"""
NeuralProphet — нейросетевая декомпозиция временного ряда (глава 2.6.3 ВКР).

Отличия от классического Prophet:
  - AR-Net: авторегрессионный компонент (n_lags периодов истории)
  - PyTorch backend вместо Stan → быстрее, GPU-поддержка
  - Точность на коротких горизонтах выше за счёт AR-Net
  - Современный API, активная разработка

Архитектура модели:
  Trend(changepoints) + Seasonality(Fourier) + AR-Net(n_lags → Dense → yhat)

Ключевые параметры:
  n_lags     — глубина авторегрессии (в периодах); автовычисляется как ~24ч
  n_forecasts=1 — one-step-ahead прогноз (согласован с XGBoost/LSTM)
  epochs     — 50 по умолчанию (достаточно для ВКР, не тормозит)

Сериализация:
  NeuralProphet.save() / NeuralProphet.load() — штатный способ (.np модель)
  Не используем joblib — модель содержит PyTorch state dict, joblib не умеет.

КРИТИЧНО по валидации:
  train_df и val_df ДОЛЖНЫ быть строго разными срезами.
  При вызове m.fit(train_df, validation_df=val_df) NeuralProphet обучается
  только на train_df и оценивает на val_df — честная схема.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Утилиты (дублируем из prophet_model — чтобы не создавать кросс-зависимость)
# ---------------------------------------------------------------------------

def _infer_freq(ds: pd.Series) -> str:
    """Определяет pandas-частоту из ряда дат."""
    ds = pd.to_datetime(ds).sort_values()
    if len(ds) < 3:
        return "h"
    inferred = pd.infer_freq(ds.iloc[:50])
    if inferred:
        return inferred
    deltas_sec = ds.diff().dropna().dt.total_seconds()
    step_sec = float(deltas_sec.median())
    if step_sec <= 90:
        return "1min"
    if step_sec <= 600:
        return f"{int(round(step_sec / 60))}min"
    if step_sec <= 3600 * 1.5:
        return "h"
    if step_sec <= 86400 * 1.5:
        return "D"
    return "h"


def _enough_for_yearly(ds: pd.Series) -> bool:
    span = (pd.to_datetime(ds.max()) - pd.to_datetime(ds.min())).days
    return span >= 540


def _auto_n_lags(ds: pd.Series) -> int:
    """
    Возвращает число лагов для AR-Net равное ~24 часам в периодах данных.
    Минимум 12, максимум 48 (чтобы не тормозило на очень мелком шаге).
    """
    ds = pd.to_datetime(ds).sort_values()
    if len(ds) < 2:
        return 24
    step_sec = float(ds.diff().dropna().dt.total_seconds().median())
    step_min = step_sec / 60
    n = int(round(24 * 60 / step_min))
    return max(12, min(n, 48))


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def _build_np_model(n_lags, n_changepoints, trend_reg, seasonality_reg,
                    ar_reg, yearly, epochs):
    from neuralprophet import NeuralProphet
    kwargs = dict(
        n_forecasts=1,
        n_lags=n_lags,
        n_changepoints=n_changepoints,
        trend_reg=trend_reg,
        seasonality_reg=seasonality_reg,
        ar_reg=ar_reg,
        yearly_seasonality=yearly,
        weekly_seasonality=True,
        daily_seasonality=True,
        epochs=epochs,
        early_stopping=True,
        trainer_config={"enable_progress_bar": True},
    )
    try:
        return NeuralProphet(**kwargs)
    except TypeError:
        kwargs.pop("trainer_config", None)
        try:
            return NeuralProphet(**kwargs)
        except TypeError:
            kwargs.pop("early_stopping", None)
            return NeuralProphet(**kwargs)


def train_neural_prophet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_lags: Optional[int] = None,
    epochs: int = 50,
    save_path: Optional[str] = None,
    verbose: bool = True,
    n_trials: int = 15,
) -> object:
    """
    Обучает NeuralProphet с Bayesian-поиском гиперпараметров (Optuna).

    КРИТИЧНО: train_df и val_df — строго разные временные срезы.

    Parameters
    ----------
    train_df, val_df : DataFrame с колонками 'ds', 'y'
    n_lags           : окно AR-Net (None = автоматически ~24ч, макс 48)
    epochs           : максимум эпох на trial; early_stopping остановит раньше
    save_path        : путь для сохранения лучшей модели
    n_trials         : число Optuna-trials (15 ≈ random-36 по качеству)

    Returns
    -------
    Лучшая обученная модель NeuralProphet
    """
    try:
        from neuralprophet import NeuralProphet  # noqa: F401
    except ImportError as exc:
        raise ImportError("pip install neuralprophet") from exc

    try:
        import optuna as _optuna
    except ImportError as exc:
        raise ImportError("pip install optuna") from exc

    if "ds" not in train_df.columns or "y" not in train_df.columns:
        raise ValueError("train_df должен содержать колонки 'ds' и 'y'")

    train_df = train_df[["ds", "y"]].copy()
    val_df   = val_df[["ds", "y"]].copy()
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    val_df["ds"]   = pd.to_datetime(val_df["ds"])

    freq   = _infer_freq(train_df["ds"])
    yearly = _enough_for_yearly(train_df["ds"])
    if n_lags is None:
        n_lags = _auto_n_lags(train_df["ds"])

    import logging
    import numpy as np
    logging.getLogger("NP.df_utils").setLevel(logging.ERROR)
    logging.getLogger("NP.forecaster").setLevel(logging.ERROR)
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    if verbose:
        print(f"  NeuralProphet: freq={freq}, n_lags={n_lags}, "
              f"epochs={epochs}, yearly={yearly}")
        print(f"  NeuralProphet Optuna search: {n_trials} trials")

    # Патч torch.load для совместимости с PyTorch 2.6
    import torch as _torch
    _orig_load = _torch.load
    _torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

    val_full = pd.concat([train_df, val_df]).sort_values("ds")
    best_model: object = None
    best_mae = float("inf")
    best_params_found: dict = {}

    def objective(trial: "_optuna.Trial") -> float:
        nonlocal best_model, best_mae, best_params_found
        params = {
            "n_changepoints":  trial.suggest_categorical("n_changepoints", [10, 20, 30]),
            "trend_reg":       trial.suggest_float("trend_reg", 0.05, 1.0, log=True),
            "seasonality_reg": trial.suggest_float("seasonality_reg", 0.05, 1.0, log=True),
            "ar_reg":          trial.suggest_float("ar_reg", 0.05, 0.1, log=True),
        }
        try:
            m = _build_np_model(
                n_lags=n_lags,
                yearly=yearly,
                epochs=epochs,
                **params,
            )
            m.fit(train_df, freq=freq, validation_df=val_df)
            pred_df = m.predict(val_full)
            col = "yhat1" if "yhat1" in pred_df.columns else "yhat"
            preds = pred_df[col].values[-len(val_df):]
            mae = float(np.mean(np.abs(val_df["y"].values[:len(preds)] - preds)))
            if mae < best_mae:
                best_mae = mae
                best_model = m
                best_params_found = params
            return mae
        except Exception:
            return float("inf")

    study = _optuna.create_study(
        direction="minimize",
        sampler=_optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    _torch.load = _orig_load  # восстанавливаем torch.load

    if best_model is None:
        raise RuntimeError("Все trials NeuralProphet завершились с ошибкой")

    if verbose:
        print(f"  Best NeuralProphet: {best_params_found}  →  val MAE={best_mae:.2f}")
        print(f"  NeuralProphet обучен: n_lags={best_model.n_lags}, freq={freq}")

    if save_path:
        save_neural_prophet(best_model, save_path)

    return best_model


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def predict_neural_prophet(
    model,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """
    One-step-ahead прогноз на тестовом периоде.

    NeuralProphet с AR-Net нуждается в n_lags предшествующих значениях y
    для каждой предсказываемой точки. Передаём контекст из конца train+val.

    Это «teacher forcing» — в качестве AR-входа используются реальные
    (а не предсказанные) значения y. Это согласовано с подходом XGBoost,
    который тоже использует реальные лаговые значения через FeatureBuilder.

    Parameters
    ----------
    model        : обученная модель NeuralProphet
    train_val_df : объединённый train+val (контекст для AR-Net)
    test_df      : тестовый датасет (ds + y — y нужны как AR-вход)

    Returns
    -------
    np.ndarray длины len(test_df): прогноз RPS ≥ 0
    """
    n_lags = getattr(model, "n_lags", 0)

    test_df = test_df[["ds", "y"]].copy()
    test_df["ds"] = pd.to_datetime(test_df["ds"])

    if n_lags > 0:
        context = train_val_df[["ds", "y"]].tail(n_lags).copy()
        context["ds"] = pd.to_datetime(context["ds"])
        df_pred = pd.concat([context, test_df], ignore_index=True)
    else:
        df_pred = test_df.copy()

    forecast = model.predict(df_pred)

    # Колонка прогноза называется 'yhat1' для n_forecasts=1
    if "yhat1" not in forecast.columns:
        # Fallback: ищем первую yhat-колонку
        yhat_cols = [c for c in forecast.columns if c.startswith("yhat")]
        if not yhat_cols:
            raise RuntimeError(
                f"NeuralProphet не вернул колонку yhat. "
                f"Доступные: {list(forecast.columns)}"
            )
        yhat_col = yhat_cols[0]
    else:
        yhat_col = "yhat1"

    preds = forecast[yhat_col].iloc[-len(test_df):].values
    return np.clip(np.asarray(preds, dtype=float), 0, None)


# ---------------------------------------------------------------------------
# Сериализация
# ---------------------------------------------------------------------------

def save_neural_prophet(model, path: str) -> None:
    """Сохраняет NeuralProphet (совместимость с разными версиями API)."""
    np_path = path if path.endswith(".np") else path + ".np"
    os.makedirs(os.path.dirname(np_path) or ".", exist_ok=True)
    try:
        from neuralprophet import save_model as _np_save
        _np_save(np_path, model)
    except (ImportError, TypeError):
        try:
            from neuralprophet import save as _np_save
            _np_save(model, np_path)
        except (ImportError, TypeError):
            import joblib
            joblib.dump(model, np_path)


def load_neural_prophet(path: str):
    """Загружает NeuralProphet из .np файла."""
    try:
        from neuralprophet import NeuralProphet
    except ImportError as exc:
        raise ImportError("pip install neuralprophet") from exc
    np_path = path if path.endswith(".np") else path + ".np"
    return NeuralProphet.load(np_path)
