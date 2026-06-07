"""
Прогнозные модели системы (XGBoost, NeuralProphet, LSTM).

Все модели объединены в одном модуле. Различия во входных форматах
инкапсулированы внутри каждой секции:
  - XGBoost / LSTM  : принимают матрицу признаков X (от FeatureBuilder)
  - NeuralProphet   : принимает DataFrame с колонками ds, y
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# NumPy 2.0 удалил np.NaN; NeuralProphet всё ещё использует его внутри.
# Патч должен быть применён ДО импорта neuralprophet.
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]


# ===========================================================================
# XGBoost
# ===========================================================================

def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 20,
    save_path: str = None,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 3,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
):
    """
    Обучает XGBRegressor с ранней остановкой.

    Parameters
    ----------
    X_train, y_train : обучающая выборка
    X_val,   y_val   : валидационная выборка (для ранней остановки)
    early_stopping_rounds : число раундов без улучшения до остановки
    save_path        : путь для сохранения через joblib (None — не сохранять)
    """
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)

    print(f"  XGBoost обучен: {model.best_iteration + 1} деревьев "
          f"(early stopping @ {early_stopping_rounds})")
    return model


def predict_xgboost(model, X) -> np.ndarray:
    """Точечный прогноз. Значения < 0 обнуляются."""
    preds = model.predict(X)
    return np.clip(preds, 0, None)


def predict_xgboost_wf(model, X) -> np.ndarray:
    """Walk-forward обёртка: прогноз по последней строке X.
    Выравнивает набор признаков по тому, на чём обучена модель —
    защита от рассинхронизации при адаптивном переобучении.
    """
    last = X.iloc[[-1]].copy()
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in last.columns:
                last[col] = 0.0
        last = last[expected]
    return predict_xgboost(model, last)


def train_xgboost_random_search(
    X_train,
    y_train,
    X_val,
    y_val,
    n_iter: int = 30,
    early_stopping_rounds: int = 20,
    save_path: str = None,
    random_state: int = 42,
    sample_weight=None,
) -> tuple:
    """
    Optuna TPE поиск гиперпараметров XGBoost (замена Random Search).
    При отсутствии optuna — fallback на Random Search.

    Возвращает (best_model, best_params, best_rmse).
    """
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error

    def _fit(params: dict):
        m = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="rmse",
            verbosity=0,
            random_state=random_state,
            **params,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              sample_weight=sample_weight, verbose=False)
        return m

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        best_model, best_params, best_rmse = None, None, float("inf")

        def objective(trial):
            nonlocal best_model, best_params, best_rmse
            params = {
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 0.3),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
            }
            m = _fit(params)
            val_preds = np.clip(m.predict(X_val), 0, None)
            rmse = float(mean_squared_error(y_val, val_preds) ** 0.5)
            if rmse < best_rmse:
                best_rmse  = rmse
                best_model = m
                best_params = params
            return rmse

        print(f"  XGBoost Optuna TPE: {n_iter} trials...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
        print(f"  XGBoost Optuna завершён: RMSE={best_rmse:.4f}  params={best_params}")

    except ImportError:
        # Fallback: Random Search если optuna не установлена
        param_grid = {
            "max_depth":        [4, 5, 6, 7, 8],
            "learning_rate":    [0.01, 0.03, 0.05, 0.07, 0.10],
            "subsample":        [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5, 7],
            "reg_alpha":        [0.0, 0.05, 0.1, 0.3],
            "reg_lambda":       [0.5, 1.0, 1.5, 2.0],
        }
        rng = np.random.default_rng(random_state)
        best_model, best_params, best_rmse = None, None, float("inf")

        print(f"  XGBoost Random Search: {n_iter} итераций...")
        for i in range(n_iter):
            params = {k: rng.choice(v).item() for k, v in param_grid.items()}
            m = _fit(params)
            val_preds = np.clip(m.predict(X_val), 0, None)
            rmse = float(mean_squared_error(y_val, val_preds) ** 0.5)
            if rmse < best_rmse:
                best_rmse  = rmse
                best_model = m
                best_params = params
                print(f"    [{i+1}/{n_iter}] новый лучший RMSE={rmse:.4f}  params={params}")
        print(f"  XGBoost Random Search завершён: RMSE={best_rmse:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(best_model, save_path)
    return best_model, best_params, best_rmse


def train_xgboost_quantile(
    X_train,
    y_train,
    X_val,
    y_val,
    quantile: float = 0.9,
    n_estimators: int = 300,
    early_stopping_rounds: int = 20,
    save_path: str = None,
):
    """Обучает модель квантильной регрессии XGBoost."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="quantile",
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)

    return model


def get_confidence_interval(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
    save_dir: str = None,
):
    """
    Возвращает нижнюю и верхнюю границы прогноза (80% CI).

    Returns
    -------
    (lower, upper) — два numpy-массива того же размера, что X_test
    """
    if len(X_val) == 0 or len(X_test) == 0:
        n = len(X_test) if len(X_test) > 0 else len(X_train)
        return np.zeros(n), np.zeros(n)

    model_low = train_xgboost_quantile(
        X_train, y_train, X_val, y_val,
        quantile=lower_q,
        save_path=os.path.join(save_dir, "xgb_q10.pkl") if save_dir else None,
    )
    model_high = train_xgboost_quantile(
        X_train, y_train, X_val, y_val,
        quantile=upper_q,
        save_path=os.path.join(save_dir, "xgb_q90.pkl") if save_dir else None,
    )
    lower = np.clip(model_low.predict(X_test), 0, None)
    upper = np.clip(model_high.predict(X_test), 0, None)
    return lower, upper


def feature_importance(model, feature_names=None) -> pd.DataFrame:
    """Возвращает DataFrame с важностью признаков XGBoost (gain)."""
    imp = model.feature_importances_
    names = feature_names if feature_names is not None else [
        f"f{i}" for i in range(len(imp))
    ]
    df = pd.DataFrame({"feature": names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ===========================================================================
# NeuralProphet
# ===========================================================================

def _infer_freq(ds: pd.Series) -> str:
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


def _build_neuralprophet(params: dict, yearly: bool):
    from neuralprophet import NeuralProphet
    return NeuralProphet(
        n_changepoints=params["n_changepoints"],
        trend_reg=params["trend_reg"],
        seasonality_reg=params["seasonality_reg"],
        ar_reg=params.get("ar_reg", 0.1),
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=yearly,
        n_forecasts=1,
        n_lags=params.get("n_lags", 0),
        quantiles=[0.1, 0.9],
        epochs=150,
        trainer_config={"enable_progress_bar": True},
    )


def train_prophet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_path: Optional[str] = None,
    use_holidays: bool = False,
    country_code: str = "RU",
    verbose: bool = True,
    exog_cols: Optional[List[str]] = None,
):
    """
    Обучает NeuralProphet с grid-search по гиперпараметрам.

    КРИТИЧНО: train_df и val_df — строго раздельные срезы.
    Обучение идёт только на train_df, оценка — на val_df.
    """
    from neuralprophet import NeuralProphet
    from sklearn.metrics import mean_absolute_error
    import torch
    # PyTorch 2.6 изменил weights_only=True по умолчанию, что ломает NeuralProphet.
    # Патчим torch.load на время обучения NeuralProphet.
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

    if "ds" not in train_df.columns or "y" not in train_df.columns:
        raise ValueError("train_df должен содержать колонки 'ds' и 'y'")

    exog_cols = [c for c in (exog_cols or [])
                 if c in train_df.columns and c in val_df.columns]
    keep = ["ds", "y"] + exog_cols
    train_df = train_df[keep].copy().reset_index(drop=True)
    val_df   = val_df[keep].copy().reset_index(drop=True)
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    val_df["ds"]   = pd.to_datetime(val_df["ds"])
    # Защита: NeuralProphet не принимает NaN в y или экзогенных колонках
    train_df["y"] = train_df["y"].interpolate(method="linear").bfill().ffill()
    val_df["y"]   = val_df["y"].interpolate(method="linear").bfill().ffill()
    for col in exog_cols:
        train_df[col] = train_df[col].fillna(0)
        val_df[col]   = val_df[col].fillna(0)
    # Колонки с нулевой дисперсией в train ломают NeuralProphet: StandardScaler
    # внутри NP вычисляет (x - mean) / std, и при std=0 получает NaN для будущих
    # строк (y=NaN), что вызывает "Future values of all user specified regressors
    # not provided" — даже если мы явно заполнили колонку нулями через _attach_exog.
    exog_cols = [c for c in exog_cols if train_df[c].nunique() > 1]

    freq   = _infer_freq(train_df["ds"])
    yearly = _enough_for_yearly(train_df["ds"])

    # n_lags зависит от частоты: для часовых данных = 24 (1 сутки),
    # для суб-часовых используем 1 час назад в периодах, но не больше 48.
    _step_sec = float(
        pd.Series(train_df["ds"]).diff().dropna().dt.total_seconds().median()
    )
    _step_min = _step_sec / 60
    if _step_min >= 30:          # hourly или реже
        _ar_lags = 24
    elif _step_min >= 5:         # 5–30 min
        _ar_lags = max(12, int(round(60 / _step_min)))   # ~1 час
    else:
        _ar_lags = 12

    param_grid = {
        "n_changepoints":  [10, 20, 30],
        "trend_reg":       [0.05, 0.1, 1.0],
        "seasonality_reg": [0.05, 0.1, 1.0],
        "n_lags":          [0, _ar_lags],   # 0 = чистый Prophet; >0 = AR-Net
        "ar_reg":          [0.05, 0.1],
    }
    all_params = [dict(zip(param_grid.keys(), v))
                  for v in product(*param_grid.values())]
    # Ограничиваем случайной выборкой из полного grid
    max_evals = min(len(all_params), 36)
    rng_grid = __import__("random").Random(42)
    all_params = rng_grid.sample(all_params, max_evals)

    best_model = None
    best_params = None
    best_mae = float("inf")

    if verbose:
        print(f"  NeuralProphet grid-search: {len(all_params)} комбинаций "
              f"(freq={freq}, yearly={yearly}, holidays={use_holidays})")

    for params in all_params:
        m = _build_neuralprophet(params, yearly)
        if use_holidays:
            try:
                m.add_country_holidays(country_name=country_code)
            except Exception:
                pass

        for col in exog_cols:
            m.add_future_regressor(col)
        m.fit(train_df, freq=freq)

        # AR-модели (n_lags > 0) нуждаются в контексте train для предсказания val.
        # Передаём train+val, берём только последние len(val_df) строк прогноза.
        if params.get("n_lags", 0) > 0:
            combined_pred = pd.concat([train_df, val_df]).reset_index(drop=True)
            forecast_full = m.predict(combined_pred)
            forecast = forecast_full.iloc[-len(val_df):].reset_index(drop=True)
        else:
            forecast = m.predict(val_df)

        yhat  = forecast["yhat1"].values
        ytrue = val_df["y"].values[-len(yhat):]
        mask  = ~np.isnan(yhat)
        if mask.sum() == 0:
            continue   # комбинация без валидных предсказаний — пропускаем
        mae = mean_absolute_error(ytrue[mask], yhat[mask])

        if mae < best_mae:
            best_mae   = mae
            best_model = m
            best_params = params

    torch.load = _orig_torch_load  # восстанавливаем после NeuralProphet

    if verbose and best_params is not None:
        print(f"  Best NeuralProphet: {best_params}  →  val MAE={best_mae:.2f}")

    if best_model is not None:
        best_model._train_df   = train_df
        best_model._best_params = best_params
        best_model._exog_cols   = exog_cols

    if save_path and best_model is not None:
        save_prophet(best_model, save_path)

    return best_model


def refit_prophet_full(
    base_model,
    train_val_df: pd.DataFrame,
    use_holidays: bool = False,
    country_code: str = "RU",
    exog_cols: Optional[List[str]] = None,
):
    """Переобучает NeuralProphet с теми же гиперпараметрами на train+val."""
    import torch as _torch
    _orig_load = _torch.load
    _torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

    exog_cols = [c for c in (exog_cols or getattr(base_model, "_exog_cols", []))
                 if c in train_val_df.columns]
    keep = ["ds", "y"] + exog_cols
    train_val_df = train_val_df[keep].copy().reset_index(drop=True)
    train_val_df["ds"] = pd.to_datetime(train_val_df["ds"])
    # Защита: NeuralProphet не принимает NaN
    train_val_df["y"] = train_val_df["y"].interpolate(method="linear").bfill().ffill()
    for col in exog_cols:
        train_val_df[col] = train_val_df[col].fillna(0)
    # Колонки с нулевой дисперсией ломают NP-нормализацию при прогнозе (std=0 → NaN)
    exog_cols = [c for c in exog_cols if train_val_df[c].nunique() > 1]

    freq   = _infer_freq(train_val_df["ds"])
    yearly = _enough_for_yearly(train_val_df["ds"])

    params = getattr(base_model, "_best_params", {
        "n_changepoints":  getattr(base_model, "n_changepoints",  30),
        "trend_reg":       getattr(base_model, "trend_reg",        0.1),
        "seasonality_reg": getattr(base_model, "seasonality_reg",  0.1),
    })

    m = _build_neuralprophet(params, yearly)
    if use_holidays:
        try:
            m.add_country_holidays(country_name=country_code)
        except Exception:
            pass
    for col in exog_cols:
        m.add_future_regressor(col)

    m.fit(train_val_df, freq=freq)
    _torch.load = _orig_load  # восстанавливаем после fit
    m._train_df    = train_val_df
    m._best_params = params
    m._exog_cols   = exog_cols
    return m


def predict_prophet_wf(model, X) -> np.ndarray:
    """Walk-forward обёртка: прогноз Prophet на текущий шаг через DatetimeIndex X."""
    try:
        train_end = pd.to_datetime(model._train_df["ds"]).max()
        current_ts = X.index[-1] if hasattr(X.index, "dtype") and str(X.index.dtype).startswith("datetime") else train_end
        delta_sec = (current_ts - train_end).total_seconds()
        freq_str = _infer_freq(model._train_df["ds"])
        try:
            step_sec = pd.tseries.frequencies.to_offset(freq_str).nanos / 1e9
        except Exception:
            step_sec = 3600.0
        n_ahead = max(1, round(delta_sec / step_sec) + 1)
        pred_df = predict_prophet(model, periods=n_ahead, history_ds=model._train_df["ds"])
        return np.clip(pred_df["yhat"].values[[-1]], 0, None)
    except Exception:
        return np.array([np.nan])


def predict_prophet(
    model,
    periods: int,
    freq: Optional[str] = None,
    history_ds: Optional[pd.Series] = None,
    future_regressors: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Прогноз на `periods` шагов вперёд.

    Возвращает DataFrame с колонками ds, yhat, yhat_lower, yhat_upper.
    future_regressors — DataFrame с колонками ds + экзогенные переменные
    для тестового горизонта (нужно при add_future_regressor).
    """
    train_df = getattr(model, "_train_df", None)
    if train_df is None:
        if history_ds is not None:
            train_df = pd.DataFrame({"ds": pd.to_datetime(history_ds), "y": 0.0})
        else:
            raise ValueError("Модель не хранит _train_df. Передайте history_ds.")

    if freq is None:
        freq = _infer_freq(train_df["ds"])

    exog_cols = getattr(model, "_exog_cols", [])
    # NeuralProphet хранит n_lags в config_ar.n_lags, а не как прямой атрибут.
    # Используем _best_params (который мы сами пишем в refit_prophet_full).
    _bp = getattr(model, "_best_params", {}) or {}
    n_lags = int(
        getattr(model, "n_lags", None)
        or getattr(getattr(model, "config_ar", None), "n_lags", None)
        or _bp.get("n_lags", 0)
        or 0
    )

    # Строим единый lookup: train-период из _train_df + test-период из future_regressors.
    # Используем map() вместо merge(), чтобы избежать дублирования колонок (col_x/col_y).
    _exog_lookup: Dict[str, pd.Series] = {}
    if exog_cols:
        for col in exog_cols:
            if col in train_df.columns:
                _exog_lookup[col] = train_df.set_index(
                    pd.to_datetime(train_df["ds"])
                )[col]
        if future_regressors is not None:
            _fr = future_regressors.copy()
            _fr["ds"] = pd.to_datetime(_fr["ds"])
            for col in exog_cols:
                if col in _fr.columns:
                    s = _fr.set_index("ds")[col]
                    _exog_lookup[col] = (
                        pd.concat([_exog_lookup[col], s])
                        if col in _exog_lookup else s
                    )

    def _attach_exog(df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет/перезаписывает колонки экзогенных признаков по ds."""
        if not exog_cols:
            return df
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        for col in exog_cols:
            lookup = _exog_lookup.get(col, pd.Series(dtype=float))
            df[col] = df["ds"].map(lookup).fillna(0).astype(int)
        return df

    if n_lags and n_lags > 0:
        # NeuralProphet с n_forecasts=1 принудительно ставит periods=1,
        # поэтому multi-step прогноз делается итеративно: каждый шаг
        # добавляет своё предсказание обратно в историю как "known y".
        # n_historic_predictions=n_lags даёт AR-контекст (последние n_lags строк),
        # иначе модель предсказывает без авторегрессивной компоненты.
        # История должна содержать экзогенные колонки — make_future_dataframe
        # валидирует входной df на наличие зарегистрированных регрессоров ещё
        # до того, как мы сами успеваем их добавить через _attach_exog.
        history = _attach_exog(train_df[["ds", "y"]].copy())
        rows: list = []
        for step in range(periods):
            fut = model.make_future_dataframe(
                df=history,
                periods=model.n_forecasts,
                n_historic_predictions=int(n_lags),
            )
            fut = _attach_exog(fut)
            fc = model.predict(fut)
            if fc.empty or "yhat1" not in fc.columns:
                break
            last = fc.iloc[-1]
            new_yhat = float(last["yhat1"]) if not pd.isna(last["yhat1"]) else 0.0
            lc = next((c for c in fc.columns if "10.0%" in c), None)
            uc = next((c for c in fc.columns if "90.0%" in c), None)
            rows.append({
                "ds":         last["ds"],
                "yhat":       new_yhat,
                "yhat_lower": float(last[lc]) if lc and not pd.isna(last[lc]) else new_yhat,
                "yhat_upper": float(last[uc]) if uc and not pd.isna(last[uc]) else new_yhat,
            })
            # Добавляем предсказание в историю ВМЕСТЕ с экзогенными колонками,
            # иначе следующий шаг получит NaN в последней строке history.
            new_row = _attach_exog(pd.DataFrame({"ds": [last["ds"]], "y": [new_yhat]}))
            history = pd.concat([history, new_row]).reset_index(drop=True)
        out = pd.DataFrame(rows).reset_index(drop=True)
    else:
        _n_lags = getattr(model, "n_lags", 0) or 0
        future = model.make_future_dataframe(
            df=train_df,
            periods=periods,
            n_historic_predictions=int(_n_lags) if _n_lags > 0 else False,
        )
        future = _attach_exog(future)
        forecast = model.predict(future)
        out = forecast[["ds", "yhat1"]].rename(columns={"yhat1": "yhat"}).copy()
        lower_col = next((c for c in forecast.columns if "10.0%" in c), None)
        upper_col = next((c for c in forecast.columns if "90.0%" in c), None)
        out["yhat_lower"] = forecast[lower_col].values if lower_col else out["yhat"]
        out["yhat_upper"] = forecast[upper_col].values if upper_col else out["yhat"]
        out = out.iloc[-periods:].reset_index(drop=True)

    for col in ("yhat", "yhat_lower", "yhat_upper"):
        if out[col].isna().any():
            fill_val = float(out[col].median())
            if np.isnan(fill_val):
                fill_val = 0.0
            out[col] = out[col].fillna(fill_val)

    return out


def save_prophet(model, path: str) -> None:
    """Сохраняет NeuralProphet (поддерживает разные версии API)."""
    import joblib as _joblib
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_path = path.replace(".json", "") + ".np"
    try:
        from neuralprophet import save_model as _np_save
        _np_save(save_path, model)
    except ImportError:
        try:
            from neuralprophet import save as _np_save
            _np_save(model, save_path)
        except (ImportError, TypeError):
            _joblib.dump(model, save_path)


def load_prophet(path: str):
    """Загружает NeuralProphet (поддерживает разные версии API)."""
    import joblib as _joblib
    load_path = path.replace(".json", "") + ".np"
    try:
        from neuralprophet import load_model as _np_load
        return _np_load(load_path)
    except ImportError:
        try:
            from neuralprophet import load as _np_load
            return _np_load(load_path)
        except (ImportError, TypeError):
            return _joblib.load(load_path)


# ===========================================================================
# Facebook Prophet (классический)
# ===========================================================================

def train_fbprophet(
    train_df: pd.DataFrame,
    use_holidays: bool = False,
    country_code: str = "RU",
    verbose: bool = True,
):
    """Обучает классический Facebook Prophet на train_df (ds, y)."""
    from prophet import Prophet
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    if "ds" not in train_df.columns or "y" not in train_df.columns:
        raise ValueError("train_df должен содержать колонки 'ds' и 'y'")

    df = train_df[["ds", "y"]].copy().reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].interpolate(method="linear").bfill().ffill()

    yearly = _enough_for_yearly(df["ds"])

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=yearly,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.8,
    )
    if use_holidays:
        try:
            model.add_country_holidays(country_name=country_code)
        except Exception:
            pass

    model.fit(df)
    return model


def predict_fbprophet(
    model,
    periods: int,
    history_ds: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Предсказывает periods шагов вперёд, возвращает df с yhat."""
    freq = _infer_freq(history_ds) if history_ds is not None else "h"
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].iloc[-periods:].reset_index(drop=True)


def save_fbprophet(model, path: str) -> None:
    from prophet.serialize import model_to_json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(model_to_json(model))


def load_fbprophet(path: str):
    from prophet.serialize import model_from_json
    with open(path, "r", encoding="utf-8") as f:
        return model_from_json(f.read())


# ===========================================================================
# LSTM
# ===========================================================================

@dataclass
class LSTMArtifact:
    """
    Связка «модель + скейлеры + конфиг», достаточная для инференса.

    Архитектура сети:
      Input(window_size, n_features)
        → LSTM(64, return_sequences=True) → Dropout(0.3)
        → LSTM(32) → Dropout(0.3)
        → Dense(16, relu) → Dense(1)
    Loss: Huber(δ=1.0) — робастен к выбросам-пикам.
    """
    keras_model: object
    feature_scaler: object
    target_scaler: object
    feature_names: List[str]
    window_size: int
    target_name: str = "y"

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        keras_path = path + ".keras"
        meta_path  = path + ".meta.pkl"
        self.keras_model.save(keras_path)
        joblib.dump(
            {
                "feature_scaler": self.feature_scaler,
                "target_scaler":  self.target_scaler,
                "feature_names":  self.feature_names,
                "window_size":    self.window_size,
                "target_name":    self.target_name,
                "keras_path":     keras_path,
            },
            meta_path,
        )

    @classmethod
    def load(cls, path: str) -> "LSTMArtifact":
        from tensorflow.keras.models import load_model
        meta_path = path + ".meta.pkl"
        meta = joblib.load(meta_path)
        keras_model = load_model(meta["keras_path"], compile=False)
        return cls(
            keras_model=keras_model,
            feature_scaler=meta["feature_scaler"],
            target_scaler=meta["target_scaler"],
            feature_names=meta["feature_names"],
            window_size=meta["window_size"],
            target_name=meta["target_name"],
        )


def _make_windows(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Преобразует (T, F) → (T-W, W, F) скользящими окнами."""
    T = X_scaled.shape[0]
    if T <= window_size:
        raise ValueError(
            f"Недостаточно строк ({T}) для окна размера {window_size}."
        )
    n = T - window_size
    F = X_scaled.shape[1]
    Xw = np.empty((n, window_size, F), dtype=np.float32)
    yw = np.empty((n,), dtype=np.float32)
    for i in range(n):
        Xw[i] = X_scaled[i:i + window_size]
        yw[i] = y_scaled[i + window_size]
    return Xw, yw


def train_lstm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
    window_size: int = 24,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    units_l1: int = 64,
    units_l2: int = 32,
    dropout: float = 0.3,
    huber_delta: float = 1.0,
    patience: int = 10,
    save_path: Optional[str] = None,
    verbose: int = 0,
) -> LSTMArtifact:
    """
    Обучает multivariate LSTM. Скейлеры fit() только на train — без утечки.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    tf.random.set_seed(42)

    feature_names = list(X_train.columns)

    feature_scaler = StandardScaler()
    target_scaler  = StandardScaler()
    feature_scaler.fit(X_train.values)
    target_scaler.fit(y_train.values.reshape(-1, 1))

    Xtr_s = feature_scaler.transform(X_train.values)
    Xvl_s = feature_scaler.transform(X_val.values)
    ytr_s = target_scaler.transform(y_train.values.reshape(-1, 1)).flatten()
    yvl_s = target_scaler.transform(y_val.values.reshape(-1, 1)).flatten()

    Xtr_w, ytr_w = _make_windows(Xtr_s, ytr_s, window_size)
    Xvl_w, yvl_w = _make_windows(Xvl_s, yvl_s, window_size)

    n_features = Xtr_w.shape[2]

    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(units_l1, return_sequences=True),
        Dropout(dropout),
        LSTM(units_l2, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=Huber(delta=huber_delta),
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-5, verbose=verbose),
    ]

    history = model.fit(
        Xtr_w, ytr_w,
        validation_data=(Xvl_w, yvl_w),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,
    )

    artifact = LSTMArtifact(
        keras_model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=feature_names,
        window_size=window_size,
    )
    artifact.history_ = history.history

    if save_path:
        artifact.save(save_path)

    return artifact


def train_lstm_random_search(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
    n_iter: int = 10,
    search_epochs: int = 30,
    final_epochs: int = 80,
    window_size: int = 24,
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> LSTMArtifact:
    """
    Random Search по гиперпараметрам LSTM.

    Каждая комбинация обучается search_epochs эпох (быстрый скрининг),
    лучшая переобучается финально на final_epochs.
    """
    param_grid = {
        "units_l1":  [32, 64, 128],
        "units_l2":  [16, 32, 64],
        "dropout":   [0.2, 0.3, 0.4],
        "lr":        [5e-4, 1e-3, 2e-3],
        "batch_size":[32, 64],
    }

    rng = np.random.default_rng(random_state)
    best_params, best_mae = None, float("inf")

    print(f"  LSTM Random Search: {n_iter} итераций × {search_epochs} эпох...")
    for i in range(n_iter):
        params = {k: rng.choice(v).item() for k, v in param_grid.items()}
        artifact = train_lstm(
            X_train, y_train, X_val, y_val,
            window_size=window_size,
            epochs=search_epochs,
            batch_size=params["batch_size"],
            lr=params["lr"],
            units_l1=params["units_l1"],
            units_l2=params["units_l2"],
            dropout=params["dropout"],
            patience=5,
            verbose=0,
        )
        val_preds = predict_lstm(artifact, X_val)
        valid = ~np.isnan(val_preds)
        mae = float(np.mean(np.abs(val_preds[valid] - y_val.values[-len(val_preds):][valid])))
        if mae < best_mae:
            best_mae = mae
            best_params = params
            print(f"    [{i+1}/{n_iter}] новый лучший MAE={mae:.4f}  params={params}")

    print(f"  LSTM Random Search завершён, финальное обучение на {final_epochs} эпохах...")
    final = train_lstm(
        X_train, y_train, X_val, y_val,
        window_size=window_size,
        epochs=final_epochs,
        batch_size=best_params["batch_size"],
        lr=best_params["lr"],
        units_l1=best_params["units_l1"],
        units_l2=best_params["units_l2"],
        dropout=best_params["dropout"],
        patience=10,
        save_path=save_path,
        verbose=0,
    )
    return final


def predict_lstm(
    artifact: LSTMArtifact,
    X_test: pd.DataFrame,
    y_history: Optional[pd.Series] = None,
) -> np.ndarray:
    """One-step прогноз для каждой строки X_test."""
    W = artifact.window_size
    feature_names = artifact.feature_names

    missing = set(feature_names) - set(X_test.columns)
    if missing:
        raise ValueError(f"X_test не содержит обученные признаки: {missing}")

    X = X_test[feature_names].values
    X_s = artifact.feature_scaler.transform(X)

    n = X_s.shape[0]
    if n <= W:
        return np.full(n, np.nan, dtype=np.float64)

    windows = np.empty((n - W, W, X_s.shape[1]), dtype=np.float32)
    for i in range(n - W):
        windows[i] = X_s[i:i + W]

    pred_scaled = artifact.keras_model.predict(windows, verbose=0).flatten()
    pred = artifact.target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()

    out = np.full(n, np.nan, dtype=np.float64)
    out[W:] = np.clip(pred, 0, None)
    return out


def predict_lstm_wf(artifact: LSTMArtifact, X) -> np.ndarray:
    """Walk-forward обёртка: прогноз LSTM с полным окном из X."""
    preds = predict_lstm(artifact, X)
    valid = preds[~np.isnan(preds)]
    return np.array([float(valid[-1])]) if len(valid) > 0 else np.array([np.nan])


def predict_lstm_aligned(
    artifact: LSTMArtifact,
    X_full: pd.DataFrame,
    n_test: int,
) -> np.ndarray:
    """
    Context-aware inference: прогноз на последние n_test точек
    с полным окном из train+val+(текущая часть теста).
    """
    feature_names = artifact.feature_names
    W = artifact.window_size

    X = X_full[feature_names].values
    X_s = artifact.feature_scaler.transform(X)

    if X_s.shape[0] < W + n_test:
        raise ValueError(
            f"Недостаточно данных: нужно ≥ {W + n_test} строк, "
            f"получено {X_s.shape[0]}"
        )

    start = X_s.shape[0] - n_test
    windows = np.empty((n_test, W, X_s.shape[1]), dtype=np.float32)
    for i in range(n_test):
        windows[i] = X_s[start + i - W:start + i]

    pred_scaled = artifact.keras_model.predict(windows, verbose=0).flatten()
    pred = artifact.target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()
    return np.clip(pred, 0, None)




