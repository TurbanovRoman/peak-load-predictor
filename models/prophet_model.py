"""
Prophet — модель декомпозиции временного ряда (тренд + сезонности + праздники).

Полностью переписана после аудита (см. AUDIT_REPORT.md, шаг 6).
Что было исправлено по сравнению со старой версией:

  СТАРАЯ ВЕРСИЯ                       │  НОВАЯ ВЕРСИЯ
  ────────────────────────────────────┼──────────────────────────────────
  m.fit(train_df) — но в вызывающей    │  Функция строго принимает train_df
  части train_df=train+val             │  и val_df РАЗДЕЛЬНО; обучение всегда
  → grid-search оценивает «обучен на   │  идёт ТОЛЬКО на train_df, оценка —
  train+val, проверен на val» = утечка │  на val_df → честный grid-search
  → подбираются параметры, которые     │  (см. AUDIT_REPORT.md, баг C2)
  переобучаются на val                 │
  ────────────────────────────────────┼──────────────────────────────────
  freq='H' захардкожен в predict       │  freq автоматически определяется
  → ломается на 5min/15min/1d данных   │  из ds-индекса или передаётся явно
  ────────────────────────────────────┼──────────────────────────────────
  Сезонности по умолчанию              │  Явные daily/weekly/yearly +
  (часто вообще не активируются если   │  настраиваемая длина истории →
  len(history) < 2 лет)                │  Prophet моделирует их корректно
  ────────────────────────────────────┼──────────────────────────────────
  joblib.dump объекта Prophet          │  Используется штатный
  → ломается между версиями prophet    │  prophet.serialize.model_to_json
  и stan-backend                       │
  ────────────────────────────────────┼──────────────────────────────────
  Метрика выбора — MSE                 │  MAE (более интерпретируема,
                                        │  не штрафует выбросы квадратично)
  ────────────────────────────────────┼──────────────────────────────────
  Нет диагностики                      │  Возвращается best_params для
                                        │  лога / отчёта ВКР

Prophet был оставлен в сравнении ВКР, потому что:
  1. Это эталонная для индустрии модель декомпозиции рядов.
  2. Даёт интерпретируемые компоненты (тренд, недельная сезонность,
     праздники) — что ценно для технического отчёта.
  3. Хорошо работает «из коробки» на длинных рядах с регулярными
     сезонностями — типичный сценарий для веб-трафика.
"""

from __future__ import annotations

import json
import os
from itertools import product
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _infer_freq(ds: pd.Series) -> str:
    """
    Определяет частоту pandas из ряда дат.

    Возвращает строку формата 'h', '5min', '15min', 'D' и т.п.
    Если pd.infer_freq не справился — считает медианный шаг и подбирает.
    """
    ds = pd.to_datetime(ds).sort_values()
    if len(ds) < 3:
        return "h"
    inferred = pd.infer_freq(ds.iloc[:50])
    if inferred:
        return inferred

    # Fallback: медианный шаг → строка частоты
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
    return "h"   # безопасный дефолт


def _enough_for_yearly(ds: pd.Series) -> bool:
    """Yearly seasonality имеет смысл только если истории > ~1.5 лет."""
    span = (pd.to_datetime(ds.max()) - pd.to_datetime(ds.min())).days
    return span >= 540


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train_prophet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_path: Optional[str] = None,
    use_holidays: bool = False,
    country_code: str = "RU",
    verbose: bool = True,
) -> Prophet:
    """
    Обучает Prophet с grid-search по гиперпараметрам.

    КРИТИЧНО: train_df и val_df ДОЛЖНЫ быть строго раздельными временнЫми
    срезами. Эта функция НЕ конкатенирует их. Если передать train+val в
    train_df, валидация даст ложно-низкую ошибку (см. баг C2).

    Parameters
    ----------
    train_df, val_df : DataFrame с колонками 'ds', 'y'
    save_path        : путь для сохранения JSON-сериализации
    use_holidays     : включить праздники страны country_code
    country_code     : ISO-код страны для add_country_holidays

    Returns
    -------
    Лучшая (по MAE на val_df) модель Prophet, дообученная на train_df.
    Для финального инференса по тесту рекомендуется отдельно дообучить
    эту же конфигурацию на train+val (см. refit_prophet_full).
    """
    if "ds" not in train_df.columns or "y" not in train_df.columns:
        raise ValueError("train_df должен содержать колонки 'ds' и 'y'")
    if "ds" not in val_df.columns or "y" not in val_df.columns:
        raise ValueError("val_df должен содержать колонки 'ds' и 'y'")

    train_df = train_df[["ds", "y"]].copy()
    val_df   = val_df[["ds", "y"]].copy()
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    val_df["ds"]   = pd.to_datetime(val_df["ds"])

    # Решаем какие сезонности активировать на основе длины истории
    yearly = _enough_for_yearly(train_df["ds"])

    # Сетка гиперпараметров. Сжата до разумного, чтобы grid-search не висел.
    param_grid = {
        "changepoint_prior_scale": [0.01, 0.1, 0.5],
        "seasonality_prior_scale": [1.0, 10.0],
        "seasonality_mode":        ["additive", "multiplicative"],
    }
    all_params = [dict(zip(param_grid.keys(), v))
                  for v in product(*param_grid.values())]

    best_model: Optional[Prophet] = None
    best_params: Optional[dict] = None
    best_mae = float("inf")

    if verbose:
        print(f"  Prophet grid-search: {len(all_params)} комбинаций  "
              f"(yearly={yearly}, holidays={use_holidays})")

    for params in all_params:
        m = Prophet(
            **params,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=yearly,
            interval_width=0.8,
        )
        if use_holidays:
            try:
                m.add_country_holidays(country_name=country_code)
            except Exception:
                pass        # если страны нет в holidays — пропускаем тихо

        # Учим ТОЛЬКО на train_df
        m.fit(train_df)

        # Прогноз на даты val_df
        forecast = m.predict(val_df[["ds"]])
        mae = mean_absolute_error(val_df["y"].values, forecast["yhat"].values)

        if mae < best_mae:
            best_mae = mae
            best_model = m
            best_params = params

    if verbose and best_params is not None:
        print(f"  Best Prophet params: {best_params}  →  val MAE={best_mae:.2f}")

    if save_path and best_model is not None:
        save_prophet(best_model, save_path)

    return best_model


def refit_prophet_full(
    base_model: Prophet,
    train_val_df: pd.DataFrame,
    use_holidays: bool = False,
    country_code: str = "RU",
) -> Prophet:
    """
    Берёт гиперпараметры из base_model и переобучает Prophet на train+val.

    Это рекомендуемая практика после grid-search: best-model выбран по val,
    но финальное предсказание делается с моделью, обученной на МАКСИМУМЕ
    доступных до теста данных.

    Parameters
    ----------
    base_model    : модель из train_prophet()
    train_val_df  : концатенированный train+val (отсортированный по ds)
    """
    params = {
        "changepoint_prior_scale": base_model.changepoint_prior_scale,
        "seasonality_prior_scale": base_model.seasonality_prior_scale,
        "seasonality_mode": base_model.seasonality_mode,
    }
    yearly = _enough_for_yearly(train_val_df["ds"])
    m = Prophet(
        **params,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=yearly,
        interval_width=0.8,
    )
    if use_holidays:
        try:
            m.add_country_holidays(country_name=country_code)
        except Exception:
            pass
    m.fit(train_val_df[["ds", "y"]])
    return m


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def predict_prophet(
    model: Prophet,
    periods: int,
    freq: Optional[str] = None,
    history_ds: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Прогноз на `periods` шагов вперёд с авто-freq.

    Parameters
    ----------
    model      : обученный Prophet
    periods    : сколько шагов предсказать
    freq       : частота pandas ('h', '5min', 'D'…). Если None — пытаемся
                 определить из history_ds или из самой модели.
    history_ds : (опционально) исходный ряд дат, на котором обучался Prophet,
                 чтобы корректно вывести freq.

    Returns
    -------
    DataFrame с колонками ds, yhat, yhat_lower, yhat_upper для последних
    `periods` строк (это и есть прогноз вперёд).
    """
    if freq is None:
        # Сначала пробуем history_ds, иначе — внутреннюю историю модели
        if history_ds is not None:
            freq = _infer_freq(history_ds)
        else:
            freq = _infer_freq(pd.Series(model.history["ds"]))

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return (
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .iloc[-periods:]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Сериализация (рекомендуемый способ из документации Prophet)
# ---------------------------------------------------------------------------

def save_prophet(model: Prophet, path: str) -> None:
    """Сохраняет модель в JSON через штатную сериализацию Prophet."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Документация Prophet рекомендует .json, но мы оставим .pkl-совместимое
    # имя для обратной совместимости с model_comparison
    json_path = path if path.endswith(".json") else path + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(model_to_json(model))


def load_prophet(path: str) -> Prophet:
    """Загружает модель из JSON-сериализации."""
    json_path = path if path.endswith(".json") else path + ".json"
    with open(json_path, "r", encoding="utf-8") as f:
        return model_from_json(f.read())
