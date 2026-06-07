"""
Многошаговое прогнозирование (multi-step forecasting).

Реализует рекурсивный прогноз на горизонте H > 1 шаг для моделей,
обученных на one-step ahead. Работает по схеме direct-recursive:
  1. Получить прогноз ŷ_{t+1} от модели
  2. Подставить ŷ_{t+1} как новое наблюдение в историю
  3. Пересчитать признаки X_{t+2} с этим псевдо-наблюдением
  4. Получить ŷ_{t+2}, повторить до t+H

Альтернатива production-систем (например, AWS Predictive Scaling),
которые делают прогноз на 48 часов вперёд.

Ограничения:
  • Накопление ошибки: ŷ_{t+h} использует все предыдущие
    предсказания как «факт», поэтому при h → ∞ дисперсия растёт.
  • Применимо только для одношаговых моделей XGBoost/LSTM;
    Prophet поддерживает многошаговый прогноз нативно.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from preprocessing.feature_engineering import FeatureBuilder


def recursive_forecast(
    history_df: pd.DataFrame,
    model: object,
    predict_fn: Callable,
    builder: FeatureBuilder,
    horizon: int,
    step_minutes: Optional[float] = None,
    exog_future: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Рекурсивный многошаговый прогноз.

    Parameters
    ----------
    history_df    : DataFrame с колонками ds, y (+ опционально экзогенные)
    model         : обученная модель (XGBoost, LSTMArtifact, ...)
    predict_fn    : функция predict(model, X) -> np.ndarray
    builder       : FeatureBuilder для построения признаков
    horizon       : сколько шагов вперёд предсказать
    step_minutes  : шаг ряда; если None — определяется автоматически
    exog_future   : DataFrame с колонками ds + экзогенные (is_campaign, is_promo)
                    для будущих шагов. Если None — экзогенные = 0.

    Returns
    -------
    DataFrame с колонками ds, y_hat — прогноз на horizon шагов вперёд.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    # Сохраняем экзогенные колонки из истории (известны заранее — calendar flags)
    exog_cols: List[str] = [c for c in history_df.columns
                            if c not in ("ds", "y")]

    df = history_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # Определяем шаг ряда
    if step_minutes is None:
        deltas = df["ds"].diff().dropna().dt.total_seconds() / 60
        step_minutes = float(deltas.median()) if len(deltas) else 60.0

    step = pd.Timedelta(minutes=step_minutes)
    last_ts = df["ds"].iloc[-1]

    # Индекс будущих экзогенных значений (если переданы)
    exog_idx: Optional[pd.DataFrame] = None
    if exog_future is not None and exog_cols:
        exog_idx = exog_future.copy()
        exog_idx["ds"] = pd.to_datetime(exog_idx["ds"])
        exog_idx = exog_idx.set_index("ds")

    forecasts = []
    for h in range(1, horizon + 1):
        next_ts = last_ts + h * step

        # Для h > 1 добавляем псевдо-наблюдение с последним прогнозом
        if h > 1:
            new_row = {"ds": next_ts - step, "y": forecasts[-1]["y_hat"]}
            # Добавляем экзогенные для предыдущего шага
            for col in exog_cols:
                if exog_idx is not None and (next_ts - step) in exog_idx.index:
                    new_row[col] = int(exog_idx.loc[next_ts - step, col])
                else:
                    new_row[col] = 0
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Добавляем строку для текущего шага h с экзогенными = известны заранее
        future_row = {"ds": next_ts, "y": np.nan}
        for col in exog_cols:
            if exog_idx is not None and next_ts in exog_idx.index:
                future_row[col] = int(exog_idx.loc[next_ts, col])
            else:
                future_row[col] = 0
        synthetic = pd.concat([df, pd.DataFrame([future_row])], ignore_index=True)
        # Временно заполняем y=nan последней известной точкой, чтобы dropna не убрал строку
        synthetic["y"] = synthetic["y"].fillna(synthetic["y"].ffill())

        X = builder.get_X(synthetic)
        if len(X) == 0:
            forecasts.append({"ds": next_ts, "y_hat": np.nan})
            continue

        x_last = X.iloc[[-1]]
        y_pred = max(0.0, float(predict_fn(model, x_last)[0]))
        forecasts.append({"ds": next_ts, "y_hat": y_pred})

    return pd.DataFrame(forecasts)
