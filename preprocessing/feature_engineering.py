"""
Модуль формирования признаков (feature engineering) для прогнозирования RPS.

Адаптировано для данных с гранулярностью 1 час.
Группы признаков:
  1. Лаговые       — rps_lag_1h, rps_lag_2h, rps_lag_3h, rps_lag_24h, rps_lag_168h
  2. Скользящие    — rps_mean/std за 3ч, 6ч, 24ч 
  3. Производные   — rps_diff, rps_momentum 
  4. Календарные   — hour, day_of_week, is_weekend
  5. Праздничные   — halo_signal, days_to_holiday и т.д.
"""

import datetime
import pandas as pd
import numpy as np
import holidays as _hol_lib
from typing import Optional, Tuple
import logging

# ─── Праздничный календарь и параметры ореола ─────────────────────────────────
try:
    from config import HALO_BEFORE as _HOL_BEFORE, HALO_AFTER as _HOL_AFTER
except (ImportError, Exception):
    _HOL_BEFORE = 3
    _HOL_AFTER  = 2

_RU_HOL: set[datetime.date] = set(_hol_lib.Russia(years=range(2019, 2040)).keys())

def _days_to_next_holiday(ts: pd.Timestamp) -> int:
    d = ts.date()
    if d in _RU_HOL:
        return 0
    for i in range(1, _HOL_BEFORE + 1):
        if (d + datetime.timedelta(days=i)) in _RU_HOL:
            return i
    return _HOL_BEFORE + 1

def _days_since_last_holiday(ts: pd.Timestamp) -> int:
    d = ts.date()
    if d in _RU_HOL:
        return 0
    for i in range(1, _HOL_AFTER + 1):
        if (d - datetime.timedelta(days=i)) in _RU_HOL:
            return i
    return _HOL_AFTER + 1
# ──────────────────────────────────────────────────────────────────────────────

def _infer_step_minutes(index: pd.DatetimeIndex) -> float:
    """Определяет шаг временного ряда в минутах."""
    if len(index) < 2:
        return 60.0  
    deltas = pd.Series(index).diff().dropna().dt.total_seconds() / 60
    return float(deltas.median())

def _hours_to_periods(hours: float, step_minutes: float) -> int:
    """Переводит временной лаг в часах в число периодов (строк)."""
    periods = hours * 60 / step_minutes
    return max(1, int(round(periods)))

class FeatureBuilder:
    """Формирует вектор признаков для прогнозирования RPS."""
    
    LAG_HOURS = [1, 2, 3, 24, 168]
    ROLL_HOURS = [3, 6, 24]

    _LAG_HOURS_NAMES = ["rps_lag_1h", "rps_lag_2h", "rps_lag_3h", "rps_lag_24h", "rps_lag_168h"]
    _ROLL_MEAN_NAMES = ["rps_mean_3h", "rps_mean_6h", "rps_mean_24h"]
    _ROLL_STD_NAMES  = ["rps_std_3h",  "rps_std_6h",  "rps_std_24h"]
    
    _EXTRA_NAMES = [
        "rps_diff", "rps_momentum", 
        "hour", "day_of_week", "is_weekend",
        "is_holiday", "is_halo",
        "days_to_holiday", "days_since_holiday", "halo_signal",
    ]

    def __init__(self, exog_cols=None):
        self.exog_cols = exog_cols or []

    @property
    def FEATURE_COLS(self):
        cols = (
            self._LAG_HOURS_NAMES
            + self._ROLL_MEAN_NAMES
            + self._ROLL_STD_NAMES
            + self._EXTRA_NAMES
            + self.exog_cols
        )
        # ЗАЩИТА №1: Удаляем возможные дубликаты признаков, сохраняя порядок
        return list(dict.fromkeys(cols))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data = data.set_index("ds").sort_index()
        data.index = pd.to_datetime(data.index)

        step_min = _infer_step_minutes(data.index)

        # --- 1. Лаговые признаки ---
        for hours, name in zip(self.LAG_HOURS, self._LAG_HOURS_NAMES):
            n = _hours_to_periods(hours, step_min)
            data[name] = data["y"].shift(n)

        # --- 2. Скользящие статистики ---
        shifted = data["y"].shift(1)
        for hours, mean_name, std_name in zip(
            self.ROLL_HOURS, self._ROLL_MEAN_NAMES, self._ROLL_STD_NAMES
        ):
            n = _hours_to_periods(hours, step_min)
            data[mean_name] = shifted.rolling(n, min_periods=1).mean()
            data[std_name]  = shifted.rolling(n, min_periods=1).std().fillna(0)

        # --- 3. Производные признаки ---
        data["rps_diff"] = data["y"].diff().shift(1).fillna(0)
        data["rps_momentum"] = (data["rps_lag_1h"] - data["rps_lag_2h"]).fillna(0)

        # --- 4. Календарные признаки ---
        data["hour"]        = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["is_weekend"]  = (data.index.dayofweek >= 5).astype(int)

        # --- 5. Праздничные признаки ---
        if "is_holiday" in data.columns:
            # Используем .iloc[:, 0] на случай, если колонка is_holiday сдублировалась
            if isinstance(data["is_holiday"], pd.DataFrame):
                data["is_holiday"] = data["is_holiday"].iloc[:, 0]
            data["is_holiday"] = data["is_holiday"].fillna(0).astype(int)
        else:
            data["is_holiday"] = data.index.normalize().map(
                lambda d: 1 if d.date() in _RU_HOL else 0
            )

        data["days_to_holiday"]    = [_days_to_next_holiday(d)   for d in data.index]
        data["days_since_holiday"] = [_days_since_last_holiday(d) for d in data.index]

        data["halo_signal"] = data["days_to_holiday"].apply(
            lambda d: max(0, _HOL_BEFORE + 1 - d)
        )

        if "is_halo" in data.columns:
            if isinstance(data["is_halo"], pd.DataFrame):
                data["is_halo"] = data["is_halo"].iloc[:, 0]
            data["is_halo"] = data["is_halo"].fillna(0).astype(int)
        else:
            data["is_halo"] = (
                (data["days_to_holiday"]    <= _HOL_BEFORE) |
                (data["days_since_holiday"] <= _HOL_AFTER)
            ).astype(int)

        data = data.dropna()
        
        # ЗАЩИТА №2: Предупреждение, если лаги "съели" весь датасет
        if len(data) == 0:
            logging.warning("FeatureBuilder: После dropna() не осталось данных! Проверьте объем train/val (нужно больше истории для лага 168ч).")

        return data

    def get_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        transformed = self.transform(df)
        available = [c for c in self.FEATURE_COLS if c in transformed.columns]
        X = transformed[available]
        y = transformed["y"]
        
        # ЗАЩИТА №3: Жестко преобразуем y в Series, если Pandas вернул DataFrame
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
            
        return X, y

    def get_X(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transform(df)
        available = [c for c in self.FEATURE_COLS if c in transformed.columns]
        return transformed[available]

    def transform_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        full = pd.concat([train, val, test], ignore_index=True).sort_values("ds")
        transformed = self.transform(full)
        available = [c for c in self.FEATURE_COLS if c in transformed.columns]

        n_test = len(test)
        n_val  = len(val)

        feat_test  = transformed.iloc[-n_test:]
        feat_val   = transformed.iloc[-(n_test + n_val):-n_test]
        feat_train = transformed.iloc[:-(n_test + n_val)]

        def _xy(subset):
            y_out = subset["y"]
            # Снова жесткая защита таргета
            if isinstance(y_out, pd.DataFrame):
                y_out = y_out.iloc[:, 0]
            return subset[available], y_out

        return _xy(feat_train), _xy(feat_val), _xy(feat_test)

    def feature_count(self, df: pd.DataFrame) -> int:
        return len(self.get_X(df.head(200)).columns)

# ---------------------------------------------------------------------------
def create_features(df: pd.DataFrame, label: Optional[str] = None):
    builder = FeatureBuilder()
    if label:
        return builder.get_X_y(df)
    return builder.get_X(df)

def split_train_val_test(df: pd.DataFrame, test_hours: int = 288, val_hours: int = 288) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("ds").reset_index(drop=True)
    n = len(df)
    split_test = n - test_hours
    split_val  = split_test - val_hours
    train = df.iloc[:split_val].copy()
    val   = df.iloc[split_val:split_test].copy()
    test  = df.iloc[split_test:].copy()
    return train, val, test