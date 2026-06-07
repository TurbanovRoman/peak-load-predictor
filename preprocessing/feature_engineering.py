"""
Модуль формирования признаков (feature engineering) для прогнозирования RPS.

Группы признаков (глава 2.6.1 ВКР):
  1. Лаговые       — rps_lag_1h, rps_lag_24h, rps_lag_168h
  2. Скользящие    — rps_mean/std за 1ч, 6ч, 24ч
  3. Производные   — rps_diff (первая разность — ранний индикатор всплеска)
  4. Календарные   — hour, day_of_week, is_weekend

Ключевое отличие от простого подхода:
  Лаги указаны в ЧАСАХ, а не в числе строк. FeatureBuilder автоматически
  вычисляет нужное количество сдвигов (shift) на основе частоты данных.
  Это позволяет работать с любым шагом: 1min, 5min, 1h и т.д.
"""

import datetime

import pandas as pd
import numpy as np
import holidays as _hol_lib
from typing import Optional, Tuple

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
    """
    Определяет шаг временного ряда в минутах по разности соседних меток.
    Использует медиану для устойчивости к пропускам.
    """
    if len(index) < 2:
        return 60.0  # по умолчанию: 1 час
    deltas = pd.Series(index).diff().dropna().dt.total_seconds() / 60
    return float(deltas.median())


def _hours_to_periods(hours: float, step_minutes: float) -> int:
    """Переводит временной лаг в часах в число периодов (строк)."""
    periods = hours * 60 / step_minutes
    return max(1, int(round(periods)))


class FeatureBuilder:
    """
    Формирует вектор признаков для прогнозирования RPS.

    Параметры, определяющие какие лаги и окна строить:
      LAG_HOURS   — лаги в часах (преобразуются в shift() по частоте данных)
      ROLL_HOURS  — окна скользящих статистик в часах

    При данных с шагом 5мин:
      lag  1ч  →  shift(12)
      lag 24ч  →  shift(288)
      lag 168ч →  shift(2016)  — нужно ≥7 дней истории!

    Если данных меньше чем нужно для максимального лага, лаговый признак
    будет содержать NaN → строки удалятся через dropna(). Это нормально:
    модель просто обучится на том объёме, который есть.
    """

    LAG_HOURS = [1, 24, 168]
    ROLL_HOURS = [1, 6, 24]

    # Базовые имена признаков (без учёта разрешения данных)
    _LAG_NAMES = ["rps_lag_1h", "rps_lag_24h", "rps_lag_168h"]
    _ROLL_MEAN_NAMES = ["rps_mean_1h", "rps_mean_6h", "rps_mean_24h"]
    _ROLL_STD_NAMES  = ["rps_std_1h",  "rps_std_6h",  "rps_std_24h"]
    _EXTRA_NAMES = [
        "rps_diff", "hour", "day_of_week", "is_weekend",
        "is_holiday", "is_halo",
        "days_to_holiday", "days_since_holiday", "halo_signal",
    ]

    def __init__(self, exog_cols=None):
        # exog_cols принимается для совместимости с Diplome-версией, не используется
        self.exog_cols = exog_cols or []

    @property
    def FEATURE_COLS(self):
        return (
            self._LAG_NAMES
            + self._ROLL_MEAN_NAMES
            + self._ROLL_STD_NAMES
            + self._EXTRA_NAMES
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Принимает DataFrame с колонками ds (datetime) и y (RPS).
        Возвращает DataFrame с признаками (без строк с NaN).

        Шаг данных определяется автоматически из временного индекса.
        """
        data = df.copy()
        data = data.set_index("ds").sort_index()
        data.index = pd.to_datetime(data.index)

        step_min = _infer_step_minutes(data.index)

        # --- 1. Лаговые признаки ---
        # shift(n) — это значение n периодов назад, оно НЕ включает текущую точку.
        for hours, name in zip(self.LAG_HOURS, self._LAG_NAMES):
            n = _hours_to_periods(hours, step_min)
            data[name] = data["y"].shift(n)

        # --- 2. Скользящие статистики ---
        # КРИТИЧНО: используем shift(1).rolling(n) — без этого окно включает
        # текущую строку y, которая является таргетом → утечка таргета в признаки
        # (target leakage). Модель тогда «угадывает» y через y и достигает
        # неестественно низкого MAE на train/val, но проваливается в проде.
        # Правильно: на момент t признак опирается ТОЛЬКО на t-1, t-2, …, t-n.
        shifted = data["y"].shift(1)
        for hours, mean_name, std_name in zip(
            self.ROLL_HOURS, self._ROLL_MEAN_NAMES, self._ROLL_STD_NAMES
        ):
            n = _hours_to_periods(hours, step_min)
            data[mean_name] = shifted.rolling(n, min_periods=1).mean()
            data[std_name]  = shifted.rolling(n, min_periods=1).std().fillna(0)

        # --- 3. Производный признак ---
        # diff() = y[t] - y[t-1] также содержит y[t]. Сдвигаем на 1, чтобы
        # признак был «изменение в предыдущем шаге», а не «изменение, ведущее
        # к таргету». Это ранний индикатор всплеска БЕЗ утечки.
        data["rps_diff"] = data["y"].diff().shift(1).fillna(0)

        # --- 4. Календарные признаки ---
        data["hour"]        = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["is_weekend"]  = (data.index.dayofweek >= 5).astype(int)

        # --- 5. Праздничные признаки и эффект ореола ---
        # Если флаги уже пришли из TimeSeriesCleaner — используем их,
        # иначе вычисляем самостоятельно.
        if "is_holiday" in data.columns:
            data["is_holiday"] = data["is_holiday"].fillna(0).astype(int)
        else:
            data["is_holiday"] = data.index.normalize().map(
                lambda d: 1 if d.date() in _RU_HOL else 0
            )

        data["days_to_holiday"]    = [_days_to_next_holiday(d)   for d in data.index]
        data["days_since_holiday"] = [_days_since_last_holiday(d) for d in data.index]

        # Нарастающий сигнал ореола: 0 вне окна, 1..HALO_BEFORE+1 внутри
        data["halo_signal"] = data["days_to_holiday"].apply(
            lambda d: max(0, _HOL_BEFORE + 1 - d)
        )

        if "is_halo" in data.columns:
            data["is_halo"] = data["is_halo"].fillna(0).astype(int)
        else:
            data["is_halo"] = (
                (data["days_to_holiday"]    <= _HOL_BEFORE) |
                (data["days_since_holiday"] <= _HOL_AFTER)
            ).astype(int)

        data = data.dropna()
        return data

    def get_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Возвращает (X, y) — признаки и целевая переменная."""
        transformed = self.transform(df)
        available = [c for c in self.FEATURE_COLS if c in transformed.columns]
        X = transformed[available]
        y = transformed["y"]
        return X, y

    def get_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Только признаки (для инференса без целевой переменной)."""
        transformed = self.transform(df)
        available = [c for c in self.FEATURE_COLS if c in transformed.columns]
        return transformed[available]

    def transform_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ):
        """
        Context-aware feature building для train/val/test сплитов.

        Вычисляет признаки на полном объединённом датасете, затем возвращает
        срезы по позиции (не по timestamp — устойчиво к дубликатам меток времени).
        """
        full = pd.concat([train, val, test], ignore_index=True).sort_values("ds")
        transformed = self.transform(full)
        available = [c for c in self.FEATURE_COLS if c in transformed.columns]

        n_test = len(test)
        n_val  = len(val)

        feat_test  = transformed.iloc[-n_test:]
        feat_val   = transformed.iloc[-(n_test + n_val):-n_test]
        feat_train = transformed.iloc[:-(n_test + n_val)]

        def _xy(subset):
            return subset[available], subset["y"]

        return _xy(feat_train), _xy(feat_val), _xy(feat_test)

    def feature_count(self, df: pd.DataFrame) -> int:
        """Число признаков, которые будут построены для данного датасета."""
        return len(self.get_X(df.head(200)).columns)

# ---------------------------------------------------------------------------
# Утилиты (обратная совместимость с main.py)
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame, label: Optional[str] = None):
    """
    Обёртка вокруг FeatureBuilder для обратной совместимости.
    Если label задан — возвращает (X, y), иначе только X.
    """
    builder = FeatureBuilder()
    if label:
        return builder.get_X_y(df)
    return builder.get_X(df)


def split_train_val_test(
    df: pd.DataFrame,
    test_hours: int = 288,
    val_hours: int = 288,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Хронологическое разбиение без перемешивания.

    Параметр test_hours/val_hours — число ПЕРИОДОВ (не обязательно часов).
    При DEFAULT_STEP="5min" передавайте 288 для 24ч, 576 для 48ч и т.д.

    Перемешивание недопустимо для временных рядов (утечка из будущего).
    """
    df = df.sort_values("ds").reset_index(drop=True)
    n = len(df)
    split_test = n - test_hours
    split_val  = split_test - val_hours
    train = df.iloc[:split_val].copy()
    val   = df.iloc[split_val:split_test].copy()
    test  = df.iloc[split_test:].copy()
    return train, val, test
