from __future__ import annotations

import datetime
import logging
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import holidays as holidays_lib
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.seasonal import STL as _STL

logger = logging.getLogger(__name__)

try:
    from config import HALO_BEFORE, HALO_AFTER
except (ImportError, Exception):
    HALO_BEFORE = 3
    HALO_AFTER  = 2

_RU_HOLIDAYS = holidays_lib.Russia(years=range(2019, 2036))

# Все даты в окне ореола (исключаются из «нормального» фона при обучении Тьюки)
_HALO_DATES: set[datetime.date] = set()
for _hdate in _RU_HOLIDAYS:
    for _i in range(1, HALO_BEFORE + 1):
        _HALO_DATES.add(_hdate - datetime.timedelta(days=_i))
    for _i in range(1, HALO_AFTER + 1):
        _HALO_DATES.add(_hdate + datetime.timedelta(days=_i))

_CLEANER_FILENAME = "cleaner.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _parse_and_grid(df: pd.DataFrame, ts_col: str, value_col: str, min_step_seconds: int = 1):
    """Парсинг, сортировка, дедупликация и выравнивание на равномерную сетку."""
    work = df[[ts_col, value_col]].copy()
    work[ts_col] = pd.to_datetime(work[ts_col], errors="coerce")
    work = work.dropna(subset=[ts_col])
    work = work.set_index(ts_col).sort_index()
    work = work[~work.index.duplicated(keep="last")]

    deltas = work.index.to_series().diff().dropna()
    if deltas.empty:
        step_seconds = min_step_seconds
    else:
        median_delta = deltas.median()
        step_seconds = max(int(median_delta.total_seconds()), min_step_seconds)

    freq = pd.Timedelta(seconds=step_seconds)
    grid = pd.date_range(start=work.index.min(), end=work.index.max(), freq=freq)
    work = work.reindex(grid)
    work["is_holiday"] = work.index.normalize().map(
        lambda d: d.date() in _RU_HOLIDAYS
    ).astype(int)
    work["is_halo"] = work.index.normalize().map(
        lambda d: d.date() in _HALO_DATES
    ).astype(int)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").astype("float64")
    return work, step_seconds


def _impute(work: pd.DataFrame, value_col: str, stl_period: int, gap_alpha: float):
    """
    Двухэтапная импутация:
    короткие пропуски — линейная интерполяция, длинные — RandomForest.
    """
    is_nan = work[value_col].isna()
    if not is_nan.any():
        return work

    nan_groups = (is_nan != is_nan.shift()).cumsum()
    gap_sizes = is_nan.groupby(nan_groups).transform("sum")
    gap_threshold = max(1, int(gap_alpha * stl_period))

    short_gap_mask = is_nan & (gap_sizes <= gap_threshold)
    long_gap_mask  = is_nan & (gap_sizes > gap_threshold)

    if short_gap_mask.any():
        work.loc[short_gap_mask, value_col] = (
            work[value_col].interpolate(method="time").loc[short_gap_mask]
        )

    if long_gap_mask.any():
        work_feats = pd.DataFrame(index=work.index)
        work_feats["_hour"] = work_feats.index.hour
        work_feats["_dow"]  = work_feats.index.dayofweek
        # Событийный контекст: восстановление дыр с учётом праздников/ореола/кампаний
        event_feats = [c for c in ("is_holiday", "is_halo", "is_campaign", "is_promo")
                       if c in work.columns]
        for c in event_feats:
            work_feats[c] = work[c].fillna(0)
        work_feats["_lag1"] = work[value_col].shift(1).fillna(work[value_col].median())

        features    = ["_hour", "_dow", *event_feats, "_lag1"]
        known_mask  = ~work[value_col].isna()
        if known_mask.any():
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(work_feats.loc[known_mask, features], work.loc[known_mask, value_col])
            work.loc[long_gap_mask, value_col] = rf.predict(
                work_feats.loc[long_gap_mask, features]
            )

    if work[value_col].isna().any():
        work[value_col] = work[value_col].bfill().ffill()

    return work


def _tukey_thresholds(series: pd.Series, k: float = 1.5):
    """Возвращает (нижний, верхний) порог по правилу Тьюки: Q1/Q3 ± k·IQR."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


# ─────────────────────────────────────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesCleaner:
    """
    Очистка временного ряда.

    Алгоритм (fit → transform):
      1. Нормализация временной оси: сортировка, дедупликация, равномерная сетка.
      2. Контекстное обогащение: признак is_holiday.
      3. Двухэтапная импутация: короткие пропуски — интерполяция, длинные — RF.
      4. RobustSTL: декомпозиция на тренд + сезонность + остаток.
      5. Метод Тьюки к остатку:
           - отрицательный выброс (провал/сбой) → медиана остатка.
           - положительный выброс на обычном дне  → медиана остатка.
           - положительный выброс в праздник/кампанию → сохранить.
      6. Рекомпозиция: trend + seasonal + residual_cleaned.
    """

    def __init__(
        self,
        stl_period: int = 24,
        tukey_k: float = 1.5,
        gap_alpha: float = 0.01,
    ):
        self.stl_period = stl_period
        self.tukey_k    = tukey_k
        self.gap_alpha  = gap_alpha

        self.seasonal_:          np.ndarray | None = None
        self.thresh_tukey_down_: float | None      = None
        self.thresh_tukey_up_:   float | None      = None
        self.median_residual_:   float             = 0.0
        self.step_seconds_:      int               = 3600

    # ------------------------------------------------------------------
    # Конвейер обучения
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        ts_col: str = "ds",
        value_col: str = "y",
    ) -> "TimeSeriesCleaner":
        """Извлекает сезонный профиль и пороги Тьюки из обучающих данных."""
        work, self.step_seconds_ = _parse_and_grid(df, ts_col, value_col)
        work = _impute(work, value_col, self.stl_period, self.gap_alpha)

        series = work[value_col].values
        stl_result = _STL(series, period=self.stl_period, robust=True).fit()
        trend    = stl_result.trend
        seasonal = stl_result.seasonal
        residual = stl_result.resid

        n_full = (len(seasonal) // self.stl_period) * self.stl_period
        if n_full >= self.stl_period:
            self.seasonal_ = seasonal[:n_full].reshape(-1, self.stl_period).mean(axis=0)
        else:
            self.seasonal_ = np.resize(seasonal, self.stl_period)

        resid_series = pd.Series(residual)
        holiday_mask = work["is_holiday"].values.astype(bool)
        halo_mask    = work["is_halo"].values.astype(bool)
        # Исключаем праздники И дни ореола: их пики — не шум, а легитимный сигнал
        normal_resid = resid_series[~holiday_mask & ~halo_mask]

        self.median_residual_    = float(normal_resid.median())
        self.thresh_tukey_down_, self.thresh_tukey_up_ = _tukey_thresholds(normal_resid, self.tukey_k)

        logger.info(
            "TimeSeriesCleaner обучен: period=%d, нижний порог Тьюки=%.4f",
            self.stl_period, self.thresh_tukey_down_,
        )
        return self

    # ------------------------------------------------------------------
    # Конвейер инференса
    # ------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        ts_col: str = "ds",
        value_col: str = "y",
    ) -> Tuple[pd.DataFrame, dict]:
        """Применяет очистку: импутация → декомпозиция → фильтр Тьюки → рекомпозиция."""
        if self.seasonal_ is None:
            raise RuntimeError("Вызовите fit() перед transform()")

        work, _ = _parse_and_grid(df, ts_col, value_col, min_step_seconds=self.step_seconds_)

        # Событийные колонки из df подвязываем ДО импутации, чтобы RF восстанавливал
        # длинные пропуски с учётом праздников/ореола/кампаний (без утечки — они
        # известны заранее по календарю). Берём только объявленные в EXOG_COLS.
        try:
            import config as _cfg
            event_cols = [c for c in getattr(_cfg, "EXOG_COLS", []) if c in df.columns]
        except (ImportError, Exception):
            event_cols = [c for c in df.columns if c.startswith("is_") and c != "is_holiday"]
        for ecol in event_cols:
            work[ecol] = (
                df.set_index(ts_col)[ecol]
                .reindex(work.index)
                .ffill().bfill().fillna(0)
                .astype(int)
                .values
            )

        work = _impute(work, value_col, self.stl_period, self.gap_alpha)

        series = work[value_col].values

        # Тренд: каузальное скользящее среднее (без заглядывания в будущее)
        trend = pd.Series(series).rolling(
            window=self.stl_period, min_periods=1, center=False
        ).mean().values

        # Сезонность: применяем сохранённый профиль по позиции в цикле
        time_deltas = (work.index - pd.Timestamp("2020-01-01")) // pd.Timedelta(seconds=self.step_seconds_)
        positions   = (time_deltas % self.stl_period).astype(int)
        seasonal    = self.seasonal_[positions]

        residual = series - trend - seasonal

        # Контекстные маски: праздники + ореол + все событийные флаги
        holiday_mask = work["is_holiday"].values.astype(bool)
        halo_mask    = work["is_halo"].values.astype(bool)
        event_mask   = holiday_mask | halo_mask
        for ecol in event_cols:
            if ecol in work.columns:
                event_mask |= work[ecol].values.astype(bool)

        # Тьюки к остатку: пороги из fit() — не из текущих данных (нет утечки)
        is_negative_outlier = residual < self.thresh_tukey_down_
        is_positive_outlier = residual > self.thresh_tukey_up_

        # Классификация:
        #   провал (вниз)            → шум (сбой инфраструктуры)
        #   рост на обычном дне      → шум
        #   рост в праздник/кампанию → легитимный пик, сохранить
        noise_mask      = is_negative_outlier | (is_positive_outlier & ~event_mask)
        legitimate_peak = is_positive_outlier & event_mask

        residual_clipped = residual.copy()
        residual_clipped[noise_mask] = self.median_residual_

        work[value_col] = trend + seasonal + residual_clipped

        out = work[[value_col]].copy()
        out.index.name = "ds"
        out = out.reset_index().rename(columns={value_col: "y"})

        # Флаги ореола — всегда передаём вниз по конвейеру (для FeatureBuilder и аналитики)
        out["is_holiday"] = work["is_holiday"].values
        out["is_halo"]    = work["is_halo"].values

        # Добавляем обратно экзогенные колонки из исходного df (по метке времени)
        try:
            import config as _cfg
            exog_cols_cfg = getattr(_cfg, "EXOG_COLS", [])
        except ImportError:
            exog_cols_cfg = []
        for ecol in exog_cols_cfg:
            if ecol not in df.columns:
                continue
            try:
                df_indexed = df.drop_duplicates(subset=[ts_col]).set_index(ts_col)
                ecol_series = (
                    df_indexed[ecol]
                    .reindex(work.index)
                    .ffill()
                    .bfill()
                    .fillna(0)
                )
                out[ecol] = ecol_series.values
            except Exception:
                out[ecol] = 0

        stats = {
            "n_output":           len(out),
            "n_outliers_removed": int(noise_mask.sum()),
            "n_peaks_preserved":  int(legitimate_peak.sum()),
            "residual":           pd.Series(residual_clipped, index=work.index),
        }
        logger.info(
            "transform: удалено шума=%d, легитимных пиков сохранено=%d",
            stats["n_outliers_removed"], stats["n_peaks_preserved"],
        )
        return out, stats

    # ------------------------------------------------------------------
    # Онлайн-очистка одной точки (инференс)
    # ------------------------------------------------------------------

    def clip_one(
        self,
        ts: pd.Timestamp,
        y: float,
        is_event: bool,
        recent_y: np.ndarray,
    ) -> float:
        """
        Очищает одну новую точку без полного STL.

        Использует каузальный тренд (среднее stl_period последних точек истории)
        и сохранённый сезонный профиль. Применяет пороги Тьюки из fit().

        Параметры
        ----------
        ts        : метка времени новой точки
        y         : сырое значение
        is_event  : True если точка в окне праздника/ореола (не обрезать)
        recent_y  : последние значения истории для расчёта тренда
        """
        if self.seasonal_ is None or self.thresh_tukey_up_ is None:
            return float(y)

        arr = np.asarray(recent_y, dtype=float)
        trend = float(np.nanmean(arr[-self.stl_period:])) if len(arr) >= 1 else float(y)

        pos = int(
            ((ts - pd.Timestamp("2020-01-01")).total_seconds() // self.step_seconds_)
            % self.stl_period
        )
        seasonal = float(self.seasonal_[pos])
        residual = float(y) - trend - seasonal

        if not is_event:
            if residual < self.thresh_tukey_down_:
                residual = self.median_residual_
            elif residual > self.thresh_tukey_up_:
                residual = self.thresh_tukey_up_

        return trend + seasonal + residual

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self, os.path.join(directory, _CLEANER_FILENAME))

    @classmethod
    def load(cls, directory: str) -> "TimeSeriesCleaner":
        return joblib.load(os.path.join(directory, _CLEANER_FILENAME))


# ─────────────────────────────────────────────────────────────────────────────
# Обратная совместимость
# ─────────────────────────────────────────────────────────────────────────────

def clean_timeseries(
    df: pd.DataFrame,
    ts_col: str = "ds",
    value_col: str = "y",
    stl_period: int = 24,
    tukey_k: float = 1.5,
    gap_alpha: float = 0.01,
) -> Tuple[pd.DataFrame, dict]:
    cleaner = TimeSeriesCleaner(stl_period=stl_period, tukey_k=tukey_k, gap_alpha=gap_alpha)
    cleaner.fit(df, ts_col=ts_col, value_col=value_col)
    return cleaner.transform(df, ts_col=ts_col, value_col=value_col)
