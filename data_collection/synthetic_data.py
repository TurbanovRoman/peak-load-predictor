"""
Генератор синтетических данных трафика.

Позволяет запускать и тестировать систему прогнозирования без реального
Prometheus. Воспроизводит типичные паттерны веб-трафика:
  - суточная сезонность (пики утром и вечером)
  - недельная сезонность (спад в выходные)
  - долгосрочный тренд (линейный рост)
  - случайные пики нагрузки (запланированные и непредсказуемые)
  - гауссовый шум
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_synthetic_traffic(
    days: int = 30,
    freq: str = "1h",
    base_rps: float = 500.0,
    trend_per_day: float = 2.0,
    noise_std: float = 30.0,
    peak_probability: float = 0.02,
    peak_multiplier: float = 3.5,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Генерирует синтетический временной ряд RPS.

    Parameters
    ----------
    days            : количество дней в датасете
    freq            : шаг временного ряда (pandas offset alias, напр. '1h', '5min')
    base_rps        : базовый уровень RPS в «тихий» час
    trend_per_day   : прирост базового уровня RPS каждый день
    noise_std       : стандартное отклонение случайного шума
    peak_probability: вероятность случайного пика в каждый момент времени
    peak_multiplier : во сколько раз пик превышает текущий уровень
    seed            : seed для воспроизводимости

    Returns
    -------
    DataFrame с колонками ds (datetime) и y (RPS ≥ 0)
    """
    rng = np.random.default_rng(seed)

    periods = int(pd.Timedelta(f"{days}D") / pd.tseries.frequencies.to_offset(freq))
    timestamps = pd.date_range(start="2025-01-01", periods=periods, freq=freq)

    t = np.arange(periods)
    hours = timestamps.hour.values
    weekday = timestamps.dayofweek.values  # 0=Mon … 6=Sun

    # --- Тренд ---
    hours_per_day = int(pd.Timedelta("1D") / pd.tseries.frequencies.to_offset(freq))
    trend = trend_per_day * (t / hours_per_day)

    # --- Суточная сезонность (двойной пик: утро 9ч + вечер 19ч) ---
    daily = (
        0.6 * np.sin(2 * np.pi * (hours - 9) / 24)
        + 0.4 * np.sin(2 * np.pi * (hours - 19) / 24)
    )
    daily_amplitude = base_rps * 0.45
    daily_component = daily_amplitude * daily

    # --- Недельная сезонность (выходные ≈ 30% снижение) ---
    weekend_mask = (weekday >= 5).astype(float)
    weekly_component = -base_rps * 0.3 * weekend_mask

    # --- Шум ---
    noise = rng.normal(0, noise_std, size=periods)

    # --- Периодические пики (повторяются каждый день — модель их предсказывает) ---
    # Пик 1: каждый день в 10:00–11:00 (утренний трафик)
    # Пик 2: каждый день в 19:00–20:00 (вечерний прайм)
    # Пик 3: пятница 18:00 — недельный максимум

    peak_hour1 = 10
    peak_hour2 = 19
    peak_width = 2

    hour_float = timestamps.hour + timestamps.minute / 60.0
    is_weekday = (weekday < 5).astype(float)

    periodic_peaks = (
        base_rps * (peak_multiplier - 1) * is_weekday * (
            np.exp(-0.5 * ((hour_float - peak_hour1) / (peak_width / 2)) ** 2) * 0.7
            + np.exp(-0.5 * ((hour_float - peak_hour2) / (peak_width / 2)) ** 2)
        )
        # Пятничный вечерний всплеск
        + base_rps * (peak_multiplier - 1) * 0.5
        * (weekday == 4).astype(float)
        * np.exp(-0.5 * ((hour_float - 18) / 1.0) ** 2)
    )

    # --- Малые случайные аномалии поверх (непредсказуемые, ~2% точек) ---
    anomaly_mask = rng.random(size=periods) < peak_probability
    anomaly_values = np.where(anomaly_mask, (base_rps + trend) * 0.5, 0)

    rps = (
        base_rps
        + trend
        + daily_component
        + weekly_component
        + noise
        + periodic_peaks
        + anomaly_values
    )
    rps = np.clip(rps, 0, None)

    df = pd.DataFrame({"ds": timestamps, "y": rps})
    return df


def get_demo_split(
    days: int = 30,
    freq: str = "1h",
    test_hours: int = 48,
    val_hours: int = 48,
    seed: int = 42,
):
    """
    Удобная функция: генерирует датасет и сразу разбивает на train/val/test.

    Returns
    -------
    (df_full, train, val, test)
    """
    from preprocessing.feature_engineering import split_train_val_test

    df = generate_synthetic_traffic(days=days, freq=freq, seed=seed)
    train, val, test = split_train_val_test(df, test_hours=test_hours, val_hours=val_hours)
    return df, train, val, test


if __name__ == "__main__":
    df = generate_synthetic_traffic(days=14)
    print(df.describe())
    print(f"\nПериодов: {len(df)}")
    print(f"Диапазон: {df['ds'].min()} … {df['ds'].max()}")
    print(f"RPS min={df['y'].min():.1f}, max={df['y'].max():.1f}, mean={df['y'].mean():.1f}")
