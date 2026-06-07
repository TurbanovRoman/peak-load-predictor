"""
Генерация синтетических данных на основе Azure Functions Dataset 2019.
Тайлинг реальных 14 дней → 90 дней с шумом и событийными пиками.
"""

import io
import gc
import tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays as holidays_lib
from matplotlib.patches import Patch

# ── Параметры ──────────────────────────────────────────────────────────────
ARCHIVE  = "/content/drive/MyDrive/azurefunctions-dataset2019.tar.xz"
OUT_CSV  = "/content/drive/MyDrive/Colab Notebooks/SDV.csv"
PLOT_PNG = "/content/drive/MyDrive/Colab Notebooks/synthetic_plot.png"
CHUNK    = 10_000
MAX_DAYS = 14   # дней из архива для базового паттерна
GEN_DAYS = 90   # дней синтетики

# ── 1. Читаем реальные 14 дней (все приложения суммарно) ──────────────────
print("Читаем реальные данные из архива...")
parts = []
days_seen = 0

with tarfile.open(ARCHIVE, "r|xz") as tar:
    for member in tar:
        if not (member.isfile() and "invocations_per_function_md" in member.name):
            continue
        if days_seen >= MAX_DAYS:
            break
        day = int(member.name.split(".d")[-1].split(".")[0])
        days_seen += 1
        f = tar.extractfile(member)
        if f is None:
            continue

        buf = io.BytesIO(f.read())
        mc = None
        day_chunks = []

        for chunk in pd.read_csv(buf, chunksize=CHUNK, low_memory=False):
            if mc is None:
                mc = [c for c in chunk.columns if c.isdigit()]
            minute_sums = chunk[mc].sum() / 60
            minutes = [int(m) for m in mc]
            timestamps = [
                pd.Timestamp("2019-01-01") + pd.Timedelta(minutes=(day - 1) * 1440 + m)
                for m in minutes
            ]
            day_chunks.append(pd.DataFrame({"ds": timestamps, "rps": minute_sums.values}))

        parts.append(pd.concat(day_chunks, ignore_index=True))
        del buf
        gc.collect()
        print(f"  День {day} загружен")

base = (
    pd.concat(parts, ignore_index=True)
    .groupby("ds")["rps"].sum()
    .sort_index().reset_index()
)
del parts
gc.collect()

# Убеждаемся что в базе ровно MAX_DAYS * 1440 минут
assert len(base) == MAX_DAYS * 1440, f"Ожидалось {MAX_DAYS * 1440}, получено {len(base)}"

# Лёгкое сглаживание базы (5-мин rolling) для удаления одиночных выбросов
base["rps"] = base["rps"].rolling(5, center=True, min_periods=1).mean()
print(f"  База: {len(base):,} точек, RPS мин={base['rps'].min():.1f} макс={base['rps'].max():.1f}")

# ── 2. Тайлинг на 90 дней с мультипликативным шумом ──────────────────────
print(f"\nТайлинг {MAX_DAYS} дней → {GEN_DAYS} дней...")
rng = np.random.default_rng(42)
tiles = []
start_ts = pd.Timestamp("2023-01-01")

for day_offset in range(GEN_DAYS):
    src_start = (day_offset % MAX_DAYS) * 1440
    src_end   = src_start + 1440
    day_rps   = base["rps"].iloc[src_start:src_end].values.copy()

    # Мультипликативный шум ±6% (медленная компонента на день + быстрая на минуты)
    slow_noise = rng.normal(1.0, 0.04)                           # сдвиг всего дня
    fast_noise = rng.normal(1.0, 0.03, size=len(day_rps))        # минутная дрожь
    day_rps   *= slow_noise * fast_noise
    day_rps    = np.clip(day_rps, 0, None)

    ts = pd.date_range(
        start=start_ts + pd.Timedelta(days=day_offset),
        periods=len(day_rps),
        freq="1min",
    )
    tiles.append(pd.DataFrame({"ds": ts, "rps": day_rps}))

synthetic = pd.concat(tiles, ignore_index=True).sort_values("ds").reset_index(drop=True)
del tiles
gc.collect()
print(f"  Синтетика: {len(synthetic):,} точек")

# ── 3. Календарные признаки ───────────────────────────────────────────────
print("\nДобавляем признаки...")
synthetic["ds"] = pd.to_datetime(synthetic["ds"])

synthetic["is_weekend"]  = synthetic["ds"].dt.dayofweek.isin([5, 6]).astype(int)
synthetic["hour"]        = synthetic["ds"].dt.hour
synthetic["day_of_week"] = synthetic["ds"].dt.dayofweek

ru_holidays = holidays_lib.Russia(years=[2023])
synthetic["is_holiday"] = (
    synthetic["ds"].dt.normalize()
    .apply(lambda d: d.date() in ru_holidays)
    .astype(int)
)

# ── 4. Событийные пики ────────────────────────────────────────────────────
# is_campaign ×2.2 : 1 событие в train (дни 10–70) + 1 в test (день 79)
# is_promo    ×1.6 : 3 события в train (дни 10–70) + 1 в test (день 83)

campaign_train = rng.choice(range(10, 70), size=1, replace=False)
campaign_dates = set()
for d in list(campaign_train) + [79]:
    for offset in range(3):
        campaign_dates.add(
            (pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(d) + offset)).normalize()
        )
synthetic["is_campaign"] = synthetic["ds"].dt.normalize().isin(campaign_dates).astype(int)

used_days  = {int(d) + o for d in campaign_train for o in range(3)}
promo_pool = [d for d in range(10, 70) if not any((d + o) in used_days for o in range(3))]
promo_train = rng.choice(promo_pool, size=3, replace=False)
promo_dates = set()
for d in list(promo_train) + [83]:
    for offset in range(3):
        promo_dates.add(
            (pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(d) + offset)).normalize()
        )
synthetic["is_promo"] = synthetic["ds"].dt.normalize().isin(promo_dates).astype(int)

print(f"  Праздничных точек:  {synthetic['is_holiday'].sum():,}")
print(f"  Кампанийных точек:  {synthetic['is_campaign'].sum():,}  (1 train + 1 test, 3 дня каждое)")
print(f"  Промо-точек:        {synthetic['is_promo'].sum():,}    (3 train + 1 test, 3 дня каждое)")

# ── 5. Плавные огибающие (трапеция 2 ч нарастание / 2 ч спад) ────────────
RAMP_MIN = 120

def make_smooth_multiplier(series_ds, date_set, peak_mult, ramp_min):
    mult = pd.Series(1.0, index=series_ds.index)
    for date in date_set:
        mask = series_ds.dt.normalize() == date
        if not mask.any():
            continue
        idx   = np.where(mask)[0]
        n_day = len(idx)
        env   = np.ones(n_day) * peak_mult
        r     = min(ramp_min, n_day // 4)
        env[:r]  = np.linspace(1.0, peak_mult, r)
        env[-r:] = np.linspace(peak_mult, 1.0, r)
        mult.iloc[idx[0]: idx[-1] + 1] = env
    return mult

campaign_mult = make_smooth_multiplier(synthetic["ds"], campaign_dates, 2.2, RAMP_MIN)
promo_mult    = make_smooth_multiplier(synthetic["ds"], promo_dates,    1.6, RAMP_MIN)

# ── 6. Применяем все множители к rps ─────────────────────────────────────
n = len(synthetic)
trend = np.linspace(1.0, 1.3, n)   # +30% за 90 дней

synthetic["rps"] *= trend
synthetic["rps"] *= np.where(synthetic["is_weekend"], 0.75, 1.0)
synthetic["rps"] *= np.where(synthetic["is_holiday"],  1.8, 1.0)
synthetic["rps"] *= campaign_mult.values
synthetic["rps"] *= promo_mult.values
synthetic["rps"]  = synthetic["rps"].clip(lower=0)

synthetic = synthetic.sort_values("ds").reset_index(drop=True)

# ── 7. Сохраняем ──────────────────────────────────────────────────────────
synthetic.to_csv(OUT_CSV, index=False)
print(f"\nГотово: {len(synthetic):,} строк → {OUT_CSV}")
print(f"Колонки: {list(synthetic.columns)}")
print(synthetic.head())

# ── 8. График ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(synthetic["ds"], synthetic["rps"], linewidth=0.4, alpha=0.85, color="steelblue")

for date in sorted(campaign_dates):
    seg = synthetic[synthetic["ds"].dt.normalize() == date]
    if not seg.empty:
        ax.axvspan(seg["ds"].iloc[0], seg["ds"].iloc[-1], color="red", alpha=0.18)

for date in sorted(promo_dates):
    seg = synthetic[synthetic["ds"].dt.normalize() == date]
    if not seg.empty:
        ax.axvspan(seg["ds"].iloc[0], seg["ds"].iloc[-1], color="orange", alpha=0.18)

ax.legend(handles=[
    Patch(color="steelblue", alpha=0.8,  label="RPS (синтетика)"),
    Patch(color="red",        alpha=0.35, label="campaign ×2.2"),
    Patch(color="orange",     alpha=0.35, label="promo ×1.6"),
])
ax.set_title("Синтетический RPS на основе Azure Functions 2019 (90 дней)")
ax.set_xlabel("Дата")
ax.set_ylabel("RPS (запросов/сек)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=120)
plt.show()
print(f"График → {PLOT_PNG}")
