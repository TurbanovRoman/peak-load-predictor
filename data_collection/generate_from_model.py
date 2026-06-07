"""
Генерация synthetic_azure.csv из сохранённой PAR-модели.
Гибридный подход:
  1. PAR генерирует короткий «сид» (3 дня) — сохраняет стохастику модели.
  2. Статистическая аугментация расширяет сид до 30 дней — без многочасового сэмплинга.
"""

import gc
import math
import numpy as np
import pandas as pd
from sdv.sequential import PARSynthesizer

MODEL_PATH = r"C:\Users\qwesd\Desktop\Obshaya\Vuz 4 kurs\Diplome\proactive-scaler\data_collection\par_model.pkl"
OUT_CSV    = r"C:\Users\qwesd\Desktop\Obshaya\Vuz 4 kurs\Diplome\proactive-scaler\data\synthetic_azure.csv"

TOP_N    = 3
GEN_DAYS = 30
SEED_DAYS = 3   # сколько дней генерировать через PAR

# ── 1. Генерация сида через PAR ───────────────────────────────────────────
print(f"Загружаем модель из {MODEL_PATH}...")
syn = PARSynthesizer.load(MODEL_PATH)
print(f"Генерация сида ({SEED_DAYS} дней × {TOP_N} приложений)...")

seed = syn.sample(num_sequences=TOP_N, sequence_length=SEED_DAYS * 1440)
seed["ds"] = pd.to_datetime(seed["ds"])
seed = seed.sort_values(["app_id", "ds"]).reset_index(drop=True)
del syn
gc.collect()
print(f"  Сид: {len(seed):,} строк")

# ── 2. Статистическая аугментация сида до GEN_DAYS ───────────────────────
print("Аугментация сида до 30 дней...")
rng = np.random.default_rng(42)
n_variants = math.ceil(GEN_DAYS / SEED_DAYS)
parts_aug = [seed.copy()]

for variant in range(1, n_variants):
    aug = seed.copy()
    for app in aug["app_id"].unique():
        mask = aug["app_id"] == app
        scale = rng.uniform(0.85, 1.35)
        noise = rng.normal(0, aug.loc[mask, "rps"].std() * 0.05, mask.sum())
        aug.loc[mask, "rps"] = (aug.loc[mask, "rps"] * scale + noise).clip(lower=0)
    aug["ds"] = aug["ds"] + pd.Timedelta(days=SEED_DAYS * variant)
    parts_aug.append(aug)

synthetic = pd.concat(parts_aug, ignore_index=True).sort_values(["app_id", "ds"]).reset_index(drop=True)
del parts_aug, seed
gc.collect()

# ── 3. Экзогенные признаки ────────────────────────────────────────────────
synthetic["is_weekend"]  = synthetic["ds"].dt.dayofweek.isin([5, 6]).astype(int)
synthetic["hour"]        = synthetic["ds"].dt.hour
synthetic["day_of_week"] = synthetic["ds"].dt.dayofweek

n = len(synthetic)
synthetic["month_trend"] = np.linspace(1.0, 1.3, n)
synthetic["rps"] *= synthetic["month_trend"]
synthetic["rps"] *= np.where(synthetic["is_weekend"], 0.75, 1.0)
synthetic["rps"]         = synthetic["rps"].clip(lower=0)

synthetic.to_csv(OUT_CSV, index=False)
print(f"\nГотово: {len(synthetic):,} строк → {OUT_CSV}")
print(synthetic.head())
