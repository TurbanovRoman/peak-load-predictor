import io
import os
import math
import gc
import tarfile
import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer

# =============================================================================
# КОНФИГУРАЦИЯ СТЕНДА
# =============================================================================
ARCHIVE    = "/root/sdv_project/azurefunctions-dataset2019.tar.xz"
OUT_CSV    = "/root/sdv_project/synthetic_azure.csv"
MODEL_PATH = "/root/sdv_project/par_model.pkl"

TOP_N      = 9       # Количество моделируемых приложений
EPOCHS     = 20      # Эпохи обучения PARSynthesizer
GEN_DAYS   = 30      # Итоговая длина синтезируемого ряда (в днях)
SEQ_DAYS   = 14      # Окно контекста модели (в днях)
CHUNK      = 1000    # Размер чанка при потоковом чтении логов
MAX_DAYS   = 14      # Сколько дней из оригинального архива берем для обучения

np.random.seed(42)   # Фиксация seed для воспроизводимости аномалий

# =============================================================================
# БЛОК 1: СТРИМИНГОВЫЙ ПАРСИНГ ЛОГОВ (Поиск Top-N приложений)
# =============================================================================
print("БЛОК 1: Определяем топ-приложения (потоковое чтение)...")
totals = pd.Series(dtype=float)
days_seen = 0

with tarfile.open(ARCHIVE, "r|xz") as tar:
    for member in tar:
        if not (member.isfile() and "invocations_per_function_md" in member.name):
            continue
        if days_seen >= MAX_DAYS:
            break
        
        days_seen += 1
        f = tar.extractfile(member)
        if f is None:
            continue
            
        buf = io.BytesIO(f.read())
        for chunk in pd.read_csv(buf, chunksize=CHUNK, low_memory=False):
            mc = [c for c in chunk.columns if c.isdigit()]
            totals = totals.add(
                chunk.groupby("HashApp")[mc].sum().sum(axis=1), fill_value=0
            )
        del buf
        gc.collect()

top_apps = set(totals.nlargest(TOP_N).index)
del totals
gc.collect()
print(f"  > Найдено {len(top_apps)} самых активных приложений.")

# =============================================================================
# БЛОК 2: ТРАНСФОРМАЦИЯ И МАСШТАБИРОВАНИЕ (Формирование RPS)
# =============================================================================
print("БЛОК 2: Извлечение временных рядов (преобразование в RPS)...")
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
        agg_chunks = []
        for chunk in pd.read_csv(buf, chunksize=CHUNK, low_memory=False):
            mc = [c for c in chunk.columns if c.isdigit()]
            chunk = chunk[chunk["HashApp"].isin(top_apps)]
            if len(chunk):
                agg_chunks.append(chunk.groupby("HashApp")[mc].sum())

        if not agg_chunks:
            continue

        agg = pd.concat(agg_chunks).groupby(level=0).sum()
        mc = [c for c in agg.columns if c.isdigit()]
        melted = agg.reset_index().melt(
            id_vars="HashApp", value_vars=mc,
            var_name="minute", value_name="rps"
        )
        
        # Формирование временной оси и перевод Invocations/min -> RPS
        melted["ds"] = pd.Timestamp("2019-01-01") + pd.to_timedelta(
            (day - 1) * 1440 + melted["minute"].astype(int), unit="min"
        )
        melted["rps"] /= 60.0
        parts.append(melted[["HashApp", "ds", "rps"]])
        
        del buf, agg, melted, agg_chunks
        gc.collect()
        print(f"  > Данные за день {day} успешно загружены.")

df = pd.concat(parts, ignore_index=True).sort_values(["HashApp", "ds"]).reset_index(drop=True)
df = df.rename(columns={"HashApp": "app_id"})
del parts
gc.collect()
print(f"  > Матрица сформирована: {len(df):,} строк.")

# =============================================================================
# БЛОК 3: ГЕНЕРАТИВНОЕ МОДЕЛИРОВАНИЕ (PARSynthesizer)
# =============================================================================
print("БЛОК 3: Обучение и генерация (PARSynthesizer)...")
metadata = Metadata.detect_from_dataframe(df)
metadata.update_column("ds",     sdtype="datetime", datetime_format="%Y-%m-%d %H:%M:%S")
metadata.update_column("app_id", sdtype="id")
metadata.update_column("rps",    sdtype="numerical", computer_representation="Float")
metadata.set_sequence_key("app_id")
metadata.set_sequence_index("ds")

# Обучение или загрузка модели
if os.path.exists(MODEL_PATH):
    print(f"  > Загрузка весов из {MODEL_PATH}...")
    syn = PARSynthesizer.load(MODEL_PATH)
else:
    print("  > Инициализация обучения модели (это может занять время)...")
    syn = PARSynthesizer(metadata, epochs=EPOCHS, verbose=True)
    syn.fit(df)
    syn.save(MODEL_PATH)
    print(f"  > Веса сохранены: {MODEL_PATH}")

del df
gc.collect()

# Генерация с хронологическим сдвигом
print("  > Запуск пакетной генерации...")
n_batches = math.ceil(GEN_DAYS / SEQ_DAYS)
parts_syn = []

for batch_idx in range(n_batches):
    print(f"    - Синтез батча {batch_idx + 1}/{n_batches}...")
    chunk_syn = syn.sample(
        num_sequences=TOP_N,
        sequence_length=SEQ_DAYS * 1440,
    )
    
    # СДВИГ ОСИ: Чтобы батчи выстраивались в единый длинный ряд
    chunk_syn["ds"] = pd.to_datetime(chunk_syn["ds"])
    time_shift = pd.Timedelta(days=batch_idx * SEQ_DAYS)
    chunk_syn["ds"] = chunk_syn["ds"] + time_shift
    
    parts_syn.append(chunk_syn)
    gc.collect()

synthetic = pd.concat(parts_syn, ignore_index=True)

# Усечение ровно до требуемого количества дней (GEN_DAYS)
max_allowed_date = synthetic["ds"].min() + pd.Timedelta(days=GEN_DAYS)
synthetic = synthetic[synthetic["ds"] < max_allowed_date]

del parts_syn
gc.collect()

# =============================================================================
# БЛОК 4: КОНТЕКСТНОЕ ОБОГАЩЕНИЕ И ИНЪЕКЦИЯ ИСКАЖЕНИЙ
# =============================================================================
print("БЛОК 4: Экзогенные признаки и инъекция аномалий...")

# Календарные фичи
synthetic["ds"]          = pd.to_datetime(synthetic["ds"])
synthetic["is_weekend"]  = synthetic["ds"].dt.dayofweek.isin([5, 6]).astype(int)
synthetic["hour"]        = synthetic["ds"].dt.hour
synthetic["day_of_week"] = synthetic["ds"].dt.dayofweek

# Тренд и базовая недельная сезонность
n = len(synthetic)
synthetic["month_trend"] = np.linspace(1.0, 1.3, n)
synthetic["rps"] *= synthetic["month_trend"]
synthetic["rps"] *= np.where(synthetic["is_weekend"], 0.75, 1.0)

# 4.1 Легитимные пики (event_mask) — целевая переменная для классификатора
# 1% точек - это запланированные события, увеличивающие трафик в 2.5 раза
synthetic["event_mask"] = np.random.choice([0, 1], size=n, p=[0.99, 0.01])
synthetic["rps"] = np.where(synthetic["event_mask"] == 1, synthetic["rps"] * 2.5, synthetic["rps"])

# 4.2 Технический шум — выбросы вверх (0.2% точек), ложные срабатывания
noise_mask = np.random.choice([0, 1], size=n, p=[0.998, 0.002])
synthetic["rps"] = np.where(noise_mask == 1, synthetic["rps"] * 3.0, synthetic["rps"])

# 4.3 Технические сбои — падение серверов (0.1% точек), просадки до 5% от нормы
crash_mask = np.random.choice([0, 1], size=n, p=[0.999, 0.001])
synthetic["rps"] = np.where(crash_mask == 1, synthetic["rps"] * 0.05, synthetic["rps"])

# Защита от отрицательных значений RPS после преобразований
synthetic["rps"] = synthetic["rps"].clip(lower=0)

# Финальное сохранение
synthetic.to_csv(OUT_CSV, index=False)
print(f"\n✅ ГОТОВО! Итоговый датасет сохранён: {OUT_CSV}")
print(f"Размерность: {synthetic.shape}. Количество дней в ряду: {GEN_DAYS}")