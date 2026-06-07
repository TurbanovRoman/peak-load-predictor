"""
Загрузчик данных из CSV-файла.

Читает web_traffic.csv, сгенерированный Code/ML.py (или любой CSV с временным рядом),
и приводит к стандартному формату {ds, y}, ожидаемому всей системой.

Поддерживаемые форматы:
  - Code/ML.py: timestamp, rps, concurrent_users, cpu_usage, memory_usage, latency_ms
  - Prometheus export: колонка ds и любая y-колонка
  - Любой CSV с одной datetime-колонкой и одной числовой
  - Автоопределение колонок если timestamp_col/value_col не указаны

Путь к данным по умолчанию: относительно proactive-scaler/
  "../Code/data/web_traffic.csv"
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Путь к датасету из Code/ML.py (относительно proactive-scaler)
DEFAULT_CSV_PATH = Path(__file__).parent.parent.parent / "Code" / "data" / "web_traffic.csv"


def _detect_timestamp_col(df: pd.DataFrame) -> str:
    """Возвращает имя первой колонки, которую удаётся распарсить как datetime."""
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col])
            if parsed.notna().mean() > 0.9:
                return col
        except Exception:
            continue
    raise ValueError(
        f"Не удалось автоматически определить колонку с датой/временем.\n"
        f"Доступные колонки: {list(df.columns)}\n"
        f"Укажите явно: --timestamp-col <имя>"
    )


def _detect_value_col(df: pd.DataFrame, timestamp_col: str) -> str:
    """Возвращает имя первой числовой колонки, не являющейся timestamp."""
    numeric_cols = [
        col for col in df.columns
        if col != timestamp_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        raise ValueError(
            f"Не найдено числовых колонок (кроме '{timestamp_col}').\n"
            f"Доступные колонки: {list(df.columns)}\n"
            f"Укажите явно: --value-col <имя>"
        )
    return numeric_cols[0]


def load_csv(
    path: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    value_col: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    extra_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Загружает CSV и возвращает DataFrame {ds, y[, extra_cols...]}.

    Parameters
    ----------
    path          : путь к CSV (None = web_traffic.csv)
    timestamp_col : колонка с датой/временем (None = автоопределение)
    value_col     : целевая метрика (None = первая числовая колонка)
    start, end    : фильтр по дате ("2023-06-01" и т.д.)
    extra_cols    : дополнительные колонки для сохранения (например,
                    ["is_holiday", "is_campaign", "is_promo"])

    Returns
    -------
    DataFrame с колонками ds (datetime), y (float) и extra_cols (если заданы).
    """
    csv_path = Path(path) if path else DEFAULT_CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV-файл не найден: {csv_path}\n"
            "Убедитесь что файл существует или укажите --path."
        )

    # Читаем без parse_dates — сначала автоопределим нужные колонки
    df = pd.read_csv(csv_path)

    ts_col = timestamp_col or _detect_timestamp_col(df)
    val_col = value_col or _detect_value_col(df, ts_col)

    if ts_col not in df.columns:
        raise ValueError(
            f"Колонка '{ts_col}' не найдена. Доступные: {list(df.columns)}"
        )
    if val_col not in df.columns:
        raise ValueError(
            f"Колонка '{val_col}' не найдена. Доступные: {list(df.columns)}"
        )

    print(f"  timestamp → '{ts_col}',  value → '{val_col}'")

    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.rename(columns={ts_col: "ds", val_col: "y"})
    keep = ["ds", "y"]
    if extra_cols:
        keep += [c for c in extra_cols if c in df.columns]
    df = df[keep].sort_values("ds").reset_index(drop=True)

    if start:
        df = df[df["ds"] >= pd.to_datetime(start)]
    if end:
        df = df[df["ds"] <= pd.to_datetime(end)]

    df["y"] = df["y"].astype("float64")
    return df.reset_index(drop=True)


def load_web_traffic(
    months: int = 12,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """
    Удобная обёртка: загружает web_traffic.csv за указанное число месяцев.

    Parameters
    ----------
    months : сколько месяцев взять начиная с start
    start  : начальная дата
    """
    end_dt = pd.to_datetime(start) + pd.DateOffset(months=months)
    return load_csv(
        timestamp_col="timestamp",
        value_col="rps",
        start=start,
        end=str(end_dt.date()),
    )


def describe_csv(path: Optional[str] = None) -> None:
    """Выводит статистику по датасету (удобно для проверки перед обучением)."""
    df = load_csv(path)
    n = len(df)
    step = (df["ds"].iloc[1] - df["ds"].iloc[0]) if n > 1 else None
    print(f"Файл:    {path or DEFAULT_CSV_PATH}")
    print(f"Строк:   {n}")
    print(f"Период:  {df['ds'].iloc[0]}  ->  {df['ds'].iloc[-1]}")
    print(f"Шаг:     {step}")
    print(f"RPS:     min={df['y'].min():.1f}  max={df['y'].max():.1f}  "
          f"mean={df['y'].mean():.1f}  std={df['y'].std():.1f}")


if __name__ == "__main__":
    describe_csv()
