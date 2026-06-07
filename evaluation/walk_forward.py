"""
Walk-forward валидация с адаптивным переобучением (глава 4 ВКР).

Реализует скользящую оценку: модель прогнозирует точку за точкой,
при обнаружении концепт-дрейфа переобучается на накопленной истории.
Параллельно запускается фиксированная модель-baseline для сравнения.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd


def run_walk_forward(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    initial_model,
    predict_fn,
    train_fn,
    builder,
    drift_detector,
    save_dir: str,
    cleaner=None,
    verbose: bool = True,
) -> dict:
    """
    Walk-forward цикл по тестовой выборке.

    Parameters
    ----------
    train, val, test   : очищенные DataFrame с колонками ds, y
    initial_model      : обученная модель (XGBoost или другая)
    predict_fn         : функция predict(model, X) -> np.ndarray
    train_fn           : функция для переобучения (из make_multi_model_train_fn)
    builder            : FeatureBuilder
    drift_detector     : ADWINDriftDetector или PerformanceDriftDetector
    save_dir           : директория для сохранения результатов
    verbose            : печатать прогресс

    Returns
    -------
    dict с ключами: results_df, retrain_timestamps, summary
    """
    from retraining.scheduler import RetrainScheduler
    from models.forecasters import predict_xgboost

    audit_path = os.path.join(save_dir, "walk_forward_log.csv")
    scheduler = RetrainScheduler(
        initial_model=initial_model,
        predict_fn=predict_fn,
        train_fn=train_fn,
        builder=builder,
        drift_detector=drift_detector,
        audit_path=audit_path,
        cooldown_seconds=0.0,
        history_buffer_size=20000,  # 20000 мин ≈ 14 суток — достаточно для 168h лага
    )

    # Засеваем историю обучающими данными
    history_seed = pd.concat([train, val]).sort_values("ds")
    scheduler.seed_history(history_seed)
    X_boot = builder.get_X(history_seed)
    y_boot = history_seed["y"].iloc[-len(X_boot):]
    preds_boot = predict_fn(initial_model, X_boot)
    drift_detector.set_baseline(np.abs(y_boot.values - preds_boot))

    if verbose:
        print(f"Baseline MAE: {drift_detector.baseline_mae:.3f}\n")

    records = []
    baseline_records = []
    retrain_timestamps = []
    retrain_intervals = []
    last_retrain_i = 0

    # Список вместо растущего DataFrame: O(1) append, без pd.concat на каждом шаге.
    # Для построения признаков берём только последние _FEAT_WINDOW точек —
    # достаточно для самого длинного лага (168 ч = 10080 мин).
    _FEAT_WINDOW = 12000
    history_records: list = history_seed.reset_index(drop=True).to_dict("records")

    def _history_df() -> pd.DataFrame:
        return pd.DataFrame(history_records[-_FEAT_WINDOW:])

    for i, row in test.reset_index(drop=True).iterrows():
        current_ts = pd.Timestamp(row["ds"])
        y_true = float(row["y"])

        hist_df = _history_df()
        y_pred = scheduler.predict_one(hist_df)
        X_last = builder.get_X(hist_df)
        y_pred_baseline = float(
            np.asarray(predict_fn(initial_model, X_last)).flatten()[0]
        ) if len(X_last) > 0 else float("nan")

        if np.isnan(y_pred):
            history_records.append({"ds": current_ts, "y": y_true})
            continue

        mae_step = abs(y_true - y_pred)
        mae_baseline = abs(y_true - y_pred_baseline) if not np.isnan(y_pred_baseline) else np.nan

        scheduler.observe(ts=current_ts, y_true=y_true, y_pred=y_pred)
        event = scheduler.check_and_retrain()
        if event is not None:
            retrain_timestamps.append(current_ts)
            retrain_intervals.append(i - last_retrain_i)
            last_retrain_i = i

            if cleaner is not None and hasattr(cleaner, "seasonal_") \
                    and cleaner.seasonal_ is not None:
                season_len = len(cleaner.seasonal_)
                n_hist = len(history_records)
                if n_hist >= 2 * season_len:
                    cleaner.fit(pd.DataFrame(history_records[-max(2 * season_len, n_hist):]))
                    if verbose:
                        print(f"  [cleaner.fit] сезонная составляющая обновлена "
                              f"(history={n_hist}, season_len={season_len})")

            if verbose:
                print(f"  [retrain] шаг {i}: MAE {event.rolling_mae_before:.2f} → "
                      f"{event.new_baseline_mae:.2f} "
                      f"(причина: {event.reason}, "
                      f"интервал: {retrain_intervals[-1]} шагов)")

        scheduler.check_rollback()

        records.append({
            "timestamp": current_ts,
            "y_true": y_true,
            "y_pred": y_pred,
            "mae": mae_step,
            "retrain": event is not None,
        })
        baseline_records.append({
            "timestamp": current_ts,
            "y_pred_baseline": y_pred_baseline,
            "mae_baseline": mae_baseline,
        })

        history_records.append({"ds": current_ts, "y": y_true})

        if verbose and i % 50 == 0:
            print(f"  шаг {i}/{len(test)}, MAE адапт.={mae_step:.2f}, "
                  f"MAE baseline={mae_baseline:.2f}")

    results_df = pd.DataFrame(records)
    baseline_df = pd.DataFrame(baseline_records)

    csv_out = os.path.join(save_dir, "walk_forward_results.csv")
    results_df.to_csv(csv_out, index=False)

    total_mae_adaptive = results_df["mae"].mean()
    total_mae_baseline = baseline_df["mae_baseline"].mean()
    improvement = (
        (total_mae_baseline - total_mae_adaptive) / total_mae_baseline * 100
        if total_mae_baseline > 0 else 0.0
    )

    summary = {
        "mae_adaptive":       round(total_mae_adaptive, 3),
        "mae_baseline":       round(total_mae_baseline, 3),
        "improvement_pct":    round(improvement, 1),
        "n_retrains":         len(retrain_timestamps),
        "retrain_intervals":  retrain_intervals,
        "retrain_timestamps": retrain_timestamps,
    }

    if verbose:
        print(f"\n--- Итог walk-forward ---")
        print(f"  MAE адаптивная:   {summary['mae_adaptive']}")
        print(f"  MAE фиксированная:{summary['mae_baseline']}")
        print(f"  Улучшение:        {summary['improvement_pct']}%")
        print(f"  Переобучений:     {summary['n_retrains']}")

    plot_walk_forward(results_df, baseline_df, retrain_timestamps, save_dir)

    return {
        "results_df":  results_df,
        "baseline_df": baseline_df,
        "summary":     summary,
    }


def plot_walk_forward(
    results_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    retrain_timestamps: list,
    save_dir: str,
) -> str:
    """График MAE по времени с метками переобучения."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    rolling_adaptive = results_df["mae"].rolling(24, min_periods=1).mean()
    rolling_baseline = baseline_df["mae_baseline"].rolling(24, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(results_df["timestamp"], rolling_adaptive, color="steelblue",
            linewidth=1.5, label="Адаптивная модель (с переобучением)")
    ax.plot(baseline_df["timestamp"], rolling_baseline, color="orange",
            linewidth=1.5, linestyle="--", label="Фиксированная модель")

    for i, ts in enumerate(retrain_timestamps):
        ax.axvline(ts, color="crimson", linestyle=":", linewidth=1.0, alpha=0.7,
                   label="Переобучение" if i == 0 else None)

    ax.set_xlabel("Время")
    ax.set_ylabel("MAE (скользящее среднее 24 шага)")
    ax.set_title("Walk-forward валидация: адаптивная vs фиксированная модель")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=30)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "walk_forward_mae.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path
