"""
Точка входа системы прогнозирования пиковых нагрузок.

Режимы запуска:
  python main.py demo       -- РЕКОМЕНДУЕТСЯ: полный пайплайн всех экспериментов
                               (сравнение моделей, детекция пиков, дрейф, праздники)
  python main.py csv        -- обучение на готовом CSV (web_traffic.csv)
  python main.py train      -- XGBoost на Prometheus / синтетике
  python main.py compare    -- сравнение всех моделей
  python main.py simulate   -- симуляция проактивного масштабирования

Примеры:
  python main.py demo
  python main.py demo --fast    # без LSTM/NeuralProphet, только XGBoost
  python main.py csv
  python main.py csv --path data.csv --months 6
  python main.py train --synthetic
"""

import argparse
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
from preprocessing.feature_engineering import FeatureBuilder, split_train_val_test
from preprocessing.data_cleaning import TimeSeriesCleaner
from models.xgboost_model import (
    train_xgboost, predict_xgboost,
    get_confidence_interval, feature_importance,
)
from models.model_comparison import ModelComparison
from scaling.scaler import MomentumScaler
from evaluation.metrics import evaluate, plot_forecast, plot_all_forecasts
from evaluation.peak_detection import PeakDetector, IsolationForestAnomalyDetector


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_data(use_synthetic=False, days=None):
    """Загружает данные из Prometheus или генерирует синтетические."""
    if use_synthetic:
        from data_collection.synthetic_data import generate_synthetic_traffic
        days = days or config.SYNTHETIC_DAYS
        print(f"Генерация синтетических данных: {days} дней...")
        df = generate_synthetic_traffic(
            days=days,
            freq=config.DEFAULT_STEP,
            base_rps=config.SYNTHETIC_BASE_RPS,
            noise_std=config.SYNTHETIC_NOISE_STD,
            peak_probability=config.SYNTHETIC_PEAK_PROB,
            peak_multiplier=config.SYNTHETIC_PEAK_MULTIPLIER,
        )
    else:
        from data_collection.prometheus_connector import fetch_metric
        print("Загрузка данных из Prometheus...")
        df = fetch_metric(
            config.PROMETHEUS_QUERY,
            days_ago=days or config.DEFAULT_DAYS_AGO,
            step=config.DEFAULT_STEP,
        )
    print(f"Загружено {len(df)} точек | "
          f"RPS: min={df['y'].min():.0f}, max={df['y'].max():.0f}, "
          f"mean={df['y'].mean():.0f}")
    return df


def _apply_cleaner(df, n_test, n_val, save_dir):
    """Обучает TimeSeriesCleaner на train-части и применяет ко всему датасету."""
    n = len(df)
    n_train_end = max(n - n_test - n_val, n // 2)
    cleaner = TimeSeriesCleaner()
    cleaner.fit(df.iloc[:n_train_end])
    cleaned_df, stats = cleaner.transform(df)
    os.makedirs(save_dir, exist_ok=True)
    cleaner.save(save_dir)
    print(f"  Предобработка: шума удалено={stats['n_outliers_removed']}, "
          f"пиков сохранено={stats['n_peaks_preserved']}")
    return cleaned_df


# ---------------------------------------------------------------------------
# Генераторы синтетики для экспериментов 3 и 4
# ---------------------------------------------------------------------------

def _gen_drift_series(base_before=4000.0, base_after=6500.0,
                      days_before=30, days_drift=2, days_after=28,
                      freq="1h", noise_std=150.0, seed=42):
    """
    Временной ряд с концепт-дрейфом.
    days_before дней на base_before → days_drift линейного перехода
    → days_after дней на base_after.
    """
    rng = np.random.default_rng(seed)
    total_days = days_before + days_drift + days_after
    periods = int(pd.Timedelta(f"{total_days}D") / pd.tseries.frequencies.to_offset(freq))
    h_per_day = int(pd.Timedelta("1D") / pd.tseries.frequencies.to_offset(freq))
    timestamps = pd.date_range(start="2025-01-01", periods=periods, freq=freq)

    rps = np.empty(periods)
    for i in range(periods):
        day = i // h_per_day
        hour = timestamps[i].hour
        seasonal = 0.15 * np.sin(2 * np.pi * hour / 24)
        noise = rng.normal(0, noise_std)
        if day < days_before:
            base = base_before
        elif day < days_before + days_drift:
            frac = (day - days_before) / max(days_drift, 1)
            base = base_before + frac * (base_after - base_before)
        else:
            base = base_after
        rps[i] = max(0.0, base * (1 + seasonal) + noise)

    return pd.DataFrame({"ds": timestamps, "y": rps})


def _gen_holiday_series(days=730, freq="1h", base_rps=3000.0,
                        noise_std=80.0, seed=42):
    """
    Двухлетний временной ряд с праздничным ореолом.
    Праздники: 9 мая и 4 ноября.
    Эффект: нарастание за 3 дня до + пик + спад за 2 дня после.
    Возвращает DataFrame с колонками: ds, y, is_holiday, is_halo.
    """
    rng = np.random.default_rng(seed)
    periods = int(pd.Timedelta(f"{days}D") / pd.tseries.frequencies.to_offset(freq))
    timestamps = pd.date_range(start="2025-01-01", periods=periods, freq=freq)

    # Праздничные даты (9 мая, 4 ноября)
    holiday_dates = {}
    for year in range(2025, 2028):
        for m, d in [(5, 9), (11, 4)]:
            try:
                holiday_dates[pd.Timestamp(year=year, month=m, day=d).date()] = True
            except Exception:
                pass

    def _halo_mult(ts):
        """Мультипликатор нагрузки в зависимости от близости к ближайшему празднику."""
        for year in range(2025, 2028):
            for m, d in [(5, 9), (11, 4)]:
                try:
                    h = pd.Timestamp(year=year, month=m, day=d)
                    delta = (ts.normalize() - h).days
                    if -3 <= delta < 0:   # нарастание: 3 дня до
                        return 1.0 + 0.8 * (delta + 3) / 3
                    elif delta == 0:       # сам праздник
                        return 1.8
                    elif 0 < delta <= 2:   # спад: 2 дня после
                        return 1.8 - 0.4 * delta
                except Exception:
                    pass
        return 1.0

    rps_arr   = np.empty(periods)
    is_hol    = np.zeros(periods, dtype=int)
    is_halo   = np.zeros(periods, dtype=int)

    for i, ts in enumerate(timestamps):
        hour    = ts.hour
        weekday = ts.dayofweek
        seasonal = 0.3 * np.sin(2 * np.pi * hour / 24)
        weekly   = -0.2 if weekday >= 5 else 0.0
        noise    = rng.normal(0, noise_std)
        mult     = _halo_mult(ts)
        rps_arr[i] = max(0.0, base_rps * mult * (1 + seasonal + weekly) + noise)
        if ts.date() in holiday_dates:
            is_hol[i] = 1
        if mult > 1.0:
            is_halo[i] = 1

    return pd.DataFrame({"ds": timestamps, "y": rps_arr,
                          "is_holiday": is_hol, "is_halo": is_halo})


# ---------------------------------------------------------------------------
# Эксперимент 1: Сравнение моделей прогнозирования (Раздел 4.2)
# ---------------------------------------------------------------------------

def _run_model_comparison(save_dir, fast=False):
    from data_collection.synthetic_data import generate_synthetic_traffic

    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 1: Сравнение моделей прогнозирования")
    print("=" * 60)

    # 90 суток, шаг 1ч, 2160 наблюдений (как в разделе 4.2)
    df = generate_synthetic_traffic(
        days=90, freq="1h", base_rps=4000.0,
        noise_std=200.0, peak_probability=0.02,
        peak_multiplier=3.5, seed=42,
    )
    print(f"Данные: {len(df)} точек | "
          f"RPS min={df['y'].min():.0f} max={df['y'].max():.0f}")

    # Разбиение: 70 / 15 / 15
    TEST_H = int(90 * 24 * 0.15)
    VAL_H  = TEST_H
    df = _apply_cleaner(df, TEST_H, VAL_H, save_dir)
    train, val, test = split_train_val_test(df, test_hours=TEST_H, val_hours=VAL_H)
    print(f"Train={len(train)}, Val={len(val)}, Test={len(test)}")

    # Сравнение моделей
    comparator = ModelComparison(model_save_dir=save_dir)
    comparator.run(train, val, test,
                   include_lstm=not fast,
                   include_prophet=not fast)
    comparator.save_best()

    # XGBoost: доверительный интервал + важность признаков
    builder = FeatureBuilder()
    (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te) = builder.transform_splits(train, val, test)
    model_xgb = joblib.load(os.path.join(save_dir, "xgboost.pkl"))
    preds = predict_xgboost(model_xgb, X_te)
    print("\nОбучение квантильных моделей (доверительный интервал)...")
    lower, upper = get_confidence_interval(X_tr, y_tr, X_vl, y_vl, X_te, save_dir=save_dir)

    imp_df = feature_importance(model_xgb, feature_names=list(X_tr.columns))
    print("\nТоп-5 признаков:")
    print(imp_df.head(5).to_string(index=False))

    # Графики
    plot_forecast(train, val, test, preds, "XGBoost",
                  lower=lower, upper=upper,
                  save_path=os.path.join(save_dir, "e1_xgboost_forecast.png"),
                  zoom=False)
    plot_forecast(train, val, test, preds, "XGBoost",
                  lower=lower, upper=upper,
                  save_path=os.path.join(save_dir, "e1_xgboost_forecast_zoom.png"),
                  zoom=True)
    plot_all_forecasts(train, val, test,
                       predictions_dict=comparator.predictions_,
                       save_path=os.path.join(save_dir, "e1_all_models.png"),
                       zoom=False)

    # CSV
    metrics_rows = [{"model": n, **m} for n, m in comparator.results_.items()]
    pd.DataFrame(metrics_rows).to_csv(os.path.join(save_dir, "e1_metrics.csv"), index=False)
    imp_df.to_csv(os.path.join(save_dir, "e1_feature_importance.csv"), index=False)
    min_len = min(len(test), len(preds), len(lower), len(upper))
    pred_df = test[["ds", "y"]].iloc[:min_len].reset_index(drop=True)
    for name, mp in comparator.predictions_.items():
        pred_df[name] = np.array(mp[:min_len])
    pred_df["XGBoost_lower"] = lower[:min_len]
    pred_df["XGBoost_upper"] = upper[:min_len]
    pred_df.to_csv(os.path.join(save_dir, "e1_predictions.csv"), index=False)

    print(f"\nСохранено в {save_dir}/e1_*")
    return train, val, test, preds, lower, upper, comparator


# ---------------------------------------------------------------------------
# Эксперимент 2: Детекция пиков + аномалии (Раздел 4.3)
# ---------------------------------------------------------------------------

def _plot_peak_detection(events_df, save_path, ci_lower=None, ci_upper=None,
                         event_mask=None, min_rep=None, max_rep=None):
    """Двухпанельный график детекции пиков: прогноз+порог+уровни сверху, реплики снизу."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = events_df.copy()
    ts = pd.to_datetime(df["timestamp"]).to_numpy()
    min_rep = config.MIN_REPLICAS if min_rep is None else min_rep
    max_rep = config.MAX_REPLICAS if max_rep is None else max_rep

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # контекстные события (праздники/ореол)
    if event_mask is not None:
        em = np.asarray(event_mask, dtype=bool)
        ax1.fill_between(ts, float(df["rps"].min()), float(df["rps"].max()),
                         where=em, color="red", alpha=0.07, label="Контекстное событие")

    ax1.plot(ts, df["rps"],       color="gray",  lw=0.9, alpha=0.8, label="Факт")
    ax1.plot(ts, df["predicted"], color="green", lw=1.4,            label="Прогноз")
    ax1.plot(ts, df["threshold"], color="red",   lw=1.2, ls="--",   label="Адаптивный порог")

    # доверительный интервал Q90 (если передан)
    if ci_lower is not None and ci_upper is not None:
        ax1.fill_between(ts, np.asarray(ci_lower)[:len(ts)], np.asarray(ci_upper)[:len(ts)],
                         color="green", alpha=0.12, label="Q90 интервал")

    # уровни серьёзности
    for sev, col in {"warning": "gold", "critical": "darkorange", "exceeded": "red"}.items():
        m = (df["severity"].values == sev)
        if m.any():
            ax1.scatter(ts[m], df["predicted"].values[m], s=18, color=col, zorder=5, label=sev)

    # аномалии → выделяем треугольником, в этот момент реплик = max
    if "is_anomaly" in df.columns and df["is_anomaly"].any():
        m = df["is_anomaly"].values.astype(bool)
        ax1.scatter(ts[m], df["rps"].values[m], s=70, marker="^",
                    color="purple", zorder=6, label="аномалия")

    ax1.set_ylabel("RPS")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=10)
    ax1.grid(alpha=0.3)

    # нижняя панель — рекомендованное число реплик
    ax2.plot(ts, df["recommended_replicas"], color="steelblue", lw=1.3,
             drawstyle="steps-post", label="Рекомендовано реплик")
    ax2.axhline(min_rep, color="green", ls=":", lw=1.0, label=f"min={min_rep}")
    ax2.axhline(max_rep, color="red",   ls=":", lw=1.0, label=f"max={max_rep}")
    ax2.set_ylabel("Реплики"); ax2.set_xlabel("Время")
    ax2.legend(loc="upper right", frameon=False, fontsize=9, ncol=3)
    ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax2.get_xticklabels(), rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def _run_peak_detection(train, val, test, preds, lower, upper,
                        comparator, save_dir):
    from data_collection.synthetic_data import generate_synthetic_traffic

    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 2: Детекция пиковых нагрузок")
    print("=" * 60)

    builder = FeatureBuilder()
    _, _, (_, y_te) = builder.transform_splits(train, val, test)

    # Масштабируем target_per_replica под реальные RPS данных
    rps_max = float(train["y"].max())
    target_per_replica = rps_max / config.MAX_REPLICAS

    # --- Часть А: пороговый детектор (95-й перцентиль) + IsolationForest ---
    detector = PeakDetector(
        method="percentile",
        percentile=95.0,
        target_rps_per_replica=target_per_replica,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
        k_safety=1.5,
    )
    detector.fit(train["y"])

    # IsolationForest: обучаем на остатках тренировочной части
    X_tr_if, y_tr_if = builder.get_X_y(train)
    model_xgb = joblib.load(os.path.join(save_dir, "xgboost.pkl"))
    residuals_train = y_tr_if.values - predict_xgboost(model_xgb, X_tr_if)
    detector.fit_anomaly_detector(residuals_train)

    # Маска контекстных событий (праздники / ореол — не считать аномалией)
    campaign_mask = None
    if "is_halo" in test.columns:
        campaign_mask = pd.Series(
            (test["is_halo"].values.astype(bool) |
             test.get("is_holiday", pd.Series(0, index=test.index)).values.astype(bool)),
            index=y_te.index,
        )

    predicted_series = pd.Series(preds, index=y_te.index)
    upper_series     = pd.Series(upper[:len(y_te)], index=y_te.index)
    events_df = detector.detect_series(
        y_te, predicted_series,
        upper_ci_series=upper_series,
        campaign_mask=campaign_mask,
    )
    summary = detector.summary(events_df)

    print(f"  Метод: {summary['method']}  порог={summary['threshold']:.0f} RPS")
    print(f"  Пиков: {summary['peaks_detected']} / {summary['total_points']} "
          f"({summary['peak_ratio_pct']}%)")
    print(f"  Уровни: {summary['severity_counts']}")
    print(f"  Аномалий IF: {int(events_df['is_anomaly'].sum())}")

    if events_df["is_peak"].any():
        top5 = events_df[events_df["is_peak"]].head(5)
        print(f"\n  Первые 5 пиков (с рекомендациями по репликам):")
        for _, row in top5.iterrows():
            mark = "  [IF-аномалия]" if row["is_anomaly"] else ""
            print(f"    {row['timestamp']}  "
                  f"RPS факт={row['rps']:.0f}  прогноз={row['predicted']:.0f}  "
                  f"[{row['severity']}]  реплик={row['recommended_replicas']}{mark}")

    events_df.to_csv(os.path.join(save_dir, "e2_peak_events.csv"), index=False)
    _plot_peak_detection(
        events_df,
        save_path=os.path.join(save_dir, "e2_peak_detection.png"),
        ci_lower=lower[:len(events_df)],
        ci_upper=upper[:len(events_df)],
        event_mask=(campaign_mask.values if campaign_mask is not None else None),
    )

    # --- Часть Б: 60-дневный ряд с инъецированными аномалиями ---
    print("\n  [IF] Синтетический ряд с инъецированными аномалиями (60 дней)...")
    df_anom = generate_synthetic_traffic(
        days=60, freq="1h", base_rps=4000.0,
        peak_probability=0.04, peak_multiplier=4.5, seed=99,
    )
    df_anom = _apply_cleaner(df_anom, 240, 240, save_dir)
    tr_a, vl_a, te_a = split_train_val_test(df_anom, test_hours=240, val_hours=240)

    X_tr_a, y_tr_a = builder.get_X_y(tr_a)
    X_vl_a, y_vl_a = builder.get_X_y(vl_a)
    m_anom = train_xgboost(X_tr_a, y_tr_a, X_vl_a, y_vl_a)
    X_te_a, y_te_a = builder.get_X_y(te_a)
    preds_a  = predict_xgboost(m_anom, X_te_a)
    res_tr_a = y_tr_a.values - predict_xgboost(m_anom, X_tr_a)

    det_a = PeakDetector(
        method="percentile", percentile=95.0,
        target_rps_per_replica=target_per_replica,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
        k_safety=1.5,
    )
    det_a.fit(tr_a["y"])
    det_a.fit_anomaly_detector(res_tr_a)

    cam_a = None
    if "is_halo" in te_a.columns:
        cam_a = pd.Series(te_a["is_halo"].values.astype(bool), index=y_te_a.index)

    ev_a = det_a.detect_series(
        y_te_a, pd.Series(preds_a, index=y_te_a.index),
        campaign_mask=cam_a,
    )
    n_anom = int(ev_a["is_anomaly"].sum())
    n_peaks = int(ev_a["is_peak"].sum())
    print(f"  [IF] Аномалий обнаружено: {n_anom}, "
          f"пиков порогом: {n_peaks}, "
          f"реплик max: {int(ev_a['recommended_replicas'].max())}")
    ev_a.to_csv(os.path.join(save_dir, "e2_anomaly_events.csv"), index=False)
    _plot_peak_detection(
        ev_a,
        save_path=os.path.join(save_dir, "e2_anomaly_detection.png"),
        event_mask=(cam_a.values if cam_a is not None else None),
    )
    print(f"\nСохранено в {save_dir}/e2_*")


# ---------------------------------------------------------------------------
# Эксперимент 3: Адаптивное переобучение при концепт-дрейфе (Раздел 4.4)
# ---------------------------------------------------------------------------

def _run_drift_retraining(save_dir, fast=False):
    from retraining.drift_detector import ADWINDriftDetector
    from retraining.scheduler import make_multi_model_train_fn
    from evaluation.walk_forward import run_walk_forward

    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 3: Адаптивное переобучение при концепт-дрейфе")
    print("=" * 60)

    builder = FeatureBuilder()
    summary_rows = []

    # На каждом переобучении make_multi_model_train_fn обучает кандидатов на свежей
    # истории и оставляет ЛУЧШУЮ по MAE — т.е. модель для адаптивного прогноза
    # выбирается динамически, как и в сравнении (Эксп. 1).
    #
    # NeuralProphet (AR) тоже участвует: run_walk_forward синхронизирует его контекст
    # с недавней реальной историей (_sync_ar_context) → прогноз на 1 шаг от конца,
    # как в проде, без O(n²). Гиперпараметры заданы разумными значениями, чтобы не
    # запускать отдельный поиск на каждом ретрейне.
    xgb_params = dict(max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=3,
                      reg_alpha=0.1, reg_lambda=1.0)
    prophet_params = None if fast else dict(
        n_changepoints=20, trend_reg=0.1, seasonality_reg=0.1, n_lags=24, ar_reg=0.1)

    scenarios = [
        (0,  "Дрейф ДО тестового периода (модель устарела до теста)"),
        (2,  "Дрейф ВНУТРИ тестового периода (постепенный сдвиг)"),
    ]

    for days_drift, label in scenarios:
        print(f"\n  Сценарий: {label}")
        df_full = _gen_drift_series(
            base_before=4000.0, base_after=6500.0,
            days_before=30, days_drift=days_drift, days_after=28,
            freq="1h", noise_std=150.0, seed=42,
        )
        TEST_H = 28 * 24
        VAL_H  = 5 * 24
        df_clean = _apply_cleaner(df_full, TEST_H, VAL_H, save_dir)
        train, val, test = split_train_val_test(df_clean,
                                                test_hours=TEST_H, val_hours=VAL_H)
        print(f"    Train={len(train)}, Val={len(val)}, Test={len(test)}")

        wf_save = os.path.join(save_dir, f"e3_drift{days_drift}d")
        os.makedirs(wf_save, exist_ok=True)

        # train_fn переобучает XGBoost / LSTM / NeuralProphet и выбирает лучшую по MAE
        train_fn = make_multi_model_train_fn(
            builder, val_fraction=0.15, save_dir=wf_save,
            include_lstm=not fast, include_prophet=not fast,
            xgb_params=xgb_params, prophet_best_params=prophet_params,
        )

        # Начальная (фиксированная) модель = лучшая на train+val.
        # run_walk_forward ведёт её без обновления как baseline, а адаптивную —
        # с переобучением через тот же train_fn.
        init_seed = pd.concat([train, val]).sort_values("ds").reset_index(drop=True)
        initial_model, predict_fn, _ = train_fn(init_seed)

        drift_det = ADWINDriftDetector()
        result = run_walk_forward(
            train=train, val=val, test=test,
            initial_model=initial_model,
            predict_fn=predict_fn,
            train_fn=train_fn,
            builder=builder,
            drift_detector=drift_det,
            save_dir=wf_save,
            verbose=False,
        )
        s = result["summary"]
        print(f"    MAE фикс.={s['mae_baseline']:.1f} | адапт.={s['mae_adaptive']:.1f}  "
              f"(улучшение {s['improvement_pct']:.1f}%,  переобучений: {s['n_retrains']})")

        summary_rows.append({
            "label": label,
            "days_drift": days_drift,
            "mae_fixed":    round(s["mae_baseline"], 1),
            "mae_adaptive": round(s["mae_adaptive"], 1),
            "improvement_pct": round(s["improvement_pct"], 1),
            "n_retrains": s["n_retrains"],
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(save_dir, "e3_drift_summary.csv"), index=False)
    print(f"\nСохранено в {save_dir}/e3_*  "
          f"(графики дрейфа: e3_drift*/walk_forward_forecast.png)")


# ---------------------------------------------------------------------------
# Эксперимент 4: Прогноз нагрузки в период праздников (Раздел 4.5)
# ---------------------------------------------------------------------------

def _run_holiday_forecast(save_dir, fast=False):
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 4: Прогноз нагрузки в период праздников")
    print("=" * 60)

    if fast:
        print("  [FAST] Пропущен (требует NeuralProphet). Запустите без --fast.")
        return

    try:
        from models.forecasters import train_prophet, predict_prophet
    except ImportError:
        print("  Пропущен: models.forecasters не содержит train_prophet.")
        return

    # Двухлетний ряд с праздничным ореолом (9 мая, 4 ноября)
    df = _gen_holiday_series(days=730, freq="1h", base_rps=3000.0, seed=42)
    print(f"Данные: {len(df)} точек (2 года), "
          f"RPS min={df['y'].min():.0f} max={df['y'].max():.0f}")

    # Разбиение: последние 60 дней = тест + валидация
    TEST_H = 30 * 24
    VAL_H  = 30 * 24
    n      = len(df)
    train_df = df.iloc[:n - TEST_H - VAL_H][["ds", "y"]].copy()
    val_df   = df.iloc[n - TEST_H - VAL_H:n - TEST_H][["ds", "y"]].copy()
    test_df  = df.iloc[n - TEST_H:][["ds", "y"]].copy()
    print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    results = {}

    # --- Абляция 1: без праздничных признаков ---
    print("\n  [1/2] NeuralProphet БЕЗ праздничных признаков...")
    try:
        m_no   = train_prophet(train_df, val_df, use_holidays=False,
                                save_path=os.path.join(save_dir, "e4_prophet_no_hols"),
                                verbose=False)
        fc_no  = predict_prophet(m_no, periods=len(test_df))
        p_no   = np.clip(fc_no["yhat"].values[:len(test_df)], 0, None)
        r_no   = evaluate(test_df["y"].values[:len(p_no)], p_no,
                           "Prophet (без праздников)", verbose=True)
        results["без_праздников"] = r_no
    except Exception as e:
        print(f"    Ошибка: {e}")

    # --- Абляция 2: с праздничными признаками ---
    print("\n  [2/2] NeuralProphet С праздничными признаками (RU)...")
    try:
        m_yes  = train_prophet(train_df, val_df, use_holidays=True,
                                country_code="RU",
                                save_path=os.path.join(save_dir, "e4_prophet_with_hols"),
                                verbose=False)
        fc_yes = predict_prophet(m_yes, periods=len(test_df))
        p_yes  = np.clip(fc_yes["yhat"].values[:len(test_df)], 0, None)
        r_yes  = evaluate(test_df["y"].values[:len(p_yes)], p_yes,
                           "Prophet (с праздниками)", verbose=True)
        results["с_праздниками"] = r_yes
    except Exception as e:
        print(f"    Ошибка: {e}")

    if results:
        pd.DataFrame([{"вариант": k, **v} for k, v in results.items()]).to_csv(
            os.path.join(save_dir, "e4_holiday_ablation.csv"), index=False)
        print(f"\nСохранено в {save_dir}/e4_*")


# ---------------------------------------------------------------------------
# Режим: полный пайплайн всех экспериментов
# ---------------------------------------------------------------------------

def mode_demo(args):
    """
    Запускает все 4 эксперимента главы 4 в одном процессе:
      Эксперимент 1 — сравнение моделей (XGBoost / LSTM / NeuralProphet)
      Эксперимент 2 — детекция пиков + IsolationForest на аномалиях
      Эксперимент 3 — адаптивное переобучение при концепт-дрейфе (2 сценария)
      Эксперимент 4 — прогноз нагрузки в период праздников (абляция)
    """
    save_dir = config.MODEL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    t_start = time.time()
    print("=" * 60)
    print("ПОЛНЫЙ ДЕМОНСТРАЦИОННЫЙ ПАЙПЛАЙН")
    print("=" * 60)

    # Эксперимент 1: сравнение моделей
    train, val, test, preds, lower, upper, comparator = \
        _run_model_comparison(save_dir, fast=args.fast)

    # Эксперимент 2: детекция пиков + аномалий
    _run_peak_detection(train, val, test, preds, lower, upper, comparator, save_dir)

    # Эксперимент 3: адаптивное переобучение
    _run_drift_retraining(save_dir, fast=args.fast)

    # Эксперимент 4: праздники
    _run_holiday_forecast(save_dir, fast=args.fast)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ за {elapsed/60:.1f} мин")
    print(f"Результаты: {os.path.abspath(save_dir)}/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Режим: XGBoost + оценка
# ---------------------------------------------------------------------------

def mode_train(args):
    df = load_data(use_synthetic=args.synthetic)
    df = _apply_cleaner(df, config.SPLIT_TEST_HOURS, config.SPLIT_VAL_HOURS,
                        config.MODEL_SAVE_DIR)
    train, val, test = split_train_val_test(
        df, test_hours=config.SPLIT_TEST_HOURS, val_hours=config.SPLIT_VAL_HOURS)
    print(f"Train={len(train)}, Val={len(val)}, Test={len(test)}\n")

    builder = FeatureBuilder()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        builder.transform_splits(train, val, test)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    xgb_save = os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl")
    model = train_xgboost(
        X_train, y_train, X_val, y_val,
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        early_stopping_rounds=config.XGB_EARLY_STOPPING,
        save_path=xgb_save,
    )
    preds = predict_xgboost(model, X_test)
    evaluate(y_test.values, preds, "XGBoost", verbose=True)

    print("\nОбучение квантильных моделей...")
    lower, upper = get_confidence_interval(
        X_train, y_train, X_val, y_val, X_test, save_dir=config.MODEL_SAVE_DIR)

    imp_df = feature_importance(model, feature_names=list(X_train.columns))
    print("\nТоп-5 признаков:")
    print(imp_df.head(5).to_string(index=False))

    plot_forecast(train, val, test, preds, "XGBoost", lower=lower, upper=upper,
                  save_path=os.path.join(config.MODEL_SAVE_DIR, "forecast_xgboost.png"))
    plot_forecast(train, val, test, preds, "XGBoost", lower=lower, upper=upper,
                  save_path=os.path.join(config.MODEL_SAVE_DIR, "forecast_xgboost_zoom.png"),
                  zoom=True)

    # Детекция пиков
    rps_max = float(train["y"].max())
    detector = PeakDetector(
        method="percentile", percentile=95.0,
        target_rps_per_replica=rps_max / config.MAX_REPLICAS,
        min_replicas=config.MIN_REPLICAS, max_replicas=config.MAX_REPLICAS,
    )
    detector.fit(train["y"])
    ev_df = detector.detect_series(y_test, pd.Series(preds, index=y_test.index))
    s = detector.summary(ev_df)
    print(f"\n--- Детекция пиков ---  порог={s['threshold']:.0f}  "
          f"пиков={s['peaks_detected']}/{s['total_points']}  "
          f"{s['severity_counts']}")

    pd.DataFrame([{"model": "XGBoost",
                   "MAE": float(np.mean(np.abs(y_test.values - preds)))}]).to_csv(
        os.path.join(config.MODEL_SAVE_DIR, "metrics_comparison.csv"), index=False)
    imp_df.to_csv(os.path.join(config.MODEL_SAVE_DIR, "feature_importance.csv"), index=False)
    print(f"\nМодель: {xgb_save}")


# ---------------------------------------------------------------------------
# Режим: сравнение всех моделей
# ---------------------------------------------------------------------------

def mode_compare(args):
    df = load_data(use_synthetic=args.synthetic)
    df = _apply_cleaner(df, config.SPLIT_TEST_HOURS, config.SPLIT_VAL_HOURS,
                        config.MODEL_SAVE_DIR)
    train, val, test = split_train_val_test(
        df, test_hours=config.SPLIT_TEST_HOURS, val_hours=config.SPLIT_VAL_HOURS)
    print(f"Train={len(train)}, Val={len(val)}, Test={len(test)}\n")

    comparator = ModelComparison(model_save_dir=config.MODEL_SAVE_DIR)
    comparator.run(train, val, test,
                   include_prophet=not args.fast,
                   include_lstm=not args.fast)
    comparator.save_best()

    winner_name, winner_model = comparator.best_model()
    if winner_name == "XGBoost" and winner_model is not None:
        builder = FeatureBuilder()
        X_test, y_test = builder.get_X_y(test)
        preds = predict_xgboost(winner_model, X_test)
        rps_max = float(train["y"].max())
        det = PeakDetector(
            method="percentile", percentile=95.0,
            target_rps_per_replica=rps_max / config.MAX_REPLICAS,
            min_replicas=config.MIN_REPLICAS, max_replicas=config.MAX_REPLICAS,
        )
        det.fit(train["y"])
        ev_df = det.detect_series(y_test, pd.Series(preds, index=y_test.index))
        s = det.summary(ev_df)
        print(f"\n--- Детекция пиков ---  порог={s['threshold']:.0f}  "
              f"пиков={s['peaks_detected']}")


# ---------------------------------------------------------------------------
# Режим: симуляция проактивного масштабирования (бесконечный цикл)
# ---------------------------------------------------------------------------

def mode_simulate(args):
    model_path = os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl")
    if not os.path.exists(model_path):
        print("Модель не найдена. Запустите сначала: python main.py demo")
        sys.exit(1)

    model    = joblib.load(model_path)
    scaler   = MomentumScaler()
    builder  = FeatureBuilder()
    detector = PeakDetector(
        method=config.PEAK_METHOD,
        k=config.PEAK_K,
        target_rps_per_replica=config.TARGET_LOAD_PER_REPLICA,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
        warning_ratio=config.ALERT_WARNING_RATIO,
        critical_ratio=config.ALERT_CRITICAL_RATIO,
    )

    cleaner_path = os.path.join(config.MODEL_SAVE_DIR, "cleaner.pkl")
    sim_cleaner = (TimeSeriesCleaner.load(config.MODEL_SAVE_DIR)
                   if os.path.exists(cleaner_path) else None)
    if sim_cleaner is None:
        print("Предупреждение: cleaner.pkl не найден — запустите demo/csv сначала")

    print("Симуляция (Ctrl+C для остановки)...")
    iteration = 0
    while True:
        try:
            df = load_data(use_synthetic=args.synthetic, days=2)
            if len(df) < 30:
                time.sleep(60); continue

            if sim_cleaner is not None:
                df, _ = sim_cleaner.transform(df)

            detector.fit(df.iloc[:-1]["y"])
            X = builder.get_X(df)
            if len(X) == 0:
                time.sleep(60); continue

            next_load    = float(predict_xgboost(model, X.iloc[[-1]])[0])
            current_load = float(df["y"].iloc[-1])
            event = detector.detect(
                predicted_rps=next_load,
                current_rps=current_load,
                timestamp=pd.Timestamp.now(),
            )
            print(f"\n[{iteration}] {event}")
            scaler.scale(next_load)
            iteration += 1
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nСимуляция остановлена.")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(30)


# ---------------------------------------------------------------------------
# Режим: обучение на готовом CSV (web_traffic.csv)
# ---------------------------------------------------------------------------

def mode_csv(args):
    from data_collection.csv_loader import load_web_traffic

    print("=" * 60)
    print("ОБУЧЕНИЕ НА ДАННЫХ: web_traffic.csv")
    print("=" * 60)

    if args.path:
        from data_collection.csv_loader import load_csv
        df = load_csv(args.path,
                      timestamp_col=args.timestamp_col or None,
                      value_col=args.value_col or None)
    else:
        df = load_web_traffic(months=args.months)

    n = len(df)
    print(f"Загружено {n} точек | "
          f"RPS: min={df['y'].min():.0f}  max={df['y'].max():.0f}  "
          f"mean={df['y'].mean():.0f}")
    print(f"Период: {df['ds'].iloc[0]}  ->  {df['ds'].iloc[-1]}\n")

    if n < 200:
        print("Недостаточно данных.")
        sys.exit(1)

    TEST_H = min(480, n // 6)
    VAL_H  = TEST_H
    df = _apply_cleaner(df, TEST_H, VAL_H, config.MODEL_SAVE_DIR)
    train, val, test = split_train_val_test(df, test_hours=TEST_H, val_hours=VAL_H)
    print(f"Split: train={len(train)}  val={len(val)}  test={len(test)}\n")

    comparator = ModelComparison(model_save_dir=config.MODEL_SAVE_DIR)
    comparator.run(train, val, test,
                   include_prophet=not args.fast,
                   include_lstm=not args.fast)
    comparator.save_best()

    builder = FeatureBuilder()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        builder.transform_splits(train, val, test)
    model = joblib.load(os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl"))
    preds = predict_xgboost(model, X_test)

    print("\nОбучение квантильных моделей...")
    lower, upper = get_confidence_interval(
        X_train, y_train, X_val, y_val, X_test, save_dir=config.MODEL_SAVE_DIR)

    imp_df = feature_importance(model, feature_names=list(X_train.columns))
    print("\nВажность признаков:")
    print(imp_df.to_string(index=False))

    xgb_plot = os.path.join(config.MODEL_SAVE_DIR, "forecast_csv.png") if args.save_plots else None
    plot_forecast(train, val, test, preds, "XGBoost (CSV)",
                  lower=lower, upper=upper, save_path=xgb_plot)
    plot_all_forecasts(train, val, test,
                       predictions_dict=comparator.predictions_,
                       save_path=os.path.join(config.MODEL_SAVE_DIR, "comparison_csv.png"),
                       zoom=False)

    rps_max = float(train["y"].max())
    detector = PeakDetector(
        method="percentile", percentile=95.0,
        target_rps_per_replica=rps_max / config.MAX_REPLICAS,
        min_replicas=config.MIN_REPLICAS, max_replicas=config.MAX_REPLICAS,
    )
    detector.fit(train["y"])
    ev_df = detector.detect_series(y_test, pd.Series(preds, index=y_test.index))
    s = detector.summary(ev_df)
    print(f"\n--- Детекция пиков ---")
    print(f"  Порог: {s['threshold']:.0f} RPS (95-й перцентиль)")
    print(f"  Пиков: {s['peaks_detected']} / {s['total_points']} ({s['peak_ratio_pct']}%)")
    print(f"  Уровни: {s['severity_counts']}")

    if ev_df["is_peak"].any():
        top5 = ev_df[ev_df["is_peak"]].head(5)
        print(f"\n  Первые 5 пиков:")
        for _, row in top5.iterrows():
            print(f"    {row['timestamp']}  RPS={row['rps']:.0f}  "
                  f"прогноз={row['predicted']:.0f}  "
                  f"[{row['severity']}]  реплик={row['recommended_replicas']}")

    # CSV-экспорт
    metrics_rows = [{"model": nm, **m} for nm, m in comparator.results_.items()]
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(config.MODEL_SAVE_DIR, "metrics_comparison.csv"), index=False)
    min_len = min(len(test), len(preds), len(lower), len(upper))
    pred_df = test[["ds", "y"]].iloc[:min_len].reset_index(drop=True)
    for nm, mp in comparator.predictions_.items():
        pred_df[nm] = np.array(mp[:min_len])
    pred_df["XGBoost_lower"] = lower[:min_len]
    pred_df["XGBoost_upper"] = upper[:min_len]
    pred_df.to_csv(os.path.join(config.MODEL_SAVE_DIR, "predictions.csv"), index=False)
    imp_df.to_csv(os.path.join(config.MODEL_SAVE_DIR, "feature_importance.csv"), index=False)
    ev_df.to_csv(os.path.join(config.MODEL_SAVE_DIR, "peak_events.csv"), index=False)
    print(f"\nМодели сохранены в {config.MODEL_SAVE_DIR}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Система прогнозирования пиковых нагрузок (ВКР)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # demo — полный пайплайн
    p_demo = sub.add_parser("demo", help="Полный пайплайн всех экспериментов")
    p_demo.add_argument("--fast", action="store_true",
                        help="Только XGBoost (пропустить LSTM, NeuralProphet)")

    # train
    p_train = sub.add_parser("train", help="Обучить XGBoost и оценить")
    p_train.add_argument("--synthetic", action="store_true",
                         help="Синтетические данные вместо Prometheus")

    # compare
    p_cmp = sub.add_parser("compare", help="Сравнить все модели")
    p_cmp.add_argument("--synthetic", action="store_true")
    p_cmp.add_argument("--fast", action="store_true",
                       help="Только XGBoost")

    # simulate
    p_sim = sub.add_parser("simulate", help="Симуляция масштабирования (цикл)")
    p_sim.add_argument("--synthetic", action="store_true")

    # csv
    p_csv = sub.add_parser("csv",
                            help="Обучение на CSV-файле (web_traffic.csv)")
    p_csv.add_argument("--path", default=None,
                       help="Путь к CSV (по умолчанию: web_traffic.csv)")
    p_csv.add_argument("--timestamp-col", default=None)
    p_csv.add_argument("--value-col",     default=None)
    p_csv.add_argument("--months", type=int, default=12)
    p_csv.add_argument("--fast", action="store_true")
    p_csv.add_argument("--save-plots", action="store_true")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    dispatch = {
        "demo":     mode_demo,
        "train":    mode_train,
        "compare":  mode_compare,
        "simulate": mode_simulate,
        "csv":      mode_csv,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
