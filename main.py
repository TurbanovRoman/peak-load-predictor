"""
Точка входа системы прогнозирования пиковых нагрузок.

Режимы запуска:
  python main.py csv       -- РЕКОМЕНДУЕТСЯ: обучение на готовом CSV (web_traffic.csv)
  python main.py train     -- обучение моделей (Prometheus или синтетика)
  python main.py compare   -- сравнение всех моделей с таблицей результатов
  python main.py simulate  -- симуляция проактивного масштабирования
  python main.py demo      -- полный пайплайн на синтетических данных (без Prometheus)

Примеры:
  python main.py csv                     # web_traffic.csv, сравнение моделей
  python main.py csv --fast              # без NeuralProphet и LSTM
  python main.py csv --months 3          # только первые 3 месяца
  python main.py demo
  python main.py train --synthetic
"""
g
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
from models.xgboost_model import (
    train_xgboost, predict_xgboost,
    get_confidence_interval, feature_importance,
)
from models.model_comparison import ModelComparison
from scaling.scaler import MomentumScaler
from evaluation.metrics import evaluate, plot_forecast, plot_all_forecasts
from evaluation.peak_detection import PeakDetector


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_data(use_synthetic: bool = False, days: int = None) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Режим: обучение одной модели (XGBoost) + оценка
# ---------------------------------------------------------------------------

def mode_train(args):
    df = load_data(use_synthetic=args.synthetic)
    train, val, test = split_train_val_test(
        df,
        test_hours=config.SPLIT_TEST_HOURS,
        val_hours=config.SPLIT_VAL_HOURS,
    )
    print(f"Train={len(train)}, Val={len(val)}, Test={len(test)} точек\n")

    builder = FeatureBuilder()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        builder.transform_splits(train, val, test)

    # --- Обучение XGBoost ---
    print("--- XGBoost ---")
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

    # --- Доверительный интервал ---
    print("\nОбучение квантильных моделей (доверительный интервал)...")
    lower, upper = get_confidence_interval(
        X_train, y_train, X_val, y_val, X_test,
        save_dir=config.MODEL_SAVE_DIR,
    )

    # --- Важность признаков ---
    imp_df = feature_importance(model, feature_names=list(X_train.columns))
    print("\nТоп-5 важных признаков:")
    print(imp_df.head(5).to_string(index=False))

    # --- Графики ---
    plot_forecast(
        train, val, test, preds, "XGBoost", lower=lower, upper=upper,
        save_path=os.path.join(config.MODEL_SAVE_DIR, "forecast_xgboost_full.png"),
        zoom=False,
    )
    plot_forecast(
        train, val, test, preds, "XGBoost", lower=lower, upper=upper,
        save_path=os.path.join(config.MODEL_SAVE_DIR, "forecast_xgboost_zoom.png"),
        zoom=True,
    )

    # --- Детекция пиков ---
    _run_peak_detection(train, y_test, preds)

    # --- CSV экспорт ---
    metrics = {"model": "XGBoost"}
    from evaluation.metrics import safe_mape
    from sklearn.metrics import mean_absolute_error
    metrics["MAE"] = float(mean_absolute_error(y_test.values, preds))
    metrics["MAPE"] = safe_mape(y_test.values, preds)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(config.MODEL_SAVE_DIR, "metrics_comparison.csv"), index=False
    )

    pred_df = test[["ds", "y"]].iloc[:len(preds)].copy().reset_index(drop=True)
    pred_df["XGBoost"] = preds
    pred_df["XGBoost_lower"] = lower[:len(pred_df)]
    pred_df["XGBoost_upper"] = upper[:len(pred_df)]
    pred_df.to_csv(
        os.path.join(config.MODEL_SAVE_DIR, "predictions.csv"), index=False
    )

    imp_df.to_csv(
        os.path.join(config.MODEL_SAVE_DIR, "feature_importance.csv"), index=False
    )

    print(f"\nМодель сохранена: {xgb_save}")
    print(f"Графики и CSV: {config.MODEL_SAVE_DIR}/")


# ---------------------------------------------------------------------------
# Режим: сравнение всех моделей
# ---------------------------------------------------------------------------

def mode_compare(args):
    df = load_data(use_synthetic=args.synthetic)
    train, val, test = split_train_val_test(
        df,
        test_hours=config.SPLIT_TEST_HOURS,
        val_hours=config.SPLIT_VAL_HOURS,
    )
    print(f"Train={len(train)}, Val={len(val)}, Test={len(test)}\n")

    comparator = ModelComparison(model_save_dir=config.MODEL_SAVE_DIR)
    comparator.run(
        train, val, test,
        include_prophet=not args.fast,
        include_lstm=not args.fast,
    )

    # Сохраняем лучшую модель
    comparator.save_best()

    # Детекция пиков с лучшей моделью
    winner_name, winner_model = comparator.best_model()
    if winner_name == "XGBoost" and winner_model is not None:
        builder = FeatureBuilder()
        X_test, y_test = builder.get_X_y(test)
        preds = winner_model.predict(X_test)
        _run_peak_detection(train, y_test, preds)


# ---------------------------------------------------------------------------
# Режим: симуляция (реальное время)
# ---------------------------------------------------------------------------

def mode_simulate(args):
    model_path = os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl")
    if not os.path.exists(model_path):
        print("Модель не найдена. Запустите сначала: python main.py train")
        sys.exit(1)

    model = joblib.load(model_path)
    scaler = MomentumScaler()
    builder = FeatureBuilder()
    detector = PeakDetector(
        method=config.PEAK_METHOD,
        k=config.PEAK_K,
        target_rps_per_replica=config.TARGET_LOAD_PER_REPLICA,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
        warning_ratio=config.ALERT_WARNING_RATIO,
        critical_ratio=config.ALERT_CRITICAL_RATIO,
    )

    use_synthetic = args.synthetic
    print("Запуск симуляции проактивного масштабирования (Ctrl+C для остановки)...")
    iteration = 0

    while True:
        try:
            df = load_data(use_synthetic=use_synthetic, days=2)

            if len(df) < 30:
                print("Недостаточно данных, ожидание...")
                time.sleep(60)
                continue

            # Инициализируем детектор по последним 24ч
            history = df.iloc[:-1]["y"]
            detector.fit(history)

            X = builder.get_X(df)
            if len(X) == 0:
                time.sleep(60)
                continue

            next_load = float(predict_xgboost(model, X.iloc[[-1]])[0])
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
# Режим: полный демонстрационный пайплайн (без Prometheus)
# ---------------------------------------------------------------------------

def mode_demo(args):
    """
    Полный пайплайн на синтетических данных:
    1. Генерация данных
    2. Сравнение моделей
    3. Доверительный интервал XGBoost
    4. Детекция пиков
    5. График
    """
    print("=" * 60)
    print("ДЕМОНСТРАЦИОННЫЙ РЕЖИМ (синтетические данные)")
    print("=" * 60)

    from data_collection.synthetic_data import generate_synthetic_traffic

    df = generate_synthetic_traffic(
        days=config.SYNTHETIC_DAYS,
        freq=config.SYNTHETIC_FREQ,
        base_rps=config.SYNTHETIC_BASE_RPS,
        noise_std=config.SYNTHETIC_NOISE_STD,
        peak_probability=config.SYNTHETIC_PEAK_PROB,
        peak_multiplier=config.SYNTHETIC_PEAK_MULTIPLIER,
    )
    print(f"\nДанные: {len(df)} точек, "
          f"RPS min={df['y'].min():.0f} max={df['y'].max():.0f}\n")

    train, val, test = split_train_val_test(
        df,
        test_hours=config.SPLIT_TEST_HOURS,
        val_hours=config.SPLIT_VAL_HOURS,
    )

    # Сравнение моделей
    comparator = ModelComparison(model_save_dir=config.MODEL_SAVE_DIR)
    comparator.run(train, val, test,
                   include_prophet=not args.fast,
                   include_lstm=not args.fast)
    comparator.save_best()

    # XGBoost + доверительный интервал
    builder = FeatureBuilder()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        builder.transform_splits(train, val, test)

    model = joblib.load(os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl"))
    preds = predict_xgboost(model, X_test)

    lower, upper = get_confidence_interval(
        X_train, y_train, X_val, y_val, X_test,
        save_dir=config.MODEL_SAVE_DIR,
    )

    plot_forecast(
        train, val, test, preds, "XGBoost",
        lower=lower, upper=upper,
        save_path=os.path.join(config.MODEL_SAVE_DIR, "forecast_xgboost_full.png"),
        zoom=False,
    )
    plot_forecast(
        train, val, test, preds, "XGBoost",
        lower=lower, upper=upper,
        save_path=os.path.join(config.MODEL_SAVE_DIR, "forecast_xgboost_zoom.png"),
        zoom=True,
    )

    # Детекция пиков
    summary = _run_peak_detection(train, y_test, preds)
    print(f"\nИтог детекции пиков: {summary}")

    # Важность признаков
    imp_df = feature_importance(model, feature_names=list(X_train.columns))
    print("\nВажность признаков (XGBoost):")
    print(imp_df.to_string(index=False))

    # --- Сравнительный график всех моделей ---
    plot_all_forecasts(
        train, val, test,
        predictions_dict=comparator.predictions_,
        save_path=os.path.join(config.MODEL_SAVE_DIR, "comparison_all_models_full.png"),
        zoom=False,
    )
    plot_all_forecasts(
        train, val, test,
        predictions_dict=comparator.predictions_,
        save_path=os.path.join(config.MODEL_SAVE_DIR, "comparison_all_models_zoom.png"),
        zoom=True,
    )

    # --- CSV-экспорт для Typst / внешних инструментов ---
    # 1. Метрики сравнения моделей
    metrics_rows = [
        {"model": name, **m}
        for name, m in comparator.results_.items()
    ]
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(config.MODEL_SAVE_DIR, "metrics_comparison.csv"), index=False
    )
    print(f"\nСохранено: saved_models/metrics_comparison.csv")

    # 2. Предсказания всех моделей на тестовом периоде
    min_len = min(len(test), len(lower), len(upper),
                  *[len(p) for p in comparator.predictions_.values()])
    pred_df = test[["ds", "y"]].iloc[:min_len].copy().reset_index(drop=True)
    for name, model_preds in comparator.predictions_.items():
        pred_df[name] = np.array(model_preds[:min_len])
    pred_df["XGBoost_lower"] = lower[:min_len]
    pred_df["XGBoost_upper"] = upper[:min_len]
    pred_df.to_csv(
        os.path.join(config.MODEL_SAVE_DIR, "predictions.csv"), index=False
    )
    print(f"Сохранено: saved_models/predictions.csv")

    # 3. Важность признаков
    imp_df.to_csv(
        os.path.join(config.MODEL_SAVE_DIR, "feature_importance.csv"), index=False
    )
    print(f"Сохранено: saved_models/feature_importance.csv")


# ---------------------------------------------------------------------------
# Режим: обучение на готовом CSV (web_traffic.csv)
# ---------------------------------------------------------------------------

def mode_csv(args) -> None:
    """
    Полный пайплайн на данных Code/data/web_traffic.csv.
    Не требует Prometheus, Docker или интернета.

    Данные: 8760 точек, шаг 1ч, 2023-01-01 — 2023-12-31.
    Колонки: timestamp, rps, concurrent_users, cpu_usage, memory_usage, latency_ms.
    """
    from data_collection.csv_loader import load_web_traffic, describe_csv

    print("=" * 60)
    print("ОБУЧЕНИЕ НА ДАННЫХ: Code/data/web_traffic.csv")
    print("=" * 60)

    # --- Загрузка ---
    if args.path:
        from data_collection.csv_loader import load_csv
        df = load_csv(
            args.path,
            timestamp_col=args.timestamp_col or None,
            value_col=args.value_col or None,
        )
    else:
        df = load_web_traffic(months=args.months)

    n = len(df)
    print(f"\nЗагружено {n} точек | "
          f"RPS: min={df['y'].min():.0f}  max={df['y'].max():.0f}  "
          f"mean={df['y'].mean():.0f}")
    print(f"Период: {df['ds'].iloc[0]}  ->  {df['ds'].iloc[-1]}\n")

    if n < 200:
        print("Недостаточно данных. Укажите --months больше или проверьте файл.")
        sys.exit(1)

    # --- Разбиение: для 1h данных используем часы напрямую ---
    # 48ч теста + 48ч валидации, остальное — обучение
    TEST_H  = min(480, n // 6)   # ~20% но не более 480ч (20 дней)
    VAL_H   = TEST_H

    train, val, test = split_train_val_test(df, test_hours=TEST_H, val_hours=VAL_H)
    print(f"Split: train={len(train)}ч  val={len(val)}ч  test={len(test)}ч\n")

    # --- Сравнение моделей ---
    comparator = ModelComparison(model_save_dir=config.MODEL_SAVE_DIR)
    comparator.run(train, val, test,
                   include_prophet=not args.fast,
                   include_lstm=not args.fast)
    comparator.save_best()

    # --- XGBoost: доверительный интервал + важность признаков ---
    builder = FeatureBuilder()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        builder.transform_splits(train, val, test)

    xgb_path = os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl")
    model = joblib.load(xgb_path)
    preds = predict_xgboost(model, X_test)

    print("\nОбучение квантильных моделей (доверительный интервал)...")
    lower, upper = get_confidence_interval(
        X_train, y_train, X_val, y_val, X_test,
        save_dir=config.MODEL_SAVE_DIR,
    )

    # --- Важность признаков ---
    imp_df = feature_importance(model, feature_names=list(X_train.columns))
    print("\nВажность признаков (XGBoost gain):")
    print(imp_df.to_string(index=False))

    # --- Графики ---
    xgb_plot_path = os.path.join(config.MODEL_SAVE_DIR, "forecast_csv.png") if args.save_plots else None
    plot_forecast(
        train, val, test, preds, "XGBoost (web_traffic.csv)",
        lower=lower, upper=upper,
        save_path=xgb_plot_path,
    )
    plot_all_forecasts(
        train, val, test,
        predictions_dict=comparator.predictions_,
        save_path=os.path.join(config.MODEL_SAVE_DIR, "comparison_all_models.png"),
        zoom=False,
    )
    plot_all_forecasts(
        train, val, test,
        predictions_dict=comparator.predictions_,
        save_path=os.path.join(config.MODEL_SAVE_DIR, "comparison_all_models_zoom.png"),
        zoom=True,
    )

    # --- Детекция пиков ---
    # При 1h данных TARGET_LOAD_PER_REPLICA нужно перевести в реальный масштаб
    # web_traffic.csv: RPS 400-1300, конфиг: TARGET=10 (для Litestar). Пересчитываем:
    rps_max = float(train["y"].max())
    target_per_replica = rps_max / config.MAX_REPLICAS
    detector = PeakDetector(
        method=config.PEAK_METHOD,
        k=config.PEAK_K,
        target_rps_per_replica=target_per_replica,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
    )
    detector.fit(train["y"])
    predicted_series = pd.Series(preds, index=y_test.index)
    events_df = detector.detect_series(y_test, predicted_series)
    summary = detector.summary(events_df)

    print(f"\n--- Детекция пиков ---")
    print(f"  Метод: {config.PEAK_METHOD}  "
          f"порог={summary['threshold']:.0f} RPS  "
          f"(= среднее + {config.PEAK_K}σ за 24ч)")
    print(f"  Пиков: {summary['peaks_detected']} / {summary['total_points']} "
          f"({summary['peak_ratio_pct']}%)")
    print(f"  Уровни severity: {summary['severity_counts']}")

    if events_df["is_peak"].any():
        top5 = events_df[events_df["is_peak"]].head(5)
        print(f"\n  Первые 5 пиков на тестовом периоде:")
        for _, row in top5.iterrows():
            print(f"    {row['timestamp']}  RPS факт={row['rps']:.0f}  "
                  f"прогноз={row['predicted']:.0f}  "
                  f"[{row['severity']}]  реплик={row['recommended_replicas']}")

    print(f"\nМодели сохранены в {config.MODEL_SAVE_DIR}/")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _run_peak_detection(train, y_test, preds) -> dict:
    """Инициализирует детектор и выводит сводку по пикам."""
    detector = PeakDetector(
        method=config.PEAK_METHOD,
        k=config.PEAK_K,
        target_rps_per_replica=config.TARGET_LOAD_PER_REPLICA,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
    )
    detector.fit(train["y"])

    predicted_series = pd.Series(preds, index=y_test.index)
    events_df = detector.detect_series(y_test, predicted_series)

    summary = detector.summary(events_df)
    n_peaks = summary["peaks_detected"]
    total = summary["total_points"]
    threshold = summary["threshold"]
    print(f"\n--- Детекция пиков ---")
    print(f"  Метод: {config.PEAK_METHOD}, порог={threshold:.0f} RPS")
    print(f"  Пиков: {n_peaks} из {total} точек ({summary['peak_ratio_pct']}%)")
    print(f"  Уровни: {summary['severity_counts']}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Система прогнозирования пиковых нагрузок (ВКР)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # train
    p_train = sub.add_parser("train", help="Обучить XGBoost и оценить")
    p_train.add_argument("--synthetic", action="store_true",
                         help="Использовать синтетические данные вместо Prometheus")

    # compare
    p_cmp = sub.add_parser("compare", help="Сравнить все модели")
    p_cmp.add_argument("--synthetic", action="store_true")
    p_cmp.add_argument("--fast", action="store_true",
                        help="Пропустить NeuralProphet и LSTM (ускоряет сравнение)")

    # simulate
    p_sim = sub.add_parser("simulate", help="Симуляция проактивного масштабирования")
    p_sim.add_argument("--synthetic", action="store_true")

    # demo
    p_demo = sub.add_parser("demo", help="Полный пайплайн на синтетических данных")
    p_demo.add_argument("--fast", action="store_true",
                         help="Пропустить NeuralProphet и LSTM")

    # csv — обучение на реальном CSV (web_traffic.csv)
    p_csv = sub.add_parser(
        "csv",
        help="Обучение на готовом CSV (Code/data/web_traffic.csv) — без Prometheus",
    )
    p_csv.add_argument(
        "--path", default=None,
        help="Путь к CSV-файлу (по умолчанию: ../Code/data/web_traffic.csv)",
    )
    p_csv.add_argument(
        "--timestamp-col", default=None,
        help="Имя колонки с датой/временем (по умолчанию: автоопределение)",
    )
    p_csv.add_argument(
        "--value-col", default=None,
        help="Имя колонки со значениями (по умолчанию: автоопределение первой числовой)",
    )
    p_csv.add_argument(
        "--months", type=int, default=12,
        help="Сколько месяцев данных использовать (по умолчанию 12)",
    )
    p_csv.add_argument(
        "--fast", action="store_true",
        help="Пропустить NeuralProphet",
    )
    p_csv.add_argument(
        "--save-plots", action="store_true",
        help="Сохранить графики в saved_models/",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    dispatch = {
        "train":    mode_train,
        "compare":  mode_compare,
        "simulate": mode_simulate,
        "demo":     mode_demo,
        "csv":      mode_csv,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
