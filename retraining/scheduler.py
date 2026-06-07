"""
RetrainScheduler — оркестратор адаптивного переобучения.

Жизненный цикл:
  1. Загружается стартовая модель (например, лучшая из ModelComparison).
  2. На каждую новую точку:
       a) предсказываем (как обычно)
       b) когда становится известно фактическое y — observe()
       c) check_drift() решает, нужен ли retrain
       d) если да — fit на скользящем окне свежих данных, replace
  3. Все события retrain пишутся в audit-CSV для отчёта ВКР.

Так система действительно «адаптируется», а не использует замороженную
модель из последней ручной тренировки.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from preprocessing.feature_engineering import FeatureBuilder, split_train_val_test
from retraining.drift_detector import ADWINDriftDetector, DriftSignal


logger = logging.getLogger(__name__)


@dataclass
class RetrainEvent:
    """Запись о переобучении (для лога и отчёта ВКР)."""
    timestamp: str
    reason: str
    n_observations: int
    baseline_mae_before: float
    rolling_mae_before: float
    new_baseline_mae: float
    train_size: int
    duration_s: float

    def to_row(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Тип функции переобучения: принимает train+val DataFrame, возвращает модель
# и массив ошибок на свежем тесте (для нового baseline)
# ---------------------------------------------------------------------------

# train_fn(history_df) -> (new_model, predict_fn, baseline_errors)
TrainFn = Callable[[pd.DataFrame], tuple]


class RetrainScheduler:
    """
    Управляет онлайн-моделью с автопереобучением.

    Parameters
    ----------
    initial_model    : начальная обученная модель
    predict_fn       : функция (model, X) → np.ndarray прогнозов
    train_fn         : функция переобучения (см. TrainFn)
    builder          : FeatureBuilder для построения признаков
    drift_detector   : экземпляр ADWINDriftDetector
    history_buffer_size : сколько последних (ds, y) точек хранить для retrain
    min_history_to_retrain : минимум точек для безопасного переобучения
    audit_path       : CSV-файл для аудита retrain-событий
    cooldown_seconds : минимум секунд между retrain (защита от шторма)
    """

    def __init__(
        self,
        initial_model: object,
        predict_fn: Callable,
        train_fn: TrainFn,
        builder: FeatureBuilder,
        drift_detector: ADWINDriftDetector,
        history_buffer_size: int = 10000,
        min_history_to_retrain: int = 500,
        audit_path: Optional[str] = None,
        cooldown_seconds: float = 300.0,
        rollback_window: int = 48,
        rollback_threshold: float = 1.3,
    ):
        self.model = initial_model
        self.predict_fn = predict_fn
        self.train_fn = train_fn
        self.builder = builder
        self.drift = drift_detector

        self.min_history_to_retrain = int(min_history_to_retrain)
        self.audit_path = audit_path
        self.cooldown_seconds = float(cooldown_seconds)
        self.rollback_window = int(rollback_window)
        self.rollback_threshold = float(rollback_threshold)

        # Скользящий буфер — deque даёт O(1) append/trim без pd.concat
        self._history_buffer_size = int(history_buffer_size)
        self._history_deque: Deque[Dict] = deque(maxlen=self._history_buffer_size)
        self._last_retrain_ts: float = 0.0
        self.events: List[RetrainEvent] = []

        # Резервная копия для rollback
        self._backup_model: Optional[object] = None
        self._backup_predict_fn: Optional[Callable] = None
        self._backup_baseline_mae: Optional[float] = None
        self._monitoring_cycles: int = 0
        self._monitoring_active: bool = False

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    @property
    def _history(self) -> pd.DataFrame:
        """DataFrame из внутреннего deque (только для совместимости с retrain)."""
        return pd.DataFrame(list(self._history_deque))

    def seed_history(self, df: pd.DataFrame) -> None:
        """Заливает стартовую историю (например, train+val датасет)."""
        df = df[["ds", "y"]].copy().sort_values("ds").tail(self._history_buffer_size)
        self._history_deque.clear()
        self._history_deque.extend(df.to_dict("records"))

    def predict_one(self, current_df: pd.DataFrame) -> float:
        """
        Прогнозирует одну точку вперёд по DataFrame с историей+текущей.
        Передаёт полный X в predict_fn — каждая функция сама извлекает нужные
        строки (последнюю для XGBoost, окно для LSTM, шаг для Prophet).
        """
        X = self.builder.get_X(current_df)
        if len(X) == 0:
            return float("nan")
        pred = self.predict_fn(self.model, X)
        return float(np.asarray(pred).flatten()[0])

    def observe(self, ts: pd.Timestamp, y_true: float, y_pred: float) -> None:
        """Добавляет наблюдение в буфер O(1) и обновляет drift detector."""
        self._history_deque.append({"ds": pd.Timestamp(ts), "y": float(y_true)})
        self.drift.observe(y_true=y_true, y_pred=y_pred)

    def check_and_retrain(self) -> Optional[RetrainEvent]:
        """
        Проверяет дрифт и при необходимости запускает retrain.
        Возвращает RetrainEvent если retrain выполнен, иначе None.
        """
        signal = self.drift.check()
        if not signal.triggered:
            return None

        # Cooldown: не чаще раза в N секунд
        now = time.time()
        if now - self._last_retrain_ts < self.cooldown_seconds:
            logger.info(
                "Drift detected (%s), но retrain отложен (cooldown). "
                "current_mae=%.3f baseline=%.3f",
                signal.reason, signal.current_mae, signal.baseline_mae,
            )
            return None

        if len(self._history) < self.min_history_to_retrain:
            logger.warning(
                "Drift detected (%s), но в буфере мало точек (%d < %d) — "
                "пропуск retrain.",
                signal.reason, len(self._history), self.min_history_to_retrain,
            )
            return None

        return self._do_retrain(signal)

    # ------------------------------------------------------------------
    # Внутренний retrain
    # ------------------------------------------------------------------

    def _do_retrain(self, signal: DriftSignal) -> RetrainEvent:
        t0 = time.time()
        logger.info(
            "Retrain запущен. reason=%s current_mae=%.3f baseline=%.3f "
            "history=%d точек",
            signal.reason, signal.current_mae, signal.baseline_mae,
            len(self._history),
        )

        new_model, new_predict_fn, baseline_errors = self.train_fn(
            self._history.copy()
        )

        new_baseline_mae = float(np.mean(np.abs(baseline_errors))) \
            if len(baseline_errors) else float("nan")

        # Сохранить резервную копию перед заменой
        self._backup_model = self.model
        self._backup_predict_fn = self.predict_fn
        self._backup_baseline_mae = self.drift.baseline_mae
        self._monitoring_cycles = 0
        self._monitoring_active = True

        # Атомарная замена модели и сброс детектора
        self.model = new_model
        self.predict_fn = new_predict_fn
        self.drift.reset_after_retrain(baseline_errors)
        self._last_retrain_ts = time.time()

        event = RetrainEvent(
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            reason=signal.reason,
            n_observations=signal.n_observations,
            baseline_mae_before=signal.baseline_mae,
            rolling_mae_before=signal.current_mae,
            new_baseline_mae=new_baseline_mae,
            train_size=len(self._history),
            duration_s=round(time.time() - t0, 2),
        )
        self.events.append(event)
        self._append_audit(event)
        logger.info(
            "Retrain завершён за %.1fs. new_baseline_mae=%.3f",
            event.duration_s, new_baseline_mae,
        )
        return event

    def check_rollback(self) -> bool:
        """
        Проверяет нужен ли откат к резервной модели.
        Вызывать на каждом цикле после observe().
        Возвращает True если откат выполнен.
        """
        if not self._monitoring_active or self._backup_model is None:
            return False

        self._monitoring_cycles += 1
        if self._monitoring_cycles < self.rollback_window:
            return False

        current_mae = self.drift.baseline_mae
        if (
            current_mae is not None
            and self._backup_baseline_mae is not None
            and current_mae > self._backup_baseline_mae * self.rollback_threshold
        ):
            logger.warning(
                "Rollback: новая модель хуже резервной "
                "(MAE %.3f > %.3f × %.1f). Откат.",
                current_mae, self._backup_baseline_mae, self.rollback_threshold,
            )
            self.model = self._backup_model
            self.predict_fn = self._backup_predict_fn
            self._monitoring_active = False
            return True

        self._monitoring_active = False
        self._backup_model = None
        self._backup_predict_fn = None
        return False

    def _append_audit(self, event: RetrainEvent) -> None:
        if not self.audit_path:
            return
        os.makedirs(os.path.dirname(self.audit_path) or ".", exist_ok=True)
        write_header = not os.path.exists(self.audit_path)
        with open(self.audit_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=event.to_row().keys())
            if write_header:
                writer.writeheader()
            writer.writerow(event.to_row())


# ---------------------------------------------------------------------------
# Готовая фабрика train_fn для XGBoost (используется по умолчанию)
# ---------------------------------------------------------------------------

def make_xgb_train_fn(
    builder: FeatureBuilder,
    val_fraction: float = 0.15,
    save_dir: Optional[str] = None,
    best_params: Optional[dict] = None,
) -> TrainFn:
    """
    Возвращает train_fn, переобучающую XGBoost на накопленной истории.

    История делится «временнО» на train/val (без перемешивания),
    последние val_fraction точек служат валидацией для early stopping
    и одновременно дают baseline_errors для нового детектора.

    best_params : параметры, найденные Random Search при начальном обучении.
                  Если None — Random Search запускается заново на каждом retrain.
    """
    from models.forecasters import train_xgboost, train_xgboost_random_search, predict_xgboost, predict_xgboost_wf

    def _train(history_df: pd.DataFrame):
        history_df = history_df.sort_values("ds").reset_index(drop=True)
        n = len(history_df)
        n_val = max(50, int(n * val_fraction))
        train_df = history_df.iloc[:-n_val].copy()
        val_df   = history_df.iloc[-n_val:].copy()

        (X_tr, y_tr), (X_vl, y_vl), (X_ts, y_ts) = builder.transform_splits(
            train_df, val_df, val_df,
        )

        save_path = (
            os.path.join(save_dir, "xgboost_retrained.pkl")
            if save_dir else None
        )

        if best_params is not None:
            model = train_xgboost(
                X_tr, y_tr, X_vl, y_vl,
                save_path=save_path,
                **best_params,
            )
        else:
            model, _, _ = train_xgboost_random_search(
                X_tr, y_tr, X_vl, y_vl, save_path=save_path,
            )

        preds = predict_xgboost(model, X_vl)
        baseline_errors = np.abs(y_vl.values - preds)
        return model, predict_xgboost_wf, baseline_errors

    return _train


# ---------------------------------------------------------------------------
# Фабрика train_fn, переобучающая все три модели и выбирающая лучшую по MAE
# ---------------------------------------------------------------------------

def make_multi_model_train_fn(
    builder: FeatureBuilder,
    val_fraction: float = 0.15,
    save_dir: Optional[str] = None,
    xgb_params: Optional[dict] = None,
    prophet_best_params: Optional[dict] = None,
    include_lstm: bool = True,
    include_prophet: bool = True,
) -> TrainFn:
    """
    Возвращает train_fn, переобучающую XGBoost, LSTM и (опционально) Prophet
    на накопленной истории и возвращающую лучшую модель по MAE на валидации.

    xgb_params         : гиперпараметры XGBoost из начального Random Search.
                         Если None — Random Search запускается заново.
    prophet_best_params: гиперпараметры Prophet из начального сравнения.
                         Если None — Prophet пропускается.
    include_lstm       : включить LSTM (отключить если нет TensorFlow).
    include_prophet    : включить Prophet (медленно, по умолчанию True).
    """
    import types as _types
    from models.forecasters import (
        train_xgboost, train_xgboost_random_search, predict_xgboost, predict_xgboost_wf,
        train_lstm, predict_lstm, predict_lstm_wf,
        refit_prophet_full, predict_prophet_wf,
    )

    def _train(history_df: pd.DataFrame):
        history_df = history_df.sort_values("ds").reset_index(drop=True)
        n = len(history_df)
        n_val = max(50, int(n * val_fraction))
        train_df = history_df.iloc[:-n_val].copy()
        val_df   = history_df.iloc[-n_val:].copy()

        (X_tr, y_tr), (X_vl, y_vl), _ = builder.transform_splits(
            train_df, val_df, val_df,
        )

        candidates = []  # (mae, model, predict_fn, baseline_errors)

        # --- XGBoost ---
        try:
            save_path = os.path.join(save_dir, "xgboost_retrained.pkl") if save_dir else None
            if xgb_params is not None:
                xgb_model = train_xgboost(X_tr, y_tr, X_vl, y_vl,
                                          save_path=save_path, **xgb_params)
            else:
                xgb_model, _, _ = train_xgboost_random_search(
                    X_tr, y_tr, X_vl, y_vl, save_path=save_path,
                )
            xgb_errors = np.abs(y_vl.values - predict_xgboost(xgb_model, X_vl))
            candidates.append((float(xgb_errors.mean()), xgb_model, predict_xgboost_wf, xgb_errors))
            logger.info("XGBoost retrain MAE=%.3f", xgb_errors.mean())
        except Exception as e:
            logger.warning("XGBoost retrain failed: %s", e)

        # --- LSTM ---
        if include_lstm:
            try:
                lstm_artifact = train_lstm(X_tr, y_tr, X_vl, y_vl)
                lstm_preds = predict_lstm(lstm_artifact, X_vl)
                valid = ~np.isnan(lstm_preds)
                if valid.any():
                    lstm_errors = np.abs(y_vl.values[valid] - lstm_preds[valid])
                    candidates.append((float(lstm_errors.mean()), lstm_artifact, predict_lstm_wf, lstm_errors))
                    logger.info("LSTM retrain MAE=%.3f", lstm_errors.mean())
            except Exception as e:
                logger.warning("LSTM retrain failed: %s", e)

        # --- Prophet ---
        if include_prophet and prophet_best_params is not None:
            try:
                base = _types.SimpleNamespace(_best_params=prophet_best_params)
                prophet_model = refit_prophet_full(
                    base_model=base,
                    train_val_df=history_df,
                )
                from models.forecasters import predict_prophet
                pred_df = predict_prophet(prophet_model, periods=len(val_df),
                                          history_ds=train_df["ds"])
                prophet_preds = np.clip(pred_df["yhat"].values[:len(y_vl)], 0, None)
                prophet_errors = np.abs(y_vl.values - prophet_preds)
                candidates.append((float(prophet_errors.mean()), prophet_model, predict_prophet_wf, prophet_errors))
                logger.info("Prophet retrain MAE=%.3f", prophet_errors.mean())
            except Exception as e:
                logger.warning("Prophet retrain failed: %s", e)

        if not candidates:
            raise RuntimeError("Все модели не смогли переобучиться")

        best_mae, best_model, best_predict_fn, best_errors = min(candidates, key=lambda x: x[0])
        logger.info("Победитель после retrain: MAE=%.3f", best_mae)
        return best_model, best_predict_fn, best_errors

    return _train
