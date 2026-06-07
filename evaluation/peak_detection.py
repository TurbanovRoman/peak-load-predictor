"""
Детекция пиковых нагрузок (глава 2.6.3, 2.7.1 ВКР).

Полностью переработано после аудита (см. AUDIT_REPORT.md, шаг 7).

Что было исправлено:

  1. Severity «exceeded» добавлен как отдельный уровень: фактическое
     превышение порога — это качественно иное событие, чем «приближение
     к порогу» (critical). Раньше эти два сценария сливались в одно
     значение, что ломало интерпретацию метрик.

  2. Метод `rolling_std` переименован в `last_window_std`, потому что
     старая реализация считала статистику ОДИН раз на всём истории и
     возвращала скаляр — это не «rolling», а «фиксированный порог из
     последнего окна». Имя теперь честно отражает поведение.

  3. Добавлен метод `adaptive_percentile` — пересчитывает порог через
     заданное число шагов (re-fit окном). Это настоящий адаптивный порог.

  4. Добавлен метод `capacity_aware` — порог = (max_replicas - safety) *
     target_rps_per_replica. Привязан к реальной ёмкости кластера, а не
     к статистике трафика. Идеален для production: даёт «жёсткий» SLO.

  5. Default target_rps_per_replica = 10.0 согласован с config.py
     (Litestar bench: ~10 RPS / реплика). Раньше стоял 1000.0 — ошибка
     порядка величины, ломала рекомендации по числу реплик.

  6. Учтена монотонность severity-уровней: если предсказание превышает
     threshold, severity ВСЕГДА = "exceeded", независимо от current_rps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------

# Порядок важен: чем правее — тем серьёзнее. Используется для сортировки.
Severity = Literal["ok", "info", "warning", "critical", "exceeded"]

_SEVERITY_RANK = {
    "ok": 0, "info": 1, "warning": 2, "critical": 3, "exceeded": 4,
}

Method = Literal[
    "last_window_std",      # бывший «rolling_std» — фиксированный порог из конца истории
    "percentile",           # фиксированный квантиль на всей истории
    "adaptive_percentile",  # квантиль, пересчитываемый в скользящем окне
    "capacity_aware",       # порог = capacity * (max_replicas - safety_margin)
]


@dataclass
class PeakEvent:
    timestamp: pd.Timestamp
    current_rps: float
    predicted_rps: float
    threshold: float
    severity: Severity
    recommended_replicas: int

    def __str__(self):
        icon = {
            "ok": "[OK]", "info": "[i]",
            "warning": "[!]", "critical": "[!!]", "exceeded": "[X]",
        }[self.severity]
        return (
            f"{icon} [{self.severity.upper()}] {self.timestamp} | "
            f"RPS текущий={self.current_rps:.0f}, прогноз={self.predicted_rps:.0f}, "
            f"порог={self.threshold:.0f} | реплик={self.recommended_replicas}"
        )

    @property
    def is_peak(self) -> bool:
        return self.severity in ("warning", "critical", "exceeded")


# ---------------------------------------------------------------------------
# Isolation Forest — детектор аномальных пиков на остатках ряда
# ---------------------------------------------------------------------------

class IsolationForestAnomalyDetector:
    """
    Обучается на остатках r_t после RobustSTL-декомпозиции.
    Аномальный пик (стохастический) → recommended_replicas умножается на k_safety.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._model = None

    def fit(self, residuals) -> "IsolationForestAnomalyDetector":
        from sklearn.ensemble import IsolationForest
        r = np.asarray(residuals, dtype=float).reshape(-1, 1)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
        )
        self._model.fit(r)
        return self

    def is_anomaly(self, value: float) -> bool:
        if self._model is None:
            return False
        return bool(self._model.predict([[float(value)]])[0] == -1)


# ---------------------------------------------------------------------------
# Детектор
# ---------------------------------------------------------------------------

class PeakDetector:
    """
    Определяет, является ли прогнозируемое значение RPS пиковым,
    и рекомендует количество реплик.

    Parameters
    ----------
    method            : метод определения порога (см. type Method)
    k                 : коэффициент σ для last_window_std
    percentile        : квантиль (0..100) для percentile / adaptive_percentile
    window            : окно (в периодах) для last_window_std / adaptive_percentile
    target_rps_per_replica : нагрузка RPS на одну реплику (из config.py)
    min_replicas      : минимальное число реплик
    max_replicas      : максимальное число реплик
    safety_margin     : доля «запаса» под выбросы для capacity_aware (0..1)
    warning_ratio     : доля от порога, при которой возникает warning
    critical_ratio    : доля от порога для critical
    spike_growth      : предиктивный рост current→predicted, при котором
                        выдаётся «info» даже если ratio < warning_ratio
    k_safety          : коэффициент запаса реплик при аномальном пике (IF)
    """

    def __init__(
        self,
        method: Method = "last_window_std",
        k: float = 2.0,
        percentile: float = 95.0,
        window: int = 24,
        target_rps_per_replica: float = 10.0,
        min_replicas: int = 1,
        max_replicas: int = 10,
        safety_margin: float = 0.1,
        warning_ratio: float = 0.70,
        critical_ratio: float = 0.85,
        spike_growth: float = 1.2,
        k_safety: float = 1.5,
    ):
        # Обратная совместимость: старое имя «rolling_std» маппим на новое
        if method == "rolling_std":
            method = "last_window_std"

        self.method: Method = method
        self.k = k
        self.percentile = percentile
        self.window = window
        self.k_safety = float(k_safety)
        self._if_detector: Optional[IsolationForestAnomalyDetector] = None
        self._drift_detector = None   # ADWINDriftDetector для дрейфа базовой линии
        self._baseline_mean: Optional[float] = None
        self.drift_recompute_timestamps: list = []
        self.target_rps = float(target_rps_per_replica)
        self.min_replicas = int(min_replicas)
        self.max_replicas = int(max_replicas)
        self.safety_margin = float(safety_margin)
        self.warning_ratio = float(warning_ratio)
        self.critical_ratio = float(critical_ratio)
        self.spike_growth = float(spike_growth)

        self._threshold: Optional[float] = None
        self._fit_history: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, historical_rps: pd.Series) -> "PeakDetector":
        """Вычисляет порог по историческим данным. Вызывать перед detect()."""
        s = pd.Series(historical_rps).dropna().astype(float)
        if s.empty:
            raise ValueError("historical_rps пустой — невозможно посчитать порог")

        self._fit_history = s

        if self.method == "percentile":
            self._threshold = float(np.percentile(s.values, self.percentile))

        elif self.method == "last_window_std":
            tail = s.tail(self.window)
            mu = float(tail.mean())
            sigma = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0
            self._threshold = mu + self.k * sigma

        elif self.method == "adaptive_percentile":
            # Базовый порог берём из последнего окна; реальный пересчёт
            # выполняется в detect_series (см. ниже)
            tail = s.tail(self.window)
            self._threshold = float(np.percentile(tail.values, self.percentile))

        elif self.method == "capacity_aware":
            # Порог = ёмкость кластера за вычетом safety-margin
            capacity = self.max_replicas * self.target_rps
            self._threshold = capacity * (1.0 - self.safety_margin)

        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

        return self

    def fit_anomaly_detector(self, residuals) -> "PeakDetector":
        """Обучает Isolation Forest на остатках r_t из RobustSTL."""
        self._if_detector = IsolationForestAnomalyDetector()
        self._if_detector.fit(residuals)
        return self

    def fit_drift_detector(
        self,
        historical_rps: pd.Series,
        delta: float = 0.002,
        min_obs: int = 50,
        cooldown_n: int = 48,
        confirmation_n: int = 5,
    ) -> "PeakDetector":
        """
        Инициализирует ADWIN-детектор дрейфа базовой линии трафика.

        Когда среднее RPS устойчиво смещается (концепт-дрейф), детектор
        сигнализирует и порог детекции пиков пересчитывается автоматически.
        cooldown_n=48 и confirmation_n=5 защищают от ложных срабатываний
        на кратковременных пиках.
        """
        from retraining.drift_detector import ADWINDriftDetector
        arr = np.asarray(historical_rps.dropna().values, dtype=float)
        self._baseline_mean = float(np.mean(arr))
        self._drift_detector = ADWINDriftDetector(
            delta=delta,
            min_obs=min_obs,
            cooldown_n=cooldown_n,
            confirmation_n=confirmation_n,
            n_fresh=0,
        )
        self._drift_detector.set_baseline(np.abs(arr - self._baseline_mean))
        self.drift_recompute_timestamps = []
        return self

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Вызовите .fit() перед обращением к threshold")
        return self._threshold

    # ------------------------------------------------------------------
    # Single-point detection
    # ------------------------------------------------------------------

    def detect(
        self,
        predicted_rps: float,
        current_rps: float = 0.0,
        timestamp: Optional[pd.Timestamp] = None,
        threshold_override: Optional[float] = None,
    ) -> PeakEvent:
        """
        Классифицирует прогнозируемую нагрузку.

        Уровни (монотонные!):
          exceeded — predicted ≥ threshold (фактический пик)
          critical — predicted ≥ threshold * critical_ratio
          warning  — predicted ≥ threshold * warning_ratio
          info     — predicted ≥ current * spike_growth (резкий рост)
          ok       — иначе
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        threshold = (
            threshold_override
            if threshold_override is not None
            else self.threshold
        )
        if threshold <= 0:
            severity: Severity = "ok"
        else:
            ratio = predicted_rps / threshold
            if ratio >= 1.0:
                severity = "exceeded"
            elif ratio >= self.critical_ratio:
                severity = "critical"
            elif ratio >= self.warning_ratio:
                severity = "warning"
            elif current_rps > 0 and predicted_rps > current_rps * self.spike_growth:
                severity = "info"
            else:
                severity = "ok"

        replicas = self._calculate_replicas(predicted_rps)

        return PeakEvent(
            timestamp=timestamp,
            current_rps=float(current_rps),
            predicted_rps=float(predicted_rps),
            threshold=float(threshold),
            severity=severity,
            recommended_replicas=replicas,
        )

    # ------------------------------------------------------------------
    # Series detection
    # ------------------------------------------------------------------

    def detect_series(
        self,
        rps_series: pd.Series,
        predicted_series: pd.Series,
        recompute_every: Optional[int] = None,
        upper_ci_series: Optional[pd.Series] = None,
        campaign_p95: Optional[float] = None,
        campaign_mask: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Применяет детекцию ко всему ряду.

        Parameters
        ----------
        recompute_every : если задан и method='adaptive_percentile' — пересчитывает
                          порог каждые N точек по последнему окну self.window.
        upper_ci_series : верхняя граница ДИ (Q90). Если передана — реплики
                          считаются по ней, а не по точечному прогнозу.
        campaign_p95    : P95 RPS из исторических кампаний. Если передан вместе
                          с campaign_mask — в периоды кампании реплики считаются
                          по max(predicted, upper_ci, campaign_p95).
        campaign_mask   : булева серия той же длины что rps_series; True = кампания.

        Returns
        -------
        DataFrame: timestamp, rps, predicted, threshold,
                   severity, recommended_replicas, is_peak, is_anomaly
        """
        records = []
        rolling_threshold = self.threshold
        history_buffer: list = list(self._fit_history.values) \
            if self._fit_history is not None else []

        # Верхняя граница CI выровнена по длине predicted_series
        upper_ci_arr: Optional[np.ndarray] = None
        if upper_ci_series is not None:
            upper_ci_arr = np.asarray(upper_ci_series.values, dtype=float)

        campaign_arr: Optional[np.ndarray] = None
        if campaign_mask is not None:
            campaign_arr = np.asarray(campaign_mask.values, dtype=bool)

        # Скользящее среднее для вычисления остатка (приближение r_t)
        rps_arr = np.asarray(rps_series.values, dtype=float)
        roll_size = min(self.window, len(rps_arr))
        rolling_med = pd.Series(rps_arr).rolling(roll_size, min_periods=1).median().values

        for i, (ts, cur, pred) in enumerate(zip(
            rps_series.index, rps_series.values, predicted_series.values,
        )):
            # Адаптивный пересчёт порога по расписанию
            if (
                self.method == "adaptive_percentile"
                and recompute_every
                and i > 0
                and i % recompute_every == 0
                and len(history_buffer) >= self.window
            ):
                tail = history_buffer[-self.window:]
                rolling_threshold = float(np.percentile(tail, self.percentile))

            # Пересчёт порога при обнаружении дрейфа базовой линии (ADWIN)
            if self._drift_detector is not None and self._baseline_mean is not None:
                self._drift_detector.observe(
                    y_true=float(cur), y_pred=self._baseline_mean
                )
                signal = self._drift_detector.check()
                if signal.triggered and len(history_buffer) >= self.window:
                    tail = history_buffer[-self.window:]
                    rolling_threshold = float(np.percentile(tail, self.percentile))
                    self._baseline_mean = float(np.mean(tail))
                    self._drift_detector.reset_after_retrain(
                        np.abs(np.array(tail) - self._baseline_mean)
                    )
                    self.drift_recompute_timestamps.append(pd.Timestamp(ts))

            # Остаток ≈ отклонение от скользящей медианы (приближение r_t)
            residual = float(cur) - float(rolling_med[i])
            in_event = bool(campaign_arr[i]) if campaign_arr is not None and i < len(campaign_arr) else False
            is_anom = (
                self._if_detector.is_anomaly(residual)
                if self._if_detector is not None and not in_event else False
            )

            event = self.detect(
                predicted_rps=float(pred),
                current_rps=float(cur),
                timestamp=pd.Timestamp(ts),
                threshold_override=rolling_threshold,
            )

            # Реплики: берём max(predicted, upper_ci, campaign_p95) при кампании
            rps_for_replicas = float(pred)
            if upper_ci_arr is not None and i < len(upper_ci_arr):
                rps_for_replicas = max(rps_for_replicas, float(upper_ci_arr[i]))
            if (campaign_p95 is not None
                    and campaign_arr is not None
                    and i < len(campaign_arr)
                    and campaign_arr[i]):
                rps_for_replicas = max(rps_for_replicas, campaign_p95)
            replicas = self._calculate_replicas(rps_for_replicas)

            # При аномальном пике — дополнительный запас k_safety
            if is_anom and event.is_peak:
                replicas = int(np.clip(
                    np.ceil(rps_for_replicas * self.k_safety / max(self.target_rps, 1)),
                    self.min_replicas, self.max_replicas,
                ))

            records.append({
                "timestamp": event.timestamp,
                "rps": event.current_rps,
                "predicted": event.predicted_rps,
                "threshold": event.threshold,
                "severity": event.severity,
                "recommended_replicas": replicas,
                "is_peak": event.is_peak,
                "is_anomaly": is_anom,
            })

            # Обновляем историю для адаптивного порога
            history_buffer.append(float(cur))

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Replicas
    # ------------------------------------------------------------------

    def _calculate_replicas(self, predicted_rps: float) -> int:
        """Минимальное число реплик, удовлетворяющее load: ceil(load/target)."""
        if self.target_rps <= 0:
            return self.min_replicas
        desired = int(np.ceil(max(0.0, predicted_rps) / self.target_rps))
        return int(np.clip(desired, self.min_replicas, self.max_replicas))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, events_df: pd.DataFrame) -> dict:
        """Сводная статистика по серии детекций."""
        total = len(events_df)
        peaks = int(events_df["is_peak"].sum()) if total else 0

        # Упорядоченный по серьёзности словарь
        sev_counts_raw = (
            events_df["severity"].value_counts().to_dict() if total else {}
        )
        sev_counts = {
            sev: int(sev_counts_raw.get(sev, 0))
            for sev in ("ok", "info", "warning", "critical", "exceeded")
        }

        return {
            "total_points": int(total),
            "peaks_detected": peaks,
            "peak_ratio_pct": round(peaks / total * 100, 2) if total else 0.0,
            "threshold": round(self.threshold, 2) if self._threshold else None,
            "severity_counts": sev_counts,
            "method": self.method,
        }
