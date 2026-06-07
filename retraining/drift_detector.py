"""
Детекция концепт-дрейфа на основе ADWIN (Bifet & Gavalda, 2007).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np


@dataclass
class DriftSignal:
    """Результат проверки дрейфа."""
    triggered: bool
    reason: str
    current_mae: float
    baseline_mae: float
    n_observations: int
    threshold: float



# ---------------------------------------------------------------------------
# ADWIN-based drift detector (Bifet & Gavalda, 2007)
# ---------------------------------------------------------------------------

class ADWINDriftDetector:
    """
    Детектор концепт-дрейфа на основе алгоритма ADWIN.

    ADWIN (Adaptive WINdowing) автоматически определяет размер окна:
    при обнаружении статистически значимого изменения среднего значения
    ошибки прогноза окно сжимается до момента изменения.

    Граница Хёффдинга гарантирует: P(ложное срабатывание) ≤ delta.

    Помимо ADWIN поддерживает принудительный периодический retrain каждые
    n_fresh шагов (соответствует N_fresh на блок-схеме алгоритма, Рисунок 2.7):
    даже без статистически значимого дрейфа модель освежается раз в N_fresh
    шагов, чтобы не пропустить медленный дрейф, который ADWIN не замечает.

    Параметры
    ----------
    delta          : уровень доверия (стандарт из статьи: 0.002 → ≤0.2% ложных)
    min_obs        : минимум наблюдений перед первой проверкой
    cooldown_n     : минимум шагов между двумя retrain-событиями
    n_fresh        : принудительный retrain каждые n_fresh шагов (0 = отключить)
    confirmation_n : сколько шагов подряд ADWIN должен сигналить дрейф прежде
                     чем ретрейн запустится. Отсекает разовые выбросы (пики):
                     при пике ошибки скачут и возвращаются → счётчик сбрасывается;
                     при концепт-дрейфе ошибки остаются высокими → ретрейн.
    """

    def __init__(
        self,
        delta: float | None = None,
        min_obs: int | None = None,
        cooldown_n: int | None = None,
        n_fresh: int | None = None,
        confirmation_n: int | None = None,
    ):
        try:
            import config as _cfg
            delta         = delta         if delta         is not None else getattr(_cfg, "ADWIN_DELTA",        0.002)
            min_obs       = min_obs       if min_obs       is not None else getattr(_cfg, "ADWIN_MIN_OBS",      30)
            cooldown_n    = cooldown_n    if cooldown_n    is not None else getattr(_cfg, "ADWIN_COOLDOWN",     20)
            n_fresh       = n_fresh       if n_fresh       is not None else getattr(_cfg, "RETRAIN_N_FRESH",    0)
            confirmation_n= confirmation_n if confirmation_n is not None else getattr(_cfg, "ADWIN_CONFIRMATION", 10)
        except (ImportError, Exception):
            delta          = delta          or 0.002
            min_obs        = min_obs        or 30
            cooldown_n     = cooldown_n     or 20
            n_fresh        = n_fresh        or 0
            confirmation_n = confirmation_n or 10
        from river.drift import ADWIN as _ADWIN
        self._adwin = _ADWIN(delta=delta)
        self.delta = delta
        self.min_obs = int(min_obs)
        self.cooldown_n = int(cooldown_n)
        self.n_fresh = int(n_fresh)
        self.confirmation_n = int(confirmation_n)

        self._n: int = 0
        self._n_since_retrain: int = 0
        self._consecutive_drift: int = 0  # счётчик подтверждения
        self._baseline_mae: Optional[float] = None
        self._last_errors: list = []   # для вычисления current_mae в DriftSignal

    # ------------------------------------------------------------------

    def set_baseline(self, baseline_errors: np.ndarray) -> None:
        arr = np.abs(np.asarray(baseline_errors, dtype=float))
        self._baseline_mae = float(np.mean(arr)) if arr.size else 0.0
        self._n_since_retrain = 0
        # Засеваем ADWIN обучающими ошибками: он должен знать,
        # что такое «нормальная» ошибка, чтобы детектировать изменение.
        for e in arr[-100:]:
            self._adwin.update(float(e))

    def reset_after_retrain(self, baseline_errors: np.ndarray) -> None:
        from river.drift import ADWIN as _ADWIN
        self._adwin = _ADWIN(delta=self.delta)   # свежий экземпляр
        self._last_errors.clear()
        self._n_since_retrain = 0
        self._consecutive_drift = 0
        self.set_baseline(baseline_errors)

    def observe(self, y_true: float, y_pred: float) -> None:
        err = abs(float(y_true) - float(y_pred))
        self._adwin.update(err)
        self._last_errors.append(err)
        if len(self._last_errors) > 200:
            self._last_errors.pop(0)
        self._n += 1
        self._n_since_retrain += 1

    def check(self) -> DriftSignal:
        if self._n < self.min_obs:
            return DriftSignal(
                triggered=False,
                reason="warming_up",
                current_mae=float("nan"),
                baseline_mae=self._baseline_mae or float("nan"),
                n_observations=self._n,
                threshold=float("nan"),
            )

        if self._n_since_retrain < self.cooldown_n:
            return DriftSignal(
                triggered=False,
                reason="cooldown",
                current_mae=float(np.mean(self._last_errors)) if self._last_errors else float("nan"),
                baseline_mae=self._baseline_mae or float("nan"),
                n_observations=self._n,
                threshold=float("nan"),
            )

        current_mae = float(np.mean(self._last_errors)) if self._last_errors else float("nan")

        # N_fresh: принудительный retrain каждые n_fresh шагов (схема 2.7, верхняя ветвь).
        # Проверяется до ADWIN — даже без статистического дрейфа модель обновляется.
        if self.n_fresh > 0 and self._n_since_retrain >= self.n_fresh:
            return DriftSignal(
                triggered=True,
                reason="n_fresh",
                current_mae=current_mae,
                baseline_mae=self._baseline_mae or float("nan"),
                n_observations=self._n,
                threshold=float(self.n_fresh),
            )

        if self._adwin.drift_detected:
            self._consecutive_drift += 1
        else:
            # Дрейф пропал — пик прошёл, сбрасываем счётчик подтверждения
            self._consecutive_drift = 0

        if self._consecutive_drift >= self.confirmation_n:
            self._consecutive_drift = 0
            return DriftSignal(
                triggered=True,
                reason="adwin",
                current_mae=current_mae,
                baseline_mae=self._baseline_mae or float("nan"),
                n_observations=self._n,
                threshold=float(self.confirmation_n),
            )

        return DriftSignal(
            triggered=False,
            reason="ok",
            current_mae=current_mae,
            baseline_mae=self._baseline_mae or float("nan"),
            n_observations=self._n,
            threshold=float("nan"),
        )

    @property
    def baseline_mae(self) -> Optional[float]:
        return self._baseline_mae

    @property
    def n_since_retrain(self) -> int:
        return self._n_since_retrain
