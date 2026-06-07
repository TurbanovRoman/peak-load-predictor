import math
import time
from threading import Timer
from config import TARGET_LOAD_PER_REPLICA, MIN_REPLICAS, MAX_REPLICAS, SCALE_DOWN_BACKOFF

class MomentumScaler:
    def __init__(self, target_load_per_replica=TARGET_LOAD_PER_REPLICA,
                 min_replicas=MIN_REPLICAS, max_replicas=MAX_REPLICAS,
                 scale_down_backoff=SCALE_DOWN_BACKOFF):
        self.target_load = target_load_per_replica
        self.min = min_replicas
        self.max = max_replicas
        self.scale_down_backoff = scale_down_backoff
        self.current_replicas = 1   # для симуляции
        self.scale_timer = None

    def calculate_desired_replicas(self, predicted_load):
        """По предсказанной нагрузке вычисляет необходимое число реплик"""
        if self.target_load <= 0:
            return self.min
        return max(self.min, min(self.max, math.ceil(predicted_load / self.target_load)))

    def scale(self, predicted_load):
        """Принимает решение о масштабировании (симуляция)"""
        desired = self.calculate_desired_replicas(predicted_load)
        if desired > self.current_replicas:
            if self.scale_timer:
                self.scale_timer.cancel()
            self.current_replicas = desired
            print(f"[UP]   Scale UP   -> {desired} replicas (load {predicted_load:.0f})")
        elif desired < self.current_replicas:
            if self.scale_timer:
                self.scale_timer.cancel()
            # отложенное уменьшение
            self.scale_timer = Timer(self.scale_down_backoff, self._scale_down, args=[desired])
            self.scale_timer.start()
            print(f"[WAIT] Scale DOWN -> {desired} replicas in {self.scale_down_backoff}s")
        else:
            print(f"[OK]   Replicas={self.current_replicas} optimal")

    def _scale_down(self, replicas):
        self.current_replicas = replicas
        print(f"[DOWN] Scale DOWN -> {replicas} replicas")