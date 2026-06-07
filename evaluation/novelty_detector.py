"""
LSTM Autoencoder для детекции новизны паттернов нагрузки (unsupervised).

Принцип: обучается восстанавливать нормальные последовательности метрик.
Аномалия = высокая ошибка восстановления (паттерн не похож на обученные).

Получает сырые данные (без удаления выбросов) — именно выбросы и есть сигнал.
"""

from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# Колонки сырых метрик для детектора (без предобработки)
RAW_COLS = ["y", "p99_ms", "error_rate_pct"]


def _make_sequences(values: np.ndarray, window: int) -> np.ndarray:
    """Нарезает массив (N, F) на окна (N-window+1, window, F)."""
    n = len(values)
    if n < window:
        return np.empty((0, window, values.shape[1]))
    return np.stack([values[i: i + window] for i in range(n - window + 1)])


class LSTMNoveltyDetector:
    """
    LSTM Autoencoder — детектор новизны паттернов нагрузки.

    Параметры
    ----------
    window      : длина временного окна (число периодов)
    epochs      : число эпох обучения
    threshold_q : квантиль ошибок обучающей выборки для порога
    """

    def __init__(
        self,
        window: int = 24,
        epochs: int = 30,
        threshold_q: float = 0.95,
    ):
        self.window = window
        self.epochs = epochs
        self.threshold_q = threshold_q
        self.threshold_: float | None = None
        self.model_ = None
        self._scaler = None

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def fit(self, df_raw: pd.DataFrame) -> "LSTMNoveltyDetector":
        """
        Обучает автоэнкодер на нормальных паттернах.

        Parameters
        ----------
        df_raw : DataFrame с колонками из RAW_COLS и временной меткой ds.
                 Выбросы НЕ удалять — они нужны для правильного порога.
        """
        from sklearn.preprocessing import StandardScaler
        import tensorflow as tf

        cols = [c for c in RAW_COLS if c in df_raw.columns]
        if not cols:
            raise ValueError(f"DataFrame должен содержать хотя бы одну из: {RAW_COLS}")

        values = df_raw[cols].ffill().fillna(0).values.astype(float)

        # Нормализация — fit только на train, сохраняем для инференса
        self._scaler = StandardScaler()
        values_scaled = self._scaler.fit_transform(values)

        sequences = _make_sequences(values_scaled, self.window)
        if len(sequences) == 0:
            raise ValueError(f"Недостаточно данных для окна window={self.window}")

        n_features = values_scaled.shape[1]
        self.model_ = self._build_model(self.window, n_features)

        self.model_.fit(
            sequences, sequences,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )

        # Порог: 95-й перцентиль ошибок на обучающих окнах
        errors = self._reconstruction_errors(sequences)
        self.threshold_ = float(np.percentile(errors, self.threshold_q * 100))
        logger.info(
            "LSTMNoveltyDetector обучен: окно=%d, признаков=%d, порог=%.4f",
            self.window, n_features, self.threshold_,
        )
        return self

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------

    def is_novel(self, df_raw: pd.DataFrame) -> bool:
        """
        Возвращает True если последние window точек образуют новый паттерн.

        Parameters
        ----------
        df_raw : свежие сырые данные, минимум window строк
        """
        if self.model_ is None or self.threshold_ is None:
            return False

        cols = [c for c in RAW_COLS if c in df_raw.columns]
        values = df_raw[cols].ffill().fillna(0).values.astype(float)

        if len(values) < self.window:
            return False

        values_scaled = self._scaler.transform(values[-self.window:])
        seq = values_scaled[np.newaxis, :, :]   # (1, window, features)
        error = self._reconstruction_errors(seq)[0]
        return bool(error > self.threshold_)

    def reconstruction_error(self, df_raw: pd.DataFrame) -> float:
        """Возвращает ошибку восстановления последнего окна (для мониторинга)."""
        if self.model_ is None:
            return 0.0
        cols = [c for c in RAW_COLS if c in df_raw.columns]
        values = df_raw[cols].ffill().fillna(0).values.astype(float)
        if len(values) < self.window:
            return 0.0
        values_scaled = self._scaler.transform(values[-self.window:])
        seq = values_scaled[np.newaxis, :, :]
        return float(self._reconstruction_errors(seq)[0])

    def novelty_score(
        self,
        df_raw: pd.DataFrame,
        alpha: float = 0.5,
        k_max: float = 1.5,
    ) -> float:
        """
        Непрерывный коэффициент новизны k_safety ∈ [1.0, k_max].

        Формула: k = 1.0 + clip((error/threshold - 1) * alpha, 0, k_max - 1)

        При error <= threshold  → k = 1.0  (нет новизны, реплики не добавляем)
        При error = 2*threshold → k = 1.0 + alpha
        При error >> threshold  → k → k_max (потолок)

        Parameters
        ----------
        alpha  : чувствительность (насколько быстро растёт k с ростом ошибки)
        k_max  : максимальный коэффициент (защита от переаллокации)
        """
        if self.model_ is None or self.threshold_ is None:
            return 1.0
        error = self.reconstruction_error(df_raw)
        if self.threshold_ <= 0:
            return 1.0
        ratio = error / self.threshold_
        k = 1.0 + float(np.clip((ratio - 1.0) * alpha, 0.0, k_max - 1.0))
        return k

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.model_.save(os.path.join(directory, "novelty_ae.keras"))
        joblib.dump(
            {"threshold": self.threshold_, "scaler": self._scaler,
             "window": self.window},
            os.path.join(directory, "novelty_meta.pkl"),
        )
        logger.info("LSTMNoveltyDetector сохранён в %s", directory)

    @classmethod
    def load(cls, directory: str) -> "LSTMNoveltyDetector":
        import tensorflow as tf
        meta = joblib.load(os.path.join(directory, "novelty_meta.pkl"))
        obj = cls(window=meta["window"])
        obj.threshold_ = meta["threshold"]
        obj._scaler = meta["scaler"]
        obj.model_ = tf.keras.models.load_model(
            os.path.join(directory, "novelty_ae.keras")
        )
        return obj

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(window: int, n_features: int):
        import tensorflow as tf
        inp = tf.keras.Input(shape=(window, n_features))
        # Encoder
        x = tf.keras.layers.LSTM(32, activation="relu")(inp)
        # Bottleneck → Decoder
        x = tf.keras.layers.RepeatVector(window)(x)
        x = tf.keras.layers.LSTM(32, activation="relu", return_sequences=True)(x)
        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer="adam", loss="mae")
        return model

    def _reconstruction_errors(self, sequences: np.ndarray) -> np.ndarray:
        pred = self.model_.predict(sequences, verbose=0)
        return np.mean(np.abs(sequences - pred), axis=(1, 2))
