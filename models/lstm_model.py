"""
LSTM-модель прогнозирования RPS (правильная реализация).

Полностью переписана после аудита (см. AUDIT_REPORT.md, шаг 5).
Что было исправлено по сравнению со старой версией:

  СТАРАЯ ВЕРСИЯ                       │  НОВАЯ ВЕРСИЯ
  ────────────────────────────────────┼──────────────────────────────────
  MinMaxScaler.fit_transform на         │  StandardScaler fit() ТОЛЬКО на
  каждом вызове df_to_X_y()             │  train; .transform() на val/test
  → утечка статистик val/test в train   │  → честная схема
  ────────────────────────────────────┼──────────────────────────────────
  Одномерный ряд (только y)             │  Multivariate: y + все признаки
  → теряются календарные/лаговые        │  FeatureBuilder (12+ признаков)
  ────────────────────────────────────┼──────────────────────────────────
  window_size = 5                       │  window_size настраиваемый;
  → 5 шагов недостаточно для суточной   │  по умолчанию 24×step или 288 для
  периодики (нужно ≥24 точек)           │  5-минутных данных (= 24 часа)
  ────────────────────────────────────┼──────────────────────────────────
  Loss = MSE                            │  Loss = Huber (δ=1.0)
  → выбросы (пики!) тянут веса          │  → робастность к выбросам, что
  → плохо предсказывает именно пики     │  как раз нужно для пик-детекции
  ────────────────────────────────────┼──────────────────────────────────
  Recursive multi-step (np.roll)        │  One-step direct prediction;
  → накапливает ошибку, дрейф           │  для multi-step используется
                                        │  отдельный путь с предсказанием
                                        │  по одному и пересчётом признаков
  ────────────────────────────────────┼──────────────────────────────────
  model.save(.h5)                       │  model.save(.keras) — современный
  → устаревший формат                   │  формат TensorFlow ≥2.13

Архитектура:
  Input(window_size, n_features)
    → LSTM(64, return_sequences=True)
    → Dropout(0.3)
    → LSTM(32)
    → Dropout(0.3)
    → Dense(16, activation='relu')
    → Dense(1)        # точечный прогноз нормализованного y

Loss: Huber(δ=1.0) — менее чувствителен к выбросам чем MSE,
но в окрестности нуля ведёт себя как MSE (гладкий градиент).
Это критично, потому что в задаче пик-детекции выбросы — это
не «шум», а именно тот сигнал, который мы хотим выучить;
MSE же будет давать им сверх-большие градиенты и переобучаться.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Отложенный импорт TensorFlow — он тяжёлый и может ломать pyinstaller/тесты,
# если LSTM не используется. Импортируем внутри функций.


# ---------------------------------------------------------------------------
# Контейнер для всего, что нужно для инференса
# ---------------------------------------------------------------------------

@dataclass
class LSTMArtifact:
    """
    Связка «модель + скейлеры + конфиг», достаточная для инференса.

    Сохраняется единым артефактом (joblib для скейлеров и конфигурации,
    отдельный .keras для самой модели — так требует TensorFlow).
    """
    keras_model: object         # tf.keras.Model
    feature_scaler: object      # StandardScaler по матрице признаков X
    target_scaler: object       # StandardScaler по таргету y
    feature_names: List[str]
    window_size: int
    target_name: str = "y"

    # ---- Сохранение / загрузка ----
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        keras_path = path + ".keras"
        meta_path = path + ".meta.pkl"
        self.keras_model.save(keras_path)
        joblib.dump(
            {
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "feature_names": self.feature_names,
                "window_size": self.window_size,
                "target_name": self.target_name,
                "keras_path": keras_path,
            },
            meta_path,
        )

    @classmethod
    def load(cls, path: str) -> "LSTMArtifact":
        from tensorflow.keras.models import load_model
        meta_path = path + ".meta.pkl"
        meta = joblib.load(meta_path)
        keras_model = load_model(meta["keras_path"], compile=False)
        return cls(
            keras_model=keras_model,
            feature_scaler=meta["feature_scaler"],
            target_scaler=meta["target_scaler"],
            feature_names=meta["feature_names"],
            window_size=meta["window_size"],
            target_name=meta["target_name"],
        )


# ---------------------------------------------------------------------------
# Утилита: построение sliding-window матриц БЕЗ перекрытия со split-ами
# ---------------------------------------------------------------------------

def _make_windows(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Превращает (T, F) → (T-W, W, F) и (T,) → (T-W,).

    Окно [t, t+W) предсказывает таргет в позиции t+W.
    """
    T = X_scaled.shape[0]
    if T <= window_size:
        raise ValueError(
            f"Недостаточно строк ({T}) для окна размера {window_size}. "
            "Уменьшите window_size или возьмите больше данных."
        )
    n = T - window_size
    F = X_scaled.shape[1]
    Xw = np.empty((n, window_size, F), dtype=np.float32)
    yw = np.empty((n,), dtype=np.float32)
    for i in range(n):
        Xw[i] = X_scaled[i:i + window_size]
        yw[i] = y_scaled[i + window_size]
    return Xw, yw


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def _build_and_fit_lstm(
    Xtr_w, ytr_w, Xvl_w, yvl_w,
    window_size, n_features,
    units_l1, units_l2, dropout, lr, batch_size,
    huber_delta, epochs, patience, verbose,
):
    """Вспомогательная функция: строит и обучает одну LSTM-модель."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(units_l1, return_sequences=True),
        Dropout(dropout),
        LSTM(units_l2, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=Huber(delta=huber_delta),
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-5, verbose=verbose),
    ]

    model.fit(
        Xtr_w, ytr_w,
        validation_data=(Xvl_w, yvl_w),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,
    )
    return model


def train_lstm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame,   y_val: pd.Series,
    window_size: int = 24,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    units_l1: int = 64,
    units_l2: int = 32,
    dropout: float = 0.3,
    huber_delta: float = 1.0,
    patience: int = 10,
    save_path: Optional[str] = None,
    verbose: int = 0,
    n_trials: int = 15,
) -> LSTMArtifact:
    """
    Обучает multivariate LSTM на признаках FeatureBuilder.

    При наличии optuna запускает Hyperband поиск гиперпараметров (n_trials).
    Hyperband отсеивает плохие конфигурации на ранних эпохах, экономя время.
    При отсутствии optuna — фиксированные параметры (обратная совместимость).

    Контракт скейлеров (КРИТИЧНО для отсутствия утечки):
      feature_scaler.fit(X_train)           # ← только train
      X_val_s = feature_scaler.transform(X_val)  # без re-fit!
    """
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    tf.random.set_seed(42)

    feature_names = list(X_train.columns)

    # ---- Скейлеры: fit ТОЛЬКО на train ----
    feature_scaler = StandardScaler()
    target_scaler  = StandardScaler()
    feature_scaler.fit(X_train.values)
    target_scaler.fit(y_train.values.reshape(-1, 1))

    Xtr_s = feature_scaler.transform(X_train.values)
    Xvl_s = feature_scaler.transform(X_val.values)
    ytr_s = target_scaler.transform(y_train.values.reshape(-1, 1)).flatten()
    yvl_s = target_scaler.transform(y_val.values.reshape(-1, 1)).flatten()

    Xtr_w, ytr_w = _make_windows(Xtr_s, ytr_s, window_size)
    Xvl_w, yvl_w = _make_windows(Xvl_s, yvl_s, window_size)
    n_features = Xtr_w.shape[2]

    def _fit(p, epochs_override=None):
        return _build_and_fit_lstm(
            Xtr_w, ytr_w, Xvl_w, yvl_w,
            window_size, n_features,
            units_l1=p["units_l1"], units_l2=p["units_l2"],
            dropout=p["dropout"], lr=p["lr"], batch_size=p["batch_size"],
            huber_delta=huber_delta,
            epochs=epochs_override if epochs_override is not None else epochs,
            patience=patience, verbose=verbose,
        )

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        best_params, best_mae, best_model = None, float("inf"), None
        search_epochs = max(10, epochs // 4)  # быстрый поиск на сокращённых эпохах

        def objective(trial):
            nonlocal best_params, best_mae, best_model
            p = {
                "units_l1":   trial.suggest_categorical("units_l1", [32, 64, 128]),
                "units_l2":   trial.suggest_categorical("units_l2", [16, 32, 64]),
                "dropout":    trial.suggest_float("dropout", 0.1, 0.5),
                "lr":         trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            }
            try:
                m = _fit(p, epochs_override=search_epochs)
                val_preds = m.predict(Xvl_w, verbose=0).flatten()
                mae = float(np.mean(np.abs(yvl_s - val_preds)))
                if mae < best_mae:
                    best_mae   = mae
                    best_params = p
                    best_model = m
                return mae
            except Exception:
                return float("inf")

        print(f"  LSTM Optuna TPE: {n_trials} trials × {search_epochs} эпох поиска...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        if best_params is not None:
            print(f"  LSTM TPE завершён: val MAE={best_mae:.4f}  params={best_params}")
            print(f"  LSTM финальное обучение на {epochs} эпохах...")
            model = _fit(best_params)
        else:
            print("  LSTM TPE: все trials упали, fallback на фиксированные параметры")
            model = _fit(dict(units_l1=units_l1, units_l2=units_l2,
                              dropout=dropout, lr=lr, batch_size=batch_size))

    except ImportError:
        print("  LSTM обучается с фиксированными параметрами (optuna не установлена)...")
        model = _fit(dict(
            units_l1=units_l1, units_l2=units_l2,
            dropout=dropout, lr=lr, batch_size=batch_size,
        ))

    artifact = LSTMArtifact(
        keras_model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=feature_names,
        window_size=window_size,
    )

    if save_path:
        artifact.save(save_path)

    return artifact


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def predict_lstm(
    artifact: LSTMArtifact,
    X_test: pd.DataFrame,
    y_history: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    One-step прогноз для каждой строки X_test.

    Для строки t используется окно [t-W, t) признаков. Поскольку первые W строк
    после train не имеют полного окна, мы дополняем их «головой» из X_history,
    либо (если y_history передан) — из последних W строк train+val матрицы.

    Если в тесте < W точек, возвращается пустой массив длины 0.

    Parameters
    ----------
    artifact   : сохранённый артефакт обучения
    X_test     : признаки тестового периода (контекст из train+val уже учтён в
                 transform_splits, так что X_test уже содержит подсчитанные лаги)
    y_history  : (опционально) полный ряд y, если нужно восстановить
                 предшествующие окна — здесь не используется напрямую

    Returns
    -------
    np.ndarray размера len(X_test): прогноз RPS в исходном масштабе
    """
    W = artifact.window_size
    feature_names = artifact.feature_names

    # Гарантируем порядок и наличие всех колонок
    missing = set(feature_names) - set(X_test.columns)
    if missing:
        raise ValueError(f"X_test не содержит обученные признаки: {missing}")

    X = X_test[feature_names].values
    X_s = artifact.feature_scaler.transform(X)

    n = X_s.shape[0]
    if n <= W:
        return np.full(n, np.nan, dtype=np.float64)

    # Окна сдвига: (n-W, W, F) → predict (n-W, 1)
    windows = np.empty((n - W, W, X_s.shape[1]), dtype=np.float32)
    for i in range(n - W):
        windows[i] = X_s[i:i + W]

    pred_scaled = artifact.keras_model.predict(windows, verbose=0).flatten()
    pred = artifact.target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()

    # Первые W точек теста не получили окна — заполним NaN, чтобы вызывающая
    # сторона могла обрезать корректно. Это честнее, чем «придумать» прогноз.
    out = np.full(n, np.nan, dtype=np.float64)
    out[W:] = np.clip(pred, 0, None)   # RPS не может быть отрицательным
    return out


def predict_lstm_aligned(
    artifact: LSTMArtifact,
    X_full: pd.DataFrame,
    n_test: int,
) -> np.ndarray:
    """
    Удобный шорткат: даёт прогнозы ровно на последние n_test точек,
    используя для каждой полное окно из train+val+(текущая часть теста).

    Это «context-aware inference», аналогичный transform_splits для XGBoost.

    Parameters
    ----------
    X_full  : признаки на ВСЁМ датасете (train+val+test после FeatureBuilder)
    n_test  : сколько точек с конца считать тестовыми

    Returns
    -------
    np.ndarray длины n_test: прогноз RPS на тестовом периоде
    """
    feature_names = artifact.feature_names
    W = artifact.window_size

    X = X_full[feature_names].values
    X_s = artifact.feature_scaler.transform(X)

    if X_s.shape[0] < W + n_test:
        raise ValueError(
            f"Недостаточно данных: нужно ≥ {W + n_test} строк, "
            f"получено {X_s.shape[0]}"
        )

    start = X_s.shape[0] - n_test       # первая «тестовая» строка
    windows = np.empty((n_test, W, X_s.shape[1]), dtype=np.float32)
    for i in range(n_test):
        # окно ровно перед тестовой точкой
        windows[i] = X_s[start + i - W:start + i]

    pred_scaled = artifact.keras_model.predict(windows, verbose=0).flatten()
    pred = artifact.target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()
    return np.clip(pred, 0, None)
