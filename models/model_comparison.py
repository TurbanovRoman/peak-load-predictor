"""
Сравнение трёх моделей прогнозирования (глава 2 ВКР).

Единый интерфейс для обучения и оценки на одних данных:
  1. XGBoost        — лаговые + календарные признаки
  2. LSTM           — скользящее окно, Huber-loss, StandardScaler
  3. NeuralProphet  — тренд/сезонность + AR-Net

Использование:
    comparator = ModelComparison()
    results = comparator.run(train, val, test)
    best_name, best_model = comparator.best_model()
"""

import os
import time
import warnings
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from preprocessing.feature_engineering import (
    FeatureBuilder,
    _infer_step_minutes,
    _hours_to_periods,
)
from models.xgboost_model import train_xgboost, predict_xgboost
from models.forecasters import train_xgboost_random_search
from evaluation.metrics import evaluate, print_comparison_table


class ModelComparison:
    """
    Запускает сравнительный эксперимент: одни данные, разные модели.

    Атрибуты после вызова .run():
        results_      : dict {model_name -> metrics_dict}
        models_       : dict {model_name -> обученный объект модели}
        predictions_  : dict {model_name -> np.ndarray прогнозов на test}
        winner_       : имя лучшей модели
    """

    def __init__(
        self,
        model_save_dir: str = "./saved_models",
        primary_metric: str = "MAE",
        exog_cols=None,
    ):
        self.model_save_dir = model_save_dir
        self.primary_metric = primary_metric
        self.results_: Dict[str, dict] = {}
        self.models_: Dict[str, object] = {}
        self.predictions_: Dict[str, np.ndarray] = {}
        self.winner_: Optional[str] = None
        if exog_cols is None:
            try:
                import config as _cfg
                exog_cols = getattr(_cfg, "EXOG_COLS", [])
            except Exception:
                exog_cols = []
        self.exog_cols = list(exog_cols)
        # FeatureBuilder включит exog в признаки только если они есть в данных.
        self._builder = FeatureBuilder(exog_cols=self.exog_cols)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def run(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        include_prophet: bool = True,
        include_lstm: bool = True,
    ) -> Dict[str, dict]:
        """
        Обучает все модели и оценивает на test.

        Parameters
        ----------
        train, val, test  : DataFrame с колонками ds, y
        include_prophet   : включить NeuralProphet
        include_lstm      : включить LSTM (требует TensorFlow)

        Returns
        -------
        dict {model_name -> {"MAE": ..., "RMSE": ..., "MAPE": ...}}
        """
        print("=" * 60)
        print("СРАВНЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
        print("=" * 60)

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            self._builder.transform_splits(train, val, test)

        n_models = 1 + int(include_lstm) + int(include_prophet)

        # --- 1. XGBoost ---
        self._fit_xgboost(X_train, y_train, X_val, y_val, X_test, y_test,
                          label=f"[1/{n_models}]")

        step = 2
        # --- 2. LSTM ---
        if include_lstm:
            self._fit_lstm(X_train, y_train, X_val, y_val, X_test, y_test,
                           label=f"[{step}/{n_models}]")
            step += 1

        # --- 3. NeuralProphet ---
        if include_prophet:
            self._fit_neural_prophet(train, val, test, y_test,
                                     label=f"[{step}/{n_models}]")

        # --- Итоговая таблица и выбор победителя ---
        self.winner_ = self._select_winner()
        print_comparison_table(self.results_, primary_metric=self.primary_metric)
        print(f"\nЛучшая модель по {self.primary_metric}: {self.winner_}")
        print("=" * 60)

        return self.results_

    def best_model(self) -> Tuple[str, object]:
        """Возвращает (имя, объект) лучшей модели."""
        if self.winner_ is None:
            raise RuntimeError("Вызовите .run() перед .best_model()")
        return self.winner_, self.models_[self.winner_]

    def save_best(self, path: Optional[str] = None) -> str:
        """Сохраняет лучшую модель. Возвращает путь к файлу."""
        name, model = self.best_model()
        os.makedirs(self.model_save_dir, exist_ok=True)

        if model is None:
            print(f"Best model ({name}) — already saved during training.")
            return ""

        # LSTMArtifact — собственный метод сохранения
        from models.lstm_model import LSTMArtifact
        if isinstance(model, LSTMArtifact):
            save_path = path or os.path.join(self.model_save_dir, f"best_{name}")
            model.save(save_path)
            print(f"Best model ({name}) saved -> {save_path}")
            return save_path

        # NeuralProphet — собственная сериализация
        try:
            from neuralprophet import NeuralProphet as _NP
            if isinstance(model, _NP):
                from models.forecasters import save_prophet
                np_path = path or os.path.join(
                    self.model_save_dir, f"best_{name}.np"
                )
                save_prophet(model, np_path)
                print(f"Best model ({name}) saved -> {np_path}")
                return np_path
        except ImportError:
            pass

        # XGBoost / sklearn — joblib
        save_path = path or os.path.join(self.model_save_dir, f"best_{name}.pkl")
        joblib.dump(model, save_path)
        print(f"Best model ({name}) saved -> {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # Внутренние методы обучения
    # ------------------------------------------------------------------

    def _fit_xgboost(self, X_train, y_train, X_val, y_val, X_test, y_test,
                     label: str = "[1/3]"):
        print(f"\n{label} XGBoost...")
        t0 = time.time()
        save_path = os.path.join(self.model_save_dir, "xgboost.pkl")
        model, _, _ = train_xgboost_random_search(X_train, y_train, X_val, y_val, save_path=save_path)
        preds = predict_xgboost(model, X_test)
        elapsed = time.time() - t0
        metrics = evaluate(y_test.values, preds, "XGBoost", verbose=True)
        metrics["train_time_s"] = round(elapsed, 2)
        self.results_["XGBoost"] = metrics
        self.models_["XGBoost"] = model
        self.predictions_["XGBoost"] = preds

    def _fit_lstm(self, X_train, y_train, X_val, y_val, X_test, y_test,
                  label: str = "[2/3]"):
        print(f"\n{label} LSTM...")
        try:
            from models.lstm_model import train_lstm, predict_lstm_aligned

            # window_size = ~24ч в периодах (по частоте данных)
            step_min = _infer_step_minutes(y_train.index)
            window_size = _hours_to_periods(24, step_min)
            window_size = max(12, min(window_size, 288))

            t0 = time.time()
            save_path = os.path.join(self.model_save_dir, "lstm")
            artifact = train_lstm(
                X_train, y_train, X_val, y_val,
                window_size=window_size,
                save_path=save_path,
            )

            # Context-aware инференс: передаём полный X (train+val+test)
            X_full = pd.concat([X_train, X_val, X_test])
            preds = predict_lstm_aligned(artifact, X_full, n_test=len(X_test))

            elapsed = time.time() - t0
            metrics = evaluate(y_test.values, preds, "LSTM", verbose=True)
            metrics["train_time_s"] = round(elapsed, 2)
            self.results_["LSTM"] = metrics
            self.models_["LSTM"] = artifact
            self.predictions_["LSTM"] = preds
        except Exception as e:
            print(f"  LSTM пропущен: {e}")

    def _fit_neural_prophet(self, train, val, test, y_test,
                            label: str = "[3/3]"):
        print(f"\n{label} NeuralProphet (AR-Net + декомпозиция)...")
        try:
            from models.neural_prophet_model import (
                train_neural_prophet,
                predict_neural_prophet,
            )
            t0 = time.time()
            save_path = os.path.join(self.model_save_dir, "neural_prophet")
            model = train_neural_prophet(
                train_df=train,
                val_df=val,
                save_path=save_path,
            )
            train_val = pd.concat([train, val]).sort_values("ds")
            preds = predict_neural_prophet(
                model, train_val_df=train_val, test_df=test
            )
            preds = preds[: len(y_test)]
            elapsed = time.time() - t0
            metrics = evaluate(y_test.values, preds, "NeuralProphet", verbose=True)
            metrics["train_time_s"] = round(elapsed, 2)
            self.results_["NeuralProphet"] = metrics
            self.models_["NeuralProphet"] = model
            self.predictions_["NeuralProphet"] = preds
        except Exception as e:
            print(f"  NeuralProphet пропущен: {e}")

    def _select_winner(self) -> str:
        return min(
            self.results_,
            key=lambda name: self.results_[name].get(self.primary_metric, float("inf")),
        )
