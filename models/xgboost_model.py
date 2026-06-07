"""
XGBoost — основная прогностическая модель (глава 2.6.2 ВКР).

Особенности реализации:
  - Early stopping по валидационной выборке (параметр early_stopping_rounds)
  - Квантильная регрессия для доверительных интервалов (q=0.1 и q=0.9)
  - Сохранение важности признаков
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ---------------------------------------------------------------------------
# Основная модель (точечный прогноз)
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 20,
    save_path: str = None,
):
    """
    Обучает XGBRegressor с ранней остановкой.

    Parameters
    ----------
    X_train, y_train : обучающая выборка
    X_val,   y_val   : валидационная выборка (для ранней остановки)
    early_stopping_rounds : число раундов без улучшения до остановки
    save_path        : путь для сохранения через joblib (None — не сохранять)
    """
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        joblib.dump(model, save_path)

    print(f"  XGBoost обучен: {model.best_iteration + 1} деревьев "
          f"(early stopping @ {early_stopping_rounds})")
    return model


def predict_xgboost(model, X) -> np.ndarray:
    """Точечный прогноз. Значения < 0 обнуляются."""
    preds = model.predict(X)
    return np.clip(preds, 0, None)


# ---------------------------------------------------------------------------
# Квантильная регрессия — доверительные интервалы
# ---------------------------------------------------------------------------

def train_xgboost_quantile(
    X_train,
    y_train,
    X_val,
    y_val,
    quantile: float = 0.9,
    n_estimators: int = 300,
    early_stopping_rounds: int = 20,
    save_path: str = None,
):
    """
    Обучает модель квантильной регрессии XGBoost.
    Для 80%-го доверительного интервала нужны q=0.1 и q=0.9.
    """
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="quantile",
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        joblib.dump(model, save_path)

    return model


def get_confidence_interval(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
    save_dir: str = None,
):
    """
    Возвращает нижнюю и верхнюю границы прогноза.

    Returns
    -------
    (lower, upper) — два numpy-массива того же размера, что X_test
    """
    if len(X_val) == 0 or len(X_test) == 0:
        n = len(X_test) if len(X_test) > 0 else len(X_train)
        return np.zeros(n), np.zeros(n)

    model_low = train_xgboost_quantile(
        X_train, y_train, X_val, y_val,
        quantile=lower_q,
        save_path=os.path.join(save_dir, "xgb_q10.pkl") if save_dir else None,
    )
    model_high = train_xgboost_quantile(
        X_train, y_train, X_val, y_val,
        quantile=upper_q,
        save_path=os.path.join(save_dir, "xgb_q90.pkl") if save_dir else None,
    )
    lower = np.clip(model_low.predict(X_test), 0, None)
    upper = np.clip(model_high.predict(X_test), 0, None)
    return lower, upper


# ---------------------------------------------------------------------------
# Важность признаков
# ---------------------------------------------------------------------------

def feature_importance(model, feature_names=None) -> pd.DataFrame:
    """
    Возвращает DataFrame с важностью признаков XGBoost (gain).
    """
    imp = model.feature_importances_
    names = feature_names if feature_names is not None else [
        f"f{i}" for i in range(len(imp))
    ]
    df = pd.DataFrame({"feature": names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
