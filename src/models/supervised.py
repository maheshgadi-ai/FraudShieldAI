"""
Supervised Learning Models for FraudShieldAI.

Implements:
  - Logistic Regression
  - Random Forest
  - XGBoost

Each model exposes:
  - build(cfg)           → untrained estimator with config params
  - tune(X, y, cfg)      → Optuna study → best estimator
  - train(X, y, cfg)     → fit best (or default) estimator

All functions return a fitted sklearn-compatible estimator and the
Optuna study (or None when tuning is skipped).
"""

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared cross-validation helper
# ---------------------------------------------------------------------------

def _cv_score(estimator, X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> float:
    """
    Return mean AUC-PR (Average Precision) via stratified k-fold.
    AUC-PR is more informative than AUC-ROC on severely imbalanced data.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = cross_val_predict(
        estimator, X, y, cv=skf, method="predict_proba", n_jobs=-1
    )[:, 1]
    return average_precision_score(y, probs)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def _build_lr(params: dict) -> LogisticRegression:
    return LogisticRegression(
        C=params.get("C", 1.0),
        max_iter=params.get("max_iter", 1000),
        class_weight=params.get("class_weight", "balanced"),
        solver=params.get("solver", "lbfgs"),
        random_state=params.get("random_state", 42),
    )


def _tune_lr(
    X: np.ndarray, y: np.ndarray, cfg: dict, n_trials: int = 30
) -> tuple[LogisticRegression, optuna.Study]:
    space = cfg.get("tuning", {}).get("param_space", {})

    def objective(trial: optuna.Trial) -> float:
        C_range = space.get("C", [0.001, 100.0])
        params = {
            "C": trial.suggest_float("C", C_range[0], C_range[1], log=True),
            "max_iter": trial.suggest_int(
                "max_iter", *space.get("max_iter", [500, 2000])
            ),
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": 42,
        }
        return _cv_score(_build_lr(params), X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = _build_lr(study.best_params)
    best.fit(X, y)
    logger.info("[LR] Best AUC-PR: %.4f | params: %s", study.best_value, study.best_params)
    return best, study


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[LogisticRegression, optuna.Study | None]:
    """Train Logistic Regression, with optional Optuna tuning."""
    if cfg.get("tuning", {}).get("method") == "optuna":
        n_trials = cfg["tuning"].get("n_trials", 30)
        logger.info("[LR] Starting Optuna tuning (%d trials)…", n_trials)
        return _tune_lr(X, y, cfg, n_trials=n_trials)

    logger.info("[LR] Training with default params…")
    model = _build_lr(cfg.get("params", {}))
    model.fit(X, y)
    return model, None


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def _build_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 15),
        min_samples_split=params.get("min_samples_split", 5),
        min_samples_leaf=params.get("min_samples_leaf", 2),
        class_weight=params.get("class_weight", "balanced"),
        n_jobs=params.get("n_jobs", -1),
        random_state=params.get("random_state", 42),
    )


def _tune_rf(
    X: np.ndarray, y: np.ndarray, cfg: dict, n_trials: int = 30
) -> tuple[RandomForestClassifier, optuna.Study]:
    space = cfg.get("tuning", {}).get("param_space", {})

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", *space.get("n_estimators", [100, 500])
            ),
            "max_depth": trial.suggest_int(
                "max_depth", *space.get("max_depth", [5, 30])
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", *space.get("min_samples_split", [2, 20])
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", *space.get("min_samples_leaf", [1, 10])
            ),
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
        }
        return _cv_score(_build_rf(params), X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = _build_rf(study.best_params)
    best.fit(X, y)
    logger.info("[RF] Best AUC-PR: %.4f | params: %s", study.best_value, study.best_params)
    return best, study


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[RandomForestClassifier, optuna.Study | None]:
    """Train Random Forest, with optional Optuna tuning."""
    if cfg.get("tuning", {}).get("method") == "optuna":
        n_trials = cfg["tuning"].get("n_trials", 30)
        logger.info("[RF] Starting Optuna tuning (%d trials)…", n_trials)
        return _tune_rf(X, y, cfg, n_trials=n_trials)

    logger.info("[RF] Training with default params…")
    model = _build_rf(cfg.get("params", {}))
    model.fit(X, y)
    return model, None


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def _build_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 8),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        scale_pos_weight=params.get("scale_pos_weight", 10),
        eval_metric=params.get("eval_metric", "aucpr"),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", -1),
        tree_method="hist",           # fast histogram method
        verbosity=0,
    )


def _tune_xgb(
    X: np.ndarray, y: np.ndarray, cfg: dict, n_trials: int = 50
) -> tuple[XGBClassifier, optuna.Study]:
    space = cfg.get("tuning", {}).get("param_space", {})

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", *space.get("n_estimators", [200, 1000])
            ),
            "max_depth": trial.suggest_int(
                "max_depth", *space.get("max_depth", [3, 12])
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *space.get("learning_rate", [0.01, 0.3]), log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", *space.get("subsample", [0.6, 1.0])
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *space.get("colsample_bytree", [0.6, 1.0])
            ),
            "scale_pos_weight": 10,
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }
        return _cv_score(_build_xgb(params), X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = _build_xgb(study.best_params)
    best.fit(X, y)
    logger.info("[XGB] Best AUC-PR: %.4f | params: %s", study.best_value, study.best_params)
    return best, study


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[XGBClassifier, optuna.Study | None]:
    """Train XGBoost, with optional Optuna tuning."""
    if cfg.get("tuning", {}).get("method") == "optuna":
        n_trials = cfg["tuning"].get("n_trials", 50)
        logger.info("[XGB] Starting Optuna tuning (%d trials)…", n_trials)
        return _tune_xgb(X, y, cfg, n_trials=n_trials)

    logger.info("[XGB] Training with default params…")
    model = _build_xgb(cfg.get("params", {}))
    model.fit(X, y)
    return model, None


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def train_supervised_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_cfg: dict,
) -> dict[str, Any]:
    """
    Train all enabled supervised models.

    Returns:
        dict mapping model name → {"model": fitted_estimator, "study": optuna_study | None}
    """
    results: dict[str, Any] = {}

    if model_cfg.get("logistic_regression", {}).get("enabled", True):
        model, study = train_logistic_regression(X_train, y_train, model_cfg["logistic_regression"])
        results["logistic_regression"] = {"model": model, "study": study}

    if model_cfg.get("random_forest", {}).get("enabled", True):
        model, study = train_random_forest(X_train, y_train, model_cfg["random_forest"])
        results["random_forest"] = {"model": model, "study": study}

    if model_cfg.get("xgboost", {}).get("enabled", True):
        model, study = train_xgboost(X_train, y_train, model_cfg["xgboost"])
        results["xgboost"] = {"model": model, "study": study}

    return results
