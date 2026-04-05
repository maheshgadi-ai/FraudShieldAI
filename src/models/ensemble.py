"""
Hybrid Ensemble for FraudShieldAI.

Two ensemble strategies (configurable via models.yaml):

1. Soft Voting
   Averages predicted fraud probabilities across all base models.
   Simple, interpretable, robust. Default choice.

2. Stacking
   Uses out-of-fold (OOF) predictions from base models as features
   for a meta-learner (Logistic Regression by default).
   More powerful but slower to train.

Both strategies expose predict / predict_proba for downstream evaluation
and the Streamlit app.
"""

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft Voting Ensemble
# ---------------------------------------------------------------------------

class SoftVotingEnsemble:
    """
    Averages predict_proba() outputs from a collection of fitted base models.
    Compatible with any model that implements predict_proba(X) -> (n, 2).
    """

    def __init__(self, models: dict[str, Any], weights: dict[str, float] | None = None):
        """
        Args:
            models:  {name: fitted_model} mapping.
            weights: Optional per-model weights (uniform if None).
        """
        self.models = models
        self.weights = weights

    def predict_proba(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """
        Args:
            inputs: {model_name: X_array} — allows models expecting different
                    input shapes (e.g. LSTM expects 3-D sequences vs 2-D for others).

        Returns:
            Averaged probability array of shape (n_samples, 2).
        """
        all_probs: list[np.ndarray] = []
        model_names = list(self.models.keys())

        for name, model in self.models.items():
            X = inputs.get(name, inputs.get("default"))
            probs = model.predict_proba(X)   # (n, 2)
            all_probs.append(probs)

        if self.weights is None:
            avg_probs = np.mean(all_probs, axis=0)
        else:
            w = np.array([self.weights.get(n, 1.0) for n in model_names])
            w = w / w.sum()
            avg_probs = np.average(all_probs, axis=0, weights=w)

        return avg_probs

    def predict(self, inputs: dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(inputs)[:, 1] >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """
    Stacking ensemble with out-of-fold meta-features.

    Training:
      1. For each base model, generate OOF predictions on the training set
         (via k-fold cross-validation) to avoid target leakage.
      2. Stack OOF predictions as columns → meta-feature matrix.
      3. Fit meta-learner on meta-features + true labels.

    Inference:
      1. Each base model predicts on the test set.
      2. Stack predictions → meta-feature matrix.
      3. Meta-learner predicts final probability.
    """

    def __init__(
        self,
        base_models: dict[str, Any],
        meta_learner: Any | None = None,
        cv_folds: int = 5,
    ):
        self.base_models = base_models
        self.meta_learner = meta_learner or LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42
        )
        self.cv_folds = cv_folds
        self._fitted_base: dict[str, Any] = {}

    def fit(
        self,
        inputs_train: dict[str, np.ndarray],
        y_train: np.ndarray,
    ) -> "StackingEnsemble":
        """
        Args:
            inputs_train: {model_name: X_array} for the training set.
            y_train:      Labels for the training set.
        """
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        n_samples = len(y_train)
        n_models = len(self.base_models)
        oof_preds = np.zeros((n_samples, n_models), dtype=np.float32)

        logger.info("[Stacking] Generating OOF predictions (%d folds)…", self.cv_folds)

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(
            next(iter(inputs_train.values())), y_train
        )):
            for model_j, (name, model) in enumerate(self.base_models.items()):
                X = inputs_train[name]
                X_fold_tr = X[train_idx]
                X_fold_val = X[val_idx]
                y_fold_tr = y_train[train_idx]

                # Clone + refit base model on fold
                import copy
                fold_model = copy.deepcopy(model)
                fold_model.fit(X_fold_tr, y_fold_tr)
                oof_preds[val_idx, model_j] = fold_model.predict_proba(X_fold_val)[:, 1]

            logger.info("[Stacking] Fold %d/%d complete", fold_i + 1, self.cv_folds)

        # Train meta-learner on full OOF predictions
        logger.info("[Stacking] Fitting meta-learner…")
        self.meta_learner.fit(oof_preds, y_train)

        # Refit all base models on full training data for inference
        for name, model in self.base_models.items():
            model.fit(inputs_train[name], y_train)
            self._fitted_base[name] = model

        logger.info("[Stacking] Ensemble training complete.")
        return self

    def predict_proba(self, inputs_test: dict[str, np.ndarray]) -> np.ndarray:
        n_samples = len(next(iter(inputs_test.values())))
        meta_features = np.zeros((n_samples, len(self._fitted_base)), dtype=np.float32)
        for j, (name, model) in enumerate(self._fitted_base.items()):
            meta_features[:, j] = model.predict_proba(inputs_test[name])[:, 1]
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, inputs_test: dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(inputs_test)[:, 1] >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ensemble(
    fitted_models: dict[str, Any],
    cfg: dict,
) -> SoftVotingEnsemble | StackingEnsemble:
    """
    Build the ensemble from already-fitted base models.

    Args:
        fitted_models: {name: fitted_model}
        cfg:           models.yaml['ensemble'] section

    Returns:
        A SoftVotingEnsemble or StackingEnsemble instance.
    """
    method = cfg.get("method", "soft_voting")
    enabled_names = cfg.get("models", list(fitted_models.keys()))
    models = {k: v for k, v in fitted_models.items() if k in enabled_names}

    logger.info("[Ensemble] Method: %s | Models: %s", method, list(models.keys()))

    if method == "soft_voting":
        return SoftVotingEnsemble(models=models)

    if method == "stacking":
        meta_cfg = cfg.get("stacking", {})
        return StackingEnsemble(
            base_models=models,
            cv_folds=meta_cfg.get("cv_folds", 5),
        )

    raise ValueError(f"Unknown ensemble method: {method!r}")
