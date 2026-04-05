"""
Metric computation for FraudShieldAI.

All functions take (y_true, y_pred_proba) or (y_true, y_pred_binary)
and return either a scalar or a dict of scalars ready for reporting.

Metrics tracked:
  - Precision, Recall, F1 (at 0.5 threshold and optimal threshold)
  - AUC-ROC
  - AUC-PR  (Average Precision — primary metric for imbalanced data)
  - Matthews Correlation Coefficient (MCC)
  - Confusion matrix components (TP, FP, TN, FN)
  - Optimal threshold (maximises F1 on the PR curve)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def optimal_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Return the probability threshold that maximises F1 on the
    precision-recall curve.  Avoids scanning every sample by using the
    thresholds already computed by sklearn.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision and recall have one extra element (for threshold=0)
    f1_scores = np.where(
        (precision[:-1] + recall[:-1]) == 0,
        0.0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9),
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float | None = None,
    model_name: str = "",
) -> dict[str, Any]:
    """
    Compute the full set of evaluation metrics for a binary classifier.

    Args:
        y_true:     Ground-truth labels (0/1).
        y_proba:    Predicted fraud probability (class-1 probability).
        threshold:  Decision threshold.  If None, the optimal F1 threshold
                    is selected automatically.
        model_name: Optional label for logging.

    Returns:
        Dict of metric name → value.
    """
    if threshold is None:
        threshold = optimal_f1_threshold(y_true, y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auc_roc  = roc_auc_score(y_true, y_proba)
    auc_pr   = average_precision_score(y_true, y_proba)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true, y_pred, zero_division=0)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    mcc      = matthews_corrcoef(y_true, y_pred)

    results = {
        "model": model_name,
        "threshold": round(threshold, 4),
        "auc_roc":   round(auc_roc, 4),
        "auc_pr":    round(auc_pr, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "mcc":       round(mcc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    logger.info(
        "[%s] AUC-PR: %.4f | AUC-ROC: %.4f | F1: %.4f | "
        "Prec: %.4f | Recall: %.4f | MCC: %.4f | threshold: %.4f",
        model_name or "model",
        auc_pr, auc_roc, f1, prec, rec, mcc, threshold,
    )
    return results


def build_metrics_table(all_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of per-model metric dicts into a sorted comparison DataFrame.
    Sorted by AUC-PR descending (primary metric).
    """
    df = pd.DataFrame(all_metrics)
    if df.empty or "auc_pr" not in df.columns:
        return df
    display_cols = [
        "model", "auc_pr", "auc_roc", "f1", "precision", "recall", "mcc", "threshold"
    ]
    df = df[[c for c in display_cols if c in df.columns]]
    return df.sort_values("auc_pr", ascending=False).reset_index(drop=True)
