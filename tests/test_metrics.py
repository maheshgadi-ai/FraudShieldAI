"""
Unit tests for src/evaluation/metrics.py
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    build_metrics_table,
    compute_metrics,
    optimal_f1_threshold,
)


# ---------------------------------------------------------------------------
# optimal_f1_threshold
# ---------------------------------------------------------------------------

class TestOptimalThreshold:
    def test_returns_float_in_unit_interval(self, binary_predictions):
        y_true, y_proba = binary_predictions
        t = optimal_f1_threshold(y_true, y_proba)
        assert isinstance(t, float)
        assert 0.0 <= t <= 1.0

    def test_perfect_separator_threshold_near_boundary(self):
        """When classes are perfectly separated the threshold sits between them."""
        y_true  = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        t = optimal_f1_threshold(y_true, y_proba)
        # The optimal cut sits somewhere between 0.3 and 0.8
        assert 0.3 <= t <= 0.9

    def test_all_same_proba_does_not_crash(self):
        y_true  = np.array([0, 1, 0, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])
        # Should not raise even if F1 is degenerate
        t = optimal_f1_threshold(y_true, y_proba)
        assert 0.0 <= t <= 1.0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_required_keys(self, binary_predictions):
        y_true, y_proba = binary_predictions
        result = compute_metrics(y_true, y_proba, model_name="test")
        required = {"model", "auc_roc", "auc_pr", "precision", "recall",
                    "f1", "mcc", "tp", "fp", "tn", "fn", "threshold"}
        assert required.issubset(result.keys())

    def test_metrics_in_valid_range(self, binary_predictions):
        y_true, y_proba = binary_predictions
        m = compute_metrics(y_true, y_proba)
        for key in ["auc_roc", "auc_pr", "precision", "recall", "f1"]:
            assert 0.0 <= m[key] <= 1.0, f"{key} = {m[key]} out of range"

    def test_mcc_in_valid_range(self, binary_predictions):
        y_true, y_proba = binary_predictions
        m = compute_metrics(y_true, y_proba)
        assert -1.0 <= m["mcc"] <= 1.0

    def test_confusion_matrix_components_sum_to_n(self, binary_predictions):
        y_true, y_proba = binary_predictions
        m = compute_metrics(y_true, y_proba)
        assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(y_true)

    def test_perfect_classifier(self):
        y_true  = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.01, 0.02, 0.03, 0.97, 0.98])
        m = compute_metrics(y_true, y_proba, threshold=0.5)
        assert m["auc_roc"] == pytest.approx(1.0, abs=1e-6)
        assert m["recall"]  == pytest.approx(1.0, abs=1e-6)
        assert m["precision"] == pytest.approx(1.0, abs=1e-6)

    def test_random_classifier_auc_near_half(self):
        rng = np.random.default_rng(99)
        y_true  = rng.integers(0, 2, 1000)
        y_proba = rng.uniform(0, 1, 1000)
        m = compute_metrics(y_true, y_proba)
        assert 0.3 < m["auc_roc"] < 0.7, "Random classifier AUC should be near 0.5"

    def test_explicit_threshold_respected(self):
        y_true  = np.array([0, 0, 1, 1])
        y_proba = np.array([0.4, 0.6, 0.4, 0.6])
        # With threshold=0.55, only indices 1 and 3 are flagged
        m = compute_metrics(y_true, y_proba, threshold=0.55)
        assert m["threshold"] == pytest.approx(0.55, abs=1e-6)

    def test_model_name_stored(self, binary_predictions):
        y_true, y_proba = binary_predictions
        m = compute_metrics(y_true, y_proba, model_name="xgboost")
        assert m["model"] == "xgboost"


# ---------------------------------------------------------------------------
# build_metrics_table
# ---------------------------------------------------------------------------

class TestBuildMetricsTable:
    def _make_metrics(self) -> list[dict]:
        return [
            {"model": "lr",  "auc_pr": 0.60, "auc_roc": 0.80, "f1": 0.55},
            {"model": "rf",  "auc_pr": 0.75, "auc_roc": 0.90, "f1": 0.70},
            {"model": "xgb", "auc_pr": 0.82, "auc_roc": 0.93, "f1": 0.78},
        ]

    def test_returns_dataframe(self):
        import pandas as pd
        df = build_metrics_table(self._make_metrics())
        assert isinstance(df, pd.DataFrame)

    def test_sorted_by_auc_pr_desc(self):
        df = build_metrics_table(self._make_metrics())
        auc_pr_vals = df["auc_pr"].tolist()
        assert auc_pr_vals == sorted(auc_pr_vals, reverse=True)

    def test_all_models_present(self):
        df = build_metrics_table(self._make_metrics())
        assert set(df["model"]) == {"lr", "rf", "xgb"}

    def test_empty_input_returns_empty_df(self):
        import pandas as pd
        df = build_metrics_table([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
