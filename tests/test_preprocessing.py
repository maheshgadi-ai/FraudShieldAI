"""
Unit tests for src/data/preprocessing.py

Tests the individual helper functions directly — the full run_preprocessing()
end-to-end is an integration test that requires files on disk, so we test
each sub-step in isolation using synthetic DataFrames from conftest.py.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import (
    _apply_smote,
    _drop_columns,
    _label_encode,
    _scale_features,
    _target_encode,
    _DROP_COLS,
    _LABEL_ENCODE_COLS,
    _TARGET_ENCODE_COLS,
)


# ---------------------------------------------------------------------------
# _drop_columns
# ---------------------------------------------------------------------------

class TestDropColumns:
    def test_identifier_columns_removed(self, raw_df):
        out = _drop_columns(raw_df)
        for col in _DROP_COLS:
            assert col not in out.columns, f"'{col}' should have been dropped"

    def test_target_column_preserved(self, raw_df):
        out = _drop_columns(raw_df)
        assert "is_fraud" in out.columns

    def test_amt_preserved(self, raw_df):
        out = _drop_columns(raw_df)
        assert "amt" in out.columns

    def test_missing_drop_cols_handled_gracefully(self):
        """Dropping columns that don't exist should not raise."""
        df = pd.DataFrame({"amt": [1.0, 2.0], "is_fraud": [0, 1]})
        out = _drop_columns(df)
        assert list(out.columns) == ["amt", "is_fraud"]


# ---------------------------------------------------------------------------
# _label_encode
# ---------------------------------------------------------------------------

class TestLabelEncode:
    def test_gender_becomes_integer(self, raw_df):
        train, test, encoders = _label_encode(raw_df.copy(), raw_df.copy())
        assert train["gender"].dtype in (np.int64, np.int32, int, np.intp)

    def test_encoder_stored(self, raw_df):
        _, _, encoders = _label_encode(raw_df.copy(), raw_df.copy())
        for col in _LABEL_ENCODE_COLS:
            if col in raw_df.columns:
                assert col in encoders

    def test_values_are_zero_or_one_for_binary(self, raw_df):
        train, _, _ = _label_encode(raw_df.copy(), raw_df.copy())
        if "gender" in train.columns:
            assert set(train["gender"].unique()).issubset({0, 1})

    def test_train_test_same_encoding(self, raw_df):
        """Same category in train and test should map to the same integer."""
        train, test, _ = _label_encode(raw_df.copy(), raw_df.copy())
        if "gender" in train.columns:
            # Train and test are identical here so values must match
            pd.testing.assert_series_equal(
                train["gender"].reset_index(drop=True),
                test["gender"].reset_index(drop=True),
            )


# ---------------------------------------------------------------------------
# _target_encode
# ---------------------------------------------------------------------------

class TestTargetEncode:
    def test_target_encode_cols_become_float(self, raw_df):
        train = raw_df.copy()
        test  = raw_df.copy()
        y     = train.pop("is_fraud")
        test.pop("is_fraud")
        train_out, test_out, _ = _target_encode(train, test, y_train=y)
        for col in _TARGET_ENCODE_COLS:
            if col in train_out.columns:
                assert train_out[col].dtype in (np.float32, np.float64)

    def test_no_nulls_after_encoding(self, raw_df):
        train = raw_df.copy()
        test  = raw_df.copy()
        y     = train.pop("is_fraud")
        test.pop("is_fraud")
        train_out, test_out, _ = _target_encode(train, test, y_train=y)
        for col in _TARGET_ENCODE_COLS:
            if col in train_out.columns:
                assert train_out[col].notna().all()
                assert test_out[col].notna().all()

    def test_unseen_category_gets_global_mean(self):
        train = pd.DataFrame({"category": ["A", "A", "B", "B"]})
        test  = pd.DataFrame({"category": ["C"]})   # unseen
        y     = pd.Series([0, 1, 0, 0], name="is_fraud")

        import src.data.preprocessing as prep
        old = prep._TARGET_ENCODE_COLS
        prep._TARGET_ENCODE_COLS = ["category"]
        _, test_out, _ = prep._target_encode(train, test, y_train=y)
        prep._TARGET_ENCODE_COLS = old

        global_mean = float(y.mean())
        assert test_out["category"].iloc[0] == pytest.approx(global_mean, abs=0.2)


# ---------------------------------------------------------------------------
# _scale_features
# ---------------------------------------------------------------------------

class TestScaleFeatures:
    def _make_frames(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        test  = pd.DataFrame({"a": [4.0, 5.0],       "b": [40.0, 50.0]})
        return train, test

    def test_standard_scaling_mean_zero(self):
        train, test, scaler = _scale_features(
            *self._make_frames(), "standard", ["a", "b"]
        )
        assert train["a"].mean() == pytest.approx(0.0, abs=1e-6)

    def test_scaler_is_returned(self):
        _, _, scaler = _scale_features(*self._make_frames(), "standard", ["a", "b"])
        assert isinstance(scaler, StandardScaler)

    def test_test_uses_train_stats(self):
        """Test set values should be scaled with train's mean/std, not their own."""
        train, test, scaler = _scale_features(
            *self._make_frames(), "standard", ["a", "b"]
        )
        # The test mean should NOT be zero (scaled with train stats)
        assert test["a"].mean() != pytest.approx(0.0, abs=1e-6)

    def test_minmax_range(self):
        train, _, _ = _scale_features(*self._make_frames(), "minmax", ["a", "b"])
        assert train["a"].min() == pytest.approx(0.0, abs=1e-6)
        assert train["a"].max() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _apply_smote
# ---------------------------------------------------------------------------

class TestApplySmote:
    def _imbalanced(self) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((100, 5)), columns=list("abcde"))
        y = pd.Series(np.where(rng.random(100) < 0.05, 1, 0), name="is_fraud")
        return X, y

    def test_minority_class_increases(self):
        X, y = self._imbalanced()
        n_fraud_before = int(y.sum())
        cfg = {"enabled": True, "sampling_strategy": 0.1, "random_state": 42}
        X_res, y_res = _apply_smote(X, y, cfg)
        assert int(y_res.sum()) > n_fraud_before

    def test_returns_dataframe_and_series(self):
        X, y = self._imbalanced()
        cfg = {"enabled": True, "sampling_strategy": 0.1, "random_state": 42}
        X_res, y_res = _apply_smote(X, y, cfg)
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)

    def test_disabled_smote_returns_unchanged(self):
        X, y = self._imbalanced()
        cfg = {"enabled": False}
        X_res, y_res = _apply_smote(X, y, cfg)
        assert len(X_res) == len(X)
        assert int(y_res.sum()) == int(y.sum())

    def test_column_names_preserved(self):
        X, y = self._imbalanced()
        cfg = {"enabled": True, "sampling_strategy": 0.1, "random_state": 42}
        X_res, _ = _apply_smote(X, y, cfg)
        assert list(X_res.columns) == list(X.columns)
