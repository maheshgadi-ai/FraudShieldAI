"""
Shared pytest fixtures for FraudShieldAI tests.

Provides minimal synthetic DataFrames and arrays that mirror the
Kaggle dataset schema so tests run without the real data on disk.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic raw transaction DataFrame (Kaggle schema)
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """
    50-row synthetic transaction DataFrame with all Kaggle columns.
    Contains 5 fraudulent rows (is_fraud=1) to exercise imbalance handling.
    """
    rng = np.random.default_rng(42)
    n = 50

    base_dt = pd.Timestamp("2020-01-01")
    timestamps = [base_dt + pd.Timedelta(hours=int(h)) for h in rng.integers(0, 720, n)]

    df = pd.DataFrame({
        "trans_date_trans_time": timestamps,
        "cc_num": rng.integers(4000_0000_0000_0000, 4999_9999_9999_9999, n).astype(str),
        "merchant":  rng.choice(["merchant_A", "merchant_B", "merchant_C"], n),
        "category":  rng.choice(["grocery_pos", "gas_transport", "shopping_net"], n),
        "amt":       rng.uniform(1.0, 500.0, n).round(2),
        "first":     ["John"] * n,
        "last":      ["Doe"] * n,
        "gender":    rng.choice(["M", "F"], n),
        "street":    ["123 Main St"] * n,
        "city":      ["Springfield"] * n,
        "state":     ["IL"] * n,
        "zip":       ["62701"] * n,
        "lat":       rng.uniform(25.0, 50.0, n),
        "long":      rng.uniform(-125.0, -65.0, n),
        "city_pop":  rng.integers(1000, 500_000, n),
        "job":       rng.choice(["Engineer", "Teacher", "Doctor"], n),
        "dob":       pd.date_range("1960-01-01", periods=n, freq="180D").strftime("%Y-%m-%d"),
        "trans_num": [f"txn_{i:04d}" for i in range(n)],
        "unix_time": [int(t.timestamp()) for t in timestamps],
        "merch_lat": rng.uniform(25.0, 50.0, n),
        "merch_long": rng.uniform(-125.0, -65.0, n),
        "is_fraud":  np.where(rng.random(n) < 0.10, 1, 0),
    })
    return df


# ---------------------------------------------------------------------------
# Minimal feature matrix and labels
# ---------------------------------------------------------------------------

@pytest.fixture()
def X_y() -> tuple[np.ndarray, np.ndarray]:
    """Simple 200×10 numeric array with ~10% positive class."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    y = (rng.random(200) < 0.10).astype(np.float32)
    return X, y


@pytest.fixture()
def binary_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Fixed y_true / y_proba pair for deterministic metric tests."""
    rng = np.random.default_rng(7)
    y_true = np.array([0] * 90 + [1] * 10, dtype=int)
    # Fraud class gets higher scores on average
    y_proba = np.concatenate([
        rng.uniform(0.0, 0.4, 90),
        rng.uniform(0.5, 1.0, 10),
    ])
    return y_true, y_proba
