"""
Shared utilities for the FraudShieldAI Streamlit app.

Handles:
  - Loading trained models from outputs/models/
  - Running inference on uploaded DataFrames
  - Applying the saved preprocessing artifacts (encoders, scaler)
    so the app can accept raw transaction CSVs identical to the Kaggle format
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs/models")
FEATURE_COLS: list[str] | None = None   # populated on first load


# ---------------------------------------------------------------------------
# Model loading (cached so Streamlit doesn't reload on every interaction)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models…")
def load_all_models() -> dict[str, Any]:
    """
    Load every saved model artifact from outputs/models/.
    Returns a dict: {model_name: fitted_model}.
    """
    models: dict[str, Any] = {}

    sklearn_names = ["logistic_regression", "random_forest", "xgboost"]
    for name in sklearn_names:
        path = OUTPUTS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)

    fnn_path = OUTPUTS_DIR / "fnn.pt"
    if fnn_path.exists():
        from src.models.fnn import FNNClassifier
        models["fnn"] = FNNClassifier.load(fnn_path, cfg={})

    lstm_path = OUTPUTS_DIR / "lstm.pt"
    if lstm_path.exists():
        from src.models.lstm import LSTMClassifier
        models["lstm"] = LSTMClassifier.load(lstm_path, cfg={})

    ensemble_path = OUTPUTS_DIR / "ensemble.joblib"
    if ensemble_path.exists():
        models["ensemble"] = joblib.load(ensemble_path)

    return models


@st.cache_resource(show_spinner="Loading preprocessing artifacts…")
def load_preprocessing_artifacts() -> dict[str, Any]:
    """Load scaler + encoders saved during preprocessing."""
    artifacts: dict[str, Any] = {}
    for name in ("scaler", "label_encoders", "target_encoders", "numeric_cols"):
        path = OUTPUTS_DIR / f"{name}.joblib"
        if path.exists():
            artifacts[name] = joblib.load(path)
    return artifacts


# ---------------------------------------------------------------------------
# Preprocessing raw uploads to match the trained feature space
# ---------------------------------------------------------------------------

def preprocess_upload(df: pd.DataFrame, artifacts: dict[str, Any]) -> np.ndarray:
    """
    Apply feature engineering + encoding + scaling to a freshly uploaded
    CSV so it matches what the models were trained on.

    This mirrors src/data/preprocessing.py but uses the already-fitted
    transformers (no re-fitting).
    """
    from src.data.features import build_features

    # Feature engineering — use default flags (all enabled)
    feat_cfg = {
        "geo_distance": True,
        "time_features": True,
        "velocity_windows": [1, 7, 30],
        "spending_features": True,
    }
    df = build_features(df, feat_cfg)

    # Drop identifier / coordinate columns
    DROP = [
        "trans_num", "first", "last", "street", "city", "state", "zip",
        "dob", "unix_time", "trans_date_trans_time",
        "lat", "long", "merch_lat", "merch_long",
        "is_fraud",   # may or may not be present in uploaded file
    ]
    df = df.drop(columns=[c for c in DROP if c in df.columns])

    # Label encode (gender etc.)
    label_encoders: dict = artifacts.get("label_encoders", {})
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen categories gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # Target encode (merchant, category, job)
    target_encoders: dict = artifacts.get("target_encoders", {})
    for col, mapping in target_encoders.items():
        if col in df.columns:
            global_mean = mapping.mean()
            df[col] = df[col].map(mapping).fillna(global_mean).astype(np.float32)

    # Hash cc_num
    if "cc_num" in df.columns:
        df["cc_num"] = df["cc_num"].astype(str).apply(
            lambda x: int(hash(x) % (2 ** 31))
        ).astype(np.int32)

    df = df.fillna(0)

    # Scale
    scaler = artifacts.get("scaler")
    numeric_cols: list[str] = artifacts.get("numeric_cols", [])
    present_num = [c for c in numeric_cols if c in df.columns]
    if scaler is not None and present_num:
        df[present_num] = scaler.transform(df[present_num])

    return df.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    X: np.ndarray,
    models: dict[str, Any],
    model_name: str,
    card_indices: np.ndarray | None = None,
    seq_len: int = 10,
) -> np.ndarray:
    """
    Run a single model on X and return fraud probability for each row.

    For the LSTM, sequences are built on the fly from X + card_indices.
    For the ensemble, a dict of inputs is assembled automatically.

    Returns:
        proba: np.ndarray of shape (n_samples,) — fraud probability
    """
    model = models[model_name]

    if model_name == "lstm":
        from src.models.lstm import build_sequences
        if card_indices is None:
            card_indices = np.zeros(len(X), dtype=np.int32)
        X_seq = build_sequences(X, card_indices, seq_len=seq_len)
        return model.predict_proba(X_seq)[:, 1]

    if model_name == "ensemble":
        inputs: dict[str, np.ndarray] = {}
        for name in models:
            if name in ("ensemble",):
                continue
            if name == "lstm":
                from src.models.lstm import build_sequences
                ci = card_indices if card_indices is not None else np.zeros(len(X), dtype=np.int32)
                inputs[name] = build_sequences(X, ci, seq_len=seq_len)
            else:
                inputs[name] = X
        return model.predict_proba(inputs)[:, 1]

    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# SHAP (single-row explanation for the Predict page)
# ---------------------------------------------------------------------------

def shap_explain_row(
    model_name: str,
    model: Any,
    X_background: np.ndarray,
    X_row: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]] | None:
    """
    Compute SHAP values for a single transaction row.

    Returns (shap_values_1d, feature_names) or None on failure.
    """
    try:
        import shap
    except ImportError:
        return None

    try:
        n_bg = min(200, len(X_background))
        bg = X_background[np.random.choice(len(X_background), n_bg, replace=False)]

        is_deep = hasattr(model, "model")
        if is_deep:
            import torch
            device = model.device
            bg_t = torch.tensor(bg, dtype=torch.float32).to(device)
            row_t = torch.tensor(X_row[np.newaxis], dtype=torch.float32).to(device)
            explainer = shap.GradientExplainer(model.model, bg_t)
            sv = explainer.shap_values(row_t)
            if isinstance(sv, list):
                sv = sv[0]
            return sv[0].flatten(), feature_names
        else:
            explainer = shap.TreeExplainer(model) if hasattr(model, "estimators_") or \
                        hasattr(model, "get_booster") else shap.LinearExplainer(model, bg)
            sv = explainer.shap_values(X_row[np.newaxis])
            if isinstance(sv, list):
                sv = sv[1]
            return sv[0].flatten(), feature_names
    except Exception as exc:
        logger.warning("SHAP explain failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "fnn": "Neural Network (FNN)",
    "lstm": "LSTM",
    "ensemble": "Ensemble",
}

RISK_COLORS = {
    "LOW": "#2ecc71",
    "MEDIUM": "#f39c12",
    "HIGH": "#e74c3c",
}

def risk_label(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    if prob < 0.7:
        return "MEDIUM"
    return "HIGH"
