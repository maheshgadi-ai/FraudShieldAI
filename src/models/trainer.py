"""
Training Orchestrator for FraudShieldAI.

Loads processed data, trains all enabled models (supervised + deep learning
+ ensemble), saves artifacts, and logs a summary of training outcomes.

Called via:
    python main.py --stage train
or:
    from src.models.trainer import run_training
    run_training(pipeline_cfg, model_cfg)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.ensemble import build_ensemble
from src.models.fnn import train_fnn
from src.models.lstm import build_sequences, train_lstm
from src.models.supervised import train_supervised_models
from src.utils.helpers import Timer, save_artifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_processed(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = cfg["data"]["target_column"]

    train_df = pd.read_parquet(cfg["data"]["processed_train_path"])
    test_df = pd.read_parquet(cfg["data"]["processed_test_path"])

    y_train = train_df.pop(target).values.astype(np.float32)
    y_test = test_df.pop(target).values.astype(np.float32)
    X_train = train_df.values.astype(np.float32)
    X_test = test_df.values.astype(np.float32)

    logger.info(
        "Loaded — X_train: %s | X_test: %s | Fraud rate train: %.3f%% | test: %.3f%%",
        X_train.shape, X_test.shape,
        y_train.mean() * 100, y_test.mean() * 100,
    )
    return X_train, y_train, X_test, y_test


def _get_card_indices(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Load hashed cc_num column from processed data for LSTM sequence building.
    Returns (card_indices_train, card_indices_test).
    """
    target = cfg["data"]["target_column"]
    train_df = pd.read_parquet(cfg["data"]["processed_train_path"])
    test_df = pd.read_parquet(cfg["data"]["processed_test_path"])

    col = "cc_num" if "cc_num" in train_df.columns else None
    if col is None:
        n_tr, n_te = len(train_df), len(test_df)
        return np.zeros(n_tr, dtype=np.int32), np.zeros(n_te, dtype=np.int32)

    return train_df[col].values.astype(np.int32), test_df[col].values.astype(np.int32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_training(pipeline_cfg: dict, model_cfg: dict) -> None:
    """
    Full training run:
      1. Load processed train / test splits
      2. Train supervised models (LR, RF, XGBoost) with Optuna tuning
      3. Train FNN
      4. Build LSTM sequences and train LSTM
      5. Build and save ensemble
      6. Persist all artifacts

    Args:
        pipeline_cfg: Contents of pipeline.yaml
        model_cfg:    Contents of models.yaml
    """
    out_dir = Path(pipeline_cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Load
    with Timer("load processed data"):
        X_train, y_train, X_test, y_test = _load_processed(pipeline_cfg)

    fitted_models: dict = {}
    # Also track test inputs for ensemble evaluation (keyed by model name)
    test_inputs: dict[str, np.ndarray] = {}

    # ------------------------------------------------- Supervised models
    with Timer("supervised model training"):
        sup_results = train_supervised_models(X_train, y_train, model_cfg)

    for name, result in sup_results.items():
        fitted_models[name] = result["model"]
        test_inputs[name] = X_test
        save_artifact(result["model"], out_dir / f"{name}.joblib")
        if result["study"] is not None:
            save_artifact(result["study"], out_dir / f"{name}_optuna_study.joblib")

    # ------------------------------------------------------------------ FNN
    if model_cfg.get("fnn", {}).get("enabled", True):
        with Timer("FNN training"):
            fnn_clf = train_fnn(X_train, y_train, model_cfg["fnn"])
        fnn_clf.save(out_dir / "fnn.pt")
        fitted_models["fnn"] = fnn_clf
        test_inputs["fnn"] = X_test

    # ---------------------------------------------------------------- LSTM
    if model_cfg.get("lstm", {}).get("enabled", True):
        seq_len = model_cfg["lstm"].get("sequence_length", 10)

        logger.info("[Trainer] Building LSTM sequences (seq_len=%d)…", seq_len)
        card_tr, card_te = _get_card_indices(pipeline_cfg)

        with Timer("build train sequences"):
            X_seq_train = build_sequences(X_train, card_tr, seq_len=seq_len)
        with Timer("build test sequences"):
            X_seq_test = build_sequences(X_test, card_te, seq_len=seq_len)

        with Timer("LSTM training"):
            lstm_clf = train_lstm(X_seq_train, y_train, model_cfg["lstm"])
        lstm_clf.save(out_dir / "lstm.pt")
        fitted_models["lstm"] = lstm_clf
        test_inputs["lstm"] = X_seq_test

        # Persist sequences for evaluation stage
        save_artifact(X_seq_test, out_dir / "X_seq_test.joblib")
        save_artifact(card_tr, out_dir / "card_indices_train.joblib")
        save_artifact(card_te, out_dir / "card_indices_test.joblib")

    # --------------------------------------------------------------- Ensemble
    if model_cfg.get("ensemble", {}).get("enabled", True):
        with Timer("build ensemble"):
            ensemble = build_ensemble(fitted_models, model_cfg["ensemble"])
        save_artifact(ensemble, out_dir / "ensemble.joblib")
        fitted_models["ensemble"] = ensemble
        test_inputs["ensemble"] = test_inputs  # ensemble expects the full dict

    # ---------------------------------------- Persist test labels + inputs
    save_artifact(y_test, out_dir / "y_test.joblib")
    save_artifact(X_test, out_dir / "X_test.joblib")
    save_artifact(list(fitted_models.keys()), out_dir / "model_names.joblib")

    logger.info("=== Training complete. Models saved to %s ===", out_dir)
    logger.info("Trained models: %s", list(fitted_models.keys()))
