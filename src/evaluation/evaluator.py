"""
Evaluation pipeline for FraudShieldAI.

For each trained model:
  1. Compute all metrics (metrics.py)
  2. Plot ROC curve, PR curve, confusion matrix, training history (DL models)
  3. Run SHAP explanation (tree-based models → TreeExplainer, neural → DeepExplainer)
  4. Save a model comparison table as CSV + markdown

Outputs land in the directories defined in pipeline.yaml:
  outputs/metrics/   — CSV / markdown comparison table
  outputs/plots/     — all matplotlib figures (PNG)
  outputs/shap/      — SHAP summary plots + values (NPY)
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
    roc_curve,
)

from src.evaluation.metrics import compute_metrics, build_metrics_table
from src.utils.helpers import load_artifact

logger = logging.getLogger(__name__)

# Seaborn style for all plots
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved plot → %s", path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    plots_dir: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = np.trapz(tpr, fpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC-ROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    _save_fig(fig, plots_dir / f"roc_{model_name}.png")


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    plots_dir: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = np.trapz(precision[::-1], recall[::-1])
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"AUC-PR = {auc_pr:.4f}")
    ax.axhline(baseline, color="grey", linestyle="--", lw=1,
               label=f"Baseline (fraud rate = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    _save_fig(fig, plots_dir / f"pr_{model_name}.png")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Legit", "Fraud"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    _save_fig(fig, plots_dir / f"cm_{model_name}.png")


def plot_training_history(
    history: dict[str, list[float]],
    model_name: str,
    plots_dir: Path,
) -> None:
    """Plot train loss and val AUC-PR curves for FNN/LSTM."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, history["train_loss"], lw=2)
    axes[0].set_title(f"{model_name} — Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")

    axes[1].plot(epochs, history["val_auc_pr"], lw=2, color="darkorange")
    axes[1].set_title(f"{model_name} — Validation AUC-PR")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-PR")

    fig.tight_layout()
    _save_fig(fig, plots_dir / f"history_{model_name}.png")


def plot_model_comparison(metrics_df: pd.DataFrame, plots_dir: Path) -> None:
    """Bar chart comparing key metrics across all models."""
    metric_cols = ["auc_pr", "auc_roc", "f1", "recall", "precision"]
    present = [c for c in metric_cols if c in metrics_df.columns]

    melted = metrics_df.melt(
        id_vars="model", value_vars=present, var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Model Comparison — Key Metrics")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save_fig(fig, plots_dir / "model_comparison.png")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def _shap_tree(model, X: np.ndarray, model_name: str, shap_dir: Path) -> None:
    """SHAP TreeExplainer for LR (linear), RF, XGBoost."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP for %s", model_name)
        return

    logger.info("[SHAP] Running TreeExplainer for %s…", model_name)
    # Cap sample size to keep SHAP runtime reasonable
    n_sample = min(2000, len(X))
    idx = np.random.choice(len(X), n_sample, replace=False)
    X_sample = X[idx]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For binary classifiers shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        logger.info("[SHAP] TreeExplainer failed for %s, trying LinearExplainer…", model_name)
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {model_name}")
    _save_fig(plt.gcf(), shap_dir / f"shap_bar_{model_name}.png")

    # Summary beeswarm plot
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Summary — {model_name}")
    _save_fig(plt.gcf(), shap_dir / f"shap_summary_{model_name}.png")

    # Persist raw SHAP values
    np.save(shap_dir / f"shap_values_{model_name}.npy", shap_values)
    logger.info("[SHAP] Saved SHAP artifacts for %s", model_name)


def _shap_deep(model, X: np.ndarray, model_name: str, shap_dir: Path) -> None:
    """SHAP GradientExplainer for PyTorch FNN / LSTM."""
    try:
        import shap
        import torch
    except ImportError:
        logger.warning("shap/torch not installed — skipping SHAP for %s", model_name)
        return

    logger.info("[SHAP] Running GradientExplainer for %s…", model_name)
    n_background = min(500, len(X))
    n_explain = min(200, len(X))
    bg_idx = np.random.choice(len(X), n_background, replace=False)
    ex_idx = np.random.choice(len(X), n_explain, replace=False)

    nn_model = model.model
    nn_model.eval()
    device = model.device

    # GradientExplainer needs the model to return 2-D output (batch, 1).
    # FNN/LSTM squeeze to 1-D, so wrap them.
    class _Unsqueeze(torch.nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x): return self.base(x).unsqueeze(1)

    wrapped = _Unsqueeze(nn_model).to(device)
    wrapped.eval()

    # GradientExplainer requires 2-D or 3-D tensors
    X_bg = torch.tensor(X[bg_idx], dtype=torch.float32).to(device)
    X_ex = torch.tensor(X[ex_idx], dtype=torch.float32).to(device)

    try:
        explainer = shap.GradientExplainer(wrapped, X_bg)
        shap_values = explainer.shap_values(X_ex)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        # shap_values shape: (n_explain, seq_len_or_features, 1) → squeeze last dim
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values.squeeze(-1)

        # For LSTM sequences (3-D), average across the sequence dimension
        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=1)
            X_display = X[ex_idx].mean(axis=1)
        else:
            X_display = X[ex_idx]

        shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance — {model_name}")
        _save_fig(plt.gcf(), shap_dir / f"shap_bar_{model_name}.png")

        np.save(shap_dir / f"shap_values_{model_name}.npy", shap_values)
        logger.info("[SHAP] Saved SHAP artifacts for %s", model_name)
    except Exception as exc:
        logger.warning("[SHAP] GradientExplainer failed for %s: %s", model_name, exc)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    model: Any,
    X_input: np.ndarray,
    y_true: np.ndarray,
    plots_dir: Path,
    shap_dir: Path,
    run_shap: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a single model:
      - compute metrics
      - save ROC, PR, CM plots
      - run SHAP (if enabled)
      - plot training history for DL models

    Returns a metrics dict.
    """
    logger.info("Evaluating %s…", model_name)

    # --- Get probabilities ---
    y_proba = model.predict_proba(X_input)[:, 1]

    # --- Metrics ---
    metrics = compute_metrics(y_true, y_proba, model_name=model_name)
    threshold = metrics["threshold"]
    y_pred = (y_proba >= threshold).astype(int)

    # --- Standard plots ---
    plot_roc_curve(y_true, y_proba, model_name, plots_dir)
    plot_pr_curve(y_true, y_proba, model_name, plots_dir)
    plot_confusion_matrix(y_true, y_pred, model_name, plots_dir)

    # Training history (FNN / LSTM)
    if hasattr(model, "history") and model.history.get("train_loss"):
        plot_training_history(model.history, model_name, plots_dir)

    # --- SHAP ---
    if run_shap and model_name != "ensemble":
        is_deep = hasattr(model, "model")  # FNNClassifier / LSTMClassifier
        if is_deep:
            _shap_deep(model, X_input, model_name, shap_dir)
        else:
            _shap_tree(model, X_input, model_name, shap_dir)

    return metrics


# ---------------------------------------------------------------------------
# All-models overlay plots
# ---------------------------------------------------------------------------

def plot_roc_overlay(
    model_results: dict[str, tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
    plots_dir: Path,
) -> None:
    """Overlay ROC curves for all models on a single figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_proba in model_results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, plots_dir / "roc_all_models.png")


def plot_pr_overlay(
    model_results: dict[str, np.ndarray],
    y_true: np.ndarray,
    plots_dir: Path,
) -> None:
    """Overlay PR curves for all models on a single figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_true.mean()
    ax.axhline(baseline, color="grey", linestyle="--", lw=1,
               label=f"Baseline ({baseline:.3f})")
    for name, y_proba in model_results.items():
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        auc_pr = np.trapz(prec[::-1], rec[::-1])
        ax.plot(rec, prec, lw=2, label=f"{name} (AP={auc_pr:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, plots_dir / "pr_all_models.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_evaluation(pipeline_cfg: dict, model_cfg: dict) -> None:
    """
    Load all trained models and test data, evaluate each one,
    produce plots + SHAP outputs, and save a metrics comparison table.

    Args:
        pipeline_cfg: Contents of pipeline.yaml
        model_cfg:    Contents of models.yaml
    """
    out_dir   = Path(pipeline_cfg["training"]["output_dir"])
    metrics_dir = Path(pipeline_cfg["training"]["metrics_dir"])
    plots_dir   = Path(pipeline_cfg["training"]["plots_dir"])
    shap_dir    = Path(pipeline_cfg["training"]["shap_dir"])

    for d in (metrics_dir, plots_dir, shap_dir):
        d.mkdir(parents=True, exist_ok=True)

    target = pipeline_cfg["data"]["target_column"]

    # --- Load test labels ---
    y_test = load_artifact(out_dir / "y_test.joblib")
    X_test = load_artifact(out_dir / "X_test.joblib")

    # --- Load LSTM sequences if they exist ---
    seq_path = out_dir / "X_seq_test.joblib"
    X_seq_test = load_artifact(seq_path) if seq_path.exists() else None

    model_names: list[str] = load_artifact(out_dir / "model_names.joblib")

    all_metrics: list[dict] = []
    probas_for_overlay: dict[str, np.ndarray] = {}

    for name in model_names:
        if name == "ensemble":
            continue  # handled separately below

        # --- Load model ---
        if name == "fnn":
            from src.models.fnn import FNNClassifier
            model = FNNClassifier.load(out_dir / "fnn.pt", model_cfg.get("fnn", {}))
            X_input = X_test
        elif name == "lstm":
            from src.models.lstm import LSTMClassifier
            model = LSTMClassifier.load(out_dir / "lstm.pt", model_cfg.get("lstm", {}))
            X_input = X_seq_test
        else:
            model = load_artifact(out_dir / f"{name}.joblib")
            X_input = X_test

        metrics = evaluate_model(
            model_name=name,
            model=model,
            X_input=X_input,
            y_true=y_test,
            plots_dir=plots_dir,
            shap_dir=shap_dir,
        )
        all_metrics.append(metrics)
        probas_for_overlay[name] = model.predict_proba(X_input)[:, 1]

    # --- Ensemble ---
    ensemble_path = out_dir / "ensemble.joblib"
    if ensemble_path.exists():
        ensemble = load_artifact(ensemble_path)
        # Build the inputs dict the ensemble expects
        ensemble_inputs: dict[str, np.ndarray] = {}
        for n in (model_names if model_names else []):
            if n == "ensemble":
                continue
            ensemble_inputs[n] = X_seq_test if n == "lstm" else X_test

        y_proba_ens = ensemble.predict_proba(ensemble_inputs)[:, 1]
        metrics_ens = compute_metrics(y_test, y_proba_ens, model_name="ensemble")
        all_metrics.append(metrics_ens)
        probas_for_overlay["ensemble"] = y_proba_ens

    # --- Overlay plots ---
    plot_roc_overlay(probas_for_overlay, y_test, plots_dir)
    plot_pr_overlay(probas_for_overlay, y_test, plots_dir)

    # --- Comparison table ---
    metrics_df = build_metrics_table(all_metrics)
    csv_path = metrics_dir / "model_comparison.csv"
    md_path  = metrics_dir / "model_comparison.md"
    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_markdown(md_path, index=False)
    logger.info("Saved metrics table → %s", csv_path)

    # --- Model comparison bar chart ---
    plot_model_comparison(metrics_df, plots_dir)

    # --- Print table to stdout ---
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by AUC-PR)")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60 + "\n")

    logger.info("Evaluation complete. Outputs in: %s  %s  %s",
                metrics_dir, plots_dir, shap_dir)
