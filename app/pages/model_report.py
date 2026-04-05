"""
Model Report Page — FraudShieldAI Streamlit App

Displays:
  - Model comparison metrics table (sorted by AUC-PR)
  - Overlay ROC and PR curves (saved by evaluator)
  - Per-model confusion matrix, ROC, PR curve images
  - FNN/LSTM training history plots
  - SHAP summary plots
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

METRICS_DIR = Path("outputs/metrics")
PLOTS_DIR   = Path("outputs/plots")
SHAP_DIR    = Path("outputs/shap")

MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "fnn", "lstm", "ensemble"]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "fnn": "Neural Network (FNN)",
    "lstm": "LSTM",
    "ensemble": "Ensemble",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metrics() -> pd.DataFrame | None:
    path = METRICS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _show_image(path: Path, caption: str = "") -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.caption(f"*Plot not found: {path.name}*")


def _metrics_bar_chart(df: pd.DataFrame) -> go.Figure:
    cols = ["auc_pr", "auc_roc", "f1", "precision", "recall"]
    present = [c for c in cols if c in df.columns]
    melted = df.melt(id_vars="model", value_vars=present,
                     var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="model", y="Score", color="Metric", barmode="group",
        title="Model Comparison — Key Metrics",
        labels={"model": "Model", "Score": "Score"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=420,
        yaxis_range=[0, 1.05],
        xaxis_tickangle=-20,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _radar_chart(df: pd.DataFrame) -> go.Figure:
    metrics = ["auc_pr", "auc_roc", "f1", "precision", "recall"]
    present = [m for m in metrics if m in df.columns]
    fig = go.Figure()
    for _, row in df.iterrows():
        vals = [row[m] for m in present]
        vals += [vals[0]]   # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=present + [present[0]],
            fill="toself",
            name=DISPLAY_NAMES.get(row["model"], row["model"]),
            opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Radar Chart",
        height=420,
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("📈 Model Performance Report")

    metrics_df = _load_metrics()

    if metrics_df is None:
        st.warning(
            "No metrics found. Run the full pipeline first:\n\n"
            "```bash\npython main.py --stage all\n```"
        )
        return

    # ---------------------------------------------------------------- Table
    st.subheader("Metrics Comparison Table")
    st.caption("Sorted by AUC-PR (primary metric for imbalanced fraud detection)")

    fmt: dict = {}
    for col in ["auc_pr", "auc_roc", "f1", "precision", "recall", "mcc"]:
        if col in metrics_df.columns:
            fmt[col] = "{:.4f}"

    st.dataframe(
        metrics_df.style
            .background_gradient(subset=["auc_pr"], cmap="Greens")
            .background_gradient(subset=["f1"], cmap="Blues")
            .format(fmt),
        use_container_width=True,
    )

    # Download metrics
    csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download metrics CSV", csv_bytes,
                       "model_comparison.csv", "text/csv")

    # --------------------------------------------------------- Bar + Radar
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_metrics_bar_chart(metrics_df), use_container_width=True)
    with col2:
        st.plotly_chart(_radar_chart(metrics_df), use_container_width=True)

    # ------------------------------------------- Overlay ROC & PR curves
    st.markdown("---")
    st.subheader("All-Model Curve Overlays")
    ov1, ov2 = st.columns(2)
    with ov1:
        _show_image(PLOTS_DIR / "roc_all_models.png", "ROC Curves — All Models")
    with ov2:
        _show_image(PLOTS_DIR / "pr_all_models.png", "PR Curves — All Models")

    # ----------------------------------------- Per-model detail section
    st.markdown("---")
    st.subheader("Per-Model Detail")

    available_models = [
        m for m in MODEL_ORDER
        if (PLOTS_DIR / f"roc_{m}.png").exists()
        or (PLOTS_DIR / f"cm_{m}.png").exists()
    ]

    if not available_models:
        st.info("No individual model plots found. Run the evaluation stage.")
        return

    selected = st.selectbox(
        "Select model",
        options=available_models,
        format_func=lambda k: DISPLAY_NAMES.get(k, k),
    )

    # ROC / PR / CM in a row
    img1, img2, img3 = st.columns(3)
    with img1:
        _show_image(PLOTS_DIR / f"roc_{selected}.png", "ROC Curve")
    with img2:
        _show_image(PLOTS_DIR / f"pr_{selected}.png", "Precision-Recall Curve")
    with img3:
        _show_image(PLOTS_DIR / f"cm_{selected}.png", "Confusion Matrix")

    # Training history (FNN / LSTM)
    hist_path = PLOTS_DIR / f"history_{selected}.png"
    if hist_path.exists():
        st.markdown("**Training History**")
        _show_image(hist_path)

    # ---------------------------------------------------- SHAP plots
    shap_bar  = SHAP_DIR / f"shap_bar_{selected}.png"
    shap_summ = SHAP_DIR / f"shap_summary_{selected}.png"

    if shap_bar.exists() or shap_summ.exists():
        st.markdown("---")
        st.subheader(f"SHAP Explanations — {DISPLAY_NAMES.get(selected, selected)}")
        s1, s2 = st.columns(2)
        with s1:
            _show_image(shap_bar, "Feature Importance (SHAP bar)")
        with s2:
            _show_image(shap_summ, "SHAP Summary (beeswarm)")
    else:
        st.info("SHAP plots not generated yet for this model.")

    # ------------------------------------------ Inline metrics for selection
    if "model" in metrics_df.columns:
        row = metrics_df[metrics_df["model"] == selected]
        if not row.empty:
            st.markdown("---")
            st.subheader(f"Metrics — {DISPLAY_NAMES.get(selected, selected)}")
            row = row.iloc[0]
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("AUC-PR",    f"{row.get('auc_pr', '—'):.4f}")
            m2.metric("AUC-ROC",   f"{row.get('auc_roc', '—'):.4f}")
            m3.metric("F1",        f"{row.get('f1', '—'):.4f}")
            m4.metric("Precision", f"{row.get('precision', '—'):.4f}")
            m5.metric("Recall",    f"{row.get('recall', '—'):.4f}")
            m6.metric("MCC",       f"{row.get('mcc', '—'):.4f}")
