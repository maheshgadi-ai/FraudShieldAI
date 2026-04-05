"""
Predict Page — FraudShieldAI Streamlit App

Features:
  - Upload a CSV of raw transactions (same format as Kaggle dataset)
  - Choose which model to run
  - See a colour-coded results table (LOW / MEDIUM / HIGH risk)
  - Drill into any row for a SHAP waterfall explanation
  - Download results as CSV
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    MODEL_DISPLAY_NAMES,
    RISK_COLORS,
    load_all_models,
    load_preprocessing_artifacts,
    preprocess_upload,
    risk_label,
    run_inference,
    shap_explain_row,
)

OUTPUTS_DIR = Path("outputs/models")


# ---------------------------------------------------------------------------
# SHAP waterfall chart (Plotly — no matplotlib dependency in the UI)
# ---------------------------------------------------------------------------

def _shap_waterfall(shap_vals: np.ndarray, feature_names: list[str], base_value: float) -> go.Figure:
    """Horizontal waterfall chart showing each feature's SHAP contribution."""
    # Sort by absolute value, show top 15
    order = np.argsort(np.abs(shap_vals))[::-1][:15]
    sv = shap_vals[order]
    fn = [feature_names[i] for i in order]

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in sv]

    fig = go.Figure(go.Bar(
        x=sv,
        y=fn,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in sv],
        textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Feature Contributions (red = increases fraud risk)",
        xaxis_title="SHAP value",
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("🔍 Transaction Fraud Predictor")
    st.markdown(
        "Upload a CSV of transactions in the **Kaggle Credit Card Fraud** format. "
        "The pipeline applies feature engineering and your chosen model in real-time."
    )

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.markdown("### Prediction Settings")
        models = load_all_models()

        if not models:
            st.error("No trained models found in `outputs/models/`. Run `python main.py --stage train` first.")
            st.stop()

        model_choice = st.selectbox(
            "Model",
            options=list(models.keys()),
            format_func=lambda k: MODEL_DISPLAY_NAMES.get(k, k),
        )
        threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help="Transactions with fraud probability ≥ threshold are flagged as FRAUD.",
        )
        show_shap = st.checkbox("Enable SHAP explanations", value=True)

    # --------------------------------------------------------------- upload
    uploaded = st.file_uploader(
        "Upload transactions CSV",
        type=["csv"],
        help="Must follow the Kaggle fraudTrain/fraudTest column schema.",
    )

    if uploaded is None:
        st.info("Upload a CSV file to get started. You can use a sample from `data/raw/`.")
        _show_expected_columns()
        return

    # ----------------------------------------------------------------- load
    with st.spinner("Reading file…"):
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            return

    st.success(f"Loaded **{len(raw_df):,}** transactions.")

    # Store true labels if present (for accuracy reporting)
    true_labels: pd.Series | None = None
    if "is_fraud" in raw_df.columns:
        true_labels = raw_df["is_fraud"].copy()

    # ----------------------------------------------------------- preprocess
    artifacts = load_preprocessing_artifacts()
    with st.spinner("Running feature engineering + preprocessing…"):
        try:
            X = preprocess_upload(raw_df.copy(), artifacts)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.exception(e)
            return

    # ------------------------------------------------------------ inference
    with st.spinner(f"Running {MODEL_DISPLAY_NAMES.get(model_choice, model_choice)}…"):
        try:
            proba = run_inference(X, models, model_choice)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.exception(e)
            return

    # ----------------------------------------------------------- build results
    pred_labels = (proba >= threshold).astype(int)
    risk_labels = [risk_label(p) for p in proba]

    results_df = raw_df.copy()
    results_df["fraud_probability"] = np.round(proba, 4)
    results_df["prediction"] = pred_labels
    results_df["risk_level"] = risk_labels
    if true_labels is not None:
        results_df["true_label"] = true_labels.values

    # ----------------------------------------------------------------- KPIs
    n_flagged = int(pred_labels.sum())
    fraud_rate = proba.mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(raw_df):,}")
    col2.metric("Flagged as Fraud", f"{n_flagged:,}",
                delta=f"{n_flagged / len(raw_df) * 100:.2f}%")
    col3.metric("Avg Fraud Probability", f"{fraud_rate:.3f}")
    if true_labels is not None:
        from sklearn.metrics import f1_score, roc_auc_score
        try:
            auc = roc_auc_score(true_labels, proba)
            f1  = f1_score(true_labels, pred_labels, zero_division=0)
            col4.metric("AUC-ROC", f"{auc:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")
        except Exception:
            pass

    st.markdown("---")

    # -------------------------------------------------------- probability histogram
    st.subheader("Fraud Probability Distribution")
    fig_hist = px.histogram(
        results_df,
        x="fraud_probability",
        nbins=50,
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_level": ["LOW", "MEDIUM", "HIGH"]},
        labels={"fraud_probability": "Fraud Probability", "risk_level": "Risk Level"},
        title="Distribution of Predicted Fraud Probabilities",
    )
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black",
                       annotation_text=f"Threshold ({threshold})")
    fig_hist.update_layout(bargap=0.05, height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------------------------------------- results table
    st.subheader("Transaction Results")

    # Colour-code risk level in the display
    display_cols = ["amt", "merchant", "category", "fraud_probability",
                    "risk_level", "prediction"]
    if true_labels is not None:
        display_cols.append("true_label")
    display_cols = [c for c in display_cols if c in results_df.columns]

    show_only_fraud = st.checkbox("Show flagged transactions only", value=False)
    display_df = results_df[results_df["prediction"] == 1] if show_only_fraud else results_df

    st.dataframe(
        display_df[display_cols].style
            .background_gradient(subset=["fraud_probability"], cmap="RdYlGn_r")
            .format({"fraud_probability": "{:.4f}"}),
        use_container_width=True,
        height=380,
    )

    # -------------------------------------------------------- download
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv_bytes,
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )

    # -------------------------------------------------------- SHAP drill-down
    if show_shap:
        st.markdown("---")
        st.subheader("Transaction SHAP Explanation")
        st.caption(
            "Select a row index to see which features pushed the model toward "
            "or away from flagging that transaction as fraud."
        )

        row_idx = st.number_input(
            "Row index (0-based)",
            min_value=0,
            max_value=len(X) - 1,
            value=0,
            step=1,
        )

        selected_prob = float(proba[row_idx])
        selected_risk = risk_label(selected_prob)
        badge_html = f'<span class="badge-{selected_risk}">{selected_risk}</span>'

        col_a, col_b = st.columns(2)
        col_a.metric("Fraud Probability", f"{selected_prob:.4f}")
        col_b.markdown(f"**Risk Level:** {badge_html}", unsafe_allow_html=True)

        # Attempt to load background data for SHAP context
        bg_path = OUTPUTS_DIR / "X_test.joblib"
        X_background: np.ndarray | None = None
        if bg_path.exists():
            import joblib
            X_background = joblib.load(bg_path)

        if X_background is not None and model_choice not in ("ensemble", "lstm"):
            with st.spinner("Computing SHAP values…"):
                try:
                    # Recover feature names from processed parquet if available
                    processed_path = Path("data/processed/test.parquet")
                    if processed_path.exists():
                        feat_names = [
                            c for c in pd.read_parquet(processed_path).columns
                            if c != "is_fraud"
                        ]
                    else:
                        feat_names = [f"feature_{i}" for i in range(X.shape[1])]

                    result = shap_explain_row(
                        model_choice,
                        models[model_choice],
                        X_background,
                        X[row_idx],
                        feat_names,
                    )
                    if result is not None:
                        sv, fn = result
                        base_val = float(selected_prob - sv.sum())
                        fig_shap = _shap_waterfall(sv, fn, base_val)
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        st.info("SHAP explanation unavailable for this model/row.")
                except Exception as exc:
                    st.warning(f"SHAP failed: {exc}")
        elif model_choice in ("ensemble", "lstm"):
            st.info("SHAP drill-down is available for LR, RF, XGBoost, and FNN models.")
        else:
            st.info("Background data not found. Run `python main.py --stage train` to generate it.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _show_expected_columns() -> None:
    with st.expander("Expected CSV columns"):
        st.markdown("""
| Column | Description |
|---|---|
| `trans_date_trans_time` | Transaction datetime |
| `cc_num` | Credit card number |
| `merchant` | Merchant name |
| `category` | Merchant category |
| `amt` | Transaction amount |
| `first`, `last` | Cardholder name |
| `gender` | M / F |
| `street`, `city`, `state`, `zip` | Cardholder address |
| `lat`, `long` | Cardholder coordinates |
| `city_pop` | City population |
| `job` | Cardholder job title |
| `dob` | Date of birth |
| `merch_lat`, `merch_long` | Merchant coordinates |
| `is_fraud` | *(optional)* Ground-truth label |
        """)
