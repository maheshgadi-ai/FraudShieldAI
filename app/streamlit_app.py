"""
FraudShieldAI — Streamlit Web Application

Entry point. Run with:
    streamlit run app/streamlit_app.py

Pages:
  1. Predict          — upload transactions CSV, get real-time fraud scores
  2. Dashboard        — fraud trend analytics and visualisations
  3. Model Report     — metrics comparison table, ROC/PR curves, SHAP plots
"""

import sys
from pathlib import Path

# Make sure project root is on the path when launched from app/ or root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShieldAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean card style, coloured risk badges
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Sidebar nav labels */
    [data-testid="stSidebarNav"] { font-size: 0.95rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Risk badge */
    .badge-LOW    { background:#d4edda; color:#155724; padding:3px 10px;
                    border-radius:12px; font-weight:600; }
    .badge-MEDIUM { background:#fff3cd; color:#856404; padding:3px 10px;
                    border-radius:12px; font-weight:600; }
    .badge-HIGH   { background:#f8d7da; color:#721c24; padding:3px 10px;
                    border-radius:12px; font-weight:600; }

    /* Section divider */
    hr { border: none; border-top: 1px solid #dee2e6; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shield.png",
        width=64,
    )
    st.title("FraudShieldAI")
    st.caption("ML-powered fraud detection")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=["Predict", "Dashboard", "Model Report"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small>Capstone Project · 2025<br>"
        "Models: LR · RF · XGBoost · FNN · LSTM · Ensemble</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "Predict":
    from app.pages.predict import render
    render()

elif page == "Dashboard":
    from app.pages.dashboard import render
    render()

elif page == "Model Report":
    from app.pages.model_report import render
    render()
