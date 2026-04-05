"""
Dashboard Page — FraudShieldAI Streamlit App

Fraud trend analytics on the raw dataset:
  - Fraud rate by hour, day, and month
  - Fraud by merchant category
  - Geographic fraud heatmap (cardholder locations)
  - Transaction amount distributions: fraud vs legit
  - Top merchants and job categories by fraud count
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAW_TRAIN = Path("data/raw/fraudTrain.csv")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…", ttl=3600)
def _load_data(sample_frac: float = 0.25) -> pd.DataFrame:
    """Load a stratified sample of the raw training data for fast rendering."""
    df = pd.read_csv(RAW_TRAIN, parse_dates=["trans_date_trans_time"])
    # Stratified sample to keep fraud representation
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0].sample(
        n=min(len(legit := df[df["is_fraud"] == 0]),
              int(len(df) * sample_frac)),
        random_state=42,
    )
    sampled = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
    sampled["hour"] = sampled["trans_date_trans_time"].dt.hour
    sampled["day_of_week"] = sampled["trans_date_trans_time"].dt.day_name()
    sampled["month"] = sampled["trans_date_trans_time"].dt.strftime("%b %Y")
    sampled["label"] = sampled["is_fraud"].map({0: "Legit", 1: "Fraud"})
    return sampled


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _fraud_by_hour(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["hour", "label"]).size().reset_index(name="count")
    fig = px.bar(
        grp, x="hour", y="count", color="label",
        color_discrete_map={"Legit": "#3498db", "Fraud": "#e74c3c"},
        barmode="group",
        title="Transactions by Hour of Day",
        labels={"hour": "Hour (24h)", "count": "Transaction Count"},
    )
    fig.update_layout(height=350)
    return fig


def _fraud_rate_by_hour(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("hour")["is_fraud"].agg(["mean", "count"]).reset_index()
    grp.columns = ["hour", "fraud_rate", "total"]
    fig = px.line(
        grp, x="hour", y="fraud_rate",
        markers=True,
        title="Fraud Rate by Hour of Day",
        labels={"hour": "Hour (24h)", "fraud_rate": "Fraud Rate"},
    )
    fig.update_traces(line_color="#e74c3c", line_width=2)
    fig.update_layout(height=350, yaxis_tickformat=".2%")
    return fig


def _fraud_by_category(df: pd.DataFrame) -> go.Figure:
    grp = df[df["is_fraud"] == 1].groupby("category").size().reset_index(name="fraud_count")
    grp = grp.sort_values("fraud_count", ascending=True)
    fig = px.bar(
        grp, x="fraud_count", y="category", orientation="h",
        title="Fraud Transactions by Merchant Category",
        labels={"fraud_count": "Fraud Count", "category": "Category"},
        color="fraud_count",
        color_continuous_scale="Reds",
    )
    fig.update_layout(height=420, coloraxis_showscale=False)
    return fig


def _fraud_rate_by_category(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("category")["is_fraud"].agg(["mean", "count"]).reset_index()
    grp.columns = ["category", "fraud_rate", "total"]
    grp = grp.sort_values("fraud_rate", ascending=True)
    fig = px.bar(
        grp, x="fraud_rate", y="category", orientation="h",
        title="Fraud Rate by Merchant Category",
        labels={"fraud_rate": "Fraud Rate", "category": "Category"},
        color="fraud_rate",
        color_continuous_scale="OrRd",
    )
    fig.update_layout(height=420, xaxis_tickformat=".1%", coloraxis_showscale=False)
    return fig


def _amount_dist(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df[df["amt"] < df["amt"].quantile(0.99)],   # clip extreme outliers
        x="label", y="amt", color="label",
        color_discrete_map={"Legit": "#3498db", "Fraud": "#e74c3c"},
        title="Transaction Amount: Fraud vs Legit",
        labels={"amt": "Amount ($)", "label": ""},
        points="outliers",
    )
    fig.update_layout(height=380, showlegend=False)
    return fig


def _amount_hist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df[df["amt"] < df["amt"].quantile(0.99)],
        x="amt", color="label", nbins=60,
        barmode="overlay", opacity=0.7,
        color_discrete_map={"Legit": "#3498db", "Fraud": "#e74c3c"},
        title="Transaction Amount Distribution",
        labels={"amt": "Amount ($)", "label": ""},
    )
    fig.update_layout(height=350)
    return fig


def _geo_map(df: pd.DataFrame) -> go.Figure:
    fraud_df = df[df["is_fraud"] == 1].dropna(subset=["lat", "long"])
    sample = fraud_df.sample(min(2000, len(fraud_df)), random_state=42)
    fig = px.scatter_mapbox(
        sample,
        lat="lat", lon="long",
        color_discrete_sequence=["#e74c3c"],
        zoom=3,
        mapbox_style="carto-positron",
        title="Geographic Distribution of Fraudulent Transactions",
        hover_data={"amt": True, "category": True, "lat": False, "long": False},
        opacity=0.6,
        size_max=8,
    )
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def _day_heatmap(df: pd.DataFrame) -> go.Figure:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grp = (
        df.groupby(["day_of_week", "hour"])["is_fraud"]
        .mean()
        .reset_index()
    )
    pivot = grp.pivot(index="day_of_week", columns="hour", values="is_fraud")
    pivot = pivot.reindex(day_order)

    fig = px.imshow(
        pivot,
        color_continuous_scale="Reds",
        title="Fraud Rate Heatmap — Day × Hour",
        labels=dict(x="Hour of Day", y="Day of Week", color="Fraud Rate"),
        aspect="auto",
    )
    fig.update_layout(height=380)
    return fig


def _top_merchants(df: pd.DataFrame) -> go.Figure:
    top = (
        df[df["is_fraud"] == 1]
        .groupby("merchant")
        .size()
        .nlargest(15)
        .reset_index(name="count")
        .sort_values("count")
    )
    fig = px.bar(
        top, x="count", y="merchant", orientation="h",
        title="Top 15 Merchants by Fraud Volume",
        color="count", color_continuous_scale="Reds",
    )
    fig.update_layout(height=420, coloraxis_showscale=False)
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("📊 Fraud Analytics Dashboard")

    if not RAW_TRAIN.exists():
        st.warning(
            "Raw training data not found at `data/raw/fraudTrain.csv`. "
            "Download the Kaggle dataset first:\n\n"
            "```bash\nkaggle datasets download -d kartik2112/fraud-detection "
            "-p data/raw --unzip\n```"
        )
        return

    with st.spinner("Loading dataset…"):
        df = _load_data()

    # ----------------------------------------------------------------- KPIs
    total = len(df)
    n_fraud = int(df["is_fraud"].sum())
    fraud_rate = df["is_fraud"].mean()
    avg_fraud_amt = df[df["is_fraud"] == 1]["amt"].mean()
    avg_legit_amt = df[df["is_fraud"] == 0]["amt"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions (sample)", f"{total:,}")
    c2.metric("Fraudulent", f"{n_fraud:,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.3%}")
    c4.metric("Avg Fraud Amount", f"${avg_fraud_amt:,.2f}")
    c5.metric("Avg Legit Amount", f"${avg_legit_amt:,.2f}")

    st.markdown("---")

    # --------------------------------------------------------- Time patterns
    st.subheader("Temporal Patterns")
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(_fraud_by_hour(df), use_container_width=True)
    with t2:
        st.plotly_chart(_fraud_rate_by_hour(df), use_container_width=True)

    st.plotly_chart(_day_heatmap(df), use_container_width=True)

    # --------------------------------------------------------- Categories
    st.markdown("---")
    st.subheader("Merchant Category Analysis")
    cat1, cat2 = st.columns(2)
    with cat1:
        st.plotly_chart(_fraud_by_category(df), use_container_width=True)
    with cat2:
        st.plotly_chart(_fraud_rate_by_category(df), use_container_width=True)

    # --------------------------------------------------------- Amounts
    st.markdown("---")
    st.subheader("Transaction Amount Patterns")
    a1, a2 = st.columns(2)
    with a1:
        st.plotly_chart(_amount_dist(df), use_container_width=True)
    with a2:
        st.plotly_chart(_amount_hist(df), use_container_width=True)

    # --------------------------------------------------------- Geography
    st.markdown("---")
    st.subheader("Geographic Distribution")
    st.plotly_chart(_geo_map(df), use_container_width=True)

    # --------------------------------------------------------- Top merchants
    st.markdown("---")
    st.subheader("Top Merchants by Fraud Volume")
    st.plotly_chart(_top_merchants(df), use_container_width=True)
