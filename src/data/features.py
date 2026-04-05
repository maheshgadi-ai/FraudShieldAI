"""
Feature Engineering for FraudShieldAI.

Adds domain-driven features on top of the raw transaction data:
  - Geospatial: Haversine distance between cardholder and merchant
  - Temporal:   Hour, day-of-week, is_weekend, transaction age buckets
  - Velocity:   Transaction count per card over rolling windows (1d, 7d, 30d)
  - Spend:      Rolling mean and std of transaction amount per card
  - Cardholder: Age derived from date-of-birth

All functions are pure transformations — they receive a DataFrame and return
a new one with additional columns, never mutating the input.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Earth radius in kilometres (for Haversine)
_EARTH_RADIUS_KM = 6_371.0


# ---------------------------------------------------------------------------
# Geospatial
# ---------------------------------------------------------------------------

def _haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """Vectorised Haversine distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Haversine distance between cardholder and merchant."""
    df = df.copy()
    df["geo_distance_km"] = _haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    logger.debug("Added geo_distance_km")
    return df


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive time-based features from the transaction datetime.
    Expects a 'trans_date_trans_time' column parseable as datetime.
    """
    df = df.copy()
    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek          # 0=Monday, 6=Sunday
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(np.int8)
    logger.debug("Added time features")
    return df


# ---------------------------------------------------------------------------
# Cardholder age
# ---------------------------------------------------------------------------

def add_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cardholder age in years from 'dob' column."""
    df = df.copy()
    trans_dt = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])
    df["age"] = ((trans_dt - dob).dt.days / 365.25).astype(np.float32)
    logger.debug("Added age")
    return df


# ---------------------------------------------------------------------------
# Transaction velocity (per card, rolling windows)
# ---------------------------------------------------------------------------

def _velocity_for_window(df: pd.DataFrame, days: int) -> pd.Series:
    """
    Count transactions per card number within the past `days` days,
    looking backward from each transaction's timestamp.

    Uses a merge-asof approach that is efficient on sorted data.
    """
    df_sorted = df.sort_values("trans_date_trans_time")
    ts = pd.to_datetime(df_sorted["trans_date_trans_time"])
    cutoff = ts - pd.Timedelta(days=days)

    # For each row, count rows in the same card with ts in (cutoff, ts]
    counts = []
    for i, (card, t, c) in enumerate(
        zip(df_sorted["cc_num"], ts, cutoff)
    ):
        mask = (df_sorted["cc_num"] == card) & (ts > c) & (ts <= t)
        counts.append(int(mask.sum()))

    result = pd.Series(counts, index=df_sorted.index, name=f"tx_velocity_{days}d")
    return result.reindex(df.index)


def add_velocity_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Add transaction velocity features for each window (in days).

    NOTE: The naive loop above is correct but slow on 1.3M rows.
    For production, a vectorised rolling-count approach is used below.
    """
    if windows is None:
        windows = [1, 7, 30]

    df = df.copy()
    df["_ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df_sorted = df.sort_values(["cc_num", "_ts"]).copy()

    for days in windows:
        col = f"tx_velocity_{days}d"
        window = f"{days}D"
        # Rolling count per card using a time-based window
        rolling_count = (
            df_sorted.groupby("cc_num")["_ts"]
            .transform(
                lambda s: s.expanding().count()  # placeholder; replaced below
            )
        )
        # Vectorised: rank within group by time, then subtract rank at (ts - window)
        df_sorted[col] = (
            df_sorted.groupby("cc_num")["_ts"]
            .transform(lambda s: _rolling_count_vectorised(s, days))
        )
        logger.debug("Added %s", col)

    df = df_sorted.sort_index()
    df = df.drop(columns=["_ts"])
    return df


def _rolling_count_vectorised(series: pd.Series, days: int) -> pd.Series:
    """
    For a sorted datetime Series (single card), compute the count of
    transactions within the past `days` days for each transaction.
    """
    ts_arr = series.values.astype("datetime64[ns]")
    delta = np.timedelta64(days, "D")
    counts = np.searchsorted(ts_arr, ts_arr, side="right") - np.searchsorted(
        ts_arr, ts_arr - delta, side="left"
    )
    return pd.Series(counts, index=series.index)


# ---------------------------------------------------------------------------
# Rolling spend statistics (per card)
# ---------------------------------------------------------------------------

def add_spend_features(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """
    Rolling mean and std of transaction amount per card over the past
    `window_days` days (exclusive of the current transaction).
    """
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df_sorted = df.sort_values(["cc_num", "_ts"]).copy()

    def _rolling_mean_std(grp: pd.DataFrame) -> pd.DataFrame:
        amt = grp["amt"].values
        ts = grp["_ts"].values.astype("datetime64[ns]")
        delta = np.timedelta64(window_days, "D")
        means, stds = [], []
        for i in range(len(amt)):
            mask = (ts >= ts[i] - delta) & (ts < ts[i])
            window_vals = amt[mask]
            means.append(window_vals.mean() if len(window_vals) > 0 else amt[i])
            stds.append(window_vals.std() if len(window_vals) > 1 else 0.0)
        grp = grp.copy()
        grp["spend_rolling_mean"] = means
        grp["spend_rolling_std"] = stds
        return grp

    df_sorted = df_sorted.groupby("cc_num", group_keys=False).apply(_rolling_mean_std)
    df_sorted["spend_rolling_std"] = df_sorted["spend_rolling_std"].fillna(0.0)
    df_sorted["spend_ratio"] = df_sorted["amt"] / (df_sorted["spend_rolling_mean"] + 1e-6)

    df = df_sorted.sort_index().drop(columns=["_ts"])
    logger.debug("Added spend_rolling_mean, spend_rolling_std, spend_ratio")
    return df


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.

    Args:
        df:  Raw transaction DataFrame.
        cfg: pipeline.yaml['feature_engineering'] section.

    Returns:
        DataFrame with all engineered features appended.
    """
    logger.info("Starting feature engineering on %d rows", len(df))

    if cfg.get("geo_distance", True):
        df = add_geo_features(df)

    if cfg.get("time_features", True):
        df = add_time_features(df)
        df = add_age_feature(df)

    windows = cfg.get("velocity_windows", [1, 7, 30])
    df = add_velocity_features(df, windows=windows)

    if cfg.get("spending_features", True):
        df = add_spend_features(df)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df
