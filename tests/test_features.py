"""
Unit tests for src/data/features.py
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_age_feature,
    add_geo_features,
    add_spend_features,
    add_time_features,
    add_velocity_features,
    build_features,
)


# ---------------------------------------------------------------------------
# Geo features
# ---------------------------------------------------------------------------

class TestGeoFeatures:
    def test_column_created(self, raw_df):
        out = add_geo_features(raw_df)
        assert "geo_distance_km" in out.columns

    def test_non_negative(self, raw_df):
        out = add_geo_features(raw_df)
        assert (out["geo_distance_km"] >= 0).all()

    def test_same_location_is_zero(self):
        df = pd.DataFrame({
            "lat": [40.0], "long": [-74.0],
            "merch_lat": [40.0], "merch_long": [-74.0],
        })
        out = add_geo_features(df)
        assert out["geo_distance_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # NYC (40.71, -74.01) to London (51.51, -0.13) ≈ 5_570 km
        df = pd.DataFrame({
            "lat": [40.71], "long": [-74.01],
            "merch_lat": [51.51], "merch_long": [-0.13],
        })
        out = add_geo_features(df)
        assert out["geo_distance_km"].iloc[0] == pytest.approx(5_570, rel=0.02)

    def test_does_not_mutate_input(self, raw_df):
        original_cols = set(raw_df.columns)
        _ = add_geo_features(raw_df)
        assert set(raw_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

class TestTimeFeatures:
    def test_columns_created(self, raw_df):
        out = add_time_features(raw_df)
        for col in ["hour", "day_of_week", "day_of_month", "month", "is_weekend", "is_night"]:
            assert col in out.columns

    def test_hour_range(self, raw_df):
        out = add_time_features(raw_df)
        assert out["hour"].between(0, 23).all()

    def test_day_of_week_range(self, raw_df):
        out = add_time_features(raw_df)
        assert out["day_of_week"].between(0, 6).all()

    def test_is_weekend_binary(self, raw_df):
        out = add_time_features(raw_df)
        assert set(out["is_weekend"].unique()).issubset({0, 1})

    def test_is_night_binary(self, raw_df):
        out = add_time_features(raw_df)
        assert set(out["is_night"].unique()).issubset({0, 1})

    def test_known_midnight_is_night(self):
        df = pd.DataFrame({"trans_date_trans_time": ["2020-06-15 00:30:00"]})
        out = add_time_features(df)
        assert out["is_night"].iloc[0] == 1

    def test_known_noon_not_night(self):
        df = pd.DataFrame({"trans_date_trans_time": ["2020-06-15 12:00:00"]})
        out = add_time_features(df)
        assert out["is_night"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Age feature
# ---------------------------------------------------------------------------

class TestAgeFeature:
    def test_column_created(self, raw_df):
        out = add_age_feature(raw_df)
        assert "age" in out.columns

    def test_positive_ages(self, raw_df):
        out = add_age_feature(raw_df)
        assert (out["age"] > 0).all()

    def test_reasonable_range(self, raw_df):
        out = add_age_feature(raw_df)
        assert out["age"].between(0, 120).all()

    def test_known_age(self):
        df = pd.DataFrame({
            "trans_date_trans_time": ["2020-01-01 00:00:00"],
            "dob": ["1990-01-01"],
        })
        out = add_age_feature(df)
        assert out["age"].iloc[0] == pytest.approx(30.0, abs=0.1)


# ---------------------------------------------------------------------------
# Velocity features
# ---------------------------------------------------------------------------

class TestVelocityFeatures:
    def test_columns_created(self, raw_df):
        out = add_velocity_features(raw_df, windows=[1, 7])
        assert "tx_velocity_1d" in out.columns
        assert "tx_velocity_7d" in out.columns

    def test_counts_non_negative(self, raw_df):
        out = add_velocity_features(raw_df, windows=[7])
        assert (out["tx_velocity_7d"] >= 0).all()

    def test_first_transaction_count_is_one(self):
        """A card's very first transaction should have velocity = 1."""
        df = pd.DataFrame({
            "trans_date_trans_time": ["2020-01-01 10:00:00"],
            "cc_num": ["4111111111111111"],
            "amt": [50.0],
        })
        out = add_velocity_features(df, windows=[7])
        assert out["tx_velocity_7d"].iloc[0] == 1

    def test_velocity_increases_over_time(self):
        """Two transactions on the same card within 7 days → velocity increases."""
        df = pd.DataFrame({
            "trans_date_trans_time": [
                "2020-01-01 10:00:00",
                "2020-01-03 10:00:00",
            ],
            "cc_num": ["4111111111111111", "4111111111111111"],
            "amt": [50.0, 60.0],
        })
        out = add_velocity_features(df, windows=[7])
        assert out["tx_velocity_7d"].iloc[1] >= out["tx_velocity_7d"].iloc[0]

    def test_no_ts_column_leaked(self, raw_df):
        out = add_velocity_features(raw_df, windows=[1])
        assert "_ts" not in out.columns


# ---------------------------------------------------------------------------
# Spend features
# ---------------------------------------------------------------------------

class TestSpendFeatures:
    def test_columns_created(self, raw_df):
        out = add_spend_features(raw_df)
        for col in ["spend_rolling_mean", "spend_rolling_std", "spend_ratio"]:
            assert col in out.columns

    def test_no_nulls(self, raw_df):
        out = add_spend_features(raw_df)
        assert out["spend_rolling_mean"].notna().all()
        assert out["spend_rolling_std"].notna().all()

    def test_spend_ratio_positive(self, raw_df):
        out = add_spend_features(raw_df)
        assert (out["spend_ratio"] > 0).all()


# ---------------------------------------------------------------------------
# build_features (integration)
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_all_features_added(self, raw_df):
        cfg = {
            "geo_distance": True,
            "time_features": True,
            "velocity_windows": [1, 7],
            "spending_features": True,
        }
        out = build_features(raw_df, cfg)
        for col in ["geo_distance_km", "hour", "age", "tx_velocity_1d", "spend_rolling_mean"]:
            assert col in out.columns

    def test_shape_preserved(self, raw_df):
        cfg = {"geo_distance": True, "time_features": True,
               "velocity_windows": [1], "spending_features": True}
        out = build_features(raw_df, cfg)
        assert len(out) == len(raw_df)

    def test_geo_disabled(self, raw_df):
        cfg = {"geo_distance": False, "time_features": True,
               "velocity_windows": [], "spending_features": False}
        out = build_features(raw_df, cfg)
        assert "geo_distance_km" not in out.columns
