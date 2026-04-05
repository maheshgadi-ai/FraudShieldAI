"""
Preprocessing pipeline for FraudShieldAI.

Steps:
  1. Load raw train / test CSVs
  2. Drop identifiers and leakage columns
  3. Run feature engineering (features.py)
  4. Encode categoricals (target-encoding for high-cardinality, label for low)
  5. Scale numerical features
  6. Apply SMOTE on training set to address class imbalance
  7. Persist processed splits as Parquet

Call via:
    python main.py --stage preprocess
or directly:
    from src.data.preprocessing import run_preprocessing
    run_preprocessing(pipeline_cfg)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

from src.data.features import build_features
from src.utils.helpers import Timer, reduce_mem_usage, save_artifact

logger = logging.getLogger(__name__)

# Columns to drop before modelling (identifiers, raw coordinates already
# encoded into geo_distance_km, and direct leakage fields)
_DROP_COLS = [
    "trans_num",         # transaction ID — identifier
    "first",             # cardholder first name
    "last",              # cardholder last name
    "street",            # address — too granular, covered by lat/long
    "city",              # covered by lat/long + city_pop
    "state",             # covered by lat/long
    "zip",               # covered by lat/long
    "dob",               # replaced by 'age'
    "unix_time",         # redundant with trans_date_trans_time
    "trans_date_trans_time",  # replaced by hour/day/month features
    # Keep lat, long, merch_lat, merch_long? No — geo_distance already captures them
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]

# Low-cardinality categoricals → LabelEncoder
_LABEL_ENCODE_COLS = ["gender"]

# High-cardinality categoricals → target mean encoding (fraud rate per category)
_TARGET_ENCODE_COLS = ["merchant", "category", "job"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path(cfg["data"]["raw_train_path"])
    test_path = Path(cfg["data"]["raw_test_path"])

    logger.info("Loading training data from %s", train_path)
    train = pd.read_csv(train_path, index_col=0)
    logger.info("Loading test data from %s", test_path)
    test = pd.read_csv(test_path, index_col=0)

    logger.info("Train shape: %s | Test shape: %s", train.shape, test.shape)
    return train, test


def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_present = [c for c in _DROP_COLS if c in df.columns]
    return df.drop(columns=cols_present)


def _label_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    encoders: dict[str, LabelEncoder] = {}
    for col in _LABEL_ENCODE_COLS:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le
    return train, test, encoders


def _target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """
    Smoothed target (mean) encoding.
    Encoded value = (n_i * mean_i + smoothing * global_mean) / (n_i + smoothing)
    Unseen categories in test get the global mean.

    Args:
        y_train: Target series (already separated from train DataFrame).
    """
    encodings: dict[str, pd.Series] = {}
    global_mean = float(y_train.mean())

    for col in _TARGET_ENCODE_COLS:
        if col not in train.columns:
            continue
        stats = y_train.groupby(train[col]).agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        train[col] = train[col].map(smooth).fillna(global_mean).astype(np.float32)
        test[col] = test[col].map(smooth).fillna(global_mean).astype(np.float32)
        encodings[col] = smooth

    return train, test, encodings


def _get_scaler(scaling: str):
    mapping = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    return mapping.get(scaling, StandardScaler())


def _scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaling: str,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    scaler = _get_scaler(scaling)
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, scaler


def _apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    smote_cfg: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    if not smote_cfg.get("enabled", True):
        return X, y

    fraud_rate = y.mean()
    logger.info("Class distribution before SMOTE — fraud rate: %.4f%%", fraud_rate * 100)

    smote = SMOTE(
        sampling_strategy=smote_cfg.get("sampling_strategy", 0.1),
        random_state=smote_cfg.get("random_state", 42),
        n_jobs=-1,
    )
    X_res, y_res = smote.fit_resample(X, y)

    logger.info(
        "After SMOTE — samples: %d | fraud rate: %.4f%%",
        len(y_res),
        y_res.mean() * 100,
    )
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_preprocessing(cfg: dict) -> None:
    """
    Full preprocessing pipeline.

    Reads raw data → engineers features → encodes → scales → SMOTE →
    saves train/test as Parquet files.

    Args:
        cfg: Contents of pipeline.yaml as a dict.
    """
    seed = cfg.get("data", {}).get("random_seed", 42)
    target = cfg["data"]["target_column"]
    feat_cfg = cfg["feature_engineering"]
    pre_cfg = cfg["preprocessing"]

    with Timer("load raw data"):
        train_raw, test_raw = _load_raw(cfg)

    # ---------- Feature engineering ----------
    with Timer("feature engineering — train"):
        train_feat = build_features(train_raw, feat_cfg)
    with Timer("feature engineering — test"):
        test_feat = build_features(test_raw, feat_cfg)

    # ---------- Drop identifier / raw coordinate columns ----------
    train_feat = _drop_columns(train_feat)
    test_feat = _drop_columns(test_feat)

    # ---------- Separate target ----------
    y_train = train_feat.pop(target)
    y_test = test_feat.pop(target)

    # ---------- Encoding ----------
    with Timer("encoding"):
        train_feat, test_feat, label_encoders = _label_encode(train_feat, test_feat)
        train_feat, test_feat, target_encoders = _target_encode(
            train_feat, test_feat, y_train=y_train,
        )

    # Encode cc_num as integer hash (anonymised, keeps cardinality low)
    if "cc_num" in train_feat.columns:
        for df in [train_feat, test_feat]:
            df["cc_num"] = df["cc_num"].astype(str).apply(
                lambda x: int(hash(x) % (2 ** 31))
            ).astype(np.int32)

    # ---------- Fill any remaining NaNs ----------
    train_feat = train_feat.fillna(0)
    test_feat = test_feat.fillna(0)

    # ---------- Scale ----------
    numeric_cols = train_feat.select_dtypes(include=[np.number]).columns.tolist()
    with Timer("scaling"):
        train_feat, test_feat, scaler = _scale_features(
            train_feat, test_feat, pre_cfg["scaling"], numeric_cols
        )

    # ---------- Memory optimisation ----------
    train_feat = reduce_mem_usage(train_feat)
    test_feat = reduce_mem_usage(test_feat)

    # ---------- SMOTE (training set only) ----------
    with Timer("SMOTE"):
        X_train_res, y_train_res = _apply_smote(
            train_feat, y_train, pre_cfg["smote"]
        )

    # ---------- Persist ----------
    out_train = Path(cfg["data"]["processed_train_path"])
    out_test = Path(cfg["data"]["processed_test_path"])
    out_train.parent.mkdir(parents=True, exist_ok=True)

    train_out = X_train_res.copy()
    train_out[target] = y_train_res.values
    test_out = test_feat.copy()
    test_out[target] = y_test.values

    train_out.to_parquet(out_train, index=False)
    test_out.to_parquet(out_test, index=False)
    logger.info("Saved processed train → %s  (%d rows)", out_train, len(train_out))
    logger.info("Saved processed test  → %s  (%d rows)", out_test, len(test_out))

    # ---------- Save preprocessing artifacts ----------
    artifacts_dir = Path(cfg["training"]["output_dir"])
    save_artifact(scaler, artifacts_dir / "scaler.joblib")
    save_artifact(label_encoders, artifacts_dir / "label_encoders.joblib")
    save_artifact(target_encoders, artifacts_dir / "target_encoders.joblib")
    save_artifact(numeric_cols, artifacts_dir / "numeric_cols.joblib")

    logger.info("Preprocessing complete.")
