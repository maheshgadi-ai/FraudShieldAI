"""
Shared utility functions used across the pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifact(obj: Any, path: str | Path) -> None:
    """Persist any Python object with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact → %s", path)


def load_artifact(path: str | Path) -> Any:
    """Load a joblib artifact."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that logs elapsed time for a named block."""

    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        label = f"[{self.name}] " if self.name else ""
        logger.info("%sElapsed: %.2fs", label, elapsed)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def memory_usage_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce DataFrame memory footprint."""
    for col in df.select_dtypes(include=[np.number]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if pd.api.types.is_integer_dtype(df[col]):
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if col_min >= np.iinfo(dtype).min and col_max <= np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
        else:
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
