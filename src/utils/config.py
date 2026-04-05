"""
Configuration loader.

Loads pipeline.yaml and models.yaml and exposes them as plain dicts.
All other modules import from here rather than reading YAML directly.
"""

from pathlib import Path
from typing import Any

import yaml


def _load(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def load_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> dict[str, Any]:
    return _load(path)


def load_model_config(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    return _load(path)


def load_all_configs(
    pipeline_path: str | Path = "configs/pipeline.yaml",
    model_path: str | Path = "configs/models.yaml",
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _load(pipeline_path), _load(model_path)
