"""
config.py
---------
Load and expose the project-wide analysis configuration from config/analysis.yml.

Usage
-----
    from src.config import cfg

    bins = cfg["bins"]
    survey = cfg["survey"]
    sample = cfg["samples"]["irac_footprint_masked"]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root is the parent of this file's directory (src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "analysis.yml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the analysis YAML config file.

    Parameters
    ----------
    path : str or Path or None
        Path to a YAML config file.  Defaults to ``config/analysis.yml``
        relative to the project root.

    Returns
    -------
    dict
        Parsed configuration.
    """
    config_path = Path(path) if path is not None else _CONFIG_PATH
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)
    return config


def resolve_path(relative: str) -> Path:
    """Resolve a path that is relative to the project root.

    Parameters
    ----------
    relative : str
        A path string as stored in config (e.g. ``data/raw/LRD_MarIRAC.fits``).

    Returns
    -------
    Path
        Absolute path.
    """
    return _PROJECT_ROOT / relative


# Module-level singleton — import this directly in scripts and notebooks.
cfg: dict[str, Any] = load_config()
