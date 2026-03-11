# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:33:39 2026

@author: jimya
"""

# quant_pipeline/pipeline/utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging
import pandas as pd


# -----------------------------
# Purpose
# - Cross-module utilities: logging, I/O, validation, experiment paths
# - Keep other modules clean; do NOT dump business logic here
# -----------------------------


@dataclass(frozen=True)
class ExperimentPaths:
    """Standard output structure for a run.
    Used by: scripts/run_backtest.py, backtester.py
    """
    run_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    tables_dir: Path
    figures_dir: Path


def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Create or get a logger with consistent formatting.
    Used by: all modules.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def read_yaml(path: Path) -> Dict[str, Any]:
    """Lightweight YAML reader."""
    import yaml  # local import to keep dependency localized
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def ensure_datetime_index(df: pd.DataFrame, name: str = "date") -> None:
    """Validate that df index contains datetime-like or has a date column."""
    if df.index.name == name:
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise TypeError(f"Index '{name}' must be datetime64; got {df.index.dtype}")
    elif name in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[name]):
            raise TypeError(f"Column '{name}' must be datetime64; got {df[name].dtype}")
    else:
        raise ValueError(f"Missing datetime index or column '{name}'")


def make_experiment_paths(base_dir: Path, run_id: str) -> ExperimentPaths:
    run_dir = base_dir / run_id
    return ExperimentPaths(
        run_dir=run_dir,
        artifacts_dir=run_dir / "artifacts",
        logs_dir=run_dir / "logs",
        tables_dir=run_dir / "tables",
        figures_dir=run_dir / "figures",
    )
