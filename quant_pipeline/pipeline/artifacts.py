# -*- coding: utf-8 -*-
"""
Artifacts and manifest helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime, timezone


@dataclass(frozen=True)
class FactorManifest:
    path: Path
    selected_factors: List[str]
    batches: List[Path]


def read_factor_manifest(path: Path) -> FactorManifest:
    if not path.exists():
        raise FileNotFoundError(f"factors_manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    batches = [Path(p) for p in data.get("batches", [])]
    return FactorManifest(
        path=path,
        selected_factors=list(data.get("selected_factors", [])),
        batches=batches,
    )


def write_factor_manifest(path: Path, selected_factors: List[str], batches: List[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_factors": selected_factors,
        "batches": [str(p) for p in batches],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_run_metadata(path: Path, payload: Dict[str, Any]) -> None:
    """Write run metadata with ISO timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(payload)
    meta["written_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
