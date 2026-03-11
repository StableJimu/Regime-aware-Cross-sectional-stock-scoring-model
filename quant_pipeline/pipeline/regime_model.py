# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:32:27 2026

@author: jimya
"""

# quant_pipeline/pipeline/regime_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import setup_logger, write_json


# -----------------------------
# Purpose
# - Fit HMM market regime model using market_features (date-indexed)
# - Provide:
#   - hard labels: regime_label[t] in {0..K-1}
#   - soft probs: gamma[t,k] posterior probabilities
# - Consumers:
#   - scoring_model.py uses (label, gamma) to train regime-wise models and ensemble
#   - backtester.py uses them for regime segmentation reporting (optional)
# -----------------------------


@dataclass(frozen=True)
class RegimeModelConfig:
    k_states: int = 3
    feature_cols: Tuple[str, ...] = ("mkt_ret_1d", "mkt_vol_20d", "y2", "term_spread")
    # walk-forward / training window controls
    train_window_days: int = 756  # ~3y
    refit_freq_days: int = 21     # monthly
    model_dir: Optional[Path] = None


class RegimeModel:
    def __init__(self, cfg: RegimeModelConfig):
        self.cfg = cfg
        self.logger = setup_logger(self.__class__.__name__)
        self._fitted = False
        # placeholders
        self.scaler_: Optional[Dict[str, np.ndarray]] = None
        self.model_: Optional[object] = None
        self.k_states_: int = self.cfg.k_states

    def _standardize_fit(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-12
        return {"mu": mu, "sd": sd}

    def _standardize_apply(self, X: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
        return (X - scaler["mu"]) / scaler["sd"]

    def fit(self, market_features_train: pd.DataFrame) -> None:
        """Fit HMM on training slice only (no look-ahead)."""
        if self.cfg.k_states != 1:
            self.logger.warning("Minimal regime model uses a single regime; overriding k_states to 1.")
            self.k_states_ = 1
        else:
            self.k_states_ = 1
        self._fitted = True
        _ = market_features_train  # placeholder for future models

    def infer(self, market_features: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Infer regimes on provided period using fitted model.
        Returns:
          - hard_labels: pd.Series index=date
          - gamma: pd.DataFrame index=date, columns=regime_0..regime_{K-1}
        """
        if not self._fitted:
            raise RuntimeError("RegimeModel not fitted")

        idx = market_features.index
        labels = pd.Series(0, index=idx, name="regime_label")
        proba = pd.DataFrame(
            0.0,
            index=idx,
            columns=[f"regime_{i}" for i in range(self.k_states_)],
        )
        proba.iloc[:, 0] = 1.0
        return labels, proba

    def save_artifacts(self) -> None:
        """Persist scaler/model metadata."""
        if self.cfg.model_dir is None:
            return
        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)
        if self.scaler_ is not None:
            write_json(self.cfg.model_dir / "scaler.json", {
                "feature_cols": list(self.cfg.feature_cols),
                "mu": self.scaler_["mu"].tolist(),
                "sd": self.scaler_["sd"].tolist(),
            })
        # TODO: save model object (pickle/joblib) when implemented
