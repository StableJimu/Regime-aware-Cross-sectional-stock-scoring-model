# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:31:33 2026

@author: jimya
"""
# quant_pipeline/pipeline/factor_processor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Literal, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from .utils import setup_logger, write_parquet

# -----------------------------
# Purpose
# - Compute factor library based on price panel (Alpha101-like single-asset formulas)
# - Expand to cross-sectional signals via per-date transforms (rank/zscore/winsorize)
# - Output: factor_panel (MultiIndex [date, ticker], columns: factor_*)
# - Consumers:
#   - scoring_model.py uses factor_panel
#   - backtester.py uses scores derived from factor_panel via scoring_model
# -----------------------------



def cs_winsorize_zscore(
    factor_panel: pd.DataFrame,
    date_level: str = "date",
    winsor_q: Optional[float] = 0.01,
    ddof: int = 0,
    clip_sigma: Optional[float] = None,
    output: Literal["zscore", "rank", "both"] = "zscore",
) -> pd.DataFrame:
    """
    Cross-sectional normalization per date for factor columns.

    Parameters
    ----------
    factor_panel:
        MultiIndex DataFrame with levels including (date, ticker).
        Columns are factor names.
    winsor_q:
        If not None, clip each factor column to [q, 1-q] quantiles per date.
    ddof:
        Std degrees of freedom for zscore.
    clip_sigma:
        If not None, clip zscores to [-clip_sigma, clip_sigma] per date.
    output:
        - "zscore": return z-scored factors
        - "rank": return cross-sectional ranks scaled to [-0.5, 0.5] (optional)
        - "both": return both with suffixes

    Returns
    -------
    DataFrame with same index; columns transformed.
    """
    if not isinstance(factor_panel.index, pd.MultiIndex):
        raise TypeError("factor_panel must have MultiIndex [date,ticker]")
    if date_level not in factor_panel.index.names:
        raise ValueError(f"MultiIndex must include level '{date_level}'")

    def _transform_one_date(df_date: pd.DataFrame) -> pd.DataFrame:
        X = df_date.copy()

        # winsorize per column
        if winsor_q is not None:
            lo = X.quantile(winsor_q)
            hi = X.quantile(1 - winsor_q)
            X = X.clip(lower=lo, upper=hi, axis=1)

        # z-score per column
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=ddof).replace(0.0, np.nan)
        Z = (X - mu) / sd

        if clip_sigma is not None:
            Z = Z.clip(lower=-clip_sigma, upper=clip_sigma)

        if output == "zscore":
            return Z
        if output == "rank":
            # rank -> [0,1] then center to [-0.5,0.5]
            R = Z.rank(pct=True, axis=0) - 0.5
            return R
        if output == "both":
            R = Z.rank(pct=True, axis=0) - 0.5
            out = pd.concat([Z.add_suffix("__z"), R.add_suffix("__r")], axis=1)
            return out

        raise ValueError("output must be one of {'zscore','rank','both'}")

    out = (
        factor_panel
        .groupby(level=date_level, group_keys=False)
        .apply(_transform_one_date)
    )
    return out







FactorFn = Callable[[pd.DataFrame, Dict], pd.Series]
"""FactorFn may accept per-ticker data or full panel; mode is stored in registry."""


@dataclass(frozen=True)
class FactorProcessorConfig:
    factors_dir: Path
    # cross-sectional transforms
    do_rank: bool = True
    do_zscore: bool = False
    winsorize: Optional[float] = 0.01  # clip at quantiles if not None
    # rolling windows etc. can be added here


class FactorRegistry:
    """Register factor definitions in a single place."""
    def __init__(self):
        self._factors: Dict[str, Tuple[FactorFn, str]] = {}

    def register(self, name: str, fn: FactorFn, mode: str = "ticker") -> None:
        if mode not in {"ticker", "panel"}:
            raise ValueError("mode must be 'ticker' or 'panel'")
        if name in self._factors:
            raise KeyError(f"Factor '{name}' already registered")
        self._factors[name] = (fn, mode)

    def names(self, mode: Optional[str] = None) -> List[str]:
        if mode is None:
            return sorted(self._factors.keys())
        return sorted([k for k, (_, m) in self._factors.items() if m == mode])

    def get(self, name: str) -> FactorFn:
        return self._factors[name][0]

    def get_mode(self, name: str) -> str:
        return self._factors[name][1]

    def items(self) -> List[Tuple[str, Tuple[FactorFn, str]]]:
        return list(self._factors.items())


def _default_factor_library() -> Dict[str, FactorFn]:
    """Minimal built-in factor library. Keep it small and stable."""
    def mom_20d(df: pd.DataFrame, _: Dict) -> pd.Series:
        return np.log(df["close"]).diff(20)
    return {"mom_20d": mom_20d}


class FactorProcessor:
    def __init__(self, cfg: FactorProcessorConfig, registry: FactorRegistry):
        self.cfg = cfg
        self.registry = registry
        self.logger = setup_logger(self.__class__.__name__)

    def compute_factors(self, prices_panel: pd.DataFrame, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute raw factor values.
        Output index: [date, ticker]
        """
        if not self.registry.names():
            for name, fn in _default_factor_library().items():
                self.registry.register(name, fn, mode="ticker")

        if factor_names is None:
            factor_names = self.registry.names()

        if not factor_names:
            raise ValueError("No factor names provided or registered")

        panel_names = [n for n in factor_names if self.registry.get_mode(n) == "panel"]
        ticker_names = [n for n in factor_names if self.registry.get_mode(n) == "ticker"]

        panel_frames: List[pd.Series] = []
        for name in panel_names:
            fn = self.registry.get(name)
            s = fn(prices_panel, {})
            if isinstance(s, pd.DataFrame):
                if s.shape[1] != 1:
                    raise ValueError(f"Panel factor '{name}' must return a Series or single-column DataFrame")
                s = s.iloc[:, 0]
            if not isinstance(s.index, pd.MultiIndex):
                raise ValueError(f"Panel factor '{name}' must return MultiIndex [date,ticker]")
            panel_frames.append(s.rename(name))

        panel_df = pd.concat(panel_frames, axis=1) if panel_frames else None

        out_rows: List[pd.DataFrame] = []
        for ticker, df_t in prices_panel.groupby(level="ticker"):
            df_t = df_t.reset_index(level="ticker", drop=True).sort_index()
            cols = []
            for name in ticker_names:
                fn = self.registry.get(name)
                s = fn(df_t, {})
                s.name = name
                cols.append(s)
            if cols:
                df_f = pd.concat(cols, axis=1)
                df_f["ticker"] = ticker
                out_rows.append(df_f.reset_index())

        ticker_df = None
        if out_rows:
            ticker_df = pd.concat(out_rows, ignore_index=True)
            ticker_df = ticker_df.set_index(["date", "ticker"]).sort_index()

        if panel_df is None:
            if ticker_df is None:
                raise ValueError("No factors computed")
            return ticker_df
        if ticker_df is None:
            return panel_df.sort_index()
        return ticker_df.join(panel_df, how="outer")

    def transform_cross_section(self, factor_panel: pd.DataFrame) -> pd.DataFrame:
        """Apply per-date transforms: winsorize -> rank/zscore, etc."""
        if not self.cfg.do_rank and not self.cfg.do_zscore:
            return factor_panel

        if self.cfg.do_rank and self.cfg.do_zscore:
            return cs_winsorize_zscore(
                factor_panel,
                winsor_q=self.cfg.winsorize,
                output="both",
            )
        if self.cfg.do_rank:
            return cs_winsorize_zscore(
                factor_panel,
                winsor_q=self.cfg.winsorize,
                output="rank",
            )
        return cs_winsorize_zscore(
            factor_panel,
            winsor_q=self.cfg.winsorize,
            output="zscore",
        )

    def save_factor_panel(self, factor_panel: pd.DataFrame, tag: str) -> Path:
        """Persist computed factors (single parquet for now)."""
        out_path = self.cfg.factors_dir / f"factors_{tag}.parquet"
        write_parquet(factor_panel, out_path)
        return out_path
