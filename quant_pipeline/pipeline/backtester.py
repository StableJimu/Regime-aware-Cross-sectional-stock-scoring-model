# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:34:00 2026

@author: jimya
"""

# quant_pipeline/pipeline/backtester.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from .utils import setup_logger, write_parquet, ExperimentPaths


# -----------------------------
# Purpose
# - Convert daily scores into positions and compute PnL under specified rules
# - Output standardized result artifacts for analysis & reporting
# - Consumer: scripts/run_backtest.py
# -----------------------------


@dataclass(frozen=True)
class BacktestConfig:
    long_short: bool = True
    top_q: float = 0.1
    bottom_q: float = 0.1
    holding_period: int = 1  # days
    rebalance_freq: int = 1  # days
    cost_bps: float = 5.0
    max_weight: float = 0.3  # optional constraint
    signal_lag: int = 1  # T+1 signal
    short_fee_bps: float = 0.0
    score_norm: str | None = None  # "zscore" | None
    portfolio_method: str = "top_q"  # "top_q" | "robust_spo"
    robust_window: int = 60
    robust_uncertainty_k: float = 1.0
    robust_lambda_l2: float = 0.5
    robust_turnover_mult: float = 1.0
    robust_turnover_cap: float | None = None  # L1 cap on daily turnover
    robust_penalty_scale: str | None = None  # "score_std" | None
    robust_max_leverage: float = 1.0
    robust_min_leverage: float = 1.0
    robust_target_leverage: float = 1.0
    robust_breadth_n: int = 0


class Backtester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.logger = setup_logger(self.__class__.__name__)

    def build_positions(self, score_panel: pd.DataFrame, prices_panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Input: MultiIndex [date,ticker], column 'score'
        Output: positions MultiIndex [date,ticker], column 'w'
        """
        if "score" not in score_panel.columns:
            raise ValueError("score_panel must contain column 'score'")
        if self.cfg.portfolio_method == "robust_spo":
            if prices_panel is None:
                raise ValueError("prices_panel is required for robust_spo")
            return self._build_positions_robust_spo(score_panel, prices_panel)

        scores = score_panel["score"].unstack("ticker")
        if self.cfg.score_norm == "zscore":
            scores = self._zscore(scores)
        positions = pd.DataFrame(index=scores.index, columns=scores.columns, dtype=float)

        if self.cfg.rebalance_freq <= 0:
            raise ValueError("rebalance_freq must be >= 1")

        rebalance_dates = scores.index[:: self.cfg.rebalance_freq]
        for d in rebalance_dates:
            s = scores.loc[d].dropna()
            if s.empty:
                continue
            n = len(s)
            top_n = max(1, int(np.floor(n * self.cfg.top_q)))
            bottom_n = max(1, int(np.floor(n * self.cfg.bottom_q))) if self.cfg.long_short else 0

            w = pd.Series(0.0, index=s.index)
            longs = s.nlargest(top_n).index
            w.loc[longs] = 1.0 / top_n

            if self.cfg.long_short and bottom_n > 0:
                shorts = s.nsmallest(bottom_n).index
                w.loc[shorts] = -1.0 / bottom_n

            if self.cfg.max_weight is not None:
                w = w.clip(lower=-self.cfg.max_weight, upper=self.cfg.max_weight)

            positions.loc[d] = w

        positions = positions.ffill().fillna(0.0)
        self.logger.info(
            f"positions: rebalance_dates={len(rebalance_dates)}, "
            f"avg_longs={(positions > 0).sum(axis=1).mean():.1f}, avg_shorts={(positions < 0).sum(axis=1).mean():.1f}"
        )
        return positions.stack().to_frame("w")

    def _compute_uncertainty(
        self,
        score_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        window: int,
        k: float,
    ) -> pd.DataFrame:
        """Rolling RMSE of prediction error per ticker, shifted to avoid lookahead."""
        ret = prices_panel["ret_1d"]
        shift_n = max(1, int(self.cfg.signal_lag))
        fwd_ret = ret.groupby(level="ticker").shift(-shift_n)

        df = score_panel[["score"]].join(fwd_ret.rename("fwd_ret"), how="inner")
        df = df.dropna(subset=["score", "fwd_ret"])
        err = df["score"] - df["fwd_ret"]

        def _rmse(x: pd.Series) -> float:
            return float(np.sqrt(np.mean(np.square(x.to_numpy()))))

        rmse = err.groupby(level="ticker").rolling(window, min_periods=max(5, window // 4)).apply(_rmse, raw=False)
        rmse = rmse.reset_index(level=0, drop=True).shift(1)
        delta = (k * rmse).to_frame("delta")

        delta = delta.reindex(score_panel.index)
        by_date = delta.groupby(level="date")["delta"].transform("median")
        overall_med = float(delta["delta"].median()) if not delta["delta"].isna().all() else 0.0
        delta["delta"] = delta["delta"].fillna(by_date).fillna(overall_med)
        return delta["delta"].unstack("ticker")

    def _build_positions_robust_spo(self, score_panel: pd.DataFrame, prices_panel: pd.DataFrame) -> pd.DataFrame:
        try:
            import cvxpy as cp
        except Exception as e:
            raise ImportError("cvxpy is required for robust_spo portfolio method") from e

        scores = score_panel["score"].unstack("ticker")
        if self.cfg.score_norm == "zscore":
            scores = self._zscore(scores)
        dates = scores.index
        positions = pd.DataFrame(index=dates, columns=scores.columns, dtype=float)

        if self.cfg.rebalance_freq <= 0:
            raise ValueError("rebalance_freq must be >= 1")

        delta = self._compute_uncertainty(
            score_panel=score_panel,
            prices_panel=prices_panel,
            window=int(self.cfg.robust_window),
            k=float(self.cfg.robust_uncertainty_k),
        )

        rebalance_dates = dates[:: self.cfg.rebalance_freq]
        w_prev = pd.Series(0.0, index=scores.columns)
        solver = "SCS"

        for d in rebalance_dates:
            s = scores.loc[d].dropna()
            if s.empty:
                continue
            if self.cfg.robust_breadth_n and self.cfg.robust_breadth_n > 0:
                k = int(self.cfg.robust_breadth_n)
                top = s.nlargest(k)
                bot = s.nsmallest(k)
                s = pd.concat([top, bot]).groupby(level=0).first()
            tickers = s.index
            mu = s.to_numpy(dtype=float)
            delta_d = delta.loc[d, tickers].to_numpy(dtype=float)
            if np.isnan(delta_d).any():
                finite = delta_d[np.isfinite(delta_d)]
                fill = float(np.median(finite)) if finite.size else 0.0
                delta_d = np.nan_to_num(delta_d, nan=fill, posinf=fill, neginf=fill)
            w0 = w_prev.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
            if not self.cfg.long_short:
                w0 = np.clip(w0, 0.0, None)

            w = cp.Variable(len(tickers))
            penalty_scale = 1.0
            if self.cfg.robust_penalty_scale == "score_std":
                penalty_scale = float(np.nanstd(mu))
                if not np.isfinite(penalty_scale) or penalty_scale <= 0.0:
                    penalty_scale = 1.0
            obj = (
                mu @ w
                - float(self.cfg.robust_lambda_l2) * penalty_scale * cp.sum_squares(w)
                - float(self.cfg.robust_turnover_mult)
                * penalty_scale
                * (float(self.cfg.cost_bps) / 10000.0)
                * cp.norm1(w - w0)
                - cp.sum(cp.multiply(delta_d, cp.abs(w)))
            )

            constraints = []
            if self.cfg.long_short:
                constraints.append(cp.sum(w) == 0.0)
            else:
                constraints.append(cp.sum(w) == 1.0)
                constraints.append(w >= 0.0)
            if self.cfg.max_weight is not None:
                constraints.append(w <= float(self.cfg.max_weight))
                constraints.append(w >= -float(self.cfg.max_weight))

            # warm start: long-short top/bottom k
            k = min(5, max(0, len(tickers) // 2))
            if k > 0:
                w_init = np.zeros(len(tickers), dtype=float)
                top_idx = np.argsort(-mu)[:k]
                bot_idx = np.argsort(mu)[:k]
                w_init[top_idx] = 1.0 / (2.0 * k)
                w_init[bot_idx] = -1.0 / (2.0 * k)
                w.value = w_init

            prob = cp.Problem(cp.Maximize(obj), constraints)
            try:
                prob.solve(
                    solver=solver,
                    verbose=False,
                    max_iters=2000,
                    eps=5e-3,
                    warm_start=True,
                )
            except Exception:
                prob.solve(
                    solver="OSQP",
                    verbose=False,
                    max_iter=5000,
                    eps_abs=1e-3,
                    eps_rel=1e-3,
                    polish=True,
                    warm_start=True,
                )

            w_val = None
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                w_val = np.asarray(w.value, dtype=float)
            else:
                self.logger.warning(f"robust_spo: solve failed on {d} (status={prob.status}); skip rebalance")
                w_val = w0  # keep previous weights if infeasible or failed
            if not self.cfg.long_short:
                w_val = np.clip(w_val, 0.0, None)
            positions.loc[d, tickers] = w_val
            w_prev = pd.Series(0.0, index=scores.columns)
            w_prev.loc[tickers] = w_val

        positions = positions.ffill().fillna(0.0)
        self.logger.info(
            f"robust_spo: rebalance_dates={len(rebalance_dates)}, "
            f"avg_longs={(positions > 0).sum(axis=1).mean():.1f}, avg_shorts={(positions < 0).sum(axis=1).mean():.1f}"
        )
        return positions.stack().to_frame("w")

    @staticmethod
    def _zscore(scores: pd.DataFrame) -> pd.DataFrame:
        mean = scores.mean(axis=1)
        std = scores.std(axis=1)
        std = std.replace(0.0, np.nan)
        z = scores.sub(mean, axis=0).div(std, axis=0)
        return z.fillna(0.0)

    def compute_pnl(self, positions: pd.DataFrame, prices_panel: pd.DataFrame) -> pd.DataFrame:
        """Compute daily portfolio returns and equity curve."""
        if "ret_1d" not in prices_panel.columns:
            raise ValueError("prices_panel missing ret_1d")

        # ret_1d is stored as log return; convert to simple return for PnL
        ret_log = prices_panel["ret_1d"].unstack("ticker")
        ret = np.exp(ret_log) - 1.0
        w = positions["w"].unstack("ticker").reindex(ret.index).fillna(0.0)

        # T+1 signal: use yesterday's weights for today's return
        if self.cfg.signal_lag > 0:
            w = w.shift(self.cfg.signal_lag).fillna(0.0)

        if self.cfg.signal_lag > 0:
            # weights already lagged -> use same-day return
            gross = (w * ret).sum(axis=1)
        else:
            fwd_ret = ret.shift(-1)
            gross = (w * fwd_ret).sum(axis=1)

        turnover = w.diff().abs().sum(axis=1) * 0.5
        cost = turnover * (self.cfg.cost_bps / 10000.0)

        # Short fee on short notional (daily)
        if self.cfg.short_fee_bps and self.cfg.short_fee_bps > 0:
            short_notional = (-w.clip(upper=0.0)).sum(axis=1)
            daily_fee = (self.cfg.short_fee_bps / 10000.0) / 252.0
            short_fee = short_notional * daily_fee
        else:
            short_fee = 0.0

        net = gross - cost - short_fee

        perf = pd.DataFrame({"gross": gross, "cost": cost, "net": net})
        if isinstance(short_fee, pd.Series):
            perf["short_fee"] = short_fee
        perf["equity"] = (1.0 + perf["net"].fillna(0.0)).cumprod()
        self.logger.info(f"turnover avg={turnover.mean():.4f}, cost avg={cost.mean():.6f}")
        return perf

    def run(
        self,
        score_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        out: Optional[ExperimentPaths] = None,
    ) -> pd.DataFrame:
        """Main entry. Returns perf DataFrame indexed by date."""
        pos = self.build_positions(score_panel, prices_panel=prices_panel)
        perf = self.compute_pnl(pos, prices_panel)

        if out is not None:
            out.tables_dir.mkdir(parents=True, exist_ok=True)
            write_parquet(perf, out.tables_dir / "performance.parquet")
            write_parquet(pos, out.tables_dir / "positions.parquet")
            write_parquet(score_panel, out.tables_dir / "scores.parquet")

        return perf
