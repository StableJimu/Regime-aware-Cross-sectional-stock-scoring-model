# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:34:00 2026

@author: jimya
"""

# quant_pipeline/pipeline/backtester.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from statistics import NormalDist

import pandas as pd
import numpy as np

from .spo import estimate_covariance, solve_spo_day
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
    long_budget: float = 1.0
    short_budget: float = 1.0
    holding_period: int = 1  # days
    rebalance_freq: int = 1  # days
    cost_bps: float = 5.0
    max_weight: float = 0.3  # optional constraint
    signal_lag: int = 1  # T+1 signal
    short_fee_bps: float = 0.0
    score_norm: str | None = None  # "zscore" | None
    portfolio_method: str = "diag_mv"  # "top_q" | "spo" | "robust_spo" | "proportional" | "diag_mv"
    initial_capital: float = 1_000_000.0
    optimizer_alpha_source: str = "auto"  # "auto" | "calibrated" | "raw" | "score"
    optimizer_alpha_method: str = "rank_normal"  # "raw" | "zscore" | "rank_normal"
    spo_cov_window: int = 60
    spo_risk_aversion: float = 5.0
    spo_cov_shrinkage: float = 0.2
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


@dataclass(frozen=True)
class BacktestResult:
    performance: pd.DataFrame
    positions: pd.DataFrame


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
        if self.cfg.portfolio_method == "proportional":
            return self._build_positions_proportional(score_panel)
        if self.cfg.portfolio_method == "diag_mv":
            if prices_panel is None:
                raise ValueError("prices_panel is required for diag_mv portfolio method")
            return self._build_positions_diag_mv(score_panel, prices_panel)
        if self.cfg.portfolio_method in {"spo", "robust_spo"}:
            if prices_panel is None:
                raise ValueError("prices_panel is required for optimizer-based portfolio methods")
            return self._build_positions_spo(
                score_panel,
                prices_panel,
                robust=(self.cfg.portfolio_method == "robust_spo"),
            )

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
            w.loc[longs] = float(self.cfg.long_budget) / top_n

            if self.cfg.long_short and bottom_n > 0:
                shorts = s.nsmallest(bottom_n).index
                w.loc[shorts] = -float(self.cfg.short_budget) / bottom_n

            if self.cfg.max_weight is not None:
                w = w.clip(lower=-self.cfg.max_weight, upper=self.cfg.max_weight)

            positions.loc[d] = w

        positions = positions.ffill().fillna(0.0)
        self.logger.info(
            f"positions: rebalance_dates={len(rebalance_dates)}, "
            f"avg_longs={(positions > 0).sum(axis=1).mean():.1f}, avg_shorts={(positions < 0).sum(axis=1).mean():.1f}"
        )
        return positions.stack().to_frame("w")

    def _build_positions_proportional(self, score_panel: pd.DataFrame) -> pd.DataFrame:
        alpha_panel = self._select_optimizer_alpha(score_panel)
        alpha = alpha_panel["alpha"].unstack("ticker")
        alpha = self._transform_optimizer_alpha(alpha)
        positions = pd.DataFrame(index=alpha.index, columns=alpha.columns, dtype=float)
        if self.cfg.rebalance_freq <= 0:
            raise ValueError("rebalance_freq must be >= 1")

        rebalance_dates = alpha.index[:: self.cfg.rebalance_freq]
        for d in rebalance_dates:
            s = self._candidate_tickers(alpha.loc[d].dropna())
            if s.empty:
                continue
            if self.cfg.long_short:
                raise ValueError("proportional method currently supports long-only only")
            s = s.clip(lower=0.0)
            if float(s.sum()) <= 0.0:
                continue
            w = s / float(s.sum())
            if self.cfg.max_weight is not None:
                w = self._cap_and_renormalize_long_only(w, float(self.cfg.max_weight), 1.0)
            positions.loc[d] = 0.0
            positions.loc[d, w.index] = w.to_numpy(dtype=float)

        positions = positions.ffill().fillna(0.0)
        self.logger.info(
            f"proportional: rebalance_dates={len(rebalance_dates)}, "
            f"avg_longs={(positions > 0).sum(axis=1).mean():.1f}"
        )
        return positions.stack().to_frame("w")

    def _select_optimizer_alpha(
        self,
        score_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        source = str(getattr(self.cfg, "optimizer_alpha_source", "auto")).strip().lower()
        if source == "calibrated":
            if "score_calibrated" not in score_panel.columns:
                raise ValueError("optimizer_alpha_source='calibrated' requires 'score_calibrated'")
            alpha = score_panel["score_calibrated"].copy()
            self.logger.info("optimizer alpha source=score_calibrated")
        elif source == "raw":
            if "score_raw" not in score_panel.columns:
                raise ValueError("optimizer_alpha_source='raw' requires 'score_raw'")
            alpha = score_panel["score_raw"].copy()
            self.logger.info("optimizer alpha source=score_raw")
        elif source == "score":
            if "score" not in score_panel.columns:
                raise ValueError("optimizer_alpha_source='score' requires 'score'")
            alpha = score_panel["score"].copy()
            self.logger.info("optimizer alpha source=score")
        elif "score_calibrated" in score_panel.columns:
            alpha = score_panel["score_calibrated"].copy()
            self.logger.info("optimizer alpha source=score_calibrated")
        elif "score_raw" in score_panel.columns:
            alpha = score_panel["score_raw"].copy()
            self.logger.info("optimizer alpha source=score_raw")
        elif "score" in score_panel.columns:
            alpha = score_panel["score"].copy()
            self.logger.warning("optimizer alpha source=score (score_raw missing)")
        else:
            raise ValueError("score_panel must contain 'score_calibrated', 'score', or 'score_raw'")
        return alpha.to_frame("alpha")

    @staticmethod
    def _rank_normalize_row(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if s.empty:
            return s
        n = len(s)
        if n == 1:
            return pd.Series(0.0, index=s.index)
        ranks = s.rank(method="average")
        u = (ranks - 0.5) / n
        nd = NormalDist()
        vals = np.array([nd.inv_cdf(float(np.clip(x, 1e-6, 1.0 - 1e-6))) for x in u], dtype=float)
        return pd.Series(vals, index=s.index)

    def _transform_optimizer_alpha(
        self,
        alpha: pd.DataFrame,
    ) -> pd.DataFrame:
        method = str(self.cfg.optimizer_alpha_method).strip().lower()
        if method == "raw":
            return alpha
        if method == "zscore":
            return self._zscore(alpha)
        if method == "rank_normal":
            return alpha.apply(self._rank_normalize_row, axis=1).fillna(0.0)
        raise ValueError("optimizer_alpha_method must be one of: raw, zscore, rank_normal")

    def _compute_uncertainty(
        self,
        alpha_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        window: int,
        k: float,
    ) -> pd.DataFrame:
        """Rolling RMSE of prediction error per ticker, shifted to avoid lookahead."""
        ret = prices_panel["ret_1d"]
        shift_n = max(1, int(self.cfg.signal_lag))
        fwd_ret = ret.groupby(level="ticker").shift(-shift_n)

        df = alpha_panel[["alpha"]].join(fwd_ret.rename("fwd_ret"), how="inner")
        df = df.dropna(subset=["alpha", "fwd_ret"])
        err = df["alpha"] - df["fwd_ret"]

        def _rmse(x: pd.Series) -> float:
            return float(np.sqrt(np.mean(np.square(x.to_numpy()))))

        rmse = err.groupby(level="ticker").rolling(window, min_periods=max(5, window // 4)).apply(_rmse, raw=False)
        rmse = rmse.reset_index(level=0, drop=True).shift(1)
        delta = (k * rmse).to_frame("delta")

        delta = delta.reindex(alpha_panel.index)
        by_date = delta.groupby(level="date")["delta"].transform("median")
        overall_med = float(delta["delta"].median()) if not delta["delta"].isna().all() else 0.0
        delta["delta"] = delta["delta"].fillna(by_date).fillna(overall_med)
        return delta["delta"].unstack("ticker")

    def _estimate_diag_variance(
        self,
        prices_panel: pd.DataFrame,
        tickers: pd.Index,
        rebalance_date: pd.Timestamp,
    ) -> pd.Series:
        ret_log = prices_panel["ret_1d"].unstack("ticker")
        ret = np.exp(ret_log) - 1.0
        cols = pd.Index(tickers)
        hist = ret.loc[ret.index < rebalance_date].reindex(columns=cols)
        hist = hist.tail(int(self.cfg.spo_cov_window))
        var = hist.var(axis=0, ddof=0).replace(0.0, np.nan)
        fallback = float(np.nanvar(hist.to_numpy())) if len(hist) else 1e-4
        if not np.isfinite(fallback) or fallback <= 0.0:
            fallback = 1e-4
        return var.fillna(fallback).clip(lower=1e-8)

    def _candidate_tickers(
        self,
        alpha_day: pd.Series,
    ) -> pd.Series:
        if self.cfg.robust_breadth_n and self.cfg.robust_breadth_n > 0:
            k = int(self.cfg.robust_breadth_n)
            if self.cfg.long_short:
                top = alpha_day.nlargest(k)
                bot = alpha_day.nsmallest(k)
                return pd.concat([top, bot]).groupby(level=0).first()
            return alpha_day.nlargest(k)
        return alpha_day

    @staticmethod
    def _cap_and_renormalize_long_only(
        w: pd.Series,
        max_weight: float,
        target_sum: float,
    ) -> pd.Series:
        w = w.clip(lower=0.0)
        if w.empty or float(w.sum()) <= 0.0:
            return w
        w = w / float(w.sum()) * float(target_sum)
        tol = 1e-10
        for _ in range(len(w) + 1):
            over = w > max_weight + tol
            if not over.any():
                break
            capped_sum = float(w[over].clip(upper=max_weight).sum())
            free = ~over
            free_sum = float(w[free].sum())
            w.loc[over] = max_weight
            remaining = float(target_sum) - capped_sum
            if remaining <= tol or free_sum <= tol:
                break
            w.loc[free] = w.loc[free] / free_sum * remaining
        total = float(w.sum())
        if total > 0.0 and abs(total - float(target_sum)) > 1e-8:
            w = w / total * float(target_sum)
        return w.clip(lower=0.0)

    def _build_positions_diag_mv(
        self,
        score_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.cfg.long_short:
            raise ValueError("diag_mv currently supports long-only only")
        alpha_panel = self._select_optimizer_alpha(score_panel)
        alpha = alpha_panel["alpha"].unstack("ticker")
        alpha = self._transform_optimizer_alpha(alpha)
        positions = pd.DataFrame(index=alpha.index, columns=alpha.columns, dtype=float)
        if self.cfg.rebalance_freq <= 0:
            raise ValueError("rebalance_freq must be >= 1")

        rebalance_dates = alpha.index[:: self.cfg.rebalance_freq]
        target_sum = float(self.cfg.robust_target_leverage if self.cfg.robust_target_leverage is not None else 1.0)
        for d in rebalance_dates:
            s = self._candidate_tickers(alpha.loc[d].dropna())
            if s.empty:
                continue
            var = self._estimate_diag_variance(prices_panel, s.index, d)
            raw = (s / var.reindex(s.index)).replace([np.inf, -np.inf], np.nan).dropna()
            raw = raw.clip(lower=0.0)
            if raw.empty or float(raw.sum()) <= 0.0:
                continue
            w = raw / float(raw.sum()) * target_sum
            if self.cfg.max_weight is not None:
                w = self._cap_and_renormalize_long_only(w, float(self.cfg.max_weight), target_sum)
            positions.loc[d] = 0.0
            positions.loc[d, w.index] = w.to_numpy(dtype=float)

        positions = positions.ffill().fillna(0.0)
        self.logger.info(
            f"diag_mv: rebalance_dates={len(rebalance_dates)}, "
            f"avg_longs={(positions > 0).sum(axis=1).mean():.1f}"
        )
        return positions.stack().to_frame("w")

    def _build_positions_spo(
        self,
        score_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        robust: bool,
    ) -> pd.DataFrame:
        alpha_panel = self._select_optimizer_alpha(score_panel)
        alpha = alpha_panel["alpha"].unstack("ticker")
        alpha = self._transform_optimizer_alpha(alpha)
        dates = alpha.index
        positions = pd.DataFrame(index=dates, columns=alpha.columns, dtype=float)

        if self.cfg.rebalance_freq <= 0:
            raise ValueError("rebalance_freq must be >= 1")

        delta = None
        if robust:
            delta = self._compute_uncertainty(
                alpha_panel=alpha_panel,
                prices_panel=prices_panel,
                window=int(self.cfg.robust_window),
                k=float(self.cfg.robust_uncertainty_k),
            )

        rebalance_dates = dates[:: self.cfg.rebalance_freq]
        w_prev = pd.Series(0.0, index=alpha.columns)
        solver = "OSQP"
        turnover_cap_warned = False

        for d in rebalance_dates:
            s = alpha.loc[d].dropna()
            if s.empty:
                continue
            s = self._candidate_tickers(s)
            tickers = s.index
            mu = s.to_numpy(dtype=float)
            sigma = estimate_covariance(
                prices_panel,
                tickers,
                d,
                window=int(self.cfg.spo_cov_window),
                shrinkage=float(self.cfg.spo_cov_shrinkage),
            )
            if robust and delta is not None:
                delta_d = delta.loc[d, tickers].to_numpy(dtype=float)
                if np.isnan(delta_d).any():
                    finite = delta_d[np.isfinite(delta_d)]
                    fill = float(np.median(finite)) if finite.size else 0.0
                    delta_d = np.nan_to_num(delta_d, nan=fill, posinf=fill, neginf=fill)
            else:
                delta_d = np.zeros(len(tickers), dtype=float)
            w0 = w_prev.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
            if not self.cfg.long_short:
                w0 = np.clip(w0, 0.0, None)
            name = "robust_spo" if robust else "spo"
            turnover_cap = self.cfg.robust_turnover_cap
            w_val, status = solve_spo_day(
                mu,
                sigma,
                w0,
                long_short=bool(self.cfg.long_short),
                risk_aversion=float(self.cfg.spo_risk_aversion),
                lambda_l2=float(self.cfg.robust_lambda_l2),
                turnover_mult=float(self.cfg.robust_turnover_mult),
                cost_bps=float(self.cfg.cost_bps),
                max_weight=(None if self.cfg.max_weight is None else float(self.cfg.max_weight)),
                max_leverage=(None if self.cfg.robust_max_leverage is None else float(self.cfg.robust_max_leverage)),
                min_leverage=float(self.cfg.robust_min_leverage),
                target_leverage=(
                    None if self.cfg.robust_target_leverage is None else float(self.cfg.robust_target_leverage)
                ),
                turnover_cap=(None if turnover_cap is None else float(turnover_cap)),
                penalty_scale_mode=self.cfg.robust_penalty_scale,
                robust_delta=(delta_d if robust else None),
                solver=solver,
                logger=(None if turnover_cap_warned else self.logger),
                name=name,
            )
            if status not in ("optimal", "optimal_inaccurate"):
                self.logger.warning(f"{name}: solve failed on {d} (status={status}); skip rebalance")
                turnover_cap_warned = True
            if not self.cfg.long_short:
                w_val = np.clip(w_val, 0.0, None)
            positions.loc[d] = 0.0
            positions.loc[d, tickers] = w_val
            w_prev = pd.Series(0.0, index=alpha.columns)
            w_prev.loc[tickers] = w_val

        positions = positions.ffill().fillna(0.0)
        name = "robust_spo" if robust else "spo"
        self.logger.info(
            f"{name}: rebalance_dates={len(rebalance_dates)}, "
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
        if not self.cfg.long_short:
            turnover = w.diff().abs().sum(axis=1)
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
        perf["equity"] = self.cfg.initial_capital * (1.0 + perf["net"].fillna(0.0)).cumprod()
        perf["cash_weight"] = 1.0 - w.clip(lower=0.0).sum(axis=1)
        self.logger.info(f"turnover avg={turnover.mean():.4f}, cost avg={cost.mean():.6f}")
        return perf

    def run(
        self,
        score_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        out: Optional[ExperimentPaths] = None,
    ) -> pd.DataFrame:
        return self.run_with_details(score_panel, prices_panel, out=out).performance

    def run_with_details(
        self,
        score_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        out: Optional[ExperimentPaths] = None,
    ) -> BacktestResult:
        """Main entry. Returns perf DataFrame indexed by date."""
        pos = self.build_positions(score_panel, prices_panel=prices_panel)
        perf = self.compute_pnl(pos, prices_panel)

        if out is not None:
            out.tables_dir.mkdir(parents=True, exist_ok=True)
            write_parquet(perf, out.tables_dir / "performance.parquet")
            write_parquet(pos, out.tables_dir / "positions.parquet")
            write_parquet(score_panel, out.tables_dir / "scores.parquet")

        return BacktestResult(performance=perf, positions=pos)
