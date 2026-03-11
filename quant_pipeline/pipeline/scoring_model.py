# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:32:49 2026

@author: jimya
"""

# quant_pipeline/pipeline/scoring_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

from .utils import setup_logger


# -----------------------------
# Purpose
# - Within each market regime k:
#     (1) evaluate factors (IC/IR) and select top N
#     (2) fit a base model f_k
#     (3) produce regime-specific score s_hat^{(k)}_{i,t}
# - Combine using regime posterior_toggle gamma[t,k]:
#     s_{i,t} = sum_k gamma[t,k] * s_hat^{(k)}_{i,t}
# - Consumers:
#   - backtester.py uses final score panel
# -----------------------------


@dataclass(frozen=True)
class ScoringModelConfig:
    max_factors_per_regime: int = 10
    label_horizon: int = 1  # e.g., next-day return
    signal_lag: int = 1     # execution lag in days
    rolling_train_days: int = 0  # 0 disables rolling ridge
    factor_lag: int = 0  # lag factor values by N days
    # model choice placeholder
    model_family: str = "ridge"  # "ridge" | "glm" | "lstm"
    ridge_alpha: float = 1.0
    # LSTM params
    lstm_lookback: int = 20
    lstm_hidden_size: int = 32
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    lstm_epochs: int = 5
    lstm_batch_size: int = 256
    lstm_lr: float = 1e-3
    ic_decay_half_life_days: int = 0
    selection_refit_days: int = 0  # 0 disables rolling re-selection
    selection_window_days: int = 252
    soft_regime_weights: bool = False
    soft_regime_min_weight: float = 0.0
    soft_regime_min_eff_n: int = 0
    # stock regime params
    trend_short_window: int = 5
    trend_long_window: int = 20
    neutral_buffer: float = 0.0
    vol_window: int = 20


class FactorSelector:
    """Regime-wise factor selection using IC/IR."""
    def __init__(self, max_factors: int):
        self.max_factors = max_factors
        self.logger = setup_logger(self.__class__.__name__)

    def compute_ic(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute cross-sectional IC for one date; caller aggregates."""
        # TODO: implement per-date spearman correlation
        raise NotImplementedError

    def select(self, factor_panel: pd.DataFrame, label: pd.Series, regime_dates: pd.Index) -> List[str]:
        """Return selected factor names for a given market regime.
        Inputs:
          - factor_panel: MultiIndex [date,ticker], columns=factors
          - label: MultiIndex [date,ticker] aligned series
          - regime_dates: dates belonging to this regime (hard label)
        """
        # TODO: compute regime-conditional IC/IR per factor, rank, pick top N
        raise NotImplementedError

    def _ic_by_date(
        self,
        factor_panel: pd.DataFrame,
        label: pd.Series,
        regime_dates: Optional[pd.Index],
    ) -> pd.DataFrame:
        """Compute per-date Spearman IC for each factor within optional regime dates."""
        df = factor_panel.copy()
        df["label"] = label.reindex(df.index)
        if regime_dates is not None:
            df = df.loc[regime_dates]
        df = df.dropna(subset=["label"])
        factor_cols = [c for c in factor_panel.columns if c != "label"]

        def _ic_one(g: pd.DataFrame) -> pd.Series:
            y = g["label"]
            if y.nunique() < 2:
                return pd.Series(np.nan, index=factor_cols)
            rank_y = y.rank()
            rank_x = g[factor_cols].rank()
            return rank_x.corrwith(rank_y)

        ic = df.groupby(level="date", group_keys=False).apply(_ic_one)
        self.logger.info(f"IC matrix: dates={ic.shape[0]}, factors={ic.shape[1]}")
        return ic

    def select_orthogonal_stepwise(
        self,
        factor_panel: pd.DataFrame,
        label: pd.Series,
        regime_dates: Optional[pd.Index],
        top_n: int = 10,
        min_icir: float = 0.0,
        decay_half_life_days: int = 0,
        date_weights: Optional[pd.Series] = None,
        min_weight: float = 0.0,
        min_effective_n: int = 0,
    ) -> List[str]:
        """Greedy orthogonal stepwise selection using IC time series.
        Uses IC series (length T) per factor to keep compute fast.
        """
        ic = self._ic_by_date(factor_panel, label, regime_dates)
        ic = ic.dropna(how="all")
        if ic.empty:
            return []

        weights = None
        if date_weights is not None:
            w = np.array(date_weights.reindex(ic.index).fillna(0.0).to_numpy(dtype=float), copy=True)
            if min_weight > 0.0:
                w[w < min_weight] = 0.0
            if w.sum() <= 0.0:
                self.logger.warning("stepwise IC: all regime weights are zero after thresholding")
                return []
            if min_effective_n and min_effective_n > 0:
                n_eff = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-12)
                if n_eff < min_effective_n:
                    self.logger.warning(
                        f"stepwise IC: effective_n={n_eff:.1f} below min_effective_n={min_effective_n}"
                    )
                    return []
            weights = w / w.sum()

        if decay_half_life_days and decay_half_life_days > 0:
            idx = np.arange(len(ic), dtype=float)
            decay = np.log(2.0) / float(decay_half_life_days)
            w_decay = np.exp(-decay * (len(ic) - 1 - idx))
            w_decay = w_decay / w_decay.sum()
            if weights is None:
                weights = w_decay
            else:
                w = weights * w_decay
                weights = w / w.sum()
            self._log_ic_decay_stats(ic, weights, decay_half_life_days)
        else:
            self._log_ic_decay_stats(ic, weights if weights is not None else None, None)

        cols = list(ic.columns)
        ic_mat = ic.to_numpy()

        def _score_residual(x: np.ndarray, S: Optional[np.ndarray]) -> float:
            if S is None:
                resid = x
            else:
                mask = np.isfinite(x)
                mask &= np.isfinite(S).all(axis=1)
                if mask.sum() < 20:
                    return -np.inf
                Xs = S[mask]
                ys = x[mask]
                if weights is not None:
                    w = weights[mask]
                    sw = np.sqrt(w)
                    beta, *_ = np.linalg.lstsq(Xs * sw[:, None], ys * sw, rcond=None)
                else:
                    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
                resid = np.full_like(x, np.nan, dtype=float)
                resid[mask] = ys - Xs @ beta
            if weights is not None:
                w = weights[np.isfinite(resid)]
                r = resid[np.isfinite(resid)]
                mean = np.sum(w * r)
                var = np.sum(w * (r - mean) ** 2)
                std = np.sqrt(var) + 1e-8
            else:
                mean = np.nanmean(resid)
                std = np.nanstd(resid) + 1e-8
            return abs(mean / std)

        selected: List[str] = []
        selected_mat: Optional[np.ndarray] = None
        remaining = cols.copy()

        for _ in range(min(top_n, len(remaining))):
            best = None
            best_score = -np.inf
            for c in remaining:
                idx = cols.index(c)
                score = _score_residual(ic_mat[:, idx], selected_mat)
                if score > best_score:
                    best_score = score
                    best = c
            if best is None or best_score < min_icir:
                break
            selected.append(best)
            self.logger.info(f"selected {best} (ICIR={best_score:.4f})")
            remaining.remove(best)
            bidx = cols.index(best)
            x = ic_mat[:, bidx]
            selected_mat = x[:, None] if selected_mat is None else np.column_stack([selected_mat, x])

        return selected

    def _log_ic_decay_stats(
        self,
        ic: pd.DataFrame,
        weights: Optional[np.ndarray],
        half_life_days: Optional[int],
        top_k: int = 5,
    ) -> None:
        if ic.empty:
            return
        cols = list(ic.columns)
        mat = ic.to_numpy()
        stats = []
        for j, c in enumerate(cols):
            x = mat[:, j]
            mask = np.isfinite(x)
            if mask.sum() < 20:
                continue
            if weights is not None:
                w = weights[mask]
                xm = x[mask]
                mean = float(np.sum(w * xm))
                var = float(np.sum(w * (xm - mean) ** 2))
                std = np.sqrt(var) + 1e-8
            else:
                mean = float(np.nanmean(x))
                std = float(np.nanstd(x)) + 1e-8
            icir = mean / std if std > 0 else np.nan
            stats.append((c, mean, icir))
        if not stats:
            return
        stats.sort(key=lambda t: abs(t[2]), reverse=True)
        tag = f"decay_half_life={half_life_days}" if weights is not None else "decay=off"
        self.logger.info(f"IC decay stats ({tag}): top {top_k} by |ICIR|")
        for c, mean, icir in stats[:top_k]:
            self.logger.info(f"  {c}: mean_ic={mean:.4f}, icir={icir:.4f}")


class StockRegimeBuilder:
    """Build stock-level regime one-hot features (vol high/low × trend up/down/neutral)."""
    def __init__(self, cfg: ScoringModelConfig):
        self.cfg = cfg
        self.logger = setup_logger(self.__class__.__name__)

    def build(self, prices_panel: pd.DataFrame) -> pd.DataFrame:
        """Output: MultiIndex [date,ticker], columns=stk_regime_* one-hot."""
        # TODO:
        #  - compute realized vol per ticker with cfg.vol_window
        #  - compute MA crossover signals with neutral buffer
        #  - map to 6 states, then one-hot encode
        raise NotImplementedError


class BaseModel:
    """Abstract base model interface."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GLMRegressor(BaseModel):
    """Gaussian GLM with identity link (equivalent to OLS)."""
    def __init__(self, l2: float = 1e-8):
        self.l2 = float(l2)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.size == 0:
            raise ValueError("Empty training data")
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        reg = np.eye(X1.shape[1]) * self.l2
        reg[0, 0] = 0.0
        w = np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)
        self.intercept_ = float(w[0])
        self.coef_ = w[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


class LinearRidgeModel(BaseModel):
    """Minimal ridge regression with closed-form solution."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.size == 0:
            raise ValueError("Empty training data")
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        reg = np.eye(X1.shape[1])
        reg[0, 0] = 0.0
        w = np.linalg.solve(X1.T @ X1 + self.alpha * reg, X1.T @ y)
        self.intercept_ = float(w[0])
        self.coef_ = w[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


class ScoringModel:
    def __init__(self, cfg: ScoringModelConfig):
        self.cfg = cfg
        self.logger = setup_logger(self.__class__.__name__)
        self.selector = FactorSelector(cfg.max_factors_per_regime)
        self.stock_regime = StockRegimeBuilder(cfg)
        self.models_: Dict[int, BaseModel] = {}
        self.selected_factors_: Dict[int, List[str]] = {}
        self.candidate_factors_: Optional[List[str]] = None

    @staticmethod
    def _normalize_gamma(gamma: pd.DataFrame, k: int) -> pd.DataFrame:
        if gamma is None or gamma.empty:
            return gamma
        g = gamma.copy()
        g = g.fillna(0.0)
        row_sum = g.sum(axis=1)
        zero_mask = row_sum <= 0.0
        if zero_mask.any():
            g.loc[zero_mask, :] = 1.0 / float(k)
            row_sum = g.sum(axis=1)
        g = g.div(row_sum, axis=0)
        return g

    def _fit_model_from_panel(
        self,
        factor_panel: pd.DataFrame,
        labels: pd.Series,
        mask: Optional[pd.Series],
        factor_cols: List[str],
    ) -> Tuple[BaseModel, List[str]]:
        df = factor_panel.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.join(labels, how="left")
        if mask is not None:
            if mask.index.equals(df.index):
                df = df[mask.values]
            else:
                df = df[mask.reindex(df.index).fillna(False).values]
        if df.empty:
            raise ValueError("No samples after masking")

        good_cols = self._filter_factor_columns(df, factor_cols)

        if self.cfg.model_family == "lstm":
            model = TorchLSTMModel(self.cfg, good_cols)
            model.fit_panel(factor_panel[good_cols], labels, mask)
            return model, good_cols

        df = df[good_cols + ["label"]].dropna()
        if df.empty:
            raise ValueError("No training samples after NaN filtering")

        X = df[good_cols].to_numpy()
        yv = df["label"].to_numpy()

        if self.cfg.model_family == "glm":
            model = GLMRegressor()
        elif self.cfg.model_family == "ridge":
            model = LinearRidgeModel(alpha=self.cfg.ridge_alpha)
        else:
            raise ValueError(f"Unsupported model_family={self.cfg.model_family}")
        model.fit(X, yv)
        return model, good_cols

    def _filter_factor_columns(self, df: pd.DataFrame, factor_cols: List[str]) -> List[str]:
        min_non_nan = max(1, int(0.2 * len(df)))
        good_cols = [c for c in factor_cols if df[c].notna().sum() >= min_non_nan]
        if not good_cols:
            raise ValueError("All factor columns are too sparse after NaN filtering")
        return good_cols

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        reg = np.eye(X1.shape[1])
        reg[0, 0] = 0.0
        w = np.linalg.solve(X1.T @ X1 + alpha * reg, X1.T @ y)
        return w[1:], float(w[0])

    def _fit_ridge_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        sw = np.sqrt(np.clip(weights, 0.0, None))
        Xw = X * sw[:, None]
        yw = y * sw
        X1 = np.hstack([np.ones((Xw.shape[0], 1)), Xw])
        reg = np.eye(X1.shape[1])
        reg[0, 0] = 0.0
        w = np.linalg.solve(X1.T @ X1 + alpha * reg, X1.T @ yw)
        return w[1:], float(w[0])

    def build_label(self, prices_panel: pd.DataFrame) -> pd.Series:
        """Build supervised label aligned with factor_panel: MultiIndex [date,ticker].
        Default: next-day return (shift -horizon).
        """
        if "ret_1d" not in prices_panel.columns:
            raise ValueError("prices_panel missing ret_1d")
        h = int(self.cfg.label_horizon)
        lag = int(self.cfg.signal_lag)
        shift_n = max(1, h + lag - 1)
        y = prices_panel.groupby(level="ticker")["ret_1d"].shift(-shift_n)
        y.name = "label"
        return y

    def fit_regime_models(
        self,
        factor_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        market_regime_label: pd.Series,  # index=date, value=int
        sample_mask: Optional[pd.Series] = None,
    ) -> None:
        """Train one model per market regime based on hard labels."""
        # Build label + stock regime
        y = self.build_label(prices_panel)
        df = factor_panel.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.join(y, how="left")
        if sample_mask is not None:
            df = df[sample_mask.reindex(df.index).fillna(False)]
        if df.empty:
            raise ValueError("No samples after masking")

        factor_cols = [c for c in factor_panel.columns if c in df.columns]
        model, good_cols = self._fit_model_from_panel(
            factor_panel=factor_panel,
            labels=y,
            mask=sample_mask,
            factor_cols=factor_cols,
        )
        self.models_ = {0: model}
        self.selected_factors_ = {0: good_cols}

    def fit_regime_models_with_masks(
        self,
        factor_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        regime_masks: Dict[int, pd.Series],
        regime_factors: Dict[int, List[str]],
    ) -> None:
        """Fit one model per regime using explicit masks and factor lists."""
        y = self.build_label(prices_panel)
        models: Dict[int, BaseModel] = {}
        selected: Dict[int, List[str]] = {}

        for k, mask in regime_masks.items():
            if k not in regime_factors:
                raise ValueError(f"Missing factor list for regime {k}")
            model, cols = self._fit_model_from_panel(
                factor_panel=factor_panel[regime_factors[k]],
                labels=y,
                mask=mask,
                factor_cols=regime_factors[k],
            )
            models[k] = model
            selected[k] = cols

        self.models_ = models
        self.selected_factors_ = selected

    def fit_regime_models_with_selection(
        self,
        factor_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        regime_masks: Dict[int, pd.Series],
        candidate_factors: List[str],
        top_n: int = 10,
        regime_date_weights: Optional[Dict[int, pd.Series]] = None,
        soft_min_weight: float = 0.0,
        soft_min_eff_n: int = 0,
        fallback_to_hard: bool = True,
    ) -> None:
        """Fit one model per regime using orthogonal stepwise selection."""
        y = self.build_label(prices_panel)
        models: Dict[int, BaseModel] = {}
        selected: Dict[int, List[str]] = {}
        self.candidate_factors_ = list(candidate_factors)

        dates = factor_panel.index.get_level_values("date")
        for k, mask in regime_masks.items():
            regime_dates = dates[mask.values].unique()
            date_weights = None if regime_date_weights is None else regime_date_weights.get(k)
            cols = self.selector.select_orthogonal_stepwise(
                factor_panel=factor_panel[candidate_factors],
                label=y,
                regime_dates=None if date_weights is not None else regime_dates,
                top_n=top_n,
                decay_half_life_days=int(self.cfg.ic_decay_half_life_days),
                date_weights=date_weights,
                min_weight=soft_min_weight,
                min_effective_n=soft_min_eff_n,
            )
            if not cols and fallback_to_hard and date_weights is not None:
                self.logger.warning(f"soft selection empty for regime {k}; falling back to hard regime dates")
                cols = self.selector.select_orthogonal_stepwise(
                    factor_panel=factor_panel[candidate_factors],
                    label=y,
                    regime_dates=regime_dates,
                    top_n=top_n,
                    decay_half_life_days=int(self.cfg.ic_decay_half_life_days),
                )
            if not cols:
                raise ValueError(f"No factors selected for regime {k}")
            model, good_cols = self._fit_model_from_panel(
                factor_panel=factor_panel[cols],
                labels=y,
                mask=mask,
                factor_cols=cols,
            )
            models[k] = model
            selected[k] = good_cols

        self.models_ = models
        self.selected_factors_ = selected

    def fit_regime_models_global_selection(
        self,
        factor_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        regime_masks: Dict[int, pd.Series],
        candidate_factors: List[str],
        top_n: int = 15,
    ) -> None:
        """Global stepwise selection on train only, then ridge per regime."""
        y = self.build_label(prices_panel)
        self.candidate_factors_ = list(candidate_factors)
        all_dates = factor_panel.index.get_level_values("date").unique()
        selected_global = self.selector.select_orthogonal_stepwise(
            factor_panel=factor_panel[candidate_factors],
            label=y,
            regime_dates=all_dates,
            top_n=top_n,
            decay_half_life_days=int(self.cfg.ic_decay_half_life_days),
        )
        if not selected_global:
            raise ValueError("Global selection returned no factors")

        # force ridge for this path
        prev = self.cfg
        self.cfg = ScoringModelConfig(
            max_factors_per_regime=prev.max_factors_per_regime,
            label_horizon=prev.label_horizon,
            signal_lag=prev.signal_lag,
            model_family="ridge",
            ridge_alpha=prev.ridge_alpha,
        )

        models: Dict[int, BaseModel] = {}
        selected: Dict[int, List[str]] = {}
        for k, mask in regime_masks.items():
            model, cols = self._fit_model_from_panel(
                factor_panel=factor_panel[selected_global],
                labels=y,
                mask=mask,
                factor_cols=selected_global,
            )
            models[k] = model
            selected[k] = cols

        self.models_ = models
        self.selected_factors_ = selected
    def score(
        self,
        factor_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        market_regime_proba: pd.DataFrame,  # index=date, columns=regime_0..K-1
        market_regime_label: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Return score panel: MultiIndex [date,ticker], columns:
           - score (final ensemble)
           - score_regime_k (optional decomposition)
        """
        if not self.models_:
            raise RuntimeError("No fitted models. Call fit_regime_models first.")

        out = pd.DataFrame(index=factor_panel.index)
        for k, model in self.models_.items():
            cols = self.selected_factors_[k]
            valid = factor_panel[cols].notna().all(axis=1)
            if hasattr(model, "predict_panel"):
                preds = model.predict_panel(factor_panel[cols])
            else:
                X = factor_panel.loc[valid, cols].to_numpy()
                preds = pd.Series(np.nan, index=factor_panel.index, dtype=float)
                if len(X) > 0:
                    preds.loc[valid] = model.predict(X)
            out[f"score_regime_{k}"] = preds

        dates = factor_panel.index.get_level_values("date")
        gamma = market_regime_proba.reindex(dates)
        gamma = self._normalize_gamma(gamma, len(self.models_))
        if gamma is None or gamma.empty:
            out["score"] = out["score_regime_0"]
            return out

        score = np.zeros(len(out), dtype=float)
        for k in self.models_.keys():
            gk = gamma.get(f"regime_{k}")
            if gk is None:
                continue
            score += out[f"score_regime_{k}"].fillna(0.0).to_numpy() * gk.to_numpy()
        out["score"] = score
        return out

    def score_with_rolling_ridge(
        self,
        factor_panel: pd.DataFrame,
        prices_panel: pd.DataFrame,
        regime_masks: Dict[int, pd.Series],
        market_regime_proba: pd.DataFrame,
        window_days: int,
        max_train_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Rolling ridge per regime over trailing window_days."""
        if not self.selected_factors_:
            raise RuntimeError("No selected factors. Fit selection first.")
        if self.cfg.selection_refit_days and not self.candidate_factors_:
            raise RuntimeError("selection_refit_days set but candidate_factors_ missing")
        y = self.build_label(prices_panel)
        dates = factor_panel.index.get_level_values("date")
        unique_dates = dates.unique().sort_values()

        out = pd.DataFrame(index=factor_panel.index)
        for k in self.selected_factors_.keys():
            out[f"score_regime_{k}"] = np.nan

        # pre-join for speed
        base = factor_panel.copy()
        base["label"] = y
        train_cutoff = None if max_train_date is None else pd.Timestamp(max_train_date)

        for i, d in enumerate(unique_dates):
            hist_dates = unique_dates[unique_dates < d]
            if train_cutoff is not None:
                hist_dates = hist_dates[hist_dates <= train_cutoff]
            if len(hist_dates) == 0:
                continue
            win_dates = hist_dates[-window_days:]
            win_mask_dates = dates.isin(win_dates)

            allow_selection_refit = train_cutoff is None or d <= train_cutoff
            if allow_selection_refit and self.cfg.selection_refit_days and (i % int(self.cfg.selection_refit_days) == 0):
                sel_hist_dates = hist_dates
                if len(sel_hist_dates) == 0:
                    continue
                sel_win_days = int(self.cfg.selection_window_days or window_days)
                sel_dates = sel_hist_dates[-sel_win_days:]
                sel_mask_dates = dates.isin(sel_dates)
                new_selected: Dict[int, List[str]] = {}
                for k, mask in regime_masks.items():
                    regime_dates = dates[mask.values & sel_mask_dates].unique()
                    cols = self.selector.select_orthogonal_stepwise(
                        factor_panel=factor_panel[self.candidate_factors_],
                        label=y,
                        regime_dates=regime_dates,
                        top_n=int(self.cfg.max_factors_per_regime),
                        decay_half_life_days=int(self.cfg.ic_decay_half_life_days),
                    )
                    if cols:
                        new_selected[k] = cols
                if new_selected:
                    self.selected_factors_.update(new_selected)
                    self.logger.info(
                        f"rolling selection update on {d.date()}: "
                        + ", ".join([f"regime_{k}={len(v)}" for k, v in new_selected.items()])
                    )

            for k, cols in self.selected_factors_.items():
                if self.cfg.soft_regime_weights:
                    w_dates = market_regime_proba.get(f"regime_{k}")
                    if w_dates is None:
                        continue
                    w_dates = w_dates.reindex(win_dates).fillna(0.0)
                    df = base.loc[win_mask_dates, cols + ["label"]].dropna()
                    if df.empty:
                        continue
                    w = np.array(
                        w_dates.reindex(df.index.get_level_values("date")).to_numpy(dtype=float),
                        copy=True,
                    )
                    if self.cfg.soft_regime_min_weight > 0.0:
                        w[w < self.cfg.soft_regime_min_weight] = 0.0
                    keep = w > 0.0
                    if not keep.any():
                        continue
                    w = w[keep]
                    df = df.iloc[np.where(keep)[0]]
                    if self.cfg.soft_regime_min_eff_n and self.cfg.soft_regime_min_eff_n > 0:
                        n_eff = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-12)
                        if n_eff < self.cfg.soft_regime_min_eff_n:
                            mask = regime_masks[k] & win_mask_dates
                            df = base.loc[mask, cols + ["label"]].dropna()
                            if df.empty:
                                continue
                            X = df[cols].to_numpy()
                            yv = df["label"].to_numpy()
                            coef, intercept = self._fit_ridge(X, yv, self.cfg.ridge_alpha)
                        else:
                            X = df[cols].to_numpy()
                            yv = df["label"].to_numpy()
                            coef, intercept = self._fit_ridge_weighted(X, yv, self.cfg.ridge_alpha, w)
                    else:
                        X = df[cols].to_numpy()
                        yv = df["label"].to_numpy()
                        coef, intercept = self._fit_ridge_weighted(X, yv, self.cfg.ridge_alpha, w)
                else:
                    mask = regime_masks[k] & win_mask_dates
                    df = base.loc[mask, cols + ["label"]].dropna()
                    if df.empty:
                        continue
                    X = df[cols].to_numpy()
                    yv = df["label"].to_numpy()
                    coef, intercept = self._fit_ridge(X, yv, self.cfg.ridge_alpha)

                idx_day = dates == d
                Xd = factor_panel.loc[idx_day, cols]
                Xd = Xd.dropna()
                if Xd.empty:
                    continue
                preds = Xd.to_numpy() @ coef + intercept
                out.loc[Xd.index, f"score_regime_{k}"] = preds

        # ensemble
        gamma = market_regime_proba.reindex(dates)
        gamma = self._normalize_gamma(gamma, len(self.selected_factors_))
        score = np.zeros(len(out), dtype=float)
        for k in self.selected_factors_.keys():
            gk = gamma.get(f"regime_{k}")
            if gk is None:
                continue
            score += out[f"score_regime_{k}"].fillna(0.0).to_numpy() * gk.to_numpy()
        out["score"] = score
        return out


class TorchLSTMModel:
    """LSTM model for sequence prediction with Huber loss."""
    def __init__(self, cfg: ScoringModelConfig, factor_cols: List[str]):
        self.cfg = cfg
        self.factor_cols = factor_cols
        self.model_ = None
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.sd_ is None:
            return X
        return (X - self.mu_) / self.sd_

    def _build_sequences(
        self,
        panel: pd.DataFrame,
        labels: Optional[pd.Series],
        mask: Optional[pd.Series],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[Tuple[pd.Timestamp, str]]]:
        lookback = self.cfg.lstm_lookback
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        idx_list: List[Tuple[pd.Timestamp, str]] = []

        for ticker, df_t in panel.groupby(level="ticker"):
            df_t = df_t.sort_index()
            X = df_t[self.factor_cols].to_numpy()
            if labels is not None:
                y = labels.reindex(df_t.index).to_numpy()
            else:
                y = None

            for i in range(lookback - 1, len(df_t)):
                idx = df_t.index[i]
                if mask is not None and not bool(mask.reindex([idx]).fillna(False).iloc[0]):
                    continue
                window = X[i - lookback + 1 : i + 1]
                if np.isnan(window).any():
                    continue
                if y is not None:
                    if np.isnan(y[i]):
                        continue
                X_list.append(window)
                idx_list.append(idx)
                if y is not None:
                    y_list.append(float(y[i]))

        X_out = np.array(X_list, dtype=np.float32)
        y_out = np.array(y_list, dtype=np.float32) if labels is not None else None
        return X_out, y_out, idx_list

    def fit_panel(
        self,
        factor_panel: pd.DataFrame,
        labels: pd.Series,
        mask: Optional[pd.Series],
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as e:
            raise ImportError("PyTorch is required for LSTM model_family") from e

        X, y, _ = self._build_sequences(factor_panel, labels, mask)
        if y is None or len(y) == 0:
            raise ValueError("No training sequences for LSTM")

        # normalize over all samples
        self.mu_ = X.mean(axis=(0, 1), keepdims=True)
        self.sd_ = X.std(axis=(0, 1), keepdims=True) + 1e-8
        Xn = self._normalize(X)

        device = torch.device("cpu")
        dataset = TensorDataset(torch.tensor(Xn), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=self.cfg.lstm_batch_size, shuffle=True)

        model = nn.LSTM(
            input_size=X.shape[2],
            hidden_size=self.cfg.lstm_hidden_size,
            num_layers=self.cfg.lstm_num_layers,
            dropout=self.cfg.lstm_dropout if self.cfg.lstm_num_layers > 1 else 0.0,
            batch_first=True,
        )
        head = nn.Linear(self.cfg.lstm_hidden_size, 1)
        model.to(device)
        head.to(device)

        optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=self.cfg.lstm_lr)
        loss_fn = nn.SmoothL1Loss()  # Huber loss

        model.train()
        head.train()
        for _ in range(self.cfg.lstm_epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                out, _ = model(xb)
                pred = head(out[:, -1, :]).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        self.model_ = (model, head)

    def predict_panel(self, factor_panel: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("LSTM model not fitted")
        try:
            import torch
        except Exception as e:
            raise ImportError("PyTorch is required for LSTM model_family") from e

        X, _, idx = self._build_sequences(factor_panel, labels=None, mask=None)
        if len(idx) == 0:
            return pd.Series(np.nan, index=factor_panel.index)

        Xn = self._normalize(X)
        model, head = self.model_
        model.eval()
        head.eval()

        preds = []
        with torch.no_grad():
            xb = torch.tensor(Xn)
            out, _ = model(xb)
            pred = head(out[:, -1, :]).squeeze(-1).cpu().numpy()
            preds = pred

        out_series = pd.Series(np.nan, index=factor_panel.index, dtype=float)
        for (d, t), v in zip(idx, preds):
            out_series.loc[(d, t)] = float(v)
        return out_series
