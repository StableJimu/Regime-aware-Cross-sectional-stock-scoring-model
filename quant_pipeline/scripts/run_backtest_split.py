# -*- coding: utf-8 -*-
"""
Backtest with train/validation split:
- Train for N years
- Validate for M years
Model is trained on train period only.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import argparse
import warnings
import numpy as np
import pandas as pd

from quant_pipeline.pipeline.utils import read_yaml, setup_logger, make_experiment_paths
from quant_pipeline.pipeline.data_loader import DataLoader, DataLoaderConfig
from quant_pipeline.pipeline.factor_processing import FactorProcessor, FactorProcessorConfig, FactorRegistry
from quant_pipeline.pipeline.alpha101_factors import register_alpha101
from quant_pipeline.pipeline.artifacts import read_factor_manifest, write_run_metadata
from quant_pipeline.pipeline.scoring_model import ScoringModel, ScoringModelConfig
from quant_pipeline.pipeline.backtester import Backtester, BacktestConfig
from quant_pipeline.pipeline.masks import all_samples_mask


def _make_run_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _split_dates(
    all_dates: pd.Index,
    train_years: int,
    val_years: int,
    anchor: str = "latest",
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    d0 = pd.to_datetime(all_dates.min())
    d_last = pd.to_datetime(all_dates.max())
    total_years = train_years + val_years

    if anchor not in {"latest", "earliest"}:
        raise ValueError("anchor must be 'latest' or 'earliest'")
    # If the requested window is shorter than the full dataset, anchor to latest or earliest.
    if d0 + pd.DateOffset(years=total_years) - pd.Timedelta(days=1) < d_last:
        if anchor == "latest":
            val_end = d_last
            train_end = val_end - pd.DateOffset(years=val_years)
            start = train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
            return start, train_end, val_end
        train_end = d0 + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        val_end = train_end + pd.DateOffset(years=val_years)
        return d0, train_end, val_end

    # Otherwise, use the earliest available dates.
    train_end = d0 + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
    val_end = train_end + pd.DateOffset(years=val_years)
    return d0, train_end, val_end


def _mask_dates(index: pd.MultiIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    dates = index.get_level_values("date")
    mask = (dates >= start) & (dates <= end)
    return pd.Series(mask, index=index, name="date_mask")


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _gaussian_logpdf_diag(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    T, D = X.shape
    log_det = np.sum(np.log(var), axis=1)
    diff = X[:, None, :] - mu[None, :, :]
    quad = np.sum((diff ** 2) / var[None, :, :], axis=2)
    return -0.5 * (D * np.log(2 * np.pi) + log_det[None, :] + quad)


def _forward_backward(
    logB: np.ndarray,
    logA: np.ndarray,
    logpi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    T, K = logB.shape
    log_alpha = np.zeros((T, K))
    log_beta = np.zeros((T, K))

    log_alpha[0] = logpi + logB[0]
    for t in range(1, T):
        log_alpha[t] = logB[t] + _logsumexp(log_alpha[t - 1][:, None] + logA, axis=0)

    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp(logA + logB[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

    loglik = _logsumexp(log_alpha[-1], axis=0)
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)

    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t][:, None]
            + logA
            + logB[t + 1][None, :]
            + log_beta[t + 1][None, :]
        )
        log_xi_t = log_xi_t - _logsumexp(log_xi_t, axis=None)
        xi_sum += np.exp(log_xi_t)
    return gamma, xi_sum, float(loglik)


def _viterbi(logB: np.ndarray, logA: np.ndarray, logpi: np.ndarray) -> np.ndarray:
    T, K = logB.shape
    dp = np.zeros((T, K))
    ptr = np.zeros((T, K), dtype=int)
    dp[0] = logpi + logB[0]
    for t in range(1, T):
        scores = dp[t - 1][:, None] + logA
        ptr[t] = np.argmax(scores, axis=0)
        dp[t] = logB[t] + np.max(scores, axis=0)
    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = ptr[t + 1, states[t + 1]]
    return states


def _fit_kmeans_fallback(X: np.ndarray, K: int, n_iter: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Small NumPy k-means fallback for environments where sklearn KMeans fails."""
    n_samples = X.shape[0]
    if n_samples < K:
        raise ValueError(f"Not enough samples for k-means init: n_samples={n_samples}, K={K}")

    rng = np.random.default_rng(42)
    centers = X[rng.choice(n_samples, size=K, replace=False)].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(n_iter):
        dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[rng.integers(0, n_samples)]
    return labels, centers


def _fit_hmm(
    X: np.ndarray,
    K: int = 3,
    n_iter: int = 50,
    emit_temp: float = 0.5,
    var_floor: float = 1.0,
    init_diag: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, D = X.shape
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=K, n_init=5, random_state=42)
        labels = km.fit_predict(X)
        mu = km.cluster_centers_
    except Exception:
        labels, mu = _fit_kmeans_fallback(X, K)

    var = np.zeros((K, D))
    for k in range(K):
        xk = X[labels == k]
        if len(xk) == 0:
            var[k] = np.var(X, axis=0) + var_floor
        else:
            var[k] = np.var(xk, axis=0) + var_floor

    pi = np.full(K, 1.0 / K)
    A = np.full((K, K), 1e-3)
    np.fill_diagonal(A, init_diag)
    A = A / A.sum(axis=1, keepdims=True)

    logpi = np.log(pi + 1e-12)
    logA = np.log(A + 1e-12)

    last_ll = -np.inf
    for _ in range(n_iter):
        logB = _gaussian_logpdf_diag(X, mu, var) * emit_temp
        gamma, xi_sum, ll = _forward_backward(logB, logA, logpi)

        pi = gamma[0] / (gamma[0].sum() + 1e-12)
        A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-12)
        mu = (gamma.T @ X) / (gamma.sum(axis=0)[:, None] + 1e-12)
        diff = X[:, None, :] - mu[None, :, :]
        var = (gamma[:, :, None] * (diff ** 2)).sum(axis=0) / (gamma.sum(axis=0)[:, None] + 1e-12)
        var = np.maximum(var, var_floor)

        logpi = np.log(pi + 1e-12)
        logA = np.log(A + 1e-12)

        if ll - last_ll < 1e-4:
            break
        last_ll = ll

    return A, pi, mu, var


def _infer_hmm(
    X: np.ndarray,
    A: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    emit_temp: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    logpi = np.log(pi + 1e-12)
    logA = np.log(A + 1e-12)
    logB = _gaussian_logpdf_diag(X, mu, var) * emit_temp
    gamma, _, _ = _forward_backward(logB, logA, logpi)
    states = _viterbi(logB, logA, logpi)
    return states, gamma


def _fit_market_hmm(
    market_features: pd.DataFrame,
    feature_cols: list[str],
    k_states: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    emit_temp: float,
    var_floor: float,
    hard_prob: float,
    regime_lag: int,
    init_diag: float,
    logger,
) -> tuple[pd.Series, pd.DataFrame]:
    feats = market_features[feature_cols].dropna()
    train_feats = feats.loc[train_start:train_end]
    if train_feats.empty or len(train_feats) < 50:
        raise ValueError("Not enough market features in training window for HMM fit")

    mu = train_feats.mean(axis=0).to_numpy()
    sd = train_feats.std(axis=0).to_numpy() + 1e-8
    X_train = (train_feats.to_numpy() - mu) / sd
    X_all = (feats.to_numpy() - mu) / sd

    A, pi, mu_hmm, var = _fit_hmm(
        X_train,
        K=k_states,
        n_iter=50,
        emit_temp=emit_temp,
        var_floor=var_floor,
        init_diag=init_diag,
    )
    states, gamma = _infer_hmm(X_all, A=A, pi=pi, mu=mu_hmm, var=var, emit_temp=emit_temp)

    prob = gamma
    prob = prob / (prob.sum(axis=1, keepdims=True) + 1e-12)
    max_prob = prob.max(axis=1)
    hard = np.eye(prob.shape[1])[prob.argmax(axis=1)]
    use_hard = (max_prob >= hard_prob)[:, None]
    prob = np.where(use_hard, hard, prob)

    idx = feats.index
    mkt_label = pd.Series(prob.argmax(axis=1), index=idx, name="regime_label")
    mkt_proba = pd.DataFrame(prob, index=idx, columns=[f"regime_{i}" for i in range(prob.shape[1])])

    if regime_lag > 0:
        mkt_proba = mkt_proba.shift(regime_lag).ffill()
        mkt_label = mkt_label.shift(regime_lag)

    mkt_label = mkt_label.reindex(market_features.index).ffill().bfill()
    mkt_proba = mkt_proba.reindex(market_features.index).ffill().bfill()
    logger.info(f"HMM fitted on train window: K={k_states}, hard_share={(max_prob >= hard_prob).mean():.3f}")
    return mkt_label, mkt_proba


def _fit_glm(X: np.ndarray, y: np.ndarray, l2: float = 1e-8) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X1 = np.hstack([np.ones((X.shape[0], 1)), X])
    reg = np.eye(X1.shape[1]) * l2
    reg[0, 0] = 0.0
    w = np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)
    return w[1:], float(w[0])


def _score_with_rolling_glm(
    scorer: ScoringModel,
    factor_panel: pd.DataFrame,
    prices_panel: pd.DataFrame,
    regime_masks: dict[int, pd.Series],
    market_regime_proba: pd.DataFrame,
    window_days: int,
    max_train_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if not scorer.selected_factors_:
        raise RuntimeError("No selected factors. Fit selection first.")

    y = scorer.build_label(prices_panel)
    dates = factor_panel.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()

    out = pd.DataFrame(index=factor_panel.index)
    for k in scorer.selected_factors_.keys():
        out[f"score_regime_{k}"] = np.nan

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

        for k, cols in scorer.selected_factors_.items():
            mask = regime_masks[k] & win_mask_dates
            df = base.loc[mask, cols + ["label"]].dropna()
            if df.empty:
                continue
            X = df[cols].to_numpy()
            yv = df["label"].to_numpy()
            coef, intercept = _fit_glm(X, yv, l2=1e-8)

            idx_day = dates == d
            Xd = factor_panel.loc[idx_day, cols].dropna()
            if Xd.empty:
                continue
            preds = Xd.to_numpy() @ coef + intercept
            out.loc[Xd.index, f"score_regime_{k}"] = preds

    gamma = market_regime_proba.reindex(dates)
    gamma = scorer._normalize_gamma(gamma, len(scorer.selected_factors_))
    score = np.zeros(len(out), dtype=float)
    for k in scorer.selected_factors_.keys():
        gk = gamma.get(f"regime_{k}")
        if gk is None:
            continue
        score += out[f"score_regime_{k}"].fillna(0.0).to_numpy() * gk.to_numpy()
    out["score"] = score
    return out


def _calibrate_scores_linear(
    score_panel: pd.DataFrame,
    prices_panel: pd.DataFrame,
    train_mask: pd.Series,
    label_horizon: int,
    signal_lag: int,
    logger,
) -> pd.DataFrame:
    if "score" not in score_panel.columns:
        return score_panel
    if "ret_1d" not in prices_panel.columns:
        return score_panel

    shift_n = max(1, int(label_horizon) + int(signal_lag) - 1)
    y = prices_panel.groupby(level="ticker")["ret_1d"].shift(-shift_n)
    y.name = "label"

    df = score_panel[["score"]].join(y, how="left")
    if train_mask is not None:
        if train_mask.index.equals(df.index):
            df = df[train_mask.values]
        else:
            df = df[train_mask.reindex(df.index).fillna(False).values]
    df = df.dropna(subset=["score", "label"])
    if df.empty or len(df) < 1000:
        logger.warning("score calibration skipped: insufficient samples")
        return score_panel

    X = df["score"].to_numpy(dtype=float)
    Y = df["label"].to_numpy(dtype=float)
    A = np.column_stack([np.ones_like(X), X])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    logger.info(f"score calibration (linear): a={a:.6g}, b={b:.6g}, n={len(df):,}")

    out = score_panel.copy()
    out["score_raw"] = out["score"]
    out["score_calibrated"] = a + b * out["score_raw"]
    # Keep ranking on raw cross-sectional scores; calibrated values are diagnostic only.
    out["score"] = out["score_raw"]
    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--train-years", type=int, default=5)
    p.add_argument("--val-years", type=int, default=2)
    p.add_argument("--split-anchor", type=str, default="latest", choices=["latest", "earliest"])
    p.add_argument(
        "--stress-level",
        type=str,
        default="high",
        choices=["low", "medium", "high", "all"],
    )
    return p.parse_args()


def main() -> None:
    logger = setup_logger("run_backtest_split")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    np.seterr(all="ignore")
    args = parse_args()
    cfg = read_yaml(Path(args.config))

    run_id = _make_run_id(cfg)
    out = make_experiment_paths(Path(cfg["paths"]["backtest_dir"]), run_id)
    logger.info(f"run_id={run_id}, out={out.run_dir}")

    # 1) Load data
    dl_cfg = DataLoaderConfig(
        project_root=Path(cfg["paths"]["project_root"]),
        raw_root_dir=Path(cfg["paths"]["raw_dir"]),
        universe_path=Path(cfg["paths"]["universe_path"]) if cfg["paths"].get("universe_path") else None,
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        panel_path=Path(cfg["paths"]["panel_path"]) if cfg["paths"].get("panel_path") else None,
        market_index_path=Path(cfg["paths"]["market_index_path"]) if cfg["paths"].get("market_index_path") else None,
        rates_path=Path(cfg["paths"]["rates_path"]) if cfg["paths"].get("rates_path") else None,
        market_proxy_ticker=cfg["data"].get("market_proxy_ticker", "SPY"),
    )
    loader = DataLoader(dl_cfg)
    prices_panel, market_features = loader.load_and_build()
    if "vix_proxy" not in market_features.columns and "mkt_ret_1d" in market_features.columns:
        market_features["vix_proxy"] = (
            market_features["mkt_ret_1d"].pow(2).rolling(20, min_periods=10).mean()
        )

    # 2) Factors from manifest
    registry = FactorRegistry()
    register_alpha101(registry)
    fp_cfg = FactorProcessorConfig(factors_dir=Path(cfg["paths"]["factors_dir"]))
    factor_proc = FactorProcessor(fp_cfg, registry)

    manifest_path = Path(cfg["paths"]["factors_manifest"])
    manifest = read_factor_manifest(manifest_path)
    if not manifest.batches:
        raise ValueError("factors_manifest has no batches")
    factor_panel_raw = pd.concat([pd.read_parquet(p) for p in manifest.batches]).sort_index()
    factor_panel = factor_proc.transform_cross_section(factor_panel_raw)
    factor_lag = int(cfg["model"].get("factor_lag", 0))
    if factor_lag > 0:
        factor_panel = factor_panel.groupby(level="ticker").shift(factor_lag)

    # 3) Split dates
    all_dates = factor_panel.index.get_level_values("date").unique().sort_values()
    start, train_end, val_end = _split_dates(
        all_dates,
        args.train_years,
        args.val_years,
        anchor=args.split_anchor,
    )
    val_start = train_end + pd.Timedelta(days=1)
    logger.info(f"train: {start.date()} -> {train_end.date()} | val: {val_start.date()} -> {val_end.date()}")

    # 4) Market regime from HMM fit on train window
    regime_cfg = cfg.get("regime", {})
    k_states = int(regime_cfg.get("k_states", 3))
    feature_cols = list(regime_cfg.get("feature_cols", ["mkt_ret_1d", "mkt_vol_20d"]))
    emit_temp = float(regime_cfg.get("emit_temp", 0.5))
    var_floor = float(regime_cfg.get("var_floor", 1.0))
    hard_prob = float(regime_cfg.get("hard_prob", 0.99))
    init_diag = float(regime_cfg.get("init_diag", 0.4))
    regime_lag = int(cfg["model"].get("regime_lag_days", 0))
    mkt_label, mkt_proba = _fit_market_hmm(
        market_features=market_features,
        feature_cols=feature_cols,
        k_states=k_states,
        train_start=start,
        train_end=train_end,
        emit_temp=emit_temp,
        var_floor=var_floor,
        hard_prob=hard_prob,
        regime_lag=regime_lag,
        init_diag=init_diag,
        logger=logger,
    )

    # 5) Scoring model fit on train only (regime-aware selection)
    sm_cfg = ScoringModelConfig(
        max_factors_per_regime=cfg["model"]["max_factors_per_regime"],
        label_horizon=cfg["model"]["label_horizon"],
        signal_lag=int(cfg["backtest"].get("signal_lag", 1)),
        rolling_train_days=int(cfg["model"].get("rolling_train_days", 0)),
        factor_lag=int(cfg["model"].get("factor_lag", 0)),
        model_family=cfg["model"]["model_family"],
        ridge_alpha=cfg["model"].get("ridge_alpha", 1.0),
        ic_decay_half_life_days=int(cfg["model"].get("ic_decay_half_life_days", 0)),
        selection_refit_days=int(cfg["model"].get("selection_refit_days", 0)),
        selection_window_days=int(cfg["model"].get("selection_window_days", 252)),
        soft_regime_weights=bool(cfg["model"].get("soft_regime_weights", False)),
        soft_regime_min_weight=float(cfg["model"].get("soft_regime_min_weight", 0.0)),
        soft_regime_min_eff_n=int(cfg["model"].get("soft_regime_min_eff_n", 0)),
    )
    validation_mode = str(cfg["model"].get("validation_mode", "walk_forward")).strip().lower()
    if validation_mode not in {"walk_forward", "frozen"}:
        raise ValueError("model.validation_mode must be 'walk_forward' or 'frozen'")
    score_train_cutoff = train_end if validation_mode == "frozen" else None
    logger.info(f"split validation_mode={validation_mode} (t-1 scoring enforced)")
    scorer = ScoringModel(sm_cfg)
    train_mask = _mask_dates(factor_panel.index, start, train_end)
    dates = factor_panel.index.get_level_values("date")
    use_regime = bool(cfg["model"].get("use_regime", True))
    if use_regime:
        masks = {
            i: pd.Series((mkt_label == i).reindex(dates).fillna(False).to_numpy(), index=factor_panel.index) & train_mask
            for i in range(k_states)
        }
    else:
        masks = {0: train_mask}
    candidate_factors = manifest.selected_factors
    selection_method = cfg["model"].get("selection_method", "regime_stepwise")
    top_n = int(cfg["model"].get("selection_top_n", 10))
    if not use_regime:
        y = scorer.build_label(prices_panel)
        train_dates = dates[train_mask.values].unique()
        selected_global = scorer.selector.select_orthogonal_stepwise(
            factor_panel=factor_panel[candidate_factors],
            label=y,
            regime_dates=train_dates,
            top_n=top_n,
            decay_half_life_days=int(sm_cfg.ic_decay_half_life_days),
        )
        if not selected_global:
            raise ValueError("Global selection returned no factors")
        scorer.fit_regime_models_with_masks(
            factor_panel=factor_panel[selected_global],
            prices_panel=prices_panel,
            regime_masks={0: train_mask},
            regime_factors={0: selected_global},
        )
        masks_full = {0: all_samples_mask(factor_panel)}
        if sm_cfg.rolling_train_days and sm_cfg.rolling_train_days > 0:
            proba = pd.DataFrame({"regime_0": 1.0}, index=factor_panel.index.get_level_values("date").unique())
            if sm_cfg.model_family == "glm":
                score_panel = _score_with_rolling_glm(
                    scorer=scorer,
                    factor_panel=factor_panel,
                    prices_panel=prices_panel,
                    regime_masks=masks_full,
                    market_regime_proba=proba,
                    window_days=sm_cfg.rolling_train_days,
                    max_train_date=score_train_cutoff,
                )
            else:
                score_panel = scorer.score_with_rolling_ridge(
                    factor_panel=factor_panel,
                    prices_panel=prices_panel,
                    regime_masks=masks_full,
                    market_regime_proba=proba,
                    window_days=sm_cfg.rolling_train_days,
                    max_train_date=score_train_cutoff,
                )
        else:
            score_panel = scorer.score(factor_panel, prices_panel, None, None)
    else:
        if selection_method == "global_stepwise_ridge":
            fp_train = factor_panel[train_mask]
            pp_train = prices_panel.loc[(slice(start, train_end), slice(None)), :]
            scorer.fit_regime_models_global_selection(
                factor_panel=fp_train,
                prices_panel=pp_train,
                regime_masks=masks,
                candidate_factors=candidate_factors,
                top_n=top_n,
            )
        else:
            use_soft_selection = bool(
                cfg["model"].get("selection_use_soft_regime", cfg["model"].get("soft_regime_weights", False))
            )
            regime_date_weights = None
            if use_soft_selection:
                train_dates = dates[train_mask.values].unique()
                regime_date_weights = {
                    i: mkt_proba.reindex(train_dates).get(f"regime_{i}")
                    for i in range(k_states)
                }
            scorer.fit_regime_models_with_selection(
                factor_panel=factor_panel,
                prices_panel=prices_panel,
                regime_masks=masks,
                candidate_factors=candidate_factors,
                top_n=top_n,
                regime_date_weights=regime_date_weights,
                soft_min_weight=float(cfg["model"].get("soft_regime_min_weight", 0.0)),
                soft_min_eff_n=int(cfg["model"].get("soft_regime_min_eff_n", 0)),
                fallback_to_hard=True,
            )
        if sm_cfg.rolling_train_days and sm_cfg.rolling_train_days > 0:
            if sm_cfg.model_family == "glm":
                full_masks = {
                    i: pd.Series((mkt_label == i).reindex(dates).fillna(False).to_numpy(), index=factor_panel.index)
                    for i in range(k_states)
                }
                score_panel = _score_with_rolling_glm(
                    scorer=scorer,
                    factor_panel=factor_panel,
                    prices_panel=prices_panel,
                    regime_masks=full_masks,
                    market_regime_proba=mkt_proba,
                    window_days=sm_cfg.rolling_train_days,
                    max_train_date=score_train_cutoff,
                )
            else:
                full_masks = {
                    i: pd.Series((mkt_label == i).reindex(dates).fillna(False).to_numpy(), index=factor_panel.index)
                    for i in range(k_states)
                }
                score_panel = scorer.score_with_rolling_ridge(
                    factor_panel=factor_panel,
                    prices_panel=prices_panel,
                    regime_masks=full_masks,
                    market_regime_proba=mkt_proba,
                    window_days=sm_cfg.rolling_train_days,
                    max_train_date=score_train_cutoff,
                )
        else:
            score_panel = scorer.score(factor_panel, prices_panel, mkt_proba, mkt_label)

    if cfg["model"].get("score_calibration") == "linear":
        score_panel = _calibrate_scores_linear(
            score_panel=score_panel,
            prices_panel=prices_panel,
            train_mask=train_mask,
            label_horizon=sm_cfg.label_horizon,
            signal_lag=sm_cfg.signal_lag,
            logger=logger,
        )

    # 6) Backtest on train/val separately
    out.tables_dir.mkdir(parents=True, exist_ok=True)
    mkt_label.to_frame("regime_label").to_parquet(out.tables_dir / "regime_label.parquet")
    mkt_proba.to_parquet(out.tables_dir / "regime_proba.parquet")
    stress_costs = cfg["backtest"].get("stress_costs_bps", [cfg["backtest"]["cost_bps"]])
    stress_short_fees = cfg["backtest"].get("stress_short_fee_bps", [0.0])
    if len(stress_costs) != len(stress_short_fees):
        raise ValueError("stress_costs_bps and stress_short_fee_bps must have same length")
    if len(stress_costs) == 4:
        level_map = {"low": 1, "medium": 2, "high": 3}
    else:
        level_map = {"low": 0, "medium": 1, "high": 2}

    if args.stress_level == "all":
        pairs = list(zip(stress_costs, stress_short_fees))
    else:
        idx = level_map[args.stress_level]
        pairs = [(stress_costs[idx], stress_short_fees[idx])]

    train_scores = score_panel[train_mask]
    train_prices = prices_panel.loc[(slice(start, train_end), slice(None)), :]
    val_mask = _mask_dates(factor_panel.index, train_end + pd.Timedelta(days=1), val_end)
    val_scores = score_panel[val_mask]
    val_prices = prices_panel.loc[(slice(train_end + pd.Timedelta(days=1), val_end), slice(None)), :]

    for cost_bps, short_fee_bps in pairs:
        suffix = f"cost{int(cost_bps)}_fee{int(short_fee_bps)}"
        bt_cfg = BacktestConfig(
            long_short=cfg["backtest"]["long_short"],
            top_q=cfg["backtest"]["top_q"],
            bottom_q=cfg["backtest"]["bottom_q"],
            holding_period=cfg["backtest"]["holding_period"],
            rebalance_freq=cfg["backtest"]["rebalance_freq"],
            cost_bps=float(cost_bps),
            max_weight=float(cfg["backtest"].get("max_weight", 0.2)),
            signal_lag=int(cfg["backtest"].get("signal_lag", 1)),
            score_norm=cfg["backtest"].get("score_norm"),
            short_fee_bps=float(short_fee_bps),
            portfolio_method=cfg["backtest"].get("portfolio_method", "top_q"),
            robust_window=int(cfg["backtest"].get("robust_window", 60)),
            robust_uncertainty_k=float(cfg["backtest"].get("robust_uncertainty_k", 1.0)),
            robust_lambda_l2=float(cfg["backtest"].get("robust_lambda_l2", 0.5)),
            robust_turnover_mult=float(cfg["backtest"].get("robust_turnover_mult", 1.0)),
            robust_penalty_scale=cfg["backtest"].get("robust_penalty_scale"),
            robust_turnover_cap=cfg["backtest"].get("robust_turnover_cap"),
            robust_max_leverage=float(cfg["backtest"].get("robust_max_leverage", 1.0)),
            robust_min_leverage=float(cfg["backtest"].get("robust_min_leverage", 1.0)),
            robust_target_leverage=(
                None
                if cfg["backtest"].get("robust_target_leverage") is None
                else float(cfg["backtest"].get("robust_target_leverage", 1.0))
            ),
            robust_breadth_n=int(cfg["backtest"].get("robust_breadth_n", 0)),
        )
        bt = Backtester(bt_cfg)
        perf_train = bt.run(train_scores, train_prices, out=None)
        perf_val = bt.run(val_scores, val_prices, out=None)
        pos_train = bt.build_positions(train_scores, prices_panel=train_prices)
        pos_val = bt.build_positions(val_scores, prices_panel=val_prices)
        perf_train.to_parquet(out.tables_dir / f"performance_train_{suffix}.parquet")
        perf_val.to_parquet(out.tables_dir / f"performance_val_{suffix}.parquet")
        nz_eps = 0.0
        pos_train = pos_train.loc[pos_train["w"].abs() > nz_eps]
        pos_val = pos_val.loc[pos_val["w"].abs() > nz_eps]
        pos_train.to_csv(out.tables_dir / f"holdings_train_{suffix}.csv")
        pos_val.to_csv(out.tables_dir / f"holdings_val_{suffix}.csv")
    score_panel.to_parquet(out.tables_dir / "scores.parquet")
    write_run_metadata(
        out.artifacts_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "mode": "split",
            "config_path": str(args.config),
            "factors_manifest": str(manifest.path),
            "selected_factors": manifest.selected_factors,
            "date_range": {
                "start": cfg["data"]["start_date"],
                "end": cfg["data"]["end_date"],
            },
            "train_years": args.train_years,
            "val_years": args.val_years,
            "validation_mode": validation_mode,
            "model_family": cfg["model"]["model_family"],
        },
    )

    logger.info("split backtest finished")


if __name__ == "__main__":
    main()
