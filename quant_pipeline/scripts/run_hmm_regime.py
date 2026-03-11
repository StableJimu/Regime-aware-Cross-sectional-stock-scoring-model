# -*- coding: utf-8 -*-
"""
HMM market regime learner (Gaussian emissions, diagonal cov).
Features:
  - SPY log return
  - SPY 20d realized vol (VIX proxy) OR VIX level if provided
  - DGS2 (2y yield)
  - term spread (10y-2y)
Outputs:
  - market_regime_hmm.csv with date, regime, prob_1..prob_K
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import argparse
import numpy as np
import pandas as pd

from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def _gaussian_logpdf_diag(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    T, D = X.shape
    K = mu.shape[0]
    log_det = np.sum(np.log(var), axis=1)
    diff = X[:, None, :] - mu[None, :, :]
    quad = np.sum((diff ** 2) / var[None, :, :], axis=2)
    return -0.5 * (D * np.log(2 * np.pi) + log_det[None, :] + quad)


def _forward_backward(
    logB: np.ndarray,
    logA: np.ndarray,
    logpi: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
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
        wt = 1.0 if weights is None else float(weights[t + 1])
        log_xi_t = log_xi_t - _logsumexp(log_xi_t, axis=None)
        xi_sum += wt * np.exp(log_xi_t)
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


def _fit_hmm(
    X: np.ndarray,
    K: int = 3,
    n_iter: int = 50,
    emit_temp: float = 0.5,
    var_floor: float = 1.0,
    half_life: int = 63,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    T, D = X.shape
    km = KMeans(n_clusters=K, n_init=5, random_state=42)
    labels = km.fit_predict(X)
    mu = km.cluster_centers_

    var = np.zeros((K, D))
    for k in range(K):
        xk = X[labels == k]
        if len(xk) == 0:
            var[k] = np.var(X, axis=0) + var_floor
        else:
            var[k] = np.var(xk, axis=0) + var_floor

    pi = np.full(K, 1.0 / K)
    A = np.full((K, K), 1e-3)
    np.fill_diagonal(A, 0.4)
    A = A / A.sum(axis=1, keepdims=True)

    logpi = np.log(pi + 1e-12)
    logA = np.log(A + 1e-12)

    last_ll = -np.inf
    # Exponential decay weights (recent data higher weight)
    T = X.shape[0]
    ages = np.arange(T - 1, -1, -1)  # 0 for most recent
    decay = np.log(2) / max(1, half_life)
    w = np.exp(-decay * ages)
    w = w / w.sum()
    for _ in range(n_iter):
        logB = _gaussian_logpdf_diag(X, mu, var) * emit_temp
        gamma, xi_sum, ll = _forward_backward(logB, logA, logpi, weights=w)

        # Apply decay weights in M-step
        w_gamma = gamma * w[:, None]
        pi = w_gamma[0] / (w_gamma[0].sum() + 1e-12)
        A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-12)
        mu = (w_gamma.T @ X) / (w_gamma.sum(axis=0)[:, None] + 1e-12)
        diff = X[:, None, :] - mu[None, :, :]
        var = (w_gamma[:, :, None] * (diff ** 2)).sum(axis=0) / (w_gamma.sum(axis=0)[:, None] + 1e-12)
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
) -> Tuple[np.ndarray, np.ndarray]:
    logpi = np.log(pi + 1e-12)
    logA = np.log(A + 1e-12)
    logB = _gaussian_logpdf_diag(X, mu, var) * emit_temp
    gamma, _, _ = _forward_backward(logB, logA, logpi)
    states = _viterbi(logB, logA, logpi)
    return states, gamma


def _load_spy_features(mkt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(mkt_path, usecols=["ts_event", "symbol", "close"])
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_event"])
    df["date"] = df["ts_event"].dt.tz_convert(None).dt.normalize()
    df = df.rename(columns={"symbol": "ticker"})
    spy = df[df["ticker"] == "SPY"].copy()
    spy = spy.sort_values("date")
    spy["ret_1d"] = np.log(spy["close"]).diff()
    spy["ret_5d"] = spy["ret_1d"].rolling(5, min_periods=5).sum()
    spy["neg_ret_5d"] = spy["ret_5d"].clip(upper=0.0)
    spy["vol_60d"] = spy["ret_1d"].rolling(60, min_periods=20).std()
    return spy[["date", "ret_5d", "neg_ret_5d", "vol_60d"]]


def _load_vix(vix_path: Path) -> pd.DataFrame:
    df = pd.read_csv(vix_path)
    if "date" not in df.columns:
        if "DATE" in df.columns:
            df = df.rename(columns={"DATE": "date"})
        elif "observation_date" in df.columns:
            df = df.rename(columns={"observation_date": "date"})
        else:
            raise ValueError("VIX file missing date column")
    if "close" not in df.columns:
        if "VIXCLS" in df.columns:
            df = df.rename(columns={"VIXCLS": "close"})
        else:
            raise ValueError("VIX file missing close/VIXCLS column")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].rename(columns={"close": "vix"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--use-vix", action="store_true", help="Use vix.csv if present")
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--emit-temp", type=float, default=0.5)
    p.add_argument("--var-floor", type=float, default=1.0)
    p.add_argument("--half-life", type=int, default=63)
    p.add_argument("--rolling-window-days", type=int, default=252)
    p.add_argument("--refit-freq-days", type=int, default=21)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("run_hmm_regime")
    cfg = read_yaml(Path(args.config))

    mkt_path = Path(cfg["paths"]["raw_dir"]) / "market_index_panel.csv"
    rates_path = Path(cfg["paths"]["raw_dir"]) / "treasury_yields.csv"
    out_path = Path(cfg["paths"]["regime_hmm_path"])
    vix_path = Path(cfg["paths"]["vix_path"])

    if not mkt_path.exists():
        raise FileNotFoundError(f"market_index_panel.csv not found: {mkt_path}")
    if not rates_path.exists():
        raise FileNotFoundError(f"treasury_yields.csv not found: {rates_path}")

    spy = _load_spy_features(mkt_path)
    rates = pd.read_csv(rates_path)
    rates["date"] = pd.to_datetime(rates["date"])
    if "DGS2" in rates.columns:
        rates = rates.rename(columns={"DGS2": "y2"})
    if "DGS10" in rates.columns:
        rates = rates.rename(columns={"DGS10": "y10"})
    if "term_spread" not in rates.columns and {"y2", "y10"}.issubset(rates.columns):
        rates["term_spread"] = rates["y10"] - rates["y2"]

    feats = pd.merge(spy, rates[["date", "y2", "term_spread"]], on="date", how="inner")

    if args.use_vix and vix_path.exists():
        vix = _load_vix(vix_path)
        feats = pd.merge(feats, vix, on="date", how="inner")
        feats = feats.rename(columns={"vix": "vol_proxy"})
    else:
        feats = feats.rename(columns={"vol_60d": "vol_proxy"})

    feats = feats.dropna()
    X = feats[["ret_5d", "neg_ret_5d", "vol_proxy", "y2", "term_spread"]].to_numpy()
    Xs, mu, sd = _standardize(X)

    if args.rolling_window_days and args.rolling_window_days > 0:
        window = int(args.rolling_window_days)
        refit = max(1, int(args.refit_freq_days))
        T = Xs.shape[0]
        states = np.full(T, -1, dtype=int)
        gamma = np.full((T, args.k), np.nan, dtype=float)
        for end_idx in range(window - 1, T, refit):
            start_idx = max(0, end_idx - window + 1)
            X_train = Xs[start_idx : end_idx + 1]
            A, pi, mu, var = _fit_hmm(
                X_train,
                K=args.k,
                n_iter=args.max_iter,
                emit_temp=args.emit_temp,
                var_floor=args.var_floor,
                half_life=args.half_life,
            )
            seg_end = min(end_idx + refit, T)
            X_seg = Xs[end_idx:seg_end]
            s_seg, g_seg = _infer_hmm(
                X_seg,
                A=A,
                pi=pi,
                mu=mu,
                var=var,
                emit_temp=args.emit_temp,
            )
            states[end_idx:seg_end] = s_seg
            gamma[end_idx:seg_end, :] = g_seg
    else:
        A, pi, mu, var = _fit_hmm(
            Xs,
            K=args.k,
            n_iter=args.max_iter,
            emit_temp=args.emit_temp,
            var_floor=args.var_floor,
            half_life=args.half_life,
        )
        states, gamma = _infer_hmm(
            Xs,
            A=A,
            pi=pi,
            mu=mu,
            var=var,
            emit_temp=args.emit_temp,
        )

    out = pd.DataFrame({"date": feats["date"].values})
    out["regime"] = np.where(states >= 0, states + 1, np.nan)  # 1..K
    for k in range(args.k):
        out[f"prob_{k+1}"] = gamma[:, k]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
