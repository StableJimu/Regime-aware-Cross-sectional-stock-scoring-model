from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _gaussian_logpdf_diag(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    t_dim, d_dim = X.shape
    log_det = np.sum(np.log(var), axis=1)
    diff = X[:, None, :] - mu[None, :, :]
    quad = np.sum((diff ** 2) / var[None, :, :], axis=2)
    return -0.5 * (d_dim * np.log(2 * np.pi) + log_det[None, :] + quad)


def _forward_backward(
    logB: np.ndarray,
    logA: np.ndarray,
    logpi: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    t_dim, k_dim = logB.shape
    log_alpha = np.zeros((t_dim, k_dim))
    log_beta = np.zeros((t_dim, k_dim))

    log_alpha[0] = logpi + logB[0]
    for t in range(1, t_dim):
        log_alpha[t] = logB[t] + _logsumexp(log_alpha[t - 1][:, None] + logA, axis=0)

    log_beta[-1] = 0.0
    for t in range(t_dim - 2, -1, -1):
        log_beta[t] = _logsumexp(logA + logB[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

    loglik = _logsumexp(log_alpha[-1], axis=0)
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)

    xi_sum = np.zeros((k_dim, k_dim))
    for t in range(t_dim - 1):
        log_xi_t = log_alpha[t][:, None] + logA + logB[t + 1][None, :] + log_beta[t + 1][None, :]
        wt = 1.0 if weights is None else float(weights[t + 1])
        log_xi_t = log_xi_t - _logsumexp(log_xi_t, axis=None)
        xi_sum += wt * np.exp(log_xi_t)
    return gamma, xi_sum, float(loglik)


def _viterbi(logB: np.ndarray, logA: np.ndarray, logpi: np.ndarray) -> np.ndarray:
    t_dim, k_dim = logB.shape
    dp = np.zeros((t_dim, k_dim))
    ptr = np.zeros((t_dim, k_dim), dtype=int)
    dp[0] = logpi + logB[0]
    for t in range(1, t_dim):
        scores = dp[t - 1][:, None] + logA
        ptr[t] = np.argmax(scores, axis=0)
        dp[t] = logB[t] + np.max(scores, axis=0)
    states = np.zeros(t_dim, dtype=int)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(t_dim - 2, -1, -1):
        states[t] = ptr[t + 1, states[t + 1]]
    return states


def _fit_kmeans_fallback(X: np.ndarray, k_states: int, n_iter: int = 25) -> tuple[np.ndarray, np.ndarray]:
    n_samples = X.shape[0]
    if n_samples < k_states:
        raise ValueError(f"Not enough samples for k-means init: n_samples={n_samples}, K={k_states}")

    rng = np.random.default_rng(42)
    centers = X[rng.choice(n_samples, size=k_states, replace=False)].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(n_iter):
        dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(k_states):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[rng.integers(0, n_samples)]
    return labels, centers


def fit_hmm(
    X: np.ndarray,
    k_states: int = 3,
    n_iter: int = 50,
    emit_temp: float = 0.5,
    var_floor: float = 1.0,
    init_diag: float = 0.4,
    half_life: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k_states, n_init=5, random_state=42)
        labels = km.fit_predict(X)
        mu = km.cluster_centers_
    except Exception:
        labels, mu = _fit_kmeans_fallback(X, k_states)

    var = np.zeros((k_states, X.shape[1]))
    for k in range(k_states):
        xk = X[labels == k]
        var[k] = (np.var(xk, axis=0) if len(xk) else np.var(X, axis=0)) + var_floor

    pi = np.full(k_states, 1.0 / k_states)
    A = np.full((k_states, k_states), 1e-3)
    np.fill_diagonal(A, init_diag)
    A = A / A.sum(axis=1, keepdims=True)

    logpi = np.log(pi + 1e-12)
    logA = np.log(A + 1e-12)

    weights = None
    if half_life is not None and half_life > 0:
        ages = np.arange(X.shape[0] - 1, -1, -1)
        decay = np.log(2.0) / float(max(1, half_life))
        weights = np.exp(-decay * ages)
        weights = weights / weights.sum()

    last_ll = -np.inf
    for _ in range(n_iter):
        logB = _gaussian_logpdf_diag(X, mu, var) * emit_temp
        gamma, xi_sum, ll = _forward_backward(logB, logA, logpi, weights=weights)

        gamma_w = gamma if weights is None else gamma * weights[:, None]
        pi = gamma_w[0] / (gamma_w[0].sum() + 1e-12)
        A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-12)
        mu = (gamma_w.T @ X) / (gamma_w.sum(axis=0)[:, None] + 1e-12)
        diff = X[:, None, :] - mu[None, :, :]
        var = (gamma_w[:, :, None] * (diff ** 2)).sum(axis=0) / (gamma_w.sum(axis=0)[:, None] + 1e-12)
        var = np.maximum(var, var_floor)

        logpi = np.log(pi + 1e-12)
        logA = np.log(A + 1e-12)
        if ll - last_ll < 1e-4:
            break
        last_ll = ll

    return A, pi, mu, var


def infer_hmm(
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


@dataclass(frozen=True)
class MarketRegimeConfig:
    k_states: int = 3
    feature_cols: tuple[str, ...] = ("mkt_ret_1d", "mkt_vol_20d")
    emit_temp: float = 0.5
    var_floor: float = 1.0
    hard_prob: float = 0.99
    init_diag: float = 0.4
    regime_lag_days: int = 0
    half_life: Optional[int] = None
    prob_smooth_days: int = 0
    min_regime_duration: int = 0


def _compute_state_stats(
    market_features: pd.DataFrame,
    feature_cols: list[str],
    label: pd.Series,
    k_states: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    feats = market_features[feature_cols].copy()
    df = feats.join(label.rename("regime_label"), how="inner").loc[start:end]
    if df.empty:
        return pd.DataFrame(index=range(k_states))
    return df.groupby("regime_label")[feature_cols].mean().reindex(range(k_states))


def _stress_order_from_stats(stats: pd.DataFrame) -> list[int]:
    if stats.empty:
        return list(range(len(stats)))
    stress = pd.Series(0.0, index=stats.index, dtype=float)
    positive_cols = [
        "mkt_vol_20d",
        "vix",
        "vix_proxy",
        "vix_ret_5d",
        "vix_z_20",
        "vix_ratio_20",
        "mkt_abs_ret_1d",
        "mkt_abs_ret_5d",
        "mkt_neg_ret_1d",
        "mkt_neg_ret_5d",
        "y2",
        "y2_chg_20d",
    ]
    negative_cols = [
        "mkt_ret_1d",
        "mkt_ret_5d",
        "mkt_ret_20d",
        "mkt_mom_20_5",
        "term_spread",
        "term_spread_chg_20d",
    ]
    for col in positive_cols:
        if col in stats.columns:
            stress = stress.add(stats[col].fillna(stats[col].mean()).astype(float), fill_value=0.0)
    for col in negative_cols:
        if col in stats.columns:
            stress = stress.sub(stats[col].fillna(stats[col].mean()).astype(float), fill_value=0.0)
    ordered = (
        pd.DataFrame({"stress": stress, "_orig": np.arange(len(stats))}, index=stats.index)
        .sort_values(["stress", "_orig"], ascending=[True, True], na_position="last")
        .index.tolist()
    )
    return [int(x) for x in ordered]


def _reorder_hmm_states(
    market_features: pd.DataFrame,
    feature_cols: list[str],
    label: pd.Series,
    proba: pd.DataFrame,
    transition: np.ndarray,
    k_states: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    stats = _compute_state_stats(market_features, feature_cols, label, k_states, train_start, train_end)
    ordered_index = _stress_order_from_stats(stats)
    mapping = {int(old): int(new) for new, old in enumerate(ordered_index)}
    label_ord = label.map(mapping).astype(float)
    proba_ord = proba[[f"regime_{old}" for old in ordered_index]].copy()
    proba_ord.columns = [f"regime_{i}" for i in range(k_states)]
    trans_ord = transition[np.ix_(ordered_index, ordered_index)]
    return label_ord, proba_ord, trans_ord


def _smooth_regime_probabilities(proba: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if window_days <= 1 or proba.empty:
        return proba
    smoothed = proba.rolling(window_days, min_periods=1).mean()
    smoothed = smoothed.div(smoothed.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return smoothed


def _enforce_min_regime_duration(
    label: pd.Series,
    proba: pd.DataFrame,
    min_duration: int,
) -> tuple[pd.Series, pd.DataFrame]:
    if min_duration <= 1 or label.empty:
        return label, proba

    values = label.astype(int).to_numpy(copy=True)
    changed = False
    n_obs = len(values)
    start = 0
    while start < n_obs:
        end = start + 1
        while end < n_obs and values[end] == values[start]:
            end += 1
        run_len = end - start
        if run_len < min_duration:
            prev_state = values[start - 1] if start > 0 else None
            next_state = values[end] if end < n_obs else None
            if prev_state is not None and next_state is not None:
                prev_score = float(proba.iloc[start:end][f"regime_{prev_state}"].mean())
                next_score = float(proba.iloc[start:end][f"regime_{next_state}"].mean())
                fill_state = int(prev_state if prev_score >= next_score else next_state)
            elif prev_state is not None:
                fill_state = int(prev_state)
            elif next_state is not None:
                fill_state = int(next_state)
            else:
                fill_state = int(values[start])
            values[start:end] = fill_state
            changed = True
        start = end

    if not changed:
        return label, proba

    label_out = pd.Series(values, index=label.index, name=label.name, dtype=float)
    proba_out = proba.copy()
    eye = np.eye(proba.shape[1], dtype=float)
    forced = eye[values]
    proba_out.loc[:, :] = forced
    return label_out, proba_out


def fit_market_hmm(
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
    half_life: Optional[int] = None,
    prob_smooth_days: int = 0,
    min_regime_duration: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    feats = market_features[feature_cols].dropna()
    train_feats = feats.loc[train_start:train_end]
    if train_feats.empty or len(train_feats) < 50:
        raise ValueError("Not enough market features in training window for HMM fit")

    mu = train_feats.mean(axis=0).to_numpy()
    sd = train_feats.std(axis=0).to_numpy() + 1e-8
    X_train = (train_feats.to_numpy() - mu) / sd
    X_all = (feats.to_numpy() - mu) / sd

    A, pi, mu_hmm, var = fit_hmm(
        X_train,
        k_states=k_states,
        n_iter=50,
        emit_temp=emit_temp,
        var_floor=var_floor,
        init_diag=init_diag,
        half_life=half_life,
    )
    _, gamma = infer_hmm(X_all, A=A, pi=pi, mu=mu_hmm, var=var, emit_temp=emit_temp)

    prob = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-12)
    max_prob = prob.max(axis=1)
    hard = np.eye(prob.shape[1])[prob.argmax(axis=1)]
    prob = np.where((max_prob >= hard_prob)[:, None], hard, prob)

    idx = feats.index
    mkt_label = pd.Series(prob.argmax(axis=1), index=idx, name="regime_label")
    mkt_proba = pd.DataFrame(prob, index=idx, columns=[f"regime_{i}" for i in range(prob.shape[1])])
    mkt_label, mkt_proba, A = _reorder_hmm_states(
        market_features=market_features,
        feature_cols=feature_cols,
        label=mkt_label,
        proba=mkt_proba,
        transition=A,
        k_states=k_states,
        train_start=train_start,
        train_end=train_end,
    )
    mkt_proba = _smooth_regime_probabilities(mkt_proba, int(prob_smooth_days))
    mkt_label = pd.Series(mkt_proba.to_numpy().argmax(axis=1), index=mkt_proba.index, name="regime_label", dtype=float)
    mkt_label, mkt_proba = _enforce_min_regime_duration(mkt_label, mkt_proba, int(min_regime_duration))

    if regime_lag > 0:
        mkt_proba = mkt_proba.shift(regime_lag).ffill()
        mkt_label = mkt_label.shift(regime_lag)

    mkt_label = mkt_label.reindex(market_features.index).ffill().bfill()
    mkt_proba = mkt_proba.reindex(market_features.index).ffill().bfill()
    logger.info(
        f"HMM fitted on train window: K={k_states}, hard_share={(max_prob >= hard_prob).mean():.3f}, "
        f"prob_smooth_days={int(prob_smooth_days)}, min_regime_duration={int(min_regime_duration)}"
    )
    return mkt_label, mkt_proba


def load_spy_features(mkt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(mkt_path, usecols=["ts_event", "symbol", "close"])
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_event"])
    df["date"] = df["ts_event"].dt.tz_convert(None).dt.normalize()
    df = df.rename(columns={"symbol": "ticker"})
    spy = df[df["ticker"] == "SPY"].copy().sort_values("date")
    spy["ret_1d"] = np.log(spy["close"]).diff()
    spy["ret_5d"] = spy["ret_1d"].rolling(5, min_periods=5).sum()
    spy["neg_ret_5d"] = spy["ret_5d"].clip(upper=0.0)
    spy["vol_60d"] = spy["ret_1d"].rolling(60, min_periods=20).std()
    return spy[["date", "ret_5d", "neg_ret_5d", "vol_60d"]]


def load_vix(vix_path: Path) -> pd.DataFrame:
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


def build_standalone_hmm_features(
    market_index_path: Path,
    rates_path: Path,
    vix_path: Optional[Path] = None,
    use_vix: bool = False,
) -> pd.DataFrame:
    spy = load_spy_features(market_index_path)
    rates = pd.read_csv(rates_path)
    rates["date"] = pd.to_datetime(rates["date"])
    if "DGS2" in rates.columns:
        rates = rates.rename(columns={"DGS2": "y2"})
    if "DGS10" in rates.columns:
        rates = rates.rename(columns={"DGS10": "y10"})
    if "term_spread" not in rates.columns and {"y2", "y10"}.issubset(rates.columns):
        rates["term_spread"] = rates["y10"] - rates["y2"]

    feats = pd.merge(spy, rates[["date", "y2", "term_spread"]], on="date", how="inner")
    if use_vix and vix_path is not None and vix_path.exists():
        vix = load_vix(vix_path)
        feats = pd.merge(feats, vix, on="date", how="inner")
        feats = feats.rename(columns={"vix": "vol_proxy"})
    else:
        feats = feats.rename(columns={"vol_60d": "vol_proxy"})
    return feats.dropna()
