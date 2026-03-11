from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def stabilize_covariance(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance matrix must be square")
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(np.diag(cov))
    cov = (1.0 - shrinkage) * cov + shrinkage * diag
    eig_min = float(np.linalg.eigvalsh(cov).min()) if cov.size else 0.0
    if eig_min < 1e-8:
        cov = cov + np.eye(cov.shape[0]) * (1e-8 - eig_min)
    return cov


def estimate_covariance(
    prices_panel: pd.DataFrame,
    tickers: pd.Index,
    rebalance_date: pd.Timestamp,
    window: int,
    shrinkage: float,
) -> np.ndarray:
    ret_log = prices_panel["ret_1d"].unstack("ticker")
    ret = np.exp(ret_log) - 1.0
    cols = pd.Index(tickers)
    hist = ret.loc[ret.index < rebalance_date].reindex(columns=cols)
    hist = hist.tail(int(window))
    min_hist = max(10, int(window) // 4)
    if len(hist) < min_hist:
        var = float(np.nanvar(hist.to_numpy())) if len(hist) else 1e-4
        if not np.isfinite(var) or var <= 0.0:
            var = 1e-4
        return np.eye(len(tickers)) * var
    hist = hist.fillna(0.0)
    cov = np.cov(hist.to_numpy(), rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.eye(len(tickers)) * float(cov)
    return stabilize_covariance(cov, shrinkage)


def solve_spo_day(
    mu: np.ndarray,
    sigma: np.ndarray,
    w0: np.ndarray,
    *,
    long_short: bool,
    risk_aversion: float,
    lambda_l2: float,
    turnover_mult: float,
    cost_bps: float,
    max_weight: Optional[float],
    max_leverage: Optional[float],
    min_leverage: float,
    target_leverage: Optional[float],
    turnover_cap: Optional[float],
    penalty_scale_mode: Optional[str],
    robust_delta: Optional[np.ndarray] = None,
    solver: str = "OSQP",
    logger=None,
    name: str = "spo",
) -> tuple[np.ndarray, str]:
    try:
        import cvxpy as cp
    except Exception as e:
        raise ImportError("cvxpy is required for optimizer-based portfolio methods") from e

    penalty_scale = 1.0
    if penalty_scale_mode == "score_std":
        penalty_scale = float(np.nanstd(mu))
        if not np.isfinite(penalty_scale) or penalty_scale <= 0.0:
            penalty_scale = 1.0

    w = cp.Variable(len(mu))
    obj = (
        mu @ w
        - float(risk_aversion) * penalty_scale * cp.quad_form(w, sigma)
        - float(lambda_l2) * penalty_scale * cp.sum_squares(w)
        - float(turnover_mult) * penalty_scale * (float(cost_bps) / 10000.0) * cp.norm1(w - w0)
    )
    if robust_delta is not None:
        obj -= cp.sum(cp.multiply(robust_delta, cp.abs(w)))

    constraints = []
    turnover_cap_eff = turnover_cap
    if long_short:
        constraints.append(cp.sum(w) == 0.0)
        if max_leverage is not None:
            constraints.append(cp.norm1(w) <= float(max_leverage))
    else:
        constraints.append(w >= 0.0)
        target = target_leverage
        if target is None:
            target = min(float(max_leverage) if max_leverage is not None else 1.0, 1.0)
        constraints.append(cp.sum(w) <= float(target))
        min_leverage_eff = max(0.0, float(min_leverage))
        if min_leverage_eff > 0.0:
            constraints.append(cp.sum(w) >= min_leverage_eff)
        if turnover_cap_eff is not None:
            prev_gross = float(np.clip(w0, 0.0, None).sum())
            min_turnover = max(0.0, min_leverage_eff - prev_gross)
            if float(turnover_cap_eff) < min_turnover:
                if logger is not None:
                    logger.warning(
                        "%s: robust_turnover_cap is below the minimum required turnover for long-only; "
                        "relaxing the cap to keep the problem feasible",
                        name,
                    )
                turnover_cap_eff = min_turnover
    if max_weight is not None:
        constraints.append(w <= float(max_weight))
        constraints.append(w >= -float(max_weight))
    if turnover_cap_eff is not None:
        constraints.append(cp.norm1(w - w0) <= float(turnover_cap_eff))

    k = min(5, max(0, len(mu) // 2))
    if k > 0:
        w_init = np.zeros(len(mu), dtype=float)
        top_idx = np.argsort(-mu)[:k]
        if long_short:
            bot_idx = np.argsort(mu)[:k]
            w_init[top_idx] = 1.0 / (2.0 * k)
            w_init[bot_idx] = -1.0 / (2.0 * k)
        else:
            w_init[top_idx] = 1.0 / k
        w.value = w_init

    prob = cp.Problem(cp.Maximize(obj), constraints)
    try:
        prob.solve(
            solver=solver,
            verbose=False,
            max_iter=20000,
            eps_abs=1e-5,
            eps_rel=1e-5,
            polish=True,
            warm_start=True,
        )
    except Exception:
        prob.solve(
            solver="SCS",
            verbose=False,
            max_iters=10000,
            eps=1e-5,
            warm_start=True,
        )

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        return np.asarray(w.value, dtype=float), str(prob.status)
    return np.asarray(w0, dtype=float), str(prob.status)
