from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .alpha101_factors import register_alpha101
from .artifacts import read_factor_manifest, write_run_metadata
from .backtester import Backtester, BacktestConfig, BacktestResult
from .data_loader import DataLoader, DataLoaderConfig
from .factor_batches import compute_factor_batches
from .factor_processing import FactorProcessor, FactorProcessorConfig, FactorRegistry
from .masks import all_samples_mask
from .regime_hmm import build_standalone_hmm_features, fit_hmm, fit_market_hmm, infer_hmm, load_vix
from .score_calibration import BucketCalibrationConfig, calibrate_scores_bucketed
from .scoring_model import ScoringModel, ScoringModelConfig
from .utils import make_experiment_paths, read_yaml


def make_run_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def build_data_loader_config(cfg: dict) -> DataLoaderConfig:
    return DataLoaderConfig(
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


def load_prices_and_market_features(cfg: dict):
    loader = DataLoader(build_data_loader_config(cfg))
    return loader, *loader.load_and_build()


def load_factor_panel(cfg: dict) -> tuple[FactorProcessor, object, pd.DataFrame]:
    registry = FactorRegistry()
    register_alpha101(registry)
    fp_cfg = FactorProcessorConfig(factors_dir=Path(cfg["paths"]["factors_dir"]))
    factor_proc = FactorProcessor(fp_cfg, registry)
    manifest = read_factor_manifest(Path(cfg["paths"]["factors_manifest"]))
    if not manifest.batches:
        raise ValueError("factors_manifest has no batches")
    factor_panel_raw = pd.concat([pd.read_parquet(p) for p in manifest.batches]).sort_index()
    factor_panel = factor_proc.transform_cross_section(factor_panel_raw)
    factor_lag = int(cfg["model"].get("factor_lag", 0))
    if factor_lag > 0:
        factor_panel = factor_panel.groupby(level="ticker").shift(factor_lag)
    return factor_proc, manifest, factor_panel


def load_factor_panel_selected(
    cfg: dict,
    columns: list[str] | None = None,
    dtype: str | None = "float32",
) -> tuple[FactorProcessor, object, pd.DataFrame]:
    registry = FactorRegistry()
    register_alpha101(registry)
    fp_cfg = FactorProcessorConfig(factors_dir=Path(cfg["paths"]["factors_dir"]))
    factor_proc = FactorProcessor(fp_cfg, registry)
    manifest = read_factor_manifest(Path(cfg["paths"]["factors_manifest"]))
    if not manifest.batches:
        raise ValueError("factors_manifest has no batches")

    requested = None if not columns else list(dict.fromkeys(columns))
    frames: list[pd.DataFrame] = []
    for p in manifest.batches:
        if requested is None:
            frame = pd.read_parquet(p)
        else:
            available = pq.ParquetFile(p).schema.names
            cols = [c for c in requested if c in available]
            if not cols:
                continue
            frame = pd.read_parquet(p, columns=cols)
        frames.append(frame)
    if not frames:
        raise ValueError("No factor columns loaded from factor batches")

    factor_panel_raw = pd.concat(frames, axis=1).sort_index()
    factor_panel_raw = factor_panel_raw.loc[:, ~factor_panel_raw.columns.duplicated()]
    factor_panel = factor_proc.transform_cross_section(factor_panel_raw)
    if dtype is not None:
        factor_panel = factor_panel.astype(dtype, copy=False)
    factor_lag = int(cfg["model"].get("factor_lag", 0))
    if factor_lag > 0:
        factor_panel = factor_panel.groupby(level="ticker").shift(factor_lag)
    return factor_proc, manifest, factor_panel


def build_scoring_model_config(cfg: dict) -> ScoringModelConfig:
    return ScoringModelConfig(
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
        min_factor_non_nan_frac=float(cfg["model"].get("min_factor_non_nan_frac", 0.02)),
        min_factor_non_nan_count=int(cfg["model"].get("min_factor_non_nan_count", 1000)),
        regime_family_prior_strength=float(cfg["model"].get("regime_family_prior_strength", 0.0)),
        thin_regime_family_prior_strength=float(cfg["model"].get("thin_regime_family_prior_strength", 0.0)),
        thin_regime_eff_n_threshold=int(cfg["model"].get("thin_regime_eff_n_threshold", 0)),
        thin_regime_top_n=int(cfg["model"].get("thin_regime_top_n", 0)),
    )


def _expand_factor_family_entry(value: str) -> list[str]:
    token = str(value).strip()
    if "-" not in token:
        return [token]
    left, right = token.split("-", 1)
    left = left.strip()
    right = right.strip()
    if not left.startswith("alpha_") or not right.startswith("alpha_"):
        return [token]
    start = int(left.split("_")[1])
    end = int(right.split("_")[1])
    if end < start:
        start, end = end, start
    return [f"alpha_{i:03d}" for i in range(start, end + 1)]


def load_factor_family_map(cfg: dict) -> dict[str, str]:
    path_str = str(cfg["model"].get("regime_factor_family_map_path", "")).strip()
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(cfg["paths"]["project_root"]) / path
    payload = read_yaml(path)
    default_family = str(payload.get("default_family", "residual_cross_section"))
    mapping: dict[str, str] = {}
    for family, entries in dict(payload.get("families", {})).items():
        for entry in entries or []:
            for factor_name in _expand_factor_family_entry(str(entry)):
                mapping[factor_name] = str(family)
    if default_family:
        mapping["__default__"] = default_family
    return mapping


def load_regime_family_priors(cfg: dict) -> dict[int, dict[str, float]]:
    raw = dict(cfg["model"].get("regime_family_priors", {}))
    priors: dict[int, dict[str, float]] = {}
    for key, weights in raw.items():
        if isinstance(key, str) and key.startswith("regime_"):
            regime_id = int(key.split("_")[1])
        else:
            regime_id = int(key)
        priors[regime_id] = {str(fam): float(val) for fam, val in dict(weights or {}).items()}
    return priors


def build_backtest_config(cfg: dict, cost_bps: float, short_fee_bps: float) -> BacktestConfig:
    return BacktestConfig(
        long_short=cfg["backtest"]["long_short"],
        top_q=cfg["backtest"]["top_q"],
        bottom_q=cfg["backtest"]["bottom_q"],
        long_budget=float(cfg["backtest"].get("long_budget", 1.0)),
        short_budget=float(cfg["backtest"].get("short_budget", 1.0 if cfg["backtest"]["long_short"] else 0.0)),
        holding_period=cfg["backtest"]["holding_period"],
        rebalance_freq=cfg["backtest"]["rebalance_freq"],
        cost_bps=float(cost_bps),
        max_weight=float(cfg["backtest"].get("max_weight", 0.2)),
        signal_lag=int(cfg["backtest"].get("signal_lag", 1)),
        score_norm=cfg["backtest"].get("score_norm"),
        short_fee_bps=float(short_fee_bps),
        portfolio_method=cfg["backtest"].get("portfolio_method", "diag_mv"),
        initial_capital=float(cfg["backtest"].get("initial_capital", 1_000_000.0)),
        optimizer_alpha_source=cfg["backtest"].get("optimizer_alpha_source", "auto"),
        optimizer_alpha_method=cfg["backtest"].get("optimizer_alpha_method", "rank_normal"),
        spo_cov_window=int(cfg["backtest"].get("spo_cov_window", 60)),
        spo_risk_aversion=float(cfg["backtest"].get("spo_risk_aversion", 5.0)),
        spo_cov_shrinkage=float(cfg["backtest"].get("spo_cov_shrinkage", 0.2)),
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


def resolve_stress_pairs(cfg: dict, stress_level: str, split_mode: bool = False) -> list[tuple[float, float]]:
    stress_costs = cfg["backtest"].get("stress_costs_bps", [cfg["backtest"]["cost_bps"]])
    stress_short_fees = cfg["backtest"].get("stress_short_fee_bps", [0.0])
    if len(stress_costs) != len(stress_short_fees):
        raise ValueError("stress_costs_bps and stress_short_fee_bps must have same length")

    if stress_level == "all":
        return list(zip(stress_costs, stress_short_fees))

    level_map = {"low": 0, "medium": 1, "high": 2}
    if split_mode and len(stress_costs) == 4:
        level_map = {"low": 0, "medium": 1, "high": 2}
    idx = min(level_map[stress_level], len(stress_costs) - 1)
    return [(stress_costs[idx], stress_short_fees[idx])]


def maybe_calibrate_scores(
    score_panel: pd.DataFrame,
    prices_panel: pd.DataFrame,
    cfg: dict,
    logger,
    label_horizon: int,
    signal_lag: int,
    train_mask: Optional[pd.Series] = None,
    max_train_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if str(cfg["model"].get("score_calibration", "")).strip().lower() not in {"bucketed", "bucket"}:
        return score_panel
    return calibrate_scores_bucketed(
        score_panel=score_panel,
        prices_panel=prices_panel,
        label_horizon=label_horizon,
        signal_lag=signal_lag,
        logger=logger,
        cfg=BucketCalibrationConfig(
            lookback_days=int(cfg["model"].get("score_calibration_lookback_days", 252)),
            n_buckets=int(cfg["model"].get("score_calibration_buckets", 10)),
            min_total_obs=int(cfg["model"].get("score_calibration_min_total_obs", 5000)),
            min_bucket_obs=int(cfg["model"].get("score_calibration_min_bucket_obs", 100)),
        ),
        train_mask=train_mask,
        max_train_date=max_train_date,
    )


def _fit_glm(X: np.ndarray, y: np.ndarray, l2: float = 1e-8) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X1 = np.hstack([np.ones((X.shape[0], 1)), X])
    reg = np.eye(X1.shape[1]) * l2
    reg[0, 0] = 0.0
    w = np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)
    return w[1:], float(w[0])


def score_with_rolling_glm(
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

    for d in unique_dates:
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
            coef, intercept = _fit_glm(df[cols].to_numpy(), df["label"].to_numpy(), l2=1e-8)

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
        if gk is not None:
            score += out[f"score_regime_{k}"].fillna(0.0).to_numpy() * gk.to_numpy()
    out["score"] = score
    return out


def split_dates(
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
    if d0 + pd.DateOffset(years=total_years) - pd.Timedelta(days=1) < d_last:
        if anchor == "latest":
            val_end = d_last
            train_end = val_end - pd.DateOffset(years=val_years)
            start = train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
            return start, train_end, val_end
        train_end = d0 + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        val_end = train_end + pd.DateOffset(years=val_years)
        return d0, train_end, val_end
    train_end = d0 + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
    val_end = train_end + pd.DateOffset(years=val_years)
    return d0, train_end, val_end


def mask_dates(index: pd.MultiIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    dates = index.get_level_values("date")
    return pd.Series((dates >= start) & (dates <= end), index=index, name="date_mask")


def _prepare_regime_feature_frame(
    market_features: pd.DataFrame,
    vix_path: Path | None,
) -> pd.DataFrame:
    feats = market_features.copy()
    if vix_path is not None and vix_path.exists():
        vix = load_vix(vix_path).set_index("date").sort_index()
        feats["vix"] = vix["vix"].reindex(feats.index).ffill().bfill()
    if "vix_proxy" not in feats.columns and "mkt_ret_1d" in feats.columns:
        feats["vix_proxy"] = feats["mkt_ret_1d"].pow(2).rolling(20, min_periods=10).mean()
    if "mkt_ret_1d" in feats.columns:
        feats["mkt_abs_ret_1d"] = feats["mkt_ret_1d"].abs()
        feats["mkt_neg_ret_1d"] = feats["mkt_ret_1d"].clip(upper=0.0)
        feats["mkt_ret_5d"] = feats["mkt_ret_1d"].rolling(5, min_periods=5).sum()
        feats["mkt_ret_20d"] = feats["mkt_ret_1d"].rolling(20, min_periods=10).sum()
        feats["mkt_abs_ret_5d"] = feats["mkt_ret_5d"].abs()
        feats["mkt_neg_ret_5d"] = feats["mkt_ret_5d"].clip(upper=0.0)
        feats["mkt_mom_20_5"] = feats["mkt_ret_20d"] - feats["mkt_ret_5d"]
    if "vix" in feats.columns:
        vix_log = np.log(feats["vix"].clip(lower=1e-6))
        feats["vix_ret_5d"] = vix_log.diff(5)
        vix_mean_20 = feats["vix"].rolling(20, min_periods=10).mean()
        vix_std_20 = feats["vix"].rolling(20, min_periods=10).std()
        feats["vix_z_20"] = (feats["vix"] - vix_mean_20) / (vix_std_20 + 1e-8)
        feats["vix_ratio_20"] = feats["vix"] / (vix_mean_20 + 1e-8)
    if "y2" in feats.columns:
        feats["y2_chg_20d"] = feats["y2"].diff(20)
    if "term_spread" in feats.columns:
        feats["term_spread_chg_20d"] = feats["term_spread"].diff(20)
    return feats


def _fit_market_hmm_detailed(
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
    half_life: int | None = None,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray, pd.DataFrame]:
    feats = market_features[feature_cols].dropna()
    train_feats = feats.loc[train_start:train_end]
    if train_feats.empty or len(train_feats) < 50:
        raise ValueError("Not enough market features in training window for HMM fit")

    mu_scale = train_feats.mean(axis=0).to_numpy()
    sd_scale = train_feats.std(axis=0).to_numpy() + 1e-8
    X_train = (train_feats.to_numpy() - mu_scale) / sd_scale
    X_all = (feats.to_numpy() - mu_scale) / sd_scale

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
    prob_hard = np.where((max_prob >= hard_prob)[:, None], hard, prob)

    idx = feats.index
    mkt_label = pd.Series(prob_hard.argmax(axis=1), index=idx, name="regime_label")
    mkt_proba = pd.DataFrame(prob_hard, index=idx, columns=[f"regime_{i}" for i in range(prob.shape[1])])

    if regime_lag > 0:
        mkt_proba = mkt_proba.shift(regime_lag).ffill()
        mkt_label = mkt_label.shift(regime_lag)

    mkt_label = mkt_label.reindex(market_features.index).ffill().bfill()
    mkt_proba = mkt_proba.reindex(market_features.index).ffill().bfill()
    transition = pd.DataFrame(A, index=[f"regime_{i}" for i in range(k_states)], columns=[f"regime_{i}" for i in range(k_states)])
    return mkt_label, mkt_proba, max_prob, transition


def _order_regime_states(
    market_features: pd.DataFrame,
    feature_cols: list[str],
    label: pd.Series,
    proba: pd.DataFrame,
    transition: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    k_states: int,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feats = market_features[feature_cols].copy()
    df = feats.join(label.rename("regime_label"), how="inner").loc[start:end]
    if df.empty:
        ordered_stats = pd.DataFrame(index=range(k_states))
        return label, proba, transition, ordered_stats

    stats = df.groupby("regime_label")[feature_cols].mean().reindex(range(k_states))
    sort_cols = [c for c in ["mkt_vol_20d", "vix", "vix_proxy", "mkt_ret_1d"] if c in stats.columns]
    if not sort_cols:
        sort_cols = list(stats.columns)
    ordered_index = (
        stats.assign(_orig=np.arange(len(stats)))
        .sort_values(sort_cols + ["_orig"], ascending=[True] * len(sort_cols) + [True], na_position="last")
        .index.tolist()
    )
    mapping = {int(old): int(new) for new, old in enumerate(ordered_index)}
    label_ord = label.map(mapping)
    prob_cols = [f"regime_{i}" for i in range(k_states)]
    proba_ord = proba[[f"regime_{old}" for old in ordered_index]].copy()
    proba_ord.columns = prob_cols
    trans_ord = transition.loc[[f"regime_{old}" for old in ordered_index], [f"regime_{old}" for old in ordered_index]].copy()
    trans_ord.index = prob_cols
    trans_ord.columns = prob_cols
    stats_ord = stats.loc[ordered_index].copy()
    stats_ord.index = range(k_states)
    return label_ord, proba_ord, trans_ord, stats_ord


def _regime_slice_metrics(
    label: pd.Series,
    proba: pd.DataFrame,
    max_prob: pd.Series,
    transition: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    k_states: int,
    hard_prob: float,
) -> dict:
    label_s = label.loc[start:end].dropna()
    proba_s = proba.loc[start:end].dropna(how="all")
    max_prob_s = max_prob.loc[start:end].dropna()
    if label_s.empty or proba_s.empty:
        return {}
    label_change_rate = float((label_s != label_s.shift(1)).iloc[1:].mean()) if len(label_s) > 1 else 0.0
    prob_delta_l1 = proba_s.diff().abs().sum(axis=1)
    occupancy = label_s.value_counts(normalize=True).reindex(range(k_states), fill_value=0.0)
    return {
        "n_days": int(len(label_s)),
        "hard_share": float(max_prob_s.ge(float(hard_prob)).mean()),
        "mean_max_prob": float(max_prob_s.mean()),
        "label_change_rate": label_change_rate,
        "mean_prob_delta_l1": float(prob_delta_l1.iloc[1:].mean()) if len(prob_delta_l1) > 1 else 0.0,
        "self_transition_mean": float(np.mean(np.diag(transition.to_numpy()))),
        "transition_entropy": float(
            -np.mean(np.sum(transition.to_numpy() * np.log(np.clip(transition.to_numpy(), 1e-12, 1.0)), axis=1))
        ),
        "occupancy_min": float(occupancy.min()),
        "occupancy_max": float(occupancy.max()),
        "occupancy_gap": float(occupancy.max() - occupancy.min()),
        **{f"occupancy_{i}": float(occupancy.iloc[i]) for i in range(k_states)},
    }


def _regime_structure_score(row: dict, k_states: int) -> float:
    val_occ = [float(row.get(f"val_occupancy_{i}", 0.0)) for i in range(k_states)]
    if not val_occ:
        return -1.0
    balance = 1.0 - float(row.get("val_occupancy_gap", 1.0))
    threshold = 1.0 - abs(float(row.get("val_hard_share", 0.0)) - float(row.get("train_hard_share", 0.0)))
    smooth = 1.0 - abs(float(row.get("val_label_change_rate", 1.0)) - float(row.get("train_label_change_rate", 1.0)))
    if k_states >= 4:
        targets = [0.35, 0.40, 0.18, 0.07]
        tolerances = [0.25, 0.25, 0.15, 0.07]
        shape = [
            1.0 - min(abs(val_occ[i] - targets[i]) / tolerances[i], 1.0)
            for i in range(4)
        ]
        tail_presence = min((val_occ[2] + val_occ[3]) / 0.12, 1.0)
        return (
            0.18 * shape[0]
            + 0.22 * shape[1]
            + 0.18 * shape[2]
            + 0.12 * shape[3]
            + 0.15 * tail_presence
            + 0.10 * balance
            + 0.03 * threshold
            + 0.02 * smooth
        )
    low_occ = val_occ[0]
    mid_occ = val_occ[1] if k_states > 1 else 0.0
    high_occ = val_occ[2] if k_states > 2 else 0.0
    crisis_presence = min(high_occ / 0.10, 1.0)
    low_presence = min(low_occ / 0.10, 1.0)
    mid_target = 1.0 - min(abs(mid_occ - 0.50) / 0.50, 1.0)
    return 0.25 * crisis_presence + 0.20 * low_presence + 0.20 * mid_target + 0.20 * balance + 0.10 * threshold + 0.05 * smooth


def run_regime_parameter_sweep_workflow(
    cfg: dict,
    logger,
    train_years: int,
    val_years: int,
    split_anchor: str = "latest",
    parameter_sets: list[dict] | None = None,
) -> Path:
    sweep_cfg = json.loads(json.dumps(cfg))
    base_backtest_dir = Path(sweep_cfg["paths"]["backtest_dir"])
    sweep_cfg["paths"]["backtest_dir"] = str(base_backtest_dir / "regime_sweeps")
    if parameter_sets is None:
        parameter_sets = []
    run_id = make_run_id(sweep_cfg | {"regime_parameter_sets": parameter_sets, "mode": "regime_sweep"})
    out = make_experiment_paths(Path(sweep_cfg["paths"]["backtest_dir"]), run_id)
    logger.info(f"regime sweep run_id={run_id}, out={out.run_dir}")

    _, _, market_features = load_prices_and_market_features(sweep_cfg)
    vix_path = Path(sweep_cfg["paths"].get("vix_path", ""))
    market_features = _prepare_regime_feature_frame(market_features, vix_path if vix_path.exists() else None)
    regime_cfg = sweep_cfg.get("regime", {})
    macro_full = [
        c for c in [
            "mkt_ret_1d",
            "mkt_ret_5d",
            "mkt_ret_20d",
            "mkt_mom_20_5",
            "mkt_abs_ret_1d",
            "mkt_abs_ret_5d",
            "mkt_neg_ret_1d",
            "mkt_neg_ret_5d",
            "vix",
            "vix_ret_5d",
            "vix_z_20",
            "vix_ratio_20",
            "y2",
            "y2_chg_20d",
            "term_spread",
            "term_spread_chg_20d",
        ]
        if c in market_features.columns
    ]
    feature_sets = [
        {"feature_set": "rv_only", "feature_cols": ["mkt_vol_20d"]},
        {"feature_set": "vix_only", "feature_cols": ["vix"] if "vix" in market_features.columns else ["vix_proxy"]},
        {"feature_set": "macro_full", "feature_cols": macro_full},
    ]
    all_feature_cols = sorted({c for fs in feature_sets for c in fs["feature_cols"]})
    feats = market_features[all_feature_cols].dropna()
    all_dates = feats.index.unique().sort_values()
    start, train_end, val_end = split_dates(all_dates, train_years, val_years, anchor=split_anchor)
    val_start = train_end + pd.Timedelta(days=1)
    regime_lag = int(sweep_cfg["model"].get("regime_lag_days", 0))
    k_states = int(regime_cfg.get("k_states", 3))

    if not parameter_sets:
        parameter_sets = [
            {
                "name": "baseline",
                "emit_temp": float(regime_cfg.get("emit_temp", 0.9)),
                "var_floor": float(regime_cfg.get("var_floor", 1.0)),
                "hard_prob": float(regime_cfg.get("hard_prob", 0.99)),
                "init_diag": float(regime_cfg.get("init_diag", 0.7)),
            },
            {"name": "temp_down_thresh_down", "emit_temp": 0.8, "var_floor": 1.0, "hard_prob": 0.97, "init_diag": 0.75},
            {"name": "temp_up_thresh_down", "emit_temp": 1.0, "var_floor": 1.0, "hard_prob": 0.97, "init_diag": 0.75},
            {"name": "temp_down_var_down", "emit_temp": 0.85, "var_floor": 0.75, "hard_prob": 0.98, "init_diag": 0.75},
            {"name": "temp_up_var_up", "emit_temp": 1.1, "var_floor": 1.25, "hard_prob": 0.98, "init_diag": 0.8},
        ]

    rows: list[dict] = []
    out.tables_dir.mkdir(parents=True, exist_ok=True)
    for feature_set in feature_sets:
        feature_cols = feature_set["feature_cols"]
        for params in parameter_sets:
            name = f"{feature_set['feature_set']}__{params['name']}"
            label, proba, max_prob_raw, transition = _fit_market_hmm_detailed(
                market_features=market_features,
                feature_cols=feature_cols,
                k_states=k_states,
                train_start=start,
                train_end=train_end,
                emit_temp=float(params["emit_temp"]),
                var_floor=float(params["var_floor"]),
                hard_prob=float(params["hard_prob"]),
                regime_lag=regime_lag,
                init_diag=float(params["init_diag"]),
                half_life=int(regime_cfg.get("half_life", 0)) or None,
            )
            label, proba, transition, state_stats = _order_regime_states(
                market_features=market_features,
                feature_cols=feature_cols,
                label=label,
                proba=proba,
                transition=transition,
                start=start,
                end=train_end,
                k_states=k_states,
            )
            max_prob = pd.Series(max_prob_raw, index=market_features[feature_cols].dropna().index, name="max_prob")
            max_prob = max_prob.reindex(market_features.index).ffill().bfill()
            proba_out = proba.copy()
            proba_out["regime_label"] = label
            proba_out["max_prob"] = max_prob
            proba_out.to_parquet(out.tables_dir / f"regime_{name}.parquet")
            transition.to_csv(out.tables_dir / f"transition_{name}.csv")
            state_stats.to_csv(out.tables_dir / f"state_stats_{name}.csv")

            train_metrics = _regime_slice_metrics(
                label, proba, max_prob, transition, start, train_end, k_states, float(params["hard_prob"])
            )
            val_metrics = _regime_slice_metrics(
                label, proba, max_prob, transition, val_start, val_end, k_states, float(params["hard_prob"])
            )
            row = {
                "name": name,
                "feature_set": feature_set["feature_set"],
                "feature_cols": ",".join(feature_cols),
                "train_start": str(start.date()),
                "train_end": str(train_end.date()),
                "val_start": str(val_start.date()),
                "val_end": str(val_end.date()),
                **{k: v for k, v in params.items() if k != "name"},
            }
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["smoothness_score"] = (
                1.0
                - abs(row.get("val_label_change_rate", 1.0) - row.get("train_label_change_rate", 1.0))
                - abs(row.get("val_mean_prob_delta_l1", 1.0) - row.get("train_mean_prob_delta_l1", 1.0))
            )
            row["threshold_score"] = 1.0 - abs(row.get("val_hard_share", 0.0) - row.get("train_hard_share", 0.0))
            row["balance_score"] = 1.0 - row.get("val_occupancy_gap", 1.0)
            row["structure_score"] = _regime_structure_score(row, k_states)
            for i in range(k_states):
                for col in feature_cols:
                    row[f"train_state{i}_{col}"] = float(state_stats.loc[i, col]) if i in state_stats.index and col in state_stats.columns else np.nan
            rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["structure_score", "balance_score", "threshold_score", "val_mean_max_prob"],
        ascending=[False, False, False, False],
    )
    summary.to_csv(out.tables_dir / "summary.csv", index=False)
    write_run_metadata(
        out.artifacts_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "mode": "regime_sweep",
            "config_path": str(sweep_cfg.get("_config_path", "")),
            "train_years": train_years,
            "val_years": val_years,
            "split_anchor": split_anchor,
            "feature_sets": feature_sets,
            "parameter_sets": parameter_sets,
        },
    )
    return out.run_dir


def run_factor_batch_workflow(cfg: dict, logger, force: bool = False) -> Path:
    return compute_factor_batches(cfg, logger=logger, force=force)


def run_standalone_hmm_workflow(
    cfg: dict,
    logger,
    k_states: int,
    use_vix: bool,
    max_iter: int,
    emit_temp: float,
    var_floor: float,
    half_life: int,
    rolling_window_days: int,
    refit_freq_days: int,
) -> Path:
    mkt_path = Path(cfg["paths"].get("market_index_path") or (Path(cfg["paths"]["raw_dir"]) / "market_index_panel.csv"))
    rates_path = Path(cfg["paths"].get("rates_path") or (Path(cfg["paths"]["raw_dir"]) / "treasury_yields.csv"))
    out_path = Path(cfg["paths"]["regime_hmm_path"])
    vix_path = Path(cfg["paths"]["vix_path"])

    feats = build_standalone_hmm_features(mkt_path, rates_path, vix_path=vix_path, use_vix=use_vix)
    X = feats[["ret_5d", "neg_ret_5d", "vol_proxy", "y2", "term_spread"]].to_numpy()
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True) + 1e-8
    Xs = (X - mu) / sd

    if rolling_window_days and rolling_window_days > 0:
        window = int(rolling_window_days)
        refit = max(1, int(refit_freq_days))
        t_dim = Xs.shape[0]
        states = np.full(t_dim, -1, dtype=int)
        gamma = np.full((t_dim, k_states), np.nan, dtype=float)
        for end_idx in range(window - 1, t_dim, refit):
            start_idx = max(0, end_idx - window + 1)
            X_train = Xs[start_idx : end_idx + 1]
            A, pi, mu_fit, var = fit_hmm(
                X_train,
                k_states=k_states,
                n_iter=max_iter,
                emit_temp=emit_temp,
                var_floor=var_floor,
                half_life=half_life,
            )
            seg_end = min(end_idx + refit, t_dim)
            s_seg, g_seg = infer_hmm(Xs[end_idx:seg_end], A=A, pi=pi, mu=mu_fit, var=var, emit_temp=emit_temp)
            states[end_idx:seg_end] = s_seg
            gamma[end_idx:seg_end, :] = g_seg
    else:
        A, pi, mu_fit, var = fit_hmm(
            Xs,
            k_states=k_states,
            n_iter=max_iter,
            emit_temp=emit_temp,
            var_floor=var_floor,
            half_life=half_life,
        )
        states, gamma = infer_hmm(Xs, A=A, pi=pi, mu=mu_fit, var=var, emit_temp=emit_temp)

    out = pd.DataFrame({"date": feats["date"].values})
    out["regime"] = np.where(states >= 0, states + 1, np.nan)
    for k in range(k_states):
        out[f"prob_{k+1}"] = gamma[:, k]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")
    return out_path


def run_full_backtest_workflow(cfg: dict, logger, stress_level: str) -> Path:
    run_id = make_run_id(cfg)
    out = make_experiment_paths(Path(cfg["paths"]["backtest_dir"]), run_id)
    logger.info(f"run_id={run_id}, out={out.run_dir}")

    _, prices_panel, market_features = load_prices_and_market_features(cfg)
    _, manifest, factor_panel = load_factor_panel(cfg)

    hmm = pd.read_csv(Path(cfg["paths"]["regime_hmm_path"]))
    hmm["date"] = pd.to_datetime(hmm["date"])
    hmm = hmm.set_index("date").sort_index().reindex(market_features.index).ffill().bfill()
    prob_cols = [c for c in hmm.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("HMM regime file must contain probability columns named prob_1..prob_K")
    prob = hmm[prob_cols].to_numpy()
    prob = prob / (prob.sum(axis=1, keepdims=True) + 1e-12)
    max_prob = prob.max(axis=1)
    hard = np.eye(prob.shape[1])[prob.argmax(axis=1)]
    prob = np.where((max_prob >= 0.99)[:, None], hard, prob)
    mkt_label = pd.Series(prob.argmax(axis=1), index=market_features.index, name="regime_label")
    mkt_proba = pd.DataFrame(prob, index=market_features.index, columns=[f"regime_{i}" for i in range(prob.shape[1])])
    regime_lag = int(cfg["model"].get("regime_lag_days", 0))
    if regime_lag > 0:
        mkt_proba = mkt_proba.shift(regime_lag).ffill()
        mkt_label = mkt_label.shift(regime_lag)

    sm_cfg = build_scoring_model_config(cfg)
    scorer = ScoringModel(sm_cfg)
    factor_family_map = load_factor_family_map(cfg)
    factor_family_map.pop("__default__", None)
    regime_family_priors = load_regime_family_priors(cfg)
    dates = factor_panel.index.get_level_values("date")
    use_regime = bool(cfg["model"].get("use_regime", True))
    k_states = prob.shape[1]
    if use_regime:
        masks = {
            i: pd.Series((mkt_label == i).reindex(dates).fillna(False).to_numpy(), index=factor_panel.index)
            for i in range(k_states)
        }
    else:
        masks = {0: all_samples_mask(factor_panel)}

    candidate_factors = manifest.selected_factors
    top_n = int(cfg["model"].get("selection_top_n", 10))
    selection_method = cfg["model"].get("selection_method", "regime_stepwise")
    if not use_regime:
        y = scorer.build_label(prices_panel)
        all_dates = factor_panel.index.get_level_values("date").unique()
        selected_global = scorer.selector.select_orthogonal_stepwise(
            factor_panel=factor_panel[candidate_factors],
            label=y,
            regime_dates=all_dates,
            top_n=top_n,
            decay_half_life_days=int(sm_cfg.ic_decay_half_life_days),
        )
        scorer.fit_regime_models_with_masks(
            factor_panel=factor_panel[selected_global],
            prices_panel=prices_panel,
            regime_masks=masks,
            regime_factors={0: selected_global},
        )
        if sm_cfg.rolling_train_days and sm_cfg.rolling_train_days > 0:
            proba = pd.DataFrame({"regime_0": 1.0}, index=all_dates)
            score_panel = scorer.score_with_rolling_ridge(
                factor_panel=factor_panel,
                prices_panel=prices_panel,
                regime_masks=masks,
                market_regime_proba=proba,
                window_days=sm_cfg.rolling_train_days,
            )
        else:
            score_panel = scorer.score(factor_panel, prices_panel, None, None)
    else:
        if selection_method == "global_stepwise_ridge":
            scorer.fit_regime_models_global_selection(
                factor_panel=factor_panel,
                prices_panel=prices_panel,
                regime_masks=masks,
                candidate_factors=candidate_factors,
                top_n=top_n,
            )
        else:
            scorer.fit_regime_models_with_selection(
                factor_panel=factor_panel,
                prices_panel=prices_panel,
                regime_masks=masks,
                candidate_factors=candidate_factors,
                top_n=top_n,
                factor_family_map=factor_family_map,
                regime_family_priors=regime_family_priors,
                thin_regime_eff_n_threshold=int(cfg["model"].get("thin_regime_eff_n_threshold", 0)),
                thin_regime_top_n=int(cfg["model"].get("thin_regime_top_n", 0)),
            )
        if sm_cfg.rolling_train_days and sm_cfg.rolling_train_days > 0:
            score_panel = scorer.score_with_rolling_ridge(
                factor_panel=factor_panel,
                prices_panel=prices_panel,
                regime_masks=masks,
                market_regime_proba=mkt_proba,
                window_days=sm_cfg.rolling_train_days,
            )
        else:
            score_panel = scorer.score(factor_panel, prices_panel, mkt_proba, mkt_label)

    score_panel = maybe_calibrate_scores(
        score_panel,
        prices_panel,
        cfg,
        logger,
        label_horizon=sm_cfg.label_horizon,
        signal_lag=sm_cfg.signal_lag,
    )

    for cost_bps, short_fee_bps in resolve_stress_pairs(cfg, stress_level):
        bt = Backtester(build_backtest_config(cfg, cost_bps, short_fee_bps))
        result = bt.run_with_details(score_panel, prices_panel, out=None)
        suffix = f"cost{int(cost_bps)}_fee{int(short_fee_bps)}"
        out.tables_dir.mkdir(parents=True, exist_ok=True)
        result.performance.to_parquet(out.tables_dir / f"performance_{suffix}.parquet")

    write_run_metadata(
        out.artifacts_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "mode": "full",
            "config_path": str(cfg.get("_config_path", "")),
            "factors_manifest": str(manifest.path),
            "selected_factors": manifest.selected_factors,
            "date_range": {"start": cfg["data"]["start_date"], "end": cfg["data"]["end_date"]},
            "model_family": cfg["model"]["model_family"],
        },
    )
    return out.run_dir


def run_split_backtest_workflow(
    cfg: dict,
    logger,
    train_years: int,
    val_years: int,
    split_anchor: str,
    stress_level: str,
) -> Path:
    run_id = make_run_id(cfg)
    out = make_experiment_paths(Path(cfg["paths"]["backtest_dir"]), run_id)
    logger.info(f"run_id={run_id}, out={out.run_dir}")

    _, prices_panel, market_features = load_prices_and_market_features(cfg)
    vix_path = Path(str(cfg["paths"].get("vix_path", "")).strip()) if cfg["paths"].get("vix_path") else None
    if vix_path is not None and not vix_path.is_absolute():
        vix_path = Path(cfg["paths"]["project_root"]) / vix_path
    market_features = _prepare_regime_feature_frame(
        market_features,
        vix_path if vix_path is not None and vix_path.exists() else None,
    )

    _, manifest, factor_panel = load_factor_panel(cfg)
    all_dates = factor_panel.index.get_level_values("date").unique().sort_values()
    start, train_end, val_end = split_dates(all_dates, train_years, val_years, anchor=split_anchor)
    val_start = train_end + pd.Timedelta(days=1)
    logger.info(f"train: {start.date()} -> {train_end.date()} | val: {val_start.date()} -> {val_end.date()}")

    regime_cfg = cfg.get("regime", {})
    k_states = int(regime_cfg.get("k_states", 3))
    feature_cols = list(regime_cfg.get("feature_cols", ["mkt_ret_1d", "mkt_vol_20d"]))
    emit_temp = float(regime_cfg.get("emit_temp", 0.5))
    var_floor = float(regime_cfg.get("var_floor", 1.0))
    hard_prob = float(regime_cfg.get("hard_prob", 0.99))
    init_diag = float(regime_cfg.get("init_diag", 0.4))
    prob_smooth_days = int(regime_cfg.get("prob_smooth_days", 0))
    min_regime_duration = int(regime_cfg.get("min_regime_duration", 0))
    regime_lag = int(cfg["model"].get("regime_lag_days", 0))
    mkt_label, mkt_proba = fit_market_hmm(
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
        prob_smooth_days=prob_smooth_days,
        min_regime_duration=min_regime_duration,
    )

    sm_cfg = build_scoring_model_config(cfg)
    validation_mode = str(cfg["model"].get("validation_mode", "walk_forward")).strip().lower()
    if validation_mode not in {"walk_forward", "frozen"}:
        raise ValueError("model.validation_mode must be 'walk_forward' or 'frozen'")
    score_train_cutoff = train_end if validation_mode == "frozen" else None
    scorer = ScoringModel(sm_cfg)
    factor_family_map = load_factor_family_map(cfg)
    factor_family_map.pop("__default__", None)
    regime_family_priors = load_regime_family_priors(cfg)
    train_mask = mask_dates(factor_panel.index, start, train_end)
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
                score_panel = score_with_rolling_glm(
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
                regime_date_weights = {i: mkt_proba.reindex(train_dates).get(f"regime_{i}") for i in range(k_states)}
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
                factor_family_map=factor_family_map,
                regime_family_priors=regime_family_priors,
                thin_regime_eff_n_threshold=int(cfg["model"].get("thin_regime_eff_n_threshold", 0)),
                thin_regime_top_n=int(cfg["model"].get("thin_regime_top_n", 0)),
            )
        full_masks = {
            i: pd.Series((mkt_label == i).reindex(dates).fillna(False).to_numpy(), index=factor_panel.index)
            for i in range(k_states)
        }
        if sm_cfg.rolling_train_days and sm_cfg.rolling_train_days > 0:
            if sm_cfg.model_family == "glm":
                score_panel = score_with_rolling_glm(
                    scorer=scorer,
                    factor_panel=factor_panel,
                    prices_panel=prices_panel,
                    regime_masks=full_masks,
                    market_regime_proba=mkt_proba,
                    window_days=sm_cfg.rolling_train_days,
                    max_train_date=score_train_cutoff,
                )
            else:
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

    score_panel = maybe_calibrate_scores(
        score_panel,
        prices_panel,
        cfg,
        logger,
        label_horizon=sm_cfg.label_horizon,
        signal_lag=sm_cfg.signal_lag,
        train_mask=train_mask,
        max_train_date=score_train_cutoff,
    )

    out.tables_dir.mkdir(parents=True, exist_ok=True)
    mkt_label.to_frame("regime_label").to_parquet(out.tables_dir / "regime_label.parquet")
    mkt_proba.to_parquet(out.tables_dir / "regime_proba.parquet")

    train_scores = score_panel[train_mask]
    train_prices = prices_panel.loc[(slice(start, train_end), slice(None)), :]
    val_mask = mask_dates(factor_panel.index, train_end + pd.Timedelta(days=1), val_end)
    val_scores = score_panel[val_mask]
    val_prices = prices_panel.loc[(slice(train_end + pd.Timedelta(days=1), val_end), slice(None)), :]

    for cost_bps, short_fee_bps in resolve_stress_pairs(cfg, stress_level, split_mode=True):
        suffix = f"cost{int(cost_bps)}_fee{int(short_fee_bps)}"
        bt = Backtester(build_backtest_config(cfg, cost_bps, short_fee_bps))
        train_result = bt.run_with_details(train_scores, train_prices, out=None)
        val_result = bt.run_with_details(val_scores, val_prices, out=None)
        train_result.performance.to_parquet(out.tables_dir / f"performance_train_{suffix}.parquet")
        val_result.performance.to_parquet(out.tables_dir / f"performance_val_{suffix}.parquet")
        train_pos = train_result.positions.loc[train_result.positions["w"].abs() > 0.0]
        val_pos = val_result.positions.loc[val_result.positions["w"].abs() > 0.0]
        train_pos.to_csv(out.tables_dir / f"holdings_train_{suffix}.csv")
        val_pos.to_csv(out.tables_dir / f"holdings_val_{suffix}.csv")

    score_panel.to_parquet(out.tables_dir / "scores.parquet")
    write_run_metadata(
        out.artifacts_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "mode": "split",
            "config_path": str(cfg.get("_config_path", "")),
            "factors_manifest": str(manifest.path),
            "selected_factors": manifest.selected_factors,
            "date_range": {"start": cfg["data"]["start_date"], "end": cfg["data"]["end_date"]},
            "train_years": train_years,
            "val_years": val_years,
            "validation_mode": validation_mode,
            "model_family": cfg["model"]["model_family"],
        },
    )
    return out.run_dir


def run_small_portfolio_experiment_workflow(
    cfg: dict,
    logger,
    start_date: str,
    end_date: str,
    portfolio_methods: list[str],
    selection_top_n: int,
    max_factors_per_regime: int,
    use_regime: bool = False,
    score_calibration: str | None = "bucketed",
) -> Path:
    exp_cfg = json.loads(json.dumps(cfg))
    exp_cfg["data"]["start_date"] = start_date
    exp_cfg["data"]["end_date"] = end_date
    exp_cfg["model"]["use_regime"] = bool(use_regime)
    exp_cfg["model"]["selection_top_n"] = int(selection_top_n)
    exp_cfg["model"]["max_factors_per_regime"] = int(max_factors_per_regime)
    exp_cfg["backtest"]["stress_costs_bps"] = [exp_cfg["backtest"]["cost_bps"]]
    exp_cfg["backtest"]["stress_short_fee_bps"] = [exp_cfg["backtest"].get("short_fee_bps", 0.0)]
    if score_calibration is not None:
        exp_cfg["model"]["score_calibration"] = score_calibration

    base_backtest_dir = Path(exp_cfg["paths"]["backtest_dir"])
    exp_cfg["paths"]["backtest_dir"] = str(base_backtest_dir / "small_portfolio_research")
    run_id = make_run_id(exp_cfg | {"portfolio_methods": portfolio_methods})
    out = make_experiment_paths(Path(exp_cfg["paths"]["backtest_dir"]), run_id)
    logger.info(f"small portfolio experiment run_id={run_id}, out={out.run_dir}")

    _, prices_panel, _ = load_prices_and_market_features(exp_cfg)
    candidate_count = max(int(selection_top_n), int(max_factors_per_regime), 1)
    candidate_factors = list(exp_cfg["factors"]["selected"])[:candidate_count]
    _, manifest, factor_panel = load_factor_panel_selected(exp_cfg, columns=candidate_factors)

    sm_cfg = build_scoring_model_config(exp_cfg)
    scorer = ScoringModel(sm_cfg)
    all_mask = all_samples_mask(factor_panel)
    y = scorer.build_label(prices_panel)
    all_dates = factor_panel.index.get_level_values("date").unique()
    selected = scorer.selector.select_orthogonal_stepwise(
        factor_panel=factor_panel[candidate_factors],
        label=y,
        regime_dates=all_dates,
        top_n=int(selection_top_n),
        decay_half_life_days=int(sm_cfg.ic_decay_half_life_days),
    )
    if not selected:
        raise ValueError("No factors selected for small portfolio experiment")

    scorer.candidate_factors_ = list(candidate_factors)
    scorer.fit_regime_models_with_masks(
        factor_panel=factor_panel[selected],
        prices_panel=prices_panel,
        regime_masks={0: all_mask},
        regime_factors={0: selected},
    )
    if sm_cfg.rolling_train_days and sm_cfg.rolling_train_days > 0:
        proba = pd.DataFrame({"regime_0": 1.0}, index=all_dates)
        score_panel = scorer.score_with_rolling_ridge(
            factor_panel=factor_panel,
            prices_panel=prices_panel,
            regime_masks={0: all_mask},
            market_regime_proba=proba,
            window_days=sm_cfg.rolling_train_days,
        )
    else:
        score_panel = scorer.score(factor_panel, prices_panel, None, None)

    score_panel = maybe_calibrate_scores(
        score_panel,
        prices_panel,
        exp_cfg,
        logger,
        label_horizon=sm_cfg.label_horizon,
        signal_lag=sm_cfg.signal_lag,
    )

    out.tables_dir.mkdir(parents=True, exist_ok=True)
    score_panel.to_parquet(out.tables_dir / "scores.parquet")
    summary_rows: list[dict] = []
    cost_bps = float(exp_cfg["backtest"]["cost_bps"])
    short_fee_bps = float(exp_cfg["backtest"].get("short_fee_bps", 0.0))
    for method in portfolio_methods:
        method_cfg = json.loads(json.dumps(exp_cfg))
        method_cfg["backtest"]["portfolio_method"] = method
        bt = Backtester(build_backtest_config(method_cfg, cost_bps, short_fee_bps))
        result = bt.run_with_details(score_panel, prices_panel, out=None)
        result.performance.to_parquet(out.tables_dir / f"performance_{method}.parquet")
        result.positions.to_parquet(out.tables_dir / f"positions_{method}.parquet")
        perf = result.performance
        ret = perf["net"].dropna()
        years = len(ret) / 252.0 if len(ret) else np.nan
        equity = (1.0 + ret).cumprod() if len(ret) else pd.Series(dtype=float)
        cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if len(ret) and years > 0 else np.nan
        sharpe = ret.mean() / ret.std(ddof=0) * np.sqrt(252.0) if ret.std(ddof=0) > 0 else np.nan
        summary_rows.append(
            {
                "portfolio_method": method,
                "selected_factors": ",".join(selected),
                "final_equity": float(perf["equity"].iloc[-1]),
                "cagr": float(cagr),
                "sharpe": float(sharpe),
                "mean_cash_weight": float(perf["cash_weight"].mean()) if "cash_weight" in perf.columns else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)
    summary.to_csv(out.tables_dir / "summary.csv", index=False)
    write_run_metadata(
        out.artifacts_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "mode": "small_portfolio_research",
            "config_path": str(exp_cfg.get("_config_path", "")),
            "factors_manifest": str(manifest.path),
            "selected_candidate_factors": candidate_factors,
            "selected_factors": selected,
            "portfolio_methods": portfolio_methods,
            "date_range": {"start": start_date, "end": end_date},
            "selection_top_n": int(selection_top_n),
            "max_factors_per_regime": int(max_factors_per_regime),
            "use_regime": bool(use_regime),
        },
    )
    return out.run_dir
