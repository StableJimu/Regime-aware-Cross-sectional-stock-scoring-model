# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:33:59 2026

@author: jimya
"""

# quant_pipeline/scripts/run_backtest.py
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument(
        "--stress-level",
        type=str,
        default="all",
        choices=["low", "medium", "high", "all"],
    )
    return p.parse_args()


def _make_run_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _calibrate_scores_linear(
    score_panel: pd.DataFrame,
    prices_panel: pd.DataFrame,
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
    out["score"] = a + b * out["score_raw"]
    return out


def main() -> None:
    args = parse_args()
    logger = setup_logger("run_backtest")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    np.seterr(all="ignore")
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
        rates_path=Path(cfg["paths"]["rates_path"]) if cfg["paths"].get("rates_path") else None,
        market_proxy_ticker=cfg["data"].get("market_proxy_ticker", "SPY"),
    )
    loader = DataLoader(dl_cfg)
    prices_panel, market_features = loader.load_and_build()

    logger.info(
        f"prices_panel rows={len(prices_panel):,}, dates={prices_panel.index.get_level_values('date').nunique():,}, "
        f"tickers={prices_panel.index.get_level_values('ticker').nunique():,}"
    )
    logger.info(
        f"market_features rows={len(market_features):,}, dates=[{market_features.index.min()}..{market_features.index.max()}]"
    )
    # 2) Factors
    registry = FactorRegistry()
    register_alpha101(registry)
    fp_cfg = FactorProcessorConfig(factors_dir=Path(cfg["paths"]["factors_dir"]))
    factor_proc = FactorProcessor(fp_cfg, registry)

    selected_factors = cfg.get("factors", {}).get("selected", [
        "alpha_001", "alpha_002", "alpha_003", "alpha_004", "alpha_005",
        "alpha_006", "alpha_007", "alpha_008", "alpha_009", "alpha_010",
    ])

    manifest_path = Path(cfg["paths"]["factors_manifest"])
    manifest = read_factor_manifest(manifest_path)
    if not manifest.batches:
        raise ValueError("factors_manifest has no batches")
    logger.info(f"factor batches={len(manifest.batches)}, factors={len(manifest.selected_factors)}")
    factor_panel_raw = pd.concat([pd.read_parquet(p) for p in manifest.batches]).sort_index()
    factor_panel = factor_proc.transform_cross_section(factor_panel_raw)
    factor_lag = int(cfg["model"].get("factor_lag", 0))
    if factor_lag > 0:
        factor_panel = factor_panel.groupby(level="ticker").shift(factor_lag)
    logger.info(f"factor_panel rows={len(factor_panel):,}, cols={len(factor_panel.columns)}")

    # 3) Market regime from HMM (probability-weighted ensemble)
    hmm_path = Path(cfg["paths"]["regime_hmm_path"])
    if not hmm_path.exists():
        raise FileNotFoundError(f"HMM regime file not found: {hmm_path}")
    hmm = pd.read_csv(hmm_path)
    hmm["date"] = pd.to_datetime(hmm["date"])
    hmm = hmm.set_index("date").sort_index()
    prob_cols = [c for c in hmm.columns if c.startswith("prob_")]
    if len(prob_cols) < 3:
        raise ValueError("HMM regime file must have prob_1..prob_3 for 3 regimes")

    # Align to market_features dates
    hmm = hmm.reindex(market_features.index).ffill().bfill()
    prob = hmm[prob_cols].to_numpy()
    prob = prob / (prob.sum(axis=1, keepdims=True) + 1e-12)
    max_prob = prob.max(axis=1)
    hard = np.eye(prob.shape[1])[prob.argmax(axis=1)]
    hard_share = float((max_prob >= 0.99).mean())
    logger.info(f"HMM regimes K={prob.shape[1]}, hard_share={hard_share:.3f}")
    use_hard = (max_prob >= 0.99)[:, None]
    prob = np.where(use_hard, hard, prob)

    mkt_label = pd.Series(prob.argmax(axis=1), index=market_features.index, name="regime_label")
    mkt_proba = pd.DataFrame(prob, index=market_features.index, columns=[f"regime_{i}" for i in range(prob.shape[1])])
    regime_lag = int(cfg["model"].get("regime_lag_days", 0))
    if regime_lag > 0:
        mkt_proba = mkt_proba.shift(regime_lag).ffill()
        mkt_label = mkt_label.shift(regime_lag)

    # 4) Scoring model with regime-specific factor sets
    dates = factor_panel.index.get_level_values("date")
    use_regime = bool(cfg["model"].get("use_regime", True))
    if use_regime:
        masks = {
            i: pd.Series((mkt_label == i).reindex(dates).fillna(False).to_numpy(), index=factor_panel.index)
            for i in range(3)
        }
        for k, m in masks.items():
            logger.info(f"regime {k}: samples={int(m.sum()):,}")
    else:
        masks = {0: all_samples_mask(factor_panel.index)}
        logger.info(f"regime disabled: samples={int(masks[0].sum()):,}")

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
    )
    scorer = ScoringModel(sm_cfg)
    candidate_factors = manifest.selected_factors
    selection_method = cfg["model"].get("selection_method", "regime_stepwise")
    top_n = int(cfg["model"].get("selection_top_n", 10))
    logger.info(f"model_family={sm_cfg.model_family}, selection_method={selection_method}, selection_top_n={top_n}")
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
        if not selected_global:
            raise ValueError("Global selection returned no factors")
        scorer.fit_regime_models_with_masks(
            factor_panel=factor_panel[selected_global],
            prices_panel=prices_panel,
            regime_masks=masks,
            regime_factors={0: selected_global},
        )
        for k, cols in scorer.selected_factors_.items():
            logger.info(f"regime {k}: selected_factors={len(cols)}")
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
            )
        for k, cols in scorer.selected_factors_.items():
            logger.info(f"regime {k}: selected_factors={len(cols)}")
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
    logger.info(f"score_panel rows={len(score_panel):,}, nan_score={score_panel['score'].isna().mean():.4f}")

    if cfg["model"].get("score_calibration") == "linear":
        score_panel = _calibrate_scores_linear(
            score_panel=score_panel,
            prices_panel=prices_panel,
            label_horizon=sm_cfg.label_horizon,
            signal_lag=sm_cfg.signal_lag,
            logger=logger,
        )

    # 5) Backtest
    stress_costs = cfg["backtest"].get("stress_costs_bps", [cfg["backtest"]["cost_bps"]])
    stress_short_fees = cfg["backtest"].get("stress_short_fee_bps", [0.0])
    if len(stress_costs) != len(stress_short_fees):
        raise ValueError("stress_costs_bps and stress_short_fee_bps must have same length")

    level_map = {"low": 0, "medium": 1, "high": 2}
    if args.stress_level != "all":
        idx = level_map[args.stress_level]
        pairs = [(stress_costs[idx], stress_short_fees[idx])]
    else:
        pairs = list(zip(stress_costs, stress_short_fees))

    logger.info(f"stress_level={args.stress_level}, scenarios={len(pairs)}")

    for cost_bps, short_fee_bps in pairs:
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
        perf = bt.run(score_panel, prices_panel, out=None)
        suffix = f"cost{int(cost_bps)}_fee{int(short_fee_bps)}"
        out.tables_dir.mkdir(parents=True, exist_ok=True)
        perf.to_parquet(out.tables_dir / f"performance_{suffix}.parquet")
    write_run_metadata(
        out.artifacts_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "mode": "full",
            "config_path": str(args.config),
            "factors_manifest": str(manifest.path),
            "selected_factors": manifest.selected_factors,
            "date_range": {
                "start": cfg["data"]["start_date"],
                "end": cfg["data"]["end_date"],
            },
            "model_family": cfg["model"]["model_family"],
        },
    )
    logger.info("backtest finished")


if __name__ == "__main__":
    main()
