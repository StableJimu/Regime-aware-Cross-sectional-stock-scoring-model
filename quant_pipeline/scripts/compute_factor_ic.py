# -*- coding: utf-8 -*-
"""
Compute single-factor IC (daily Spearman) from factor parquet batches.
Prints mean IC and ICIR to stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_pipeline.pipeline.artifacts import read_factor_manifest
from quant_pipeline.pipeline.reporting import (
    build_forward_return_series,
    compute_factor_daily_ic,
    compute_ic_ir_stats,
    read_panel_for_returns,
)
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--factors", type=str, required=True, help="Comma-separated factor names (e.g., alpha_054,alpha_025)")
    p.add_argument("--raw-panel", type=str, default="data/raw/ohlcv_1d_panel.csv")
    p.add_argument("--horizon", type=int, default=1, help="Forward return horizon in trading days")
    p.add_argument("--signal-lag", type=int, default=1, help="Signal lag in trading days")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("compute_factor_ic")
    cfg = read_yaml(Path(args.config))

    manifest = read_factor_manifest(Path(cfg["paths"]["factors_manifest"]))
    if not manifest.batches:
        raise ValueError("factors_manifest has no batches")

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    if not factors:
        raise ValueError("no factors provided")
    missing = [f for f in factors if f not in manifest.selected_factors]
    if missing:
        raise ValueError(f"factors not in manifest: {missing}")

    logger.info("loading factor batches...")
    fp = pd.concat([pd.read_parquet(p) for p in manifest.batches]).sort_index()
    for f in factors:
        if f not in fp.columns:
            raise ValueError(f"factor column missing: {f}")

    panel = read_panel_for_returns(Path(args.raw_panel))
    fwd_ret = build_forward_return_series(panel["ret_1d"], horizon=args.horizon, signal_lag=args.signal_lag)

    print("Factor,Horizon,MeanIC,ICIR")
    for f in factors:
        ic = compute_factor_daily_ic(fp[f], fwd_ret).dropna()
        if ic.empty:
            print(f"{f},{int(args.horizon)},nan,nan")
            continue
        stats = compute_ic_ir_stats(ic)
        print(f"{f},{int(args.horizon)},{stats['mean_ic']:.6f},{stats['icir']:.6f}")


if __name__ == "__main__":
    main()
