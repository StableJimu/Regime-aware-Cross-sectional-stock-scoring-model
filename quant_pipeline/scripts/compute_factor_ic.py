# -*- coding: utf-8 -*-
"""
Compute single-factor IC (daily Spearman) from factor parquet batches.
Prints mean IC and ICIR to stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_pipeline.pipeline.utils import read_yaml, setup_logger
from quant_pipeline.pipeline.artifacts import read_factor_manifest


def _read_panel_for_returns(panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        raise FileNotFoundError(f"raw panel not found: {panel_path}")
    if panel_path.suffix.lower() == ".csv":
        df = pd.read_csv(panel_path, usecols=["ts_event", "symbol", "close"])
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts_event"])
        df["date"] = df["ts_event"].dt.tz_convert(None).dt.normalize()
        df = df.rename(columns={"symbol": "ticker"})
    else:
        df = pd.read_parquet(panel_path)
        if {"date", "ticker", "close"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"])
        else:
            raise ValueError(f"{panel_path} missing required columns for returns")
    df = df[["date", "ticker", "close"]].copy()
    df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    df["ret_1d"] = df.groupby("ticker")["close"].transform(lambda s: np.log(s).diff()).astype(float)
    df["fwd_ret_1d"] = df.groupby("ticker")["ret_1d"].shift(-1)
    return df.set_index(["date", "ticker"]).sort_index()


def _daily_ic(series: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    df = pd.concat([series.rename("factor"), fwd_ret.rename("fwd_ret_1d")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def _ic_one(x: pd.DataFrame) -> float:
        if len(x) < 2:
            return np.nan
        return x["factor"].corr(x["fwd_ret_1d"], method="spearman")

    return df.groupby(level="date", group_keys=False).apply(_ic_one)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--factors", type=str, required=True, help="Comma-separated factor names (e.g., alpha_054,alpha_025)")
    p.add_argument("--raw-panel", type=str, default="data/raw/ohlcv_1d_panel.csv")
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

    panel = _read_panel_for_returns(Path(args.raw_panel))
    fwd_ret = panel["fwd_ret_1d"]

    print("Factor,MeanIC,ICIR")
    for f in factors:
        ic = _daily_ic(fp[f], fwd_ret).dropna()
        if ic.empty:
            print(f"{f},nan,nan")
            continue
        mean_ic = ic.mean()
        icir = mean_ic / ic.std(ddof=0) if ic.std(ddof=0) > 0 else np.nan
        print(f"{f},{mean_ic:.6f},{icir:.6f}")


if __name__ == "__main__":
    main()
