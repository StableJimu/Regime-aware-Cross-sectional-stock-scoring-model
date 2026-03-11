# -*- coding: utf-8 -*-
"""
Download public VIX history from Cboe and save it in the local pipeline format.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_pipeline.pipeline.utils import read_yaml, setup_logger


CBOE_VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("download_vix")
    cfg = read_yaml(Path(args.config))

    out_path = Path(args.output or cfg["paths"].get("vix_path") or "data/raw/vix.csv")
    df = pd.read_csv(CBOE_VIX_HISTORY_URL)
    cols = {str(c).strip().upper(): c for c in df.columns}
    date_col = cols.get("DATE")
    close_col = cols.get("CLOSE")
    if date_col is None or close_col is None:
        raise ValueError(f"Unexpected VIX columns: {list(df.columns)}")

    out = df[[date_col, close_col]].copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"saved vix history: {out_path} rows={len(out):,}")


if __name__ == "__main__":
    main()
