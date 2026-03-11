# -*- coding: utf-8 -*-
"""
Download FRED DGS2 and DGS10 for the dataset date range and save locally.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_pipeline.pipeline.downloaders import run_fred_yields_download_workflow
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--raw-panel", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("download_fred_yields")
    cfg = read_yaml(Path(args.config))
    raw_panel = args.raw_panel or cfg["paths"].get("panel_path") or "data/raw/ohlcv_1d_panel.csv"
    output = args.output or cfg["paths"].get("rates_path") or "data/raw/treasury_yields.csv"
    run_fred_yields_download_workflow(logger=logger, raw_panel=raw_panel, output=output)


if __name__ == "__main__":
    main()
