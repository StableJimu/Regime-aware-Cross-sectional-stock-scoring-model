# -*- coding: utf-8 -*-
"""
Download public daily OHLCV data using Yahoo Finance via yfinance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_pipeline.pipeline.downloaders import run_price_download_workflow
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--tickers-file", type=str, default=None)
    p.add_argument("--panel-path", type=str, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--pause-seconds", type=float, default=0.25)
    p.add_argument("--market-symbols", type=str, default="SPY,VTI,DIA,QQQ")
    p.add_argument("--output-panel", type=str, default=None)
    p.add_argument("--output-market", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("download_prices")
    cfg = read_yaml(Path(args.config))
    run_price_download_workflow(
        cfg=cfg,
        logger=logger,
        tickers_file=args.tickers_file,
        panel_path=args.panel_path,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        pause_seconds=args.pause_seconds,
        market_symbols=args.market_symbols,
        output_panel=args.output_panel,
        output_market=args.output_market,
    )


if __name__ == "__main__":
    main()
