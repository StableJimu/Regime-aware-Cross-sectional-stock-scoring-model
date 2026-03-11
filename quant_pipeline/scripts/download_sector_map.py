# -*- coding: utf-8 -*-
"""
Download a public sector proxy for the current ticker universe using Yahoo Finance metadata.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_pipeline.pipeline.downloaders import run_sector_map_download_workflow
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--tickers-file", type=str, default=None)
    p.add_argument("--panel-path", type=str, default=None)
    p.add_argument("--output", type=str, default="data/raw/sector_map.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("download_sector_map")
    cfg = read_yaml(Path(args.config))
    run_sector_map_download_workflow(
        cfg=cfg,
        logger=logger,
        tickers_file=args.tickers_file,
        panel_path=args.panel_path,
        output=args.output,
    )


if __name__ == "__main__":
    main()
