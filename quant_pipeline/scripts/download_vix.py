# -*- coding: utf-8 -*-
"""
Download public VIX history from Cboe and save it in the local pipeline format.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_pipeline.pipeline.downloaders import run_vix_download_workflow
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("download_vix")
    cfg = read_yaml(Path(args.config))
    run_vix_download_workflow(cfg=cfg, logger=logger, output=args.output)


if __name__ == "__main__":
    main()
