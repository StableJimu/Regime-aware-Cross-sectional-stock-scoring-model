# -*- coding: utf-8 -*-
"""
Full-history backtest CLI.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from quant_pipeline.pipeline.experiments import run_full_backtest_workflow
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


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


def main() -> None:
    args = parse_args()
    logger = setup_logger("run_backtest")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    np.seterr(all="ignore")
    cfg = read_yaml(Path(args.config))
    cfg["_config_path"] = args.config
    run_full_backtest_workflow(cfg, logger=logger, stress_level=args.stress_level)
    logger.info("backtest finished")


if __name__ == "__main__":
    main()
