# -*- coding: utf-8 -*-
"""
Compute factors in ticker batches and store raw factors to parquet files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_pipeline.pipeline.experiments import run_factor_batch_workflow
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--force", action="store_true", help="Recompute existing batch parquet files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("compute_factors_batch")
    cfg = read_yaml(Path(args.config))
    cfg["_config_path"] = args.config
    run_factor_batch_workflow(cfg, logger=logger, force=args.force)


if __name__ == "__main__":
    main()
