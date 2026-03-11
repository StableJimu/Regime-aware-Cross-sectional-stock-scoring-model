from __future__ import annotations

import argparse
from pathlib import Path

from quant_pipeline.pipeline.experiments import run_regime_parameter_sweep_workflow
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config_split_walk_forward.yaml")
    p.add_argument("--train-years", type=int, default=5)
    p.add_argument("--val-years", type=int, default=2)
    p.add_argument("--split-anchor", type=str, default="latest", choices=["latest", "earliest"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("run_regime_parameter_sweep")
    cfg = read_yaml(Path(args.config))
    cfg["_config_path"] = args.config
    run_regime_parameter_sweep_workflow(
        cfg,
        logger=logger,
        train_years=args.train_years,
        val_years=args.val_years,
        split_anchor=args.split_anchor,
    )


if __name__ == "__main__":
    main()
