# -*- coding: utf-8 -*-
"""
Run a compact non-regime portfolio-construction comparison on a shorter sample.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_pipeline.pipeline.experiments import run_small_portfolio_experiment_workflow
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--start", type=str, default="2021-01-01")
    p.add_argument("--end", type=str, default="2024-12-31")
    p.add_argument(
        "--portfolio-methods",
        type=str,
        default="top_q,proportional,diag_mv,spo",
        help="Comma-separated portfolio methods",
    )
    p.add_argument("--selection-top-n", type=int, default=5)
    p.add_argument("--max-factors", type=int, default=5)
    p.add_argument("--use-regime", action="store_true", help="Enable regime logic for the small experiment")
    p.add_argument("--score-calibration", type=str, default="bucketed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("run_small_portfolio_research")
    cfg = read_yaml(Path(args.config))
    cfg["_config_path"] = args.config
    methods = [m.strip() for m in args.portfolio_methods.split(",") if m.strip()]
    run_small_portfolio_experiment_workflow(
        cfg=cfg,
        logger=logger,
        start_date=args.start,
        end_date=args.end,
        portfolio_methods=methods,
        selection_top_n=args.selection_top_n,
        max_factors_per_regime=args.max_factors,
        use_regime=args.use_regime,
        score_calibration=args.score_calibration,
    )


if __name__ == "__main__":
    main()
