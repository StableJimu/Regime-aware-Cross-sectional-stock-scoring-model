# -*- coding: utf-8 -*-
"""
Standalone HMM market regime learner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_pipeline.pipeline.experiments import run_standalone_hmm_workflow
from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--use-vix", action="store_true", help="Use vix.csv if present")
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--emit-temp", type=float, default=0.5)
    p.add_argument("--var-floor", type=float, default=1.0)
    p.add_argument("--half-life", type=int, default=63)
    p.add_argument("--rolling-window-days", type=int, default=252)
    p.add_argument("--refit-freq-days", type=int, default=21)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("run_hmm_regime")
    cfg = read_yaml(Path(args.config))
    cfg["_config_path"] = args.config
    run_standalone_hmm_workflow(
        cfg,
        logger=logger,
        k_states=args.k,
        use_vix=args.use_vix,
        max_iter=args.max_iter,
        emit_temp=args.emit_temp,
        var_floor=args.var_floor,
        half_life=args.half_life,
        rolling_window_days=args.rolling_window_days,
        refit_freq_days=args.refit_freq_days,
    )


if __name__ == "__main__":
    main()
