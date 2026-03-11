# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:28:32 2026

@author: jimya
"""

"""
Purpose：统一入口（CLI 风格），按 flags 选择执行模块。
TODO：支持 --mode {factor_calc, update, backtest}；把所有 I/O 和路径都从 config 注入，不在代码里写死。
"""

# quant_pipeline/main.py
from __future__ import annotations
import argparse
from quant_pipeline.scripts.run_backtest import main as run_backtest_main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="backtest", choices=["backtest"])
    p.add_argument("--config", type=str, default="config/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "backtest":
        run_backtest_main(args.config)


if __name__ == "__main__":
    main()
