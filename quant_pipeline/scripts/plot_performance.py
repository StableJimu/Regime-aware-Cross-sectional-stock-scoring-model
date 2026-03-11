# -*- coding: utf-8 -*-
"""
Plot backtest performance and print basic stats + IC/IR diagnostics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quant_pipeline.pipeline.reporting import (
    build_benchmark_series,
    compute_daily_ic,
    compute_ic_ir_stats,
    compute_performance_stats,
    plot_performance,
    read_panel_for_returns,
    read_performance,
    read_regime_label,
    read_scores,
    read_split_performance,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True, help="Path to a specific run directory")
    p.add_argument("--raw-panel", type=str, default="data/raw/ohlcv_1d_panel.csv")
    p.add_argument("--benchmark-file", type=str, default="data/raw/market_index_panel.csv")
    p.add_argument("--benchmark-ticker", type=str, default="SPY")
    p.add_argument("--ic-method", type=str, default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--save", type=str, default=None, help="Save plot to file instead of showing")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    p.add_argument("--title", type=str, default="Backtest Performance")
    p.add_argument("--split", action="store_true", help="Use performance_train/val if present")
    p.add_argument("--performance-file", type=str, default=None, help="Override performance parquet file")
    p.add_argument("--split-suffix", type=str, default="", help="Suffix for split files, e.g. cost20_fee300")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    perf_val = None
    if args.performance_file and args.split:
        raise ValueError("--performance-file is incompatible with --split")
    if args.split:
        perf, perf_val = read_split_performance(run_dir, split_suffix=args.split_suffix)
    else:
        perf_file = Path(args.performance_file) if args.performance_file else None
        perf = read_performance(run_dir, perf_file=perf_file)
    stats = compute_performance_stats(perf)
    ic_series = None
    ic_series_val = None

    try:
        scores = read_scores(run_dir)
        panel = read_panel_for_returns(Path(args.raw_panel))
        fwd_ret = panel["fwd_ret_1d"]
        if perf_val is not None:
            ic_series = compute_daily_ic(scores.loc[perf.index], fwd_ret, method=args.ic_method)
            ic_series_val = compute_daily_ic(scores.loc[perf_val.index], fwd_ret, method=args.ic_method)
        else:
            ic_series = compute_daily_ic(scores, fwd_ret, method=args.ic_method)
    except Exception as e:
        print(f"[warn] IC/IR skipped: {e}")

    benchmark = None
    try:
        benchmark = build_benchmark_series(
            benchmark_file=Path(args.benchmark_file),
            benchmark_ticker=args.benchmark_ticker,
            perf=perf,
            perf_val=perf_val,
        )
    except Exception as e:
        print(f"[warn] Benchmark skipped: {e}")

    regime_label = None
    try:
        regime_label = read_regime_label(run_dir)
    except Exception as e:
        print(f"[warn] Regime label skipped: {e}")

    print("Stats:")
    print(f"  CAGR:        {stats['cagr']:.4f}")
    print(f"  Sharpe:      {stats['sharpe']:.4f}")
    print(f"  Max DD:      {stats['max_drawdown']:.4f}")
    print(f"  Calmar:      {stats['calmar']:.4f}")

    if perf_val is not None:
        stats_val = compute_performance_stats(perf_val)
        print("Validation Stats:")
        print(f"  CAGR:        {stats_val['cagr']:.4f}")
        print(f"  Sharpe:      {stats_val['sharpe']:.4f}")
        print(f"  Max DD:      {stats_val['max_drawdown']:.4f}")
        print(f"  Calmar:      {stats_val['calmar']:.4f}")

    if ic_series:
        ic_means = []
        icirs = []
        print("IC/IR:")
        for k, s in ic_series.items():
            m = compute_ic_ir_stats(s)
            ic_means.append(m["mean_ic"])
            icirs.append(m["icir"])
            print(f"  {k}: mean IC={m['mean_ic']:.4f}  ICIR={m['icir']:.4f}")
        print(f"  Average IC:  {float(np.nanmean(ic_means)) if ic_means else np.nan:.4f}")
        print(f"  Average IR:  {float(np.nanmean(icirs)) if icirs else np.nan:.4f}")

    if ic_series_val:
        ic_means = []
        icirs = []
        print("Validation IC/IR:")
        for k, s in ic_series_val.items():
            m = compute_ic_ir_stats(s)
            ic_means.append(m["mean_ic"])
            icirs.append(m["icir"])
            print(f"  {k}: mean IC={m['mean_ic']:.4f}  ICIR={m['icir']:.4f}")
        print(f"  Average IC:  {float(np.nanmean(ic_means)) if ic_means else np.nan:.4f}")
        print(f"  Average IR:  {float(np.nanmean(icirs)) if icirs else np.nan:.4f}")

    save_path = Path(args.save) if args.save else None
    plot_performance(
        perf,
        ic_series=ic_series,
        title=args.title,
        save_path=save_path,
        show=not args.no_show,
        perf_val=perf_val,
        benchmark=benchmark,
        regime_label=regime_label,
    )


if __name__ == "__main__":
    main()
