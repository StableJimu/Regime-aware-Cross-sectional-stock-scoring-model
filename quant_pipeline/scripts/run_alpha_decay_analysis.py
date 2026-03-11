from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_pipeline.pipeline.reporting import (
    build_forward_return_series,
    compute_daily_ic,
    compute_factor_daily_ic,
    compute_ic_ir_stats,
    read_panel_for_returns,
    read_scores,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default=None, help="Run directory containing scores.parquet")
    p.add_argument("--raw-panel", type=str, default="data/raw/ohlcv_1d_panel.csv")
    p.add_argument("--score-cols", type=str, default="score,score_raw,score_calibrated")
    p.add_argument("--factor-file", type=str, default=None, help="Optional parquet/csv with MultiIndex date,ticker factor columns")
    p.add_argument("--factor-cols", type=str, default="", help="Optional comma-separated factor columns for decay study")
    p.add_argument("--horizons", type=str, default="1,2,3,5,10")
    p.add_argument("--signal-lag", type=int, default=1)
    p.add_argument("--method", type=str, default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--save", type=str, default=None, help="Optional CSV path to save summary")
    return p.parse_args()


def _read_factor_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if {"date", "ticker"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index(["date", "ticker"]).sort_index()
    raise ValueError(f"Unsupported factor file: {path}")


def main() -> None:
    args = parse_args()
    panel = read_panel_for_returns(Path(args.raw_panel))
    ret_1d = panel["ret_1d"]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    rows: list[dict] = []

    if args.run_dir:
        scores = read_scores(Path(args.run_dir))
        score_cols = [c.strip() for c in args.score_cols.split(",") if c.strip() and c.strip() in scores.columns]
        for horizon in horizons:
            fwd = build_forward_return_series(ret_1d, horizon=horizon, signal_lag=int(args.signal_lag))
            ic_map = compute_daily_ic(scores[score_cols], fwd, method=args.method)
            for col, series in ic_map.items():
                stats = compute_ic_ir_stats(series)
                rows.append(
                    {
                        "source": "score",
                        "column": col,
                        "horizon": horizon,
                        "mean_ic": stats["mean_ic"],
                        "icir": stats["icir"],
                        "n_dates": int(series.dropna().shape[0]),
                    }
                )

    if args.factor_file and args.factor_cols:
        factor_frame = _read_factor_frame(Path(args.factor_file))
        factor_cols = [c.strip() for c in args.factor_cols.split(",") if c.strip() and c.strip() in factor_frame.columns]
        for horizon in horizons:
            fwd = build_forward_return_series(ret_1d, horizon=horizon, signal_lag=int(args.signal_lag))
            for col in factor_cols:
                series = compute_factor_daily_ic(factor_frame[col], fwd, method=args.method)
                stats = compute_ic_ir_stats(series)
                rows.append(
                    {
                        "source": "factor",
                        "column": col,
                        "horizon": horizon,
                        "mean_ic": stats["mean_ic"],
                        "icir": stats["icir"],
                        "n_dates": int(series.dropna().shape[0]),
                    }
                )

    out = pd.DataFrame(rows).sort_values(["source", "column", "horizon"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No decay results produced")

    print(out.to_csv(index=False))
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)


if __name__ == "__main__":
    main()
