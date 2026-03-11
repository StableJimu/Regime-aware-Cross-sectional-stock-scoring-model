# -*- coding: utf-8 -*-
"""
Plot backtest performance and print basic stats + IC/IR diagnostics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_perf(run_dir: Path, perf_file: Path | None = None) -> pd.DataFrame:
    if perf_file is not None:
        perf_path = perf_file
    else:
        perf_path = run_dir / "tables" / "performance.parquet"
    if not perf_path.exists():
        raise FileNotFoundError(f"performance.parquet not found: {perf_path}")
    return pd.read_parquet(perf_path)


def _read_scores(run_dir: Path) -> pd.DataFrame:
    scores_path = run_dir / "tables" / "scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.parquet not found: {scores_path}")
    return pd.read_parquet(scores_path)


def _stats(perf: pd.DataFrame, trading_days: int = 252) -> Dict[str, float]:
    if "net" not in perf.columns:
        raise ValueError("performance.parquet must include column 'net'")
    ret = perf["net"].dropna()
    if ret.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "calmar": np.nan}

    equity = (1.0 + ret).cumprod()
    years = len(ret) / trading_days
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    sharpe = ret.mean() / ret.std(ddof=0) * np.sqrt(trading_days) if ret.std(ddof=0) > 0 else np.nan
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
    }


def _read_panel_for_returns(panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        raise FileNotFoundError(f"raw panel not found: {panel_path}")
    if panel_path.suffix.lower() == ".csv":
        df = pd.read_csv(panel_path, usecols=["ts_event", "symbol", "close"])
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts_event"])
        df["date"] = df["ts_event"].dt.tz_convert(None).dt.normalize()
        df = df.rename(columns={"symbol": "ticker"})
    else:
        df = pd.read_parquet(panel_path)
        if {"date", "ticker", "close"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"])
        else:
            raise ValueError(f"{panel_path} missing required columns for returns")
    df = df[["date", "ticker", "close"]].copy()
    df = df.sort_values(["ticker", "date"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df["ret_1d"] = df.groupby("ticker")["close"].transform(lambda s: np.log(s).diff()).astype(float)
    df["fwd_ret_1d"] = df.groupby("ticker")["ret_1d"].shift(-1)
    return df.set_index(["date", "ticker"]).sort_index()


def _daily_ic(scores: pd.DataFrame, fwd_ret: pd.Series, method: str = "spearman") -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    fwd = fwd_ret.copy()
    if fwd.index.names != scores.index.names:
        try:
            fwd.index = fwd.index.set_names(scores.index.names)
        except Exception:
            pass
    fwd = fwd.reindex(scores.index)
    df = scores.join(fwd.rename("fwd_ret_1d"), how="inner")
    for col in scores.columns:
        def _ic_one_date(x: pd.DataFrame) -> float:
            x = x[[col, "fwd_ret_1d"]].dropna()
            if len(x) < 2:
                return np.nan
            return x[col].corr(x["fwd_ret_1d"], method=method)
        ic = df.groupby(level="date", group_keys=False).apply(_ic_one_date)
        ic.name = f"ic_{col}"
        out[col] = ic
    return out


def _ic_ir_stats(ic: pd.Series) -> Dict[str, float]:
    ic = ic.dropna()
    if ic.empty:
        return {"mean_ic": np.nan, "icir": np.nan}
    mean_ic = ic.mean()
    ic_std = ic.std(ddof=0)
    icir = mean_ic / ic_std if ic_std > 0 else np.nan
    return {"mean_ic": float(mean_ic), "icir": float(icir)}


def plot_performance(
    perf: pd.DataFrame,
    ic_series: Dict[str, pd.Series] | None = None,
    title: str = "Backtest Performance",
    save_path: Path | None = None,
    show: bool = True,
    perf_val: pd.DataFrame | None = None,
    benchmark: pd.Series | Dict[str, pd.Series] | None = None,
    regime_label: pd.Series | None = None,
) -> None:
    if "equity" not in perf.columns:
        perf = perf.copy()
        perf["equity"] = (1.0 + perf["net"].fillna(0.0)).cumprod()

    if ic_series:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax0, ax1 = axes
    else:
        fig, ax0 = plt.subplots(figsize=(10, 5))
        ax1 = None

    def _shade_regimes(ax, labels: pd.Series) -> None:
        if labels is None or labels.empty:
            return
        labels = labels.dropna()
        if labels.empty:
            return
        labels = labels.sort_index()
        unique = sorted(pd.unique(labels))
        if not unique:
            return
        cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
        color_map = {int(k): cmap(i % cmap.N) for i, k in enumerate(unique)}
        for k in unique:
            ax.plot([], [], color=color_map[int(k)], lw=6, alpha=0.6, label=f"Regime {int(k)}")
        start = labels.index[0]
        last_k = int(labels.iloc[0])
        for dt, k in labels.iloc[1:].items():
            k = int(k)
            if k != last_k:
                ax.axvspan(start, dt, color=color_map[last_k], alpha=0.18, lw=0)
                start = dt
                last_k = k
        ax.axvspan(start, labels.index[-1], color=color_map[last_k], alpha=0.18, lw=0)

    end_all = float(perf["equity"].iloc[-1])
    label_all = f"Train/All (End={end_all:.2f})"
    perf["equity"].plot(ax=ax0, color="#111111", lw=1.8, label=label_all)
    if benchmark is not None:
        if isinstance(benchmark, dict):
            bench_train = benchmark.get("train")
            bench_val = benchmark.get("val")
            if bench_train is not None and not bench_train.empty:
                bench_train.plot(ax=ax0, color="#1f77b4", lw=1.4, alpha=0.9, label="Benchmark (Train)")
            if bench_val is not None and not bench_val.empty:
                bench_val.plot(ax=ax0, color="#1f77b4", lw=1.4, alpha=0.9, ls="--", label="Benchmark (Val)")
        else:
            if not benchmark.empty:
                benchmark.plot(ax=ax0, color="#1f77b4", lw=1.4, alpha=0.9, label="Benchmark")
    if perf_val is not None:
        if "equity" not in perf_val.columns:
            perf_val = perf_val.copy()
            perf_val["equity"] = (1.0 + perf_val["net"].fillna(0.0)).cumprod()
        end_val = float(perf_val["equity"].iloc[-1])
        label_val = f"Validation (End={end_val:.2f})"
        perf_val["equity"].plot(ax=ax0, color="#d62728", lw=1.4, alpha=0.9, label=label_val)

    if regime_label is not None:
        if perf_val is not None:
            full_idx = perf.index.union(perf_val.index)
        else:
            full_idx = perf.index
        labels = regime_label.reindex(full_idx).dropna()
        _shade_regimes(ax0, labels)
    ax0.set_title(title)
    ax0.set_ylabel("Equity")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    if ic_series and ax1 is not None:
        for name, s in ic_series.items():
            s.plot(ax=ax1, lw=1.0, alpha=0.8, label=name)
        ax1.axhline(0.0, color="gray", lw=0.8)
        ax1.set_ylabel("Daily IC")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    ax0.set_xlabel("Date")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


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
        suffix = f"_{args.split_suffix}" if args.split_suffix else ""
        train_path = run_dir / "tables" / f"performance_train{suffix}.parquet"
        val_path = run_dir / "tables" / f"performance_val{suffix}.parquet"
        if train_path.exists() and val_path.exists():
            perf = pd.read_parquet(train_path)
            perf_val = pd.read_parquet(val_path)
        else:
            raise FileNotFoundError("performance_train.parquet or performance_val.parquet not found")
    else:
        perf_file = Path(args.performance_file) if args.performance_file else None
        perf = _read_perf(run_dir, perf_file=perf_file)
    stats = _stats(perf)
    ic_series = None
    ic_series_val = None

    panel = None
    try:
        scores = _read_scores(run_dir)
        panel = _read_panel_for_returns(Path(args.raw_panel))
        fwd_ret = panel["fwd_ret_1d"]
        if perf_val is not None:
            train_dates = perf.index
            val_dates = perf_val.index
            ic_series = _daily_ic(scores.loc[train_dates], fwd_ret, method=args.ic_method)
            ic_series_val = _daily_ic(scores.loc[val_dates], fwd_ret, method=args.ic_method)
        else:
            ic_series = _daily_ic(scores, fwd_ret, method=args.ic_method)
    except Exception as e:
        print(f"[warn] IC/IR skipped: {e}")

    benchmark = None
    try:
        bench_panel = _read_panel_for_returns(Path(args.benchmark_file))
        bench = bench_panel.xs(args.benchmark_ticker, level="ticker")["ret_1d"].dropna()
        if perf_val is not None:
            train_dates = perf.index
            val_dates = perf_val.index
            bench_train = (1.0 + bench.reindex(train_dates).fillna(0.0)).cumprod()
            bench_val = (1.0 + bench.reindex(val_dates).fillna(0.0)).cumprod()
            if not bench_train.empty and bench_train.iloc[0] != 0:
                bench_train = bench_train / float(bench_train.iloc[0])
            if not bench_val.empty and bench_val.iloc[0] != 0:
                bench_val = bench_val / float(bench_val.iloc[0])
            benchmark = {"train": bench_train, "val": bench_val}
        else:
            dates_all = perf.index
            bench = bench.reindex(dates_all).fillna(0.0)
            benchmark = (1.0 + bench).cumprod()
    except Exception as e:
        print(f"[warn] Benchmark skipped: {e}")

    regime_label = None
    try:
        label_path = run_dir / "tables" / "regime_label.parquet"
        if label_path.exists():
            lbl = pd.read_parquet(label_path)
            if isinstance(lbl, pd.DataFrame) and "regime_label" in lbl.columns:
                regime_label = lbl["regime_label"]
            elif isinstance(lbl, pd.Series):
                regime_label = lbl
    except Exception as e:
        print(f"[warn] Regime label skipped: {e}")

    print("Stats:")
    print(f"  CAGR:        {stats['cagr']:.4f}")
    print(f"  Sharpe:      {stats['sharpe']:.4f}")
    print(f"  Max DD:      {stats['max_drawdown']:.4f}")
    print(f"  Calmar:      {stats['calmar']:.4f}")

    if perf_val is not None:
        stats_val = _stats(perf_val)
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
            m = _ic_ir_stats(s)
            ic_means.append(m["mean_ic"])
            icirs.append(m["icir"])
            print(f"  {k}: mean IC={m['mean_ic']:.4f}  ICIR={m['icir']:.4f}")
        avg_ic = float(np.nanmean(ic_means)) if ic_means else np.nan
        avg_icir = float(np.nanmean(icirs)) if icirs else np.nan
        print(f"  Average IC:  {avg_ic:.4f}")
        print(f"  Average IR:  {avg_icir:.4f}")

    if ic_series_val:
        ic_means = []
        icirs = []
        print("Validation IC/IR:")
        for k, s in ic_series_val.items():
            m = _ic_ir_stats(s)
            ic_means.append(m["mean_ic"])
            icirs.append(m["icir"])
            print(f"  {k}: mean IC={m['mean_ic']:.4f}  ICIR={m['icir']:.4f}")
        avg_ic = float(np.nanmean(ic_means)) if ic_means else np.nan
        avg_icir = float(np.nanmean(icirs)) if icirs else np.nan
        print(f"  Average IC:  {avg_ic:.4f}")
        print(f"  Average IR:  {avg_icir:.4f}")
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
