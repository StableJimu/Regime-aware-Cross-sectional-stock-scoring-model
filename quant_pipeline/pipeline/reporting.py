from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_performance(run_dir: Path, perf_file: Path | None = None) -> pd.DataFrame:
    perf_path = perf_file if perf_file is not None else run_dir / "tables" / "performance.parquet"
    if not perf_path.exists():
        raise FileNotFoundError(f"performance.parquet not found: {perf_path}")
    return pd.read_parquet(perf_path)


def read_scores(run_dir: Path) -> pd.DataFrame:
    scores_path = run_dir / "tables" / "scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.parquet not found: {scores_path}")
    return pd.read_parquet(scores_path)


def compute_performance_stats(perf: pd.DataFrame, trading_days: int = 252) -> Dict[str, float]:
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


def benchmark_equity(bench_ret: pd.Series, anchor_equity: float) -> pd.Series:
    bench_ret = bench_ret.fillna(0.0)
    return float(anchor_equity) * (1.0 + bench_ret).cumprod()


def read_panel_for_returns(panel_path: Path) -> pd.DataFrame:
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
    df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    df["ret_1d"] = df.groupby("ticker")["close"].transform(lambda s: np.log(s).diff()).astype(float)
    df["fwd_ret_1d"] = df.groupby("ticker")["ret_1d"].shift(-1)
    return df.set_index(["date", "ticker"]).sort_index()


def compute_daily_ic(scores: pd.DataFrame, fwd_ret: pd.Series, method: str = "spearman") -> Dict[str, pd.Series]:
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


def compute_factor_daily_ic(series: pd.Series, fwd_ret: pd.Series, method: str = "spearman") -> pd.Series:
    df = pd.concat([series.rename("factor"), fwd_ret.rename("fwd_ret_1d")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def _ic_one(x: pd.DataFrame) -> float:
        if len(x) < 2:
            return np.nan
        return x["factor"].corr(x["fwd_ret_1d"], method=method)

    return df.groupby(level="date", group_keys=False).apply(_ic_one)


def compute_ic_ir_stats(ic: pd.Series) -> Dict[str, float]:
    ic = ic.dropna()
    if ic.empty:
        return {"mean_ic": np.nan, "icir": np.nan}
    mean_ic = ic.mean()
    ic_std = ic.std(ddof=0)
    icir = mean_ic / ic_std if ic_std > 0 else np.nan
    return {"mean_ic": float(mean_ic), "icir": float(icir)}


def read_split_performance(run_dir: Path, split_suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = f"_{split_suffix}" if split_suffix else ""
    train_path = run_dir / "tables" / f"performance_train{suffix}.parquet"
    val_path = run_dir / "tables" / f"performance_val{suffix}.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("performance_train.parquet or performance_val.parquet not found")
    return pd.read_parquet(train_path), pd.read_parquet(val_path)


def read_regime_label(run_dir: Path) -> pd.Series | None:
    label_path = run_dir / "tables" / "regime_label.parquet"
    if not label_path.exists():
        return None
    lbl = pd.read_parquet(label_path)
    if isinstance(lbl, pd.DataFrame) and "regime_label" in lbl.columns:
        return lbl["regime_label"]
    if isinstance(lbl, pd.Series):
        return lbl
    return None


def build_benchmark_series(
    benchmark_file: Path,
    benchmark_ticker: str,
    perf: pd.DataFrame,
    perf_val: pd.DataFrame | None = None,
) -> pd.Series | Dict[str, pd.Series]:
    bench_panel = read_panel_for_returns(benchmark_file)
    bench = bench_panel.xs(benchmark_ticker, level="ticker")["ret_1d"].dropna()
    if perf_val is not None:
        train_anchor = float(perf["equity"].iloc[0]) if "equity" in perf.columns and not perf.empty else 1.0
        val_anchor = float(perf_val["equity"].iloc[0]) if "equity" in perf_val.columns and not perf_val.empty else train_anchor
        return {
            "train": benchmark_equity(bench.reindex(perf.index), train_anchor),
            "val": benchmark_equity(bench.reindex(perf_val.index), val_anchor),
        }
    anchor = float(perf["equity"].iloc[0]) if "equity" in perf.columns and not perf.empty else 1.0
    return benchmark_equity(bench.reindex(perf.index), anchor)


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
    perf["equity"].plot(ax=ax0, color="#111111", lw=1.8, label=f"Train/All (End={end_all:.2f})")
    if benchmark is not None:
        if isinstance(benchmark, dict):
            bench_train = benchmark.get("train")
            bench_val = benchmark.get("val")
            if bench_train is not None and not bench_train.empty:
                bench_train.plot(ax=ax0, color="#1f77b4", lw=1.4, alpha=0.9, label="Benchmark (Train)")
            if bench_val is not None and not bench_val.empty:
                bench_val.plot(ax=ax0, color="#1f77b4", lw=1.4, alpha=0.9, ls="--", label="Benchmark (Val)")
        elif not benchmark.empty:
            benchmark.plot(ax=ax0, color="#1f77b4", lw=1.4, alpha=0.9, label="Benchmark")
    if perf_val is not None:
        if "equity" not in perf_val.columns:
            perf_val = perf_val.copy()
            perf_val["equity"] = (1.0 + perf_val["net"].fillna(0.0)).cumprod()
        end_val = float(perf_val["equity"].iloc[-1])
        perf_val["equity"].plot(ax=ax0, color="#d62728", lw=1.4, alpha=0.9, label=f"Validation (End={end_val:.2f})")

    if regime_label is not None:
        full_idx = perf.index.union(perf_val.index) if perf_val is not None else perf.index
        _shade_regimes(ax0, regime_label.reindex(full_idx).dropna())
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
