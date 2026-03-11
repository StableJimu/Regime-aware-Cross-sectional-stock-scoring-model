# -*- coding: utf-8 -*-
"""
Download public daily OHLCV data using Yahoo Finance via yfinance.

Outputs:
  - data/raw/ohlcv_1d_panel.csv
  - data/raw/market_index_panel.csv

This is the default daily price downloader. It keeps the file
schema expected by DataLoader:
  ts_event, open, high, low, close, volume, symbol
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_pipeline.pipeline.utils import read_yaml, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--tickers-file", type=str, default=None)
    p.add_argument("--panel-path", type=str, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--pause-seconds", type=float, default=0.25)
    p.add_argument("--market-symbols", type=str, default="SPY,VTI,DIA,QQQ")
    p.add_argument("--output-panel", type=str, default=None)
    p.add_argument("--output-market", type=str, default=None)
    return p.parse_args()


def _load_tickers(args: argparse.Namespace, cfg: dict) -> list[str]:
    if args.tickers_file:
        path = Path(args.tickers_file)
        if not path.exists():
            raise FileNotFoundError(f"tickers file not found: {path}")
        df = pd.read_csv(path)
    else:
        universe_path = cfg["paths"].get("universe_path")
        if universe_path:
            path = Path(universe_path)
            if not path.exists():
                raise FileNotFoundError(f"universe file not found: {path}")
            df = pd.read_csv(path)
        else:
            panel_path = Path(args.panel_path or cfg["paths"].get("panel_path") or "data/raw/ohlcv_1d_panel.csv")
            if not panel_path.exists():
                raise FileNotFoundError(
                    "No universe source found. Provide --tickers-file or set paths.universe_path, "
                    f"or keep an existing panel at {panel_path}."
                )
            df = pd.read_csv(panel_path, usecols=["symbol"])

    ticker_col = None
    for col in ("ticker", "symbol", "Symbol", "Ticker"):
        if col in df.columns:
            ticker_col = col
            break
    if ticker_col is None:
        raise ValueError(f"Could not find ticker column in source: {list(df.columns)}")
    tickers = sorted(df[ticker_col].dropna().astype(str).str.strip().unique().tolist())
    if not tickers:
        raise ValueError("Ticker universe is empty")
    return tickers


def _to_yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def _to_pipeline_symbol(yahoo_symbol: str) -> str:
    return yahoo_symbol.replace("-", ".")


def _batch(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _normalize_download_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return pd.DataFrame(columns=["ts_event", "open", "high", "low", "close", "volume", "symbol"])

    out = out.rename_axis("date").reset_index()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in out.columns]
    out = out[keep].copy()
    if "volume" not in out.columns:
        out["volume"] = pd.NA
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    if out.empty:
        return pd.DataFrame(columns=["ts_event", "open", "high", "low", "close", "volume", "symbol"])

    out["ts_event"] = out["date"].dt.tz_localize("UTC")
    out["symbol"] = symbol
    out = out[["ts_event", "open", "high", "low", "close", "volume", "symbol"]]
    return out


def _download_batch(
    yahoo_symbols: list[str],
    start: str,
    end_exclusive: str,
    pause_seconds: float,
    logger,
) -> pd.DataFrame:
    import time
    import yfinance as yf

    data = yf.download(
        tickers=yahoo_symbols,
        start=start,
        end=end_exclusive,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    time.sleep(max(0.0, pause_seconds))

    if data.empty:
        logger.warning(f"empty download for batch size={len(yahoo_symbols)}")
        return pd.DataFrame(columns=["ts_event", "open", "high", "low", "close", "volume", "symbol"])

    frames: list[pd.DataFrame] = []
    if isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        for yahoo_symbol in yahoo_symbols:
            if yahoo_symbol not in available:
                logger.warning(f"missing symbol in batch result: {yahoo_symbol}")
                continue
            frames.append(_normalize_download_frame(data[yahoo_symbol], _to_pipeline_symbol(yahoo_symbol)))
    else:
        if len(yahoo_symbols) != 1:
            raise ValueError("Expected a single symbol result for non-MultiIndex download")
        frames.append(_normalize_download_frame(data, _to_pipeline_symbol(yahoo_symbols[0])))

    if not frames:
        return pd.DataFrame(columns=["ts_event", "open", "high", "low", "close", "volume", "symbol"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "ts_event"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    logger = setup_logger("download_prices")
    cfg = read_yaml(Path(args.config))

    tickers = _load_tickers(args, cfg)
    market_symbols = [s.strip().upper() for s in args.market_symbols.split(",") if s.strip()]
    all_symbols = sorted(set(tickers) | set(market_symbols))
    yahoo_symbols = [_to_yahoo_symbol(s) for s in all_symbols]

    start = args.start or cfg["data"]["start_date"]
    end = args.end or cfg["data"]["end_date"]
    end_exclusive = (pd.to_datetime(end) + pd.Timedelta(days=1)).date().isoformat()

    output_panel = Path(args.output_panel or cfg["paths"].get("panel_path") or "data/raw/ohlcv_1d_panel.csv")
    output_market = Path(args.output_market or "data/raw/market_index_panel.csv")

    logger.info(f"symbols={len(all_symbols)}, market_symbols={market_symbols}, start={start}, end={end}")
    frames: list[pd.DataFrame] = []
    for batch in _batch(yahoo_symbols, max(1, args.batch_size)):
        logger.info(f"downloading batch size={len(batch)}")
        frames.append(
            _download_batch(
                yahoo_symbols=batch,
                start=start,
                end_exclusive=end_exclusive,
                pause_seconds=args.pause_seconds,
                logger=logger,
            )
        )

    if not frames:
        raise RuntimeError("No price data downloaded")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(subset=["symbol", "ts_event"], keep="last")
    panel = panel.sort_values(["ts_event", "symbol"]).reset_index(drop=True)
    if panel.empty:
        raise RuntimeError("Downloaded panel is empty after normalization")

    market_panel = panel[panel["symbol"].isin(market_symbols)].copy()
    equity_panel = panel[~panel["symbol"].isin(market_symbols)].copy()

    output_panel.parent.mkdir(parents=True, exist_ok=True)
    output_market.parent.mkdir(parents=True, exist_ok=True)
    equity_panel.to_csv(output_panel, index=False)
    market_panel.to_csv(output_market, index=False)

    logger.info(f"saved equities: {output_panel} rows={len(equity_panel):,}")
    logger.info(f"saved market: {output_market} rows={len(market_panel):,}")


if __name__ == "__main__":
    main()
