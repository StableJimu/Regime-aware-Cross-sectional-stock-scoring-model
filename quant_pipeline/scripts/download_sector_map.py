# -*- coding: utf-8 -*-
"""
Download a public sector proxy for the current ticker universe using Yahoo Finance
metadata via yfinance.

Output:
  - data/raw/sector_map.csv

This is a practical proxy, not a point-in-time industry history.
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
    p.add_argument("--output", type=str, default="data/raw/sector_map.csv")
    return p.parse_args()


def _to_yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


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
    return sorted(df[ticker_col].dropna().astype(str).str.strip().unique().tolist())


def main() -> None:
    args = parse_args()
    logger = setup_logger("download_sector_map")
    cfg = read_yaml(Path(args.config))
    tickers = _load_tickers(args, cfg)

    import yfinance as yf

    rows: list[dict] = []
    retrieved_at = pd.Timestamp.utcnow().isoformat()
    for i, ticker in enumerate(tickers, start=1):
        yahoo_symbol = _to_yahoo_symbol(ticker)
        logger.info(f"[{i}/{len(tickers)}] sector metadata for {ticker}")
        try:
            info = yf.Ticker(yahoo_symbol).get_info()
        except Exception as exc:
            rows.append(
                {
                    "ticker": ticker,
                    "yahoo_symbol": yahoo_symbol,
                    "sector": None,
                    "industry": None,
                    "quote_type": None,
                    "long_name": None,
                    "source": "yfinance",
                    "status": f"error: {exc}",
                    "retrieved_at": retrieved_at,
                }
            )
            continue

        rows.append(
            {
                "ticker": ticker,
                "yahoo_symbol": yahoo_symbol,
                "sector": info.get("sectorKey") or info.get("sectorDisp") or info.get("sector"),
                "industry": info.get("industryKey") or info.get("industryDisp") or info.get("industry"),
                "market_cap": info.get("marketCap"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "quote_type": info.get("quoteType"),
                "long_name": info.get("longName") or info.get("shortName"),
                "source": "yfinance",
                "status": "ok",
                "retrieved_at": retrieved_at,
            }
        )

    out = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(
        f"saved sector map: {out_path} rows={len(out):,}, "
        f"missing_sector={(out['sector'].isna()).sum():,}"
    )


if __name__ == "__main__":
    main()
