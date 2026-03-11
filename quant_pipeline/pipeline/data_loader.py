# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:31:33 2026

@author: jimya
"""
# quant_pipeline/pipeline/data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from .utils import setup_logger, ensure_datetime_index


@dataclass(frozen=True)
class DataLoaderConfig:
    project_root: Path
    raw_root_dir: Path          # e.g., data/raw
    universe_path: Optional[Path]
    start_date: str
    end_date: str
    market_proxy_ticker: str = "SPY"
    schema: str = "ohlcv-1d"
    dataset_tag: str = "xnas-itch"   # from XNAS.ITCH -> xnas-itch
    panel_path: Optional[Path] = None
    market_index_path: Optional[Path] = None
    # rates (optional now; can be added later)
    rates_path: Optional[Path] = None


class DataLoader:
    """
    Purpose
    -------
    Load raw daily OHLCV CSV per ticker/panel and build:
      (1) prices_panel: MultiIndex [date, ticker], columns: open, high, low, close, volume, ret_1d, vol_20d
      (2) market_features: Index [date], columns used by HMM: mkt_ret_1d, mkt_vol_20d, (optional y2, term_spread)

    Contract with downloader
    ------------------------
    Expected per-ticker file:
      raw_root_dir/<TICKER>/<schema>/<dataset_tag>_<TICKER>.<schema>.csv
    Example:
      .../AAPL/ohlcv-1d/xnas-itch_AAPL.ohlcv-1d.csv

    The file contains 'ts_event' (UTC timestamp) and OHLCV columns.
    """

    def __init__(self, cfg: DataLoaderConfig):
        self.cfg = cfg
        self.logger = setup_logger(self.__class__.__name__)

    def _detect_panel_path(self) -> Optional[Path]:
        """Find a single-file panel if present."""
        if self.cfg.panel_path is not None and self.cfg.panel_path.exists():
            return self.cfg.panel_path
        csv_candidate = self.cfg.raw_root_dir / "ohlcv_1d_panel.csv"
        if csv_candidate.exists():
            return csv_candidate
        parquet_candidate = self.cfg.raw_root_dir / "processed" / "prices_panel.parquet"
        if parquet_candidate.exists():
            return parquet_candidate
        return None

    def _detect_market_index_path(self) -> Optional[Path]:
        if self.cfg.market_index_path is not None and self.cfg.market_index_path.exists():
            return self.cfg.market_index_path
        csv_candidate = self.cfg.raw_root_dir / "market_index_panel.csv"
        if csv_candidate.exists():
            return csv_candidate
        return None

    def load_universe(self) -> List[str]:
        if self.cfg.universe_path is None:
            # fallback: infer from panel file or raw root
            panel_path = self._detect_panel_path()
            if panel_path is not None and panel_path.suffix.lower() in {".csv", ".parquet"}:
                if panel_path.suffix.lower() == ".csv":
                    df = pd.read_csv(panel_path, usecols=["symbol"])
                    tickers = sorted(df["symbol"].dropna().unique().tolist())
                    return tickers
                df = pd.read_parquet(panel_path)
                if isinstance(df.index, pd.MultiIndex) and "ticker" in df.index.names:
                    tickers = sorted(df.index.get_level_values("ticker").unique().tolist())
                    return tickers
            # raw-root directories
            if self.cfg.raw_root_dir.exists():
                tickers = sorted([p.name for p in self.cfg.raw_root_dir.iterdir() if p.is_dir()])
                if tickers:
                    return tickers
            raise ValueError("universe_path is required or no panel/raw universe detected")
        df = pd.read_csv(self.cfg.universe_path)
        if "ticker" not in df.columns:
            for alt in ["Symbol", "symbol", "Ticker"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "ticker"})
                    break
        tickers = sorted(df["ticker"].dropna().unique().tolist())
        return tickers

    # -------------------------
    # Raw: per-ticker CSV reader
    # -------------------------
    def _ticker_file(self, ticker: str) -> Path:
        fn = f"{self.cfg.dataset_tag}_{ticker}.{self.cfg.schema}.csv"
        return self.cfg.raw_root_dir / ticker / self.cfg.schema / fn

    def _read_single_ticker(self, ticker: str) -> pd.DataFrame:
        fp = self._ticker_file(ticker)
        if not fp.exists():
            raise FileNotFoundError(f"Missing raw file for {ticker}: {fp}")

        df = pd.read_csv(fp)

        # ts_event -> date index
        if "ts_event" not in df.columns:
            raise ValueError(f"{fp} missing 'ts_event' column")

        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts_event"])
        # daily: use date (UTC) as canonical date; later you can convert to exchange TZ if needed
        df["date"] = df["ts_event"].dt.tz_convert(None).dt.normalize()

        # Canonical OHLCV names; keep this mapping tolerant of source differences.
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in {"open", "high", "low", "close", "volume"}:
                rename_map[c] = lc
        df = df.rename(columns=rename_map)

        required = ["open", "high", "low", "close"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"{fp} missing required column '{c}' (cols={list(df.columns)})")
        if "volume" not in df.columns:
            df["volume"] = np.nan  # allow missing volume

        out = df[["date", "open", "high", "low", "close", "volume"]].copy()
        out["ticker"] = ticker
        return out

    def _read_panel_csv(self, panel_path: Path) -> pd.DataFrame:
        df = pd.read_csv(panel_path)
        if "ts_event" not in df.columns:
            raise ValueError(f"{panel_path} missing 'ts_event'")
        if "symbol" not in df.columns:
            # allow alternate name
            if "ticker" in df.columns:
                df = df.rename(columns={"ticker": "symbol"})
            else:
                raise ValueError(f"{panel_path} missing 'symbol' or 'ticker'")
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts_event"])
        df["date"] = df["ts_event"].dt.tz_convert(None).dt.normalize()
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in {"open", "high", "low", "close", "volume"}:
                rename_map[c] = lc
        df = df.rename(columns=rename_map)
        if "volume" not in df.columns:
            df["volume"] = np.nan
        out = df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
        out = out.rename(columns={"symbol": "ticker"})
        return out

    def _read_market_index_panel(self) -> Optional[pd.DataFrame]:
        panel_path = self._detect_market_index_path()
        if panel_path is None:
            return None
        if panel_path.suffix.lower() == ".csv":
            df = self._read_panel_csv(panel_path)
        else:
            df = self._read_panel_parquet(panel_path)
        return df

    def _read_panel_parquet(self, panel_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(panel_path)
        if isinstance(df.index, pd.MultiIndex) and {"date", "ticker"}.issubset(df.index.names):
            df = df.reset_index()
        if "date" not in df.columns or "ticker" not in df.columns:
            raise ValueError(f"{panel_path} missing date/ticker columns")
        return df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()

    def load_prices(self, tickers: List[str]) -> pd.DataFrame:
        """Return long table with columns: date, ticker, open, high, low, close, volume."""
        panel_path = self._detect_panel_path()
        if panel_path is not None:
            if panel_path.suffix.lower() == ".csv":
                df = self._read_panel_csv(panel_path)
            else:
                df = self._read_panel_parquet(panel_path)
            if tickers:
                df = df[df["ticker"].isin(tickers)]
            df["date"] = pd.to_datetime(df["date"])
            start = pd.to_datetime(self.cfg.start_date)
            end = pd.to_datetime(self.cfg.end_date)
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            return df

        rows = []
        for t in tickers:
            try:
                rows.append(self._read_single_ticker(t))
            except FileNotFoundError:
                # For now: log and skip missing tickers
                self.logger.warning(f"missing ticker file: {t}")
                continue

        if not rows:
            raise RuntimeError("No ticker files loaded")

        df = pd.concat(rows, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        # restrict date range
        start = pd.to_datetime(self.cfg.start_date)
        end = pd.to_datetime(self.cfg.end_date)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df

    # -------------------------
    # Canonical panels
    # -------------------------
    def build_prices_panel(self, raw_prices: pd.DataFrame) -> pd.DataFrame:
        """MultiIndex [date, ticker] with derived ret_1d and vol_20d."""
        if "date" not in raw_prices.columns or "ticker" not in raw_prices.columns:
            raise ValueError("raw_prices must have columns: date, ticker, ...")

        raw_prices["date"] = pd.to_datetime(raw_prices["date"])
        raw_prices = raw_prices.sort_values(["date", "ticker"])

        panel = raw_prices.set_index(["date", "ticker"]).sort_index()
        ensure_datetime_index(panel.reset_index().set_index("date"), "date")  # lightweight check

        # Derived: log return per ticker
        panel["ret_1d"] = (
            panel.groupby(level="ticker")["close"]
            .transform(lambda s: np.log(s).diff())
            .astype(float)
        )

        # Derived: rolling vol (std of ret)
        panel["vol_20d"] = (
            panel.groupby(level="ticker")["ret_1d"]
            .transform(lambda s: s.rolling(20, min_periods=10).std())
            .astype(float)
        )

        return panel

    def load_rates(self) -> Optional[pd.DataFrame]:
        """Optional. If not provided, market_features will omit y2/term_spread."""
        if self.cfg.rates_path is None:
            # default to treasury_yields.csv if present
            default_fp = self.cfg.raw_root_dir / "treasury_yields.csv"
            if not default_fp.exists():
                return None
            fp = default_fp
        else:
            fp = self.cfg.rates_path
        if not fp.exists():
            raise FileNotFoundError(f"rates_path not found: {fp}")
        df = pd.read_csv(fp)
        # Standardize:
        if "date" not in df.columns:
            # FRED exports sometimes use observation_date
            if "observation_date" in df.columns:
                df = df.rename(columns={"observation_date": "date"})
            else:
                raise ValueError(f"rates file missing date column: {fp}")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # Normalize column names if FRED DGS2/DGS10
        if "DGS2" in df.columns and "DGS10" in df.columns:
            df = df.rename(columns={"DGS2": "y2", "DGS10": "y10"})
            if "term_spread" not in df.columns:
                df["term_spread"] = df["y10"] - df["y2"]
        return df

    def build_market_features(self, prices_panel: pd.DataFrame, rates: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Build features used by HMM. Requires market_proxy ticker in prices_panel."""
        proxy = self.cfg.market_proxy_ticker
        mkt_panel = self._read_market_index_panel()

        # extract proxy series
        if mkt_panel is not None:
            mkt_panel["date"] = pd.to_datetime(mkt_panel["date"])
            mkt_panel = mkt_panel.set_index(["date", "ticker"]).sort_index()
            if proxy not in mkt_panel.index.get_level_values("ticker"):
                available = mkt_panel.index.get_level_values("ticker").unique().tolist()
                raise KeyError(f"market_proxy_ticker={proxy} not in market_index_panel (available={available})")
            px = mkt_panel.xs(proxy, level="ticker")
            px = px.sort_index()
            px["ret_1d"] = np.log(px["close"]).diff()
            px["vol_20d"] = px["ret_1d"].rolling(20, min_periods=10).std()
        else:
            try:
                px = prices_panel.xs(proxy, level="ticker")
            except KeyError:
                raise KeyError(f"market_proxy_ticker={proxy} not found in prices_panel")

        feats = pd.DataFrame(index=px.index)
        feats["mkt_ret_1d"] = px["ret_1d"]
        feats["mkt_vol_20d"] = px["ret_1d"].rolling(20, min_periods=10).std()

        if rates is not None:
            if "y2" in rates.columns:
                feats["y2"] = rates["y2"].reindex(feats.index)
            if "y10" in rates.columns and "y2" in rates.columns:
                feats["term_spread"] = (rates["y10"] - rates["y2"]).reindex(feats.index)
            if "term_spread" in rates.columns and "term_spread" not in feats.columns:
                feats["term_spread"] = rates["term_spread"].reindex(feats.index)

        feats = feats.sort_index()
        return feats

    def load_and_build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tickers = self.load_universe()
        # include market proxy if not in universe
        if self.cfg.market_proxy_ticker not in tickers:
            tickers = tickers + [self.cfg.market_proxy_ticker]

        raw_prices = self.load_prices(tickers)
        prices_panel = self.build_prices_panel(raw_prices)

        rates = self.load_rates()
        market_features = self.build_market_features(prices_panel, rates)

        return prices_panel, market_features
