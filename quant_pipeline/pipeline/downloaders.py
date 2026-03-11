from __future__ import annotations

from pathlib import Path

import pandas as pd


CBOE_VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def load_tickers(
    cfg: dict,
    tickers_file: str | None = None,
    panel_path: str | None = None,
) -> list[str]:
    if tickers_file:
        path = Path(tickers_file)
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
            panel = Path(panel_path or cfg["paths"].get("panel_path") or "data/raw/ohlcv_1d_panel.csv")
            if not panel.exists():
                raise FileNotFoundError(
                    "No universe source found. Provide --tickers-file or set paths.universe_path, "
                    f"or keep an existing panel at {panel}."
                )
            df = pd.read_csv(panel, usecols=["symbol"])

    ticker_col = next((c for c in ("ticker", "symbol", "Symbol", "Ticker") if c in df.columns), None)
    if ticker_col is None:
        raise ValueError(f"Could not find ticker column in source: {list(df.columns)}")
    tickers = sorted(df[ticker_col].dropna().astype(str).str.strip().unique().tolist())
    if not tickers:
        raise ValueError("Ticker universe is empty")
    return tickers


def to_yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def to_pipeline_symbol(yahoo_symbol: str) -> str:
    return yahoo_symbol.replace("-", ".")


def batch_items(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def normalize_download_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
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
    return out[["ts_event", "open", "high", "low", "close", "volume", "symbol"]]


def download_yahoo_batch(
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
            frames.append(normalize_download_frame(data[yahoo_symbol], to_pipeline_symbol(yahoo_symbol)))
    else:
        if len(yahoo_symbols) != 1:
            raise ValueError("Expected a single symbol result for non-MultiIndex download")
        frames.append(normalize_download_frame(data, to_pipeline_symbol(yahoo_symbols[0])))

    if not frames:
        return pd.DataFrame(columns=["ts_event", "open", "high", "low", "close", "volume", "symbol"])
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts_event"]).reset_index(drop=True)


def run_price_download_workflow(
    cfg: dict,
    logger,
    tickers_file: str | None = None,
    panel_path: str | None = None,
    start: str | None = None,
    end: str | None = None,
    batch_size: int = 50,
    pause_seconds: float = 0.25,
    market_symbols: str = "SPY,VTI,DIA,QQQ",
    output_panel: str | None = None,
    output_market: str | None = None,
) -> tuple[Path, Path]:
    tickers = load_tickers(cfg, tickers_file=tickers_file, panel_path=panel_path)
    market_symbol_list = [s.strip().upper() for s in market_symbols.split(",") if s.strip()]
    all_symbols = sorted(set(tickers) | set(market_symbol_list))
    yahoo_symbols = [to_yahoo_symbol(s) for s in all_symbols]

    start_date = start or cfg["data"]["start_date"]
    end_date = end or cfg["data"]["end_date"]
    end_exclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).date().isoformat()

    panel_out = Path(output_panel or cfg["paths"].get("panel_path") or "data/raw/ohlcv_1d_panel.csv")
    market_out = Path(output_market or "data/raw/market_index_panel.csv")

    logger.info(f"symbols={len(all_symbols)}, market_symbols={market_symbol_list}, start={start_date}, end={end_date}")
    frames: list[pd.DataFrame] = []
    for batch in batch_items(yahoo_symbols, max(1, batch_size)):
        logger.info(f"downloading batch size={len(batch)}")
        frames.append(
            download_yahoo_batch(
                yahoo_symbols=batch,
                start=start_date,
                end_exclusive=end_exclusive,
                pause_seconds=pause_seconds,
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

    market_panel = panel[panel["symbol"].isin(market_symbol_list)].copy()
    equity_panel = panel[~panel["symbol"].isin(market_symbol_list)].copy()

    panel_out.parent.mkdir(parents=True, exist_ok=True)
    market_out.parent.mkdir(parents=True, exist_ok=True)
    equity_panel.to_csv(panel_out, index=False)
    market_panel.to_csv(market_out, index=False)

    logger.info(f"saved equities: {panel_out} rows={len(equity_panel):,}")
    logger.info(f"saved market: {market_out} rows={len(market_panel):,}")
    return panel_out, market_out


def run_sector_map_download_workflow(
    cfg: dict,
    logger,
    tickers_file: str | None = None,
    panel_path: str | None = None,
    output: str = "data/raw/sector_map.csv",
) -> Path:
    import yfinance as yf

    tickers = load_tickers(cfg, tickers_file=tickers_file, panel_path=panel_path)
    rows: list[dict] = []
    retrieved_at = pd.Timestamp.utcnow().isoformat()
    for i, ticker in enumerate(tickers, start=1):
        yahoo_symbol = to_yahoo_symbol(ticker)
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
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"saved sector map: {out_path} rows={len(out):,}, missing_sector={(out['sector'].isna()).sum():,}")
    return out_path


def run_vix_download_workflow(cfg: dict, logger, output: str | None = None) -> Path:
    out_path = Path(output or cfg["paths"].get("vix_path") or "data/raw/vix.csv")
    df = pd.read_csv(CBOE_VIX_HISTORY_URL)
    cols = {str(c).strip().upper(): c for c in df.columns}
    date_col = cols.get("DATE")
    close_col = cols.get("CLOSE")
    if date_col is None or close_col is None:
        raise ValueError(f"Unexpected VIX columns: {list(df.columns)}")

    out = df[[date_col, close_col]].copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"saved vix history: {out_path} rows={len(out):,}")
    return out_path


def get_panel_date_range(raw_panel: Path) -> tuple[str, str]:
    if not raw_panel.exists():
        raise FileNotFoundError(f"raw panel not found: {raw_panel}")
    df = pd.read_csv(raw_panel, usecols=["ts_event"])
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_event"])
    start = df["ts_event"].min().date().isoformat()
    end = df["ts_event"].max().date().isoformat()
    return start, end


def fred_url(series: str, start: str, end: str) -> str:
    return "https://fred.stlouisfed.org/graph/fredgraph.csv" f"?id={series}&cosd={start}&coed={end}"


def run_fred_yields_download_workflow(
    logger,
    raw_panel: str = "data/raw/ohlcv_1d_panel.csv",
    output: str = "data/raw/treasury_yields.csv",
) -> Path:
    start, end = get_panel_date_range(Path(raw_panel))
    dgs2 = pd.read_csv(fred_url("DGS2", start, end))
    dgs10 = pd.read_csv(fred_url("DGS10", start, end))

    def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out

    dgs2 = _normalize_cols(dgs2)
    dgs10 = _normalize_cols(dgs10)

    date_col2 = "DATE" if "DATE" in dgs2.columns else "observation_date" if "observation_date" in dgs2.columns else None
    date_col10 = "DATE" if "DATE" in dgs10.columns else "observation_date" if "observation_date" in dgs10.columns else None
    if date_col2 is None or "DGS2" not in dgs2.columns:
        raise ValueError(f"Unexpected DGS2 columns: {list(dgs2.columns)}")
    if date_col10 is None or "DGS10" not in dgs10.columns:
        raise ValueError(f"Unexpected DGS10 columns: {list(dgs10.columns)}")

    dgs2 = dgs2.rename(columns={date_col2: "date", "DGS2": "DGS2"})
    dgs10 = dgs10.rename(columns={date_col10: "date", "DGS10": "DGS10"})
    df = pd.merge(dgs2, dgs10, on="date", how="outer").sort_values("date")
    df["term_spread"] = df["DGS10"] - df["DGS2"]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"saved treasury yields: {out_path} rows={len(df):,}")
    return out_path
