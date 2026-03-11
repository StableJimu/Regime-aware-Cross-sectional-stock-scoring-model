# -*- coding: utf-8 -*-
"""
Download FRED DGS2 and DGS10 for the dataset date range and save locally.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_PANEL = Path("data/raw/ohlcv_1d_panel.csv")
OUT_PATH = Path("data/raw/treasury_yields.csv")


def _get_date_range() -> tuple[str, str]:
    df = pd.read_csv(RAW_PANEL, usecols=["ts_event"])
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_event"])
    start = df["ts_event"].min().date().isoformat()
    end = df["ts_event"].max().date().isoformat()
    return start, end


def _fred_url(series: str, start: str, end: str) -> str:
    return (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series}&cosd={start}&coed={end}"
    )


def main() -> None:
    if not RAW_PANEL.exists():
        raise FileNotFoundError(f"raw panel not found: {RAW_PANEL}")

    start, end = _get_date_range()
    url2 = _fred_url("DGS2", start, end)
    url10 = _fred_url("DGS10", start, end)

    dgs2 = pd.read_csv(url2)
    dgs10 = pd.read_csv(url10)

    def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df

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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
