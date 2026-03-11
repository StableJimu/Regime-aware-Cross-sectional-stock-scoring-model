# -*- coding: utf-8 -*-
"""
Compute factors in ticker batches and store raw factors to parquet files.
This avoids loading the full universe in memory at once.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd

from quant_pipeline.pipeline.utils import read_yaml, setup_logger
from quant_pipeline.pipeline.data_loader import DataLoader, DataLoaderConfig
from quant_pipeline.pipeline.factor_processing import FactorProcessor, FactorProcessorConfig, FactorRegistry
from quant_pipeline.pipeline.alpha101_factors import register_alpha101


ALPHA101_METADATA_FACTORS = {
    "alpha_048", "alpha_056", "alpha_058", "alpha_059", "alpha_063", "alpha_067",
    "alpha_069", "alpha_070", "alpha_076", "alpha_079", "alpha_080", "alpha_082",
    "alpha_087", "alpha_089", "alpha_090", "alpha_091", "alpha_093", "alpha_097",
    "alpha_100",
}


def _chunked_read_panel_csv(panel_path: Path, tickers: List[str], chunksize: int = 1_000_000) -> pd.DataFrame:
    usecols = ["ts_event", "symbol", "open", "high", "low", "close", "volume"]
    out = []
    tickers_set = set(tickers)
    for chunk in pd.read_csv(panel_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["symbol"].isin(tickers_set)]
        if not chunk.empty:
            out.append(chunk)
    if not out:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(out, ignore_index=True)
    df = df.rename(columns={"symbol": "ticker"})
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_event"])
    df["date"] = df["ts_event"].dt.tz_convert(None).dt.normalize()
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def _load_sector_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"sector map must contain ticker column: {path}")
    keep = [c for c in ["ticker", "sector", "industry", "market_cap", "shares_outstanding"] if c in df.columns]
    if "ticker" not in keep:
        keep = ["ticker"] + keep
    df = df[keep].copy()
    return df.drop_duplicates(subset=["ticker"], keep="last")


def _batch(iterable: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--force", action="store_true", help="Recompute existing batch parquet files")
    return p.parse_args()


def _existing_batch_numbers(out_dir: Path) -> set[int]:
    nums: set[int] = set()
    for path in out_dir.glob("factors_raw_batch_*.parquet"):
        m = re.match(r"factors_raw_batch_(\d+)\.parquet$", path.name)
        if m:
            nums.add(int(m.group(1)))
    return nums


def main() -> None:
    args = parse_args()
    logger = setup_logger("compute_factors_batch")
    cfg = read_yaml(Path(args.config))

    dl_cfg = DataLoaderConfig(
        project_root=Path(cfg["paths"]["project_root"]),
        raw_root_dir=Path(cfg["paths"]["raw_dir"]),
        universe_path=Path(cfg["paths"]["universe_path"]) if cfg["paths"].get("universe_path") else None,
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        panel_path=Path(cfg["paths"]["panel_path"]) if cfg["paths"].get("panel_path") else None,
        market_index_path=Path(cfg["paths"]["market_index_path"]) if cfg["paths"].get("market_index_path") else None,
        market_proxy_ticker=cfg["data"].get("market_proxy_ticker", "SPY"),
    )
    loader = DataLoader(dl_cfg)
    tickers = loader.load_universe()

    registry = FactorRegistry()
    register_alpha101(registry)
    fp_cfg = FactorProcessorConfig(factors_dir=Path(cfg["paths"]["factors_dir"]))
    factor_proc = FactorProcessor(fp_cfg, registry)

    selected_factors = cfg.get("factors", {}).get("selected", [
        "alpha_001", "alpha_002", "alpha_003", "alpha_004", "alpha_005",
        "alpha_006", "alpha_007", "alpha_008", "alpha_009", "alpha_010",
    ])

    sector_map = None
    sector_map_path = Path(cfg["paths"].get("sector_map_path", "data/raw/sector_map.csv"))
    if any(f in ALPHA101_METADATA_FACTORS for f in selected_factors):
        if not sector_map_path.exists():
            raise FileNotFoundError(
                f"sector_map_path is required for metadata-aware Alpha101 factors: {sector_map_path}"
            )
        sector_map = _load_sector_map(sector_map_path)
        logger.info(f"loaded sector metadata: {sector_map_path} rows={len(sector_map):,}")

    batch_size = int(cfg.get("factors", {}).get("batch_size", 200))
    out_dir = Path(cfg["paths"]["factors_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(cfg["paths"]["panel_path"])
    if not panel_path.exists():
        raise FileNotFoundError(f"panel_path not found: {panel_path}")

    manifest: Dict[str, object] = {
        "selected_factors": selected_factors,
        "batches": [],
    }
    existing = _existing_batch_numbers(out_dir)

    for i, tick_batch in enumerate(_batch(tickers, batch_size), start=1):
        out_path = out_dir / f"factors_raw_batch_{i:03d}.parquet"
        if not args.force and i in existing and out_path.exists():
            logger.info(f"batch {i}: using existing {out_path.name}")
            manifest["batches"].append(str(out_path))
            continue
        logger.info(f"batch {i}: {len(tick_batch)} tickers")
        raw_prices = _chunked_read_panel_csv(panel_path, tick_batch)
        if raw_prices.empty:
            logger.warning(f"batch {i}: empty raw_prices")
            continue
        if sector_map is not None:
            raw_prices = raw_prices.merge(sector_map, on="ticker", how="left")
            if "shares_outstanding" in raw_prices.columns:
                raw_prices["market_value"] = raw_prices["shares_outstanding"] * raw_prices["close"]
            elif "market_cap" in raw_prices.columns:
                raw_prices["market_value"] = raw_prices["market_cap"]
        prices_panel = loader.build_prices_panel(raw_prices)
        factor_panel_raw = factor_proc.compute_factors(prices_panel, factor_names=selected_factors)
        factor_panel_raw.to_parquet(out_path, index=True)
        manifest["batches"].append(str(out_path))

    manifest_path = out_dir / "factors_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
