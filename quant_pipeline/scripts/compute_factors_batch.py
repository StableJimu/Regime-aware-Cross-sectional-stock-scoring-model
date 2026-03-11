# -*- coding: utf-8 -*-
"""
Compute factors in ticker batches and store raw factors to parquet files.
This avoids loading the full universe in memory at once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd

from quant_pipeline.pipeline.utils import read_yaml, setup_logger
from quant_pipeline.pipeline.data_loader import DataLoader, DataLoaderConfig
from quant_pipeline.pipeline.factor_processing import FactorProcessor, FactorProcessorConfig, FactorRegistry
from quant_pipeline.pipeline.alpha101_factors import register_alpha101


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


def _batch(iterable: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    return p.parse_args()


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

    for i, tick_batch in enumerate(_batch(tickers, batch_size), start=1):
        logger.info(f"batch {i}: {len(tick_batch)} tickers")
        raw_prices = _chunked_read_panel_csv(panel_path, tick_batch)
        if raw_prices.empty:
            logger.warning(f"batch {i}: empty raw_prices")
            continue
        prices_panel = loader.build_prices_panel(raw_prices)
        factor_panel_raw = factor_proc.compute_factors(prices_panel, factor_names=selected_factors)
        out_path = out_dir / f"factors_raw_batch_{i:03d}.parquet"
        factor_panel_raw.to_parquet(out_path, index=True)
        manifest["batches"].append(str(out_path))

    manifest_path = out_dir / "factors_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
