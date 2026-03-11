# Cross Sectional Stock Selection Project

Cross-sectional equity research pipeline using Alpha101-style factors, regime-aware model selection, and both full-history and train/validation backtests.

## Current Status

The current repo state has been validated locally against the workspace data.

- Factor generation writes parquet batches to `data/factors/`.
- Full backtests run end to end with a precomputed HMM regime file.
- Split backtests run end to end in both walk-forward and frozen validation modes.
- Plot generation works reliably with a non-interactive matplotlib backend.

Current caveats:

- The full backtest path still expects an existing `data/raw/market_regime_hmm.csv`.
- The standalone `run_hmm_regime.py` path can still be sensitive to the local scientific Python stack on Windows.
- The full backtest path still writes performance and metadata, but not `scores.parquet`.

## Repo Layout

- `config/config.yaml`
  Canonical full-backtest config.

- `config/config_split_walk_forward.yaml`
  Canonical split backtest config with walk-forward validation.

- `config/config_split_frozen.yaml`
  Canonical split backtest config with frozen validation.

- `quant_pipeline/pipeline/`
  Core library code for loading data, computing factors, selecting factors, fitting scoring models, and backtesting.

- `quant_pipeline/scripts/`
  CLI scripts for factor generation, HMM regime estimation, backtesting, plotting, and data download utilities.

- `data/`
  Local runtime data directory. This is intentionally ignored by Git.

- `docs/research/`
  Research PDFs and reference material.

## Data Layout

Expected local files:

- `data/raw/ohlcv_1d_panel.csv`
  Universe OHLCV panel with at least `ts_event`, `symbol`, `open`, `high`, `low`, `close`, `volume`.

- `data/raw/market_index_panel.csv`
  Market proxy panel, typically containing `SPY`.

- `data/raw/treasury_yields.csv`
  Treasury data with `date` plus either `DGS2` and `DGS10` or normalized `y2` and `y10`.

Optional local files:

- `data/raw/vix.csv`
  Used by `run_hmm_regime.py` only when `--use-vix` is passed.

- `data/raw/sector_map.csv`
  Current sector metadata proxy for the ticker universe. This is used for the expanded Alpha101 layer where sector or market-value proxies are needed.

Generated outputs:

- `data/factors/`
  Raw factor parquet batches plus `factors_manifest.json`.

- `data/backtest/<run_id>/`
  Backtest outputs, plots, holdings, scores, and run metadata.

- `data/model_params/`
  Reserved for saved model artifacts.

## Pipeline Modes

The pipeline currently runs in two distinct modes.

### Mode 1: Precompute Factors + Precompute HMM + Full Backtest

1. `compute_factors_batch.py`
   - Loads the configured universe from `ohlcv_1d_panel.csv` or raw ticker folders.
   - Splits tickers into batches.
   - Computes the selected Alpha101 factors for each batch.
   - Writes `data/factors/factors_raw_batch_*.parquet`.
   - Writes `data/factors/factors_manifest.json`.

2. `run_hmm_regime.py`
   - Builds market features from `market_index_panel.csv` and `treasury_yields.csv`.
   - Uses rolling or full-sample Gaussian HMM fitting.
   - Writes `data/raw/market_regime_hmm.csv` with `regime` and `prob_1..prob_K`.

3. `run_backtest.py`
   - Loads prices and market features via `DataLoader`.
   - Reads the precomputed factor manifest and concatenates raw factor batches.
   - Applies cross-sectional factor transforms using `FactorProcessor.transform_cross_section`.
   - Reads the precomputed HMM probabilities from `data/raw/market_regime_hmm.csv`.
   - Selects factors per regime or globally, depending on config.
   - Fits one model per regime.
   - Ensembles regime model predictions using HMM probabilities.
   - Runs one or more stress backtests across transaction cost / short fee scenarios.
   - Writes performance files under `data/backtest/<run_id>/tables/`.

Important detail:
- `run_backtest.py` does not fit the HMM itself. It expects `market_regime_hmm.csv` to already exist.
- This path has been validated locally.

### Mode 2: Train/Validation Split Backtest

1. `compute_factors_batch.py`
   - Same as above. Split backtests also consume the factor manifest.

2. `run_backtest_split.py`
   - Loads prices and market features.
   - Reads precomputed factor batches.
   - Splits the sample into train and validation windows.
   - Fits the market HMM inside the script using only the training window.
   - Builds regime probabilities and labels for the full date range.
   - Selects factors using only the training data.
   - Fits regime models on the training window.
   - Scores both train and validation windows.
   - Backtests train and validation separately across stress scenarios.
   - Writes:
     - `performance_train_*.parquet`
     - `performance_val_*.parquet`
     - `holdings_train_*.csv`
     - `holdings_val_*.csv`
     - `scores.parquet`
     - `regime_label.parquet`
     - `regime_proba.parquet`

Important detail:
- `run_backtest_split.py` fits the HMM inside the split workflow. It does not require a pre-existing `market_regime_hmm.csv`.
- The split path supports both `validation_mode: walk_forward` and `validation_mode: frozen`.
- This path has been validated locally with a 5-year train / 2-year validation run.

## Factor Layer

The factor library is defined in `quant_pipeline/pipeline/alpha101_factors.py`.

Current behavior:

- Factors are computed as raw panel-level Alpha101-style signals.
- The selected factors are controlled by `config/config.yaml`.
- Cross-sectional post-processing happens after loading factor batches:
  - winsorization
  - cross-sectional rank transform by default
  - z-score mode is available but not the default

## Model Layer

The scoring model supports:

- `ridge`
  - closed-form linear ridge regression

- `glm`
  - Gaussian GLM implemented as regularized OLS

- `lstm`
  - PyTorch LSTM sequence model

Current selection logic:

- `regime_stepwise`
  - orthogonal stepwise IC-based selection per regime

- `global_stepwise_ridge`
  - global factor selection followed by regime-specific model fitting

Labels:

- The supervised label is forward next-period return based on:
  - `model.label_horizon`
  - `backtest.signal_lag`

Optional behavior:

- factor lagging via `model.factor_lag`
- rolling retraining via `model.rolling_train_days`
- soft regime weighting in split backtests
- linear score calibration via `model.score_calibration: linear`

Important detail:
- Rank-based backtests such as `top_q` use `score_raw` for ranking.
- The calibrated score is retained only as a diagnostic field for future optimization-based portfolio construction such as `robust_spo`.

## Backtest Layer

Implemented portfolio methods:

- `top_q`
  - long-only or long-short quantile portfolio construction

- `robust_spo`
  - convex optimization portfolio with turnover, uncertainty, and leverage controls

Backtester behavior:

- Uses `ret_1d` as log return internally and converts to simple return for PnL.
- Applies signal lag before return realization.
- Applies turnover-based transaction costs.
- Applies daily short borrow fees when configured.

## Quick Start

### 1. Download local data inputs

Prices and market ETFs:

```bash
python -m quant_pipeline.scripts.download_prices
```

VIX:

```bash
python -m quant_pipeline.scripts.download_vix
```

Treasury yields:

```bash
python -m quant_pipeline.scripts.download_fred_yields
```

Sector metadata:

```bash
python -m quant_pipeline.scripts.download_sector_map
```

### 2. Compute factors

```bash
python -m quant_pipeline.scripts.compute_factors_batch
```

### 3. Build the standalone HMM for the full backtest

```bash
python -m quant_pipeline.scripts.run_hmm_regime --k 3 --use-vix
```

### 4. Run backtests

Full backtest:

```bash
python -m quant_pipeline.scripts.run_backtest --stress-level all
```

Walk-forward split backtest:

```bash
python -m quant_pipeline.scripts.run_backtest_split \
  --config config/config_split_walk_forward.yaml \
  --train-years 5 \
  --val-years 2 \
  --stress-level all
```

Frozen split backtest:

```bash
python -m quant_pipeline.scripts.run_backtest_split \
  --config config/config_split_frozen.yaml \
  --train-years 5 \
  --val-years 2 \
  --stress-level low
```

### 5. Plot results

Full run:

```bash
python -m quant_pipeline.scripts.plot_performance --run-dir data/backtest/<run_id>
```

Split run:

```bash
python -m quant_pipeline.scripts.plot_performance \
  --run-dir data/backtest_split_walk_forward/<run_id> \
  --split \
  --split-suffix cost0_fee0
```

In this environment, plotting is most reliable with a non-interactive backend:

```bash
set MPLBACKEND=Agg
python -m quant_pipeline.scripts.plot_performance --run-dir data/backtest/<run_id> --no-show
```

## Data Sources

- `yfinance`
  - daily OHLCV proxy for the fixed ticker universe
  - market proxy ETF panel
  - ticker metadata used as a sector and market-value proxy

- `Cboe VIX history CSV`
  - public VIX history download

- `FRED`
  - public treasury yield series

Important caveats:

- Yahoo Finance is a practical daily data source, not an exchange-grade source.
- The sector map generated from Yahoo metadata is current or recent metadata, not
  point-in-time historical GICS history.
- This is good enough for continuing research on the current fixed universe, but
  it is not a substitute for institutional-quality historical reference data.

Compute IC for specific factors:

```bash
python -m quant_pipeline.scripts.compute_factor_ic --factors alpha_054,alpha_025,alpha_033
```

## Configuration

The config surface is intentionally small:

- `config/config.yaml`
  - full backtest

- `config/config_split_walk_forward.yaml`
  - split backtest with rolling refits during validation using only prior data

- `config/config_split_frozen.yaml`
  - split backtest with train-only fitting and fixed validation scoring

Important config sections in `config/config.yaml`:

- `paths`
  - location of raw data, factor outputs, and backtest outputs

- `factors`
  - factor list and factor batch size

- `regime`
  - number of HMM states and regime-feature definitions

- `model`
  - selection method, factor lag, regime usage, model family, rolling windows

- `backtest`
  - quantile cutoffs, costs, leverage, robustness controls, and stress scenarios

## Dependency Notes

Core runtime dependencies:

- `numpy`
- `pandas`
- `PyYAML`
- `pyarrow`
- `scikit-learn`

Used by optional or non-default paths:

- `matplotlib`
  - required for `plot_performance.py`

- `yfinance`
  - required for the public price and sector download scripts

- `torch`
  - required only when `model.model_family: lstm`

- `cvxpy`, `scs`, `osqp`
  - required only when `backtest.portfolio_method: robust_spo`

## Current Limitations

- The full backtest path relies on a separately generated HMM CSV rather than fitting the regime model inline.
- The standalone HMM script can still fail in some Windows scientific Python environments because of the `sklearn` / `threadpoolctl` stack.
- The full backtest path currently does not save `scores.parquet`, so full-run IC/IR plotting is limited.
- Sector metadata is currently a static current snapshot and is not point-in-time.
- There is not yet a formal `tests/` suite.
