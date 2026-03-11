# Cross Sectional Stock Selection Project

Validated quant research pipeline for cross-sectional equity selection using Alpha101-style factors, regime-aware model selection, train/validation backtesting, and full-history backtesting.

## Current Status

The project has now been exercised locally against the current workspace data.

- Factor generation works and writes outputs to `data/factors/`.
- Train/validation split backtests work end-to-end and write outputs under a separate run folder.
- Full backtests work end-to-end using a precomputed regime file.
- Plot generation works when a non-interactive matplotlib backend is used.

Current caveats:

- The full backtest path still depends on an existing `data/raw/market_regime_hmm.csv`.
- The standalone `run_hmm_regime.py` path may still be environment-sensitive because of the local scientific Python stack.
- The full backtest path currently writes performance outputs and metadata, but not `scores.parquet`.

## Current Structure

- `config/config.yaml`
  Central runtime configuration for paths, dates, factors, model selection, and backtest settings.

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

Generated outputs:

- `data/factors/`
  Raw factor parquet batches plus `factors_manifest.json`.

- `data/backtest/<run_id>/`
  Backtest outputs, plots, holdings, scores, and run metadata.

- `data/model_params/`
  Reserved for saved model artifacts.

## How The Current Pipeline Works

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
- This path has been validated locally after fixing the broken backtest loop in `run_backtest.py`.

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
- `run_backtest_split.py` does fit the HMM itself. It does not require a pre-existing `market_regime_hmm.csv`.
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

## Typical Commands

### 1. Compute factor batches

```bash
python -m quant_pipeline.scripts.compute_factors_batch
```

Outputs:

```text
data/factors/factors_raw_batch_*.parquet
data/factors/factors_manifest.json
```

### 2. Fit standalone market HMM

```bash
python -m quant_pipeline.scripts.run_hmm_regime --k 3
```

Optional VIX input:

```bash
python -m quant_pipeline.scripts.run_hmm_regime --k 3 --use-vix
```

Output:

```text
data/raw/market_regime_hmm.csv
```

### 3. Run full backtest using precomputed HMM

```bash
python -m quant_pipeline.scripts.run_backtest --stress-level all
```

Stress level options:

- `low`
- `medium`
- `high`
- `all`

### 4. Run train/validation backtest

```bash
python -m quant_pipeline.scripts.run_backtest_split --train-years 5 --val-years 2 --stress-level high
```

Example validated no-cost split config:

```bash
python -m quant_pipeline.scripts.run_backtest_split \
  --config config/config_split_no_turnover.yaml \
  --train-years 5 \
  --val-years 2 \
  --stress-level low
```

### 5. Plot outputs

Single-run performance:

```bash
python -m quant_pipeline.scripts.plot_performance --run-dir data/backtest/<run_id>
```

Split backtest plot:

```bash
python -m quant_pipeline.scripts.plot_performance --run-dir data/backtest/<run_id> --split --split-suffix cost20_fee300
```

In this environment, plotting is most reliable with a non-interactive backend:

```bash
set MPLBACKEND=Agg
python -m quant_pipeline.scripts.plot_performance --run-dir data/backtest/<run_id> --split --split-suffix cost0_fee0 --no-show
```

### 6. Compute IC for specific factors

```bash
python -m quant_pipeline.scripts.compute_factor_ic --factors alpha_054,alpha_025,alpha_033
```

## Current Configuration Knobs

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

- `databento`
  - required for the download scripts

- `torch`
  - required only when `model.model_family: lstm`

- `cvxpy`, `scs`, `osqp`
  - required only when `backtest.portfolio_method: robust_spo`

## Current Limitations

- The package entrypoint in `quant_pipeline/scripts/main.py` is only a thin backtest wrapper.
- The full backtest path relies on a separately generated HMM CSV rather than fitting the regime model inline.
- The standalone HMM script can still fail in some Windows scientific Python environments because of the `sklearn` / `threadpoolctl` stack.
- The full backtest path currently does not save `scores.parquet`, so full-run IC/IR plotting is limited.
- Several downloader scripts still contain user-specific assumptions beyond the new path layout.
- There is not yet a formal `tests/` suite.
