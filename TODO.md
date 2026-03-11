# Project TODO

## Priority 1: Data Completeness

- Add `data/raw/vix.csv` as a standard supported input instead of relying on a realized-vol proxy fallback.
- Add sector or industry classification data keyed by ticker and date so industry-neutral alphas can be implemented.
- Add market cap inputs such as shares outstanding or a precomputed `market_cap` field.
- Define and document required raw data schemas for:
  - `ohlcv_1d_panel.csv`
  - `market_index_panel.csv`
  - `treasury_yields.csv`
  - `vix.csv`
  - sector or industry mapping files
- Review the current universe construction for survivorship bias and delisting coverage.
- Validate corporate action handling and price adjustment assumptions.

## Priority 2: Regime Model Refactor

- Move HMM fitting and inference logic out of scripts and into [`quant_pipeline/pipeline/regime_model.py`](/C:/Users/jimya/Projects/Cross sectional stock selection Project/quant_pipeline/pipeline/regime_model.py).
- Make full and split backtests call the same reusable regime-model API.
- Remove duplicated HMM code currently split between:
  - [`quant_pipeline/scripts/run_hmm_regime.py`](/C:/Users/jimya/Projects/Cross sectional stock selection Project/quant_pipeline/scripts/run_hmm_regime.py)
  - [`quant_pipeline/scripts/run_backtest_split.py`](/C:/Users/jimya/Projects/Cross sectional stock selection Project/quant_pipeline/scripts/run_backtest_split.py)
- Replace the placeholder single-regime implementation in [`quant_pipeline/pipeline/regime_model.py`](/C:/Users/jimya/Projects/Cross sectional stock selection Project/quant_pipeline/pipeline/regime_model.py) with the actual multi-regime implementation.
- Make the number of regimes fully config-driven in the full backtest instead of assuming exactly 3 regimes.

## Priority 3: Backtest Consistency

- Make full and split backtests write a consistent artifact set:
  - performance tables
  - holdings
  - scores
  - regime labels
  - regime probabilities
  - run metadata
- Make full backtests save `scores.parquet` so plotting and IC diagnostics work consistently.
- Standardize run directory structure and file naming conventions across all pipeline modes.
- Add explicit benchmark outputs for comparison against SPY or another configured benchmark.

## Priority 4: Alpha and Feature Expansion

- Extend the factor library beyond the current OHLCV-only Alpha101 subset.
- Implement sector-neutral and industry-neutral variants once classification data is available.
- Add size-aware and liquidity-aware features once market-cap data is available.
- Add feature availability checks so unsupported alphas fail clearly instead of silently being omitted.
- Document which Alpha101 factors are implemented, which are intentionally excluded, and why.

## Priority 5: Research Diagnostics

- Add IC and IR summaries by factor and by regime.
- Add turnover diagnostics by run and by rebalance date.
- Add exposure diagnostics by ticker, sector, and regime.
- Add benchmark-relative performance metrics.
- Add train-versus-validation drift diagnostics for split runs.
- Add summary reports comparing stress scenarios in one place.

## Priority 6: Data and Config Validation

- Add schema validation for required raw inputs before long runs begin.
- Validate date coverage and alignment across prices, rates, VIX, and market index inputs.
- Validate factor manifest contents before backtests start.
- Add config validation for incompatible settings and missing paths.
- Fail early with clear error messages for partial or stale data inputs.

## Priority 7: Testing

- Add a `tests/` directory.
- Add unit tests for:
  - data loading
  - factor computation shape and null handling
  - regime model output shape and probability normalization
  - backtester portfolio construction
- Add a small synthetic dataset for end-to-end smoke tests.
- Add at least one CI-friendly smoke test covering factor generation and a minimal backtest.

## Priority 8: Repository and Developer Experience

- Clean up downloader scripts to remove user-specific assumptions and comments.
- Add a clearer CLI entrypoint strategy instead of relying on script-by-script execution.
- Consider moving to a `src/` layout if packaging becomes more important.
- Add a short architecture document describing data flow from raw inputs to outputs.
- Add example commands for common workflows in the README.

## Priority 9: Longer-Term Research Work

- Add rolling or walk-forward model comparison across ridge, GLM, and LSTM variants.
- Evaluate whether the current regime feature set is sufficient or should include richer macro and volatility features.
- Compare hard regime assignment versus soft regime weighting more systematically.
- Add benchmark studies to determine whether regime-aware factor selection is adding value beyond a simpler baseline.
- Add experiment tracking for comparing configurations across runs.
