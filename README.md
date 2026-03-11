# Cross-Sectional Stock Selection Research

This repo is a research pipeline for daily cross-sectional equity selection. The current view of the project is simple: regime definition matters as much as factor selection, because the regime model decides which parts of the signal library should matter, how much data each model really sees, and how scores should be interpreted downstream.

## Current Research View

- The working regime baseline is now a `k=4` HMM, not `k=3`.
- The preferred regime feature block is macro-heavy rather than volatility-only:
  - market return and momentum
  - downside and absolute-return shock measures
  - VIX level and VIX momentum
  - short yield and term spread
- The current ordered regime interpretation is:
  - calm growth
  - constructive / rangebound
  - stressed correction
  - crisis / panic
- These regimes are not only labels for plots. They drive:
  - soft regime probabilities used in training
  - regime-specific stepwise IC factor selection
  - per-regime ridge models
  - the way model behavior is interpreted in train and validation

## What The Pipeline Does

- loads daily price panels, market proxy data, VIX, and yields
- computes Alpha101 factors in parquet batches
- cross-sectionally normalizes the factor panel
- fits regime-aware models with rolling retraining
- calibrates optimizer-facing scores with a trailing bucketed mapping
- backtests multiple portfolio constructors
- saves split and full-run artifacts, diagnostics, and plots

## Current Defaults

- Regime model:
  - `k=4`
  - macro feature block
  - `emit_temp=0.80`
  - `var_floor=0.60`
  - `hard_prob=0.97`
  - `init_diag=0.70`
- Modeling:
  - `ridge`
  - `selection_method=regime_stepwise`
  - `max_factors_per_regime=4`
  - `rolling_train_days=378`
  - `selection_window_days=378`
  - soft regime weighting enabled
- Thin-regime handling:
  - minimum effective sample size for soft selection is `50`
  - regime-family priors are blended into stepwise IC ranking
  - thin regimes are restricted to a smaller candidate set
- Portfolio construction:
  - `diag_mv` is the default allocator
  - `spo` has been isolated for separate research rather than serving as the default

## Why `k=4`

The earlier `k=3` setup was interpretable, but it compressed too much structure into a single stress bucket. With the broader macro feature set, `k=4` separates normal growth from constructive/rangebound conditions, and separates ordinary corrections from full crisis states. That matters because the factor model is conditional on these regimes. A better regime partition changes both what gets selected and how stable the rolling model fit is.

## Thin-Regime Policy

`k=4` improves interpretability, but it also makes the tail regimes thinner. The current answer is not to borrow neighboring regimes mechanically. Instead, the selector can apply family-level priors when effective sample size gets weak. This is meant to keep crisis-state selection economically sensible instead of purely noisy.

The initial factor-family taxonomy is intentionally coarse:

- `trend`
- `short_reversal`
- `volatility_stress`
- `liquidity_turnover`
- `quality_defensive`
- `beta_macro`
- `residual_cross_section`

These are used as selection priors, not as hard economic truth.

## Portfolio Construction Status

`diag_mv` is the practical baseline right now. In smaller controlled experiments it has been more stable than `spo`, and the recent work has shown that `spo` still needs separate formulation work. The optimizer code is now isolated so it can be revisited without disturbing the rest of the research workflow.

## Main Workflows

Compute factors:

```bash
python -m quant_pipeline.scripts.compute_factors_batch
```

Run the main split walk-forward backtest:

```bash
python -m quant_pipeline.scripts.run_backtest_split \
  --config config/config_split_walk_forward.yaml \
  --train-years 5 \
  --val-years 2 \
  --stress-level low
```

Sweep regime parameters only:

```bash
python -m quant_pipeline.scripts.run_regime_parameter_sweep --config config/config_split_walk_forward.yaml
```

Plot a completed run:

```bash
python -m quant_pipeline.scripts.plot_performance --run-dir data/backtest_split_walk_forward/<run_id> --split
```

## Current Conclusions

- Regime modeling is now a core modeling choice, not a reporting layer.
- `k=4` with macro features is the best current regime specification.
- Thin-regime sample efficiency is now the main constraint on regime-aware factor selection.
- `diag_mv` is the correct default until `spo` is reformulated and revalidated.
- The next research gains are more likely to come from regime tuning, factor-capacity tests, and hot-path performance work than from adding more portfolio complexity immediately.
