# Cross-Sectional Stock Selection Research

This repository is a research pipeline for cross-sectional equity selection. It builds daily Alpha101-style factor panels, conditions the model on a macro regime process, trains rolling cross-sectional scoring models, and evaluates portfolio construction under split walk-forward backtests.

The current working setup is:
- weekly decision horizon
- `k=4` macro HMM regime model
- regime-aware rolling ridge scoring
- `diag_mv` as the practical default allocator
- no score calibration as the preferred weekly baseline for portfolio construction

## Goal

The project is trying to answer one question: can a regime-aware cross-sectional model produce a stable weekly stock-selection edge that survives realistic validation, rather than only looking good in-sample or being rescued by portfolio construction?

## Current Design

### Regime layer

The market regime model is a 4-state HMM fit on macro features:
- market return and momentum
- downside and absolute-return shock measures
- VIX level and VIX momentum
- short yield and term spread

The current semantic ordering is:
- calm growth
- constructive / rangebound
- stressed correction
- crisis / panic

These regimes are used inside modeling, not only for diagnostics. Regime probabilities affect factor selection and score generation.

### Model layer

The model trains rolling cross-sectional scores on a weekly forward target. Factor selection is regime-aware and can apply family-level priors in thin regimes. The current baseline is a rolling ridge model.

### Portfolio layer

The practical baseline is long-only `diag_mv`. Weekly `130/30` top-quantile portfolios are also supported and are useful as a research comparator. `spo` remains in the codebase, but it is isolated for separate research and is not the default path.

## What Works Now

- Daily raw data loading and market feature assembly
- Alpha101 factor batch computation
- Cross-sectional factor normalization
- Split walk-forward and full-history backtests
- Weekly target training and weekly rebalancing
- Regime-aware factor selection and rolling ridge scoring
- Long-only `diag_mv`, `top_q`, proportional, `spo`, and `robust_spo` constructors
- Performance plots, benchmark overlays, IC reporting, and alpha-decay analysis

## Main Entry Points

Compute factor batches:

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

Plot a completed run:

```bash
python -m quant_pipeline.scripts.plot_performance \
  --run-dir data/backtest_split_walk_forward/<run_id> \
  --split \
  --ic-horizon 5
```

Run alpha-decay analysis:

```bash
python -m quant_pipeline.scripts.run_alpha_decay_analysis \
  --run-dir data/backtest_split_walk_forward/<run_id> \
  --raw-panel data/raw/ohlcv_1d_panel.csv \
  --score-cols score \
  --horizons 1,2,3,5,10
```

## Key Configs

- Main weekly split baseline:
  - [config_split_walk_forward.yaml](/C:/Users/jimya/Projects/Cross%20sectional%20stock%20selection%20Project/config/config_split_walk_forward.yaml)
- Weekly long-only no-calibration research:
  - [config_split_walk_forward_weekly_diagmv_raw.yaml](/C:/Users/jimya/Projects/Cross%20sectional%20stock%20selection%20Project/config/config_split_walk_forward_weekly_diagmv_raw.yaml)
- Weekly `130/30` no-calibration research:
  - [config_split_walk_forward_weekly_topq_130_30.yaml](/C:/Users/jimya/Projects/Cross%20sectional%20stock%20selection%20Project/config/config_split_walk_forward_weekly_topq_130_30.yaml)

## Current Research Read

- Weekly targets are better aligned than daily targets.
- The regime model is useful, but thin-regime modeling remains the main constraint.
- Bucket calibration is currently not a good weekly allocator input.
- Portfolio construction is no longer the main bottleneck.
- The next major gains should come from improving model thickness and stability, not from adding more allocator complexity.
