# Config Guide

The active config set is:

- `config.yaml`
  - canonical full-backtest config

- `config_split_walk_forward.yaml`
  - split backtest with walk-forward validation
  - validation can refit over time, but only on data available before each scoring date

- `config_split_frozen.yaml`
  - split backtest with frozen validation
  - validation uses train-fit models without post-split refits

All three configs share the same data, factor, regime, and model structure. The main differences are:

- `paths.backtest_dir`
- `model.validation_mode` on split runs
- the stress grid and cost settings used for the specific evaluation

Portfolio construction is controlled from the `backtest` section:

- `portfolio_method: "diag_mv"`
  - default long-only diagonal-risk allocator using calibrated alpha and per-name variance

- `portfolio_method: "top_q"`
  - rank-based portfolio construction using `score_raw`

- `portfolio_method: "spo"`
  - convex optimizer using calibrated expected returns and covariance risk

- `portfolio_method: "robust_spo"`
  - `spo` plus uncertainty-aware penalties

For normal usage:

- use `config.yaml` for full-history evaluation
- use `config_split_walk_forward.yaml` for the main split backtest
- use `config_split_frozen.yaml` only when you want a stricter fixed-model validation check
