# Research Cycle Log

## 2026-03-11 16:10 America/Chicago

Cycle:
- Weekly decision-cycle migration and alpha-decay instrumentation

Changes:
- Fixed forward-label construction so `label_horizon > 1` now means cumulative forward return, not a farther single-day shift.
- Unified target construction across:
  - scoring labels
  - bucket calibration
  - IC / factor IC reporting
- Added `run_alpha_decay_analysis.py` to measure score or factor IC across multiple horizons.
- Switched default configs to weekly research settings:
  - `label_horizon = 5`
  - `holding_period = 5`
  - `rebalance_freq = 5`

Reason:
- The daily setup showed weak and unstable aggregate IC while portfolio results were partly being rescued by allocator behavior.
- Weekly targets should better match slower cross-sectional signal decay, reduce turnover, and make regime-conditioned modeling easier to interpret.
- Research speed should improve on the backtest side, and the new decay diagnostic lets us verify whether the signal actually survives to a weekly horizon before trusting the change.

Expected qualitative effect:
- More stable model target
- Lower turnover and less micro-noise sensitivity
- Better alignment between training objective and execution horizon
- Cleaner comparison of daily-vs-weekly edge persistence

Expected quantitative effect:
- Lower average turnover than daily rebalance runs
- Potentially higher IC / ICIR at `5d` than at `1d` if the current signal truly has slower decay
- Faster end-to-end research iteration on backtest/portfolio stages, though score generation remains a significant runtime component

Open question:
- Weekly rebalance alone does not materially reduce rolling score-generation cost. If runtime is still too high, the next engineering step should be decision-date-only scoring with forward-fill between rebalance dates.

Observed result:
- The decay study supported the weekly move.
- On the prior daily run `726298a944`, aggregate score IC improved with longer horizon:
  - `score` mean IC: `-0.0017` at `1d`, `0.0000` at `5d`, `0.0026` at `10d`
  - `score_calibrated` mean IC: `-0.0015` at `1d`, `0.0049` at `5d`, `0.0122` at `10d`
- Weekly split walk-forward run `b08f5f3d9a` was then executed with:
  - `label_horizon = 5`
  - `rebalance_freq = 5`
  - `holding_period = 5`
- Weekly run outcome:
  - train CAGR `16.49%`, Sharpe `0.833`, max DD `-34.25%`
  - validation CAGR `12.73%`, Sharpe `0.902`, max DD `-13.26%`
  - average rebalance turnover fell to about `24.4%` in train and `16.6%` in validation
- Weekly 5-day IC improved materially in-sample:
  - train `score` mean IC `0.0083`, ICIR `0.1191`
  - train `score_calibrated` mean IC `0.0094`, ICIR `0.1402`
- Validation remains mixed:
  - validation `score` mean IC `0.0006`, ICIR `0.0074`
  - validation `score_calibrated` mean IC `-0.0078`, ICIR `-0.1025`
- Qualitative conclusion:
  - the slower horizon is more aligned with the learned signal than the old daily target
  - the portfolio outcome remains acceptable under weekly execution
  - the calibration layer is still the weak point out of sample, so future optimizer work should not assume the calibrated cardinal alpha is stable in validation

## 2026-03-11 17:12 America/Chicago

Cycle:
- Remove score calibration from weekly portfolio construction and compare long-only vs `130/30`

Changes:
- Added explicit optimizer alpha-source control so allocator tests can use `score`, `score_raw`, or `score_calibrated` intentionally instead of always preferring calibrated output.
- Added explicit `long_budget` / `short_budget` controls so top-quantile long/short tests can represent `130/30` rather than only market-neutral `100/100`.
- Created two weekly research configs:
  - `config_split_walk_forward_weekly_diagmv_raw.yaml`
  - `config_split_walk_forward_weekly_topq_130_30.yaml`
- Disabled score calibration in both test configs.

Reason:
- The prior weekly run showed the same problem as the daily setup: the calibrated cardinal score remained the weakest out-of-sample object.
- Before trying another nonlinear mapping, the cleaner test is whether the raw regime-combined score is already sufficient for weekly execution.

Observed result:
- Baseline weekly calibrated long-only run `b08f5f3d9a`:
  - train CAGR `16.49%`, Sharpe `0.833`
  - validation CAGR `12.73%`, Sharpe `0.902`
  - validation `score_calibrated` mean IC `-0.0078`
- Weekly long-only `diag_mv` without calibration, run `30e71d89d7`:
  - train CAGR `14.12%`, Sharpe `0.756`
  - validation CAGR `19.58%`, Sharpe `1.302`
  - validation `score` mean IC `0.0006`
  - plot: `data/backtest_split_walk_forward_weekly_diagmv_raw/30e71d89d7/figures/performance_cost0_fee0_scaled.png`
- Weekly `top_q` `130/30` without calibration, run `81120ef095`:
  - train CAGR `34.22%`, Sharpe `1.335`
  - validation CAGR `20.53%`, Sharpe `1.015`
  - uses the same raw weekly score signal, so IC is unchanged and the gain is purely from portfolio construction / leverage
  - plot: `data/backtest_split_walk_forward_weekly_topq_130_30/81120ef095/figures/performance_cost0_fee0_scaled.png`

Qualitative conclusion:
- Bucketing is not helping the weekly system. Removing calibration improved the long-only validation result materially.
- The raw weekly score is weak but usable; the calibrated score was actively hurting allocator input quality.
- `130/30` looks promising as a construction overlay, but it is not directly comparable to long-only because it adds gross exposure.
- The next calibration step should not be another bucket variant. If calibration is revisited at all, it should be a monotone shrinkage-style map or be skipped entirely for portfolio construction.
