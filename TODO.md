# TODO

## Immediate

- Run the first full `k=4` macro-regime split backtest with the new defaults and inspect train/validation behavior end to end.
- Compare `k=4` macro-full against the previous `k=3` baseline in actual split backtests, not only regime-only sweeps.
- Check whether the new regime-family priors improve thin-regime factor stability or just add bias.

## Regime Research

- Validate that the ordered 4-state interpretation remains stable across different 5y/2y anchors.
- Improve regime evaluation beyond occupancy and smoothness with explicit duration and transition diagnostics.
- Test whether regime-specific factor caps should differ by regime instead of using one shared cap of `4`.
- Revisit whether the crisis bucket should be sharpened with additional macro stress features or stronger transition regularization.

## Modeling

- Run stepwise factor-capacity experiments at `4`, `8`, and `10` with train/validation comparison.
- Decide whether rolling ridge should remain the baseline or whether a broader global-selection variant is more stable under `k=4`.
- Add diagnostics for selected factors by regime over time so regime interpretation is tied to actual model content.

## Portfolio Construction

- Revisit `spo` inside its isolated module with small controlled experiments only.
- Compare `spo` against `diag_mv`, capped proportional weights, and simpler long-only allocators before promoting it again.
- Improve optimizer robustness and solver behavior before spending more time on alpha calibration for `spo`.

## Performance Engineering

- Profile the split walk-forward workflow and remove repeated pandas-heavy work in selection, calibration, and rolling scoring.
- Reduce repeated factor-panel concatenation and repeated date masking in hot paths.
- Move more regime and factor operations toward denser array-based workflows where practical.

## Validation And Testing

- Add tests for regime feature preparation, factor-family map expansion, and thin-regime selection behavior.
- Add smoke tests for the split walk-forward workflow with a reduced synthetic panel.
- Add artifact-level validation so runs fail early when regime files, factor batches, or required macro inputs are inconsistent.
