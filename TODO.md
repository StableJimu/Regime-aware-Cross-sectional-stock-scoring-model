# TODO

## Model Upgrades

- Replace pure regime-sliced fitting with a partially pooled model:
  - global baseline score
  - regime-specific deviations
  - shrink sparse regimes toward the global model
- Test wider ridge-first screening instead of relying mainly on stepwise-first factor selection.
- Add model variants that use regime probabilities as continuous features rather than only as partitions.
- Compare weekly regression targets against rank-oriented objectives.
- Test multi-horizon forecasting (`1d`, `5d`, `10d`) and combine signals only if the added stability is real.

## Factor Capacity

- Run controlled factor-capacity tests at `8`, `10`, and broader screened sets.
- Check whether the signal improves from more factors or from better shrinkage.
- Add factor-stability diagnostics across rolling refits and across regimes.

## Regime Modeling

- Rework thin-regime handling so sparse regimes are not effectively standalone low-sample models.
- Test whether the rare tail regime should be modeled as a stress overlay rather than a full separate score model.
- Compare frozen-train HMM behavior against rolling/refit HMM behavior in validation.

## Portfolio Construction

- Keep long-only `diag_mv` as the reference baseline.
- Keep weekly `130/30` as a parallel research track.
- Revisit `spo` only after the model layer is stronger.

## Performance Engineering

- Reduce pandas-heavy rolling work in selection and scoring.
- Add decision-date-only scoring with forward-fill between weekly rebalances.
- Reduce repeated factor-panel concatenation and masking in split walk-forward runs.

## Validation

- Add tests for weekly forward-target construction.
- Add tests for explicit alpha-source selection in the backtester.
- Add smoke tests for weekly split walk-forward runs on a reduced panel.

## Documentation

- Keep appending [RESEARCH_CYCLE_LOG.md](/C:/Users/jimya/Projects/Cross%20sectional%20stock%20selection%20Project/RESEARCH_CYCLE_LOG.md) after each major experiment cycle with timestamp, reason, and quantitative outcome.
