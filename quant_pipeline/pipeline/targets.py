from __future__ import annotations

import pandas as pd


def build_forward_return_label(
    prices_panel: pd.DataFrame,
    label_horizon: int,
    signal_lag: int,
    return_col: str = "ret_1d",
) -> pd.Series:
    """Build forward cumulative log-return labels with explicit signal lag.

    For horizon h and lag l, the label at date t is:
      ret_{t+l} + ret_{t+l+1} + ... + ret_{t+l+h-1}
    where ret_* are daily log returns.
    """
    if return_col not in prices_panel.columns:
        raise ValueError(f"prices_panel missing {return_col}")

    h = max(1, int(label_horizon))
    lag = max(0, int(signal_lag))
    grouped = prices_panel.groupby(level="ticker")[return_col]

    y = None
    for step in range(lag, lag + h):
        part = grouped.shift(-step)
        y = part if y is None else (y + part)
    y.name = "label"
    return y
