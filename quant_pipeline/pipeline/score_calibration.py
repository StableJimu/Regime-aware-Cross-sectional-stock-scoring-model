from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .targets import build_forward_return_label


@dataclass(frozen=True)
class BucketCalibrationConfig:
    lookback_days: int = 252
    n_buckets: int = 10
    min_total_obs: int = 5000
    min_bucket_obs: int = 100


def calibrate_scores_bucketed(
    score_panel: pd.DataFrame,
    prices_panel: pd.DataFrame,
    label_horizon: int,
    signal_lag: int,
    logger,
    cfg: Optional[BucketCalibrationConfig] = None,
    train_mask: Optional[pd.Series] = None,
    max_train_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if "score" not in score_panel.columns or "ret_1d" not in prices_panel.columns:
        return score_panel

    cfg = cfg or BucketCalibrationConfig()
    y = build_forward_return_label(
        prices_panel=prices_panel,
        label_horizon=int(label_horizon),
        signal_lag=int(signal_lag),
    )

    base = score_panel[["score"]].copy()
    base["score_raw"] = base["score"]
    base["score_pct"] = base.groupby(level="date")["score_raw"].rank(method="average", pct=True)
    base = base.join(y, how="left")

    if train_mask is not None:
        if train_mask.index.equals(base.index):
            base["train_mask"] = train_mask.values
        else:
            base["train_mask"] = train_mask.reindex(base.index).fillna(False).values
    else:
        base["train_mask"] = True

    dates = base.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    out = score_panel.copy()
    out["score_raw"] = out["score"]
    out["score_pct"] = out.groupby(level="date")["score_raw"].rank(method="average", pct=True)
    out["score_calibrated"] = np.nan

    updates = 0
    for d in unique_dates:
        hist_dates = unique_dates[unique_dates < d]
        if max_train_date is not None:
            hist_dates = hist_dates[hist_dates <= pd.Timestamp(max_train_date)]
        if len(hist_dates) == 0:
            continue
        win_dates = hist_dates[-int(cfg.lookback_days):]
        hist_mask = dates.isin(win_dates) & base["train_mask"].to_numpy()
        hist = base.loc[hist_mask, ["score_pct", "label"]].dropna()
        if len(hist) < int(cfg.min_total_obs):
            continue

        scores_hist = hist["score_pct"].to_numpy(dtype=float)
        labels_hist = hist["label"].to_numpy(dtype=float)
        quantiles = np.linspace(0.0, 1.0, int(cfg.n_buckets) + 1)
        edges = np.unique(np.quantile(scores_hist, quantiles))
        if len(edges) < 3:
            continue

        inner_edges = edges[1:-1]
        bucket_hist = np.searchsorted(inner_edges, scores_hist, side="right")
        n_eff = len(edges) - 1
        bucket_means = np.full(n_eff, np.nan, dtype=float)
        bucket_counts = np.zeros(n_eff, dtype=int)
        for b in range(n_eff):
            mask_b = bucket_hist == b
            bucket_counts[b] = int(mask_b.sum())
            if bucket_counts[b] >= int(cfg.min_bucket_obs):
                bucket_means[b] = float(labels_hist[mask_b].mean())

        valid = np.isfinite(bucket_means)
        if valid.sum() < max(3, n_eff // 2):
            continue

        centers = 0.5 * (edges[:-1] + edges[1:])
        centers = centers[:n_eff]
        x = centers[valid]
        yb = bucket_means[valid]
        wb = bucket_counts[valid].astype(float)
        # Preserve the original score ordering. The optimizer needs a cardinal
        # alpha estimate, but it should not silently reverse a rank signal.
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(x, yb, sample_weight=wb)

        idx_day = dates == d
        scores_day = out.loc[idx_day, "score_pct"]
        if scores_day.empty:
            continue
        pred = iso.predict(scores_day.to_numpy(dtype=float))
        out.loc[scores_day.index, "score_calibrated"] = pred
        updates += 1

    logger.info(
        "score calibration (bucketed): "
        f"lookback_days={cfg.lookback_days}, n_buckets={cfg.n_buckets}, "
        f"updated_dates={updates}, filled_share={out['score_calibrated'].notna().mean():.4f}"
    )
    out["score"] = out["score_raw"]
    return out
