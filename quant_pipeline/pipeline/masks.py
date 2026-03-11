# -*- coding: utf-8 -*-
"""
Mask helpers for model ensembling.
Currently only a single "all samples" mask is used.
"""
from __future__ import annotations

import pandas as pd


def all_samples_mask(factor_panel: pd.DataFrame) -> pd.Series:
    """Return a boolean mask selecting all rows."""
    return pd.Series(True, index=factor_panel.index, name="mask_all")
