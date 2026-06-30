"""
Deterministic cleaning step.

Everything here is pure / stateless (no fitted parameters), so it can be applied
identically at training time and at serving time. Fitted transforms (imputation
medians, scaling) live inside the model Pipeline instead — see models/train.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils.logger import get_logger

logger = get_logger("data.preprocess")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the deterministic cleaning decisions established during analysis:

    * Reconstruct ``avg_bet_amount`` exactly as ``total_turnover / live_bets_count``
      where it is missing (no information lost), falling back to 0.
    * Add ``no_deposit_flag`` BEFORE filling deposit fields — missingness is
      informative (no-deposit players churn slightly less).
    * Fill deposit fields and ``ggr_margin`` with 0.

    Date columns are parsed (left as datetime) for the feature step. Leakage
    columns are intentionally NOT dropped here — that happens at feature
    selection so validation can still inspect them.
    """
    df = df.copy()

    # Ensure date columns are datetime (load_data already parses, but batch
    # inputs may arrive as strings via the API).
    for c in config.DATE_COLS:
        if c in df.columns and not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # avg_bet_amount: exact reconstruction, then 0.
    if "avg_bet_amount" in df.columns:
        recomputed = df["total_turnover"] / df["live_bets_count"].replace(0, np.nan)
        df["avg_bet_amount"] = df["avg_bet_amount"].fillna(recomputed).fillna(0.0)

    # Informative missingness flag for deposits (must precede the fill).
    if "deposit_count" in df.columns:
        df["no_deposit_flag"] = df["deposit_count"].isna().astype(int)
    else:
        df["no_deposit_flag"] = 0

    for c in ["deposit_count", "total_deposit_amount", "deposit_to_turnover_ratio"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    if "ggr_margin" in df.columns:
        df["ggr_margin"] = df["ggr_margin"].fillna(0.0)

    logger.info("Cleaned data | %d rows x %d cols", df.shape[0], df.shape[1])
    return df
