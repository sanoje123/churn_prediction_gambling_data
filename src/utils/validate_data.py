"""
Data-quality validation for the live-betting churn dataset.

Runs before training (and before batch scoring) so bad inputs fail fast and
loudly instead of silently degrading the model. Returns a (is_valid, failures)
tuple; failures are logged and can be persisted by the caller.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from src import config
from src.utils.logger import get_logger

logger = get_logger("data.validate")

# Raw columns the feature pipeline depends on.
REQUIRED_RAW_COLS = [
    "tenure_days", "live_bets_count", "total_turnover", "total_payout",
    "ggr", "days_active_in_observation", "days_since_last_bet",
]


def validate_churn_data(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate schema, ranges and business logic of a churn DataFrame.

    Args:
        df: Raw (pre-feature-engineering) data.
        require_target: Whether the ``churn`` target must be present (training).

    Returns:
        (is_valid, failed_checks)
    """
    failed: List[str] = []

    def check(name: str, condition: bool) -> None:
        if not bool(condition):
            failed.append(name)

    # --- Schema -----------------------------------------------------------
    for col in REQUIRED_RAW_COLS:
        check(f"column_present:{col}", col in df.columns)
    if require_target:
        check(f"column_present:{config.TARGET}", config.TARGET in df.columns)

    existing = set(df.columns)

    # --- Non-emptiness -----------------------------------------------------
    check("not_empty", len(df) > 0)

    # --- Numeric ranges / business logic ----------------------------------
    if "live_bets_count" in existing:
        check("live_bets_count_non_negative", (df["live_bets_count"].fillna(0) >= 0).all())
    if "tenure_days" in existing:
        check("tenure_days_non_negative", (df["tenure_days"].fillna(0) >= 0).all())
    if "days_active_in_observation" in existing:
        check("days_active_in_window",
              df["days_active_in_observation"].fillna(0).between(0, config.OBSERVATION_WINDOW_DAYS + 1).all())
    if "days_since_last_bet" in existing:
        check("days_since_last_bet_non_negative", (df["days_since_last_bet"].fillna(0) >= 0).all())
    if "total_turnover" in existing:
        check("total_turnover_non_negative", (df["total_turnover"].fillna(0) >= 0).all())

    if require_target and config.TARGET in existing:
        check("target_is_binary", set(df[config.TARGET].dropna().unique()) <= {0, 1})

    # --- Leakage guard: warn (not fail) if the leak column is still present.
    leak_present = [c for c in config.LEAKAGE_COLS if c in existing]
    if leak_present:
        logger.warning("Leakage columns present and will be dropped: %s", leak_present)

    is_valid = len(failed) == 0
    if is_valid:
        logger.info("Data validation PASSED (%d rows).", len(df))
    else:
        logger.error("Data validation FAILED: %s", failed)
    return is_valid, failed
