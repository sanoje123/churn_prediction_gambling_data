"""
Feature engineering — the 8 RFM-motivated behavioural features.

This is the single most important module for train/serve consistency: it must
produce byte-identical features whether called from training, batch scoring, or
the real-time API. It is pure and stateless (no fitted state), so the same code
path guarantees no skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils.logger import get_logger

logger = get_logger("features")

EPS = 1.0  # smoothing added to denominators to avoid divide-by-zero


def _observation_start(df: pd.DataFrame) -> pd.Series:
    """Per-row observation start date (snapshot). Falls back to config default."""
    if "observation_start_date" in df.columns:
        s = pd.to_datetime(df["observation_start_date"], errors="coerce")
        return s.fillna(pd.Timestamp(config.OBSERVATION_START_FALLBACK))
    return pd.Series(
        pd.Timestamp(config.OBSERVATION_START_FALLBACK), index=df.index
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered behavioural features to a cleaned DataFrame.

    Expects the output of ``data.preprocess.clean_data`` (dates parsed, deposit
    fields filled, ``no_deposit_flag`` present). Returns the DataFrame with the
    engineered columns added; column selection happens in ``get_feature_matrix``.
    """
    df = df.copy()
    obs_start = _observation_start(df)

    df["bets_per_active_day"] = df["live_bets_count"] / (df["days_active_in_observation"] + EPS)

    if {"last_live_bet_date", "first_live_bet_date"} <= set(df.columns):
        df["active_span_days"] = (
            df["last_live_bet_date"] - df["first_live_bet_date"]
        ).dt.days.fillna(0.0)
        df["days_to_first_bet"] = (
            (df["first_live_bet_date"] - obs_start).dt.days.fillna(config.OBSERVATION_WINDOW_DAYS)
        )
    else:
        df["active_span_days"] = 0.0
        df["days_to_first_bet"] = float(config.OBSERVATION_WINDOW_DAYS)

    df["activity_consistency"] = df["days_active_in_observation"] / (df["active_span_days"] + EPS)
    df["lifetime_bet_rate"] = df["live_bets_count"] / (df["tenure_days"] + EPS)
    df["turnover_per_active_day"] = df["total_turnover"] / (df["days_active_in_observation"] + EPS)
    df["avg_deposit_amount"] = df["total_deposit_amount"] / (df["deposit_count"] + EPS)
    df["deposit_per_bet"] = df["total_deposit_amount"] / (df["live_bets_count"] + EPS)

    logger.info("Built features | %d engineered columns added", 8)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the modelling feature names: everything except DROP_COLS."""
    return [c for c in df.columns if c not in config.DROP_COLS]


def get_feature_matrix(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Select and order the modelling features.

    Args:
        df: Output of ``build_features``.
        feature_columns: If given (serving), reindex to this exact order and
            fill any missing column with 0. If None (training), derive from the
            DataFrame.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(df)
        return df[feature_columns].copy()
    # Serving: enforce exact training-time schema and order.
    return df.reindex(columns=feature_columns, fill_value=0).copy()


def make_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Convenience: clean + build + select in one call.

    Note: importing here avoids a circular import at module load time.
    """
    from src.data.preprocess import clean_data

    cleaned = clean_data(df)
    built = build_features(cleaned)
    return get_feature_matrix(built, feature_columns)
