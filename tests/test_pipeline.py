"""
Smoke + consistency tests for the churn pipeline.

Run:  pytest -q
The tests use the real dataset if present, else a small synthetic frame, so CI
can run without shipping data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data.preprocess import clean_data
from src.features.build_features import build_features, get_feature_matrix
from src.models.evaluate import compute_metrics
from src.models.train import build_model, fit_model


def _synthetic(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    start = pd.Timestamp(config.OBSERVATION_START_FALLBACK)
    df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(n)],
        "observation_start_date": start,
        "first_live_bet_date": start + pd.to_timedelta(rng.integers(0, 80, n), "D"),
        "last_live_bet_date": start + pd.to_timedelta(rng.integers(0, 90, n), "D"),
        "tenure_days": rng.integers(1, 900, n),
        "live_bets_count": rng.integers(1, 300, n),
        "avg_bet_amount": rng.uniform(1, 50, n),
        "total_turnover": rng.uniform(10, 5000, n),
        "total_payout": rng.uniform(10, 5000, n),
        "ggr": rng.uniform(-500, 500, n),
        "ggr_margin": rng.uniform(-1, 1, n),
        "deposit_count": rng.integers(0, 20, n).astype(float),
        "total_deposit_amount": rng.uniform(0, 2000, n),
        "deposit_to_turnover_ratio": rng.uniform(0, 1, n),
        "days_active_in_observation": rng.integers(1, 90, n),
        "days_since_last_bet": rng.integers(0, 60, n),
        "churn": rng.integers(0, 2, n),
    })
    return df


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    if os.path.exists(config.RAW_DATA_PATH):
        return pd.read_csv(config.RAW_DATA_PATH, parse_dates=[
            c for c in config.DATE_COLS])
    return _synthetic()


def test_feature_engineering_shape_and_no_nan(raw_df):
    X = get_feature_matrix(build_features(clean_data(raw_df)))
    assert X.shape[0] == len(raw_df)
    assert config.TARGET not in X.columns
    for leak in config.LEAKAGE_COLS:
        assert leak not in X.columns          # leakage never reaches the model
    assert X.isna().sum().sum() == 0          # imputation handled downstream/here


def test_train_and_metrics(raw_df):
    df = build_features(clean_data(raw_df))
    X = get_feature_matrix(df)
    y = df[config.TARGET].astype(int)
    model = fit_model(build_model(list(X.columns)), X, y)
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    m = compute_metrics(y, proba)
    assert set(m) == {"roc_auc", "pr_auc", "brier"}


def test_train_serve_feature_parity(raw_df):
    """Single-row serving features must equal the batch features for that row."""
    df = build_features(clean_data(raw_df))
    feature_columns = list(get_feature_matrix(df).columns)
    batch = get_feature_matrix(df, feature_columns)

    one_raw = raw_df.iloc[[0]]
    one = get_feature_matrix(build_features(clean_data(one_raw)), feature_columns)
    assert list(one.columns) == feature_columns
    np.testing.assert_allclose(
        one.iloc[0].to_numpy(dtype=float),
        batch.iloc[0].to_numpy(dtype=float),
        rtol=1e-9, atol=1e-9,
    )
