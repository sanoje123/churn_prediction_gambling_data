"""
Central configuration for the live-betting churn pipeline.

Everything that the training, serving, and monitoring code needs to agree on
lives here, so there is a single source of truth and no risk of train/serve
skew creeping in through duplicated constants.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # .../churn_production
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# A specific model version is written into a timestamped folder by retrain.py;
# the "current" symlink/folder is what serving loads.
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
REFERENCE_STATS_PATH = ARTIFACTS_DIR / "reference_stats.json"   # for drift monitoring

# Default raw dataset (overridable via env var, like the notebook).
RAW_DATA_PATH = os.environ.get("CHURN_DATA", str(RAW_DIR / "churn_dataset.csv"))

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
TARGET = "churn"

DATE_COLS = [
    "registration_date", "observation_start_date", "observation_end_date",
    "last_live_bet_date", "first_live_bet_date",
]

# Leaks the label by construction (bets placed AFTER the observation window).
LEAKAGE_COLS = ["live_bets_count_outcome_window"]

# Identifier / raw columns that must never enter the model.
ID_COLS = ["user_id"]

# Columns dropped before modelling: leakage + ids + raw dates + low-value raw
# rate column + the target itself. Matches the analysis notebook exactly.
DROP_COLS = LEAKAGE_COLS + ID_COLS + [
    "observation_start_date", "observation_end_date",
    "registration_date", "last_live_bet_date", "first_live_bet_date",
    "bet_day_rate", TARGET,
]

# Heavy-tailed monetary / count columns that get a log1p transform inside the
# model pipeline before scaling.
LOG_COLS = [
    "total_turnover", "total_payout", "total_deposit_amount",
    "turnover_per_active_day", "avg_deposit_amount", "deposit_per_bet",
    "live_bets_count",
]

# Observation window length (days). Used as the fill value for days_to_first_bet
# when a player never placed a first live bet inside the window.
OBSERVATION_WINDOW_DAYS = 90
# Fallback observation-start date if the raw column is absent (snapshot date).
OBSERVATION_START_FALLBACK = "2025-10-03"

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Production model: Logistic Regression with log1p + standardisation pipeline.
# These are the values validated in the analysis (C=1, L2 was already optimal).
# `l1_ratio=0` is pure L2; it replaces the deprecated `penalty="l2"`, which is
# removed in scikit-learn 1.10 (requires scikit-learn >= 1.8, solver "saga").
LOGREG_PARAMS = dict(
    C=1.0,
    l1_ratio=0,
    solver="saga",
    max_iter=3000,
)

# Minimum held-out PR-AUC a freshly trained model must reach to be deployable.
MIN_ACCEPTABLE_PR_AUC = 0.50
MIN_ACCEPTABLE_ROC_AUC = 0.82

# --------------------------------------------------------------------------- #
# Scoring / business thresholds
# --------------------------------------------------------------------------- #
# Risk tiers are assigned from the predicted probability. Cut-offs chosen from
# the decile-lift analysis (top decile ~60% churn => High).
RISK_TIER_BINS = [
    (0.50, "High"),     # p >= 0.50
    (0.25, "Medium"),   # 0.25 <= p < 0.50
    (0.00, "Low"),      # p <  0.25
]

# Default operating threshold for a binary "contact / don't" decision.
DECISION_THRESHOLD = 0.50

# --------------------------------------------------------------------------- #
# MLflow (optional)
# --------------------------------------------------------------------------- #
# MLflow 3.x puts the filesystem tracking backend (file://.../mlruns) into
# maintenance mode and raises unless this opt-in flag is set. We default to the
# local file store, so enable it here (without clobbering a user override).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri()
)
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "live-betting-churn")
