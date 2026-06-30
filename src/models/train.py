"""
Model construction and training.

The production model is a single scikit-learn ``Pipeline`` that bundles ALL
preprocessing (median imputation, log1p of heavy-tailed columns, standardisation)
with the Logistic Regression classifier. Because the preprocessing is *inside*
the pipeline, serving simply loads the one pickled object — there is no separate
preprocessing artefact to keep in sync, which removes a whole class of
train/serve skew bugs.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config
from src.utils.logger import get_logger

logger = get_logger("models.train")


def build_model(feature_columns: list[str]) -> Pipeline:
    """
    Construct the unfitted LogReg pipeline for the given feature schema.

    Heavy-tailed monetary/count columns (``config.LOG_COLS``) get a log1p
    transform before scaling; everything else is just imputed + scaled.
    """
    log_cols = [c for c in config.LOG_COLS if c in feature_columns]
    other_cols = [c for c in feature_columns if c not in log_cols]

    pre = ColumnTransformer([
        ("log", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("sc", StandardScaler()),
        ]), log_cols),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), other_cols),
    ])

    clf = LogisticRegression(random_state=config.RANDOM_STATE, **config.LOGREG_PARAMS)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_model(model: Pipeline, X, y) -> Pipeline:
    """Fit the pipeline. Thin wrapper so callers don't import sklearn directly."""
    logger.info("Fitting model on %d samples, %d features", X.shape[0], X.shape[1])
    model.fit(X, y)
    return model
