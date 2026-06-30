"""
Inference / serving.

Loads the pickled LogReg pipeline + the training-time feature schema once, then
exposes ``score`` for both batch jobs and the real-time API. Raw player records
go through the SAME clean -> build_features -> select path as training, so there
is no train/serve skew.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features.build_features import make_features
from src.utils.logger import get_logger

logger = get_logger("serving.inference")


class ChurnModel:
    """Wraps the fitted pipeline and feature schema for scoring."""

    def __init__(self, model, feature_columns: list[str], metadata: dict | None = None):
        self.model = model
        self.feature_columns = feature_columns
        self.metadata = metadata or {}

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def load(cls, model_path: Path | str | None = None) -> "ChurnModel":
        model_path = Path(model_path or config.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artefact not found at {model_path}. "
                "Train one first: python -m scripts.run_pipeline"
            )
        model = joblib.load(model_path)

        feats_path = Path(config.FEATURE_COLUMNS_PATH)
        feature_columns = json.loads(feats_path.read_text()) if feats_path.exists() else None
        if feature_columns is None:
            raise FileNotFoundError(f"Feature schema not found at {feats_path}")

        meta_path = Path(config.METADATA_PATH)
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        logger.info("Loaded model v%s with %d features",
                    metadata.get("version", "?"), len(feature_columns))
        return cls(model, feature_columns, metadata)

    # -- scoring ---------------------------------------------------------- #
    def predict_proba(self, records: pd.DataFrame | list[dict] | dict) -> np.ndarray:
        """Return churn probabilities for one or many raw player records."""
        df = self._to_frame(records)
        X = make_features(df, feature_columns=self.feature_columns)
        return self.model.predict_proba(X)[:, 1]

    def score(self, records: pd.DataFrame | list[dict] | dict) -> pd.DataFrame:
        """
        Score records and attach business fields.

        Returns a DataFrame with churn_probability, risk_decile, risk_tier and
        the original user_id (if provided).
        """
        df = self._to_frame(records)
        proba = self.predict_proba(df)

        out = pd.DataFrame(index=df.index)
        if "user_id" in df.columns:
            out["user_id"] = df["user_id"].values
        out["churn_probability"] = proba
        out["risk_decile"] = self._risk_decile(proba)
        out["risk_tier"] = [self._risk_tier(p) for p in proba]
        return out.reset_index(drop=True)

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _to_frame(records) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records.copy()
        if isinstance(records, dict):
            return pd.DataFrame([records])
        return pd.DataFrame(list(records))

    @staticmethod
    def _risk_tier(p: float) -> str:
        for cutoff, label in config.RISK_TIER_BINS:
            if p >= cutoff:
                return label
        return config.RISK_TIER_BINS[-1][1]

    @staticmethod
    def _risk_decile(proba: np.ndarray) -> np.ndarray:
        """Decile 1 = riskiest. Ranked within the scored batch."""
        if len(proba) == 0:
            return np.array([], dtype=int)
        order = (-proba).argsort().argsort()                # rank, 0 = highest p
        return (np.floor(order / (len(proba) / 10)).astype(int) + 1).clip(1, 10)


@lru_cache(maxsize=1)
def get_model() -> ChurnModel:
    """Process-wide singleton so the API loads the model only once."""
    return ChurnModel.load()


def predict(record: dict) -> dict:
    """Convenience single-record prediction for the API."""
    row = get_model().score(record).iloc[0].to_dict()
    row["churn_probability"] = float(row["churn_probability"])
    return row
