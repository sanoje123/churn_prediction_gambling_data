"""Evaluation metrics and business-facing decile-lift table."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

from src import config


def compute_metrics(y_true, proba) -> dict:
    """Return the three headline metrics used throughout the project."""
    proba = np.clip(np.asarray(proba, dtype=float), 0, 1)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def decile_lift(y_true, proba, k: int = 10) -> pd.DataFrame:
    """
    Rank players by predicted probability and bucket into ``k`` equal groups.

    Returns a table with churn rate, lift over base rate, and cumulative recall —
    the artefact the retention team uses to decide how deep to contact.
    """
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    d = d.sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = (np.floor(np.arange(len(d)) / (len(d) / k)).astype(int) + 1).clip(1, k)
    base = d["y"].mean()
    g = d.groupby("decile").agg(n=("y", "size"), churners=("y", "sum"))
    g["churn_rate"] = g["churners"] / g["n"]
    g["lift"] = g["churn_rate"] / base
    g["cum_recall"] = g["churners"].cumsum() / d["y"].sum()
    return g.reset_index()


def passes_quality_gate(metrics: dict) -> tuple[bool, list[str]]:
    """Check a model's metrics against the minimum deployable thresholds."""
    reasons = []
    if metrics["roc_auc"] < config.MIN_ACCEPTABLE_ROC_AUC:
        reasons.append(
            f"roc_auc {metrics['roc_auc']:.4f} < {config.MIN_ACCEPTABLE_ROC_AUC}")
    if metrics["pr_auc"] < config.MIN_ACCEPTABLE_PR_AUC:
        reasons.append(
            f"pr_auc {metrics['pr_auc']:.4f} < {config.MIN_ACCEPTABLE_PR_AUC}")
    return (len(reasons) == 0, reasons)
