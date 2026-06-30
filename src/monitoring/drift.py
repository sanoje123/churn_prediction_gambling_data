"""
Monitoring: data drift (PSI) and performance tracking.

These are the signals that trigger retraining. ``reference_stats.json`` is
written at training time (the training feature distribution); in production a
scheduled job compares the latest scored batch against it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.models.evaluate import compute_metrics
from src.utils.logger import get_logger

logger = get_logger("monitoring.drift")

# PSI rule of thumb: <0.1 no shift, 0.1-0.25 moderate, >0.25 significant.
PSI_WARN = 0.10
PSI_ALERT = 0.25


def _psi(edges, expected, actual: np.ndarray) -> float:
    """
    Population Stability Index for one numeric feature.

    Uses the training-time equal-frequency bin ``edges`` and ``expected``
    proportions; compares them against the new batch's proportions in the same
    bins. Scoring the training data itself yields PSI ~ 0.
    """
    if expected is None or edges is None or len(edges) < 3:
        return 0.0
    actual = actual[~np.isnan(actual)]
    if len(actual) == 0:
        return 0.0
    e_edges = np.array(edges, dtype=float)
    e_edges[0], e_edges[-1] = -np.inf, np.inf
    a = np.histogram(actual, bins=e_edges)[0] / len(actual)
    e = np.array(expected, dtype=float)
    a, e = np.clip(a, 1e-6, None), np.clip(e, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference_stats(X: pd.DataFrame) -> dict:
    """
    Summarise the training feature distribution for later PSI checks.

    For each feature we store the unique decile edges and the training
    proportion falling in each bin (the PSI 'expected' distribution).
    """
    stats = {"n": int(len(X)), "features": {}}
    for c in X.columns:
        col = pd.to_numeric(X[c], errors="coerce").dropna().to_numpy()
        edges = np.unique(np.quantile(col, np.linspace(0, 1, 11))) if len(col) else np.array([])
        if len(edges) < 3:
            stats["features"][c] = {"edges": edges.tolist(), "expected": None}
            continue
        e_edges = edges.copy().astype(float)
        e_edges[0], e_edges[-1] = -np.inf, np.inf
        expected = np.histogram(col, bins=e_edges)[0] / len(col)
        stats["features"][c] = {"edges": edges.tolist(), "expected": expected.tolist()}
    return stats


def save_reference_stats(X: pd.DataFrame, path: Path | str | None = None) -> None:
    path = Path(path or config.REFERENCE_STATS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_reference_stats(X)))
    logger.info("Saved reference stats -> %s", path)


def data_drift_report(X_new: pd.DataFrame, path: Path | str | None = None) -> dict:
    """Compute PSI per feature between training reference and new data."""
    path = Path(path or config.REFERENCE_STATS_PATH)
    if not path.exists():
        logger.warning("No reference stats at %s; skipping drift.", path)
        return {"status": "no_reference", "features": {}}

    ref = json.loads(path.read_text())["features"]
    report: dict = {"features": {}, "max_psi": 0.0, "n_alert": 0}
    for c, meta in ref.items():
        if c not in X_new.columns:
            continue
        psi = _psi(meta.get("edges"), meta.get("expected"),
                   pd.to_numeric(X_new[c], errors="coerce").to_numpy())
        report["features"][c] = round(psi, 4)
        report["max_psi"] = max(report["max_psi"], psi)
        if psi >= PSI_ALERT:
            report["n_alert"] += 1

    report["status"] = "alert" if report["n_alert"] else (
        "warn" if report["max_psi"] >= PSI_WARN else "ok")
    logger.info("Drift status=%s max_psi=%.3f alerts=%d",
                report["status"], report["max_psi"], report["n_alert"])
    return report


def performance_report(y_true, proba) -> dict:
    """Metrics on a labelled batch once outcomes are observed (30 days later)."""
    m = compute_metrics(y_true, proba)
    m["degraded"] = (m["roc_auc"] < config.MIN_ACCEPTABLE_ROC_AUC
                     or m["pr_auc"] < config.MIN_ACCEPTABLE_PR_AUC)
    return m


def should_retrain(drift: dict, perf: dict | None = None) -> tuple[bool, list[str]]:
    """Decide whether to trigger a retraining run."""
    reasons = []
    if drift.get("status") == "alert":
        reasons.append(f"data drift alert (max PSI {drift.get('max_psi', 0):.2f})")
    if perf and perf.get("degraded"):
        reasons.append(
            f"performance below gate (roc_auc {perf['roc_auc']:.3f}, pr_auc {perf['pr_auc']:.3f})")
    return (len(reasons) > 0, reasons)
