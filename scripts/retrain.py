#!/usr/bin/env python3
"""
Scheduled retraining with a champion / challenger gate.

Trains a fresh "challenger" on the latest data and compares it to the current
"champion" on a common hold-out. The challenger is promoted ONLY if it clears
the absolute quality gate AND does not regress versus the champion beyond a
small tolerance. Old models are archived so a rollback is always one copy away.

Usage:
    python -m scripts.retrain --input data/raw/churn_dataset.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config                                       # noqa: E402
from src.data.load_data import load_data                     # noqa: E402
from src.data.preprocess import clean_data                   # noqa: E402
from src.features.build_features import build_features, get_feature_matrix  # noqa: E402
from src.models.evaluate import compute_metrics, passes_quality_gate  # noqa: E402
from src.models.train import build_model, fit_model          # noqa: E402
from src.utils.logger import get_logger                      # noqa: E402
from src.utils.validate_data import validate_churn_data      # noqa: E402

logger = get_logger("retrain")

# Allow the challenger to be at most this much worse on PR-AUC and still pass
# (guards against promoting a model that regressed on noise).
REGRESSION_TOLERANCE = 0.005


def _archive_champion() -> None:
    if config.MODEL_PATH.exists():
        archive = config.ARTIFACTS_DIR / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        meta = json.loads(config.METADATA_PATH.read_text()) if config.METADATA_PATH.exists() else {}
        tag = meta.get("version", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        shutil.copy2(config.MODEL_PATH, archive / f"model_{tag}.pkl")
        if config.METADATA_PATH.exists():
            shutil.copy2(config.METADATA_PATH, archive / f"metadata_{tag}.json")
        logger.info("Archived champion %s", tag)


def main(args: argparse.Namespace) -> None:
    # Prepare data + a common hold-out for a fair champion/challenger compare.
    df = load_data(args.input)
    ok, failed = validate_churn_data(df, require_target=True)
    if not ok:
        raise ValueError(f"Data validation failed: {failed}")

    df = build_features(clean_data(df))
    X = get_feature_matrix(df)
    y = df[config.TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    # Challenger.
    challenger = fit_model(build_model(list(X.columns)), X_tr, y_tr)
    chal_metrics = compute_metrics(y_te, challenger.predict_proba(X_te)[:, 1])
    logger.info("Challenger: %s", {k: round(v, 4) for k, v in chal_metrics.items()})

    gate_ok, reasons = passes_quality_gate(chal_metrics)
    if not gate_ok:
        logger.error("Challenger fails absolute quality gate: %s. Keeping champion.", reasons)
        return

    # Champion (if any) on the same hold-out.
    champ_pr = None
    if config.MODEL_PATH.exists():
        champion = joblib.load(config.MODEL_PATH)
        champ_metrics = compute_metrics(y_te, champion.predict_proba(X_te)[:, 1])
        champ_pr = champ_metrics["pr_auc"]
        logger.info("Champion:   %s", {k: round(v, 4) for k, v in champ_metrics.items()})

    if champ_pr is not None and chal_metrics["pr_auc"] < champ_pr - REGRESSION_TOLERANCE:
        logger.warning(
            "Challenger PR-AUC %.4f regresses vs champion %.4f (tol %.3f). Keeping champion.",
            chal_metrics["pr_auc"], champ_pr, REGRESSION_TOLERANCE)
        return

    # Promote: archive old, then re-fit challenger on ALL data and save.
    _archive_champion()
    final_model = fit_model(build_model(list(X.columns)), X, y)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    joblib.dump(final_model, config.MODEL_PATH)
    config.FEATURE_COLUMNS_PATH.write_text(json.dumps(list(X.columns), indent=2))
    config.METADATA_PATH.write_text(json.dumps({
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "LogisticRegression",
        "params": config.LOGREG_PARAMS,
        "n_samples": int(len(X)),
        "test_metrics": chal_metrics,
        "promoted_over_champion_pr_auc": champ_pr,
    }, indent=2))
    logger.info("PROMOTED challenger as new champion v%s.", version)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Retrain + champion/challenger gate.")
    p.add_argument("--input", default=config.RAW_DATA_PATH)
    main(p.parse_args())
