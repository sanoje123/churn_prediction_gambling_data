#!/usr/bin/env python3
"""
Training pipeline:  load -> validate -> clean -> features -> train -> evaluate
                    -> quality gate -> persist artefacts (+ optional MLflow).

Usage:
    python -m scripts.run_pipeline --input data/raw/churn_dataset.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

# Make `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config                                              # noqa: E402
from src.data.load_data import load_data                           # noqa: E402
from src.data.preprocess import clean_data                         # noqa: E402
from src.features.build_features import build_features, get_feature_matrix  # noqa: E402
from src.models.evaluate import compute_metrics, decile_lift, passes_quality_gate  # noqa: E402
from src.models.train import build_model, fit_model                # noqa: E402
from src.monitoring.drift import save_reference_stats              # noqa: E402
from src.utils.logger import get_logger                            # noqa: E402
from src.utils.validate_data import validate_churn_data            # noqa: E402

logger = get_logger("pipeline")


def _try_mlflow():
    """Return the mlflow module if installed and reachable, else None.

    MLflow is an optional dependency, so a missing install is downgraded to a
    warning. A *misconfigured* backend (unreachable server, unsupported store)
    is a real problem the user usually wants to see, so it is logged loudly with
    the traceback before we fall back to local artefacts only.
    """
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        logger.warning("MLflow not installed; logging to local artefacts only.")
        return None
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        return mlflow
    except Exception:  # noqa: BLE001
        logger.error(
            "MLflow is installed but tracking is misconfigured (uri=%s); "
            "logging to local artefacts only.",
            config.MLFLOW_TRACKING_URI,
            exc_info=True,
        )
        return None


def main(args: argparse.Namespace) -> None:
    mlflow = None if args.no_mlflow else _try_mlflow()

    # 1. Load + validate ---------------------------------------------------
    df = load_data(args.input)
    ok, failed = validate_churn_data(df, require_target=True)
    if not ok:
        raise ValueError(f"Data validation failed: {failed}")

    # 2. Clean + engineer features ----------------------------------------
    df = clean_data(df)
    df = build_features(df)
    X = get_feature_matrix(df)                  # derives feature columns
    y = df[config.TARGET].astype(int)
    feature_columns = list(X.columns)
    logger.info("Feature matrix: %d rows x %d features", *X.shape)

    # Persist the processed dataset for reproducibility/debugging.
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X.assign(**{config.TARGET: y}).to_csv(
        config.PROCESSED_DIR / "churn_processed.csv", index=False)

    # 3. Hold-out split + train -------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=config.RANDOM_STATE)

    model = fit_model(build_model(feature_columns), X_tr, y_tr)

    # 4. Evaluate on the untouched test set -------------------------------
    proba_te = model.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, proba_te)
    lift = decile_lift(y_te, proba_te)
    logger.info("Held-out: ROC-AUC %.4f | PR-AUC %.4f | Brier %.4f",
                metrics["roc_auc"], metrics["pr_auc"], metrics["brier"])
    logger.info("Top-decile churn rate %.1f%% (lift %.2fx); top-30%% recall %.0f%%",
                lift.iloc[0]["churn_rate"] * 100, lift.iloc[0]["lift"],
                lift.iloc[2]["cum_recall"] * 100)

    gate_ok, reasons = passes_quality_gate(metrics)
    if not gate_ok:
        raise RuntimeError(f"Model failed quality gate: {reasons}")

    # 5. Refit on ALL data for the deployed model -------------------------
    final_model = fit_model(build_model(feature_columns), X, y)

    # 6. Persist artefacts ------------------------------------------------
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "LogisticRegression",
        "params": config.LOGREG_PARAMS,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "test_metrics": metrics,
        "data_md5": md5(Path(args.input).read_bytes()).hexdigest(),
    }

    joblib.dump(final_model, config.MODEL_PATH)
    config.FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_columns, indent=2))
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    save_reference_stats(X)
    logger.info("Saved model v%s -> %s", version, config.MODEL_PATH)

    # 7. Optional MLflow tracking -----------------------------------------
    if mlflow:
        with mlflow.start_run(run_name=f"logreg_{version}"):
            mlflow.log_params(config.LOGREG_PARAMS)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(config.MODEL_PATH))
            mlflow.log_artifact(str(config.FEATURE_COLUMNS_PATH))
            mlflow.log_artifact(str(config.METADATA_PATH))
        logger.info("Logged run to MLflow at %s", config.MLFLOW_TRACKING_URI)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the churn model.")
    p.add_argument("--input", default=config.RAW_DATA_PATH, help="raw CSV path")
    p.add_argument("--test_size", type=float, default=config.TEST_SIZE)
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    main(p.parse_args())
