#!/usr/bin/env python3
"""
Batch scoring job — the primary On-Prem deliverable.

Reads a CSV of players, scores them with the current production model, writes a
CSV with churn_probability / risk_decile / risk_tier, and emits a drift report.
Designed to be run on a schedule (cron / Airflow) against the nightly export.

Usage:
    python -m scripts.batch_score --input data/raw/churn_dataset.csv \
                                  --output data/processed/scored_players.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config                                       # noqa: E402
from src.data.load_data import load_data                     # noqa: E402
from src.data.preprocess import clean_data                   # noqa: E402
from src.features.build_features import build_features, get_feature_matrix  # noqa: E402
from src.monitoring.drift import data_drift_report           # noqa: E402
from src.serving.inference import ChurnModel                 # noqa: E402
from src.utils.logger import get_logger                      # noqa: E402
from src.utils.validate_data import validate_churn_data      # noqa: E402

logger = get_logger("batch_score")


def main(args: argparse.Namespace) -> None:
    df = load_data(args.input)
    ok, failed = validate_churn_data(df, require_target=False)
    if not ok:
        raise ValueError(f"Input failed validation: {failed}")

    model = ChurnModel.load(args.model)
    scored = model.score(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    logger.info("Scored %d players -> %s", len(scored), out_path)
    logger.info("Risk tiers: %s", scored["risk_tier"].value_counts().to_dict())

    # Drift check on the scored batch (writes a JSON report alongside output).
    feats = get_feature_matrix(build_features(clean_data(df)), model.feature_columns)
    drift = data_drift_report(feats)
    report = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_players": int(len(scored)),
        "model_version": model.metadata.get("version", "unknown"),
        "risk_tiers": scored["risk_tier"].value_counts().to_dict(),
        "drift": drift,
    }
    report_path = out_path.with_name(out_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote run report -> %s (drift status: %s)", report_path, drift.get("status"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch-score players for churn risk.")
    p.add_argument("--input", default=config.RAW_DATA_PATH)
    p.add_argument("--output", default=str(config.PROCESSED_DIR / "scored_players.csv"))
    p.add_argument("--model", default=str(config.MODEL_PATH))
    main(p.parse_args())
