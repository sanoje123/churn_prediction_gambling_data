# Live-Betting Player Churn — Production Pipeline

Production-ready packaging of the churn model developed in
`..notebook/churn_final_presentation.ipynb`. The model flags players likely to place
**zero live bets in the next 30 days**, ranked so the retention team contacts the
riskiest first.

The production model is a **scikit-learn `Pipeline`** = `log1p` of heavy-tailed
columns → standardisation → **Logistic Regression**. Because *all* preprocessing
lives inside the pipeline object, serving loads a single `.pkl` and there is no
separate preprocessing artefact to keep in sync.

> **Validated parity:** the production training run reproduces the notebook
> exactly — held-out **ROC-AUC 0.854, PR-AUC 0.537, Brier 0.106**, top-decile
> churn 60% (3.5× lift), top-30% recall 74%.

---

## Project structure

```
churn_production/
├── src/
│   ├── config.py              # single source of truth: paths, schema, params, thresholds
│   ├── data/
│   │   ├── load_data.py       # CSV load + date parsing
│   │   └── preprocess.py      # deterministic cleaning (avg_bet reconstruct, no_deposit_flag, NA fills)
│   ├── features/
│   │   └── build_features.py  # the 8 RFM features — shared by train AND serve (no skew)
│   ├── models/
│   │   ├── train.py           # builds the LogReg sklearn Pipeline + fit
│   │   └── evaluate.py        # ROC-AUC / PR-AUC / Brier + decile-lift + quality gate
│   ├── serving/
│   │   ├── inference.py       # ChurnModel: load pickle, score raw records -> risk tiers
│   │   └── api.py             # FastAPI: /health, /predict, /predict_batch
│   ├── monitoring/
│   │   └── drift.py           # PSI data-drift + performance tracking -> retrain trigger
│   └── utils/                 # logger, data validation
├── scripts/
│   ├── run_pipeline.py        # TRAIN: load->validate->clean->features->train->gate->persist
│   ├── batch_score.py         # SCORE: nightly cron job -> scored CSV + drift report
│   └── retrain.py             # RETRAIN: champion/challenger promotion gate + archive
├── notebook/                  # analysis notebook with data exploration, feature engineering, and model evaluation
│   └── churn_final_presentation.ipynb
├── tests/test_pipeline.py     # smoke + train/serve feature-parity tests
├── artifacts/                 # model.pkl, feature_columns.json, model_metadata.json, reference_stats.json
├── Dockerfile · .github/workflows/ci.yml · requirements.txt
└── ARCHITECTURE.md            # On-Prem deployment design + retraining strategy
```

---

## Jupyter notebook

The analysis notebook is `notebook/churn_final_presentation.ipynb`. It documents the full workflow for this project, including:
- raw data loading and leakage detection
- cleaning, missing-value handling, and feature engineering
- model training, cross-validation, and comparison of LogReg/LightGBM/XGBoost
- probability calibration, decile lift, and business thresholding
- final held-out test evaluation and scored player export

To run it locally:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r notebook/requirements_jupyter.txt
jupyter lab
```

The notebook uses the same `CHURN_DATA` environment override as the production code. By default it loads `data/raw/churn_dataset.csv`.

The notebook is the source of truth for the analysis results shown in this README.

---

## Quickstart

```bash
# 0. install
pip install -r requirements.txt

# 1. train (writes artifacts/model.pkl + metadata + reference stats)
python scripts/run_pipeline.py --input data/raw/churn_dataset.csv

# 2. batch-score a player export (the primary use case)
python scripts/batch_score.py --input data/raw/churn_dataset.csv \
                              --output data/processed/scored_players.csv

# 3. serve the real-time API
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
#   GET  /health
#   POST /predict        {single player record}
#   POST /predict_batch  {"players": [ ... ]}

# 4. scheduled retraining with champion/challenger gate
python scripts/retrain.py --input data/raw/churn_dataset.csv

# tests
pytest -q
```

`CHURN_DATA` env var overrides the default input path; MLflow logging activates
automatically if `mlflow` is installed (skip with `--no-mlflow`).

---

## Outputs

| Artefact | Written by | Purpose |
|----------|-----------|---------|
| `artifacts/model.pkl` | `run_pipeline` / `retrain` | the deployed LogReg pipeline |
| `artifacts/feature_columns.json` | training | exact feature schema + order for serving |
| `artifacts/model_metadata.json` | training | version, metrics, params, data hash |
| `artifacts/reference_stats.json` | training | training feature distribution for PSI drift |
| `data/processed/scored_players.csv` | `batch_score` | `user_id, churn_probability, risk_decile, risk_tier` |
| `data/processed/*_report.json` | `batch_score` | run summary + drift status |

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the On-Prem deployment design and
the retraining strategy.
