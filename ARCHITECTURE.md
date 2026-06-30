# Production Architecture — On-Premise Deployment & Retraining

This document proposes a conceptual architecture for running the live-betting
churn pipeline on the company's **own infrastructure** (no public cloud), and
describes how the model is **regularly retrained** and kept healthy.

---

## 1. Design principles

1. **Batch-first.** Churn is not a millisecond decision. The retention team works
   from a daily/weekly ranked list, so the *primary* product is a **scheduled
   batch scoring job**. A real-time API exists for on-demand/ad-hoc lookups but
   is secondary.
2. **One artefact, no skew.** The whole preprocessing chain lives inside the
   scikit-learn `Pipeline`. Training and serving call the *same*
   `clean_data → build_features` code, so there is no separate transformer to
   drift out of sync. This is enforced by a `test_train_serve_feature_parity`
   unit test.
3. **Everything reproducible & versioned.** Code in Git, data versioned by
   snapshot, every model in a registry with metrics + the hash of the data it was
   trained on (`model_metadata.json`).
4. **Keep it boring.** The model is a calibrated Logistic Regression — cheap to
   train (~3 s on CPU), trivial to serve, easy to explain to compliance. No GPU
   needed in production.

---

## 2. Conceptual architecture (On-Prem)

```
                          ON-PREMISE DATA CENTRE
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  ┌────────────────┐     nightly ETL      ┌───────────────────────────┐     │
│  │  Operational   │ ───────────────────► │   Data Warehouse / Lake   │     │
│  │  DBs (bets,    │   (Airflow / NiFi)   │  PostgreSQL  +  MinIO     │     │
│  │  deposits)     │                      │  (S3-compatible object    │     │
│  └────────────────┘                      │   store, versioned)       │     │
│                                          └─────────────┬─────────────┘     │
│                                                        │player snapshot CSV│
│                ┌───────────────────────────────────────┼───────────────────┤
│                │                TRAINING PLANE         ▼                   │
│                │   ┌──────────────────────────────────────────────────┐    │
│   Git (GitHub) │   │  Orchestrator (Airflow DAG / cron)               │    │
│   ─────────────┼──►│   run_pipeline.py:                               │    │
│   CI/CD        │   │   load → validate → clean → features → train     │    │
│   (GitHub      │   │        → evaluate → QUALITY GATE → register      │    │
│    Actions,    │   └────────────────────┬─────────────────────────────┘    │
│    self-hosted)│                        │ logs params/metrics/model        │
│                │                        ▼                                  │
│                │            ┌────────────────────────┐                     │
│                │            │  MLflow Tracking +     │  artefacts ► MinIO  │
│                │            │  Model Registry        │  backend  ► Postgres│
│                │            │  (Staging / Production)│                     │
│                │            └───────────┬────────────┘                     │
│                │                        │ promote "Production" model       │
│   ─────────────┴────────────────────────┼──────────────────────────────────┤
│                         SERVING PLANE   ▼                                  │
│   ┌──────────────────────────┐   ┌──────────────────────────────────────┐  │
│   │  BATCH SCORING (primary) │   │  REAL-TIME API (secondary)           │  │
│   │  batch_score.py          │   │  FastAPI + Uvicorn, in Docker        │  │
│   │  cron/Airflow → CSV +    │   │  on Kubernetes / OpenShift           │  │
│   │  risk tiers → CRM table  │   │  /health /predict /predict_batch     │  │
│   └─────────────┬────────────┘   └──────────────────┬───────────────────┘  │
│                 │ scored list                       │ predictions          │
│                 ▼                                   ▼                      │
│        ┌──────────────────┐               ┌────────────────────┐           │
│        │ CRM / campaign   │               │ internal services  │           │
│        │ tool (retention) │               │ (apps, dashboards) │           │
│        └──────────────────┘               └────────────────────┘           │
│                                                                            │
│   ┌─────────────────────────── MONITORING ──────────────────────────────┐  │
│   │ drift.py (PSI) + performance vs gate  →  Prometheus  →  Grafana     │  │
│   │ Alerts (e-mail / Slack) → trigger retrain.py                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tooling — concern → On-Prem tool

| Concern | Tool (On-Prem) | Why |
|---------|----------------|-----|
| Source control | **GitHub (Enterprise / self-hosted)** | code + CI in one place |
| CI/CD | **GitHub Actions** on **self-hosted runners** | lint → test → build image → deploy (see `.github/workflows/ci.yml`) |
| Data export / ETL | **Apache Airflow** (or NiFi) | builds the daily player snapshot from operational DBs |
| Data / artefact store | **PostgreSQL** + **MinIO** | warehouse tables; MinIO is S3-compatible object storage on-prem |
| Experiment tracking + registry | **MLflow** (Postgres backend, MinIO artefacts) | versioned models, metrics, Staging→Production promotion |
| Packaging | **Docker** | identical runtime dev → prod |
| Orchestration / serving | **Kubernetes / OpenShift** | scale, rolling deploys, self-healing for the API |
| Scheduling (batch + retrain) | **Airflow** or **cron** | nightly scoring, weekly/monthly retrain |
| Monitoring | **Prometheus + Grafana** | system + model metrics, dashboards |
| Alerting | **Alertmanager → e-mail/Slack** | drift / decay / job-failure alerts |
| Secrets | **HashiCorp Vault** | DB creds, registry tokens |

> If Kubernetes is not available, the API runs equally well as a Docker container
> managed by **systemd** or **Docker Compose** behind an **Nginx** reverse proxy.

---

## 4. Deployment — the basic steps

1. **Package.** CI builds a Docker image from the `Dockerfile` (Python 3.11-slim
   + `requirements.txt` + `src/` + the `artifacts/` from the registry).
2. **Test gate.** CI runs `ruff` + `pytest` (including train/serve parity). A red
   build never ships.
3. **Register the model.** Training logs to MLflow; a human (or an automated gate)
   promotes the run from **Staging → Production** in the registry.
4. **Ship the image** to the internal container registry and roll it out to
   Kubernetes (`kubectl set image …`), with `/health` as the readiness probe and
   a **manual approval** before prod (a protected GitHub **Environment** with
   required reviewers — see `.github/workflows/ci.yml`).
5. **Schedule batch scoring.** An Airflow DAG runs `batch_score.py` nightly:
   pull snapshot → score → write `risk_tier`/`risk_decile` back to a CRM table →
   emit a drift report.
6. **Observe.** Prometheus scrapes job/model metrics; Grafana dashboards + alerts
   watch latency, volume, drift, and (when labels mature) live performance.

**Rollback** is trivial: the previous model is in the registry and archived under
`artifacts/archive/`, so promoting the prior version (or `kubectl rollout undo`)
restores the last-good state in seconds.

---

## 5. Regular retraining (model updating)

### 5.1 The key constraint — labels arrive late
Churn is defined over a **30-day outcome window**. A player snapshotted today is
not *known* to have churned until 30 days later. So:

* **Scoring** runs continuously on fresh snapshots (no labels needed).
* **Training labels** are only complete for snapshots that are ≥ 30 days old.
  The retraining job therefore trains on a *lagged* window of fully-observed data.

### 5.2 Retraining triggers (any one fires a run)
1. **Scheduled cadence — the baseline.** Retrain **monthly** (betting behaviour
   shifts with seasons, promotions, sporting calendars). Cheap model → cheap to
   redo often.
2. **Data drift.** `drift.py` computes **PSI** per feature between the training
   reference (`reference_stats.json`) and each new scored batch. `PSI ≥ 0.25` on
   key features raises an alert and can trigger an early retrain.
3. **Performance decay.** Once the 30-day labels mature, `performance_report`
   re-measures ROC-AUC / PR-AUC / Brier on recent cohorts. Dropping below the
   gate (`MIN_ACCEPTABLE_*`) triggers a retrain. Brier is watched specifically —
   calibration drift is the first thing to break for a probability model.

`should_retrain(drift, perf)` encapsulates this decision.

### 5.3 The retraining job — champion / challenger
`retrain.py` does **not** blindly overwrite the live model:

1. Train a **challenger** on the latest fully-labelled data.
2. Check it against the **absolute quality gate** (`passes_quality_gate`). Fail →
   keep champion, alert.
3. Compare challenger vs **champion** on the *same* held-out set. If the
   challenger regresses on PR-AUC beyond a small tolerance → keep champion.
4. Only if it clears both checks: **archive** the current champion, refit the
   challenger on *all* data, and promote it as the new champion.

This guarantees the live model never gets *worse* from an automated run, and
every previous version is one copy away for rollback.

### 5.4 Validation discipline
Because data is time-ordered, model selection during retraining should use
**rolling-origin (time-based) cross-validation** — train on months *1…k*, test on
month *k+1* — rather than a random split, to get an honest estimate of
next-month performance. (The analysis used a random split for speed; production
retraining should prefer the temporal split, noted in the recommendations.)

### 5.5 Full retraining loop

```
   (monthly)        (per batch)         (when labels mature)
   schedule  ─┐     drift alert ─┐      perf decay ─┐
              ▼                  ▼                  ▼
            ┌──────────────────────────────────────────┐
            │  retrain.py: train challenger on latest  │
            │  fully-labelled (lagged 30d) data        │
            └───────────────┬──────────────────────────┘
                            ▼
              quality gate + champion/challenger compare
                  │                         │
            pass  ▼                    fail │
        archive champion →           keep champion,
        promote challenger →            raise alert
        register "Production"
                  │
                  ▼
        CI redeploys serving image / batch job picks up new model
```

---

## 6. Why this is low-risk to operate

* **Cheap & fast** — LogReg trains in seconds on CPU; retraining is a non-event.
* **Calibrated** — probabilities are trustworthy (Brier 0.106), so risk tiers and
  campaign budgeting map directly to predicted probability.
* **Explainable** — signed coefficients satisfy compliance/audit needs in a
  regulated betting context.
* **Single pickled pipeline** — the artefact *is* the preprocessing + model, so a
  deploy is just "ship a new `.pkl`"; there is nothing else to keep in sync.
