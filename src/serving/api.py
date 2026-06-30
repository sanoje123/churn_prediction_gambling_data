"""
FastAPI serving application.

Exposes:
    GET  /health         -> liveness/readiness probe (load balancer / k8s)
    POST /predict        -> score a single player
    POST /predict_batch  -> score a list of players

Run locally:
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.inference import get_model

app = FastAPI(
    title="Live-Betting Churn Prediction API",
    description="Scores players for 30-day churn risk (Logistic Regression).",
    version="1.0.0",
)


class PlayerRecord(BaseModel):
    """Raw player features as captured at the observation snapshot.

    Optional date fields enable the recency features; if omitted, sensible
    fallbacks are used (same as training).
    """
    user_id: Optional[str] = None
    tenure_days: float
    live_bets_count: float
    avg_bet_amount: Optional[float] = None
    total_turnover: float
    total_payout: float = 0.0
    ggr: float = 0.0
    ggr_margin: Optional[float] = None
    deposit_count: Optional[float] = None
    total_deposit_amount: Optional[float] = None
    deposit_to_turnover_ratio: Optional[float] = None
    days_active_in_observation: float
    days_since_last_bet: float
    observation_start_date: Optional[str] = None
    first_live_bet_date: Optional[str] = None
    last_live_bet_date: Optional[str] = None

    model_config = {"extra": "allow"}  # tolerate extra raw columns


class BatchRequest(BaseModel):
    players: List[PlayerRecord] = Field(..., min_length=1)


@app.get("/health")
def health():
    """Liveness + readiness: confirms the model artefact is loadable."""
    try:
        meta = get_model().metadata
        return {"status": "ok", "model_version": meta.get("version", "unknown")}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model not ready: {exc}")


@app.post("/predict")
def predict_one(player: PlayerRecord):
    try:
        return get_model().score(player.model_dump()).iloc[0].to_dict()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    try:
        records = [p.model_dump() for p in req.players]
        scored = get_model().score(records)
        return {"results": scored.to_dict(orient="records")}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
