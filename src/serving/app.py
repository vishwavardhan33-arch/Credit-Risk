"""FastAPI service: POST /predict returns a calibrated default probability and an approve/reject decision.

    MODEL_DIR=models uvicorn src.serving.app:app --port 8000
    # docs at http://localhost:8000/docs

Environment variables:
    MODEL_DIR        folder with lgbm_model.joblib and model_config.json (default: models)
    PREDICTION_LOG   optional path; if set, every scored application is appended as one JSON line,
                     which src/monitoring/drift_report.py can read. Treat this file as personal data.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.serving.scoring import Scorer, clean


class PredictRequest(BaseModel):
    applications: list[dict[str, Any]] = Field(
        min_length=1, max_length=1000,
        description="One or more applicants: feature name -> value. Missing features are treated as missing values.")
    explain_top_k: int = Field(3, ge=0, le=10, description="How many top risk-increasing features to return (0 = none).")


class Factor(BaseModel):
    feature: str
    contribution: float
    value: Any = None


class Prediction(BaseModel):
    id: Any = None
    default_probability: float
    decision: str
    threshold: float
    missing_features: int = Field(0, description="How many of the model's input features were missing or unseen.")
    warnings: list[str] = []
    top_risk_factors: list[Factor] = []


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


def create_app(model_dir: str | None = None, prediction_log: str | None = None) -> FastAPI:
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = Scorer(model_dir or os.environ.get("MODEL_DIR", "models"))
        app.state.log_path = prediction_log or os.environ.get("PREDICTION_LOG")
        yield

    app = FastAPI(title="Credit risk scoring API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        s: Scorer = request.app.state.scorer
        return {"status": "ok", "model_version": s.version, "n_features": len(s.columns)}

    @app.get("/model/info")
    def model_info(request: Request):
        s: Scorer = request.app.state.scorer
        c = s.config
        return {
            "model_version": s.version,
            "threshold": s.threshold,
            "assumptions": c.get("assumptions"),
            "calibrator": s.calibration["kind"],
            "n_features": len(s.columns),
            "dropped_columns": c.get("dropped_columns", []),
            "test_metrics": c.get("test_metrics"),
            "test_summary": c.get("test_summary"),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request):
        s: Scorer = request.app.state.scorer
        try:
            preds, X = s.predict(req.applications, req.explain_top_k)
        except Exception as e:  # malformed values that survive validation
            raise HTTPException(status_code=422, detail=f"could not score input: {e}") from e

        log_path = request.app.state.log_path
        if log_path:
            ts = datetime.now(timezone.utc).isoformat()
            lines = [json.dumps({"ts": ts, "model_version": s.version,
                                 "features": {c: clean(X.iloc[i][c]) for c in s.columns},
                                 "score": preds[i]["default_probability"]})
                     for i in range(len(preds))]
            with lock, open(log_path, "a") as f:
                f.write("\n".join(lines) + "\n")
        return {"model_version": s.version, "predictions": preds}

    return app


app = create_app()
