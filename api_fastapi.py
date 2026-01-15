"""API FastAPI para predições batidas do modelo CatBoost."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import uvicorn
import yaml
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import validate_columns


class PredictionRequest(BaseModel):
    registros: List[Dict[str, Any]] = Field(
        ..., description="Lista de registros seguindo o schema do treino."
    )


class PredictionResponse(BaseModel):
    fraud_proba: List[float]
    fraud_flag: List[int]


def load_config_artifacts():
    with open("src/config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(config["paths"]["artifacts_file"], "r", encoding="utf-8") as f:
        artifacts = json.load(f)
    model = CatBoostClassifier()
    model.load_model(config["paths"]["model_file"])
    return config, artifacts, model


app = FastAPI(title="Fraud Detection API", version="1.0.0")
config, artifacts, model = load_config_artifacts()


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    if not req.registros:
        raise HTTPException(status_code=400, detail="Lista de registros vazia.")

    df = pd.DataFrame(req.registros)
    try:
        validate_columns(df, artifacts["features"])
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    X = df[artifacts["features"]]
    probs = model.predict_proba(X)[:, 1]
    flags = (probs >= artifacts["threshold"]["value"]).astype(int)

    return PredictionResponse(
        fraud_proba=list(np.round(probs, 6)),
        fraud_flag=list(map(int, flags)),
    )


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
