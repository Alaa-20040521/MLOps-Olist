from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    PredictionInput,
    PredictionResponse,
)
from src.prediction.predictor import Predictor


app = FastAPI(
    title="Olist Late Delivery Prediction API",
    version="1.0.0",
    description=(
        "Inference API for the Olist late-delivery "
        "prediction model."
    ),
)


predictor = Predictor()


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy",
        "model_version": "1.0.0",
    }


@app.get("/model")
def model_info():
    return {
        "model_name": "olist-late-delivery-model",
        "model_version": "1.0.0",
        "model_alias": "champion",
        "model_type": "XGBClassifier",
        "threshold": predictor.threshold,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(data: PredictionInput):

    try:
        input_df = pd.DataFrame(
            [data.model_dump()]
        )

        result = predictor.predict(input_df)

        return {
            "prediction": result["predictions"][0],
            "probability": result["probabilities"][0],
            "threshold": result["threshold"],
            "model_version": result["model_version"],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(data: List[PredictionInput]):

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Batch input cannot be empty.",
        )

    try:
        input_df = pd.DataFrame(
            [item.model_dump() for item in data]
        )

        result = predictor.predict(input_df)

        return {
            "predictions": result["predictions"],
            "probabilities": result["probabilities"],
            "threshold": result["threshold"],
            "model_version": result["model_version"],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed.",
        ) from exc