
import os
import cloudpickle
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

BUNDLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "fraud_detection_bundle.pkl"
)

with open(BUNDLE_PATH, "rb") as f:
    bundle = cloudpickle.load(f)

model = bundle["model"]
preprocess_features = bundle["preprocess_features"]
model_feature_names = bundle["feature_names"]
threshold = float(bundle["threshold"])

# Raw feature names come from the original training DataFrame.
raw_feature_names = list(
    preprocess_features.__globals__["X"].columns
)

app = FastAPI(
    title="Group 25 Credit Card Fraud Detection API",
    description="Educational transaction-scoring prototype",
    version="1.0.1",
)

class TransactionRequest(BaseModel):
    features: Dict[str, float]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "XGBoost",
        "raw_feature_count": len(raw_feature_names),
        "model_feature_count": len(model_feature_names),
        "threshold": threshold,
    }

@app.post("/predict")
def predict(request: TransactionRequest):
    missing = [
        name for name in raw_feature_names
        if name not in request.features
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing},
        )

    try:
        row = np.asarray(
            [request.features[name] for name in raw_feature_names],
            dtype=float,
        )

        if not np.isfinite(row).all():
            raise HTTPException(
                status_code=422,
                detail="Features must be finite numeric values.",
            )

        raw_df = pd.DataFrame(
            [row],
            columns=raw_feature_names,
        )

        processed = preprocess_features(raw_df)

        # Ensure the model receives the expected columns in the right order.
        processed = processed.loc[:, model_feature_names]

        probabilities = model.predict_proba(processed)
        fraud_probability = float(probabilities[0, 1])
        prediction = int(fraud_probability >= threshold)

        # Update monitoring counters without storing transaction features.
        with _monitor_lock:
            _monitor["successful_predictions"] += 1
            if prediction == 1:
                _monitor["fraud_flags"] += 1

        return {
            "fraud_probability": fraud_probability,
            "threshold": threshold,
            "prediction": prediction,
            "decision": (
                "FLAG_FOR_REVIEW"
                if prediction == 1
                else "NOT_FLAGGED"
            ),
            "notice": (
                "Educational prototype; not for live transaction decisions."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to score transaction: {str(exc)}",
        )


# --- Prototype API monitoring ---
import time
from threading import Lock
from collections import defaultdict

_monitor_lock = Lock()
_monitor = {
    "total_requests": 0,
    "successful_predictions": 0,
    "api_errors": 0,
    "validation_failures": 0,
    "fraud_flags": 0,
    "total_prediction_latency_ms": 0.0,
}

@app.middleware("http")
async def monitor_requests(request, call_next):
    start = time.perf_counter()
    with _monitor_lock:
        _monitor["total_requests"] += 1

    try:
        response = await call_next(request)
        if response.status_code >= 400:
            with _monitor_lock:
                _monitor["api_errors"] += 1
                if response.status_code == 422:
                    _monitor["validation_failures"] += 1
        return response
    except Exception:
        with _monitor_lock:
            _monitor["api_errors"] += 1
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        with _monitor_lock:
            _monitor["total_prediction_latency_ms"] += elapsed_ms


@app.get("/metrics")
def metrics():
    with _monitor_lock:
        data = dict(_monitor)

    successful = data["successful_predictions"]
    total = data["total_requests"]

    data["average_request_latency_ms"] = (
        data["total_prediction_latency_ms"] / total if total else 0.0
    )
    data["fraud_flag_rate"] = (
        data["fraud_flags"] / successful if successful else 0.0
    )
    data.pop("total_prediction_latency_ms", None)
    return data
