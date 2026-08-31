import os
import time
from contextlib import asynccontextmanager
from typing import List

import mlflow
import mlflow.sklearn
import numpy as np

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from feature_engineering import (
    SELECTED_FEATURES,
    SAMPLING_FREQUENCY,
    WINDOW_SIZE,
    STEP_SIZE,
    extract_features_from_recording,
)


# ============================================================
# 1. Configuration
# ============================================================

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_URI = "models:/bearing_condition_logistic@champion"

RUN_THRESHOLD = 0.5


# ============================================================
# 2. Global model container
# ============================================================

champion_model = None


# ============================================================
# 3. Prometheus metrics
# ============================================================

REQUEST_COUNT = Counter(
    "bearing_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "bearing_api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "bearing_predictions_total",
    "Total number of model predictions",
    ["prediction", "condition"],
)

ANOMALY_PROBABILITY = Histogram(
    "bearing_anomaly_probability",
    "Distribution of run-level mean anomaly probability",
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
)

MODEL_READY = Gauge(
    "bearing_model_ready",
    "Whether the MLflow champion model is loaded and ready",
)


# ============================================================
# 4. Request schema
# ============================================================

class BearingSignalRequest(BaseModel):

    ch1: List[float] = Field(
        ...,
        description="Raw vibration signal from channel 1",
    )

    ch2: List[float] = Field(
        ...,
        description="Raw vibration signal from channel 2",
    )

    ch3: List[float] = Field(
        ...,
        description="Raw vibration signal from channel 3",
    )


# ============================================================
# 5. Response schema
# ============================================================

class BearingPredictionResponse(BaseModel):

    number_of_samples: int

    number_of_windows: int

    normal_windows: int

    anomalous_windows: int

    mean_anomaly_probability: float

    threshold: float

    prediction: int

    label: str

    condition: str


# ============================================================
# 6. FastAPI startup / shutdown lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global champion_model

    print("=" * 80)
    print("STARTING BEARING FAULT API")
    print("=" * 80)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    print(
        f"\nTracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )

    print(
        f"Loading model: "
        f"{MODEL_URI}"
    )

    # --------------------------------------------------------
    # Load Champion model ONCE
    # --------------------------------------------------------

    try:

        champion_model = mlflow.sklearn.load_model(
            MODEL_URI
        )

        MODEL_READY.set(1)

        print(
            "Champion model loaded successfully."
        )

    except Exception as error:

        champion_model = None

        MODEL_READY.set(0)

        print(
            f"Champion model loading failed: "
            f"{error}"
        )

        raise

    print("=" * 80)
    print("API READY")
    print("=" * 80)

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    MODEL_READY.set(0)

    print(
        "\nBearing Fault API shutting down."
    )


# ============================================================
# 7. Create FastAPI application
# ============================================================

app = FastAPI(

    title="Bearing Fault Detection API",

    description=(
        "Bearing condition diagnosis from "
        "three-channel raw vibration signals."
    ),

    version="1.1.0",

    lifespan=lifespan,
)


# ============================================================
# 8. Prometheus middleware
# ============================================================

@app.middleware("http")
async def prometheus_middleware(
    request,
    call_next,
):

    start_time = time.perf_counter()

    status_code = 500

    try:

        response = await call_next(request)

        status_code = response.status_code

        return response

    finally:

        duration = (
            time.perf_counter()
            - start_time
        )

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(
            duration
        )


# ============================================================
# 9. Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "Bearing Fault Detection API",

        "model":
            "bearing_condition_logistic@champion",

        "sampling_frequency_hz":
            SAMPLING_FREQUENCY,

        "window_size":
            WINDOW_SIZE,

        "step_size":
            STEP_SIZE,
    }


# ============================================================
# 10. Health endpoint
# ============================================================

@app.get("/health")
def health():

    if champion_model is None:

        raise HTTPException(
            status_code=503,
            detail="Champion model is not loaded.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_uri": MODEL_URI,
    }


# ============================================================
# 11. Prometheus metrics endpoint
# ============================================================

@app.get(
    "/metrics",
    include_in_schema=False,
)
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================
# 12. Prediction endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=BearingPredictionResponse,
)
def predict(
    request: BearingSignalRequest,
):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if champion_model is None:

        raise HTTPException(
            status_code=503,
            detail="Champion model is not loaded.",
        )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    try:

        ch1 = np.asarray(
            request.ch1,
            dtype=float,
        )

        ch2 = np.asarray(
            request.ch2,
            dtype=float,
        )

        ch3 = np.asarray(
            request.ch3,
            dtype=float,
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Signals could not be converted "
                f"to numerical arrays: {error}"
            ),
        )

    # --------------------------------------------------------
    # Validate signal lengths
    # --------------------------------------------------------

    if not (
        len(ch1)
        == len(ch2)
        == len(ch3)
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "ch1, ch2 and ch3 must have "
                "the same number of samples."
            ),
        )

    number_of_samples = len(ch1)

    if number_of_samples < WINDOW_SIZE:

        raise HTTPException(
            status_code=400,
            detail=(
                f"At least {WINDOW_SIZE} "
                "samples are required."
            ),
        )

    # --------------------------------------------------------
    # Check NaN / Inf
    # --------------------------------------------------------

    for channel_name, signal in [
        ("ch1", ch1),
        ("ch2", ch2),
        ("ch3", ch3),
    ]:

        if np.isnan(signal).any():

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{channel_name} "
                    "contains NaN values."
                ),
            )

        if np.isinf(signal).any():

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{channel_name} "
                    "contains Inf values."
                ),
            )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    try:

        feature_df = (
            extract_features_from_recording(
                ch1,
                ch2,
                ch3,
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Feature extraction failed: "
                f"{error}"
            ),
        )

    # --------------------------------------------------------
    # Feature contract
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature
        in SELECTED_FEATURES
        if feature
        not in feature_df.columns
    ]

    if missing_features:

        raise HTTPException(
            status_code=500,
            detail=(
                "Feature contract violation. "
                f"Missing: {missing_features}"
            ),
        )

    X = feature_df[
        SELECTED_FEATURES
    ]

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    try:

        window_predictions = (
            champion_model.predict(
                X
            )
        )

        window_probabilities = (
            champion_model.predict_proba(
                X
            )[:, 1]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model inference failed: "
                f"{error}"
            ),
        )

    # --------------------------------------------------------
    # Run-level aggregation
    # --------------------------------------------------------

    mean_probability = float(
        np.mean(
            window_probabilities
        )
    )

    final_prediction = int(
        mean_probability
        >= RUN_THRESHOLD
    )

    normal_windows = int(
        np.sum(
            window_predictions == 0
        )
    )

    anomalous_windows = int(
        np.sum(
            window_predictions == 1
        )
    )

    # --------------------------------------------------------
    # Engineering labels
    # --------------------------------------------------------

    if final_prediction == 1:

        label = "Before"
        condition = "Anomalous"

    else:

        label = "After"
        condition = "Normal"

    # --------------------------------------------------------
    # Prometheus ML metrics
    # --------------------------------------------------------

    ANOMALY_PROBABILITY.observe(
        mean_probability
    )

    PREDICTION_COUNT.labels(
        prediction=str(
            final_prediction
        ),
        condition=condition,
    ).inc()

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return BearingPredictionResponse(

        number_of_samples=
            number_of_samples,

        number_of_windows=
            len(feature_df),

        normal_windows=
            normal_windows,

        anomalous_windows=
            anomalous_windows,

        mean_anomaly_probability=
            mean_probability,

        threshold=
            RUN_THRESHOLD,

        prediction=
            final_prediction,

        label=
            label,

        condition=
            condition,
    )