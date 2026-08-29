import numpy as np
import pytest

from fastapi.testclient import TestClient

import app as app_module


class FakeChampionModel:
    """
    Deterministic fake model used for API unit tests.

    It avoids any dependency on:
    - MLflow
    - PostgreSQL
    - MinIO
    - the real production model
    """

    def __init__(
        self,
        prediction=0,
        anomaly_probability=0.2,
    ):
        self.prediction = prediction
        self.anomaly_probability = anomaly_probability

    def predict(self, X):
        return np.full(
            len(X),
            self.prediction,
            dtype=int,
        )

    def predict_proba(self, X):
        normal_probability = (
            1.0 - self.anomaly_probability
        )

        return np.column_stack(
            [
                np.full(
                    len(X),
                    normal_probability,
                ),
                np.full(
                    len(X),
                    self.anomaly_probability,
                ),
            ]
        )


@pytest.fixture
def client():
    """
    Create TestClient without running the application's
    MLflow lifespan.
    """
    return TestClient(
        app_module.app
    )


@pytest.fixture
def valid_signal():
    """
    Minimum valid recording: exactly one analysis window.
    """
    n_samples = app_module.WINDOW_SIZE

    time = (
        np.arange(n_samples)
        / app_module.SAMPLING_FREQUENCY
    )

    return np.sin(
        2.0 * np.pi * 1000.0 * time
    ).tolist()


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert (
        data["message"]
        == "Bearing Fault Detection API"
    )

    assert (
        data["model"]
        == "bearing_condition_logistic@champion"
    )

    assert (
        data["sampling_frequency_hz"]
        == app_module.SAMPLING_FREQUENCY
    )

    assert (
        data["window_size"]
        == app_module.WINDOW_SIZE
    )

    assert (
        data["step_size"]
        == app_module.STEP_SIZE
    )


def test_health_returns_503_without_model(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "champion_model",
        None,
    )

    response = client.get("/health")

    assert response.status_code == 503

    assert response.json() == {
        "detail":
            "Champion model is not loaded."
    }


def test_health_returns_200_with_model(
    client,
    monkeypatch,
):
    fake_model = FakeChampionModel()

    monkeypatch.setattr(
        app_module,
        "champion_model",
        fake_model,
    )

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_uri"] == app_module.MODEL_URI


def test_predict_returns_503_without_model(
    client,
    valid_signal,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "champion_model",
        None,
    )

    payload = {
        "ch1": valid_signal,
        "ch2": valid_signal,
        "ch3": valid_signal,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 503


def test_predict_rejects_mismatched_lengths(
    client,
    valid_signal,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "champion_model",
        FakeChampionModel(),
    )

    payload = {
        "ch1": valid_signal,
        "ch2": valid_signal,
        "ch3": valid_signal[:-1],
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400

    assert (
        "same number of samples"
        in response.json()["detail"]
    )


def test_predict_rejects_short_recording(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "champion_model",
        FakeChampionModel(),
    )

    short_signal = [
        0.0
    ] * (
        app_module.WINDOW_SIZE - 1
    )

    payload = {
        "ch1": short_signal,
        "ch2": short_signal,
        "ch3": short_signal,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400

    assert (
        "At least"
        in response.json()["detail"]
    )


def test_predict_normal_condition(
    client,
    valid_signal,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "champion_model",
        FakeChampionModel(
            prediction=0,
            anomaly_probability=0.2,
        ),
    )

    payload = {
        "ch1": valid_signal,
        "ch2": valid_signal,
        "ch3": valid_signal,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["number_of_samples"]
        == app_module.WINDOW_SIZE
    )

    assert data["number_of_windows"] == 1

    assert data["normal_windows"] == 1
    assert data["anomalous_windows"] == 0

    assert (
        data["mean_anomaly_probability"]
        == pytest.approx(0.2)
    )

    assert (
        data["threshold"]
        == app_module.RUN_THRESHOLD
    )

    assert data["prediction"] == 0
    assert data["label"] == "After"
    assert data["condition"] == "Normal"


def test_predict_anomalous_condition(
    client,
    valid_signal,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "champion_model",
        FakeChampionModel(
            prediction=1,
            anomaly_probability=0.8,
        ),
    )

    payload = {
        "ch1": valid_signal,
        "ch2": valid_signal,
        "ch3": valid_signal,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["normal_windows"] == 0
    assert data["anomalous_windows"] == 1

    assert (
        data["mean_anomaly_probability"]
        == pytest.approx(0.8)
    )

    assert data["prediction"] == 1
    assert data["label"] == "Before"
    assert data["condition"] == "Anomalous"


def test_prediction_threshold_boundary(
    client,
    valid_signal,
    monkeypatch,
):
    """
    RUN_THRESHOLD uses >=, therefore exactly 0.5
    must be classified as anomalous.
    """
    monkeypatch.setattr(
        app_module,
        "champion_model",
        FakeChampionModel(
            prediction=1,
            anomaly_probability=0.5,
        ),
    )

    payload = {
        "ch1": valid_signal,
        "ch2": valid_signal,
        "ch3": valid_signal,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["label"] == "Before"
    assert data["condition"] == "Anomalous"