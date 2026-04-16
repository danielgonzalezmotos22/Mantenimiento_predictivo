"""Tests de la API: contratos de los endpoints y manejo de errores HTTP."""
from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient


# --- /health -----------------------------------------------------------------

def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "models_loaded" in body
    assert "dataset_loaded" in body


# --- /model-info -------------------------------------------------------------

def test_model_info(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "fault_classifier" in body
    assert "time_regressor" in body
    assert "feature_columns" in body
    assert isinstance(body["feature_columns"], list)
    assert len(body["feature_columns"]) >= 17


# --- /predict ----------------------------------------------------------------

def test_predict_happy_path(client: TestClient, sample_reading: dict) -> None:
    r = client.post("/predict", json=sample_reading)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fault_type"] in {0, 1, 2, 3, 4}
    assert 0.0 <= body["fault_probability"] <= 1.0
    assert body["time_to_fault_min"] >= 0
    assert "fault_type_label" in body
    assert "summary" in body
    assert isinstance(body["fault_probabilities"], list)
    assert len(body["fault_probabilities"]) == 5
    # Suma de probabilidades ≈ 1
    total = sum(p["probability"] for p in body["fault_probabilities"])
    assert pytest.approx(total, abs=1e-6) == 1.0


def test_predict_missing_field_returns_422(client: TestClient, sample_reading: dict) -> None:
    payload = deepcopy(sample_reading)
    del payload["reactor_temp"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_out_of_range_returns_422(client: TestClient, sample_reading: dict) -> None:
    payload = deepcopy(sample_reading)
    payload["reactor_temp"] = -50  # fuera del rango definido en el schema
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_regime_returns_422(client: TestClient, sample_reading: dict) -> None:
    payload = deepcopy(sample_reading)
    payload["operating_regime"] = "Z"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_empty_body_returns_422(client: TestClient) -> None:
    r = client.post("/predict", json={})
    assert r.status_code == 422


# --- /generate-report --------------------------------------------------------

def test_generate_report_happy_path(client: TestClient, sample_prediction) -> None:
    payload = {"prediction": sample_prediction.model_dump(), "language": "es"}
    r = client.post("/generate-report", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "es"
    assert "report" in body and len(body["report"]) > 0
    assert "model" in body


def test_generate_report_missing_prediction_returns_422(client: TestClient) -> None:
    r = client.post("/generate-report", json={"language": "es"})
    assert r.status_code == 422


# --- /sensores ---------------------------------------------------------------

def test_sensors_default(client: TestClient) -> None:
    r = client.get("/sensores")
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    assert "total" in body
    assert "returned" in body
    assert isinstance(body["data"], list)


def test_sensors_filter_by_reactor(client: TestClient) -> None:
    r = client.get("/sensores", params={"reactor_id": "A_R1"})
    assert r.status_code == 200
    body = r.json()
    assert body["reactor_id"] == "A_R1"
    for row in body["data"]:
        assert row["reactor_id"] == "A_R1"


def test_sensors_pagination(client: TestClient) -> None:
    r = client.get("/sensores", params={"limit": 1, "offset": 0})
    assert r.status_code == 200
    assert len(r.json()["data"]) <= 1


def test_sensors_invalid_limit_returns_422(client: TestClient) -> None:
    r = client.get("/sensores", params={"limit": -1})
    assert r.status_code == 422


def test_sensors_reactors_list(client: TestClient) -> None:
    r = client.get("/sensores/reactores")
    assert r.status_code == 200
    body = r.json()
    assert "reactors" in body
    assert isinstance(body["reactors"], list)


# --- 404 ---------------------------------------------------------------------

def test_unknown_route_returns_404(client: TestClient) -> None:
    r = client.get("/no-existe")
    assert r.status_code == 404
