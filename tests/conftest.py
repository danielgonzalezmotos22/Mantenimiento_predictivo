"""Fixtures compartidas para los tests.

Estrategia: los tests de la API no dependen de los .pkl reales ni de Groq ni
de Hugging Face. Sustituimos los servicios por mocks usando los hooks de
inyección de dependencias de FastAPI (`app.dependency_overrides`).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.routers.deps import (
    get_hf_service,
    get_llm_client,
    get_model_service,
    get_sensor_store,
)
from app.schemas.prediction import (
    FaultProbability,
    PredictionResponse,
    SensorReading,
    SeverityClassification,
)
from app.services.ml_model import FAULT_LABELS, ModelService

DATASET_PATH = Path("data/chemical_process_timeseries.csv")


# --- Fixtures de datos --------------------------------------------------------

@pytest.fixture(scope="session")
def dataset_path() -> Path:
    return DATASET_PATH


@pytest.fixture(scope="session")
def dataset_df(dataset_path: Path) -> pd.DataFrame:
    """Carga el CSV una sola vez por sesión para los tests de datos."""
    if not dataset_path.exists():
        pytest.skip(f"Dataset no disponible en {dataset_path}")
    return pd.read_csv(dataset_path, parse_dates=["timestamp"])


# --- Mocks de servicios -------------------------------------------------------

class _FakeFaultModel:
    classes_ = np.array([0, 1, 2, 3, 4])

    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.7, 0.1, 0.1, 0.05, 0.05]])


class _FakeTimeModel:
    def predict(self, X):
        return np.array([42.5])


class FakeModelService(ModelService):
    """ModelService que se salta `load()` y usa modelos sintéticos."""

    def __init__(self) -> None:
        super().__init__(Path("fake_fault.pkl"), Path("fake_time.pkl"))
        self.fault_model = _FakeFaultModel()
        self.time_model = _FakeTimeModel()
        self.loaded_at = datetime.utcnow()


class FakeHFService:
    model_name = "fake-hf"

    def classify(self, text: str) -> SeverityClassification:
        return SeverityClassification(label="Normal", score=0.91)


class FakeLLMClient:
    model = "fake-llm"

    def generate_report(self, prediction: PredictionResponse, language: str = "es") -> str:
        return f"[Informe simulado en {language}] Fallo: {prediction.fault_type_label}."


class FakeSensorStore:
    """Histórico mínimo en memoria, sin tocar el CSV."""

    is_ready = True
    total_rows = 3

    _rows = [
        {"timestamp": "2024-01-01 00:00:00", "reactor_id": "A_R1", "reactor_temp": 180.0},
        {"timestamp": "2024-01-01 00:01:00", "reactor_id": "A_R1", "reactor_temp": 181.0},
        {"timestamp": "2024-01-01 00:02:00", "reactor_id": "B_R1", "reactor_temp": 175.0},
    ]

    def query(self, reactor_id: str | None = None, limit: int = 100, offset: int = 0):
        rows = self._rows
        if reactor_id is not None:
            rows = [r for r in rows if r["reactor_id"] == reactor_id]
        return rows[offset : offset + limit]

    def reactors(self):
        return sorted({r["reactor_id"] for r in self._rows})


# --- Cliente de test ----------------------------------------------------------

@pytest.fixture
def app_with_mocks():
    """App FastAPI con todas las dependencias sustituidas por mocks."""
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: FakeModelService()
    app.dependency_overrides[get_hf_service] = lambda: FakeHFService()
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient()
    app.dependency_overrides[get_sensor_store] = lambda: FakeSensorStore()
    return app


@pytest.fixture
def client(app_with_mocks) -> TestClient:
    return TestClient(app_with_mocks)


@pytest.fixture
def sample_reading() -> dict[str, Any]:
    return SensorReading.model_json_schema()["example"] if False else {
        "operating_regime": "A",
        "reactor_id": "A_R1",
        "ambient_temp_effect": 0.0,
        "reactor_temp": 181.13,
        "reactor_pressure": 15.79,
        "feed_flow_rate": 101.10,
        "coolant_flow_rate": 79.15,
        "agitator_speed_rpm": 305.78,
        "reaction_rate": 0.72,
        "conversion_rate": 99.15,
        "selectivity": 91.92,
        "yield_pct": 82.03,
        "vibration_rms": 1.47,
        "motor_current": 45.88,
        "power_consumption_kw": 41.29,
        "temp_setpoint": 180.0,
        "pressure_setpoint": 12.0,
        "efficiency_loss_pct": 0.0,
    }


@pytest.fixture
def sample_prediction() -> PredictionResponse:
    return PredictionResponse(
        fault_type=0,
        fault_type_label=FAULT_LABELS[0],
        fault_probability=0.7,
        fault_probabilities=[
            FaultProbability(fault_type=i, probability=p)
            for i, p in enumerate([0.7, 0.1, 0.1, 0.05, 0.05])
        ],
        time_to_fault_min=42.5,
        severity=SeverityClassification(label="Normal", score=0.91),
        summary="Reactor A_R1 en operación normal.",
    )
