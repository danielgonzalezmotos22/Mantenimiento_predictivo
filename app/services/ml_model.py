"""Servicio que encapsula la carga y uso de los modelos propios (RandomForest).

Los modelos fueron entrenados como `sklearn.pipeline.Pipeline`, por lo que el
preprocesado (encoding de categóricas, imputación, escalado) está embebido en
el .pkl. En inferencia basta con pasarles un DataFrame con los nombres de
columna originales del dataset.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas.prediction import (
    FaultProbability,
    PredictionResponse,
    SensorReading,
)

# Mapeo legible de las clases de fallo (definido en el dataset original).
FAULT_LABELS: dict[int, str] = {
    0: "Operación normal",
    1: "Degradación del sistema de refrigeración",
    2: "Obstrucción del caudal de alimentación",
    3: "Deriva de sensor",
    4: "Desgaste mecánico",
}

# Orden esperado de columnas. Debe coincidir con el del entrenamiento.
FEATURE_COLUMNS: list[str] = [
    "operating_regime",
    "reactor_id",
    "ambient_temp_effect",
    "reactor_temp",
    "reactor_pressure",
    "feed_flow_rate",
    "coolant_flow_rate",
    "agitator_speed_rpm",
    "reaction_rate",
    "conversion_rate",
    "selectivity",
    "yield_pct",
    "vibration_rms",
    "motor_current",
    "power_consumption_kw",
    "temp_setpoint",
    "pressure_setpoint",
    "efficiency_loss_pct",
]


class ModelService:
    """Carga los pipelines de clasificación y regresión y expone `predict`."""

    def __init__(self, fault_path: Path, time_path: Path) -> None:
        self.fault_path = fault_path
        self.time_path = time_path
        self.fault_model: Any | None = None
        self.time_model: Any | None = None
        self.loaded_at: datetime | None = None

    def load(self) -> None:
        """Carga los .pkl en memoria. Se llama en el `lifespan` de FastAPI."""
        if not self.fault_path.exists():
            raise FileNotFoundError(f"Modelo de clasificación no encontrado: {self.fault_path}")
        if not self.time_path.exists():
            raise FileNotFoundError(f"Modelo de regresión no encontrado: {self.time_path}")

        self.fault_model = joblib.load(self.fault_path)
        self.time_model = joblib.load(self.time_path)
        self.loaded_at = datetime.utcnow()

    @property
    def is_ready(self) -> bool:
        return self.fault_model is not None and self.time_model is not None

    def _to_dataframe(self, reading: SensorReading) -> pd.DataFrame:
        """Convierte la entrada a un DataFrame con el orden de columnas correcto."""
        row = reading.model_dump()
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, reading: SensorReading) -> PredictionResponse:
        """Ejecuta los dos modelos y construye la respuesta agregada."""
        if not self.is_ready:
            raise RuntimeError("Los modelos no están cargados.")

        df = self._to_dataframe(reading)

        # Clasificación
        fault_pred = int(self.fault_model.predict(df)[0])
        proba = self.fault_model.predict_proba(df)[0]
        classes = self.fault_model.classes_

        probabilities = [
            FaultProbability(fault_type=int(cls), probability=float(p))
            for cls, p in zip(classes, proba)
        ]
        fault_proba = float(proba[np.where(classes == fault_pred)[0][0]])

        # Regresión
        time_pred = float(self.time_model.predict(df)[0])
        time_pred = max(0.0, time_pred)  # nunca negativo

        label = FAULT_LABELS.get(fault_pred, f"Fallo tipo {fault_pred}")
        summary = self._build_summary(reading, fault_pred, label, fault_proba, time_pred)

        return PredictionResponse(
            fault_type=fault_pred,
            fault_type_label=label,
            fault_probability=fault_proba,
            fault_probabilities=probabilities,
            time_to_fault_min=time_pred,
            severity=None,  # lo rellena el router tras llamar al modelo HF
            summary=summary,
        )

    @staticmethod
    def _build_summary(
        reading: SensorReading,
        fault_type: int,
        label: str,
        proba: float,
        time_to_fault: float,
    ) -> str:
        """Genera un resumen textual usado por el modelo HF y el LLM."""
        return (
            f"Reactor {reading.reactor_id} (régimen {reading.operating_regime}). "
            f"Temperatura {reading.reactor_temp:.1f}°C "
            f"(setpoint {reading.temp_setpoint:.1f}°C), "
            f"presión {reading.reactor_pressure:.2f} bar, "
            f"vibración {reading.vibration_rms:.2f} RMS, "
            f"corriente motor {reading.motor_current:.1f} A. "
            f"Predicción: {label} (clase {fault_type}, "
            f"probabilidad {proba:.1%}), "
            f"tiempo estimado hasta fallo {time_to_fault:.1f} min."
        )
