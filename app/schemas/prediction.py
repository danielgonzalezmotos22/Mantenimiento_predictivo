"""Esquemas Pydantic para el endpoint /predict."""
from typing import Literal

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Lectura instantánea de los sensores de un reactor.

    Los nombres y rangos siguen el dataset Chemical Process Monitoring.
    El Pipeline entrenado se encarga de codificar las variables categóricas.
    """

    operating_regime: Literal["A", "B", "C"] = Field(..., description="Régimen operativo")
    reactor_id: str = Field(..., description="Identificador del reactor (ej. A_R1)")

    ambient_temp_effect: float = Field(..., ge=-10, le=10)
    reactor_temp: float = Field(..., ge=0, le=500, description="°C")
    reactor_pressure: float = Field(..., ge=0, le=100, description="bar")
    feed_flow_rate: float = Field(..., ge=0, le=500)
    coolant_flow_rate: float = Field(..., ge=0, le=500)
    agitator_speed_rpm: float = Field(..., ge=0, le=1000)
    reaction_rate: float = Field(..., ge=0, le=10)
    conversion_rate: float = Field(..., ge=0, le=100)
    selectivity: float = Field(..., ge=0, le=100)
    yield_pct: float = Field(..., ge=0, le=100)
    vibration_rms: float = Field(..., ge=0, le=20)
    motor_current: float = Field(..., ge=0, le=200)
    power_consumption_kw: float = Field(..., ge=0, le=200)
    temp_setpoint: float = Field(..., ge=0, le=500)
    pressure_setpoint: float = Field(..., ge=0, le=100)
    efficiency_loss_pct: float = Field(..., ge=0, le=100, description="Pérdida de eficiencia (%)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class FaultProbability(BaseModel):
    """Probabilidad asociada a una clase de fallo concreta."""

    fault_type: int
    probability: float


class SeverityClassification(BaseModel):
    """Resultado del modelo Hugging Face que enriquece la predicción."""

    label: str
    score: float


class PredictionResponse(BaseModel):
    """Respuesta del endpoint /predict."""

    fault_type: int = Field(..., description="Clase de fallo predicha (0-4)")
    fault_type_label: str = Field(..., description="Descripción legible del fallo")
    fault_probability: float = Field(..., ge=0, le=1)
    fault_probabilities: list[FaultProbability]
    time_to_fault_min: float = Field(..., description="Minutos estimados hasta el fallo")
    severity: SeverityClassification | None = None
    summary: str = Field(..., description="Resumen del estado del sistema")
