"""Esquemas para el endpoint /generate-report."""
from pydantic import BaseModel, Field

from app.schemas.prediction import PredictionResponse


class ReportRequest(BaseModel):
    """Toma como entrada la respuesta del endpoint /predict."""

    prediction: PredictionResponse
    language: str = Field(default="es", description="Idioma del informe")


class ReportResponse(BaseModel):
    report: str
    model: str
    language: str
