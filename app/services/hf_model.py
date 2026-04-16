"""Servicio de clasificación zero-shot con Hugging Face.

Toma el `summary` textual de la predicción y lo clasifica en niveles de
severidad. Aporta una interpretación semántica encima de la predicción
numérica del modelo propio.

El modelo se carga bajo demanda (lazy) la primera vez que se invoca, para
no penalizar el arranque del servidor (bart-large-mnli pesa ~1.6 GB).
"""
from __future__ import annotations

import logging
from typing import Any

from app.schemas.prediction import SeverityClassification

logger = logging.getLogger(__name__)

SEVERITY_LABELS: list[str] = [
    "critical failure imminent",
    "high risk of failure",
    "moderate degradation",
    "minor anomaly",
    "stable normal operation",
]

# Mapeo a etiquetas en español que devolveremos al cliente.
_LABEL_TO_ES: dict[str, str] = {
    "critical failure imminent": "Crítico",
    "high risk of failure": "Alto",
    "moderate degradation": "Medio",
    "minor anomaly": "Bajo",
    "stable normal operation": "Normal",
}


class HFClassifierService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            # Import perezoso para que el resto de la app pueda arrancar
            # aunque transformers o torch no estén disponibles en el entorno.
            from transformers import pipeline  # type: ignore

            logger.info("Cargando modelo HF '%s'...", self.model_name)
            self._pipeline = pipeline("zero-shot-classification", model=self.model_name)
            logger.info("Modelo HF cargado.")
        except Exception as exc:  # pragma: no cover - depende del entorno
            logger.exception("No se pudo cargar el modelo HF: %s", exc)
            raise

    def classify(self, text: str) -> SeverityClassification:
        self._ensure_loaded()
        assert self._pipeline is not None
        result = self._pipeline(text, candidate_labels=SEVERITY_LABELS)
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        return SeverityClassification(
            label=_LABEL_TO_ES.get(top_label, top_label),
            score=top_score,
        )
