"""POST /predict — predicción de fallo + tiempo restante + severidad (HF)."""
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.routers.deps import get_hf_service, get_model_service
from app.schemas.prediction import PredictionResponse, SensorReading
from app.services.hf_model import HFClassifierService
from app.services.ml_model import ModelService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predicción"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    reading: SensorReading,
    model_service: ModelService = Depends(get_model_service),
    hf_service: HFClassifierService = Depends(get_hf_service),
) -> PredictionResponse:
    try:
        prediction = model_service.predict(reading)
    except Exception as exc:
        logger.exception("Error durante la predicción")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error de inferencia: {exc}",
        )

    # Enriquecemos con el modelo HF, pero su fallo no debe romper la respuesta.
    try:
        prediction.severity = hf_service.classify(prediction.summary)
    except Exception as exc:  # pragma: no cover - depende de torch/transformers
        logger.warning("No se pudo clasificar severidad con HF: %s", exc)
        prediction.severity = None

    return prediction
