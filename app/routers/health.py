"""GET /health — comprobación rápida del estado del servidor."""
from fastapi import APIRouter, Request

router = APIRouter(tags=["meta"])


@router.get("/health")
def health(request: Request) -> dict:
    state = request.app.state
    model_service = getattr(state, "model_service", None)
    sensor_store = getattr(state, "sensor_store", None)
    return {
        "status": "ok",
        "models_loaded": bool(model_service and model_service.is_ready),
        "dataset_loaded": bool(sensor_store and sensor_store.is_ready),
    }
