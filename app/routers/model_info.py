"""GET /model-info — metadatos sobre los modelos cargados."""
from datetime import datetime

from fastapi import APIRouter, Depends

from app.routers.deps import get_model_service
from app.services.ml_model import FEATURE_COLUMNS, ModelService

router = APIRouter(tags=["meta"])


@router.get("/model-info")
def model_info(model_service: ModelService = Depends(get_model_service)) -> dict:
    return {
        "fault_classifier": {
            "type": type(model_service.fault_model).__name__,
            "path": str(model_service.fault_path),
            "classes": [int(c) for c in getattr(model_service.fault_model, "classes_", [])],
        },
        "time_regressor": {
            "type": type(model_service.time_model).__name__,
            "path": str(model_service.time_path),
        },
        "feature_columns": FEATURE_COLUMNS,
        "loaded_at": (
            model_service.loaded_at.isoformat() if model_service.loaded_at else None
        ),
        "server_time": datetime.utcnow().isoformat(),
    }
