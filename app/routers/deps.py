"""Dependencias inyectables. Centralizadas aquí para que los tests las
puedan sobrescribir con `app.dependency_overrides`.
"""
from fastapi import HTTPException, Request, status

from app.services.hf_model import HFClassifierService
from app.services.llm_client import LLMClient
from app.services.ml_model import ModelService
from app.services.sensor_store import SensorStore


def get_model_service(request: Request) -> ModelService:
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos de ML no disponibles. Revisa los archivos .pkl.",
        )
    return service


def get_sensor_store(request: Request) -> SensorStore:
    store: SensorStore | None = getattr(request.app.state, "sensor_store", None)
    if store is None or not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Histórico de sensores no disponible.",
        )
    return store


def get_hf_service(request: Request) -> HFClassifierService:
    service: HFClassifierService | None = getattr(request.app.state, "hf_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio HF no disponible.",
        )
    return service


def get_llm_client(request: Request) -> LLMClient:
    client: LLMClient | None = getattr(request.app.state, "llm_client", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cliente LLM no configurado.",
        )
    return client
