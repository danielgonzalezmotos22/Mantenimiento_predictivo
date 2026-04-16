"""Punto de entrada de la API FastAPI.

Usa `lifespan` para cargar los modelos y el dataset una sola vez al arrancar
y dejarlos disponibles en `app.state`. La carga es tolerante a errores: si
falta un .pkl o el CSV, el servidor arranca igual y los endpoints afectados
responden 503, en lugar de impedir el arranque por completo.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import get_settings
from app.routers import health, model_info, predict, report, sensors
from app.services.hf_model import HFClassifierService
from app.services.llm_client import LLMClient
from app.services.ml_model import ModelService
from app.services.sensor_store import SensorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Modelos propios
    model_service = ModelService(settings.model_fault_path, settings.model_time_path)
    try:
        model_service.load()
        logger.info("Modelos ML cargados.")
    except Exception as exc:
        logger.warning("No se pudieron cargar los modelos ML: %s", exc)
    app.state.model_service = model_service

    # Dataset
    sensor_store = SensorStore(settings.dataset_path)
    try:
        sensor_store.load()
        logger.info("Dataset cargado: %d filas.", sensor_store.total_rows)
    except Exception as exc:
        logger.warning("No se pudo cargar el dataset: %s", exc)
    app.state.sensor_store = sensor_store

    # Hugging Face (lazy: aquí solo se instancia, se descarga al primer uso)
    app.state.hf_service = HFClassifierService(settings.hf_model_name)

    # LLM (Groq)
    app.state.llm_client = LLMClient(settings.groq_api_key, settings.groq_model)

    yield
    logger.info("Apagando servidor.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sistema de Detección Temprana de Fallos en Procesos Químicos",
        description=(
            "API REST que combina un modelo propio de RandomForest, un "
            "clasificador zero-shot de Hugging Face y un LLM (Groq) para "
            "predecir fallos, estimar el tiempo restante y generar informes "
            "técnicos automáticos."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS abierto: ajustar a dominios concretos en producción.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(model_info.router)
    app.include_router(predict.router)
    app.include_router(report.router)
    app.include_router(sensors.router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Redirige a la documentación interactiva."""
        return RedirectResponse(url="/docs")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
