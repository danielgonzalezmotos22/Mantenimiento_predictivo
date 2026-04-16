"""POST /generate-report — informe técnico en lenguaje natural (Groq)."""
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.routers.deps import get_llm_client
from app.schemas.report import ReportRequest, ReportResponse
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["informes"])


@router.post("/generate-report", response_model=ReportResponse)
def generate_report(
    payload: ReportRequest,
    llm: LLMClient = Depends(get_llm_client),
) -> ReportResponse:
    try:
        text = llm.generate_report(payload.prediction, language=payload.language)
    except RuntimeError as exc:
        # GROQ_API_KEY ausente u otro error de configuración → 503
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Error llamando a Groq")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error en proveedor LLM: {exc}",
        )

    return ReportResponse(report=text, model=llm.model, language=payload.language)
