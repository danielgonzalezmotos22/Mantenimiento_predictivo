"""Cliente para Groq Cloud que genera informes técnicos en lenguaje natural."""
from __future__ import annotations

import logging
from typing import Any

from app.schemas.prediction import PredictionResponse

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_ES = (
    "Eres un ingeniero de mantenimiento industrial experto en procesos químicos. "
    "Recibes la predicción de un sistema de monitorización de un reactor y debes "
    "redactar un informe técnico breve, profesional y accionable. "
    "Estructura: 1) diagnóstico del estado, 2) causas probables del fallo predicho, "
    "3) acciones de mantenimiento recomendadas con prioridad, 4) urgencia según el "
    "tiempo estimado hasta el fallo. Sé conciso (máx. 250 palabras), técnico y claro. "
    "No inventes datos que no aparezcan en la predicción."
)

SYSTEM_PROMPT_EN = (
    "You are a senior industrial maintenance engineer specialized in chemical "
    "processes. You receive a prediction from a reactor monitoring system and must "
    "write a brief, professional, actionable technical report. "
    "Structure: 1) state diagnosis, 2) probable causes of the predicted fault, "
    "3) recommended maintenance actions with priority, 4) urgency based on the "
    "estimated time to fault. Be concise (max. 250 words), technical, and clear. "
    "Do not invent data not present in the prediction."
)


def _build_user_prompt(prediction: PredictionResponse, language: str) -> str:
    severity = prediction.severity
    severity_line = (
        f"- Severidad (modelo NLP): {severity.label} (confianza {severity.score:.0%})"
        if severity is not None
        else "- Severidad (modelo NLP): no disponible"
    )

    probs_lines = "\n".join(
        f"  · Clase {p.fault_type}: {p.probability:.1%}"
        for p in prediction.fault_probabilities
    )

    intro = "Datos de la predicción:" if language == "es" else "Prediction data:"

    return (
        f"{intro}\n"
        f"- Tipo de fallo: {prediction.fault_type} ({prediction.fault_type_label})\n"
        f"- Probabilidad principal: {prediction.fault_probability:.1%}\n"
        f"- Tiempo estimado hasta fallo: {prediction.time_to_fault_min:.1f} min\n"
        f"{severity_line}\n"
        f"- Distribución de probabilidades:\n{probs_lines}\n"
        f"- Resumen sensorial: {prediction.summary}"
    )


class LLMClient:
    """Wrapper sobre el SDK oficial de Groq."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any | None = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY no configurada en el entorno.")
        # Import perezoso para que la app arranque sin la dependencia presente.
        from groq import Groq  # type: ignore

        self._client = Groq(api_key=self.api_key)

    def generate_report(self, prediction: PredictionResponse, language: str = "es") -> str:
        self._ensure_client()
        assert self._client is not None
        system_prompt = SYSTEM_PROMPT_ES if language == "es" else SYSTEM_PROMPT_EN
        user_prompt = _build_user_prompt(prediction, language)

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return completion.choices[0].message.content or ""
