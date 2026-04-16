"""Configuración global de la aplicación cargada desde variables de entorno."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Modelos
    model_fault_path: Path = Path("models/fault_classifier.pkl")
    model_time_path: Path = Path("models/time_to_fault_regressor.pkl")

    # Dataset
    dataset_path: Path = Path("data/chemical_process_timeseries.csv")

    # Hugging Face
    hf_model_name: str = "facebook/bart-large-mnli"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Cachea la instancia de Settings para no releer .env en cada llamada."""
    return Settings()
