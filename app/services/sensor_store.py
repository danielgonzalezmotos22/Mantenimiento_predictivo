"""Acceso al histórico de sensores leyendo el CSV con caché en memoria."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class SensorStore:
    """Carga el CSV una sola vez y permite consultarlo con filtros básicos."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    def load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {self.csv_path}")
        # parse_dates para poder filtrar por rango temporal
        self._df = pd.read_csv(self.csv_path, parse_dates=["timestamp"])

    @property
    def is_ready(self) -> bool:
        return self._df is not None

    @property
    def total_rows(self) -> int:
        return 0 if self._df is None else len(self._df)

    def query(
        self,
        reactor_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Devuelve un slice del histórico, opcionalmente filtrado por reactor."""
        if self._df is None:
            raise RuntimeError("El histórico no está cargado.")

        df = self._df
        if reactor_id is not None:
            df = df[df["reactor_id"] == reactor_id]

        sliced = df.iloc[offset : offset + limit].copy()
        # NaN -> None para que sea JSON-serializable
        sliced = sliced.where(pd.notnull(sliced), None)
        # timestamp a string ISO
        if "timestamp" in sliced.columns:
            sliced["timestamp"] = sliced["timestamp"].astype(str)
        return sliced.to_dict(orient="records")

    def reactors(self) -> list[str]:
        if self._df is None:
            raise RuntimeError("El histórico no está cargado.")
        return sorted(self._df["reactor_id"].dropna().unique().tolist())
