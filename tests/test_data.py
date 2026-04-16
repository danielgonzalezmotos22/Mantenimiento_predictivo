"""Tests sobre el dataset (existencia, columnas, rangos, integridad).

Cubren los tests pedidos en el README:
- archivo existe
- dataset no vacío
- implementación correcta de datos
- rangos de variables
- columnas esperadas
- valores nulos
- columnas vacías
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

EXPECTED_COLUMNS = {
    "timestamp",
    "operating_regime",
    "reactor_id",
    "ambient_temp_effect",
    "reactor_temp",
    "reactor_pressure",
    "feed_flow_rate",
    "coolant_flow_rate",
    "agitator_speed_rpm",
    "reaction_rate",
    "conversion_rate",
    "selectivity",
    "yield_pct",
    "vibration_rms",
    "motor_current",
    "power_consumption_kw",
    "temp_setpoint",
    "pressure_setpoint",
    "fault_type",
    "efficiency_loss_pct",
    "time_to_fault_min",
}

# Columnas críticas que NO deben tener nulos (las features que recibe el modelo
# pueden tener nulos puntuales por imputación; pero la etiqueta sí debe existir).
NO_NULLS_REQUIRED = {"timestamp", "operating_regime", "reactor_id", "fault_type"}

VALID_FAULT_TYPES = {0, 1, 2, 3, 4}
VALID_REGIMES = {"A", "B", "C"}


def test_file_exists(dataset_path: Path) -> None:
    """El CSV existe en la ruta esperada."""
    assert dataset_path.exists(), f"No existe el dataset en {dataset_path}"


def test_dataset_not_empty(dataset_df: pd.DataFrame) -> None:
    """El dataset tiene al menos una fila."""
    assert len(dataset_df) > 0


def test_dataset_loaded_correctly(dataset_df: pd.DataFrame) -> None:
    """La carga produce un DataFrame con dtypes razonables."""
    assert isinstance(dataset_df, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(dataset_df["timestamp"])
    assert pd.api.types.is_numeric_dtype(dataset_df["fault_type"])


def test_expected_columns_present(dataset_df: pd.DataFrame) -> None:
    """Todas las columnas esperadas están en el dataset."""
    missing = EXPECTED_COLUMNS - set(dataset_df.columns)
    assert not missing, f"Faltan columnas: {missing}"


def test_no_empty_columns(dataset_df: pd.DataFrame) -> None:
    """Ninguna columna está completamente vacía."""
    fully_empty = [c for c in dataset_df.columns if dataset_df[c].isna().all()]
    assert not fully_empty, f"Columnas vacías: {fully_empty}"


@pytest.mark.parametrize("column", sorted(NO_NULLS_REQUIRED))
def test_no_nulls_in_critical_columns(dataset_df: pd.DataFrame, column: str) -> None:
    """Las columnas clave no contienen nulos."""
    assert dataset_df[column].notna().all(), f"Hay nulos en {column}"


def test_fault_type_in_valid_range(dataset_df: pd.DataFrame) -> None:
    """fault_type solo toma los valores 0-4."""
    unique = set(dataset_df["fault_type"].dropna().astype(int).unique())
    assert unique.issubset(VALID_FAULT_TYPES), f"Valores inesperados: {unique}"


def test_operating_regime_valid(dataset_df: pd.DataFrame) -> None:
    """operating_regime es A, B o C."""
    unique = set(dataset_df["operating_regime"].dropna().unique())
    assert unique.issubset(VALID_REGIMES), f"Regímenes inesperados: {unique}"


def test_reactor_temp_physical_range(dataset_df: pd.DataFrame) -> None:
    """La temperatura del reactor cae dentro de un rango físico razonable."""
    temp = dataset_df["reactor_temp"].dropna()
    assert temp.min() >= 0, "Temperatura negativa detectada"
    assert temp.max() <= 500, "Temperatura excesiva detectada"


def test_pressure_in_range(dataset_df: pd.DataFrame) -> None:
    pressure = dataset_df["reactor_pressure"].dropna()
    assert pressure.min() >= 0
    assert pressure.max() <= 100


def test_efficiency_loss_in_range(dataset_df: pd.DataFrame) -> None:
    eff = dataset_df["efficiency_loss_pct"].dropna()
    assert eff.min() >= 0
    assert eff.max() <= 100


def test_time_to_fault_non_negative(dataset_df: pd.DataFrame) -> None:
    ttf = dataset_df["time_to_fault_min"].dropna()
    assert (ttf >= 0).all()
