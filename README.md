# Sistema de Detección Temprana de Fallos en Procesos Químicos

API REST construida con **FastAPI** que combina:

1. **Modelo propio** (RandomForest) — predice `fault_type` (clasificación 0–4) y `time_to_fault_min` (regresión).
2. **Modelo Hugging Face** (`facebook/bart-large-mnli`, zero-shot) — clasifica la severidad del estado del sistema.
3. **LLM de Groq Cloud** (`llama-3.3-70b-versatile` por defecto) — genera informes técnicos en lenguaje natural.

Frontend en **Streamlit** para interacción manual.

---

## Estructura del proyecto

```
proyecto/
├── app/
│   ├── main.py                 # Punto de entrada FastAPI + lifespan
│   ├── config.py               # Settings desde .env
│   ├── routers/                # Un archivo por endpoint
│   ├── schemas/                # Modelos Pydantic
│   └── services/               # Lógica de modelos y LLM
├── tests/                      # pytest (datos + API)
├── frontend/
│   └── streamlit_app.py
├── models/                     # .pkl entrenados (no incluidos)
├── data/                       # CSV (no incluido)
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # rellenar con tu GROQ_API_KEY
```

Coloca los modelos entrenados en `models/`:

- `models/fault_classifier.pkl`
- `models/time_to_fault_regressor.pkl`

Y el dataset en `data/chemical_process_timeseries.csv`.

> **Nota:** los `.pkl` deben ser pipelines de scikit-learn que hagan internamente el preprocesado (encoding de `operating_regime` y `reactor_id`, escalado, etc.). El servicio les pasa un DataFrame con los nombres de columna originales.

---

## Arrancar la API

```bash
uvicorn app.main:app --reload
```

Documentación interactiva: <http://localhost:8000/docs>

## Arrancar el frontend

En otra terminal, con la API ya corriendo:

```bash
streamlit run frontend/streamlit_app.py
```

---

## Endpoints

| Método | Ruta                  | Descripción                                              |
|--------|-----------------------|----------------------------------------------------------|
| GET    | `/health`             | Estado del servidor y de los recursos cargados           |
| GET    | `/model-info`         | Tipo de modelos, clases, columnas y fecha de carga       |
| POST   | `/predict`            | Predicción de `fault_type` + `time_to_fault_min` + severidad |
| POST   | `/generate-report`    | Informe técnico en lenguaje natural a partir de una predicción |
| GET    | `/sensores`           | Histórico paginado del CSV (filtro por `reactor_id`)     |
| GET    | `/sensores/reactores` | Lista de reactores disponibles                           |

### Ejemplo de `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "operating_regime": "A",
    "reactor_id": "A_R1",
    "ambient_temp_effect": 0.0,
    "reactor_temp": 181.13,
    "reactor_pressure": 15.79,
    "feed_flow_rate": 101.10,
    "coolant_flow_rate": 79.15,
    "agitator_speed_rpm": 305.78,
    "reaction_rate": 0.72,
    "conversion_rate": 99.15,
    "selectivity": 91.92,
    "yield_pct": 82.03,
    "vibration_rms": 1.47,
    "motor_current": 45.88,
    "power_consumption_kw": 41.29,
    "temp_setpoint": 180.0,
    "pressure_setpoint": 12.0
  }'
```

---

## Tests

```bash
pytest
```

- **`tests/test_data.py`** — existencia del CSV, columnas esperadas, rangos físicos, ausencia de nulos en columnas críticas, valores válidos de `fault_type` y `operating_regime`.
- **`tests/test_api.py`** — contratos de los endpoints, códigos HTTP (200, 404, 422), validación Pydantic, pagination y filtros del histórico.

Los tests de la API **no necesitan los `.pkl` reales ni el API key de Groq**: las dependencias se inyectan como mocks vía `app.dependency_overrides`. Los tests de datos sí leen el CSV real (se saltan automáticamente si no está).

---

## Mapeo de fallos

| Código | Descripción                              |
|--------|------------------------------------------|
| 0      | Operación normal                         |
| 1      | Degradación del sistema de refrigeración |
| 2      | Obstrucción del caudal de alimentación   |
| 3      | Deriva de sensor                         |
| 4      | Desgaste mecánico                        |

---

## Notas de diseño

- **Lifespan tolerante a errores**: si faltan los `.pkl` o el CSV, la API arranca igual; los endpoints afectados responden `503` en lugar de impedir el arranque.
- **Modelo HF lazy**: `bart-large-mnli` pesa ~1.6 GB; se descarga la primera vez que se llama a `/predict`. Si quieres una alternativa más ligera, cambia `HF_MODEL_NAME` en `.env` por algo como `valhalla/distilbart-mnli-12-3`.
- **El fallo del HF no rompe la respuesta**: si el clasificador semántico falla (sin red, sin GPU, etc.), `/predict` devuelve la predicción sin el campo `severity`.
- **Inyección de dependencias**: todas las dependencias se obtienen vía `Depends`, lo que permite mockear cada servicio en tests sin tocar la app real.
