from fastapi import FastAPI    # Librería: crear la API
from pydantic import BaseModel # Librería: valida datos
from typing import List        # Librería: define listas


app = FastAPI(

    title =       "API de Mantenimiento Predictivo",
    description = "Servicio para predecir fallos industriales y generar informes técnicos.",
    version =     "1.0.0 Gran Jefe Cigueño"
)


@app.get   ("/health")
def health ():
    return {"status": "ok"}      # Comprobar si la API está viva


class PredictInput(BaseModel): # Clase de Pydantic
                               # Para definir cómo debe ser el json de entrada
    reactor_temp:       float
    reactor_pressure:   float
    feed_flow_rate:     float
    coolant_flow_rate:  float
    agitator_speed_rpm: float

# Simulación de predicción_ _ _ _ _ _ _ _ _ _ _ _ 

@app.post("/predict")

def predict(data: PredictInput): # Recibe json lo convierte en PredictInput
                                 # Aquí iria el modelo de IA
    fault_type = 1               
    probability = 0.85
    time_to_fault_min = 120.0

    return {

        "fault_type":        fault_type,
        "probability":       probability,
        "time_to_fault_min": time_to_fault_min
    }


@app.get("/model-info")
def model_info():
    return {

        "model_name":            "predictive_model",
        "version":               "1.0",
        "training_date":         "2026-03-18",
        "classification_metric": "accuracy: 0.89",
        "regression_metric":     "mae: 12.4"
    }


@app.get("/sensores")           # Devuelve datos simulados de sensores
def get_sensores():            
    return {

        "sensores": [
            {
                "reactor_temp":     182.5,
                "reactor_pressure": 15.7,
                "feed_flow_rate":   100.0,
                "coolant_flow_rate": 80.2,
                "agitator_speed_rpm": 300.5
            },

            {
                "reactor_temp":       185.1,
                "reactor_pressure":   16.0,
                "feed_flow_rate":     98.4,
                "coolant_flow_rate":  79.8,
                "agitator_speed_rpm": 298.9
            }
        ]
    }


class ReportInput(BaseModel):

    fault_type: int          # Define datos que necesita el informe
    probability: float
    time_to_fault_min: float


@app.post("/generate-report") # Escribe un informe en texto

def generate_report(data: ReportInput):

    report = (                # String que construye el informe con datos

        f"Se ha detectado un posible fallo de tipo {data.fault_type} "
        f"con una probabilidad de                  {data.probability:.2f}. "
        f"El tiempo estimado hasta el fallo es de  {data.time_to_fault_min:.1f} minutos. "
        f"Se recomienda revisar el estado del reactor y programar una inspección preventiva."
    )

    return {"report": report}  # Respuesta JSON