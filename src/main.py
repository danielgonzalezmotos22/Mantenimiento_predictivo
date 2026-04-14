from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

app = FastAPI(

    title       = "API de Mantenimiento Predictivo",
    description = "Servicio para predecir fallos industriales y generar informes técnicos. " ,
    version     = "2.0.0 Gran Jefe Cigueño"

)

# Carga de modelos_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


# BASE_DIR = Path(__file__).resolve().parent

# MODELS_DIR = BASE_DIR / "notebooks"

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR

try:

    model_fault = joblib.load(MODELS_DIR / "fault_classification_model.pkl")

except Exception as e:

    model_fault = None
    
    print(f"Error cargando fault_classification_model.pkl: {e}")

try:

    model_time = joblib.load(MODELS_DIR / "time_to_fault_model.pkl")

except Exception as e:

    model_time = None
    print(f"Error cargando time_to_fault_model.pkl: {e}")

try:

    model_time = joblib.load(MODELS_DIR / "time_to_fault_model.pkl")

except Exception as e:

    model_time = None

    print(f"Error cargando time_to_fault_model.pkl: {e}")



# Modelos de entrada/salida_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

class PredictInput(BaseModel):

    reactor_temp         : float
    reactor_pressure     : float
    feed_flow_rate       : float
    coolant_flow_rate    : float
    agitator_speed_rpm   : float
    operating_regime     : str = "normal"
    reactor_id           : str = "R1"
    ambient_temp_effect  : float = 1.2

    reaction_rate: float = 0.87
    conversion_rate      : float = 0.91

    selectivity: float          = 0.88
    yield_pct: float            = 92.4
    vibration_rms: float        = 1.7
    motor_current: float        = 12.5
    power_consumption_kw: float = 45.8
    temp_setpoint: float        = 180.0
    pressure_setpoint: float    = 15.0
    efficiency_loss_pct: float  = 3.2


class ReportInput(BaseModel):

    fault_type: int
    probability: float
    time_to_fault_min: float

# Endpoints_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

@app.get("/health")

def health():

    return { "status" : "ok"}


@app.get("/model-info")

def model_info():

    return {

        "fault_classification_model_loaded" : model_fault is not None,
        "time_to_fault_model_loaded"        : model_time is not None,
        "models_path"                       : str(MODELS_DIR),
        "expected_features"                 : 18,
        "version"                           : "2.1.0"
    }


@app.get("/sensores")
def get_sensores():
    return {
        "sensores": [
            {
                "operating_regime"     : "normal",
                "reactor_id"           : "R1",
                "ambient_temp_effect"  : 1.2,
                "reactor_temp"         : 182.5,
                "reactor_pressure"     : 15.7,
                "feed_flow_rate"       : 100.0,
                "coolant_flow_rate"    : 80.2,
                "agitator_speed_rpm"   : 300.5,
                "reaction_rate"        : 0.87,
                "conversion_rate"      : 0.91,
                "selectivity"          : 0.88,
                "yield_pct"            : 92.4,
                "vibration_rms"        : 1.7,
                "motor_current"        : 12.5,
                "power_consumption_kw" : 45.8,
                "temp_setpoint"        : 180.0,
                "pressure_setpoint"    : 15.0,
                "efficiency_loss_pct"  : 3.2
            }
        ]
    }


@app.post("/predict")
def predict(data: PredictInput):
    if model_fault is None or model_time is None:
        raise HTTPException (
            status_code = 500,
            detail      = "Los modelos no están cargados correctamente."
        )


    input_df = pd.DataFrame([{

        "operating_regime"     : data.operating_regime,
        "reactor_id"           : data.reactor_id,
        "ambient_temp_effect"  : data.ambient_temp_effect,
        "reactor_temp"         : data.reactor_temp,
        "reactor_pressure"     : data.reactor_pressure,
        "feed_flow_rate"       : data.feed_flow_rate,
        "coolant_flow_rate"    : data.coolant_flow_rate,
        "agitator_speed_rpm"   : data.agitator_speed_rpm,
        "reaction_rate"        : data.reaction_rate,
        "conversion_rate"      : data.conversion_rate,
        "selectivity"          : data.selectivity,
        "yield_pct"            : data.yield_pct,
        "vibration_rms"        : data.vibration_rms,
        "motor_current"        : data.motor_current,
        "power_consumption_kw" : data.power_consumption_kw,
        "temp_setpoint"        : data.temp_setpoint,
        "pressure_setpoint"    : data.pressure_setpoint,
        "efficiency_loss_pct"  : data.efficiency_loss_pct
    }])

    try:

        fault_prediction = model_fault.predict(input_df)
        time_prediction  = model_time.predict(input_df)

        if hasattr(model_fault, "predict_proba"):
            
            probas      = model_fault.predict_proba(input_df)
            probability = float(probas.max())
        
        else:
            
            probability = 1.0

        return {
            
            "fault_type"         : int(fault_prediction[0]),
            "probability"        : round(probability, 4),
            "time_to_fault_min"  : float(time_prediction[0])
        }

    except Exception as e:

        raise HTTPException(
            
            status_code = 500,
            detail = f"Error durante la predicción: {str(e)}"
        
        )


@app.post("/generate-report")

def generate_report(data: ReportInput):

    report = (

        f" Se ha detectado un posible fallo de tipo {data.fault_type} "
        f" con una probabilidad de                  {data.probability:.2f}. "
        f" El tiempo estimado hasta el fallo es de  {data.time_to_fault_min:.1f} minutos. "
        f" Se recomienda revisar el estado del reactor y programar una inspección preventiva."
    )

    return {"report": report}
