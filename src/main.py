from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

app = FastAPI (

    title       = " API de Mantenimiento Predictivo Industrial ",
    description = " API para predecir fallos y estimar en cuánto tiempo puede pasar.",
    version     = " 3.0.0 "
)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

BASE_DIR = Path(__file__).resolve().parent # David Que no se te olvide aqui pillamos la carpeta donde esta este archivo


                                           # cargamos El dichoso modelo

try:

    modelo_fallo = joblib.load(BASE_DIR / "fault_classification_model.pkl")

except Exception as e:

    modelo_fallo = None
    print(f"Error cargando el modelo de fallos: {e}")

try:

    modelo_tiempo = joblib.load(BASE_DIR / "time_to_fault_model.pkl")

except Exception as e:

    modelo_tiempo = None
    print(f"Error cargando el modelo de tiempo: {e}")

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


                                                # Esto es lo que tiene que mandar la persona a /predict
class DatosEntrada(BaseModel):

    operating_regime: str
    reactor_id: str
    ambient_temp_effect: float
    reactor_temp: float
    reactor_pressure: float
    feed_flow_rate: float
    coolant_flow_rate: float
    agitator_speed_rpm: float
    reaction_rate: float
    conversion_rate: float
    selectivity: float
    yield_pct: float
    vibration_rms: float
    motor_current: float
    power_consumption_kw: float
    temp_setpoint: float
    pressure_setpoint: float
    efficiency_loss_pct: float


class DatosReporte(BaseModel):

    fault_type: int
    probability: float
    time_to_fault_min: float


def sacar_nivel_riesgo(probabilidad):

    if probabilidad >= 0.85:

        return "alto"

    elif probabilidad >= 0.60:

        return "medio"

    else:

        return "bajo"


def sacar_mensaje(probabilidad, tiempo):

    if probabilidad >= 0.85 or tiempo <= 30:

        return "Hay bastante riesgo. Yo revisaria el equipo cuanto antes."

    elif probabilidad >= 0.60 or tiempo <= 120:

        return "Hay algo de riesgo. Conviene preparar una revision preventiva pronto."

    else:

        return "No parece urgente ahora mismo, pero mejor seguir vigilando los datos."


tipos_fallo = {
    0: "sin fallo",
    1: "fallo de presion",
    2: "fallo de temperatura",
    3: "fallo mecanico"
}

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

@app.get("/")

def inicio():

    return {

        "Mensaje": "API funcionando bien no te preocupes",
        "Estado_modelo_fallos": modelo_fallo is not None,
        "Estado_modelo_tiempo": modelo_tiempo is not None
    }

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

@app.get("/health")

def health():

    return {

        "status": "Todo ok",
        "mensaje": "Todo bien por aqui"
    }

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

@app.get("/model-info")

def model_info():

    return {

        "modelo_fallos_cargado": modelo_fallo is not None,
        "modelo_tiempo_cargado": modelo_tiempo is not None,
        "ruta_modelos": str(BASE_DIR),
        "numero_features_esperadas": 18
    }

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

@app.post("/predict")

def predict(datos: DatosEntrada):

    if modelo_fallo is None or modelo_tiempo is None:

        raise HTTPException(

            status_code = 500,
            detail = "No se han cargado bien los modelos."
        )


    fila = {

        "operating_regime"    : datos.operating_regime,
        "reactor_id"          : datos.reactor_id,
        "ambient_temp_effect" : datos.ambient_temp_effect,
        "reactor_temp"        : datos.reactor_temp,
        "reactor_pressure"    : datos.reactor_pressure,
        "feed_flow_rate"      : datos.feed_flow_rate,
        "coolant_flow_rate"   : datos.coolant_flow_rate,
        "agitator_speed_rpm"  : datos.agitator_speed_rpm,
        "reaction_rate"       : datos.reaction_rate,
        "conversion_rate"     : datos.conversion_rate,
        "selectivity"         : datos.selectivity,
        "yield_pct"           : datos.yield_pct,
        "vibration_rms"       : datos.vibration_rms,
        "motor_current"       : datos.motor_current,
        "power_consumption_kw": datos.power_consumption_kw,
        "temp_setpoint"       : datos.temp_setpoint,
        "pressure_setpoint"   : datos.pressure_setpoint,
        "efficiency_loss_pct" : datos.efficiency_loss_pct
    }

    df = pd.DataFrame([fila])

    try:

        pred_fallo = modelo_fallo.predict(df)[0]
        pred_tiempo = modelo_tiempo.predict(df)[0]

        if hasattr(modelo_fallo, "predict_proba"):

            probabilidades = modelo_fallo.predict_proba(df)[0]
            probabilidad = float(max(probabilidades))

        else:

            probabilidad = 1.0

        nivel_riesgo = sacar_nivel_riesgo(probabilidad)
        mensaje = sacar_mensaje(probabilidad, float(pred_tiempo))
       
        return {

            "resultado": {

                "tipo_de_fallo": int(pred_fallo),
                "tipo_de_fallo": int(pred_fallo),
                "descripcion_fallo": tipos_fallo.get(int(pred_fallo), "desconocido"),
                "probabilidad": round(probabilidad, 4),
                "tiempo_estimado_hasta_fallo_min": round(float(pred_tiempo), 0)
            },

            "interpretacion": {

                "nivel_de_riesgo": nivel_riesgo,
                "mensaje": mensaje
            }
        }

    except Exception as e:

        raise HTTPException(

            status_code = 500,
            detail = f"Ha fallado la prediccion: {str(e)}"
        )

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

@app.post("/generate-report")

def generate_report(datos: DatosReporte):

    nivel_riesgo = sacar_nivel_riesgo(datos.probability)
    mensaje      = sacar_mensaje(datos.probability, datos.time_to_fault_min)

    texto = (

        f"Se ha detectado un posible fallo de tipo {datos.fault_type}. "
        f"La probabilidad estimada es de           {datos.probability * 100:.2f}%. "
        f"El tiempo estimado hasta el fallo es de  {datos.time_to_fault_min:.2f} minutos. "
        f"El nivel de riesgo se considera          {nivel_riesgo}. "
        f"{mensaje}"
    )

    return {

        "report": texto
    }

