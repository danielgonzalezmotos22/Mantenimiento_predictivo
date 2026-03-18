from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

# Simulación de histórico de sensores
sensor_history = []

# Simulación de información del modelo
model_info = {
    "model_type": "RandomForest",
    "trained_on": "2026-02-10",
    "accuracy": 0.92,
    "f1_score": 0.89
}

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API funcionando"}


# Health Check

@app.get("/health")
def health():
    return {
        "status": "running",
        "timestamp": datetime.now()
    }


# Información del modelo
@app.get("/model-info")
def get_model_info():
    return model_info



# Histórico de sensores
@app.get("/sensores")
def get_sensores():
    return {
        "sensor_history": sensor_history
    }



# Predicción
@app.post("/predict")
def predict(sensor_data: dict):

    temperature = sensor_data.get("temperature")
    pressure = sensor_data.get("pressure")
    vibration = sensor_data.get("vibration")

    # Guardamos lectura en histórico
    sensor_history.append(sensor_data)

    # Simulación de predicción
    if vibration > 0.05:
        failure = "Bearing Failure"
        probability = 0.87
        remaining_time = "48 hours"
    elif temperature > 30:
        failure = "Overheating"
        probability = 0.75
        remaining_time = "72 hours"
    else:
        failure = "Normal Operation"
        probability = 0.95
        remaining_time = "No failure expected"

    return {
        "prediction": failure,
        "probability": probability,
        "remaining_time": remaining_time
    }



# Generación de informe
@app.post("/generate-report")
def generate_report(result: dict):

    prediction = result.get("prediction")
    probability = result.get("probability")
    remaining_time = result.get("remaining_time")

    report = f"""
    Technical Maintenance Report

    The predictive model has detected a potential issue classified as: {prediction}.
    The estimated probability of occurrence is {probability*100:.1f}%.
    The estimated remaining operational time before failure is approximately {remaining_time}.

    Preventive maintenance is recommended to avoid unexpected downtime.
    """

    return {
        "report": report
    }