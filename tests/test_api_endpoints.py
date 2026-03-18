import sys # Permite interactuar con el sistema rutas o entorno
from pathlib import Path # Es para maneja rutas de archivos de forma limpia

sys.path.append(str(Path(__file__).resolve().parents[1])) # Añade la carpeta del proyecto para poder encontrar los módulos
                                                          # parents[1])) sube una carpeta
from fastapi.testclient import TestClient                 # simula peticiones HTTP como en las pruebas de Postman pero en código

from src.main import app

cliente = TestClient(app)                                 # Inicializacion de objeto


def test_health_devuelve_200():                           # Función de test pytest
    respuesta = cliente.get("/health")                    # Llamada HTTP
    assert respuesta.status_code == 200                   # Aserción test valida


def test_model_info_devuelve_200():
    respuesta = cliente.get("/model-info")
    assert respuesta.status_code == 200


def test_sensores_devuelve_200():
    respuesta = cliente.get("/sensores")
    assert respuesta.status_code == 200


def test_predict_devuelve_200_con_datos_validos():
    datos = {                                           # Datos es un diccionario Python convierte ese diccionario a JSON automáticamente

        "reactor_temp":       182.5,
        "reactor_pressure":   15.7,
        "feed_flow_rate":     100.0,
        "coolant_flow_rate":  80.2,
        "agitator_speed_rpm": 300.5

    }

    respuesta = cliente.post("/predict", json=datos)    # Convierte la respuesta en diccionario
    assert respuesta.status_code == 200


def test_predict_devuelve_campos_esperados():
    datos = {

        "reactor_temp":       182.5,
        "reactor_pressure":   15.7,
        "feed_flow_rate":     100.0,
        "coolant_flow_rate":  80.2,
        "agitator_speed_rpm": 300.5
    }

    respuesta = cliente.post("/predict", json=datos)
    resultado = respuesta.json()

    assert "fault_type" in resultado
    assert "probability" in resultado
    assert "time_to_fault_min" in resultado


def test_predict_devuelve_422_si_tipo_incorrecto():
    datos = {

        "reactor_temp": "hola",
        "reactor_pressure":   15.7,
        "feed_flow_rate":     100.0,
        "coolant_flow_rate":  80.2,
        "agitator_speed_rpm": 300.5
    }

    respuesta = cliente.post("/predict", json=datos)
    assert respuesta.status_code == 422


def test_predict_devuelve_422_si_faltan_campos():    # Test negativo
    datos = {
        "reactor_temp": 182.5
    }

    respuesta = cliente.post("/predict", json=datos) # Simula generación de informe
    assert respuesta.status_code == 422


def test_generate_report_devuelve_200():
    datos = {

        "fault_type":        2,
        "probability":       0.78,
        "time_to_fault_min": 95.0
    }

    respuesta = cliente.post("/generate-report", json=datos)
    assert respuesta.status_code == 200
    assert "report" in respuesta.json()


def test_generate_report_devuelve_422_si_faltan_datos():
    datos = {
        "fault_type": 2
    }

    respuesta = cliente.post("/generate-report", json=datos)
    assert respuesta.status_code == 422