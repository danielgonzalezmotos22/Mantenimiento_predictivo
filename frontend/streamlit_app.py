"""Frontend en Streamlit para el sistema de detección de fallos.

Permite:
- Introducir lecturas de sensores y obtener una predicción.
- Generar el informe técnico con un clic.
- Explorar el histórico de sensores.
- Consultar el estado del servidor y la información del modelo.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 60

st.set_page_config(
    page_title="Detección de Fallos en Reactores",
    page_icon="⚙️",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Helpers HTTP
# -----------------------------------------------------------------------------

def api_get(path: str, params: dict | None = None) -> Any:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict) -> Any:
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# -----------------------------------------------------------------------------
# Sidebar — estado del servidor
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Reactor Monitor")
    st.caption(f"API: `{API_URL}`")
    try:
        health = api_get("/health")
        st.success("API disponible")
        st.json(health)
    except Exception as exc:
        st.error(f"API no disponible: {exc}")

    page = st.radio(
        "Navegación",
        ["Predicción", "Histórico de sensores", "Información del modelo"],
    )


# -----------------------------------------------------------------------------
# Página: Predicción
# -----------------------------------------------------------------------------

def page_prediction() -> None:
    st.header("Predicción de fallo en reactor")
    st.write(
        "Introduce las lecturas actuales del sensor para obtener una predicción "
        "del tipo de fallo, el tiempo restante hasta avería y un informe técnico "
        "automático."
    )

    with st.form("sensor_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            operating_regime = st.selectbox("Régimen operativo", ["A", "B", "C"])
            reactor_id = st.text_input("ID del reactor", value="A_R1")
            ambient_temp_effect = st.number_input("Efecto temperatura ambiente", value=0.0)
            reactor_temp = st.number_input("Temperatura del reactor (°C)", value=181.13)
            reactor_pressure = st.number_input("Presión del reactor (bar)", value=15.79)
            feed_flow_rate = st.number_input("Caudal de alimentación", value=101.10)

        with col2:
            coolant_flow_rate = st.number_input("Caudal de refrigerante", value=79.15)
            agitator_speed_rpm = st.number_input("Velocidad agitador (RPM)", value=305.78)
            reaction_rate = st.number_input("Tasa de reacción", value=0.72)
            conversion_rate = st.number_input("Tasa de conversión (%)", value=99.15)
            selectivity = st.number_input("Selectividad (%)", value=91.92)
            yield_pct = st.number_input("Rendimiento (%)", value=82.03)

        with col3:
            vibration_rms = st.number_input("Vibración RMS", value=1.47)
            motor_current = st.number_input("Corriente motor (A)", value=45.88)
            power_consumption_kw = st.number_input("Consumo (kW)", value=41.29)
            temp_setpoint = st.number_input("Setpoint temperatura", value=180.0)
            pressure_setpoint = st.number_input("Setpoint presión", value=12.0)
            efficiency_loss_pct = st.number_input("Pérdida de eficiencia (%)", value=0.0, min_value=0.0, max_value=100.0)

        submitted = st.form_submit_button("🔮 Predecir", use_container_width=True)

    if submitted:
        payload = {
            "operating_regime": operating_regime,
            "reactor_id": reactor_id,
            "ambient_temp_effect": ambient_temp_effect,
            "reactor_temp": reactor_temp,
            "reactor_pressure": reactor_pressure,
            "feed_flow_rate": feed_flow_rate,
            "coolant_flow_rate": coolant_flow_rate,
            "agitator_speed_rpm": agitator_speed_rpm,
            "reaction_rate": reaction_rate,
            "conversion_rate": conversion_rate,
            "selectivity": selectivity,
            "yield_pct": yield_pct,
            "vibration_rms": vibration_rms,
            "motor_current": motor_current,
            "power_consumption_kw": power_consumption_kw,
            "temp_setpoint": temp_setpoint,
            "pressure_setpoint": pressure_setpoint,
            "efficiency_loss_pct": efficiency_loss_pct,
        }
        try:
            with st.spinner("Consultando modelo..."):
                pred = api_post("/predict", payload)
            st.session_state["last_prediction"] = pred
        except Exception as exc:
            st.error(f"Error en la predicción: {exc}")
            return

    pred = st.session_state.get("last_prediction")
    if not pred:
        return

    st.subheader("Resultado")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tipo de fallo", pred["fault_type"], pred["fault_type_label"])
    m2.metric("Probabilidad", f"{pred['fault_probability']:.1%}")
    m3.metric("Tiempo hasta fallo", f"{pred['time_to_fault_min']:.1f} min")
    if pred.get("severity"):
        m4.metric("Severidad (NLP)", pred["severity"]["label"], f"{pred['severity']['score']:.0%}")
    else:
        m4.metric("Severidad (NLP)", "N/D")

    st.markdown(f"**Resumen:** {pred['summary']}")

    # Distribución de probabilidades
    probs_df = pd.DataFrame(pred["fault_probabilities"]).set_index("fault_type")
    st.bar_chart(probs_df["probability"])

    # Informe LLM bajo demanda
    if st.button("📝 Generar informe técnico"):
        try:
            with st.spinner("Generando informe con Groq..."):
                report = api_post(
                    "/generate-report",
                    {"prediction": pred, "language": "es"},
                )
            st.success(f"Informe generado con `{report['model']}`")
            st.markdown(report["report"])
        except Exception as exc:
            st.error(f"Error generando informe: {exc}")


# -----------------------------------------------------------------------------
# Página: Histórico
# -----------------------------------------------------------------------------

def page_history() -> None:
    st.header("Histórico de sensores")
    try:
        reactors = api_get("/sensores/reactores")["reactors"]
    except Exception as exc:
        st.error(f"No se pudo cargar la lista de reactores: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    reactor = col1.selectbox("Reactor", ["(todos)"] + reactors)
    limit = col2.number_input("Filas a mostrar", min_value=10, max_value=1000, value=200)
    offset = col3.number_input("Offset", min_value=0, value=0)

    params: dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
    if reactor != "(todos)":
        params["reactor_id"] = reactor

    try:
        data = api_get("/sensores", params=params)
    except Exception as exc:
        st.error(f"Error: {exc}")
        return

    st.caption(f"Total filas en histórico: {data['total']:,}")
    df = pd.DataFrame(data["data"])
    if df.empty:
        st.warning("Sin datos para los filtros seleccionados.")
        return

    st.dataframe(df, use_container_width=True, height=500)

    if "reactor_temp" in df.columns and "timestamp" in df.columns:
        st.line_chart(df.set_index("timestamp")[["reactor_temp"]])


# -----------------------------------------------------------------------------
# Página: Información del modelo
# -----------------------------------------------------------------------------

def page_model_info() -> None:
    st.header("Información del modelo")
    try:
        info = api_get("/model-info")
        st.json(info)
    except Exception as exc:
        st.error(f"Error: {exc}")


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

if page == "Predicción":
    page_prediction()
elif page == "Histórico de sensores":
    page_history()
else:
    page_model_info()
