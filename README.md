# Definición del Proyecto
# Propósito del sistema
## ¿Qué problema resuelve?
El sistema resuelve el problema de detección temprana de fallos en procesos industriales.

En entornos industriales, una avería inesperada puede generar:

Paradas de producción

Pérdidas económicas

Riesgos operativos

Costes elevados de reparación

El sistema permite:

Detectar posibles fallos antes de que ocurran

Estimar el tiempo restante hasta una avería

Generar explicaciones automáticas del estado del sistema

### ¿Para quién está pensado?
Está pensado para:

Ingenieros de mantenimiento

Técnicos industriales

Responsables de planta

Empresas que trabajen con maquinaria continua

### ¿Qué valor aporta frente a analizar un dataset en un notebook?
Un notebook solo permite análisis puntual.

Este sistema aporta:

Predicción en tiempo real mediante API REST

Integración en sistemas industriales

Automatización del diagnóstico

Generación automática de informes técnicos en lenguaje natural

Uso combinado de modelos propios + modelos preentrenados + IA generativa

Es decir, se transforma el análisis en un servicio inteligente listo para producción.

# Dataset elegido
Dataset utilizado:
Chemical Process Monitoring Time-Series Dataset (Kaggle)

Fuente: Dataset público de simulación industrial con sensores de reactores químicos.

Tipo de problema
Clasificación → Predicción de tipo de fallo (fault_type)

Regresión → Predicción de tiempo hasta fallo (time_to_fault_min)

Justificación del dataset
Es adecuado porque:

Permite entrenar un modelo propio de clasificación y regresión.
Tiene múltiples variables numéricas reales (temperatura, presión, vibración, etc.).
Permite aplicar modelos preentrenados para análisis semántico o explicación.
Facilita la incorporación de generación de lenguaje natural para explicar resultados técnicos.

# Tipo de aplicación
Tipo de servicio
Servicio predictivo + Asistente inteligente basado en datos industriales

Ejemplo completo de uso real
Quién hace la petición:
Un sistema de monitorización industrial o un técnico de mantenimiento.

Endpoint utilizado:
POST /predict

Qué envía:
Datos actuales del sensor (temperatura, presión, vibración, etc.)

Qué devuelve el sistema:

Tipo de fallo predicho

Probabilidad asociada

Tiempo estimado hasta fallo

Informe técnico generado automáticamente

Cómo se combinan los modelos
Modelo propio Predice el fallo y el tiempo restante.
 Modelo Hugging Face Analiza patrones o ayuda a clasificar estados complejos.
IA Generativa Genera un informe técnico explicativo en lenguaje natural.

# Arquitectura y Encaje de las Piezas
# Modelo propio de Machine Learning
Tipo:

Clasificación multiclase

Regresión

Variable objetivo:

fault_type

time_to_fault_min

Integración en API:
El modelo se carga al iniciar el servidor y se ejecuta cuando el usuario llama al endpoint /predict.

# Modelo preentrenado de Hugging Face
Tipo de modelo:
Modelo de clasificación o análisis contextual (por ejemplo, transformer para análisis semántico).

Valor añadido:

Mejora la interpretación del estado del sistema.

Permite enriquecer la información obtenida del modelo propio.

Integración:

Se ejecuta después de la predicción principal para complementar el resultado antes de generar el informe final.

# Modelo de IA Generativa
Tipo de respuesta:

Informe técnico automático

Recomendación de mantenimiento

Explicación del fallo detectado

Control de calidad:

Plantillas estructuradas

Límites de longitud

Validación de coherencia

Uso de prompts controlados

# Exposición mediante API REST
Framework:
FastAPI

Endpoints principales (mínimo 4)
/predict
Predice tipo de fallo y tiempo restante.

Entrada: JSON con datos del sensor.
Salida: Predicción + probabilidad + tiempo estimado.

/generate-report
Genera informe técnico en lenguaje natural a partir de una predicción.

Entrada: Resultado del modelo.
Salida: Texto generado automáticamente.

/health
Verifica que la API está activa.

Entrada: ninguna.
Salida: Estado del servidor.

/model-info
Devuelve información sobre el modelo cargado.

Entrada: ninguna.
Salida: Tipo de modelo, fecha de entrenamiento, métricas.

Manejo de errores
Validación de tipos de datos.

Control de valores fuera de rango.

Respuesta HTTP adecuada (400, 422, 500).