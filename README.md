# 1. Definición del Proyecto
# 1.1 Propósito del sistema Propósito del sistema
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

# 1.2 Dataset elegido
Dataset utilizado:
Chemical Process Monitoring Time-Series Dataset (Kaggle)

Fuente: https://www.kaggle.com/datasets/rohit8527kmr7518/chemical-process-monitoring-time-series-dataset/data

Tipo de problema
Clasificación → Predicción de tipo de fallo (fault_type)

Regresión → Predicción de tiempo hasta fallo (time_to_fault_min)

Justificación del dataset
Es adecuado porque:

Permite entrenar un modelo propio de clasificación y regresión.
Tiene múltiples variables numéricas reales (temperatura, presión, vibración, etc.).
Permite aplicar modelos preentrenados para análisis semántico o explicación.
Facilita la incorporación de generación de lenguaje natural para explicar resultados técnicos.

# 1.3 Tipo de aplicación
Tipo de servicio
Servicio predictivo + Asistente inteligente basado en datos industriales

Ejemplo completo de uso real
Quién hace la petición:
Un sistema de monitorización industrial o un técnico de mantenimiento.

Endpoints utilizados:
POST /predict

Qué envía:
Datos actuales del sensor (temperatura, presión, vibración, etc.)

Qué devuelve el sistema:

Tipo de fallo predicho

Probabilidad asociada

Tiempo estimado hasta fallo

Informe técnico generado automáticamente (opcional)

GET /sensores
Recibe el usuario la lectura del histórico de los sensores.


Cómo se combinan los modelos
Modelo propio Predice el fallo y el tiempo restante.
 Modelo Hugging Face Analiza patrones o ayuda a clasificar estados complejos.
IA Generativa Genera un informe técnico explicativo en lenguaje natural.

# 2 Arquitectura y Encaje de las Piezas
## 2.1 Modelo propio de Machine Learning

Machine Learning Targets

### Classification
 fault_type — multi-class fault diagnosis (0–4)

### Regression
 efficiency_loss_pct — estimated production efficiency degradation
 
### Forecasting / Survival Analysis
 time_to_fault_min — remaining time before fault onset

Integración en API:
El modelo se carga al iniciar el servidor y se ejecuta cuando el usuario llama al endpoint /predict.

# 2.2 Modelo preentrenado de Hugging Face
Tipo de modelo:
Modelo de clasificación

Valor añadido:

Mejora la interpretación del estado del sistema.

Permite enriquecer la información obtenida del modelo propio.

Integración:

Se ejecuta después de la predicción principal para complementar el resultado antes de generar el informe final.

# 2.3 Modelo de IA Generativa
Tipo de respuesta:

Informe técnico automático

Recomendación de mantenimiento

Explicación del fallo detectado


# 2.4 Exposición mediante API REST
Framework:
FastAPI

Endpoints principales (mínimo 4)
/predict
Predice tipo de fallo y tiempo restante.

GET /sensores
Recibe el usuario la lectura del histórico de los sensores.
Entrada: None
Salida: Histórico de los sensores.

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



# 3. Tests

## Tests de datos
test de archivo que existe:
    compruebe que el archivo existe 

test de dataset:
    comprueba que el dataset no esta vacio


test de implementacion de datos:
    este test se encarga de verificar que los datos se han implementado correctamente.

test de rangos variable:
    comprueba que las variables se encuentran en el rango correcto

test de columnas:
    comprueba que tiene las columnas esperadas.

test de valor nulo:
    comprueba que no haya valores nulos

test de columnas vacias:
    comprueba que no haya columnas vacias

## Tests de API
tests de los endpoints:
    comprueba los endpoint
Respuesta HTTP adecuada (400, 422, 500).

## Mapeo de fallos

| Código | Descripción                              |
|--------|------------------------------------------|
| 0      | Operación normal                         |
| 1      | Degradación del sistema de refrigeración |
| 2      | Obstrucción del caudal de alimentación   |
| 3      | Deriva de sensor                         |
| 4      | Desgaste mecánico                        |

---