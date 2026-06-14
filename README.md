# API de Optimizacion de Transito - Puente Internacional Posadas-Encarnacion

Trabajo Integrador Final **Programacion II** (Tecnicatura Universitaria en Ciencia de Datos).

Este proyecto expone una solución de Machine Learning mediante una API REST desarrollada en **FastAPI** y desplegada en **Fly.io**. El objetivo es predecir la congestion vehicular y estimar cuantitativamente los tiempos de espera en el viaducto internacional que conecta Argentina con Paraguay.


# Enlaces del Proyecto

* **URL Publica de la API:** [https://puente-posadas-encarnacion-api.fly.dev](https://puente-posadas-encarnacion-api.fly.dev)

* **Estado de la API:** [https://puente-posadas-encarnacion-api.fly.dev/status](https://puente-posadas-encarnacion-api.fly.dev/status)

* **Endpoint de Informacion:** [https://puente-posadas-encarnacion-api.fly.dev/info](https://puente-posadas-encarnacion-api.fly.dev/info)

* **Predicciones de clasificación:** [https://puente-posadas-encarnacion-api.fly.dev/modelo1](https://puente-posadas-encarnacion-api.fly.dev/modelo1)

* **Predicciones de regresión:** [https://puente-posadas-encarnacion-api.fly.dev/modelo2](https://puente-posadas-encarnacion-api.fly.dev/modelo2)

* **Documentación automática generada por FastAPI:** [https://puente-posadas-encarnacion-api.fly.dev/docs](https://puente-posadas-encarnacion-api.fly.dev/docs)





# Integrantes
* **Luciano Kovacevich** - DNI: 41689310
* **Luis Tiscornia** - DNI: 31759076
* **Bianca Valente** - DNI: 44306880



# Modelos de Machine Learning Integrados
Para resolver el problema del colapso logistico y vial, entrenamos e integramos dos modelos predictivos utilizando **Scikit-Learn**:

# Modelo 1: Clasificador de Congestion Fronteriza (`/modelo1`)
* **Tipo de Modelo:** Clasificacion Binaria (`RandomForestClassifier`).
* **Objetivo:** Determinar si las condiciones operativas y climaticas provocaran un colapso en el viaducto.
* **Entradas (Features):** * `hora`: Hora del dia (0 a 23).
  * `dia_semana`: Dia de la semana (0: Lunes a 6: Domingo).
  * `clima_lluvia`: Estado climatico (1: Si llueve, 0: No llueve).
  * `cabinas_abiertas`: Cantidad de casillas migratorias operativas en ese momento (1 a 14).
* **Salida (Target):** `0` para Transito Fluido / Normal, `1` para Puente Congestionado / Demora Alta.

# Modelo 2: Predictor de Tiempo de Espera (`/modelo2`)
* **Tipo de Modelo:** Regresion Cuantitativa (`DecisionTreeRegressor`).
* **Objetivo:** Estimar la cantidad exacta de minutos que demorara un vehiculo en la fila antes de cruzar.
* **Entradas (Features):**
  * `vehiculos_en_cola`: Cantidad estimada de autos esperando en la fila.
  * `cabinas_abiertas`: Cantidad de casillas migratorias operativas (1 a 14).
  * `es_feriado`: Indicador de dia no laborable o fin de semana largo (1: Si, 0: No).
* **Salida (Target):** Valor numerico continuo que representa el tiempo estimado de espera en minutos.


#  Tecnologias Utilizadas
* **Backend:** Python 3.11 & FastAPI
* **Machine Learning:** Scikit-Learn, Numpy & Pickle
* **Infraestructura & Despliegue:** Docker, Container Registry y Fly.io

# Para probar la API:
1. Ingresar a la documentacion automatica: [https://puente-posadas-encarnacion-api.fly.dev/docs](https://puente-posadas-encarnacion-api.fly.dev/docs)
2. Desplegar cualquiera de los dos endpoints de prediccion (`POST /modelo1` o `POST /modelo2`).
3. Hacer clic en **"Try it out"**, ajustar los valores numericos del JSON de ejemplo si lo desea, y presionar **"Execute"** para recibir la respuesta procesada por el motor