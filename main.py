import os
import json
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

# Inicializar FastAPI
app = FastAPI(
    title="API Puente Posadas-Encarnacion",
    description="API REST para la prediccion de congestion y tiempos de espera en el puente internacional Posadas (Arg) - Encarnacion (Pry).",
    version="1.1"
)


# Carga de Modelos

PATH_CLASIF = "saved_models/modelo_clasificacion.pkl"
PATH_REGRES = "saved_models/modelo_regresion.pkl"

modelo_clasificacion = None
modelo_regresion = None

if os.path.exists(PATH_CLASIF):
    with open(PATH_CLASIF, "rb") as f:
        modelo_clasificacion = pickle.load(f)

if os.path.exists(PATH_REGRES):
    with open(PATH_REGRES, "rb") as f:
        modelo_regresion = pickle.load(f)



# Esquemas de Validacion de Datos (Pydantic)


class ClasificacionInput(BaseModel):
    hora: int = Field(..., ge=0, le=23, description="Hora del dia (0-23)", example=14)
    dia_semana: int = Field(..., ge=0, le=6, description="Dia de la semana (0:Lunes, 6:Domingo)", example=5)
    clima_lluvia: int = Field(..., ge=0, le=1, description="Llueve? (1:Si, 0:No)", example=0)
    cabinas_abiertas: int = Field(..., ge=1, le=14, description="Cabinas migratorias activas", example=4)


class RegresionInput(BaseModel):
    vehiculos_en_cola: int = Field(..., ge=0, description="Cantidad de autos en la fila", example=120)
    cabinas_abiertas: int = Field(..., ge=1, le=14, description="Cabinas migratorias activas", example=3)
    es_feriado: int = Field(..., ge=0, le=1, description="Es feriado o fin de semana largo? (1:Si, 0:No)", example=1)



# Endpoints de la API


@app.get("/")
def root():
    content_root = {"mensaje": "API del Puente Posadas-Encarnacion funcionando. Visita /docs para probar."}
    json_formateado = json.dumps(content_root, indent=2)
    return Response(content=json_formateado, media_type="application/json; charset=utf-8")


@app.get("/status")
def status():
    """Verificacion de funcionamiento de la API con formato legible."""
    content_status = {
        "status": "ok",
        "materia": "Programacion II",
        "modelos_cargados": {
            "clasificacion": modelo_clasificacion is not None,
            "regresion": modelo_regresion is not None
        },
        "version": "1.0"
    }
    json_formateado = json.dumps(content_status, indent=2)
    return Response(content=json_formateado, media_type="application/json; charset=utf-8")


@app.get("/info")
def info():
    """Informacion general del TPI e integrantes"""
    data_info = {
        "materia": "Programacion II",
        "proyecto": "API de Optimizacion de Transito - Puente Posadas-Encarnacion",
        "descripcion": "API REST desarrollada con FastAPI que permite predecir la congestion de trafico y estimar los tiempos de espera en el Puente Internacional San Roque Gonzalez de Santa Cruz utilizando modelos entrenados con Scikit-Learn.",

        "integrantes": [
            {
                "Nombre": "Luciano Kovacevich",
                "DNI": "41689310",
                "email": "luchokova.lk@gmail.com"
            },
            {
                "Nombre": "Luis Tiscornia",
                "DNI": "31759076", 
                "email": "luistiscornia@gmail.com"
            },
            {
                "Nombre": "Bianca Valente",
                "DNI": "44306880",
                "email": "valentebianca22@gmail.com"
            }
        ],

        "modelo_clasificacion": {
            "Nombre": "Clasificador de Congestion vehicular",
            "Tipo": "Clasificacion",
            "Dataset": "Historial de Transito Puente Posadas-Encarnacion",
            "Descripcion": "Determina si el estado del puente sera 'Congestionado' (1) o 'Fluido' (0).",
            "Endpoint": "/modelo1"
        },

        "modelo_regresion": {
            "Nombre": "Predictor de Tiempo de Espera en Viaducto",
            "Tipo": "Regresion",
            "Dataset": "Metricas de Demoras Registradas por Longitud de Fila",
            "Descripcion": "Estima el tiempo aproximado de espera en minutos para cruzar el centro de fronteras.",
            "Endpoint": "/modelo2"
        },

        "uso_endpoints": {
            "/modelo1": {
                "metodo": "POST",
                "descripcion": "Realiza una prediccion de clasificacion sobre congestion (0: Fluido, 1: Colapsado).",
                "ejemplo_body": {
                    "hora": 18,
                    "dia_semana": 4,
                    "clima_lluvia": 1,
                    "cabinas_abiertas": 3
                }
            },
            "/modelo2": {
                "metodo": "POST",
                "descripcion": "Realiza una prediccion de regresion estimando los minutos de espera.",
                "ejemplo_body": {
                    "vehiculos_en_cola": 85,
                    "cabinas_abiertas": 5,
                    "es_feriado": 0
                }
            }
        }
    }
    json_formateado = json.dumps(data_info, indent=2)
    return Response(content=json_formateado, media_type="application/json; charset=utf-8")



# Datos para accesos desde el navegador


@app.get("/modelo1", include_in_schema=False)
def modelo1_para_nav():
    """Devuelve instrucciones si el usuario intenta entrar a /modelo1 desde el navegador."""
    instrucciones = {
        "mensaje": "Ingresaste a la URL exclusiva del Modelo 1 (Clasificacion).",
        "para_probarlo": "Ingresar a https://puente-posadas-encarnacion-api.fly.dev/docs para ejecutarlo.",
        "ejemplo_json_requerido": {
            "hora": 14,
            "dia_semana": 5,
            "clima_lluvia": 0,
            "cabinas_abiertas": 4
        }
    }
    return Response(content=json.dumps(instrucciones, indent=2), media_type="application/json; charset=utf-8")


@app.get("/modelo2", include_in_schema=False)
def modelo2_para_nav():
    """Devuelve instrucciones si el usuario intenta entrar a /modelo2 desde el navegador."""
    instrucciones = {
        "mensaje": "Ingresaste a la URL exclusiva del Modelo 2 (Regresion).",
        "para_probarlo": "Ingresar a https://puente-posadas-encarnacion-api.fly.dev/docs para ejecutarlo.",
        "ejemplo_json_requerido": {
            "vehiculos_en_cola": 120,
            "cabinas_abiertas": 3,
            "es_feriado": 1
        }
    }
    return Response(content=json.dumps(instrucciones, indent=2), media_type="application/json; charset=utf-8")

@app.post("/modelo1", summary="Prediccion de Congestion Vial (Clasificacion)", description="Recibe variables de contexto (hora, dia, lluvia, casillas operativas) y determina mediante un RandomForest de Scikit-Learn si el puente estara fluido (0) o colapsado (1).")
def predict_modelo1(data: ClasificacionInput):
    """Prediccion de Clasificacion (Congestion)"""
    if modelo_clasificacion is None:
        congestion_simulada = 1 if (data.hora in [7, 8, 13, 18] or data.cabinas_abiertas < 3) else 0
        return {
            "estado": "Modo Simulacion (sin .pkl)",
            "prediccion": congestion_simulada,
            "resultado": "Congestionado / Demora Alta" if congestion_simulada == 1 else "Transito Fluido"
        }
    
    try:
        X = np.array([[data.hora, data.dia_semana, data.clima_lluvia, data.cabinas_abiertas]])
        pred = int(modelo_clasificacion.predict(X)[0])
        respuesta = {"prediccion": pred, "resultado": "Congestionado" if pred == 1 else "Fluido"}
        
        if hasattr(modelo_clasificacion, "predict_proba"):
            proba = modelo_clasificacion.predict_proba(X)[0]
            respuesta["confianza"] = round(float(proba[pred]), 4)
            
        return respuesta
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del modelo: {str(e)}")


@app.post("/modelo2", summary="Estimacion de Tiempo de Espera (Regresion)", description="Recibe la cantidad de autos en fila, casillas abiertas y si es feriado, devolviendo el calculo analitico exacto de minutos de demora estimados utilizando un DecisionTreeRegressor.")
def predict_modelo2(data: RegresionInput):
    """Prediccion de Regresion (tiempo de Espera en minutos)"""
    if modelo_regresion is None:
        tiempo_simulado = round((data.vehiculos_en_cola * 1.6) / data.cabinas_abiertas + (40 if data.es_feriado else 0), 2)
        return {
            "estado": "Modo Simulacion (sin archivo .pkl)",
            "tiempo_espera_estimado_minutos": tiempo_simulado
        }
        
    try:
        X = np.array([[data.vehiculos_en_cola, data.cabinas_abiertas, data.es_feriado]])
        pred = float(modelo_regresion.predict(X)[0])
        return {
            "tiempo_espera_estimado_minutos": round(max(0.0, pred), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del modelo: {str(e)}")