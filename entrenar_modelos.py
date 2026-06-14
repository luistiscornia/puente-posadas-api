import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

# Crea carpeta para guardar los modelos si no existe
os.makedirs("saved_models", exist_ok=True)

# Configura una semilla para que los datos sean reproducibles
np.random.seed(42)
n_muestras = 2000

print("Generando dataset del Puente Posadas-Encarnacion...")

# --- MODELO 1: CLASIFICACION (Congestion: 0 = Fluido, 1 = Colapsado) ---
hora = np.random.randint(0, 24, n_muestras)
dia_semana = np.random.randint(0, 7, n_muestras)
clima_lluvia = np.random.randint(0, 2, n_muestras)
cabinas_abiertas = np.random.randint(1, 10, n_muestras)

# Logica para generar la congestion real del dataset
# Horas pico (7-9, 12-14, 17-19), fines de semana, lluvia y pocas cabinas aumentan la probabilidad
score_congestion = (
    np.isin(hora, [7, 8, 9, 12, 13, 14, 17, 18, 19]).astype(int) * 4 +
    np.isin(dia_semana, [5, 6]).astype(int) * 2 +
    clima_lluvia * 2 +
    (10 - cabinas_abiertas)
)
congestion = (score_congestion > 8).astype(int)

X_clasif = np.column_stack((hora, dia_semana, clima_lluvia, cabinas_abiertas))

# Entrenar un clasificador real (Random Forest)
print("🤖 Entrenando Modelo de Clasificacion...")
modelo_clasif = RandomForestClassifier(n_estimators=50, random_state=42)
modelo_clasif.fit(X_clasif, congestion)

with open("saved_models/modelo_classification.pkl", "wb") as f:
    pickle.dump(modelo_clasif, f)


# --- MODELO 2: REGRESION (Tiempo de espera en minutos) ---
vehiculos_en_cola = np.random.randint(0, 350, n_muestras)
cabinas_regresion = np.random.randint(1, 10, n_muestras)
es_feriado = np.random.randint(0, 2, n_muestras)

# Logica para calcular el tiempo de espera + un ruido blanco normal para simular la realidad
tiempo_base = (vehiculos_en_cola * 1.8) / cabinas_regresion
tiempo_feriado = es_feriado * 45
ruido_aleatorio = np.random.normal(0, 10, n_muestras)

tiempo_espera = tiempo_base + tiempo_feriado + ruido_aleatorio
tiempo_espera = np.clip(tiempo_espera, 5, 480) # Limitar entre 5 min y 8 horas

X_regres = np.column_stack((vehiculos_en_cola, cabinas_regresion, es_feriado))

# Entrenar un regresor real (Decision Tree)
print("🤖 Entrenando Modelo de Regresion...")
modelo_regres = DecisionTreeRegressor(max_depth=10, random_state=42)
modelo_regres.fit(X_regres, tiempo_espera)

with open("saved_models/modelo_regresion.pkl", "wb") as f:
    pickle.dump(modelo_regres, f)

print("✅ ¡Modelos reales entrenados y guardados en saved_models/!")