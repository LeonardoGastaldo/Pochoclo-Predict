import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
API_KEY = "TU_API_KEY_AQUI"  # <--- Coloca tu API Key aquí (ver .env.example)
BASE_URL = "https://api.themoviedb.org/3"

def obtener_detalle_pelicula(movie_id):
    """Consulta el detalle individual de una película para obtener presupuesto y género."""
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=es-ES"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        fecha = data.get('release_date', '')
        mes = int(fecha.split('-')[1]) if fecha and '-' in fecha else 1
        
        return {
            'id': data.get('id'),
            'titulo': data.get('title'),
            'presupuesto': data.get('budget', 0),
            'recaudacion': data.get('revenue', 0),
            'duracion_min': data.get('runtime', 0),
            'popularidad': data.get('popularity', 0),
            'mes_estreno': mes,
            'genero_principal': data.get('genres')[0]['name'] if data.get('genres') else 'Otro',
            'fecha_estreno': fecha
        }
    return None

# ---------------------------------------------------------
# 1. OBTENER DATOS HISTÓRICOS (Para entrenar)
# ---------------------------------------------------------
print("Paso 1: Descargando películas históricas...")
historicas_ids = []
for page in range(1, 6): # 5 páginas = 100 películas
    res = requests.get(f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=es-ES&page={page}")
    if res.status_code == 200:
        for m in res.json().get('results', []):
            historicas_ids.append(m['id'])
    time.sleep(0.1)

datos_historicos = []
for idx, m_id in enumerate(historicas_ids, 1):
    detalle = obtener_detalle_pelicula(m_id)
    if detalle and detalle['presupuesto'] > 0 and detalle['recaudacion'] > 0:
        datos_historicos.append(detalle)
    time.sleep(0.05)

df_train = pd.DataFrame(datos_historicos)
print(f"-> Películas históricas válidas para entrenar: {len(df_train)}")

# ---------------------------------------------------------
# 2. OBTENER PRÓXIMOS ESTRENOS (Solo fechas futuras)
# ---------------------------------------------------------
from datetime import datetime
hoy = datetime.now().strftime('%Y-%m-%d')

print("\nPaso 2: Descargando próximos estrenos...")
upcoming_ids = []

# Consultar las primeras 2 páginas para tener más opciones de próximos estrenos
for page in range(1, 3):
    res = requests.get(f"{BASE_URL}/movie/upcoming?api_key={API_KEY}&language=es-ES&page={page}")
    if res.status_code == 200:
        for m in res.json().get('results', []):
            fecha_estreno = m.get('release_date', '')
            # FILTRO: Solo agregar si la fecha de estreno es posterior a hoy
            if fecha_estreno and fecha_estreno > hoy:
                upcoming_ids.append(m['id'])
    time.sleep(0.1)

datos_proximos = []
for m_id in upcoming_ids:
    detalle = obtener_detalle_pelicula(m_id)
    if detalle:
        datos_proximos.append(detalle)
    time.sleep(0.05)

df_pred = pd.DataFrame(datos_proximos)
print(f"-> Próximos estrenos futuros obtenidos: {len(df_pred)}")

# ---------------------------------------------------------
# 3. PREPROCESAMIENTO Y ENTRENAMIENTO DEL MODELO
# ---------------------------------------------------------
print("\nPaso 3: Entrenando el modelo...")

# Codificar el género (convertir texto a número)
le = LabelEncoder()
todos_los_generos = list(set(df_train['genero_principal']).union(set(df_pred['genero_principal'])))
le.fit(todos_los_generos)

df_train['genero_encoded'] = le.transform(df_train['genero_principal'])
df_pred['genero_encoded'] = le.transform(df_pred['genero_principal'])

# Variables de entrada (X) y variable objetivo (Y)
features = ['presupuesto', 'duracion_min', 'popularidad', 'mes_estreno', 'genero_encoded']

X_train = df_train[features]
y_train = df_train['recaudacion']

# Entrenar modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. PREDICCIÓN Y EXPORTACIÓN A CSV
# ---------------------------------------------------------
print("\nPaso 4: Generando predicciones y guardando CSV...")

X_pred = df_pred[features]
df_pred['recaudacion_estimada_USD'] = model.predict(X_pred)

# Formatear la recaudación en formato legible ($)
df_pred['recaudacion_estimada_USD'] = df_pred['recaudacion_estimada_USD'].round(2)

# Seleccionar y ordenar las columnas para la entrega final
columnas_finales = [
    'id', 'titulo', 'fecha_estreno', 'genero_principal', 
    'presupuesto', 'duracion_min', 'popularidad', 'recaudacion_estimada_USD'
]
df_resultado = df_pred[columnas_finales]

# *** EXPORTACIÓN A CSV ***
nombre_archivo = "predicciones_proximos_estrenos.csv"
df_resultado.to_csv(nombre_archivo, index=False, encoding='utf-8-sig')

print(f"\n¡Éxito! Se ha generado el archivo '{nombre_archivo}'.")
print("\nVista previa de los resultados:")
print(df_resultado[['titulo', 'fecha_estreno', 'recaudacion_estimada_USD']].head())