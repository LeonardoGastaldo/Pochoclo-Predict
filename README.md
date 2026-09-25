# Pochoclo Predict

Trabajo Practico Final - **Web Mining** (Magister en Ciencia de Datos)

## Objetivo

Predecir la **nota promedio** (`nota_promedio`, escala 1-10) que los usuarios
de [The Movie Database (TMDB)](https://www.themoviedb.org/) le asignan a una
pelicula, a partir de datos obtenidos directamente de la API publica de TMDB:
metadata de produccion (presupuesto, recaudacion, duracion, fecha de
estreno), genero(s), elenco, palabras clave y sinopsis.

Es un problema de **regresion supervisada**: la variable objetivo es
continua (no una clase), y se evalua con metricas de error de regresion
(MAE, RMSE, R², MAPE).

## Fuente de datos

[TMDB API v3](https://developer.themoviedb.org/reference/intro/getting-started).
Se requiere una API Key gratuita (ver seccion "Configuracion" mas abajo).

## Tecnicas utilizadas

| Etapa | Tecnica |
|---|---|
| Extraccion | Consumo de API REST paginada, con reintentos y backoff ante errores/rate-limit |
| EDA | Analisis univariado/bivariado, deteccion de asimetria y de "ceros faltantes", correlacion |
| Feature Engineering | TF-IDF + SVD (elenco y keywords), analisis de sentimiento (VADER) sobre la sinopsis, codificacion one-hot del genero principal, codificacion **multi-label** de subgeneros, encoding de experiencia del director |
| Modelado | Comparacion de modelos de **boosting** (Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost) contra un baseline, con busqueda de hiperparametros (`RandomizedSearchCV`) y seleccion por MAE en un test set held-out |

## Estructura del proyecto

```
Pochoclo Predict/
├── README.md                          <- este archivo
├── requirements.txt                   <- dependencias exactas del proyecto
├── .env.example                       <- plantilla para la API Key (copiar a .env)
├── .env                                <- API Key real (NO se versiona)
├── .gitignore
│
├── notebooks/
│   ├── 01_extraccion_datos.ipynb       <- descarga el dataset crudo desde la API de TMDB
│   ├── 02_eda.ipynb                    <- analisis exploratorio + limpieza
│   ├── 03_feature_engineering.ipynb    <- TF-IDF, sentimiento, encoding de generos
│   └── 04_modelos_predictivos.ipynb    <- entrenamiento, comparacion y seleccion del modelo
│
├── src/
│   ├── paths.py                        <- rutas del proyecto (portables, sin hardcodear)
│   └── tmdb_client.py                  <- cliente de la API de TMDB (fetch + parseo)
│
├── data/
│   ├── raw/peliculas_raw.csv           <- salida de 01 (se regenera al ejecutar el notebook)
│   └── processed/                      <- salidas de 02 y 03
│
├── models/best_model.pkl               <- mejor modelo entrenado (salida de 04)
│
└── legacy/prototipo_original.py        <- script exploratorio inicial (prediccion de recaudacion), reemplazado por los notebooks
```

## Como reproducir el proyecto

El proyecto esta pensado para poder copiarse a **cualquier carpeta o equipo**
y funcionar sin modificar ninguna ruta: todas las rutas se resuelven de
forma relativa a la ubicacion del propio proyecto (ver `src/paths.py`), no
hay ninguna ruta absoluta hardcodeada.

### 1. Crear el entorno

Con conda (recomendado):

```bash
conda create -n webmining python=3.11
conda activate webmining
pip install -r requirements.txt
```

O con un entorno virtual estandar:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configurar la API Key de TMDB

1. Crear una cuenta gratuita en https://www.themoviedb.org/ y generar una
   API Key en *Configuracion > API*.
2. Copiar `.env.example` a un nuevo archivo `.env` (misma carpeta) y
   completar `TMDB_API_KEY` con la key propia.

### 3. Ejecutar los notebooks en orden

```bash
jupyter lab
```

Y correr, en orden, `01` → `02` → `03` → `04`. Cada notebook lee la salida
del anterior desde `data/`, por lo que no se pueden saltear pasos ni
correrlos en otro orden.

> Nota sobre redes corporativas: si la red tiene un proxy que inspecciona el
> trafico HTTPS (comun en entornos corporativos), `src/tmdb_client.py` ya
> incluye el uso de `truststore` para validar los certificados contra el
> almacen de confianza del sistema operativo en lugar del bundle de
> `certifi`, evitando errores de `SSLCertVerificationError`.

## Resultados

Dataset final: **2.998 peliculas** (2.398 de entrenamiento / 600 de test),
**100 features**, tras filtrar por `cantidad_votos >= 30` y descartar
peliculas sin sinopsis.

Comparacion de modelos (metricas sobre el test set, 600 peliculas nunca
vistas durante la busqueda de hiperparametros):

| Modelo | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| **CatBoost** (ganador) | **0.311** | **0.407** | **0.666** | **4.64%** |
| XGBoost | 0.313 | 0.408 | 0.664 | 4.66% |
| HistGradientBoosting | 0.318 | 0.409 | 0.661 | 4.72% |
| LightGBM | 0.318 | 0.410 | 0.661 | 4.74% |
| Gradient Boosting | 0.329 | 0.425 | 0.635 | 4.90% |
| Baseline (media) | 0.575 | 0.705 | -0.003 | 8.43% |

**CatBoost** resulto el mejor modelo: en promedio se equivoca por apenas
**0.31 puntos** sobre una escala de 1 a 10 (MAPE ≈ 4.6%), y explica cerca
del **67% de la varianza** de `nota_promedio`. Los cinco modelos de boosting
superan holgadamente al baseline (que solo predice la nota media), lo que
confirma que las features de elenco, keywords, sentimiento y metadata de
produccion construidas en el NB03 tienen poder predictivo real. El detalle
de importancia de features (por permutacion) esta en la seccion 6 de
`04_modelos_predictivos.ipynb`.

## Autor

Trabajo Practico Final de la materia Web Mining, Magister en Ciencia de
Datos.
