# Pochoclo Predict

Trabajo Practico Final - **Web Mining** (Magister en Ciencia de Datos)

## Objetivo

Predecir la **nota promedio** (`nota_promedio`, escala 1-10) que los usuarios
de [The Movie Database (TMDB)](https://www.themoviedb.org/) le asignan a una
pelicula, a partir de datos conocidos al momento del estreno (o antes):
presupuesto, duracion, fecha de estreno, genero(s), elenco, palabras clave,
sinopsis, y el desempeño **historico** (previo) del director, el elenco, la
saga/coleccion y las productoras involucradas.

Es un problema de **regresion supervisada**: la variable objetivo es
continua (no una clase), y se evalua con metricas de error de regresion
(MAE, RMSE, R², MAPE).

## Fuente de datos

[TMDB API v3](https://developer.themoviedb.org/reference/intro/getting-started).
Se requiere una API Key gratuita (ver seccion "Configuracion" mas abajo).

### Alcance y condiciones del dataset

- Solo peliculas estrenadas **a partir de 1980**, en estado `Released`.
- Solo peliculas con **director identificado** y con **presupuesto cargado**
  en TMDB (son variables centrales para el modelado; TMDB no tiene cargado
  el presupuesto en aproximadamente dos tercios de su catalogo).
- Piso minimo de votos (`vote_count >= 30`) para que `nota_promedio` sea
  estadisticamente confiable.

### Muestreo: aleatorio, estratificado por anio, filtrado en linea

Ordenar `/discover/movie` por cantidad de votos y recorrer paginas en
secuencia (primer enfoque probado) da siempre el mismo techo: las peliculas
mas votadas de toda la historia, sistematicamente mejor calificadas (sesgo
de seleccion). Para evitarlo, `TMDBClient.sample_movies` (en
`src/tmdb_client.py`) arma la muestra anio por anio: dentro de cada anio
recorre las paginas de resultados **en orden aleatorio** (y los resultados
dentro de cada pagina tambien en orden aleatorio), pidiendo el detalle de
cada candidata hasta **alcanzar la cantidad objetivo de peliculas que
cumplen las condiciones de arriba** (o agotar el universo disponible de ese
anio, lo que ocurra primero). El filtro de director/presupuesto se aplica
**durante** la busqueda (no en un paso posterior) para maximizar cuantas
peliculas validas se obtienen de cada anio, en lugar de pedir un lote fijo
de candidatas al azar y descartar despues las que no sirven.

## Tecnicas utilizadas

| Etapa | Tecnica |
| --- | --- |
| Extraccion | Muestreo aleatorio estratificado por anio sobre la API REST de TMDB, con filtro de calidad aplicado en linea, checkpoint incremental y reintentos/backoff ante errores. Enriquecimiento posterior con el poster de cada pelicula y sus identificadores externos (`imdb_id`, `wikidata_id`), que dejan preparado un futuro cruce con otras fuentes (IMDb, Wikidata) por id exacto |
| EDA | Analisis univariado/bivariado, deteccion de asimetria, correlacion |
| Feature Engineering | TF-IDF + SVD (elenco y keywords), analisis de sentimiento (VADER) sobre la sinopsis, codificacion one-hot del genero principal, codificacion **multi-label** de subgeneros, y **features historicas calculadas de forma temporal/expansiva** (sin data leakage) para director, elenco, saga y productoras. Cierra con un heatmap de correlacion de Pearson y un PCA exploratorio |
| Modelado | Comparacion de modelos de **boosting** (Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost) y lineales regularizados (Ridge, ElasticNet) contra un baseline, con busqueda de hiperparametros (`RandomizedSearchCV`), seleccion por MAE en un test set held-out, e interpretabilidad con importancia por permutacion y **SHAP** |
| Evaluacion PCA | Reentrenamiento de los mismos modelos sobre 80 componentes principales vs. las features originales, con **test de Wilcoxon** apareado para evaluar si la diferencia es significativa |
| Dashboard | Panel interactivo (`ipywidgets`) que elige peliculas al azar del test set y muestra poster, datos, director y reparto principal, junto con la nota predicha por el modelo ganador vs. la real. Incluye un medidor tipo reloj cuyas zonas se calibran con los percentiles del error del modelo en test (buena <= P30, regular P30-P70, mala > P70) y un grafico predicho vs. real acumulado de la sesion |

### Variables excluidas por fuga de informacion (_data leakage_)

`recaudacion`, `popularidad` y `cantidad_votos` **no se usan como feature
directa de la propia pelicula**: se conocen recien _despues_ del estreno, al
mismo tiempo que `nota_promedio` (`cantidad_votos` es literalmente la
cantidad de votos con la que se calculo esa nota), asi que usarlas
directamente seria entrenar con informacion no disponible en el momento real
de prediccion. En cambio, se usan como insumo para construir **features
historicas** -desempeño de peliculas _anteriores_ del mismo director, elenco,
saga o productora-, calculadas siempre con un corte temporal (solo peliculas
estrenadas antes que la que se esta prediciendo), para que esa informacion
historica sea legitima y no filtre el resultado de la propia pelicula ni de
peliculas futuras. El detalle esta documentado en
`03_feature_engineering.ipynb`.

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
│   ├── 01_extraccion_datos.ipynb       <- descarga el dataset crudo desde la API de TMDB (+ posters e ids externos)
│   ├── 02_eda.ipynb                    <- analisis exploratorio + limpieza
│   ├── 03_feature_engineering.ipynb    <- TF-IDF, sentimiento, generos, features historicas, correlacion y PCA
│   ├── 04_modelos_predictivos.ipynb    <- entrenamiento, comparacion, SHAP y seleccion del modelo
│   ├── 05_pca_vs_original.ipynb        <- PCA vs. features originales + test de Wilcoxon
│   └── 06_dashboard.ipynb              <- dashboard interactivo con el modelo ganador
│
├── src/
│   ├── paths.py                        <- rutas del proyecto (portables, sin hardcodear)
│   └── tmdb_client.py                  <- cliente de la API de TMDB (fetch + parseo)
│
├── data/
│   ├── raw/peliculas_raw.csv           <- salida de 01 (se regenera al ejecutar el notebook)
│   └── processed/                      <- salidas de 02 y 03
│
└── models/best_model.pkl               <- mejor modelo entrenado (salida de 04)
```

> Nota: la carpeta local `legacy/` (prototipos exploratorios previos a este
> pipeline) no forma parte del repositorio (ver `.gitignore`).

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
   API Key en _Configuracion > API_.
2. Copiar `.env.example` a un nuevo archivo `.env` (misma carpeta) y
   completar `TMDB_API_KEY` con la key propia.

### 3. Ejecutar los notebooks en orden

```bash
jupyter lab
```

Y correr, en orden, `01` → `02` → `03` → `04`. Cada notebook lee la salida
del anterior desde `data/`, por lo que no se pueden saltear pasos ni
correrlos en otro orden. Despues, `05` (evaluacion PCA) y `06` (dashboard)
se pueden correr en cualquier orden: ambos solo leen lo ya generado por los
notebooks anteriores.

> `01` tarda del orden de una hora (una llamada a la API por cada pelicula
> candidata). Si se interrumpe, al volver a ejecutarlo retoma desde donde
> quedo gracias a los archivos de checkpoint `data/raw/_*_parcial.csv`.

> El dashboard (`06`) necesita un kernel activo para que funcione el boton
> (JupyterLab o notebooks de VS Code), y conexion a internet para mostrar
> los posters, que se cargan desde el CDN publico de TMDB.

> Nota sobre redes corporativas: si la red tiene un proxy que inspecciona el
> trafico HTTPS (comun en entornos corporativos), `src/tmdb_client.py` ya
> incluye el uso de `truststore` para validar los certificados contra el
> almacen de confianza del sistema operativo en lugar del bundle de
> `certifi`, evitando errores de `SSLCertVerificationError`.

## Autor

Sol Gabriele Peruilh - Leonardo Gabriel Gastaldo -
Ezequiel Baglieri- Maximiliano Vergara Dominguez

Trabajo Practico Final de la materia Web Mining, Magister en Ciencia de
Datos de la Universidad Austral.
