"""
Rutas del proyecto resueltas de forma relativa a la ubicacion de este archivo.

Esto permite que todo el proyecto (carpeta "Pochoclo Predict") pueda copiarse
o moverse a cualquier otra ubicacion / equipo sin romper ninguna ruta: nunca
se usan rutas absolutas "hardcodeadas", sino que todo se deriva de donde vive
este mismo modulo (src/paths.py), que siempre esta un nivel por debajo de la
raiz del proyecto.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ENV_FILE = PROJECT_ROOT / ".env"

RAW_MOVIES_CSV = DATA_RAW / "peliculas_raw.csv"
FEATURES_CSV = DATA_PROCESSED / "dataset_features.csv"
EDA_CLEAN_CSV = DATA_PROCESSED / "peliculas_clean.csv"
