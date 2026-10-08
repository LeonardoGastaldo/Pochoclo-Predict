"""
Cliente sencillo para consumir la API publica de The Movie Database (TMDB).

Responsable de:
  - Descubrir ids de peliculas (endpoint /discover/movie).
  - Descargar el detalle completo de una pelicula en una unica llamada
    (detalle + reparto/equipo + palabras clave + traducciones) usando
    ``append_to_response``, para minimizar la cantidad de requests.
  - Aplanar (parsear) la respuesta JSON anidada en un diccionario "flat"
    listo para convertirse en una fila de un DataFrame de pandas.

Nota sobre SSL: en redes corporativas con un proxy que inspecciona el
trafico HTTPS (Netskope, Zscaler, etc.) el certificado que llega a Python
esta firmado por una CA interna que no figura en el bundle de ``certifi``.
``truststore`` resuelve esto delegando la verificacion al almacen de
certificados del sistema operativo (Windows/mac/Linux), que si conoce esa CA.
"""

import json
import random
import time
from pathlib import Path
from typing import Optional

import truststore

truststore.inject_into_ssl()

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import paths

load_dotenv(paths.ENV_FILE)

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p"


class TMDBClient:
    def __init__(self, api_key: str, language: str = "es-ES", request_delay: float = 0.05):
        if not api_key:
            raise ValueError(
                "Falta la API Key de TMDB. Copia '.env.example' a '.env' en la raiz "
                "del proyecto y completa TMDB_API_KEY."
            )
        self.api_key = api_key
        self.language = language
        self.request_delay = request_delay
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        params = dict(params or {})
        params["api_key"] = self.api_key
        params.setdefault("language", self.language)
        resp = self.session.get(f"{BASE_URL}{endpoint}", params=params, timeout=15)
        time.sleep(self.request_delay)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Detalle completo de una pelicula
    # ------------------------------------------------------------------
    def get_movie_full(self, movie_id: int) -> Optional[dict]:
        try:
            return self._get(
                f"/movie/{movie_id}",
                params={"append_to_response": "credits,keywords,translations,external_ids"},
            )
        except requests.exceptions.RequestException:
            return None

    def get_poster_paths(self, movie_id: int) -> dict:
        """Version liviana de `get_movie_full`, sin `append_to_response`, para
        cuando solo hace falta poster/backdrop de una pelicula cuyo resto de
        datos ya se tiene (ej. enriquecer un dataset ya extraido)."""
        try:
            data = self._get(f"/movie/{movie_id}")
        except requests.exceptions.RequestException:
            return {"poster_path": None, "backdrop_path": None}
        return {"poster_path": data.get("poster_path"), "backdrop_path": data.get("backdrop_path")}

    def get_external_ids(self, movie_id: int) -> dict:
        """Usa el endpoint dedicado `/movie/{id}/external_ids` (mas liviano que
        pedir el detalle completo) para obtener identificadores externos de
        una pelicula cuyo resto de datos ya se tiene. TMDB devuelve ademas del
        `imdb_id` el `wikidata_id` (el QID de Wikidata) directamente, sin
        necesidad de matchear por titulo: ambos sirven como llave para cruzar
        en el futuro con otras fuentes (datasets oficiales de IMDb, consultas
        SPARQL a Wikidata para premios -propiedad P166-, etc.)."""
        try:
            data = self._get(f"/movie/{movie_id}/external_ids")
        except requests.exceptions.RequestException:
            return {"imdb_id": None, "wikidata_id": None}
        return {"imdb_id": data.get("imdb_id"), "wikidata_id": data.get("wikidata_id")}

    # ------------------------------------------------------------------
    # Muestreo aleatorio estratificado por anio, con filtro aplicado EN LINEA
    # ------------------------------------------------------------------
    def sample_movies(
        self,
        year_from: int,
        year_to: int,
        min_vote_count: int = 30,
        target_per_year: int = 100,
        require_director: bool = True,
        require_budget: bool = True,
        seed: int = 42,
        on_year_done=None,
    ) -> list[dict]:
        """
        Arma una muestra ALEATORIA de peliculas ya parseadas que cumplen las
        condiciones de calidad pedidas (director identificado, presupuesto
        cargado), tomando del universo real de peliculas elegibles la mayor
        cantidad posible por anio (hasta `target_per_year`).

        Esto es deliberadamente distinto a "pedir un lote fijo de ids al azar
        y descartar despues los que no cumplen": la API de TMDB no permite
        filtrar por "tiene presupuesto cargado" en `/discover/movie` (ese
        dato solo se conoce al pedir el detalle de cada pelicula), por lo que
        es inevitable pedir el detalle de algunas candidatas que terminan
        descartandose. Lo que si se puede controlar es NO frenar la busqueda
        hasta alcanzar la cantidad objetivo de peliculas que **si cumplen**
        (o hasta agotar el universo disponible de ese anio, lo que ocurra
        primero), en lugar de tomar un lote chico al azar y aceptar lo que
        sea que sobreviva al filtro.

        Para evitar el sesgo hacia las peliculas mas votadas/mas famosas
        (ver `discover_movie_ids_stratified` en versiones anteriores), por
        cada anio del rango [year_from, year_to] se recorren las paginas de
        `/discover/movie` en **orden aleatorio**, y dentro de cada pagina los
        resultados tambien se visitan en **orden aleatorio**.

        `on_year_done(anio, aceptadas, universo_visto)` es un callback
        opcional para reportar progreso (ej. imprimir desde el notebook).
        """
        rng = random.Random(seed)
        aceptadas: list[dict] = []

        for anio in range(year_from, year_to + 1):
            primera = self._get(
                "/discover/movie",
                params={
                    "primary_release_year": anio,
                    "vote_count.gte": min_vote_count,
                    "include_adult": "false",
                    "sort_by": "vote_count.desc",
                    "page": 1,
                },
            )
            total_pages = min(primera.get("total_pages", 0), 500)  # tope duro de la API
            if total_pages == 0:
                if on_year_done:
                    on_year_done(anio, 0, 0)
                continue

            paginas = list(range(1, total_pages + 1))
            rng.shuffle(paginas)

            aceptadas_anio: list[dict] = []
            ids_vistos: set[int] = set()

            for pagina in paginas:
                if len(aceptadas_anio) >= target_per_year:
                    break
                data = primera if pagina == 1 else self._get(
                    "/discover/movie",
                    params={
                        "primary_release_year": anio,
                        "vote_count.gte": min_vote_count,
                        "include_adult": "false",
                        "sort_by": "vote_count.desc",
                        "page": pagina,
                    },
                )
                resultados = list(data.get("results", []))
                rng.shuffle(resultados)

                for m in resultados:
                    if len(aceptadas_anio) >= target_per_year:
                        break
                    if m["id"] in ids_vistos:
                        continue
                    ids_vistos.add(m["id"])

                    raw = self.get_movie_full(m["id"])
                    fila = TMDBClient.parse_movie(raw)
                    if fila is None:
                        continue
                    if require_director and not fila["director"]:
                        continue
                    if require_budget and not fila["presupuesto"]:
                        continue
                    aceptadas_anio.append(fila)

            aceptadas.extend(aceptadas_anio)
            if on_year_done:
                on_year_done(anio, len(aceptadas_anio), len(ids_vistos))

        return aceptadas

    @staticmethod
    def _english_overview(raw: dict) -> str:
        for t in raw.get("translations", {}).get("translations", []):
            if t.get("iso_639_1") == "en":
                return t.get("data", {}).get("overview", "") or ""
        return ""

    @staticmethod
    def parse_movie(raw: dict) -> Optional[dict]:
        """Aplana la respuesta cruda de la API a un dict listo para un DataFrame."""
        if not raw or not raw.get("id"):
            return None

        fecha = raw.get("release_date") or ""
        anio = int(fecha.split("-")[0]) if fecha else None
        mes = int(fecha.split("-")[1]) if fecha and "-" in fecha else None

        generos = [g["name"] for g in raw.get("genres", []) or []]
        keywords = [k["name"] for k in raw.get("keywords", {}).get("keywords", []) or []]

        cast = raw.get("credits", {}).get("cast", []) or []
        cast_ordenado = sorted(cast, key=lambda c: c.get("order", 999))
        reparto_principal = [c["name"] for c in cast_ordenado[:10]]

        crew = raw.get("credits", {}).get("crew", []) or []
        directores = [c["name"] for c in crew if c.get("job") == "Director"]

        companias = [c["name"] for c in raw.get("production_companies", []) or []]
        paises = [c["name"] for c in raw.get("production_countries", []) or []]

        coleccion = raw.get("belongs_to_collection") or None

        return {
            "id": raw.get("id"),
            "titulo": raw.get("title"),
            "titulo_original": raw.get("original_title"),
            "idioma_original": raw.get("original_language"),
            "fecha_estreno": fecha or None,
            "anio_estreno": anio,
            "mes_estreno": mes,
            "estado": raw.get("status"),
            "presupuesto": raw.get("budget", 0),
            "recaudacion": raw.get("revenue", 0),
            "duracion_min": raw.get("runtime", 0),
            "popularidad": raw.get("popularity", 0.0),
            "adultos": raw.get("adult", False),
            "generos": json.dumps(generos, ensure_ascii=False),
            "genero_principal": generos[0] if generos else "Desconocido",
            "num_generos": len(generos),
            "sinopsis_es": raw.get("overview", "") or "",
            "sinopsis_en": TMDBClient._english_overview(raw),
            "companias_productoras": json.dumps(companias, ensure_ascii=False),
            "paises_produccion": json.dumps(paises, ensure_ascii=False),
            "reparto_principal": json.dumps(reparto_principal, ensure_ascii=False),
            "director": directores[0] if directores else None,
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "coleccion_id": coleccion["id"] if coleccion else None,
            "coleccion_nombre": coleccion["name"] if coleccion else None,
            "poster_path": raw.get("poster_path"),
            "backdrop_path": raw.get("backdrop_path"),
            "imdb_id": raw.get("imdb_id"),
            "wikidata_id": raw.get("external_ids", {}).get("wikidata_id") if raw.get("external_ids") else None,
            "nota_promedio": raw.get("vote_average"),
            "cantidad_votos": raw.get("vote_count", 0),
        }

    @staticmethod
    def poster_url(poster_path: Optional[str], size: str = "w342") -> Optional[str]:
        """Arma la URL publica de la imagen a partir del `poster_path` (o
        `backdrop_path`) devuelto por la API. TMDB sirve estas imagenes desde
        un CDN publico: no hace falta descargarlas/guardarlas en el proyecto,
        alcanza con guardar el path y construir la URL al momento de mostrarla.
        Tamaños tipicos de poster: w92, w154, w185, w342, w500, original."""
        if not poster_path:
            return None
        return f"{IMAGE_BASE_URL}/{size}{poster_path}"
