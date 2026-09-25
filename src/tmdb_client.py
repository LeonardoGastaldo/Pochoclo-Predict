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
    # Descubrimiento de peliculas
    # ------------------------------------------------------------------
    def discover_movie_ids(
        self,
        min_vote_count: int = 100,
        max_pages: int = 100,
        sort_by: str = "vote_count.desc",
    ) -> list[int]:
        """
        Recorre /discover/movie ordenando por cantidad de votos descendente,
        de forma de priorizar peliculas con una calificacion (nota_promedio)
        estadisticamente mas confiable (evita el ruido de titulos con 1 o 2
        votos y notas de 1 o 10).
        """
        ids = []
        for page in range(1, max_pages + 1):
            data = self._get(
                "/discover/movie",
                params={
                    "sort_by": sort_by,
                    "vote_count.gte": min_vote_count,
                    "include_adult": "false",
                    "page": page,
                },
            )
            results = data.get("results", [])
            if not results:
                break
            ids.extend(m["id"] for m in results)
            if page >= data.get("total_pages", page):
                break
        return ids

    # ------------------------------------------------------------------
    # Detalle completo de una pelicula
    # ------------------------------------------------------------------
    def get_movie_full(self, movie_id: int) -> Optional[dict]:
        try:
            return self._get(
                f"/movie/{movie_id}",
                params={"append_to_response": "credits,keywords,translations"},
            )
        except requests.exceptions.RequestException:
            return None

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
            "nota_promedio": raw.get("vote_average"),
            "cantidad_votos": raw.get("vote_count", 0),
        }
