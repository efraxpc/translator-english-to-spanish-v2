"""Configuración global: modelo, API key y glosario."""

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("PDF_TRANSLATOR_MODEL", "gpt-5.6-terra")

# Páginas cuyos bloques se agrupan en un mismo request a OpenAI.
DEFAULT_PAGES_PER_BATCH = 5

CACHE_DIR_NAME = ".translation_cache"


class MissingApiKeyError(RuntimeError):
    """No se configuró una API key válida para OpenAI."""


def get_openai_api_key(environ: Mapping[str, str] | None = None) -> str:
    """Obtiene la API key y admite el nombre antiguo usado por el proyecto."""
    variables = os.environ if environ is None else environ
    api_key = (variables.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        # Compatibilidad con el nombre que apareció en versiones anteriores
        # del .env.example. OPENAI_API_KEY sigue siendo el nombre recomendado.
        api_key = (variables.get("OPEN_AI_API_KEY") or "").strip()
    if not api_key:
        raise MissingApiKeyError(
            "falta OPENAI_API_KEY; añádela al archivo .env del proyecto"
        )
    return api_key


def load_glossary(path: Path | None = None) -> list[str]:
    """Lee glossary.txt (términos a conservar en inglés), ignorando comentarios."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "glossary.txt"
    if not path.exists():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms
