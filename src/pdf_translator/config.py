"""Configuración global: modelo, API key y glosario."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("PDF_TRANSLATOR_MODEL", "gemini-3.1-pro-preview")

# Páginas cuyos bloques se agrupan en un mismo request a Gemini.
DEFAULT_PAGES_PER_BATCH = 5

CACHE_DIR_NAME = ".translation_cache"


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
