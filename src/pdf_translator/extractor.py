"""Extracción de bloques de texto de un PDF con PyMuPDF."""

import re
from dataclasses import dataclass, field

import pymupdf

from .code_detect import is_mono_font, looks_like_code

# Bits de flags de span en PyMuPDF
_FLAG_ITALIC = 1 << 1
_FLAG_MONO = 1 << 3
_FLAG_BOLD = 1 << 4

_SKIP_RE = re.compile(r"^[\d\s\.,;:%\-–—/()\[\]]*$")  # solo números/puntuación
_URL_RE = re.compile(r"^\s*(https?://|www\.)\S+\s*$")


@dataclass
class Block:
    id: str
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float = 11.0
    bold: bool = False
    italic: bool = False
    is_code: bool = False
    translatable: bool = True
    translated: str | None = None
    lines: list[str] = field(default_factory=list)
    from_ocr: bool = False


def _should_translate(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if _SKIP_RE.match(stripped):
        return False
    if _URL_RE.match(stripped):
        return False
    return True


def extract_blocks(doc: pymupdf.Document) -> list[Block]:
    """Devuelve todos los bloques de texto del documento, clasificados."""
    blocks: list[Block] = []
    for page_index, page in enumerate(doc):
        data = page.get_text("dict")
        for b_index, raw in enumerate(data["blocks"]):
            if raw.get("type") != 0:  # 0 = texto; 1 = imagen
                continue

            lines: list[str] = []
            mono_chars = 0
            total_chars = 0
            sizes: dict[float, int] = {}
            bold_chars = 0
            italic_chars = 0

            for line in raw["lines"]:
                parts = []
                for span in line["spans"]:
                    span_text = span["text"]
                    parts.append(span_text)
                    n = len(span_text)
                    total_chars += n
                    if is_mono_font(span.get("font", "")) or span["flags"] & _FLAG_MONO:
                        mono_chars += n
                    if span["flags"] & _FLAG_BOLD:
                        bold_chars += n
                    if span["flags"] & _FLAG_ITALIC:
                        italic_chars += n
                    size = round(span["size"], 1)
                    sizes[size] = sizes.get(size, 0) + n
                lines.append("".join(parts))

            text = "\n".join(lines)
            if not text.strip():
                continue

            mono_ratio = mono_chars / max(total_chars, 1)
            is_code = looks_like_code(text, mono_ratio)
            dominant_size = max(sizes, key=sizes.get) if sizes else 11.0

            block = Block(
                id=f"p{page_index}_b{b_index}",
                page=page_index,
                bbox=tuple(raw["bbox"]),
                text=text,
                font_size=dominant_size,
                bold=bold_chars / max(total_chars, 1) > 0.5,
                italic=italic_chars / max(total_chars, 1) > 0.5,
                is_code=is_code,
                translatable=not is_code and _should_translate(text),
                lines=lines,
            )
            blocks.append(block)
    return blocks
