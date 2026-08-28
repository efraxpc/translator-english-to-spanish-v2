"""Extracción de bloques de texto de un PDF con PyMuPDF."""

import re
from collections.abc import Iterable
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
    is_table_cell: bool = False
    layout_bbox: tuple[float, float, float, float] | None = None


def _should_translate(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if _SKIP_RE.match(stripped):
        return False
    if _URL_RE.match(stripped):
        return False
    return True


def _build_block(
    block_id: str,
    page_index: int,
    raw_blocks: list[dict],
    *,
    is_table_cell: bool = False,
    layout_bbox: tuple[float, float, float, float] | None = None,
) -> Block | None:
    """Construye un bloque y calcula su tipografía a partir de texto extraído."""
    lines: list[str] = []
    mono_chars = 0
    total_chars = 0
    sizes: dict[float, int] = {}
    bold_chars = 0
    italic_chars = 0
    text_bbox: pymupdf.Rect | None = None

    for raw in raw_blocks:
        if raw.get("type") != 0:
            continue
        raw_rect = pymupdf.Rect(raw["bbox"])
        text_bbox = raw_rect if text_bbox is None else text_bbox | raw_rect
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
    if not text.strip() or text_bbox is None:
        return None

    mono_ratio = mono_chars / max(total_chars, 1)
    is_code = looks_like_code(text, mono_ratio)
    dominant_size = max(sizes, key=sizes.get) if sizes else 11.0
    return Block(
        id=block_id,
        page=page_index,
        bbox=tuple(text_bbox),
        text=text,
        font_size=dominant_size,
        bold=bold_chars / max(total_chars, 1) > 0.5,
        italic=italic_chars / max(total_chars, 1) > 0.5,
        is_code=is_code,
        translatable=not is_code and _should_translate(text),
        lines=lines,
        is_table_cell=is_table_cell,
        layout_bbox=layout_bbox,
    )


def _find_tables(page: pymupdf.Page) -> list:
    """Detecta tablas delimitadas por líneas sin comprometer la extracción."""
    try:
        return page.find_tables().tables
    except Exception:
        # Algunos PDFs malformados hacen fallar el analizador de tablas. En
        # ese caso se conserva el comportamiento previo para toda la página.
        return []


def _cell_layout_bbox(
    cell: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Deja un margen mínimo para que el texto no toque la cuadrícula."""
    rect = pymupdf.Rect(cell)
    if rect.width > 4 and rect.height > 4:
        rect += (2, 2, -2, -2)
    return tuple(rect)


def _bbox_center_inside(
    bbox: tuple[float, float, float, float],
    containers: list[pymupdf.Rect],
) -> bool:
    """Indica si el centro del texto pertenece a una tabla detectada."""
    point = (pymupdf.Rect(bbox).tl + pymupdf.Rect(bbox).br) / 2
    return any(point in container for container in containers)


def extract_blocks(
    doc: pymupdf.Document,
    page_indexes: Iterable[int] | None = None,
) -> list[Block]:
    """Devuelve los bloques clasificados de las páginas indicadas.

    Los índices son los originales del documento. Si no se especifican, se
    procesan todas las páginas.
    """
    blocks: list[Block] = []
    pages = (
        enumerate(doc)
        if page_indexes is None
        else ((page_index, doc[page_index]) for page_index in page_indexes)
    )
    for page_index, page in pages:
        data = page.get_text("dict")
        tables = _find_tables(page)
        table_rects = [pymupdf.Rect(table.bbox) for table in tables]
        page_blocks: list[Block] = []

        for b_index, raw in enumerate(data["blocks"]):
            if raw.get("type") != 0:  # 0 = texto; 1 = imagen
                continue
            if _bbox_center_inside(tuple(raw["bbox"]), table_rects):
                continue
            block = _build_block(f"p{page_index}_b{b_index}", page_index, [raw])
            if block is not None:
                page_blocks.append(block)

        seen_cells: set[tuple[float, float, float, float]] = set()
        for table_index, table in enumerate(tables):
            for row_index, row in enumerate(table.rows):
                for column_index, cell in enumerate(row.cells):
                    if cell is None:
                        continue
                    cell_key = tuple(round(value, 3) for value in cell)
                    if cell_key in seen_cells:
                        continue
                    seen_cells.add(cell_key)
                    cell_data = page.get_text("dict", clip=pymupdf.Rect(cell))
                    block = _build_block(
                        f"p{page_index}_t{table_index}_r{row_index}_c{column_index}",
                        page_index,
                        cell_data["blocks"],
                        is_table_cell=True,
                        layout_bbox=_cell_layout_bbox(cell),
                    )
                    if block is not None:
                        page_blocks.append(block)

        page_blocks.sort(key=lambda block: (block.bbox[1], block.bbox[0]))
        blocks.extend(page_blocks)
    return blocks
