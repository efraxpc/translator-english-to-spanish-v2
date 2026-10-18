"""Reconstrucción del PDF: borra el texto original e inserta la traducción."""

import html
import re
from statistics import median

import pymupdf

from .extractor import Block

# CSS base: fuente serif legible y line-height compacto para que quepa el ~20%
# extra de longitud del español.
_BASE_CSS = "* { margin: 0; padding: 0; line-height: 1.15; }"

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MD_CODE_RE = re.compile(r"`([^`\n]+)`")
_MAX_PARAGRAPH_WIDTH = 13 * 72 / 2.54  # centímetros a puntos PDF


def _layout_rects(
    page: pymupdf.Page, blocks: list[Block], structured: set[str]
) -> dict[str, pymupdf.Rect]:
    """Reserva espacio para encabezados y evita colisiones al centrar párrafos."""
    body_sizes = [
        b.font_size for b in blocks
        if not b.bold and not b.is_table_cell
        and max(len(b.text), len(b.translated or "")) > 120
    ]
    body_size = median(body_sizes) if body_sizes else 11.0
    headings = {
        b.id for b in blocks
        if b.id not in structured
        and (b.bold or b.font_size >= body_size * 1.15
             or max(len(b.text.strip()), len((b.translated or "").strip())) <= 120)
    }
    original = {b.id: pymupdf.Rect(b.layout_bbox or b.bbox) for b in blocks}
    rects = {key: pymupdf.Rect(rect) for key, rect in original.items()}
    movable = {
        b.id for b in blocks
        if b.translatable and b.translated and b.id not in structured | headings
    }
    bounds = page.rect * page.derotation_matrix
    for key in movable:
        rect = rects[key]
        width = min(rect.width, _MAX_PARAGRAPH_WIDTH, bounds.width)
        rect.x0 = bounds.x0 + (bounds.width - width) / 2
        rect.x1 = rect.x0 + width

    # Si el centrado invade otro bloque, vuelve a su columna original sin
    # ampliar el párrafo. Repite porque restaurar una columna ocupa otro lugar.
    pending = set(movable)
    while pending:
        collisions = {
            key for key in pending
            if any(key != other and rects[key].intersects(rect)
                   for other, rect in rects.items())
        }
        if not collisions:
            break
        for key in collisions:
            rects[key] = pymupdf.Rect(original[key])
            rects[key].x1 = min(rects[key].x1, rects[key].x0 + _MAX_PARAGRAPH_WIDTH)
        pending -= collisions

    # Algunas cajas OCR de títulos invaden la primera línea del bloque
    # siguiente. Reparte esa franja antes de insertar, manteniendo ambos textos.
    editable = {b.id for b in blocks if b.translatable and b.translated}
    ordered = sorted(rects, key=lambda key: (rects[key].y0, rects[key].x0))
    for index, key in enumerate(ordered):
        for other in ordered[index + 1:]:
            if key not in editable or other not in editable:
                continue
            if key in structured or other in structured:
                continue
            if key not in headings and other not in headings:
                continue
            upper, lower = rects[key], rects[other]
            if not upper.intersects(lower):
                continue
            if upper.y0 < lower.y0 < upper.y1 < lower.y1:
                boundary = (upper.y1 + lower.y0) / 2
                gap = min(2.0, (boundary - upper.y0) / 4, (lower.y1 - boundary) / 4)
                upper.y1 = boundary - gap / 2
                lower.y0 = boundary + gap / 2
    return rects


def _structured_blocks(page: pymupdf.Page, blocks: list[Block]) -> set[str]:
    """Reconoce estructuras por columnas y separadores, incluso en cachés viejos."""
    structured = {block.id for block in blocks if block.is_table_cell}
    for block in blocks:
        lines = block.text.splitlines()
        if sum(bool(re.search(r"\S(?:\t| {2,}|\s*\|\s*)\S", line)) for line in lines) >= 2:
            structured.add(block.id)

    # Las columnas pueden estar en bloques independientes o en líneas de un
    # mismo bloque PyMuPDF. Incluye los originales sin traducción (p. ej. cifras).
    entries = [(pymupdf.Rect(block.bbox), {block.id}) for block in blocks]
    if any(not block.from_ocr for block in blocks):
        for raw in page.get_text("dict")["blocks"]:
            if raw.get("type") != 0:
                continue
            for line in raw["lines"]:
                rect = pymupdf.Rect(line["bbox"])
                owners = {
                    block.id for block in blocks
                    if not block.from_ocr
                    and pymupdf.Rect(block.bbox).contains((rect.tl + rect.br) / 2)
                }
                entries.append((rect, owners))
    for index, (left, owners) in enumerate(entries):
        for right, other_owners in entries[index + 1:]:
            overlap = min(left.y1, right.y1) - max(left.y0, right.y0)
            gap = max(right.x0 - left.x1, left.x0 - right.x1)
            if gap >= 8 and overlap > 0 and overlap >= min(left.height, right.height) * 0.5:
                structured.update(owners | other_owners)
    return structured


def _block_html(block: Block, preserve_structure: bool = False) -> str:
    text = html.escape(block.translated or block.text)
    # El modelo a veces devuelve énfasis en markdown: se convierte a HTML.
    text = _MD_BOLD_RE.sub(r"<b>\1</b>", text)
    text = _MD_ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _MD_CODE_RE.sub(r"<code>\1</code>", text)
    if not preserve_structure:
        text = text.replace("\n", " ")  # los párrafos se re-fluyen dentro del bbox
    style = f"font-size: {block.font_size:.1f}px;"
    if preserve_structure:
        style += " white-space: pre-wrap;"
    if block.bold:
        style += " font-weight: bold;"
    if block.italic:
        style += " font-style: italic;"
    return f'<div style="{style}">{text}</div>'


def rebuild(doc: pymupdf.Document, blocks: list[Block]) -> None:
    """Aplica las traducciones sobre el documento, página por página.

    Los bloques de código y los no traducibles no se tocan: quedan tal cual
    en la página original, con su formato íntegro.
    """
    by_page: dict[int, list[Block]] = {}
    all_by_page: dict[int, list[Block]] = {}
    for block in blocks:
        all_by_page.setdefault(block.page, []).append(block)
        if block.translatable and block.translated:
            by_page.setdefault(block.page, []).append(block)

    for page_index, page_blocks in sorted(by_page.items()):
        page = doc[page_index]
        structured = _structured_blocks(page, all_by_page[page_index])
        layout_rects = _layout_rects(page, all_by_page[page_index], structured)

        redact_blocks = [b for b in page_blocks if not b.from_ocr]
        for block in redact_blocks:
            page.add_redact_annot(pymupdf.Rect(block.bbox))
        if redact_blocks:
            # Borra solo texto; imágenes y gráficos vectoriales quedan intactos.
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            )

        for block in page_blocks:
            source_rect = pymupdf.Rect(block.bbox)
            if block.from_ocr:
                # El texto original es vectorial (curvas): no se puede redactar
                # sin arrastrar los gráficos, así que se tapa con un rectángulo
                # (con margen, porque el bbox del OCR no es exacto al píxel).
                margin = 1 if block.is_table_cell else 3
                cover = (
                    source_rect + (-margin, -margin, margin, margin)
                ) & page.rect
                page.draw_rect(cover, color=None, fill=(1, 1, 1))

        # Tapa todos los originales antes de insertar texto desplazado, para
        # que una cobertura OCR posterior no borre parte de otro párrafo.
        for block in page_blocks:
            layout_rect = layout_rects[block.id]
            preserve_structure = block.id in structured
            # scale_low=0 permite reducir la fuente lo necesario para caber.
            page.insert_htmlbox(
                layout_rect, _block_html(block, preserve_structure),
                css=_BASE_CSS, scale_low=0
            )
