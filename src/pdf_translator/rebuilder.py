"""Reconstrucción del PDF: borra el texto original e inserta la traducción."""

import html
import re

import pymupdf

from .extractor import Block

# CSS base: fuente serif legible y line-height compacto para que quepa el ~20%
# extra de longitud del español.
_BASE_CSS = "* { margin: 0; padding: 0; line-height: 1.15; }"

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MD_CODE_RE = re.compile(r"`([^`\n]+)`")


def _block_html(block: Block) -> str:
    text = html.escape(block.translated or block.text)
    # Gemini a veces devuelve énfasis en markdown: se convierte a HTML.
    text = _MD_BOLD_RE.sub(r"<b>\1</b>", text)
    text = _MD_ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _MD_CODE_RE.sub(r"<code>\1</code>", text)
    text = text.replace("\n", " ")  # los párrafos se re-fluyen dentro del bbox
    style = f"font-size: {block.font_size:.1f}px;"
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
    for block in blocks:
        if block.translatable and block.translated:
            by_page.setdefault(block.page, []).append(block)

    for page_index, page_blocks in sorted(by_page.items()):
        page = doc[page_index]

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
            layout_rect = pymupdf.Rect(block.layout_bbox or block.bbox)
            # scale_low=0 permite reducir la fuente lo necesario para caber.
            page.insert_htmlbox(
                layout_rect, _block_html(block), css=_BASE_CSS, scale_low=0
            )
