"""CLI: traduce PDFs de inglés a español conservando el layout original."""

import argparse
import sys
from pathlib import Path

import pymupdf

from .checkpoint import Checkpoint
from .config import DEFAULT_MODEL, DEFAULT_PAGES_PER_BATCH, load_glossary
from .extractor import Block, extract_blocks
from .ocr import GeminiOcr, block_from_dict, block_to_dict, looks_english
from .rebuilder import rebuild
from .translator import GeminiTranslator


def make_batches(blocks: list[Block], pages_per_batch: int) -> list[list[Block]]:
    """Agrupa los bloques traducibles en lotes de N páginas consecutivas."""
    translatable = [b for b in blocks if b.translatable]
    batches: list[list[Block]] = []
    current: list[Block] = []
    current_pages: set[int] = set()
    for block in translatable:
        if block.page not in current_pages and len(current_pages) >= pages_per_batch:
            batches.append(current)
            current, current_pages = [], set()
        current.append(block)
        current_pages.add(block.page)
    if current:
        batches.append(current)
    return batches


def translate_pdf(
    pdf_path: Path,
    output_path: Path,
    translator: GeminiTranslator,
    pages_per_batch: int,
    workdir: Path | None,
) -> None:
    doc = pymupdf.open(pdf_path)
    blocks = extract_blocks(doc)
    translatable = [b for b in blocks if b.translatable]
    code_blocks = sum(1 for b in blocks if b.is_code)
    print(
        f"  {len(doc)} páginas, {len(blocks)} bloques "
        f"({len(translatable)} a traducir, {code_blocks} de código conservados)"
    )

    checkpoint = Checkpoint(pdf_path, translator.model, workdir)

    # Páginas sin texto extraíble (texto vectorizado/escaneado) → OCR con Gemini.
    pages_with_text = {b.page for b in blocks}
    ocr_pages = [
        i for i, page in enumerate(doc)
        if i not in pages_with_text
        and (page.get_drawings() or page.get_images() or page.get_text().strip())
    ]
    if ocr_pages:
        print(
            f"  {len(ocr_pages)} páginas sin texto extraíble: "
            "usando OCR con Gemini (texto vectorizado o escaneado)"
        )
        ocr = GeminiOcr(translator.client, translator.model, translator.glossary)
        for n, page_index in enumerate(ocr_pages, 1):
            key = f"ocr_p{page_index + 1:04d}"
            cached = checkpoint.load(key)
            if cached is not None:
                print(f"  OCR {n}/{len(ocr_pages)} (pág. {page_index + 1}): cache ✓")
                page_blocks = [block_from_dict(d) for d in cached["blocks"]]
            else:
                print(f"  OCR {n}/{len(ocr_pages)} (pág. {page_index + 1}): traduciendo...")
                page_blocks = ocr.ocr_page(doc[page_index], page_index)
                checkpoint.save(key, {"blocks": [block_to_dict(b) for b in page_blocks]})
            blocks.extend(page_blocks)

        # Repaso: a veces el OCR deja bloques en inglés (sobre todo en
        # diagramas). Se detectan y se retraducen con el traductor de texto.
        pending = [
            b for b in blocks
            if b.from_ocr and b.translatable and b.translated
            and looks_english(b.translated)
        ]
        if pending:
            print(f"  Repaso: {len(pending)} bloques quedaron en inglés; retraduciendo...")
            source = [
                Block(id=b.id, page=b.page, bbox=b.bbox, text=b.translated)
                for b in pending
            ]
            fixed = translator.translate_batch(source)
            for b in pending:
                if b.id in fixed:
                    b.translated = fixed[b.id]
            # Se actualiza el cache de las páginas corregidas.
            for page_index in {b.page for b in pending}:
                key = f"ocr_p{page_index + 1:04d}"
                page_blocks = [b for b in blocks if b.from_ocr and b.page == page_index]
                checkpoint.save(key, {"blocks": [block_to_dict(b) for b in page_blocks]})

    # Los bloques OCR ya vienen traducidos; a Gemini solo van los normales.
    batches = make_batches([b for b in blocks if not b.from_ocr], pages_per_batch)
    translations: dict[str, str] = {}

    for i, batch in enumerate(batches, 1):
        first, last = batch[0].page + 1, batch[-1].page + 1
        key = f"p{first:04d}-p{last:04d}"
        cached = checkpoint.load(key)
        if cached is not None:
            print(f"  Lote {i}/{len(batches)} (págs. {first}-{last}): cache ✓")
            translations.update(cached)
            continue
        print(
            f"  Lote {i}/{len(batches)} (págs. {first}-{last}): "
            f"traduciendo {len(batch)} bloques..."
        )
        result = translator.translate_batch(batch)
        checkpoint.save(key, result)
        translations.update(result)

    for block in blocks:
        if block.id in translations:
            block.translated = translations[block.id]

    rebuild(doc, blocks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path, garbage=3, deflate=True)
    doc.close()
    print(f"  → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pdf-translator",
        description="Traduce PDFs técnicos de inglés a español con Gemini, "
        "conservando el layout y los bloques de código intactos.",
    )
    parser.add_argument("input", type=Path, help="PDF o carpeta con PDFs")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="PDF de salida (o carpeta de salida en modo carpeta). "
        "Por defecto: <nombre>.es.pdf junto al original",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Modelo Gemini (por defecto: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pages-per-batch", type=int, default=DEFAULT_PAGES_PER_BATCH,
        help=f"Páginas por request a la API (por defecto: {DEFAULT_PAGES_PER_BATCH})",
    )
    parser.add_argument(
        "--workdir", type=Path, default=None,
        help="Carpeta para el cache de traducciones (por defecto: "
        ".translation_cache junto al PDF)",
    )
    parser.add_argument(
        "--glossary", type=Path, default=None,
        help="Archivo de glosario con términos a conservar en inglés",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        pdfs = sorted(args.input.glob("*.pdf"))
        if not pdfs:
            sys.exit(f"No hay PDFs en {args.input}")
        out_dir = args.output or args.input
        targets = [(p, out_dir / f"{p.stem}.es.pdf") for p in pdfs]
    elif args.input.is_file():
        out = args.output or args.input.with_suffix("").with_suffix(".es.pdf")
        targets = [(args.input, out)]
    else:
        sys.exit(f"No existe: {args.input}")

    glossary = load_glossary(args.glossary)
    translator = GeminiTranslator(model=args.model, glossary=glossary)

    for pdf_path, output_path in targets:
        print(f"Traduciendo {pdf_path.name} [{args.model}]")
        translate_pdf(
            pdf_path, output_path, translator, args.pages_per_batch, args.workdir
        )


if __name__ == "__main__":
    main()
