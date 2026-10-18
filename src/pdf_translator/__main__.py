"""CLI: traduce PDFs de inglés a español conservando el layout original."""

import argparse
import re
import sys
from pathlib import Path

import pymupdf

from .checkpoint import Checkpoint
from .config import (
    DEFAULT_MODEL,
    DEFAULT_PAGES_PER_BATCH,
    MissingApiKeyError,
    load_glossary,
)
from .extractor import Block, extract_blocks
from .ocr import (
    OpenAIOcr,
    block_from_dict,
    block_to_dict,
    find_broken_seams,
    looks_english,
)
from .rebuilder import rebuild
from .translator import OpenAITranslator

PageRange = tuple[int, int]

_PAGE_RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
_OCR_TABLE_CACHE_VERSION = "table_v1"


def parse_page_range(value: str) -> PageRange:
    """Convierte ``INICIO-FIN`` a un rango inclusivo numerado desde 1."""
    match = _PAGE_RANGE_RE.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "usa el formato INICIO-FIN, por ejemplo: --pages 10-25"
        )
    start, end = (int(part) for part in match.groups())
    if start < 1 or end < 1:
        raise argparse.ArgumentTypeError("las páginas deben ser mayores que cero")
    if start > end:
        raise argparse.ArgumentTypeError(
            "la página inicial no puede ser mayor que la final"
        )
    return start, end


def resolve_page_range(
    requested: PageRange,
    total_pages: int,
) -> PageRange | None:
    """Ajusta el final al documento o devuelve ``None`` si no hay solapamiento."""
    start, end = requested
    if start > total_pages:
        return None
    return start, min(end, total_pages)


def ranged_output_name(pdf_path: Path, page_range: PageRange) -> str:
    """Nombre automático de una traducción parcial."""
    start, end = page_range
    return f"{pdf_path.stem}.p{start}-{end}.es.pdf"


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
    translator: OpenAITranslator,
    pages_per_batch: int,
    workdir: Path | None,
    page_range: PageRange | None = None,
) -> None:
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    if page_range is None:
        page_indexes = list(range(total_pages))
    else:
        start, end = page_range
        if start < 1 or start > end or end > total_pages:
            doc.close()
            raise ValueError(
                f"Rango de páginas inválido para un PDF de {total_pages} páginas: "
                f"{start}-{end}"
            )
        page_indexes = list(range(start - 1, end))

    blocks = extract_blocks(doc, page_indexes)
    translatable = [b for b in blocks if b.translatable]
    code_blocks = sum(1 for b in blocks if b.is_code)
    if page_range is None:
        pages_description = f"{total_pages} páginas"
    else:
        start, end = page_range
        pages_description = (
            f"{len(page_indexes)} páginas seleccionadas ({start}-{end}) "
            f"de {total_pages}"
        )
    print(
        f"  {pages_description}, {len(blocks)} bloques "
        f"({len(translatable)} a traducir, {code_blocks} de código conservados)"
    )

    # El cache distingue el modo parafraseo para no mezclar ambos estilos.
    cache_model = translator.model + ("-parafraseo" if translator.parafrasear else "")
    checkpoint = Checkpoint(pdf_path, cache_model, workdir)

    # Páginas sin texto extraíble (texto vectorizado/escaneado) → OCR con OpenAI.
    pages_with_text = {b.page for b in blocks}
    ocr_pages = []
    for page_index in page_indexes:
        page = doc[page_index]
        has_content = (
            page.get_drawings() or page.get_images() or page.get_text().strip()
        )
        if page_index not in pages_with_text and has_content:
            ocr_pages.append(page_index)
    if ocr_pages:
        print(
            f"  {len(ocr_pages)} páginas sin texto extraíble: "
            "usando OCR con OpenAI (texto vectorizado o escaneado)"
        )
        ocr = OpenAIOcr(
            translator.client, translator.model, translator.glossary,
            parafrasear=translator.parafrasear,
        )
        for n, page_index in enumerate(ocr_pages, 1):
            key = f"ocr_{_OCR_TABLE_CACHE_VERSION}_p{page_index + 1:04d}"
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
                Block(
                    id=b.id,
                    page=b.page,
                    bbox=b.bbox,
                    text=b.translated,
                    is_table_cell=b.is_table_cell,
                )
                for b in pending
            ]
            fixed = translator.translate_batch(source)
            for b in pending:
                if b.id in fixed:
                    b.translated = fixed[b.id]
            # Se actualiza el cache de las páginas corregidas.
            for page_index in {b.page for b in pending}:
                key = f"ocr_{_OCR_TABLE_CACHE_VERSION}_p{page_index + 1:04d}"
                page_blocks = [b for b in blocks if b.from_ocr and b.page == page_index]
                checkpoint.save(key, {"blocks": [block_to_dict(b) for b in page_blocks]})

        # Costura: las oraciones que cruzan un salto de página se tradujeron
        # por separado (una imagen por llamada) y pueden no encajar: la cola
        # queda colgando y la cabeza reinicia la oración (sobre todo con
        # parafraseo). Se reenvían los pares sospechosos para reescribirlos.
        seams = find_broken_seams(blocks)
        if seams:
            print(
                f"  Costura: {len(seams)} saltos de página sospechosos; "
                "revisando..."
            )
            fixed = 0
            for tail, head in seams:
                key = (
                    f"seam_{_OCR_TABLE_CACHE_VERSION}_"
                    f"p{tail.page + 1:04d}-p{head.page + 1:04d}"
                )
                cached = checkpoint.load(key)
                if cached is not None:
                    tail.translated = cached["tail"]
                    head.translated = cached["head"]
                    continue
                new_tail, new_head = translator.stitch_pair(
                    tail.translated, head.translated
                )
                if new_tail.strip():
                    tail.translated = new_tail
                if new_head.strip():
                    head.translated = new_head
                checkpoint.save(
                    key, {"tail": tail.translated, "head": head.translated}
                )
                fixed += 1
            if fixed:
                print(f"  Costura: {fixed} pares reescritos")
                # Se actualiza el cache de las páginas corregidas.
                for page_index in {b.page for pair in seams for b in pair}:
                    key = (
                        f"ocr_{_OCR_TABLE_CACHE_VERSION}_"
                        f"p{page_index + 1:04d}"
                    )
                    page_blocks = [
                        b for b in blocks if b.from_ocr and b.page == page_index
                    ]
                    checkpoint.save(
                        key, {"blocks": [block_to_dict(b) for b in page_blocks]}
                    )

    # Los bloques OCR ya vienen traducidos; a OpenAI solo van los normales.
    batches = make_batches([b for b in blocks if not b.from_ocr], pages_per_batch)
    translations: dict[str, str] = {}

    for i, batch in enumerate(batches, 1):
        first, last = batch[0].page + 1, batch[-1].page + 1
        key = f"p{first:04d}-p{last:04d}"
        cached = checkpoint.load(key)
        missing = batch
        if cached is not None:
            missing = [block for block in batch if block.id not in cached]
        if cached is not None and not missing:
            print(f"  Lote {i}/{len(batches)} (págs. {first}-{last}): cache ✓")
            translations.update({block.id: cached[block.id] for block in batch})
            continue
        cache_note = " faltantes" if cached is not None else ""
        print(
            f"  Lote {i}/{len(batches)} (págs. {first}-{last}): "
            f"traduciendo {len(missing)} bloques{cache_note}..."
        )
        result = translator.translate_batch(missing)
        combined = dict(cached or {})
        combined.update(result)
        checkpoint.save(key, combined)
        translations.update(
            {block.id: combined[block.id] for block in batch if block.id in combined}
        )

    for block in blocks:
        if block.id in translations:
            block.translated = translations[block.id]

    rebuild(doc, blocks)
    if page_range is not None:
        # Se recorta al final para que los bloques y el cache mantengan los
        # índices absolutos del PDF original durante todo el pipeline.
        doc.select(page_indexes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path, garbage=3, deflate=True)
    doc.close()
    print(f"  → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pdf-translator",
        description="Traduce PDFs técnicos de inglés a español con OpenAI, "
        "conservando el layout y los bloques de código intactos.",
    )
    parser.add_argument("input", type=Path, help="PDF o carpeta con PDFs")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="PDF de salida (o carpeta de salida en modo carpeta). "
        "Por defecto: <nombre>.es.pdf junto al original; con --pages: "
        "<nombre>.pINICIO-FIN.es.pdf",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Modelo OpenAI (por defecto: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pages-per-batch", type=int, default=DEFAULT_PAGES_PER_BATCH,
        help=f"Páginas por request a la API (por defecto: {DEFAULT_PAGES_PER_BATCH})",
    )
    parser.add_argument(
        "--pages", type=parse_page_range, metavar="INICIO-FIN", default=None,
        help="Rango inclusivo de páginas físicas a traducir, numeradas desde 1. "
        "La salida contiene solo las páginas seleccionadas",
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
    parser.add_argument(
        "--parafrasear", action="store_true",
        help="Parafrasea los párrafos en español claro y sencillo en vez de "
        "traducirlos literalmente (títulos y viñetas se traducen directos)",
    )
    args = parser.parse_args()

    targets: list[tuple[Path, Path, PageRange | None]] = []
    if args.input.is_dir():
        pdfs = sorted(args.input.glob("*.pdf"))
        if not pdfs:
            sys.exit(f"No hay PDFs en {args.input}")
        out_dir = args.output or args.input
        for pdf_path in pdfs:
            effective_range = None
            if args.pages is not None:
                with pymupdf.open(pdf_path) as doc:
                    total_pages = len(doc)
                effective_range = resolve_page_range(args.pages, total_pages)
                if effective_range is None:
                    print(
                        f"Omitiendo {pdf_path.name}: el rango empieza en la "
                        f"página {args.pages[0]}, pero el PDF solo tiene "
                        f"{total_pages} páginas",
                        file=sys.stderr,
                    )
                    continue
                if effective_range != args.pages:
                    print(
                        f"Aviso: {pdf_path.name} solo tiene {total_pages} páginas; "
                        f"el rango {args.pages[0]}-{args.pages[1]} se ajusta a "
                        f"{effective_range[0]}-{effective_range[1]}",
                        file=sys.stderr,
                    )
                output_name = ranged_output_name(pdf_path, effective_range)
            else:
                output_name = f"{pdf_path.stem}.es.pdf"
            targets.append((pdf_path, out_dir / output_name, effective_range))
        if not targets:
            sys.exit(
                f"Ningún PDF contiene la página inicial {args.pages[0]}"
            )
    elif args.input.is_file():
        effective_range = None
        if args.pages is not None:
            with pymupdf.open(args.input) as doc:
                total_pages = len(doc)
            effective_range = resolve_page_range(args.pages, total_pages)
            if effective_range is None:
                sys.exit(
                    f"El rango empieza en la página {args.pages[0]}, pero "
                    f"{args.input.name} solo tiene {total_pages} páginas"
                )
            if effective_range != args.pages:
                print(
                    f"Aviso: {args.input.name} solo tiene {total_pages} páginas; "
                    f"el rango {args.pages[0]}-{args.pages[1]} se ajusta a "
                    f"{effective_range[0]}-{effective_range[1]}",
                    file=sys.stderr,
                )
            default_out = args.input.with_name(
                ranged_output_name(args.input, effective_range)
            )
        else:
            default_out = args.input.with_suffix("").with_suffix(".es.pdf")
        out = args.output or default_out
        targets = [(args.input, out, effective_range)]
    else:
        sys.exit(f"No existe: {args.input}")

    glossary = load_glossary(args.glossary)
    try:
        translator = OpenAITranslator(
            model=args.model, glossary=glossary, parafrasear=args.parafrasear
        )
    except MissingApiKeyError as exc:
        sys.exit(f"Error de configuración: {exc}")

    for pdf_path, output_path, page_range in targets:
        modo = " [parafraseo]" if args.parafrasear else ""
        rango = f" [págs. {page_range[0]}-{page_range[1]}]" if page_range else ""
        print(f"Traduciendo {pdf_path.name} [{args.model}]{modo}{rango}")
        translate_pdf(
            pdf_path,
            output_path,
            translator,
            args.pages_per_batch,
            args.workdir,
            page_range,
        )


if __name__ == "__main__":
    main()
