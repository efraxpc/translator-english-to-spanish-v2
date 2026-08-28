"""Smoke test sin API: genera un PDF de muestra, extrae, 'traduce' con un fake
y reconstruye. Verifica que el bloque de código quede intacto.

Ejecutar: python -m pytest tests/ -v   (o simplemente: python tests/test_smoke.py)
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdf_translator import __main__ as cli  # noqa: E402
from pdf_translator.checkpoint import Checkpoint  # noqa: E402
from pdf_translator.extractor import Block, extract_blocks  # noqa: E402
from pdf_translator.ocr import (  # noqa: E402
    GeminiOcr,
    OcrBlock,
    _remap_region_items,
    block_from_dict,
    block_to_dict,
    find_broken_seams,
)
from pdf_translator.rebuilder import rebuild  # noqa: E402
from pdf_translator.translator import (  # noqa: E402
    GeminiTranslator,
    TranslatedBlock,
    build_system_prompt,
)

CODE_SNIPPET = (
    "def train_model(dataset):\n"
    "    model = Transformer(layers=12)\n"
    "    for epoch in range(10):\n"
    "        loss = model.fit(dataset)\n"
    "    return model"
)

PARAGRAPH = (
    "Machine learning models require careful feature engineering and a robust "
    "deployment pipeline. Google Cloud offers Vertex AI for training and "
    "serving models at scale."
)

FAKE_TRANSLATION = (
    "Los modelos de Machine Learning requieren un feature engineering "
    "cuidadoso y un pipeline de deployment robusto. Google Cloud ofrece "
    "Vertex AI para entrenar y servir modelos a escala."
)


class FakeTranslator:
    """Traductor offline que registra las páginas recibidas."""

    model = "modelo-falso"
    glossary: list[str] = []
    parafrasear = False
    client = None

    def __init__(self):
        self.calls: list[list[int]] = []
        self.block_ids: list[list[str]] = []

    def translate_batch(self, blocks):
        self.calls.append([block.page for block in blocks])
        self.block_ids.append([block.id for block in blocks])
        return {
            block.id: f"Contenido traducido de la página {block.page + 1}."
            for block in blocks
        }


def make_sample_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Chapter 1: Introduction to ML Engineering",
                     fontsize=18, fontname="helv")
    rect = pymupdf.Rect(72, 110, 540, 200)
    page.insert_textbox(rect, PARAGRAPH, fontsize=11, fontname="helv")
    code_rect = pymupdf.Rect(72, 220, 540, 320)
    page.insert_textbox(code_rect, CODE_SNIPPET, fontsize=10, fontname="cour")
    doc.save(path)
    doc.close()


def make_multipage_pdf(path: Path, page_count: int = 4) -> None:
    doc = pymupdf.open()
    for page_number in range(1, page_count + 1):
        page = doc.new_page()
        page.insert_text(
            (72, 80),
            f"English content from physical page {page_number} to translate.",
            fontsize=11,
            fontname="helv",
        )
    doc.save(path)
    doc.close()


def make_table_pdf(path: Path, with_title: bool = False) -> None:
    """Genera una tabla con encabezado combinado y cuadrícula vectorial."""
    doc = pymupdf.open()
    page = doc.new_page()
    if with_title:
        page.insert_text(
            (72, 70), "Model evaluation table", fontsize=12, fontname="helv"
        )

    xs = [72, 220, 400]
    ys = [100, 130, 160, 190]
    for x in (xs[0], xs[-1]):
        page.draw_line((x, ys[0]), (x, ys[-1]))
    page.draw_line((xs[1], ys[1]), (xs[1], ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))

    values = [
        (76, 120, "Combined heading"),
        (76, 150, "Accuracy"),
        (224, 150, "Correct predictions"),
        (76, 180, "Recall"),
        (224, 180, "Found positives"),
    ]
    for x, y, value in values:
        page.insert_text((x, y), value, fontsize=9, fontname="helv")
    doc.save(path)
    doc.close()


def test_pipeline_preserves_code():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sample.pdf"
        out = Path(tmp) / "sample.es.pdf"
        make_sample_pdf(src)

        doc = pymupdf.open(src)
        blocks = extract_blocks(doc)
        assert blocks, "no se extrajo ningún bloque"

        code_blocks = [b for b in blocks if b.is_code]
        assert code_blocks, "el bloque Courier no se detectó como código"
        assert "def train_model(dataset):" in code_blocks[0].text
        assert not code_blocks[0].translatable

        para = next(b for b in blocks if "feature engineering" in b.text)
        assert para.translatable
        para.translated = FAKE_TRANSLATION

        rebuild(doc, blocks)
        doc.save(out)
        doc.close()

        result = pymupdf.open(out)
        text = result[0].get_text()
        result.close()

        # El código sigue íntegro, línea por línea con su indentación
        for line in CODE_SNIPPET.splitlines():
            assert line.strip() in text, f"línea de código perdida: {line!r}"
        # La traducción está presente y el original ya no
        assert "modelos de Machine Learning" in text
        assert "require careful feature engineering" not in text

    print("OK: código intacto, párrafo traducido, layout reconstruido")


def test_table_structure_is_preserved():
    """Traduce cada celda sin alterar filas, columnas ni celdas combinadas."""
    translations = {
        "Combined heading": "Encabezado combinado",
        "Accuracy": "Exactitud",
        "Correct predictions": "Predicciones correctas",
        "Recall": "Exhaustividad",
        "Found positives": "Positivos encontrados",
    }
    expected = [
        ["Encabezado combinado", None],
        ["Exactitud", "Predicciones correctas"],
        ["Exhaustividad", "Positivos encontrados"],
    ]

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "table.pdf"
        out = Path(tmp) / "table.es.pdf"
        make_table_pdf(src)

        doc = pymupdf.open(src)
        original_table = doc[0].find_tables().tables[0]
        original_bbox = original_table.bbox
        original_cells = [row.cells for row in original_table.rows]
        blocks = extract_blocks(doc)
        table_blocks = [block for block in blocks if block.is_table_cell]

        assert len(table_blocks) == 5
        assert len(table_blocks) == len(blocks), "duplicó filas como bloques normales"
        assert {block.text for block in table_blocks} == set(translations)
        assert all(block.layout_bbox is not None for block in table_blocks)
        for block in table_blocks:
            block.translated = translations[block.text]

        rebuild(doc, blocks)
        doc.save(out)
        doc.close()

        result = pymupdf.open(out)
        tables = result[0].find_tables().tables
        assert len(tables) == 1
        translated_table = tables[0]
        assert translated_table.bbox == original_bbox
        assert [row.cells for row in translated_table.rows] == original_cells
        assert translated_table.extract() == expected
        text = result[0].get_text()
        result.close()
        assert all(source not in text for source in translations)

    print("OK: estructura y celdas combinadas de tabla conservadas")


def test_table_metadata_reaches_translation_prompt():
    """El traductor identifica celdas sin cambiar su esquema de respuesta."""
    captured = {}
    translator = GeminiTranslator.__new__(GeminiTranslator)

    def fake_call(payload):
        captured.update(json.loads(payload)[0])
        return [TranslatedBlock(id="cell", translated="Valor traducido")]

    translator._call_with_retries = fake_call
    result = translator.translate_batch(
        [
            Block(
                id="cell",
                page=0,
                bbox=(0, 0, 100, 20),
                text="Value",
                is_table_cell=True,
            )
        ]
    )

    assert captured["is_table_cell"] is True
    assert result == {"cell": "Valor traducido"}
    prompt = build_system_prompt([], parafrasear=True)
    assert "is_table_cell=true" in prompt
    assert "Modo parafraseo" in prompt

    print("OK: metadatos de tabla enviados sin anular el parafraseo")


def test_seam_detection():
    """La heurística de costura marca uniones rotas y respeta las limpias."""

    def blk(
        page,
        y,
        text,
        from_ocr=True,
        is_code=False,
        translated=None,
        is_table_cell=False,
    ):
        return Block(
            id=f"p{page}_y{y}",
            page=page,
            bbox=(80.0, float(y), 500.0, float(y) + 15),
            text=text,
            is_code=is_code,
            translatable=not is_code,
            translated=text if translated is None else translated,
            from_ocr=from_ocr,
            is_table_cell=is_table_cell,
        )

    # Costura rota real (p42→43 del libro): cola a media oración y cabeza
    # que reinicia con mayúscula.
    tail = blk(41, 700, "Cuando se usa un consenso incorrecto sobre el patrón "
                        "de datos asociado con cualquier etiqueta objetivo para el")
    head = blk(42, 30, "Si el proceso de etiquetado es defectuoso, el problema "
                       "no se notará hasta fases avanzadas")
    # Unión limpia: punto final + mayúscula inicial.
    clean_tail = blk(43, 700, "La oración anterior terminó correctamente.")
    clean_head = blk(44, 30, "Nuevo párrafo que empieza la página.")
    # Inversión sospechosa: punto final pero la página siguiente arranca en
    # minúscula.
    inv_tail = blk(45, 700, "Un servicio de inference en múltiples dispositivos.")
    inv_head = blk(46, 30, "dispositivos en paralelo")

    blocks = [tail, head, clean_tail, clean_head, inv_tail, inv_head]
    seams = find_broken_seams(blocks)

    assert (tail, head) in seams, "no detectó la costura rota colgada"
    assert (inv_tail, inv_head) in seams, "no detectó la inversión punto+minúscula"
    assert (clean_tail, clean_head) not in seams, "falso positivo en unión limpia"

    # Los bloques de código o sin traducir no participan en costuras.
    code_tail = blk(47, 700, "def foo(x):", is_code=True)
    ocr_head = blk(48, 30, "texto al inicio de página")
    plain = blk(48, 700, "bloque sin traducir", from_ocr=False)
    assert find_broken_seams([code_tail, ocr_head, plain]) == []

    table_tail = blk(49, 700, "celda sin punto", is_table_cell=True)
    next_head = blk(50, 30, "texto de la página siguiente")
    assert find_broken_seams([table_tail, next_head]) == []

    print("OK: detección de costuras rotas en saltos de página")


def test_ocr_empty_response_uses_regions():
    """Una respuesta vacía se subdivide sin repetir la misma llamada."""

    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    parsed=None,
                    text=None,
                    candidates=[
                        SimpleNamespace(
                            finish_reason="RECITATION",
                            finish_message="respuesta detenida",
                        )
                    ],
                    prompt_feedback=None,
                    usage_metadata=None,
                )
            parsed = [
                OcrBlock(
                    box_2d=[100, 100, 200, 900],
                    text=f"Franja {len(self.calls) - 1}",
                    is_code=False,
                    line_count=1,
                    bold=False,
                )
            ]
            return SimpleNamespace(parsed=parsed, text="[]")

    models = FakeModels()
    client = SimpleNamespace(models=models)
    doc = pymupdf.open()
    page = doc.new_page()

    blocks = GeminiOcr(client, "gemini-3.1-pro-preview").ocr_page(page, 0)
    page_height = page.rect.height

    doc.close()
    assert [b.translated for b in blocks] == [
        "Franja 1",
        "Franja 2",
        "Franja 3",
        "Franja 4",
    ]
    assert [round(b.bbox[1] / page_height * 1000) for b in blocks] == [
        25,
        275,
        525,
        775,
    ], "remapeó mal las coordenadas verticales de los recortes"
    assert len(models.calls) == 5, "repitió la página completa en vez de recortarla"
    full_image = models.calls[0]["contents"][0].inline_data.data
    cropped_image = models.calls[1]["contents"][0].inline_data.data
    assert full_image != cropped_image, "el fallback siguió enviando la página completa"
    config = models.calls[0]["config"]
    assert config.thinking_config.thinking_level.value == "LOW"

    print("OK: fallback OCR por regiones ante respuesta vacía")


def test_ocr_table_cell_boxes_and_cache_roundtrip():
    """OCR separa el texto que se cubre del espacio disponible en la celda."""

    class FakeModels:
        @staticmethod
        def generate_content(**_kwargs):
            return SimpleNamespace(
                parsed=[
                    OcrBlock(
                        box_2d=[120, 120, 180, 400],
                        cell_box_2d=[100, 100, 200, 500],
                        text="Encabezado traducido",
                        is_code=False,
                        line_count=1,
                        bold=True,
                        is_table_cell=True,
                    )
                ],
                text="[]",
            )

    doc = pymupdf.open()
    page = doc.new_page()
    block = GeminiOcr(
        SimpleNamespace(models=FakeModels()), "gemini-3.1-pro-preview"
    ).ocr_page(page, 0)[0]
    assert block.is_table_cell
    assert block.layout_bbox is not None
    assert pymupdf.Rect(block.layout_bbox).contains(pymupdf.Rect(block.bbox))

    restored = block_from_dict(block_to_dict(block))
    assert restored.is_table_cell
    assert restored.layout_bbox == block.layout_bbox

    remapped = _remap_region_items(
        [
            OcrBlock(
                box_2d=[100, 100, 200, 400],
                cell_box_2d=[50, 80, 250, 500],
                text="Celda",
                is_code=False,
                line_count=1,
                bold=False,
                is_table_cell=True,
            )
        ],
        250,
        500,
    )[0]
    doc.close()
    assert remapped.box_2d == [275, 100, 300, 400]
    assert remapped.cell_box_2d == [262, 80, 312, 500]

    print("OK: cajas OCR de texto y celda conservadas en cache y recortes")


def test_existing_batch_cache_translates_only_new_table_cells():
    """Un cache anterior se completa sin volver a traducir bloques presentes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "cached-table.pdf"
        out = tmp_path / "cached-table.es.pdf"
        workdir = tmp_path / "cache"
        make_table_pdf(src, with_title=True)

        source_doc = pymupdf.open(src)
        source_blocks = extract_blocks(source_doc)
        source_doc.close()
        title = next(block for block in source_blocks if not block.is_table_cell)
        table_ids = {
            block.id for block in source_blocks if block.is_table_cell
        }
        Checkpoint(src, "modelo-falso", workdir).save(
            "p0001-p0001", {title.id: "Tabla de evaluación del modelo"}
        )

        translator = FakeTranslator()
        cli.translate_pdf(src, out, translator, 5, workdir)

        assert len(translator.block_ids) == 1
        assert set(translator.block_ids[0]) == table_ids
        result = pymupdf.open(out)
        assert "Tabla de evaluación del modelo" in result[0].get_text()
        result.close()

    print("OK: cache previo completado solo con las nuevas celdas")


def test_page_range_parsing_and_resolution():
    """Valida la sintaxis 1-based, el ajuste final y el nombre de salida."""
    assert cli.parse_page_range("10-25") == (10, 25)
    assert cli.parse_page_range(" 2 - 7 ") == (2, 7)
    assert cli.resolve_page_range((3, 99), 4) == (3, 4)
    assert cli.resolve_page_range((5, 8), 4) is None
    assert cli.ranged_output_name(Path("libro.pdf"), (3, 4)) == (
        "libro.p3-4.es.pdf"
    )

    for invalid in ("", "3", "a-b", "0-2", "3-2", "1:2", "-1-2"):
        try:
            cli.parse_page_range(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"aceptó un rango inválido: {invalid!r}")

    print("OK: validación y ajuste del rango de páginas")


def test_partial_translation_keeps_only_selected_pages():
    """Solo traduce y conserva las páginas físicas del rango inclusivo."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "book.pdf"
        out = tmp_path / "book.p2-3.es.pdf"
        make_multipage_pdf(src)

        translator = FakeTranslator()
        cli.translate_pdf(
            src,
            out,
            translator,
            pages_per_batch=5,
            workdir=tmp_path / "cache-parcial",
            page_range=(2, 3),
        )

        assert translator.calls == [[1, 2]], (
            "envió a traducir bloques fuera del rango solicitado"
        )
        result = pymupdf.open(out)
        texts = [page.get_text() for page in result]
        result.close()
        assert len(texts) == 2
        assert "página 2" in texts[0]
        assert "página 3" in texts[1]
        assert all("English content" not in text for text in texts)

        # Sin rango se mantiene el comportamiento previo: sale el PDF entero.
        full_out = tmp_path / "book.es.pdf"
        full_translator = FakeTranslator()
        cli.translate_pdf(
            src,
            full_out,
            full_translator,
            pages_per_batch=5,
            workdir=tmp_path / "cache-completo",
        )
        full_result = pymupdf.open(full_out)
        assert len(full_result) == 4
        full_result.close()
        assert full_translator.calls == [[0, 1, 2, 3]]

    print("OK: traducción y recorte del rango sin afectar el modo completo")


def test_page_range_cli_clamps_and_rejects_outside_start():
    """El CLI ajusta el final, nombra la salida y falla antes de usar Gemini."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "manual.pdf"
        make_multipage_pdf(src)

        fake = FakeTranslator()
        original_translator = cli.GeminiTranslator
        original_argv = sys.argv[:]
        try:
            cli.GeminiTranslator = lambda **_kwargs: fake
            sys.argv = ["pdf-translator", str(src), "--pages", "3-99"]
            cli.main()
        finally:
            cli.GeminiTranslator = original_translator
            sys.argv = original_argv

        clamped_out = tmp_path / "manual.p3-4.es.pdf"
        assert clamped_out.exists(), "no usó el rango efectivo en el nombre"
        clamped_doc = pymupdf.open(clamped_out)
        assert len(clamped_doc) == 2
        clamped_doc.close()
        assert fake.calls == [[2, 3]]

        outside_out = tmp_path / "no-debe-existir.pdf"
        original_argv = sys.argv[:]
        try:
            sys.argv = [
                "pdf-translator",
                str(src),
                "--pages",
                "5-8",
                "-o",
                str(outside_out),
            ]
            try:
                cli.main()
            except SystemExit as exc:
                assert "solo tiene 4 páginas" in str(exc)
            else:
                raise AssertionError("el CLI aceptó un inicio fuera del documento")
        finally:
            sys.argv = original_argv
        assert not outside_out.exists()

    print("OK: ajuste y rechazo de rangos en el CLI")


if __name__ == "__main__":
    test_pipeline_preserves_code()
    test_table_structure_is_preserved()
    test_table_metadata_reaches_translation_prompt()
    test_seam_detection()
    test_ocr_empty_response_uses_regions()
    test_ocr_table_cell_boxes_and_cache_roundtrip()
    test_existing_batch_cache_translates_only_new_table_cells()
    test_page_range_parsing_and_resolution()
    test_partial_translation_keeps_only_selected_pages()
    test_page_range_cli_clamps_and_rejects_outside_start()
