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
from pdf_translator.config import MissingApiKeyError, get_openai_api_key  # noqa: E402
from pdf_translator.extractor import Block, extract_blocks  # noqa: E402
from pdf_translator.ocr import (  # noqa: E402
    OpenAIOcr,
    OcrBlock,
    OcrPage,
    _remap_region_items,
    block_from_dict,
    block_to_dict,
    find_broken_seams,
)
from pdf_translator.rebuilder import rebuild  # noqa: E402
from pdf_translator.translator import (  # noqa: E402
    OpenAITranslator,
    StitchedFragments,
    TranslatedBlock,
    TranslationBatch,
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
    translator = OpenAITranslator.__new__(OpenAITranslator)

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


def test_openai_responses_contract():
    """Traducción y costura usan Responses con salida estructurada y sin guardar."""

    class FakeResponses:
        def __init__(self):
            self.calls = []

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["text_format"] is TranslationBatch:
                parsed = TranslationBatch(
                    translations=[
                        TranslatedBlock(id="b1", translated="Texto traducido")
                    ]
                )
            else:
                assert kwargs["text_format"] is StitchedFragments
                parsed = StitchedFragments(
                    tail="Primera parte", head="y segunda parte."
                )
            return SimpleNamespace(output_parsed=parsed, output_text="{}")

    responses = FakeResponses()
    translator = OpenAITranslator.__new__(OpenAITranslator)
    translator.client = SimpleNamespace(responses=responses)
    translator.model = "gpt-5.6-terra"
    translator.system_prompt = "Instrucciones de prueba"

    result = translator.translate_batch(
        [Block(id="b1", page=0, bbox=(0, 0, 100, 20), text="Text")]
    )
    stitched = translator.stitch_pair("Primera", "segunda")

    assert result == {"b1": "Texto traducido"}
    assert stitched == ("Primera parte", "y segunda parte.")
    assert all(call["store"] is False for call in responses.calls)
    assert all("temperature" not in call for call in responses.calls)
    assert responses.calls[0]["text_format"] is TranslationBatch
    assert responses.calls[0]["instructions"] == "Instrucciones de prueba"
    assert json.loads(responses.calls[0]["input"])[0]["id"] == "b1"
    assert responses.calls[1]["text_format"] is StitchedFragments

    print("OK: contrato de Responses API para traducción y costura")


def test_openai_api_key_configuration():
    """Acepta el nombre oficial, migra el antiguo y explica si falta la clave."""
    assert get_openai_api_key({"OPENAI_API_KEY": "oficial"}) == "oficial"
    assert get_openai_api_key({"OPEN_AI_API_KEY": "antigua"}) == "antigua"
    assert get_openai_api_key(
        {"OPENAI_API_KEY": "", "OPEN_AI_API_KEY": "antigua"}
    ) == "antigua"

    try:
        get_openai_api_key({})
    except MissingApiKeyError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("aceptó una configuración sin API key")

    print("OK: configuración y compatibilidad de OPENAI_API_KEY")


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

    class FakeResponses:
        def __init__(self):
            self.calls = []

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    output_parsed=None,
                    output_text="",
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="content_filter"),
                    error=None,
                    output=[],
                    usage=None,
                )
            parsed = OcrPage(
                blocks=[
                    OcrBlock(
                        box_2d=[100, 100, 200, 900],
                        text=f"Franja {len(self.calls) - 1}",
                        is_code=False,
                        line_count=1,
                        bold=False,
                    )
                ]
            )
            return SimpleNamespace(output_parsed=parsed, output_text="{}")

    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    doc = pymupdf.open()
    page = doc.new_page()

    blocks = OpenAIOcr(client, "gpt-5.6-terra").ocr_page(page, 0)
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
    assert len(responses.calls) == 5, "repitió la página completa en vez de recortarla"
    full_content = responses.calls[0]["input"][0]["content"]
    cropped_content = responses.calls[1]["input"][0]["content"]
    full_image = next(part for part in full_content if part["type"] == "input_image")
    cropped_image = next(
        part for part in cropped_content if part["type"] == "input_image"
    )
    assert full_image != cropped_image, "el fallback siguió enviando la página completa"
    assert responses.calls[0]["text_format"] is OcrPage
    assert responses.calls[0]["store"] is False
    assert "temperature" not in responses.calls[0]

    print("OK: fallback OCR por regiones ante respuesta vacía")


def test_ocr_table_cell_boxes_and_cache_roundtrip():
    """OCR separa el texto que se cubre del espacio disponible en la celda."""

    class FakeResponses:
        @staticmethod
        def parse(**_kwargs):
            return SimpleNamespace(
                output_parsed=OcrPage(
                    blocks=[
                        OcrBlock(
                            box_2d=[120, 120, 180, 400],
                            cell_box_2d=[100, 100, 200, 500],
                            text="Encabezado traducido",
                            is_code=False,
                            line_count=1,
                            bold=True,
                            is_table_cell=True,
                        )
                    ]
                ),
                output_text="{}",
            )

    doc = pymupdf.open()
    page = doc.new_page()
    block = OpenAIOcr(
        SimpleNamespace(responses=FakeResponses()), "gpt-5.6-terra"
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
        assert all(page.rect == pymupdf.Rect(0, 0, 595, 842) for page in result)
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
        assert all(page.rect == pymupdf.Rect(0, 0, 595, 842) for page in full_result)
        full_result.close()
        assert full_translator.calls == [[0, 1, 2, 3]]

    print("OK: traducción y recorte del rango sin afectar el modo completo")


def test_page_range_cli_clamps_and_rejects_outside_start():
    """El CLI ajusta el final, nombra la salida y falla antes de usar OpenAI."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "manual.pdf"
        make_multipage_pdf(src)

        fake = FakeTranslator()
        original_translator = cli.OpenAITranslator
        original_argv = sys.argv[:]
        try:
            cli.OpenAITranslator = lambda **_kwargs: fake
            sys.argv = ["pdf-translator", str(src), "--pages", "3-99"]
            cli.main()
        finally:
            cli.OpenAITranslator = original_translator
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


def test_paragraph_width_preserves_page_size():
    """Centra el texto normal y OCR con ancho limitado sin escalar páginas."""
    for from_ocr in (False, True):
        for width in (200, 500):
            with pymupdf.open() as doc:
                page = doc.new_page(width=595, height=842)
                bbox = (40, 40, 40 + width, 220)
                if not from_ocr:
                    page.insert_textbox(bbox, "Original en inglés " * 15)
                page.insert_text((40, 300), "def ejemplo(): pass")
                translated = "Este párrafo explica el funcionamiento del sistema. " * 12
                block = Block(
                    id="p0_b0", page=0, bbox=bbox, text="Original en inglés",
                    translated=translated, from_ocr=from_ocr,
                )
                rebuild(doc, [block])
                with pymupdf.open(stream=doc.tobytes(), filetype="pdf") as result:
                    output = result[0]
                    assert output.rect == pymupdf.Rect(0, 0, 595, 842)
                    paragraph_area = pymupdf.Rect(0, 40, 595, 220)
                    text = output.get_text(clip=paragraph_area)
                    assert " ".join(text.split()) == " ".join(translated.split())
                    assert "def ejemplo(): pass" in output.get_text()
                    words = output.get_text("words", clip=paragraph_area)
                    expected_width = min(width, 13 * 72 / 2.54)
                    expected_left = (595 - expected_width) / 2
                    # insert_htmlbox deja un margen interno mínimo.
                    assert abs(min(word[0] for word in words) - expected_left) < 2
                    assert max(word[2] for word in words) <= expected_left + expected_width + 0.001
    print("OK: párrafos centrados de hasta 13 cm con tamaño de página intacto")


def test_structured_layout_is_not_centered_or_narrowed():
    """Tablas anchas y columnas sin bordes conservan sus cajas, también en OCR."""
    from unittest.mock import patch
    from pdf_translator.rebuilder import _block_html

    for from_ocr in (False, True):
        with pymupdf.open() as doc:
            page = doc.new_page()
            blocks = [
                Block(id="tabla", page=0, bbox=(30, 40, 550, 85),
                      layout_bbox=(25, 35, 555, 90), text="Celda ancha",
                      translated="Celda ancha", is_table_cell=True, from_ocr=from_ocr),
                Block(id="izquierda", page=0, bbox=(30, 120, 160, 150),
                      text="Concepto", translated="Concepto", from_ocr=from_ocr),
                Block(id="derecha", page=0, bbox=(330, 120, 550, 150),
                      text="Explicación", translated="Explicación", from_ocr=from_ocr),
                Block(id="filas", page=0, bbox=(30, 190, 550, 260),
                      text="Nombre    Valor\nPrimero    Segundo",
                      translated="Nombre    Valor\nPrimero    Segundo", from_ocr=from_ocr),
            ]
            original_insert = pymupdf.Page.insert_htmlbox
            inserted = []

            def record_insert(target, rect, *args, **kwargs):
                inserted.append(tuple(rect))
                return original_insert(target, rect, *args, **kwargs)

            with patch.object(pymupdf.Page, "insert_htmlbox", record_insert):
                rebuild(doc, blocks)
            assert inserted == [b.layout_bbox or b.bbox for b in blocks]
            assert "white-space: pre-wrap" in _block_html(blocks[-1], True)
            assert "Nombre    Valor\nPrimero    Segundo" in _block_html(blocks[-1], True)
            assert "Explicación" in doc[0].get_text()

    # PyMuPDF puede agrupar varias columnas en un único bloque de texto.
    with pymupdf.open() as doc:
        page = doc.new_page()
        for y in (50, 80, 110):
            page.insert_text((30, y), "Concepto")
            page.insert_text((330, y), "Valor")
        from pdf_translator.rebuilder import _structured_blocks
        blocks = extract_blocks(doc)
        assert blocks
        assert _structured_blocks(page, blocks) == {b.id for b in blocks}
    print("OK: tablas y estructuras sin bordes mantienen ancho y posición")


def test_headings_do_not_overlap():
    """Encabezados cercanos y cajas OCR solapadas mantienen texto legible."""
    from pdf_translator.rebuilder import _layout_rects

    with pymupdf.open() as doc:
        page = doc.new_page()
        blocks = [
            Block(id="titulo", page=0, bbox=(40, 40, 180, 70),
                  text="Título", translated="Título principal", bold=True, font_size=18),
            Block(id="subtitulo", page=0, bbox=(184, 40, 350, 70),
                  text="Subtítulo", translated="Subtítulo cercano", bold=True, font_size=14),
            Block(id="ocr_titulo", page=0, bbox=(40, 110, 500, 142),
                  text="Encabezado OCR", translated="Encabezado OCR",
                  bold=True, font_size=18, from_ocr=True),
            Block(id="parrafo", page=0, bbox=(40, 136, 540, 250),
                  text="Este párrafo explica el contenido del documento. " * 8,
                  translated="Este párrafo explica el contenido del documento. " * 8,
                  from_ocr=True),
        ]
        rects = _layout_rects(page, blocks, set())
        assert rects["titulo"] == pymupdf.Rect(blocks[0].bbox)
        assert rects["subtitulo"] == pymupdf.Rect(blocks[1].bbox)
        assert not rects["ocr_titulo"].intersects(rects["parrafo"])
        rebuild(doc, blocks)
        with pymupdf.open(stream=doc.tobytes(), filetype="pdf") as result:
            output = result[0]
            text = " ".join(output.get_text().split())
            for block in blocks:
                assert " ".join(block.translated.split()) in text
            words = output.get_text("words")
            for index, word in enumerate(words):
                for other in words[index + 1:]:
                    if word[5] == other[5]:
                        continue  # las cajas tipográficas de líneas contiguas se cruzan
                    overlap = pymupdf.Rect(word[:4]) & pymupdf.Rect(other[:4])
                    assert overlap.is_empty, (word[4], other[4])

    # El centrado no puede invadir texto que queda en su posición original.
    with pymupdf.open() as doc:
        page = doc.new_page()
        paragraph = Block(id="p", page=0, bbox=(30, 40, 200, 170),
                          text="Contenido del párrafo. " * 10,
                          translated="Contenido del párrafo. " * 10)
        fixed = Block(id="fijo", page=0, bbox=(205, 45, 240, 60),
                      text="123", translatable=False)
        rects = _layout_rects(page, [paragraph, fixed], set())
        assert rects["p"].x0 == paragraph.bbox[0]
    print("OK: títulos y subtítulos sin solapamientos al reconstruir")


if __name__ == "__main__":
    test_headings_do_not_overlap()
    test_structured_layout_is_not_centered_or_narrowed()
    test_paragraph_width_preserves_page_size()
    test_pipeline_preserves_code()
    test_table_structure_is_preserved()
    test_table_metadata_reaches_translation_prompt()
    test_openai_responses_contract()
    test_openai_api_key_configuration()
    test_seam_detection()
    test_ocr_empty_response_uses_regions()
    test_ocr_table_cell_boxes_and_cache_roundtrip()
    test_existing_batch_cache_translates_only_new_table_cells()
    test_page_range_parsing_and_resolution()
    test_partial_translation_keeps_only_selected_pages()
    test_page_range_cli_clamps_and_rejects_outside_start()
