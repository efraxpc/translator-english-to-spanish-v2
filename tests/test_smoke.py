"""Smoke test sin API: genera un PDF de muestra, extrae, 'traduce' con un fake
y reconstruye. Verifica que el bloque de código quede intacto.

Ejecutar: python -m pytest tests/ -v   (o simplemente: python tests/test_smoke.py)
"""

import sys
import tempfile
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdf_translator.extractor import extract_blocks  # noqa: E402
from pdf_translator.rebuilder import rebuild  # noqa: E402

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


if __name__ == "__main__":
    test_pipeline_preserves_code()
