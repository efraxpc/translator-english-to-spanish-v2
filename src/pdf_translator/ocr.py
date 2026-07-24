"""OCR con Gemini para páginas sin texto extraíble (texto convertido a curvas).

Algunos PDFs no contienen texto real: los glifos están vectorizados como
trazos, sin fuentes ni operadores de texto, y PyMuPDF no puede extraer nada.
Para esas páginas se renderiza la página a imagen y se le pide a Gemini
(multimodal) que detecte los bloques de texto con sus coordenadas y los
traduzca en una sola llamada.
"""

import re
import time

import pymupdf
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from .extractor import Block

_ZOOM = 2.0  # resolución del render enviado a Gemini
_MAX_RETRIES = 5
_RETRYABLE_CODES = {429, 500, 502, 503, 504}


class OcrBlock(BaseModel):
    box_2d: list[int]  # [ymin, xmin, ymax, xmax] normalizado a 0-1000
    text: str
    is_code: bool
    line_count: int
    bold: bool


def build_ocr_prompt(glossary: list[str]) -> str:
    glossary_section = ""
    if glossary:
        glossary_section = (
            "\n\nGlosario de términos que SIEMPRE se conservan en inglés "
            "(además de las reglas anteriores):\n" + ", ".join(glossary)
        )
    return (
        "La imagen es una página de un libro técnico en inglés sobre IA, "
        "Machine Learning, ingeniería de software y Google Cloud Platform.\n\n"
        "Detecta TODOS los bloques de texto de la página (títulos, párrafos, "
        "viñetas, pies de figura, cabeceras, números de página...). Para cada "
        "bloque devuelve:\n"
        "- box_2d: [ymin, xmin, ymax, xmax] normalizado a 0-1000, ajustado al "
        "bloque completo.\n"
        "- text: la traducción del bloque a español neutro y profesional. Si "
        "el bloque es código, devuelve el texto original sin cambios.\n"
        "- is_code: true si el bloque es un listado de código, comando de "
        "terminal o salida de consola.\n"
        "- line_count: número de líneas de texto del bloque original.\n"
        "- bold: true si el bloque está mayoritariamente en negrita.\n\n"
        "Reglas de traducción estrictas:\n"
        "1. CONSERVA EN INGLÉS los términos técnicos de IA/ML/software/GCP: "
        "nombres de productos (Cloud Run, BigQuery, Vertex AI...), conceptos "
        "establecidos (embedding, transformer, pipeline, deployment, "
        "fine-tuning...), siglas (API, SDK, IAM, CI/CD...).\n"
        "2. NO modifiques nada que parezca código inline: `df.groupby()`, "
        "comandos, variables, rutas, URLs. Cópialos carácter por carácter.\n"
        "3. Conserva la puntuación estructural (viñetas, numeración...).\n"
        "4. Si un bloque no necesita traducción (número de página, nombre "
        "propio), devuélvelo sin cambios.\n"
        "5. Traduce TAMBIÉN las etiquetas de diagramas y flowcharts "
        "(YES → SÍ, NO → NO) y los pies de figura (Figure 1.1: → Figura 1.1:). "
        "Ningún texto en inglés debe quedar sin traducir salvo código y "
        "términos del glosario.\n"
        "6. No agregues comentarios: solo los bloques detectados."
        + glossary_section
    )


class GeminiOcr:
    def __init__(self, client: genai.Client, model: str, glossary: list[str] | None = None):
        self.client = client
        self.model = model
        self.prompt = build_ocr_prompt(glossary or [])

    def ocr_page(self, page: pymupdf.Page, page_index: int) -> list[Block]:
        """Renderiza la página, la manda a Gemini y devuelve bloques traducidos."""
        pix = page.get_pixmap(matrix=pymupdf.Matrix(_ZOOM, _ZOOM))
        png = pix.tobytes("png")
        items = self._call_with_retries(png)

        width, height = page.rect.width, page.rect.height
        blocks: list[Block] = []
        for i, item in enumerate(items):
            if len(item.box_2d) != 4 or not item.text.strip():
                continue
            ymin, xmin, ymax, xmax = (max(0, min(1000, v)) for v in item.box_2d)
            if ymax <= ymin or xmax <= xmin:
                continue
            bbox = (
                xmin / 1000 * width,
                ymin / 1000 * height,
                xmax / 1000 * width,
                ymax / 1000 * height,
            )
            lines = max(item.line_count, 1)
            font_size = min(max((bbox[3] - bbox[1]) / lines / 1.3, 6.0), 40.0)
            blocks.append(
                Block(
                    id=f"p{page_index}_ocr{i}",
                    page=page_index,
                    bbox=bbox,
                    text=item.text,
                    font_size=font_size,
                    bold=item.bold,
                    is_code=item.is_code,
                    # El código no se toca: el original vectorial queda visible.
                    translatable=not item.is_code,
                    translated=None if item.is_code else item.text,
                    from_ocr=True,
                )
            )
        return blocks

    def _call_with_retries(self, png: bytes) -> list[OcrBlock]:
        delay = 5.0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=png, mime_type="image/png"),
                        self.prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[OcrBlock],
                        temperature=0.1,
                    ),
                )
                parsed = response.parsed
                if parsed is None:
                    raise ValueError(f"Respuesta sin JSON parseable: {response.text!r:.200}")
                return parsed
            except (errors.APIError, ValueError) as exc:
                code = getattr(exc, "code", None)
                retryable = code in _RETRYABLE_CODES or isinstance(exc, ValueError)
                if retryable and attempt < _MAX_RETRIES:
                    print(
                        f"    OCR error ({exc.__class__.__name__}); "
                        f"reintento {attempt}/{_MAX_RETRIES} en {delay:.0f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")


# Palabras que casi nunca aparecen en español: si están, el bloque quedó en inglés.
_ENGLISH_RE = re.compile(
    r"\b(the|and|you|your|which|with|are|do you|does|need|already|"
    r"figure|this|that|from|want|is your)\b",
    re.IGNORECASE,
)


def looks_english(text: str) -> bool:
    """Heurística: ¿este texto parece haber quedado sin traducir?"""
    if text.strip().lower() in {"yes", "yes/no"}:
        return True
    return bool(_ENGLISH_RE.search(text))


def block_to_dict(block: Block) -> dict:
    return {
        "id": block.id,
        "page": block.page,
        "bbox": list(block.bbox),
        "text": block.text,
        "font_size": block.font_size,
        "bold": block.bold,
        "is_code": block.is_code,
        "translatable": block.translatable,
        "translated": block.translated,
    }


def block_from_dict(data: dict) -> Block:
    return Block(
        id=data["id"],
        page=data["page"],
        bbox=tuple(data["bbox"]),
        text=data["text"],
        font_size=data["font_size"],
        bold=data["bold"],
        is_code=data["is_code"],
        translatable=data["translatable"],
        translated=data["translated"],
        from_ocr=True,
    )
