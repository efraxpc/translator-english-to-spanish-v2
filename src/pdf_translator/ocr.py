"""OCR con OpenAI para páginas sin texto extraíble (texto convertido a curvas).

Algunos PDFs no contienen texto real: los glifos están vectorizados como
trazos, sin fuentes ni operadores de texto, y PyMuPDF no puede extraer nada.
Para esas páginas se renderiza la página a imagen y se le pide a OpenAI
(multimodal) que detecte los bloques de texto con sus coordenadas y los
traduzca en una sola llamada.
"""

import base64
import re
import time

import pymupdf
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from .extractor import Block

_ZOOM = 2.0  # resolución del render enviado a OpenAI
_MAX_RETRIES = 5
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MIN_REGION_HEIGHT = 62  # coordenadas normalizadas; hasta 16 franjas por página


class OcrBlock(BaseModel):
    box_2d: list[int]  # [ymin, xmin, ymax, xmax] normalizado a 0-1000
    text: str
    is_code: bool
    line_count: int
    bold: bool
    is_table_cell: bool = False
    cell_box_2d: list[int] | None = None


class OcrPage(BaseModel):
    blocks: list[OcrBlock]


class EmptyOcrResponse(ValueError):
    """OpenAI terminó la llamada sin devolver ningún bloque de texto."""


def build_ocr_prompt(glossary: list[str], parafrasear: bool = False) -> str:
    glossary_section = ""
    if glossary:
        glossary_section = (
            "\n\nGlosario de términos que SIEMPRE se conservan en inglés "
            "(además de las reglas anteriores):\n" + ", ".join(glossary)
        )
    paraphrase_section = ""
    if parafrasear:
        paraphrase_section = (
            "\n\nModo parafraseo (prioridad sobre la traducción literal):\n"
            "- Los párrafos completos NO se traducen literalmente: reescribe "
            "su contenido en un español claro y sencillo, explicando la idea "
            "con palabras más simples para que cualquier lector lo entienda "
            "fácilmente, sin perder precisión técnica.\n"
            "- Usa una extensión similar o menor que el original: el texto "
            "debe caber en el mismo espacio del PDF.\n"
            "- Títulos, viñetas, etiquetas cortas y pies de figura se traducen "
            "de forma directa, sin parafrasear.\n"
            "- El resto de reglas (términos técnicos en inglés, código "
            "intacto, puntuación estructural, glosario) sigue aplicando."
        )
    return (
        "La imagen es una página de un libro técnico en inglés sobre IA, "
        "Machine Learning, ingeniería de software y Google Cloud Platform.\n\n"
        "Detecta TODOS los bloques de texto de la página (títulos, párrafos, "
        "viñetas, pies de figura, cabeceras, números de página...). Para cada "
        "bloque devuelve:\n"
        "- box_2d: [ymin, xmin, ymax, xmax] normalizado a 0-1000, ajustado al "
        "texto visible del bloque.\n"
        "- text: la traducción del bloque a español neutro y profesional. Si "
        "el bloque es código, devuelve el texto original sin cambios.\n"
        "- is_code: true si el bloque es un listado de código, comando de "
        "terminal o salida de consola.\n"
        "- line_count: número de líneas de texto del bloque original.\n"
        "- bold: true si el bloque está mayoritariamente en negrita.\n"
        "- is_table_cell: true si el bloque pertenece a una celda de una "
        "tabla con filas y columnas, aunque no tenga bordes, o a una "
        "estructura de datos alineados en columnas.\n"
        "- cell_box_2d: para una celda, los límites completos de esa celda "
        "normalizados a 0-1000; usa null para los demás bloques.\n\n"
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
        "6. Si un párrafo empieza o termina a media oración porque continúa "
        "en la página anterior o siguiente, NO lo completes: parafrasea o "
        "traduce solo el fragmento y déjalo igual de incompleto que el "
        "original (sin cerrar la oración ni inventar un comienzo), para que "
        "encaje con su continuación.\n"
        "7. En tablas con o sin bordes y estructuras de datos en columnas, "
        "devuelve cada celda no vacía como un "
        "bloque independiente. Conserva las celdas combinadas mediante su "
        "cell_box_2d y nunca fusiones, dividas o reordenes su contenido ni "
        "generes sintaxis de tabla Markdown. Ajusta box_2d solo al texto para "
        "que la cuadrícula original no se cubra.\n"
        "8. No agregues comentarios: solo los bloques detectados."
        + glossary_section
        + paraphrase_section
    )


class OpenAIOcr:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        glossary: list[str] | None = None,
        parafrasear: bool = False,
    ):
        self.client = client
        self.model = model
        self.prompt = build_ocr_prompt(glossary or [], parafrasear)

    def ocr_page(self, page: pymupdf.Page, page_index: int) -> list[Block]:
        """Renderiza la página, la manda a OpenAI y devuelve bloques traducidos."""
        pix = page.get_pixmap(matrix=pymupdf.Matrix(_ZOOM, _ZOOM))
        png = pix.tobytes("png")
        try:
            items = self._call_with_retries(png)
        except EmptyOcrResponse as exc:
            print(
                "    OpenAI no devolvió texto para la página completa "
                f"({exc}); procesando recortes reales..."
            )
            # Un bloqueo de contenido puede depender de la cantidad de texto
            # visible; los recortes físicos reducen el contexto de cada llamada.
            reason = str(exc).lower()
            parts = 4 if "content_filter" in reason or "refusal" in reason else 2
            items = self._call_by_regions(page, 0, 1000, parts)

        width, height = page.rect.width, page.rect.height
        blocks: list[Block] = []
        for i, item in enumerate(items):
            bbox = _box_to_page(item.box_2d, width, height)
            if bbox is None or not item.text.strip():
                continue
            layout_bbox = None
            if item.is_table_cell and item.cell_box_2d is not None:
                cell_bbox = _box_to_page(item.cell_box_2d, width, height)
                if cell_bbox is not None:
                    cell_rect = pymupdf.Rect(cell_bbox)
                    if cell_rect.contains(pymupdf.Rect(bbox)):
                        if cell_rect.width > 4 and cell_rect.height > 4:
                            cell_rect += (2, 2, -2, -2)
                        layout_bbox = tuple(cell_rect)
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
                    is_table_cell=item.is_table_cell,
                    layout_bbox=layout_bbox,
                )
            )
        return blocks

    def _call_by_regions(
        self,
        page: pymupdf.Page,
        ymin: int,
        ymax: int,
        parts: int = 2,
    ) -> list[OcrBlock]:
        """Renderiza franjas reales y remapea sus coordenadas a la página."""
        items: list[OcrBlock] = []
        boundaries = [
            ymin + round((ymax - ymin) * index / parts)
            for index in range(parts + 1)
        ]
        for start, end in zip(boundaries, boundaries[1:]):
            png = self._render_region(page, start, end)
            try:
                region_items = self._call_with_retries(png, is_crop=True)
            except EmptyOcrResponse as exc:
                if end - start <= _MIN_REGION_HEIGHT:
                    raise EmptyOcrResponse(
                        f"recorte {start}-{end}: {exc}"
                    ) from exc
                print(
                    f"    Recorte {start}-{end} sin respuesta ({exc}); "
                    "subdividiendo..."
                )
                items.extend(self._call_by_regions(page, start, end))
                continue
            items.extend(_remap_region_items(region_items, start, end))
        return items

    @staticmethod
    def _render_region(page: pymupdf.Page, ymin: int, ymax: int) -> bytes:
        """Renderiza un intervalo vertical normalizado de la página."""
        rect = page.rect
        clip = pymupdf.Rect(
            rect.x0,
            rect.y0 + ymin / 1000 * rect.height,
            rect.x1,
            rect.y0 + ymax / 1000 * rect.height,
        )
        pix = page.get_pixmap(matrix=pymupdf.Matrix(_ZOOM, _ZOOM), clip=clip)
        return pix.tobytes("png")

    def _call_with_retries(
        self,
        png: bytes,
        is_crop: bool = False,
    ) -> list[OcrBlock]:
        delay = 5.0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                prompt = self.prompt
                if is_crop:
                    prompt += (
                        "\n\n"
                        "La imagen es un recorte horizontal de una página. "
                        "Procesa únicamente el texto visible, sin reconstruir "
                        "contenido fuera del recorte. Normaliza box_2d y "
                        "cell_box_2d de 0 a 1000 respecto de ESTE RECORTE."
                    )
                image = base64.b64encode(png).decode("ascii")
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{image}",
                                    "detail": "high",
                                },
                            ],
                        }
                    ],
                    text_format=OcrPage,
                    store=False,
                )
                parsed = response.output_parsed
                if parsed is None:
                    detail = _describe_empty_response(response)
                    text = getattr(response, "output_text", None)
                    if not text:
                        raise EmptyOcrResponse(detail)
                    raise ValueError(
                        f"Respuesta sin JSON parseable ({detail}): "
                        f"{text!r:.200}"
                    )
                return parsed.blocks
            except EmptyOcrResponse:
                # Repetir la misma imagen no corrige bloqueos ni respuestas
                # vacías deterministas; el caller la divide en regiones.
                raise
            except (OpenAIError, ValueError) as exc:
                code = getattr(exc, "status_code", None)
                retryable = (
                    isinstance(exc, ValueError)
                    or code is None
                    or code in _RETRYABLE_CODES
                )
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


def _remap_region_items(
    items: list[OcrBlock], ymin: int, ymax: int
) -> list[OcrBlock]:
    """Convierte box_2d normalizados en un recorte a coordenadas de página."""
    height = ymax - ymin
    remapped: list[OcrBlock] = []
    for item in items:
        updates = {"box_2d": _remap_region_box(item.box_2d, ymin, height)}
        if item.cell_box_2d is not None:
            updates["cell_box_2d"] = _remap_region_box(
                item.cell_box_2d, ymin, height
            )
        remapped.append(item.model_copy(update=updates))
    return remapped


def _remap_region_box(box: list[int], ymin: int, height: int) -> list[int]:
    """Remapea una caja del recorte al sistema normalizado de la página."""
    if len(box) != 4:
        return box
    local_ymin, xmin, local_ymax, xmax = box
    return [
        round(ymin + local_ymin / 1000 * height),
        xmin,
        round(ymin + local_ymax / 1000 * height),
        xmax,
    ]


def _box_to_page(
    box: list[int], width: float, height: float
) -> tuple[float, float, float, float] | None:
    """Convierte y valida una caja normalizada a coordenadas del PDF."""
    if len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = (max(0, min(1000, value)) for value in box)
    if ymax <= ymin or xmax <= xmin:
        return None
    return (
        xmin / 1000 * width,
        ymin / 1000 * height,
        xmax / 1000 * width,
        ymax / 1000 * height,
    )


def _describe_empty_response(response) -> str:
    """Resume el motivo real de una respuesta vacía sin volcar datos sensibles."""
    status = getattr(response, "status", None)
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None)
    error = getattr(response, "error", None)
    error_code = getattr(error, "code", None)
    error_message = getattr(error, "message", None)

    refusals = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "refusal":
                refusals.append(getattr(content, "refusal", None))

    usage = getattr(response, "usage", None)
    tokens = ""
    if usage is not None:
        tokens = (
            ", tokens="
            f"entrada:{getattr(usage, 'input_tokens', None)}/"
            f"salida:{getattr(usage, 'output_tokens', None)}/"
            f"total:{getattr(usage, 'total_tokens', None)}"
        )
    return (
        f"status={status!r}, reason={reason!r}, error_code={error_code!r}, "
        f"error_message={error_message!r}, refusals={refusals!r}"
        f"{tokens}"
    )


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
        "is_table_cell": block.is_table_cell,
        "layout_bbox": list(block.layout_bbox) if block.layout_bbox else None,
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
        is_table_cell=data.get("is_table_cell", False),
        layout_bbox=(
            tuple(data["layout_bbox"]) if data.get("layout_bbox") else None
        ),
    )


# --- Costura de oraciones que cruzan saltos de página ----------------------
# Cada página se OCR-traduce por separado; una oración que cruza el salto se
# traduce en dos mitades independientes que pueden no encajar (la cola queda
# colgando y la cabeza reinicia la oración, sobre todo con --parafrasear).

_TERMINAL_RE = re.compile(r"[.!?…:;»”\"')\]]\s*$")


def _ends_mid_sentence(text: str) -> bool:
    """Heurística: ¿este fragmento termina a media oración?"""
    stripped = text.strip().strip("*_` ")
    if not stripped:
        return False
    return not _TERMINAL_RE.search(stripped)


def _starts_lowercase(text: str) -> bool:
    """¿el primer carácter alfabético es minúscula?"""
    for ch in text.strip():
        if ch.isalpha():
            return ch.islower()
        if ch.isdigit():
            return False
    return False


def find_broken_seams(blocks: list[Block]) -> list[tuple[Block, Block]]:
    """Pares (fin de pág. N, inicio de pág. N+1) candidatos a estar mal cosidos.

    Se marca el par si la cola termina a media oración (la cabeza debería
    continuarla pero podría haberla reiniciado) o si la cabeza empieza en
    minúscula pese a que la cola cierra con puntuación final (inversión
    sospechosa). Son solo candidatos: el modelo confirma en el paso de
    costura si la unión necesita reescritura.
    """
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        if (
            b.from_ocr
            and not b.is_table_cell
            and b.translatable
            and b.translated
        ):
            by_page.setdefault(b.page, []).append(b)
    seams: list[tuple[Block, Block]] = []
    for page in sorted(by_page):
        if page + 1 not in by_page:
            continue
        tail = max(by_page[page], key=lambda b: b.bbox[1])
        head = min(by_page[page + 1], key=lambda b: b.bbox[1])
        if _ends_mid_sentence(tail.translated) or _starts_lowercase(head.translated):
            seams.append((tail, head))
    return seams
