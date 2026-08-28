"""Cliente Gemini: traduce bloques en lotes con structured output y retries."""

import json
import time

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from .extractor import Block

_MAX_RETRIES = 5
_RETRYABLE_CODES = {429, 500, 502, 503, 504}


class TranslatedBlock(BaseModel):
    id: str
    translated: str


class StitchedFragments(BaseModel):
    tail: str
    head: str


_STITCH_SYSTEM_PROMPT = (
    'Dos fragmentos consecutivos de un libro técnico cruzan un salto de '
    'página: "tail" termina una página y "head" empieza la siguiente. Ambos '
    'se tradujeron o parafrasearon por separado, y la oración que cruza el '
    'salto puede haber quedado mal cosida: "tail" colgado a media frase, '
    '"head" reiniciando la oración por su cuenta, o contenido repetido u '
    'omitido entre ambos.\n\n'
    'Reescribe "tail" y "head" para que el texto fluya de forma natural y '
    'gramaticalmente correcta al leerse en secuencia, conservando TODO el '
    'contenido de ambos, sin omitir ni añadir información, y con una '
    'extensión total similar a la original (deben caber en el mismo espacio '
    'del PDF). Conserva el estilo del texto (español claro, términos '
    'técnicos en inglés, código y formato markdown intactos). Si la unión '
    'ya es correcta, devuelve ambos fragmentos sin cambios. No agregues '
    'comentarios.'
)


def build_system_prompt(glossary: list[str], parafrasear: bool = False) -> str:
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
        "Eres un traductor técnico profesional de inglés a español "
        "especializado en ingeniería de IA, Machine Learning, ingeniería de "
        "software y Google Cloud Platform (GCP).\n\n"
        "Recibirás un JSON con bloques de texto extraídos de un PDF técnico: "
        '[{"id": ..., "text": ..., "is_table_cell": ...}]. Traduce cada '
        "bloque a español neutro y "
        "profesional, y devuelve un bloque traducido por cada bloque recibido, "
        "con el mismo id.\n\n"
        "Reglas estrictas:\n"
        "1. CONSERVA EN INGLÉS los términos técnicos de IA/ML/software/GCP: "
        "nombres de productos (Cloud Run, BigQuery, Vertex AI...), conceptos "
        "establecidos (embedding, transformer, pipeline, deployment, "
        "fine-tuning...), siglas (API, SDK, IAM, CI/CD...). El lector es un "
        "ingeniero que conoce esos términos en inglés.\n"
        "2. NO modifiques nada que parezca código: fragmentos inline como "
        "`df.groupby()`, comandos (`gcloud run deploy`), nombres de variables, "
        "clases, funciones, rutas de archivo, URLs, claves de configuración. "
        "Cópialos carácter por carácter.\n"
        "3. Conserva la puntuación estructural del original (viñetas, "
        "numeración, dos puntos finales, etc.).\n"
        "4. Si un bloque no necesita traducción (ya está en español, es código, "
        "un número o un nombre propio), devuélvelo sin cambios.\n"
        "5. Los bloques con is_table_cell=true son celdas independientes de "
        "una tabla. Traduce solo el contenido de esa celda: no lo fusiones "
        "con otras, no lo dividas ni reordenes y no agregues separadores ni "
        "sintaxis de tabla Markdown.\n"
        "6. No agregues comentarios ni explicaciones: solo la traducción."
        + glossary_section
        + paraphrase_section
    )


class GeminiTranslator:
    def __init__(
        self,
        model: str,
        glossary: list[str] | None = None,
        parafrasear: bool = False,
    ):
        self.client = genai.Client()
        self.model = model
        self.glossary = glossary or []
        self.parafrasear = parafrasear
        self.system_prompt = build_system_prompt(self.glossary, parafrasear)

    def translate_batch(self, blocks: list[Block]) -> dict[str, str]:
        """Traduce un lote de bloques. Devuelve {id: texto_traducido}."""
        if not blocks:
            return {}
        payload = json.dumps(
            [
                {
                    "id": b.id,
                    "text": b.text,
                    "is_table_cell": b.is_table_cell,
                }
                for b in blocks
            ],
            ensure_ascii=False,
        )
        response = self._call_with_retries(payload)
        results: dict[str, str] = {}
        for item in response:
            results[item.id] = item.translated
        # Si el modelo omitió algún bloque, se conserva el original.
        for b in blocks:
            results.setdefault(b.id, b.text)
        return results

    def stitch_pair(self, tail_text: str, head_text: str) -> tuple[str, str]:
        """Reescribe dos fragmentos que cruzan un salto de página para que encajen."""
        contents = json.dumps(
            {"tail": tail_text, "head": head_text}, ensure_ascii=False
        )
        delay = 5.0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=_STITCH_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=StitchedFragments,
                        temperature=0.2,
                    ),
                )
                parsed = response.parsed
                if parsed is None:
                    raise ValueError(f"Respuesta sin JSON parseable: {response.text!r:.200}")
                return parsed.tail, parsed.head
            except errors.APIError as exc:
                if exc.code in _RETRYABLE_CODES and attempt < _MAX_RETRIES:
                    print(
                        f"    API error {exc.code} ({exc.status}); "
                        f"reintento {attempt}/{_MAX_RETRIES} en {delay:.0f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")

    def _call_with_retries(self, payload: str) -> list[TranslatedBlock]:
        delay = 5.0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=payload,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        response_mime_type="application/json",
                        response_schema=list[TranslatedBlock],
                        temperature=0.1,
                    ),
                )
                parsed = response.parsed
                if parsed is None:
                    raise ValueError(f"Respuesta sin JSON parseable: {response.text!r:.200}")
                return parsed
            except errors.APIError as exc:
                if exc.code in _RETRYABLE_CODES and attempt < _MAX_RETRIES:
                    print(
                        f"    API error {exc.code} ({exc.status}); "
                        f"reintento {attempt}/{_MAX_RETRIES} en {delay:.0f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")
