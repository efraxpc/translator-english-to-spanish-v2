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
        '[{"id": ..., "text": ...}]. Traduce cada bloque a español neutro y '
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
        "5. No agregues comentarios ni explicaciones: solo la traducción."
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
            [{"id": b.id, "text": b.text} for b in blocks], ensure_ascii=False
        )
        response = self._call_with_retries(payload)
        results: dict[str, str] = {}
        for item in response:
            results[item.id] = item.translated
        # Si el modelo omitió algún bloque, se conserva el original.
        for b in blocks:
            results.setdefault(b.id, b.text)
        return results

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
