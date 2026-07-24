# PDF Translator (inglés → español)

Traduce PDFs técnicos completos (IA, Machine Learning, ingeniería de software, GCP) de inglés a español usando la API de Gemini, **conservando el layout original** del documento:

- Los **términos técnicos se conservan en inglés** (embedding, pipeline, deployment, Cloud Run, BigQuery...) — configurable en `glossary.txt`.
- Los **bloques de código se detectan y quedan intactos**, byte a byte, con su indentación y formato original.
- Las imágenes y gráficos no se tocan.
- **Reanudable**: las traducciones se cachean por lote; si el proceso se corta (rate limit, red), al relanzar continúa donde iba sin repagar la API.
- **OCR con Gemini**: si una página no tiene texto extraíble (texto vectorizado como curvas o página escaneada), se renderiza a imagen y Gemini detecta y traduce los bloques en una sola llamada, tapando el texto original con el bloque traducido en la misma posición.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# edita .env y pon tu GEMINI_API_KEY (https://aistudio.google.com/apikey)
```

## Uso

```bash
# Un PDF → crea libro.es.pdf junto al original
pdf-translator libro.pdf

# Con ruta de salida explícita
pdf-translator libro.pdf -o traducciones/libro-es.pdf

# Carpeta completa de PDFs
pdf-translator ./libros/ -o ./libros-es/

# Cambiar de modelo (por defecto: gemini-3.1-pro-preview)
pdf-translator libro.pdf --model gemini-2.5-pro
```

Opciones útiles:

| Opción | Descripción |
|---|---|
| `--model` | Modelo Gemini a usar (también vía env `PDF_TRANSLATOR_MODEL`) |
| `--pages-per-batch` | Páginas agrupadas por request a la API (default: 5) |
| `--glossary` | Archivo de glosario alternativo |
| `--workdir` | Carpeta del cache de reanudación (default: `.translation_cache/` junto al PDF) |

## Cómo funciona

1. **Extracción** — PyMuPDF extrae los bloques de texto de cada página con posición, tamaño de fuente y estilo.
2. **Detección de código** — bloques en fuente monoespaciada (Courier, Consolas...) o con estructura de código no se traducen ni se tocan.
   - Si una página no tiene texto extraíble (0 fuentes, texto convertido a curvas), entra el **fallback OCR**: la página se renderiza a PNG y Gemini devuelve los bloques con coordenadas ya traducidos (el código se deja intacto). El resultado se cachea por página.
3. **Traducción** — los bloques se envían a Gemini en lotes con salida JSON estructurada; el prompt exige conservar terminología técnica y código inline.
4. **Reconstrucción** — el texto original se borra con redacciones (sin afectar imágenes) y la traducción se inserta en el mismo recuadro; la fuente se reduce automáticamente si el español no cabe.

## Test

```bash
python tests/test_smoke.py   # no consume API
```
