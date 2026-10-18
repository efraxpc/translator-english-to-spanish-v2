# PDF Translator (inglés → español)

Los párrafos traducidos tienen un ancho máximo de **13 cm** y se centran
horizontalmente en la hoja, conservando el tamaño original de las páginas.
El texto mantiene la alineación a la izquierda dentro del párrafo y las celdas
de las tablas conservan su posición y su ancho original, sin el límite de 13 cm.
También se exceptúan las estructuras detectadas por columnas alineadas o por
filas con tabuladores, espacios múltiples o separadores `|`; conservan los
saltos de línea y espacios presentes en la traducción. Los bloques más estrechos
mantienen su ancho; la fuente se reduce si hace falta para que el texto quepa
en la altura disponible.

Los títulos, subtítulos y etiquetas cortas conservan su posición original.
Si centrar un párrafo invade otro bloque, se conserva su posición horizontal.
Las cajas de encabezados que se solapan verticalmente con el texto siguiente
se ajustan antes de insertar la traducción para separar ambos textos.

Traduce PDFs técnicos completos (IA, Machine Learning, ingeniería de software, GCP) de inglés a español usando la API de OpenAI, **conservando el layout original** del documento:

- Los **términos técnicos se conservan en inglés** (embedding, pipeline, deployment, Cloud Run, BigQuery...) — configurable en `glossary.txt`.
- Los **bloques de código se detectan y quedan intactos**, byte a byte, con su indentación y formato original.
- Las imágenes y gráficos no se tocan.
- Las **tablas con bordes conservan su estructura**: cada celda se traduce por
  separado dentro de su posición original, incluidas las celdas combinadas y
  las tablas detectadas mediante OCR.
- **Reanudable**: las traducciones se cachean por lote; si el proceso se corta (rate limit, red), al relanzar continúa donde iba sin repagar la API.
- **OCR con OpenAI**: si una página no tiene texto extraíble (texto vectorizado como curvas o página escaneada), se renderiza a imagen y un modelo multimodal de OpenAI detecta y traduce los bloques en una sola llamada, tapando el texto original con el bloque traducido en la misma posición.
- **Rangos de páginas**: permite traducir y extraer solo una parte del PDF usando la numeración física del documento.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# edita .env y pon tu OPENAI_API_KEY (https://platform.openai.com/api-keys)
```

## Uso

```bash
# Un PDF → crea libro.es.pdf junto al original
pdf-translator libro.pdf

# Con ruta de salida explícita
pdf-translator libro.pdf -o traducciones/libro-es.pdf

# Solo las páginas físicas 10 a 25 (ambas incluidas)
# Crea libro.p10-25.es.pdf, que contiene únicamente esas 16 páginas
pdf-translator libro.pdf --pages 10-25

# Carpeta completa de PDFs
pdf-translator ./libros/ -o ./libros-es/

# Cambiar de modelo (por defecto: gpt-5.6-terra)
pdf-translator libro.pdf --model gpt-5.6-luna
```

Opciones útiles:

| Opción | Descripción |
|---|---|
| `--model` | Modelo OpenAI a usar (también vía env `PDF_TRANSLATOR_MODEL`) |
| `--pages-per-batch` | Páginas agrupadas por request a la API (default: 5) |
| `--pages INICIO-FIN` | Traduce un rango inclusivo desde 1 y genera solo esas páginas |
| `--glossary` | Archivo de glosario alternativo |
| `--workdir` | Carpeta del cache de reanudación (default: `.translation_cache/` junto al PDF) |

Si el final del rango supera la longitud del PDF, se ajusta a su última
página y el nombre automático refleja el rango efectivo. En modo carpeta, el
mismo rango se aplica a cada PDF y se omiten con un aviso los documentos que no
alcancen la página inicial.

## Cómo funciona

1. **Extracción** — PyMuPDF extrae los bloques de texto de cada página con posición, tamaño de fuente y estilo. Las tablas delimitadas por líneas se dividen en celdas para conservar sus filas, columnas, espacios y celdas combinadas.
2. **Detección de código** — bloques en fuente monoespaciada (Courier, Consolas...) o con estructura de código no se traducen ni se tocan.
   - Si una página no tiene texto extraíble (0 fuentes, texto convertido a curvas), entra el **fallback OCR**: la página se renderiza a PNG y OpenAI devuelve los bloques con coordenadas ya traducidos (el código se deja intacto). El resultado se cachea por página.
3. **Traducción** — los bloques se envían a OpenAI mediante la API Responses en lotes con salida JSON estructurada; el prompt exige conservar terminología técnica y código inline.
4. **Reconstrucción** — el texto original se borra con redacciones (sin afectar imágenes) y la traducción se inserta en el mismo recuadro; la fuente se reduce automáticamente si el español no cabe.

## Test

```bash
python tests/test_smoke.py   # no consume API
```
