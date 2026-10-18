# AGENTS.md

CLI tool (`pdf-translator`) that translates technical PDFs English → Spanish via the OpenAI API, preserving layout, code blocks, and images. Single package, `src/` layout, no monorepo.

## Commands

```bash
pip install -e .                       # into .venv (already exists; use .venv/bin/python)
python tests/test_smoke.py             # the only test; offline, no API key needed
pdf-translator file.pdf                # real run: needs OPENAI_API_KEY in .env
```

- `pytest` is **not installed** in `.venv` even though `tests/test_smoke.py`'s docstring mentions it. Run the file directly (`.venv/bin/python tests/test_smoke.py`).
- There is no linter, formatter, typechecker, or CI configured. The smoke test is the whole verification suite.
- Real runs cost API money and hit rate limits; never run `pdf-translator` on the repo's PDFs just to "test" something — the smoke test covers the pipeline without API.

## Language convention

The entire codebase — docstrings, comments, CLI output, README, and the OpenAI system prompts — is in **Spanish**. Keep it that way when editing or adding code. The LLM prompts being in Spanish is deliberate (they instruct the model to output neutral Spanish).

## Pipeline (src/pdf_translator/)

`__main__.py` orchestrates: `extractor.py` (PyMuPDF text blocks + font metadata) → `code_detect.py` (mono-font ratio ≥ 0.5 or symbol/keyword heuristics → `is_code`, never touched) → `ocr.py` (pages with no extractable text are rendered to PNG @2x and sent to an OpenAI multimodal model, which returns already-translated blocks with `box_2d` coords normalized 0–1000) → `translator.py` (batches of ~5 pages, structured output via pydantic schema, model-compatible sampling defaults, exponential-backoff retries on 429/5xx) → `rebuilder.py` (redactions erase original text but preserve images/vector graphics; OCR text is covered with a white rectangle instead; `insert_htmlbox` with `scale_low=0` auto-shrinks font until the longer Spanish text fits).

With `--parafrasear`, an extra stage runs: `sections.py` detects sections/subsections, skips those with < 600 chars of content, and `translator.summarize_section()` generates a structured summary (`idea` + `puntos` concept/explanation table rows + optional `ejemplo`) from the already-paraphrased text. `rebuilder.insert_summary_pages()` renders each as a boxed table on pages inserted right after the page where the section ends — inserting in descending page order so indices don't shift; box height is measured exactly with `pymupdf.Story` so text is never cut. Heading detection: real-text blocks by typography (font size ≥ body×1.15 or bold+larger; largest size = level 1, next = level 2); OCR blocks via the OpenAI model's `heading_level` mark, with a text-pattern fallback for pre-mark caches ("Capítulo N"/"N.N."/"N.N.N." bold-numbered, or bold multi-word starting with a letter — validated against the repo's real cached books, where estimated OCR font sizes proved useless). Old plain-text `resumen_*` cache entries (no `puntos` key) are regenerated automatically.

Non-obvious behaviors to preserve:

- A "repaso" pass in `__main__.py` re-translates OCR blocks whose output still looks English (`looks_english` heuristic in `ocr.py`) and updates the cache.
- A "costura" (stitching) pass in `__main__.py` runs after the repaso on OCR books: each page is OCR-translated independently, so a sentence crossing a page break gets two halves that may not join — the tail dangles mid-sentence and (especially with `--parafrasear`) the head restarts as a standalone paraphrase, dropping/duplicating content. `find_broken_seams` (ocr.py) heuristically flags candidate pairs (tail ends mid-sentence, or tail ends with terminal punctuation but head starts lowercase); `translator.stitch_pair()` sends both fragments so OpenAI rewrites them to connect without losing/repeating content (it returns them unchanged if the join is already fine). Results are cached as `seam_p*-p*` entries and the affected page caches are updated.
- Blocks detected as code are never redacted or covered — the original stays visible byte-for-byte. Same for OCR blocks the OpenAI model marks `in_diagram=true` (text inside diagrams/flowcharts): diagrams are deliberately left in English (figure captions ARE translated), and the repaso pass skips them because they're non-translatable.
- The OpenAI model sometimes returns markdown emphasis (`**bold**`, `` `code` ``); `rebuilder.py` converts it to HTML. Don't strip it earlier.

## Caching / resume

- `checkpoint.py` caches per-batch results in `.translation_cache/<pdf-stem>-<sha256[:16]>-<model>/` next to the PDF. Interrupting and re-running resumes without re-paying the API.
- The cache dir name includes the **model name**: changing `--model` (or `PDF_TRANSLATOR_MODEL`) starts translation from scratch. `--parafrasear` (paragraphs are paraphrased into plain, easy-to-read Spanish instead of translated literally; headings/bullets stay literal) gets its own cache via a `-parafraseo` suffix, so both styles can coexist.
- To force re-translation of a file, delete its cache dir.

## Config & conventions

- `.env` (gitignored) holds `OPENAI_API_KEY`; loaded via `python-dotenv` at import time in `config.py`.
- Default glossary is repo-root `glossary.txt` (resolved relative to `config.py`, not cwd) — terms listed there are kept in English; it's appended to both the text and OCR prompts.
- Output convention: `<name>.es.pdf` next to the input. `.gitignore` ignores **all** `*.pdf`, so sample/input/output PDFs in the repo root are untracked — don't commit generated PDFs.
- Imports use `pymupdf` (not the legacy `fitz` alias).
- Python ≥ 3.10; code uses `X | None` annotation syntax.
