# AGENTS.md

CLI tool (`pdf-translator`) that translates technical PDFs English → Spanish via the Gemini API, preserving layout, code blocks, and images. Single package, `src/` layout, no monorepo.

## Commands

```bash
pip install -e .                       # into .venv (already exists; use .venv/bin/python)
python tests/test_smoke.py             # the only test; offline, no API key needed
pdf-translator file.pdf                # real run: needs GEMINI_API_KEY in .env
```

- `pytest` is **not installed** in `.venv` even though `tests/test_smoke.py`'s docstring mentions it. Run the file directly (`.venv/bin/python tests/test_smoke.py`).
- There is no linter, formatter, typechecker, or CI configured. The smoke test is the whole verification suite.
- Real runs cost API money and hit rate limits; never run `pdf-translator` on the repo's PDFs just to "test" something — the smoke test covers the pipeline without API.

## Language convention

The entire codebase — docstrings, comments, CLI output, README, and the Gemini system prompts — is in **Spanish**. Keep it that way when editing or adding code. The LLM prompts being in Spanish is deliberate (they instruct the model to output neutral Spanish).

## Pipeline (src/pdf_translator/)

`__main__.py` orchestrates: `extractor.py` (PyMuPDF text blocks + font metadata) → `code_detect.py` (mono-font ratio ≥ 0.5 or symbol/keyword heuristics → `is_code`, never touched) → `ocr.py` (pages with no extractable text are rendered to PNG @2x and sent to Gemini multimodal, which returns already-translated blocks with `box_2d` coords normalized 0–1000) → `translator.py` (batches of ~5 pages, structured output via pydantic schema, temperature 0.1, exponential-backoff retries on 429/5xx) → `rebuilder.py` (redactions erase original text but preserve images/vector graphics; OCR text is covered with a white rectangle instead; `insert_htmlbox` with `scale_low=0` auto-shrinks font until the longer Spanish text fits).

Non-obvious behaviors to preserve:

- A "repaso" pass in `__main__.py` re-translates OCR blocks whose output still looks English (`looks_english` heuristic in `ocr.py`) and updates the cache.
- Blocks detected as code are never redacted or covered — the original stays visible byte-for-byte.
- Gemini sometimes returns markdown emphasis (`**bold**`, `` `code` ``); `rebuilder.py` converts it to HTML. Don't strip it earlier.

## Caching / resume

- `checkpoint.py` caches per-batch results in `.translation_cache/<pdf-stem>-<sha256[:16]>-<model>/` next to the PDF. Interrupting and re-running resumes without re-paying the API.
- The cache dir name includes the **model name**: changing `--model` (or `PDF_TRANSLATOR_MODEL`) starts translation from scratch. `--parafrasear` (paragraphs are paraphrased into plain, easy-to-read Spanish instead of translated literally; headings/bullets stay literal) gets its own cache via a `-parafraseo` suffix, so both styles can coexist.
- To force re-translation of a file, delete its cache dir.

## Config & conventions

- `.env` (gitignored) holds `GEMINI_API_KEY`; loaded via `python-dotenv` at import time in `config.py`.
- Default glossary is repo-root `glossary.txt` (resolved relative to `config.py`, not cwd) — terms listed there are kept in English; it's appended to both the text and OCR prompts.
- Output convention: `<name>.es.pdf` next to the input. `.gitignore` ignores **all** `*.pdf`, so sample/input/output PDFs in the repo root are untracked — don't commit generated PDFs.
- Imports use `pymupdf` (not the legacy `fitz` alias).
- Python ≥ 3.10; code uses `X | None` annotation syntax.
