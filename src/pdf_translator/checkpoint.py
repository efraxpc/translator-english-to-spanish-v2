"""Cache de traducciones por lote para reanudar tras un fallo sin repagar la API."""

import hashlib
import json
from pathlib import Path

from .config import CACHE_DIR_NAME


class Checkpoint:
    def __init__(self, pdf_path: Path, model: str, workdir: Path | None = None):
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
        base = workdir or (pdf_path.parent / CACHE_DIR_NAME)
        self.dir = base / f"{pdf_path.stem}-{digest}-{model}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _batch_file(self, batch_key: str) -> Path:
        return self.dir / f"batch_{batch_key}.json"

    def load(self, batch_key: str) -> dict[str, str] | None:
        path = self._batch_file(batch_key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def save(self, batch_key: str, translations: dict[str, str]) -> None:
        path = self._batch_file(batch_key)
        path.write_text(
            json.dumps(translations, ensure_ascii=False, indent=1), encoding="utf-8"
        )
