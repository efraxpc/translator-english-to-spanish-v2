"""Heurísticas para clasificar bloques como código y dejarlos intactos."""

import re

MONO_FONT_RE = re.compile(
    r"mono|courier|consolas|menlo|inconsolata|roboto ?mono|source ?code|"
    r"fira ?code|dejavu ?sans ?mono|liberation ?mono|nimbus ?mono|jetbrains",
    re.IGNORECASE,
)

_CODE_LINE_RE = re.compile(
    r"^\s*($|\$ |>>> |# |// |def |class |import |from |return |if |for |while "
    r"|gcloud |kubectl |pip |docker |curl )"
)

_CODE_SYMBOLS = set("{}();=<>[]#$|&")


def is_mono_font(font_name: str) -> bool:
    return bool(MONO_FONT_RE.search(font_name or ""))


def looks_like_code(text: str, mono_ratio: float) -> bool:
    """Decide si un bloque es código.

    mono_ratio: fracción de caracteres del bloque escritos en fuente monoespaciada.
    """
    if mono_ratio >= 0.5:
        return True

    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False

    # Densidad de símbolos típicos de código
    symbol_chars = sum(1 for c in text if c in _CODE_SYMBOLS)
    symbol_density = symbol_chars / max(len(text), 1)

    # Líneas con pinta de código (comandos, keywords, prompts de shell/REPL)
    code_lines = sum(1 for l in lines if _CODE_LINE_RE.match(l))

    # Indentación consistente de varias líneas
    indented = sum(1 for l in text.splitlines() if l.startswith(("    ", "\t")))

    if code_lines / len(lines) >= 0.5:
        return True
    if symbol_density > 0.08 and (indented >= 2 or code_lines >= 1):
        return True
    return False
