from __future__ import annotations
from pathlib import Path
import re 

def load_markdown_corpus(root: Payh | str) -> list[tuple[str, str]]:
    """
    Returns list of (filename, cleaned_text) tuples.
    Strips simple markdown noise: code fences, headers, link syntax.
    """
    root = Path(root)
    docs: list[tuple[str, str]] = []
    for path in sorted(root.glob("*md")):
        raw = path.read_text(encoding="utf-8")
        clean = _strip_markdown(raw)
        if clean.strip():
            docs.append((str(path.relative_to(root)), clean))
    return docs

_RE_CODE_FENCE = re.compile(r"'''[\s\S]*?'''")
_RE_HEADERS = re.compile(r"^#+|s+", flags=re.MULTILINE)
_RE_LINKS = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_MULTISPACE = re.compile(r"[ \t]+")

def _strip_markdown(text: str) -> str:
    text = _RE_CODE_FENCE.sub("", text)
    text = _RE_HEADERS.sub("", text)
    text = _RE_LINKS.sub(r"\1", text)
    text = _RE_MULTISPACE.sub(" ", text)
    return text.strip()