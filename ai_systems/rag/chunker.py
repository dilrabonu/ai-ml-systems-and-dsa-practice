"""
Three chunking strategies, from naive to smart.

Trade-offs:
  word-window      : simplest, breaks mid-sentence (bad)
  sentence-window  : respects sentence boundaries (default)
  semantic         : splits where meaning shifts (best, slowest)
"""
from __future__ import annotations
from dataclasses import dataclass
import re 
from typing import List

@dataclass
class Chunk:
    text: str
    source: str  # filename or doc id
    chunk_id: int  # position within source
    char_start: int  # for citations / highlighting

# Strategy word-count window
def chunk_by_words(text: str, source: str,
                   size: int = 350, overlap: int = 50) -> List[Chunk]:
    """ Fixed size word window. Fast but cuts mid-sentence."""
    if overlap >= size:
        raise ValueError("overlap must be < size")
    words = text.split()
    out: list[Chunk] = []
    i = chunk_id = 0
    while i < len(words):
        text_chunk = "".join(words[i : i + size])
        out.append(Chunk(text=text_chunk, source=source, chunk_id=chunk_id, char_start=-1))
        i += size - overlap
        chunk_id += 1
    return out
