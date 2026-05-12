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
# Strategy sentence window
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

def chunk_by_sentences(text: str, source: str,
                       max_chars: int = 1500, overlap_sentences: int = 2) -> List[Chunk]:
    """ Group whole sentences into -max_chars buckets, with a small sentence overlap.
    Simple and surprisingly effective. Don't reach for fancy strategies until
    this proves insufficient on your eval set.
    """
    sentences = _SENT_SPLIT.split(text.strip())
    chunks: list[Chunk] = []
    buf: list[str] = []
    cur_len = 0
    chunk_id = 0
    for sent in sentences:
        if cur_len + len(sent) > max_chars and buf:
            chunks.append(Chunk(text="".join(buf), source=source,
                                chunk_id=chunk_id, char_start=-1))
            chunk_id += 1
            buf = buf[-overlap_sentences:]  # carry over last N sentences
            cur_len = sum(len(s) + 1 for s in buf)
        buf.append(sent)
        cur_len += len(sent) + 1
    
    if buf:
        chunks.append(Chunk(text="".join(buf), source=source,
                            chunk_id=chunk_id, char_start=-1))
    return chunks

# Strategy semantic 
def chunk_semantic(text: str, source: str, embedder, 
                    threshold: float = 0.75) -> List[Chunk]:
    """Split where consecutive sentences become semantically dissimilar.

    This is the highest-quality strategy for unstructured prose, at the cost
    of running the embedder twice (once during chunking, once for indexing).
    """
    sentences = _SENT_SPLIT.split(text.strip())
    if not sentences:
        return []
    sent_embs = embedder.encode(sentences, normalize_embeddings=True)
    chunks: list[Chunk] = []
    buf: list[str] = [sentences[0]]
    chunk_id = 0
    for i in range(1, len(sentences)):
        sim = float(sent_embs[i] @ sent_embs[i - 1])
        if sim < threshold:
            chunks.append(Chunk(text="".join(buf), source=source,
                                chunk_id=chunk_id, char_start=-1))
            chunk_id += 1
            buf = [sentences[i]]
        else:
            buf.append(sentences[i])

    if buf:
        chunks.append(Chunk(text="".join(buf), source=source,
                            chunk_id=chunk_id, char_start=-1))
    return chunks