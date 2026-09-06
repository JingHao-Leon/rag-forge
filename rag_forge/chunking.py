"""Chunking strategies: fixed-size and separator-aware recursive chunking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    text: str
    start: int  # char offset in the source doc

    def __repr__(self) -> str:  # pragma: no cover
        return f"Chunk({self.doc_id}, {len(self.text)}ch, {self.text[:24]!r})"


def fixed_chunks(doc_id: str, text: str, size: int = 256, overlap: int = 32) -> list[Chunk]:
    assert 0 <= overlap < size
    chunks = []
    step = size - overlap
    start = 0
    while start < len(text):
        chunks.append(Chunk(doc_id, text[start:start + size], start))
        if start + size >= len(text):
            break
        start += step
    return chunks


def recursive_chunks(doc_id: str, text: str, size: int = 256, overlap: int = 32,
                     separators: tuple[str, ...] = ("\n\n", "\n", "。", "；", "; ", " ")) -> list[Chunk]:
    """Separator-aligned chunking: never break inside a unit when a separator fits.

    Greedily packs separator-joined neighbors up to ``size``; a single oversized
    piece is re-split with the next (smaller) separator. Pieces partition the
    source text exactly, so ``"".join(c.text) == text`` — no lost separators.
    (``overlap`` is kept for API symmetry; only ``fixed_chunks`` uses it.)
    """
    assert 0 <= overlap < size

    def split_seg(seg: str, seps: tuple[str, ...]) -> list[str]:
        if len(seg) <= size:
            return [seg]
        if not seps:  # unsplittable run: hard-cut (still an exact partition)
            return [seg[i:i + size] for i in range(0, len(seg), size)]
        sep = seps[0]
        parts = seg.split(sep)
        # token stream: part0, sep, part1, sep, ... — greedy packing keeps every
        # token exactly once, so "".join(pieces) == seg (pieces may lead with a sep)
        out: list[str] = []
        buf = ""
        for i, p in enumerate(parts):
            for tok in ((p,) if i == 0 else (sep, p)):
                cand = buf + tok
                if len(cand) <= size:
                    buf = cand
                    continue
                if buf:
                    out.append(buf)
                if len(tok) <= size:
                    buf = tok
                else:  # oversized part: recurse with smaller separators
                    out.extend(split_seg(tok, seps[1:]))
                    buf = ""
        if buf:
            out.append(buf)
        return out

    chunks: list[Chunk] = []
    pos = 0  # pieces partition the text, so offsets are contiguous
    pending_ws = ""
    for piece in split_seg(text, separators):
        if not piece.strip():  # whitespace-only pieces ride along with a neighbor
            pending_ws += piece
            pos += len(piece)
            continue
        if pending_ws and chunks:
            chunks[-1].text += pending_ws  # attach to the previous chunk
        start = pos - len(pending_ws) if pending_ws else pos
        text_piece = pending_ws + piece if (pending_ws and not chunks) else piece
        pending_ws = ""
        chunks.append(Chunk(doc_id, text_piece, start))
        pos += len(piece)
    if pending_ws and chunks:
        chunks[-1].text += pending_ws
    elif pending_ws:  # whitespace-only document: keep it as a single chunk
        chunks.append(Chunk(doc_id, pending_ws, 0))
    return chunks
