"""把段落聚合为不超过限定长度的请求块，并记录块内偏移区间以便回溯段落."""

from __future__ import annotations

import re

from ai_reveal.models import Chunk, Paragraph

# 中英文句子结束符
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*")


def build_chunks(paragraphs: list[Paragraph], chunk_chars: int = 4000) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur_texts: list[str] = []
    cur_ranges: list[tuple[int, int, int]] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur_texts, cur_ranges, cur_len
        if not cur_texts:
            return
        chunks.append(Chunk(
            index=len(chunks),
            text="\n\n".join(cur_texts),
            paragraph_indices=[r[2] for r in cur_ranges],
            ranges=list(cur_ranges),
        ))
        cur_texts, cur_ranges, cur_len = [], [], 0

    for para in paragraphs:
        text = para.text
        if not text:
            continue
        if len(text) > chunk_chars:
            flush()
            for piece in _split_long_text(text, chunk_chars):
                chunks.append(Chunk(
                    index=len(chunks),
                    text=piece,
                    paragraph_indices=[para.index],
                    ranges=[(0, len(piece), para.index)],
                ))
            continue
        start = cur_len + (2 if cur_texts else 0)
        if start + len(text) > chunk_chars and cur_texts:
            flush()
            start = 0
        cur_texts.append(text)
        cur_ranges.append((start, start + len(text), para.index))
        cur_len = start + len(text)
    flush()
    return chunks


def _split_long_text(text: str, limit: int) -> list[str]:
    """超长段落按句子边界切分为 ≤limit 的片段."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s]
    pieces: list[str] = []
    buf = ""
    for sentence in sentences:
        while len(sentence) > limit:
            # 单句仍超长时硬切
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        candidate = f"{buf} {sentence}" if buf else sentence
        if len(candidate) > limit and buf:
            pieces.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        pieces.append(buf)
    return pieces
