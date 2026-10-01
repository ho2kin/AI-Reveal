"""把段落聚合为不超过限定长度的请求块，并记录块内偏移区间以便回溯段落."""

from __future__ import annotations

import re

from ai_reveal.models import Chunk, Paragraph

# 中英文句子结束符
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*")

# 低于该长度的段落视为碎片，与同章节相邻段落合并
MIN_PARA_CHARS = 120


def consolidate(paragraphs: list[Paragraph],
                min_chars: int = MIN_PARA_CHARS) -> list[Paragraph]:
    """整合段落：标题转为章节上下文，同章节内的短段落与相邻段落合并.

    - 标题（kind=heading）不作为检测单元，记录为后续段落的 section 上下文，
      避免把 "Introduction" 这类结构性文本送去检测；
    - 长度不足 min_chars 的段落向后合并（连续碎片链式合并），节内末尾的
      碎片向前合并；图注（kind=caption）作为边界不参与合并。
    返回重新编号后的段落列表。
    """
    # 1. 按章节分组：每个标题都新起一组（防止重名章节跨节合并），图注单独成组
    groups: list[list] = []  # [section, kind, paras]
    section = ""
    for para in paragraphs:
        if para.kind == "heading":
            section = para.text
            groups.append([section, "body", []])
            continue
        kind = para.kind if para.kind in ("abstract", "caption") else "body"
        if not groups or groups[-1][1] != kind:
            groups.append([section, kind, []])
        groups[-1][2].append(para)

    # 2. 节内合并短段落（caption 组不合并不参与）
    merged: list[Paragraph] = []
    for sec, kind, paras in groups:
        if kind == "caption":
            for para in paras:
                para.section = para.section or sec
            merged.extend(paras)
            continue
        buf: list[Paragraph] = []
        for para in paras:
            if buf and len(buf[-1].text) < min_chars:
                buf[-1] = _join_pair(buf[-1], para, sec)
            else:
                buf.append(para)
        if len(buf) >= 2 and len(buf[-1].text) < min_chars:
            tail = buf.pop()
            buf[-1] = _join_pair(buf[-1], tail, sec)
        for para in buf:
            para.section = sec
        merged.extend(buf)

    for i, para in enumerate(merged):
        para.index = i
    return merged


def _join_pair(a: Paragraph, b: Paragraph, section: str) -> Paragraph:
    """合并两个段落，保留前者的元信息."""
    a.section = section
    return Paragraph(
        index=a.index,
        text=f"{a.text} {b.text}".strip(),
        source=a.source,
        kind=a.kind,
        section=section,
    )


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
