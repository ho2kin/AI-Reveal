"""PDF 论文正文提取（PyMuPDF）。

按块读取文本 → 逐页处理双栏阅读顺序 → 去除重复页眉页脚与纯页码块
→ 组装段落 → 默认从 References/参考文献 处截断。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

try:
    import pymupdf
except ImportError:  # 旧版本 PyMuPDF 只提供 fitz 入口
    import fitz as pymupdf

from ai_reveal.models import Paragraph

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# 参考文献标题块（去除所有空白后匹配，规避 PDF 抽取的字符间空格）
_REF_HEADINGS = {"references", "bibliography", "参考文献", "引用文献"}


def extract_pdf(path: Path, keep_references: bool = False) -> tuple[list[Paragraph], list[str]]:
    """提取 PDF 正文段落，返回 (段落列表, 警告列表)."""
    path = Path(path)
    warnings: list[str] = []
    doc = pymupdf.open(path)
    try:
        if doc.is_encrypted:
            raise RuntimeError(f"PDF 已加密，无法读取: {path}")
        all_blocks = []  # (页码, 排序后的块列表)
        for page_no, page in enumerate(doc, start=1):
            ordered = _page_blocks_in_order(page)
            all_blocks.append((page_no, ordered))
        all_blocks = _drop_headers_footers(all_blocks, total_pages=doc.page_count)
    finally:
        doc.close()

    paragraphs: list[Paragraph] = []
    index = 0
    for page_no, blocks in all_blocks:
        for text in blocks:
            text = _clean_block(text)
            if not text or not _is_useful(text):
                continue
            paragraphs.append(Paragraph(index=index, text=text, source=f"p.{page_no}"))
            index += 1

    if not keep_references:
        paragraphs, cut = _cut_references(paragraphs)
        if cut:
            warnings.append(f"已截断参考文献部分（自 {cut} 起），如需保留请加 --keep-references")
    return paragraphs, warnings


def _page_blocks_in_order(page) -> list[str]:
    """读取一页的文本块；双栏页面按左列→右列输出，跨栏块按 y 位置穿插."""
    raw = [b for b in page.get_text("blocks") if b[6] == 0]  # 只留文本块
    if not raw:
        return []
    mid = page.rect.width / 2
    left, right, crossing = [], [], []
    for b in raw:
        x0, x1 = b[0], b[2]
        if x0 < mid - 8 and x1 > mid + 8:
            crossing.append(b)
        elif x1 <= mid + 8:
            left.append(b)
        else:
            right.append(b)
    two_col = len(left) >= 3 and len(right) >= 3  # 少量跨栏块（通栏图表）不影响判定
    if two_col:
        left.sort(key=lambda b: b[1])
        right.sort(key=lambda b: b[1])
        crossing.sort(key=lambda b: b[1])
        ordered = _merge_columns(left, right, crossing)
    else:
        ordered = sorted(raw, key=lambda b: (round(b[1]), b[0]))
    return [b[4] for b in ordered]


def _merge_columns(left: list, right: list, crossing: list) -> list:
    """左列读完读右列；顶部/中部的跨栏块（通栏图表标题等）按 y 插入."""
    ordered = []
    li = ri = ci = 0
    inf = float("inf")
    while li < len(left) or ri < len(right):
        next_left_y = left[li][1] if li < len(left) else inf
        next_right_y = right[ri][1] if ri < len(right) else inf
        while ci < len(crossing) and crossing[ci][1] <= min(next_left_y, next_right_y):
            ordered.append(crossing[ci])
            ci += 1
        if next_left_y <= next_right_y:
            ordered.append(left[li])
            li += 1
        else:
            ordered.append(right[ri])
            ri += 1
    ordered.extend(crossing[ci:])
    return ordered


def _drop_headers_footers(pages: list, total_pages: int) -> list:
    """删除在多页重复出现的文本块（页眉/页脚/期刊栏头）与纯页码块."""
    threshold = max(2, int(total_pages * 0.4))
    repeat_counter: Counter[str] = Counter()
    for _, blocks in pages:
        seen_on_page: set[str] = set()
        for text in blocks:
            normalized = _normalize_for_repeat(text)
            if normalized and normalized not in seen_on_page:
                seen_on_page.add(normalized)
                repeat_counter[normalized] += 1
    frequent = {k for k, v in repeat_counter.items() if v >= threshold}
    cleaned = []
    for page_no, blocks in pages:
        kept = []
        for text in blocks:
            if _normalize_for_repeat(text) in frequent:
                continue
            if re.fullmatch(r"[\s\d\-—–.]+", text):
                continue  # 纯页码/页脚数字
            kept.append(text)
        cleaned.append((page_no, kept))
    return cleaned


def _normalize_for_repeat(text: str) -> str:
    return re.sub(r"[\d\s]+", "", text).strip().lower()


def _clean_block(text: str) -> str:
    # 块内换行：CJK 之间直接拼接，其余替换为空格
    lines = [ln.strip() for ln in text.splitlines()]
    parts: list[str] = []
    for ln in lines:
        if not ln:
            continue
        if parts:
            prev_last, cur_first = parts[-1][-1], ln[0]
            if _CJK_RE.search(prev_last) and _CJK_RE.search(cur_first):
                parts[-1] += ln
            else:
                parts[-1] += " " + ln
        else:
            parts.append(ln)
    text = " ".join(parts)
    # 英文断词连字符修复
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _is_useful(text: str) -> bool:
    if len(text) < 2:
        return False
    return bool(_CJK_RE.search(text) or _LATIN_RE.search(text))


def _cut_references(paragraphs: list[Paragraph]) -> tuple[list[Paragraph], int | None]:
    for i, p in enumerate(paragraphs):
        stripped = re.sub(r"\s+", "", p.text).lower()
        if len(p.text) <= 60 and stripped in _REF_HEADINGS:
            return paragraphs[:i], p.source
        # 处理字符间被插入空格的情况，如 "R e f e r e n c e s"
        compact = re.sub(r"[\s]", "", p.text).lower()
        if len(p.text) <= 60 and compact == "references":
            return paragraphs[:i], p.source
    return paragraphs, None
