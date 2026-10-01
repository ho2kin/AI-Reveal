"""朱雀 AIGC 文本检测客户端与检测主流程.

逐块调用 /v1/providers/zhuque-text/classify（is_merge=false 获得逐段标注），
把 segment_labels 的位置映射回源段落，按段落长度加权汇总整体占比。
"""

from __future__ import annotations

import time

import requests

from ai_reveal.config import (
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    ZHUQUE_CLASSIFY_URL,
)
from ai_reveal.models import (
    Chunk,
    DetectionReport,
    LABEL_AI,
    LABEL_HUMAN,
    LABEL_SUSPECTED,
    Paragraph,
    ParagraphResult,
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ZhuqueError(RuntimeError):
    """朱雀 API 调用失败."""


class ZhuqueClient:
    def __init__(self, api_key: str, timeout: int = REQUEST_TIMEOUT,
                 max_retries: int = MAX_RETRIES, url: str = ZHUQUE_CLASSIFY_URL):
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def classify(self, text: str) -> dict:
        """检测一段文本，返回 API 原始 JSON（status=success 时）."""
        payload = {"text": text, "is_merge": False}
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(self.url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = ZhuqueError(f"网络错误: {exc}")
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "success":
                        return data
                    raise ZhuqueError(f"API 返回失败: {data.get('msg') or data}")
                if resp.status_code in _RETRYABLE_STATUS:
                    last_error = ZhuqueError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}")
                else:
                    raise ZhuqueError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 10))
        raise last_error or ZhuqueError("未知错误")


def run_detection(paragraphs: list[Paragraph], chunks: list[Chunk],
                  client: ZhuqueClient, source_path: str, source_kind: str,
                  progress=None) -> DetectionReport:
    """执行全部块的检测并聚合为 DetectionReport.

    progress: 可选回调 fn(done, total)，用于 CLI 打印进度.
    """
    report = DetectionReport(source_path=source_path, source_kind=source_kind)
    # 段落 -> [(label, conf, 片段长度), ...]
    para_segments: dict[int, list[tuple[int, float, int]]] = {}
    total = len(chunks)

    for chunk in chunks:
        try:
            data = client.classify(chunk.text)
        except ZhuqueError as exc:
            report.warnings.append(
                f"第 {chunk.index + 1}/{total} 块检测失败（{len(chunk.text)} 字符未计入）: {exc}")
            continue
        _collect_chunk(data, chunk, para_segments, report)
        if progress:
            progress(chunk.index + 1, total)

    usage = report.extras.pop("_pending_usage", {"zhuque": 0, "makers": 0})
    report.zhuque_tokens = usage["zhuque"]
    report.makers_tokens = usage["makers"]

    for para in paragraphs:
        result = ParagraphResult(paragraph=para)
        segments = para_segments.get(para.index)
        if segments:
            result.label, result.conf = _aggregate_segments(segments)
        report.paragraphs.append(result)

    _compute_ratios(report)
    _warn_unmapped(report, chunks)
    return report


def _warn_unmapped(report: DetectionReport, chunks: list[Chunk]) -> None:
    """有段落未拿到标注时显式提示，避免整体占比被误读为全量结果."""
    covered = {idx for chunk in chunks for idx in chunk.paragraph_indices}
    unmatched = [r.paragraph.index for r in report.paragraphs
                 if r.paragraph.index in covered and r.label is None]
    if unmatched:
        preview = ", ".join(f"#{i}" for i in unmatched[:10])
        more = f" 等 {len(unmatched)} 段" if len(unmatched) > 10 else ""
        report.warnings.append(
            f"{preview}{more} 未能映射到检测结果（API 标注与文本偏移不一致），"
            "整体占比仅基于已检测段落，建议重试或反馈")


def _collect_chunk(data: dict, chunk: Chunk,
                   para_segments: dict[int, list[tuple[int, float, int]]],
                   report: DetectionReport) -> None:
    usage = data.get("usage") or {}
    makers_usage = data.get("makers_models_usage") or {}
    pending = report.extras.setdefault("_pending_usage", {"zhuque": 0, "makers": 0})
    pending["zhuque"] += int(usage.get("total_tokens") or 0)
    pending["makers"] += int(makers_usage.get("total_tokens") or 0)

    segments = data.get("segment_labels") or []
    if not segments:
        # API 未返回逐段标注时，用整体比例近似：整块按占比拆分记到块内所有段落
        ratios = data.get("labels_ratio") or {}
        label = _max_ratio_label(ratios)
        if label is not None:
            for para_idx in chunk.paragraph_indices:
                para_segments.setdefault(para_idx, []).append(
                    (label, float(data.get("ratio_confidence") or 0.0), 0))
        return

    for seg in segments:
        try:
            label = int(seg.get("label"))
        except (TypeError, ValueError):
            continue
        conf = float(seg.get("conf") or 0.0)
        seg_len = len(str(seg.get("text") or ""))
        for para_idx, overlap_len in _map_segment(seg, chunk):
            para_segments.setdefault(para_idx, []).append((label, conf, overlap_len or seg_len))


def _map_segment(seg: dict, chunk: Chunk) -> list[tuple[int, int]]:
    """把一个 segment 映射到与其重叠的所有段落，返回 (段落, 重叠长度).

    朱雀 position 语义为 [起始偏移, 长度]（实测校准，非 [起, 止]）；
    其切分为句子级、与段落边界不重合，按重叠长度分摊到各段落。
    """
    seg_text = str(seg.get("text") or "")
    position = seg.get("position")
    start = None
    end = None
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        try:
            start = int(position[0])
            end = start + max(int(position[1]), 0)
        except (TypeError, ValueError):
            start = None
    if start is None and seg_text:
        # 位置缺失时的兜底：直接在块内查找 segment 文本
        start = chunk.text.find(seg_text)
        end = start + len(seg_text) if start >= 0 else None
    if start is not None and seg_text:
        # 校验位置与文本一致，不一致则以文本查找为准
        if chunk.text[start:start + len(seg_text)] != seg_text:
            alt = chunk.text.find(seg_text)
            if alt >= 0:
                start, end = alt, alt + len(seg_text)
    if start is not None and end is not None and end > start:
        end = min(end, len(chunk.text))
        overlaps = []
        for begin, stop, para_idx in chunk.ranges:
            overlap = min(end, stop) - max(start, begin)
            if overlap > 0:
                overlaps.append((para_idx, overlap))
        if overlaps:
            return overlaps
    # 无有效位置时退化为单段块直取
    if len(chunk.paragraph_indices) == 1:
        return [(chunk.paragraph_indices[0], 0)]
    return []


def _aggregate_segments(segments: list[tuple[int, float, int]]) -> tuple[int, float]:
    """一个段落含多个片段时，按片段长度加权选出主导标签."""
    weighted: dict[int, float] = {}
    total_len = 0.0
    for label, conf, seg_len in segments:
        weight = max(seg_len, 1)
        weighted[label] = weighted.get(label, 0.0) + conf * weight
        total_len += weight
    best_label = max(weighted, key=lambda k: weighted[k])
    best_conf = weighted[best_label] / max(total_len, 1.0)
    return best_label, min(best_conf, 1.0)


def _max_ratio_label(ratios: dict) -> int | None:
    best_label, best_value = None, -1.0
    for key, value in ratios.items():
        try:
            label = int(key)
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v > best_value:
            best_label, best_value = label, v
    return best_label


def _compute_ratios(report: DetectionReport) -> None:
    """按段落字符长度加权计算整体占比（仅统计已被成功检测的段落）."""
    totals = {LABEL_HUMAN: 0.0, LABEL_AI: 0.0, LABEL_SUSPECTED: 0.0}
    labeled_len = 0.0
    for result in report.paragraphs:
        if result.label is None:
            continue
        weight = max(len(result.paragraph.text), 1)
        totals[result.label] += weight
        labeled_len += weight
    if labeled_len > 0:
        report.human_ratio = totals[LABEL_HUMAN] / labeled_len
        report.ai_ratio = totals[LABEL_AI] / labeled_len
        report.suspected_ratio = totals[LABEL_SUSPECTED] / labeled_len
