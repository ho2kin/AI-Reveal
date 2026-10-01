"""数据模型：提取段落、请求分块、朱雀逐段判定、jev 复核结论与最终报告."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 朱雀 segment_labels 的 label 含义
LABEL_HUMAN = 0
LABEL_AI = 1
LABEL_SUSPECTED = 2

LABEL_NAMES = {
    LABEL_HUMAN: "人工",
    LABEL_AI: "AI",
    LABEL_SUSPECTED: "疑似AI",
}


@dataclass
class Paragraph:
    """从论文中提取出的一段正文."""

    index: int
    text: str
    source: str  # 来源文件（tex 路径或 pdf 页码区间）
    kind: str = "body"  # body / heading / abstract / caption


@dataclass
class Chunk:
    """发送给朱雀 API 的一个请求文本块，由若干段落拼接而成.

    ranges 记录块内每个段落文本的 (起始偏移, 结束偏移, 段落 index)，
    用于把 API 返回的 segment_labels 位置映射回源段落。
    """

    index: int
    text: str
    paragraph_indices: list[int] = field(default_factory=list)
    ranges: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class JevReview:
    """jev 对单个段落的复核结论."""

    verdict: str  # ai_generated / style_issue / false_positive
    reason: str = ""
    confidence: float = 0.0
    suggestion: str = ""  # 修改建议（仅确认可疑时有）
    error: str = ""  # jev 调用失败时的说明


@dataclass
class ParagraphResult:
    """单个段落的完整检测结果."""

    paragraph: Paragraph
    label: int | None = None  # 朱雀标签；None 表示该段未被检测
    conf: float = 0.0  # 朱雀返回的 softmax AI 概率（人工段接近 0，AI 段接近 1）
    review: JevReview | None = None


@dataclass
class DetectionReport:
    """一次完整检测的汇总结果."""

    source_path: str
    source_kind: str  # tex / pdf
    paragraphs: list[ParagraphResult] = field(default_factory=list)
    # 整体占比（按段落长度加权，0~1）
    human_ratio: float = 0.0
    ai_ratio: float = 0.0
    suspected_ratio: float = 0.0
    zhuque_tokens: int = 0
    makers_tokens: int = 0
    jev_tokens: int = 0
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def flagged(self) -> list[ParagraphResult]:
        """朱雀判为 AI 或疑似 AI 的段落."""
        return [
            p for p in self.paragraphs if p.label in (LABEL_AI, LABEL_SUSPECTED)
        ]
