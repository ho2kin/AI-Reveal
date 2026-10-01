"""jev 决策模型复核：对朱雀标记为 AI/疑似AI 的段落做结构化二次判定.

jev 通过专用端点 /v1/systemone 调用（POST {state, questions}），是纯分类模型：
每个可疑段落作为一道 choice 题，从 ai_generated / style_issue / false_positive
中判定。修改建议按判定类别套用改写策略模板生成（jev 不生成自由文本）。
任何失败都会降级：段落保留朱雀原始判定，报告中记录警告。
"""

from __future__ import annotations

import re
import time

import requests

from ai_reveal.config import (
    JEV_MODEL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    SYSTEMONE_URL,
)
from ai_reveal.models import DetectionReport, JevReview

MAX_REVIEW_PARAGRAPHS = 24   # 最多复核段落数，控制 token 消耗
BATCH_SIZE = 6               # 每次请求携带的题目数
MIN_REVIEW_CHARS = 15        # 短于此的段落（多为标题）不值得复核

_VERDICT_NAMES = {
    "ai_generated": "AI生成",
    "style_issue": "风格套路",
    "false_positive": "误报",
}

# 三分类判定标准（作为 jev choice 题的 criteria）
_CRITERIA = {
    "ai_generated": (
        "明显具有AI生成文本特征：套话堆砌、句式过度均匀、泛泛而谈、"
        "缺乏具体数据/实验细节/个人分析支撑"
    ),
    "style_issue": (
        "仅写作风格模板化或八股化（公式化开头、通用过渡句），"
        "但内容具体、大概率人工撰写"
    ),
    "false_positive": (
        "包含具体数据、实验结果、案例分析或明显人工写作痕迹，属于误报"
    ),
}

_INSTRUCTIONS = (
    "这是从学术论文中提取、被朱雀AIGC检测标记的段落。"
    "请依据判定标准，从三个选项中选出最符合该段落情况的一项。"
)

# 按判定类别生成的修改建议模板
_SUGGESTION_TEMPLATES = {
    "ai_generated": (
        "该段被判定为 AI 生成特征明显，建议：1) 补充具体实验数据、参数或案例细节；"
        "2) 打破均匀句式，长短句结合，加入自己的分析推理过程；"
        "3) 结合引用文献展开评述，替换泛化表述（如“取得了显著进展”）。"
    ),
    "style_issue": (
        "该段内容本身可信，但表达模板化，建议调整：替换“随着…的快速发展”"
        "“本文提出”等高频套话，用更具体的陈述替代通用过渡句，增强段落之间的逻辑衔接。"
    ),
    "false_positive": "",
    "unreviewed": "",
}


class JevError(RuntimeError):
    """jev 调用失败."""


class JevAdvisor:
    def __init__(self, api_key: str, timeout: int = REQUEST_TIMEOUT,
                 max_retries: int = 2, model: str = JEV_MODEL,
                 url: str = SYSTEMONE_URL):
        self.url = url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def run(self, report: DetectionReport, progress=None) -> None:
        flagged = report.flagged()
        targets = [r for r in flagged if len(r.paragraph.text) >= MIN_REVIEW_CHARS]
        if not targets:
            return
        if len(flagged) > MAX_REVIEW_PARAGRAPHS:
            report.warnings.append(
                f"可疑段落共 {len(flagged)} 个，jev 仅复核前 {MAX_REVIEW_PARAGRAPHS} 个以控制额度")
        targets = targets[:MAX_REVIEW_PARAGRAPHS]
        batches = [targets[i:i + BATCH_SIZE] for i in range(0, len(targets), BATCH_SIZE)]
        done = 0
        for batch in batches:
            self._review_batch(batch, report)
            done += len(batch)
            if progress:
                progress(done, len(targets))

    # ------------------------------------------------------------------
    def _review_batch(self, batch: list, report: DetectionReport) -> None:
        """一次 systemone 请求复核一批段落：每段一道 choice 题."""
        questions = {
            str(result.paragraph.index): {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": _CRITERIA,
            }
            for result in batch
        }
        state = {
            "task": "论文AI率复核",
            "note": "以下 paragraphs 中的每一段独立判定，id 对应题目 qid",
            "paragraphs": {
                str(result.paragraph.index): result.paragraph.text
                for result in batch
            },
        }
        try:
            answers, tokens = self._ask(state, questions)
        except JevError as exc:
            report.warnings.append(f"jev 复核失败（{len(batch)} 段保留朱雀判定）: {exc}")
            for result in batch:
                result.review = JevReview(verdict="unreviewed", error=str(exc))
            return
        report.jev_tokens += tokens
        for result in batch:
            qid = str(result.paragraph.index)
            answer = answers.get(qid)
            if not answer or not answer.get("choice"):
                result.review = JevReview(verdict="unreviewed", error="jev 未返回该段结论")
                continue
            verdict = str(answer["choice"]).lower()
            result.review = JevReview(
                verdict=verdict,
                confidence=_to_float(answer.get("confidence")),
                suggestion=_SUGGESTION_TEMPLATES.get(verdict, ""),
            )

    def _ask(self, state, questions: dict) -> tuple[dict, int]:
        payload = {"model": self.model, "state": state, "questions": questions}
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(self.url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = JevError(f"网络错误: {exc}")
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    tokens = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
                    return data.get("answers") or {}, tokens
                if resp.status_code in {429, 500, 502, 503, 504}:
                    last_error = JevError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                else:
                    raise JevError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise last_error or JevError("未知错误")


def _to_float(value) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def review_verdict_name(verdict: str) -> str:
    return _VERDICT_NAMES.get(verdict, "未复核" if verdict == "unreviewed" else verdict)
