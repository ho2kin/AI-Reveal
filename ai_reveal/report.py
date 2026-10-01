"""检测报告生成：report.json（机器可读）+ report.html（自包含单文件，离线可看）."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from jinja2 import Environment

from ai_reveal import __version__
from ai_reveal.models import (
    LABEL_AI,
    LABEL_HUMAN,
    LABEL_SUSPECTED,
    DetectionReport,
)
from ai_reveal.advisor import review_verdict_name

_LABEL_NAMES = {LABEL_HUMAN: "人工", LABEL_AI: "AI", LABEL_SUSPECTED: "疑似AI"}
_LABEL_CLASSES = {LABEL_HUMAN: "human", LABEL_AI: "ai", LABEL_SUSPECTED: "suspected"}
_KIND_NAMES = {"body": "正文", "heading": "标题", "abstract": "摘要", "caption": "图注"}

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-Reveal 检测报告</title>
<style>
  :root { --red:#b91c1c; --red-bg:#fee2e2; --yellow:#a16207; --yellow-bg:#fef9c3;
          --green:#15803d; --green-bg:#dcfce7; --gray:#6b7280; --gray-bg:#f3f4f6;
          --ink:#1f2937; --line:#e5e7eb; }
  * { box-sizing: border-box; }
  body { margin:0; padding:32px 20px 60px; background:#f8fafc; color:var(--ink);
         font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
  .wrap { max-width:880px; margin:0 auto; }
  h1 { font-size:24px; margin:0 0 4px; }
  .meta { color:var(--gray); font-size:13px; margin-bottom:24px; }
  .cards { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:20px; }
  .card { flex:1 1 160px; border-radius:12px; padding:18px 20px; }
  .card .num { font-size:32px; font-weight:700; line-height:1.2; }
  .card .lab { font-size:13px; opacity:.85; }
  .card.ai { background:var(--red-bg); color:var(--red); }
  .card.suspected { background:var(--yellow-bg); color:var(--yellow); }
  .card.human { background:var(--green-bg); color:var(--green); }
  .tokens { color:var(--gray); font-size:12.5px; margin-bottom:26px; }
  .legend { display:flex; gap:8px; align-items:center; font-size:12.5px; color:var(--gray); margin-bottom:12px; }
  .badge { display:inline-block; padding:1px 9px; border-radius:99px; font-size:12px; font-weight:600; }
  .badge.ai { background:var(--red-bg); color:var(--red); }
  .badge.suspected { background:var(--yellow-bg); color:var(--yellow); }
  .badge.human { background:var(--green-bg); color:var(--green); }
  .badge.none { background:var(--gray-bg); color:var(--gray); }
  .para { background:#fff; border:1px solid var(--line); border-left:5px solid var(--gray-bg);
          border-radius:10px; padding:14px 18px; margin-bottom:12px; }
  .para.ai { border-left-color:var(--red); }
  .para.suspected { border-left-color:var(--yellow); }
  .para.human { border-left-color:var(--green); }
  .phead { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:8px;
           font-size:12.5px; color:var(--gray); }
  .ptext { margin:0; white-space:pre-wrap; word-break:break-word; }
  .review { margin-top:10px; padding:10px 14px; border-radius:8px; background:var(--gray-bg); font-size:13.5px; }
  .review.warn { background:var(--yellow-bg); }
  .review b { font-weight:600; }
  .warnings { margin-top:30px; padding:14px 18px; background:#fffbeb; border:1px solid #fde68a;
              border-radius:10px; font-size:13px; color:#92400e; }
  .warnings ul { margin:6px 0 0; padding-left:18px; }
  footer { margin-top:36px; color:var(--gray); font-size:12px; text-align:center; }
</style>
</head>
<body>
<div class="wrap">
  <h1>AI-Reveal 论文 AI 率检测报告</h1>
  <div class="meta">来源：{{ source_path }} · 类型：{{ source_kind }} · 生成时间：{{ generated_at }} · 已检测 {{ detected_count }}/{{ total_count }} 段</div>

  <div class="cards">
    <div class="card ai"><div class="num">{{ "%.1f"|format(ai_pct) }}%</div><div class="lab">AI 生成占比</div></div>
    <div class="card suspected"><div class="num">{{ "%.1f"|format(suspected_pct) }}%</div><div class="lab">疑似 AI 占比</div></div>
    <div class="card human"><div class="num">{{ "%.1f"|format(human_pct) }}%</div><div class="lab">人工占比</div></div>
  </div>
  <div class="tokens">朱雀 tokens：{{ zhuque_tokens }} ｜ Makers 计费 tokens：{{ makers_tokens }} ｜ jev tokens：{{ jev_tokens }}{% if quota %}<br>本月免费额度：已用 {{ "{:,}".format(quota.used) }} / {{ "{:,}".format(quota.free) }} ({{ "%.2f"|format(quota.used_pct) }}%) · 剩余 {{ "{:,}".format(quota.remaining) }} ({{ "%.2f"|format(100 - quota.used_pct) }}%){% endif %}</div>

  <div class="legend">图例：<span class="badge ai">AI</span><span class="badge suspected">疑似AI</span><span class="badge human">人工</span><span class="badge none">未检测</span></div>

  {% for p in paragraphs %}
  <article class="para {{ p.cls }}">
    <div class="phead">
      <span class="badge {{ p.cls }}">{{ p.label_name }}</span>
      {% if p.label is not none %}<span>AI 概率 {{ "%.0f"|format(p.conf_pct) }}%</span>{% endif %}
      <span>段落 #{{ p.index }} · {{ p.source }}{% if p.kind != 'body' %} · {{ p.kind_name }}{% endif %}</span>
      {% if p.review_badge %}<span class="badge {{ p.review_cls }}">jev：{{ p.review_badge }}</span>{% endif %}
    </div>
    <p class="ptext">{{ p.text }}</p>
    {% if p.reason %}<div class="review"><b>复核理由：</b>{{ p.reason }}</div>{% endif %}
    {% if p.suggestion %}<div class="review warn"><b>修改建议：</b>{{ p.suggestion }}</div>{% endif %}
    {% if p.review_error %}<div class="review"><b>复核失败：</b>{{ p.review_error }}</div>{% endif %}
  </article>
  {% endfor %}

  {% if warnings %}
  <div class="warnings"><b>警告（{{ warnings|length }}）</b>
    <ul>{% for w in warnings %}<li>{{ w }}</li>{% endfor %}</ul>
  </div>
  {% endif %}

  <footer>AI-Reveal v{{ version }} · 检测结果为模型置信度参考，不构成学术不端认定</footer>
</div>
</body>
</html>
"""


def write_reports(report: DetectionReport, out_dir: Path) -> tuple[Path, Path]:
    """写出 JSON 与 HTML 报告，返回两个文件路径."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    json_path.write_text(
        json.dumps(_json_payload(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    html_path = out_dir / "report.html"
    env = Environment(autoescape=True)
    html_path.write_text(
        env.from_string(_HTML_TEMPLATE).render(**_html_view(report)),
        encoding="utf-8",
    )
    return json_path, html_path


def _json_payload(report: DetectionReport) -> dict:
    data = asdict(report)
    data["version"] = __version__
    data["generated_at"] = datetime.now().isoformat(timespec="seconds")
    data["summary"] = {
        "ai_ratio": round(report.ai_ratio, 4),
        "suspected_ratio": round(report.suspected_ratio, 4),
        "human_ratio": round(report.human_ratio, 4),
    }
    return data


def _html_view(report: DetectionReport) -> dict:
    paragraphs = []
    for result in report.paragraphs:
        para = result.paragraph
        cls = _LABEL_CLASSES.get(result.label, "none") if result.label is not None else "none"
        review = result.review
        review_badge = review_cls = ""
        reason = suggestion = review_error = ""
        if review is not None:
            review_badge = review_verdict_name(review.verdict)
            review_cls = {"ai_generated": "ai", "style_issue": "suspected",
                          "false_positive": "human"}.get(review.verdict, "none")
            reason = review.reason or ""
            suggestion = review.suggestion or ""
            review_error = review.error or ""
        paragraphs.append({
            "index": para.index,
            "source": para.source,
            "kind": para.kind,
            "kind_name": _KIND_NAMES.get(para.kind, para.kind),
            "text": para.text,
            "cls": cls,
            "label_name": _LABEL_NAMES.get(result.label, "未检测") if result.label is not None else "未检测",
            "label": result.label,
            "conf_pct": (result.conf or 0.0) * 100,
            "review_badge": review_badge,
            "review_cls": review_cls,
            "reason": reason,
            "suggestion": suggestion,
            "review_error": review_error,
        })
    return {
        "source_path": report.source_path,
        "source_kind": report.source_kind,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ai_pct": report.ai_ratio * 100,
        "suspected_pct": report.suspected_ratio * 100,
        "human_pct": report.human_ratio * 100,
        "zhuque_tokens": report.zhuque_tokens,
        "makers_tokens": report.makers_tokens,
        "jev_tokens": report.jev_tokens,
        "quota": report.extras.get("quota"),
        "detected_count": sum(1 for r in report.paragraphs if r.label is not None),
        "total_count": len(report.paragraphs),
        "paragraphs": paragraphs,
        "warnings": report.warnings,
        "version": __version__,
    }
