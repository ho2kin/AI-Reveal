"""命令行入口：ai-reveal detect <tex|pdf|目录> [选项]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_reveal import __version__
from ai_reveal.advisor import JevAdvisor, review_verdict_name
from ai_reveal.config import MissingApiKeyError, REQUEST_TIMEOUT, load_api_key
from ai_reveal.detector import ZhuqueClient, ZhuqueError, run_detection
from ai_reveal.models import (
    LABEL_AI,
    LABEL_NAMES,
    LABEL_SUSPECTED,
    DetectionReport,
    Paragraph,
)
from ai_reveal.pdf_extract import extract_pdf
from ai_reveal.quota import quota_summary, record_usage
from ai_reveal.report import write_reports
from ai_reveal.segment import build_chunks, consolidate
from ai_reveal.tex_extract import extract_tex


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-reveal",
        description="LaTeX/PDF 论文 AI 率检测工具：提取正文 → 朱雀模型检测 → jev 复核建议",
    )
    parser.add_argument("--version", action="version", version=f"ai-reveal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="检测论文 AI 率")
    detect.add_argument("path", help=".tex 文件、LaTeX 项目目录或 .pdf 文件")
    detect.add_argument("--api-key", help="覆盖环境变量 / .env 中的 API Key")
    detect.add_argument("--chunk-chars", type=int, default=4000,
                        help="单次请求文本块大小（默认 4000 字符）")
    detect.add_argument("--min-para-chars", type=int, default=120,
                        help="低于该长度的段落与同章节相邻段落合并（默认 120，0 关闭合并）")
    detect.add_argument("--include-captions", action="store_true",
                        help="保留图表标题（默认丢弃浮动体）")
    detect.add_argument("--keep-references", action="store_true",
                        help="保留参考文献部分（默认去除）")
    detect.add_argument("--no-jev", action="store_true", help="跳过 jev 复核与修改建议")
    detect.add_argument("--dry-run", action="store_true",
                        help="只做提取与分段，不调用 API、不消耗 token")
    detect.add_argument("-o", "--output", default="ai-reveal-report",
                        help="报告输出目录（默认 ./ai-reveal-report）")
    detect.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT,
                        help=f"单次请求超时秒数（默认 {REQUEST_TIMEOUT}）")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    args = _build_parser().parse_args(argv)
    path = Path(args.path)

    try:
        if path.is_dir() or path.suffix.lower() == ".tex":
            extraction = extract_tex(
                path,
                include_captions=args.include_captions,
                keep_references=args.keep_references,
            )
            paragraphs, warnings = extraction.paragraphs, extraction.warnings
            source_kind = "tex"
        elif path.is_file() and path.suffix.lower() == ".pdf":
            paragraphs, warnings = extract_pdf(path, keep_references=args.keep_references)
            source_kind = "pdf"
        else:
            print(f"错误：不支持的输入 {path}（需要 .tex / .pdf / 目录）", file=sys.stderr)
            return 2
    except Exception as exc:  # 提取层保证不崩，这里兜底给出清晰错误
        print(f"错误：提取正文失败：{exc}", file=sys.stderr)
        return 2

    if not paragraphs:
        print("错误：未能从输入中提取出有效正文内容", file=sys.stderr)
        for w in warnings:
            print(f"  警告: {w}", file=sys.stderr)
        return 2

    if args.min_para_chars > 0:
        paragraphs = consolidate(paragraphs, min_chars=args.min_para_chars)
    chunks = build_chunks(paragraphs, chunk_chars=max(args.chunk_chars, 200))
    total_chars = sum(len(p.text) for p in paragraphs)

    print(f"提取完成：{len(paragraphs)} 个段落 / {total_chars} 字符，"
          f"分为 {len(chunks)} 个请求块（来源: {source_kind}）")
    for w in warnings:
        print(f"  警告: {w}")

    if args.dry_run:
        print("\n--dry-run 模式：前 5 个段落预览 ——")
        for para in paragraphs[:5]:
            print(f"  [{para.kind}] {para.text[:80]}{'…' if len(para.text) > 80 else ''}")
        try:
            api_key = load_api_key(args.api_key)
            info = quota_summary(api_key)
            print(f"\n未调用任何 API。本月额度: 已用 {info['used']:,} / "
                  f"{info['free']:,} ({info['used_pct']:.2f}%) · 剩余 {info['remaining']:,}")
        except MissingApiKeyError:
            print("\n未调用任何 API。")
        print("确认提取质量后可去掉 --dry-run 执行检测。")
        return 0

    try:
        api_key = load_api_key(args.api_key)
    except MissingApiKeyError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    client = ZhuqueClient(api_key, timeout=args.timeout)
    try:
        report = run_detection(
            paragraphs, chunks, client,
            source_path=str(path), source_kind=source_kind,
            progress=lambda done, total: print(f"  朱雀检测进度: {done}/{total} 块"),
        )
    except ZhuqueError as exc:
        print(f"错误：朱雀检测失败：{exc}", file=sys.stderr)
        return 1

    report.warnings.extend(warnings)
    if not args.no_jev and report.flagged():
        print("  jev 复核中……")
        JevAdvisor(api_key, timeout=args.timeout).run(report)

    # 累计进本地额度账本并汇总额度信息（供终端与报告展示）
    record_usage(api_key, report.makers_tokens)
    report.extras["quota"] = quota_summary(api_key)

    _print_summary(report)
    json_path, html_path = write_reports(report, Path(args.output))
    print(f"\n报告已生成：\n  {json_path.resolve()}\n  {html_path.resolve()}")
    return 0


def _print_summary(report: DetectionReport) -> None:
    print("\n════════ AI-Reveal 检测结果 ════════")
    print(f"来源: {report.source_path} ({report.source_kind})")
    print(f"AI 生成占比: {report.ai_ratio * 100:.1f}%   "
          f"疑似 AI: {report.suspected_ratio * 100:.1f}%   "
          f"人工: {report.human_ratio * 100:.1f}%")
    print(f"token 消耗: 朱雀 {report.zhuque_tokens} | "
          f"Makers 计费 {report.makers_tokens} | jev {report.jev_tokens}")
    quota = report.extras.get("quota")
    if quota:
        print(f"本月额度: 已用 {quota['used']:,} / {quota['free']:,} "
              f"({quota['used_pct']:.2f}%) · 剩余 {quota['remaining']:,} "
              f"({100 - quota['used_pct']:.2f}%)")

    flagged = [r for r in report.paragraphs
               if r.label in (LABEL_AI, LABEL_SUSPECTED)]
    if flagged:
        flagged.sort(key=lambda r: r.conf, reverse=True)
        print(f"\nTop 可疑段落（{min(5, len(flagged))}/{len(flagged)}）:")
        for r in flagged[:5]:
            preview = r.paragraph.text[:60] + ("…" if len(r.paragraph.text) > 60 else "")
            review = ""
            if r.review is not None and r.review.verdict != "unreviewed":
                review = f" · jev: {review_verdict_name(r.review.verdict)}"
            print(f"  [{LABEL_NAMES.get(r.label, '未知')} {r.conf * 100:.0f}%] "
                  f"#{r.paragraph.index} {preview}{review}")
    else:
        print("\n未发现 AI / 疑似 AI 段落。")


if __name__ == "__main__":
    sys.exit(main())
