"""LaTeX 提取器与分段逻辑单元测试（使用 examples/ 下的样例项目）."""

from pathlib import Path

import pytest

from ai_reveal.models import Paragraph
from ai_reveal.segment import build_chunks, _split_long_text
from ai_reveal.tex_extract import extract_tex, _strip_comments

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def extracted():
    return extract_tex(EXAMPLES / "sample.tex")


def all_text(extraction) -> str:
    return "\n".join(p.text for p in extraction.paragraphs)


class TestTexExtract:
    def test_keeps_body_prose(self, extracted):
        text = all_text(extracted)
        assert "联邦学习允许各方在不共享原始数据" in text  # 来自 \input 展开文件
        assert "本文提出一种基于边缘计算的联邦学习" in text  # abstract
        assert "如公式所示" in text  # 行内公式之后的正文
        assert "补充实验结果表明各组件均有效" in text  # 参考文献后的附录应保留

    def test_drops_comments(self, extracted):
        assert "不应出现在检测结果中" not in all_text(extracted)

    def test_drops_citation_keys_and_preamble(self, extracted):
        text = all_text(extracted)
        assert "vaswani2017" not in text
        assert "mcmahan2017" not in text
        assert "documentclass" not in text
        assert "usepackage" not in text
        assert "基于边缘计算的联邦学习优化方法研究" not in text  # \title 被丢弃

    def test_drops_math(self, extracted):
        text = all_text(extracted)
        assert "mathcal" not in text
        assert "ell(f(x_i)" not in text

    def test_drops_references_section(self, extracted):
        assert "参考文献" not in all_text(extracted)

    def test_captions_excluded_by_default_included_on_flag(self, extracted):
        assert "系统总体架构图" not in all_text(extracted)
        flagged = extract_tex(EXAMPLES / "sample.tex", include_captions=True)
        assert "系统总体架构图" in all_text(flagged)

    def test_paragraph_kinds(self, extracted):
        kinds = {p.kind for p in extracted.paragraphs}
        assert "heading" in kinds
        assert "abstract" in kinds
        assert "body" in kinds

    def test_directory_input_finds_root(self):
        extraction = extract_tex(EXAMPLES)
        assert extraction.root_file.name == "sample.tex"
        assert extraction.paragraphs


class TestStripComments:
    def test_simple_comment(self):
        assert _strip_comments("正文 % 注释") == "正文 "

    def test_escaped_percent_kept(self):
        assert _strip_comments("100\\% 的数据") == "100\\% 的数据"

    def test_linebreak_then_comment(self):
        assert _strip_comments("上一行\\\\% 注释") == "上一行\\\\"


class TestSegment:
    def _paras(self, *texts):
        return [Paragraph(index=i, text=t, source="x") for i, t in enumerate(texts)]

    def test_merges_within_limit(self):
        chunks = build_chunks(self._paras("段落一" * 100, "段落二" * 100), chunk_chars=1000)
        assert len(chunks) == 1
        assert chunks[0].paragraph_indices == [0, 1]
        # ranges 与文本一致
        start, end, idx = chunks[0].ranges[1]
        assert chunks[0].text[start:end] == "段落二" * 100
        assert idx == 1

    def test_splits_across_limit(self):
        paras = self._paras("甲" * 450, "乙" * 450, "丙" * 450)
        chunks = build_chunks(paras, chunk_chars=1000)
        assert len(chunks) == 2  # 甲+乙=902 放得下，丙溢出
        assert chunks[0].paragraph_indices == [0, 1]
        assert chunks[1].paragraph_indices == [2]

    def test_long_paragraph_hard_split(self):
        chunks = build_chunks(self._paras("很长的一段。" * 500), chunk_chars=500)
        assert all(len(c.text) <= 500 for c in chunks)
        assert all(c.paragraph_indices == [0] for c in chunks)

    def test_split_long_text_sentence_boundary(self):
        pieces = _split_long_text("第一句。第二句。" * 3, 20)
        assert all(len(p) <= 20 for p in pieces)
        assert "".join(pieces).replace(" ", "") == "第一句。第二句。" * 3
