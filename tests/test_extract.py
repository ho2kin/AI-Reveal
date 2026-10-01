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
        assert "如式 (1)所示" in text or "损失函数由各样本损失累加而成" in text  # 公式后的正文
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

    def test_restores_references_citations_and_math(self, extracted):
        """交叉引用/文献引用/行内公式应还原为可读内容，而非删除留残句."""
        text = all_text(extracted)
        assert "如式 (1)所示" in text  # \eqref{eq:loss}（~ 为不换行空格，忠实还原）
        assert "Figure 1" in text  # \autoref{fig:arch}
        assert "第 1节" in text  # 第~\ref{sec:intro}节
        assert "[1]" in text and "[2]" in text and "[3]" in text  # \cite 按首次出现编号
        assert "其中 n 为样本数量" in text  # 行内数学 $n$
        assert "γ" in text  # $\gamma$ 经 Unicode 还原

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

    def test_position_is_start_and_length(self):
        """朱雀 position=[起始偏移, 长度]，块内各段都应拿到标注（回归测试）."""
        from ai_reveal.detector import _map_segment
        from ai_reveal.models import Chunk

        chunk = Chunk(index=0, text="甲" * 10 + "\n\n" + "乙" * 10,
                      paragraph_indices=[0, 1],
                      ranges=[(0, 10, 0), (12, 22, 1)])
        # 第二个 segment 起点 5 长度 20（旧实现会因 end<start 被丢弃）
        seg = {"position": [5, 20], "label": 1, "conf": 0.9, "text": chunk.text[5:25]}
        overlaps = dict(_map_segment(seg, chunk))
        assert overlaps == {0: 5, 1: 10}
        # 起点落在段间空隙时仍应映射到两侧段落
        seg_gap = {"position": [9, 5], "label": 0, "conf": 0.5, "text": chunk.text[9:14]}
        assert _map_segment(seg_gap, chunk)

    def test_position_missing_falls_back_to_text(self):
        from ai_reveal.detector import _map_segment
        from ai_reveal.models import Chunk

        chunk = Chunk(index=0, text="甲" * 10 + "\n\n" + "乙" * 10,
                      paragraph_indices=[0, 1],
                      ranges=[(0, 10, 0), (12, 22, 1)])
        seg = {"position": None, "label": 1, "conf": 0.9, "text": "乙乙乙"}
        assert dict(_map_segment(seg, chunk)) == {1: 3}

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


class TestConsolidate:
    def _para(self, text, kind="body", **kw):
        return Paragraph(index=0, text=text, source="x", kind=kind, **kw)

    def test_headings_become_sections_not_units(self):
        from ai_reveal.segment import consolidate
        paras = [
            self._para("引言", kind="heading"),
            self._para("长正文" * 60),
            self._para("方法", kind="heading"),
            self._para("长方法" * 60),
        ]
        out = consolidate(paras)
        assert all(p.kind != "heading" for p in out)  # 标题不作为检测单元
        assert [p.section for p in out] == ["引言", "方法"]
        assert len(out) == 2 and out[0].index == 0 and out[1].index == 1

    def test_short_merged_into_next(self):
        from ai_reveal.segment import consolidate
        out = consolidate([self._para("短引句："), self._para("长" * 130)])
        assert len(out) == 1
        assert out[0].text.startswith("短引句： 长")

    def test_chain_merge_until_long_enough(self):
        from ai_reveal.segment import consolidate
        out = consolidate([
            self._para("碎片一"), self._para("碎片二"), self._para("长" * 130),
        ])
        assert len(out) == 1  # 连续碎片链式合并到足够长

    def test_trailing_short_merges_backward(self):
        from ai_reveal.segment import consolidate
        out = consolidate([self._para("长" * 130), self._para("结尾碎片")])
        assert len(out) == 1

    def test_caption_blocks_merging(self):
        from ai_reveal.segment import consolidate
        out = consolidate([
            self._para("长" * 130), self._para("图注", kind="caption"), self._para("长" * 130),
        ])
        assert len(out) == 3  # 图注独立，两侧正文不跨它合并
        assert out[1].kind == "caption"

    def test_lone_short_paragraph_kept(self):
        from ai_reveal.segment import consolidate
        out = consolidate([
            self._para("甲节", kind="heading"), self._para("独短段"),
            self._para("乙节", kind="heading"), self._para("长" * 130),
        ])
        assert len(out) == 2  # 节内无邻居的短段落保持原样
        assert out[0].text == "独短段"

    def test_duplicate_headings_do_not_merge_across(self):
        from ai_reveal.segment import consolidate
        out = consolidate([
            self._para("讨论", kind="heading"), self._para("长" * 130),
            self._para("讨论", kind="heading"), self._para("另一段" * 60),
        ])
        assert len(out) == 2  # 重名章节之间不跨节合并


class TestQuota:
    def test_record_and_summary(self, tmp_path, monkeypatch):
        import ai_reveal.quota as quota
        monkeypatch.setattr(quota, "_STATE_DIR", tmp_path)
        monkeypatch.setattr(quota, "_STATE_FILE", tmp_path / "usage.json")
        key = "sk-test-key-000"
        before = quota.month_used_tokens(key)
        total = quota.record_usage(key, 100)
        assert total == before + 100
        info = quota.summary_with_run(key, this_run_tokens=50) if hasattr(
            quota, "summary_with_run") else quota.quota_summary(key, this_run_tokens=50)
        assert info["used"] >= 150
        assert info["remaining"] == info["free"] - info["used"]
        assert 0 <= info["used_pct"] <= 100

    def test_month_rollover(self, tmp_path, monkeypatch):
        import ai_reveal.quota as quota
        monkeypatch.setattr(quota, "_STATE_DIR", tmp_path)
        monkeypatch.setattr(quota, "_STATE_FILE", tmp_path / "usage.json")
        key = "sk-test-key-111"
        quota.record_usage(key, 200)
        # 模拟账本里是上个月的记录
        import json
        state = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
        for entry in state.values():
            entry["month"] = "2000-01"
        (tmp_path / "usage.json").write_text(json.dumps(state), encoding="utf-8")
        assert quota.month_used_tokens(key) == 0
