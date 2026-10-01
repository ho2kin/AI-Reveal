"""LaTeX 项目正文提取。

流程：定位主文件 → 去注释 → 递归展开 \\input/\\include → 截取 document body
→ （可选）截断参考文献 → pylatexenc 语法树遍历，只保留散文段落。
解析失败时回退到正则清洗，保证工具不因个别 .tex 语法问题而崩溃。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexSpecialsNode,
    LatexWalker,
    LatexWalkerError,
)

from ai_reveal.models import Paragraph

# 包裹正文文字、需要递归提取参数内容的宏
_PROSE_MACROS = {
    "text", "textbf", "textit", "textem", "emph", "textrm", "textsf",
    "texttt", "textsc", "textup", "textsl", "textnormal", "textsuperscript",
    "textsubscript", "underline", "mbox", "footnote", "enquote", "hl",
}
# 只递归第一个花括号参数（其余参数是备用文本，避免重复）
_FIRST_ARG_MACROS = {"texorpdfstring"}
# 直接丢弃的宏：引用、标注、排版控制、元信息等
_DROP_MACROS = {
    "cite", "citep", "citet", "citealp", "citeauthor", "citeyear", "nocite",
    "ref", "eqref", "autoref", "pageref", "cref", "Cref", "label",
    "bibliographystyle", "bibliography", "printbibliography", "bibitem",
    "usepackage", "documentclass", "includegraphics", "graphicspath",
    "hspace", "vspace", "vskip", "vfill", "hfill", "newpage", "clearpage",
    "cleardoublepage", "noindent", "indent", "centering", "raggedright",
    "raggedleft", "maketitle", "tableofcontents", "listoffigures",
    "listoftables", "setlength", "setcounter", "addtocounter",
    "renewcommand", "newcommand", "providecommand", "def", "let",
    "input", "include", "includeonly", "hypersetup",
    "title", "author", "date", "thanks", "and", "footnotetext",
    "appendix", "appendices", "backmatter", "frontmatter", "mainmatter",
    "parskip", "tabularnewline", "arraybackslash", "relax", "protect",
    "small", "large", "Large", "LARGE", "huge", "Huge", "normalsize",
    "footnotesize", "scriptsize", "tiny", "begin", "end", "par",
    "bibentry", "nobibliography",
}
# 章节标题宏：提取参数文本作为标题段落
_HEADING_MACROS = {
    "part", "chapter", "section", "subsection", "subsubsection",
    "paragraph", "subparagraph",
}
# 整体跳过的环境：公式、浮动体、代码、参考文献列表等非散文内容
_SKIP_ENVS = {
    "figure", "figure*", "table", "table*", "sidewaystable", "sidewaysfigure",
    "wraptable", "wrapfigure", "equation", "equation*", "align", "align*",
    "alignat", "alignat*", "gather", "gather*", "multline", "multline*",
    "eqnarray", "eqnarray*", "displaymath", "math", "thebibliography",
    "algorithm", "algorithmic", "algorithm2e", "lstlisting", "verbatim",
    "verbatim*", "minted", "tabular", "tabular*", "longtable", "tabbing",
    "picture", "tikzpicture", "filecontents", "filecontents*",
    "split", "cases", "aligned", "gathered", "array", "bmatrix", "pmatrix",
}
# 递归处理的环境（未知环境也递归，宁可多留散文也不漏正文）
_ABSTRACT_ENVS = {"abstract"}
# 图表标题宏
_CAPTION_MACROS = {"caption", "subcaption", "captionof"}

_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_REF_HEADING_RE = re.compile(
    r"\\(?:section|subsection|chapter)\*?\s*\{[^}]*(?:references|bibliography|参考文献)[^}]*\}",
    re.IGNORECASE,
)
_APPENDIX_RE = re.compile(r"\\(?:appendix|appendices|backmatter)\b|\\end\{document\}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")


@dataclass
class TexExtraction:
    root_file: Path
    paragraphs: list = field(default_factory=list)  # list[Paragraph]
    warnings: list[str] = field(default_factory=list)


def extract_tex(path: Path, include_captions: bool = False,
                keep_references: bool = False) -> TexExtraction:
    """从 .tex 文件或 LaTeX 项目目录提取正文段落."""
    path = Path(path)
    warnings: list[str] = []
    if path.is_dir():
        root = _find_root_file(path, warnings)
    else:
        root = path
        if not root.is_file():
            raise FileNotFoundError(f"文件不存在: {root}")

    text = _load_project_text(root, warnings)
    if not keep_references:
        text = _cut_references(text)

    emitter = _BlockEmitter(include_captions=include_captions)
    try:
        nodelist, _, _ = LatexWalker(text).get_latex_nodes()
        for node in nodelist:
            emitter.walk(node)
    except LatexWalkerError as exc:
        warnings.append(f"LaTeX 语法树解析失败，已回退到正则清洗: {exc}")
        emitter.blocks.append(("body", _regex_fallback(text)))
    emitter.finish()

    paragraphs = [
        Paragraph(index=i, text=block_text, source=str(root), kind=kind)
        for i, (kind, block_text) in enumerate(emitter.blocks)
    ]
    return TexExtraction(root_file=root, paragraphs=paragraphs, warnings=warnings)


# ---------------------------------------------------------------------------
# 项目级处理：主文件定位、注释、\input 展开、参考文献截断
# ---------------------------------------------------------------------------

_PREFERRED_ROOT_NAMES = {"main", "thesis", "paper", "article", "report", "manuscript", "body"}


def _find_root_file(directory: Path, warnings: list[str]) -> Path:
    tex_files = [p for p in directory.rglob("*.tex")
                 if not any(part in {"build", "out", "output", "node_modules"} for part in p.parts)]
    if not tex_files:
        raise FileNotFoundError(f"目录下没有找到 .tex 文件: {directory}")
    with_documentclass = [p for p in tex_files
                          if "\\documentclass" in _read(p)]
    candidates = with_documentclass or tex_files
    for name in _PREFERRED_ROOT_NAMES:
        for p in candidates:
            if p.stem.lower() == name:
                return p
    if not with_documentclass:
        warnings.append("未找到含 \\documentclass 的主文件，默认使用 " + str(candidates[0]))
        return candidates[0]
    if len(with_documentclass) > 1:
        warnings.append("多个文件含 \\documentclass，默认使用 " + str(with_documentclass[0]))
    return with_documentclass[0]


def _load_project_text(root: Path, warnings: list[str]) -> str:
    text = _strip_comments(_read(root))
    text = _expand_inputs(text, root.parent, warnings, {root.resolve()})
    m = re.search(r"\\begin\{document\}", text)
    if m:
        end = text.find("\\end{document}", m.end())
        return text[m.end():end if end != -1 else None]
    return text


def _expand_inputs(text: str, base_dir: Path, warnings: list[str], seen: set) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        for cand in (base_dir / name, base_dir / f"{name}.tex"):
            if cand.is_file():
                resolved = cand.resolve()
                if resolved in seen:
                    warnings.append(f"跳过循环引用: {cand}")
                    return ""
                seen.add(resolved)
                sub = _strip_comments(_read(cand))
                return _expand_inputs(sub, cand.parent, warnings, seen)
        warnings.append(f"\\input{{{name}}} 找不到对应文件，已忽略")
        return ""
    return _INPUT_RE.sub(repl, text)


def _strip_comments(text: str) -> str:
    r"""去掉未转义的 % 注释（\% 与 \\ 后的 % 不算注释起始）."""
    out = []
    for line in text.splitlines():
        cut = None
        j = 0
        while j < len(line):
            c = line[j]
            if c == "\\":
                j += 2  # 跳过转义字符或成对的行分隔符
                continue
            if c == "%":
                cut = j
                break
            j += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def _cut_references(text: str) -> str:
    """从 References/参考文献 标题截断到 \\appendix 或文档末尾."""
    m = _REF_HEADING_RE.search(text)
    if not m:
        return text
    stop = _APPENDIX_RE.search(text, m.end())
    if stop:
        return text[:m.start()] + "\n" + text[stop.start():]
    return text[:m.start()]


# ---------------------------------------------------------------------------
# 语法树遍历
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _is_useful(text: str) -> bool:
    if len(text) < 2:
        return False
    return bool(_CJK_RE.search(text) or _WORD_RE.search(text))


class _BlockEmitter:
    """把语法树遍历为 (kind, text) 块序列。kind: body/heading/abstract/caption."""

    def __init__(self, include_captions: bool = False):
        self.blocks: list[tuple[str, str]] = []
        self.include_captions = include_captions
        self._buf: list[str] = []
        self._kind = "body"

    # -- 对外 --
    def walk(self, node) -> None:
        if isinstance(node, LatexCharsNode):
            self._add_text(node.chars)
        elif isinstance(node, LatexCommentNode):
            pass
        elif isinstance(node, LatexGroupNode):
            for child in node.nodelist:
                self.walk(child)
        elif isinstance(node, LatexMathNode):
            pass  # 数学公式不参与检测
        elif isinstance(node, LatexSpecialsNode):
            self._buf.append(" ")
        elif isinstance(node, LatexMacroNode):
            self._walk_macro(node)
        elif isinstance(node, LatexEnvironmentNode):
            self._walk_env(node)
        # 其他节点类型一律忽略

    def finish(self) -> None:
        self.flush()

    # -- 宏 --
    def _walk_macro(self, node: LatexMacroNode) -> None:
        name = node.macroname or ""
        args = self._arg_nodes(node)

        if name in _HEADING_MACROS:
            self.flush()
            title = self._extract_text(args[-1] if args else None)
            if title:
                self.blocks.append(("heading", title))
            return
        if name in _CAPTION_MACROS:
            # 浮动体被整体跳过，caption 在其中被按需提取；正文里偶发的 \captionof 按标题处理
            if self.include_captions:
                self.flush()
                cap = self._extract_text(args[-1] if args else None)
                if cap:
                    self.blocks.append(("caption", cap))
            return
        if name in _DROP_MACROS:
            return
        if name in _FIRST_ARG_MACROS:
            if args:
                self.walk(args[0])
            return
        if name in _PROSE_MACROS:
            for arg in args:
                self.walk(arg)
            return
        if len(name) == 1 and not name.isalpha():
            # 转义标点 \% \& \_ 等，按字面保留
            self._buf.append(name)
            return
        if name == "item":
            self.flush()
            return
        # 其他未知宏：丢弃宏名本身，其后的参数组会作为独立 Group 节点被正常递归

    @staticmethod
    def _arg_nodes(node: LatexMacroNode) -> list:
        argd = node.nodeargd
        if argd is None or not hasattr(argd, "argnlist"):
            return []
        return [a for a in argd.argnlist if a is not None]

    # -- 环境 --
    def _walk_env(self, node: LatexEnvironmentNode) -> None:
        name = node.envname or ""
        if name in _SKIP_ENVS:
            if self.include_captions and re.match(r"(figure|table|wrap|sideways)", name):
                self._emit_captions(node.nodelist)
            return
        if name in _ABSTRACT_ENVS:
            self.flush()
            prev_kind = self._kind
            self._kind = "abstract"
            for child in node.nodelist:
                self.walk(child)
            self.flush()
            self._kind = prev_kind
            return
        for child in node.nodelist:
            self.walk(child)

    def _emit_captions(self, nodelist: list) -> None:
        """在（被跳过的）浮动体节点列表中提取 caption 文本."""
        if not self.include_captions:
            return
        for i, node in enumerate(nodelist):
            if isinstance(node, LatexMacroNode) and (node.macroname or "") in _CAPTION_MACROS:
                args = self._arg_nodes(node)
                arg_node = args[-1] if args else None
                if arg_node is None and i + 1 < len(nodelist):
                    # 默认上下文中 \caption 的 argspec 为空，参数组是紧随其后的兄弟节点
                    sibling = nodelist[i + 1]
                    if isinstance(sibling, LatexGroupNode):
                        arg_node = sibling
                cap = self._extract_text(arg_node)
                if cap:
                    self.blocks.append(("caption", cap))
            elif isinstance(node, (LatexGroupNode, LatexEnvironmentNode)):
                self._emit_captions(getattr(node, "nodelist", []))

    # -- 文本与块 --
    def _add_text(self, s: str) -> None:
        if "\n\n" in s:
            parts = s.split("\n\n")
            for i, part in enumerate(parts):
                self._buf.append(part)
                if i < len(parts) - 1:
                    self.flush()
        else:
            self._buf.append(s)

    def flush(self) -> None:
        text = _normalize("".join(self._buf))
        self._buf = []
        if _is_useful(text):
            self.blocks.append((self._kind, text))

    def _extract_text(self, node) -> str:
        """把单个参数节点提取为纯文本（用于标题/caption）."""
        if node is None:
            return ""
        sub = _BlockEmitter(include_captions=self.include_captions)
        sub.walk(node)
        sub.finish()
        return _normalize(" ".join(t for _, t in sub.blocks))


# ---------------------------------------------------------------------------
# 兜底
# ---------------------------------------------------------------------------

def _regex_fallback(text: str) -> str:
    """语法树解析失败时的粗略清洗，保住可读正文."""
    text = re.sub(r"\\begin\{(equation|align|gather|multline|eqnarray|figure|table|tabular|longtable|thebibliography|verbatim|lstlisting|algorithm|algorithmic|tikzpicture)\*?\}.*?\\end\{\1\*?\}", " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.S)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}~]", " ", text)
    return _normalize(text)


def _read(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")
