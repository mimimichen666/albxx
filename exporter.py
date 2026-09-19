"""
exporter.py —— 报告导出模块（成员C · 模块1）
================================
职责：把 Markdown 格式的综述报告导出为 Word (.docx) 和 LaTeX+BibTeX
（可直接编译的 zip 包），与 Streamlit 前端完全解耦——本模块只做
纯数据转换，不 import streamlit / 不读会话状态，可独立测试。

设计要点（对应任务要求）：
  1. 解耦业务逻辑：对外只暴露 md_to_docx() / build_latex_package()，
     输入是 Markdown 字符串 + 文献元数据列表（统一中间格式），
     与报告的来源（最终综述/模式检索报告）无关。
  2. 多引用处理：正文中的 [arXiv:2302.00232] 引用标记，
     LaTeX 侧转换为 \\cite{...} 并在 refs.bib 中生成对应条目；
     Word 侧保留原文字。
  3. 长文本边界：表格单元格超长文本自动换行（Word 表格默认行为；
     LaTeX 用 p{...} 定宽列），单元格内的换行符转为空格避免表格错乱。

输入的文献元数据统一为 dict 列表（从 papers_meta.json 或
mode_assistant 的 hits 转换而来），字段:
  {"id": "2302.00232", "title": str, "authors": [str], "year": int,
   "venue": str|None, "doi": str|None, "url": str|None, "source_type": str}
其中 id 在 LaTeX 侧兼作 bibkey（清洗为合法标识符）。

Markdown 子集解析（覆盖本项目报告实际使用的语法）:
  #/##/#### 标题 · **粗体** · *斜体* · `行内代码`
  |表格| · - 列表 · > 引用块 · --- 分隔线 · 普通段落
"""

import io
import re
import os
import zipfile

# ================================================================
# 一、Markdown 词法分析（行级 + 行内标记）
# ================================================================

def _parse_inline(text: str) -> list[tuple[str, str]]:
    """
    行内标记解析: 把一行文本拆成 [(样式, 内容), ...] 片段序列。
    样式: 'plain' / 'bold' / 'italic' / 'code'

    实现: 逐个正则扫描三种标记，取最早命中者切分（无依赖的轻量实现，
    避免引入 markdown 库——报告语法子集是已知的）。
    """
    tokens = []
    pos = 0
    patterns = [
        (re.compile(r"\*\*(.+?)\*\*"), "bold"),
        (re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)"), "italic"),
        (re.compile(r"`([^`\n]+?)`"), "code"),
    ]
    while pos < len(text):
        best = None  # (start, end, style, inner)
        for pat, style in patterns:
            m = pat.search(text, pos)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), m.end(), style, m.group(1))
        if best is None:
            tokens.append(("plain", text[pos:]))
            break
        s, e, style, inner = best
        if s > pos:
            tokens.append(("plain", text[pos:s]))
        tokens.append((style, inner))
        pos = e
    return tokens


def _split_table_row(line: str) -> list[str]:
    """| a | b | c | -> ['a','b','c']（去首尾竖线后按竖线切分）"""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_table_sep(line: str) -> bool:
    """识别表格第二行的 |---|---| 分隔行"""
    return bool(re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line)) and "-" in line


def parse_markdown(md: str) -> list[dict]:
    """
    Markdown -> 结构化块列表（供 Word/LaTeX 两个渲染器共用的中间表示）

    块类型:
      {"type": "heading",   "level": 1-4, "text": str}
      {"type": "paragraph", "text": str}
      {"type": "table",     "rows": [[cell,...],...], "header": [...]|None}
      {"type": "list",      "items": [str]}
      {"type": "quote",     "text": str}
      {"type": "hr"}
    """
    blocks = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # 空行
        if not line.strip():
            i += 1
            continue

        # 分隔线
        if re.fullmatch(r"\s*-{3,}\s*", line):
            blocks.append({"type": "hr"})
            i += 1
            continue

        # 标题（#### 在前防止被 ## 截断）
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            blocks.append({"type": "heading",
                           "level": len(m.group(1)), "text": m.group(2).strip()})
            i += 1
            continue

        # 表格：当前行是 | 开头 且 下一行是分隔行 => 带表头
        if line.lstrip().startswith("|") and i + 1 < len(lines) \
                and _is_table_sep(lines[i + 1]):
            header = _split_table_row(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_split_table_row(lines[i]))
                i += 1
            blocks.append({"type": "table", "header": header, "rows": rows})
            continue

        # 表格（无表头变体：连续 | 行但没有分隔行——按无表头处理）
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_split_table_row(lines[i]))
                i += 1
            blocks.append({"type": "table", "header": None, "rows": rows})
            continue

        # 列表（- / * 开头；引用块内的列表不特殊处理）
        stripped = line.lstrip()
        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < len(lines):
                s = lines[i].lstrip()
                m2 = re.match(r"^[-*]\s+(.*)", s)
                if not m2:
                    break
                items.append(m2.group(1))
                i += 1
            blocks.append({"type": "list", "items": items})
            continue

        # 引用块（> 开头，连续行合并）
        if stripped.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            blocks.append({"type": "quote", "text": "\n".join(quote_lines).strip()})
            continue

        # 普通段落（连续非空行合并——报告的段落都是单行，这里保守合并）
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() \
                and not lines[i].lstrip().startswith(("|", ">", "#")) \
                and not re.match(r"^\s*[-*]\s+", lines[i]) \
                and not re.fullmatch(r"\s*-{3,}\s*", lines[i]):
            para.append(lines[i].strip())
            i += 1
        blocks.append({"type": "paragraph", "text": " ".join(para)})

    return blocks


# ================================================================
# 二、Word 导出器
# ================================================================

def md_to_docx(md: str, title: str = "文献综述报告",
               meta_lines: list[str] | None = None) -> bytes:
    """
    Markdown -> Word 文档（返回 .docx 的字节流，供前端 st.download_button 直接使用）

    参数:
        md: Markdown 报告正文
        title: 文档主标题（报告首行 # 标题会重复时由调用方决定是否去重）
        meta_lines: 标题下方的说明行（如"生成时间/主题"）
    """
    from docx import Document
    from docx.shared import Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    doc = Document()

    # ---- 全局样式: 正文宋体（中文）+ 西文默认，11pt ----
    # 兼容处理: rPr/rFonts 可能为 None（python-docx 不自动创建），
    # 通过 get_or_add_ensure 元素存在后再设置东亚字体
    style = doc.styles["Normal"]
    style.font.size = Pt(11)
    style.font.name = "Calibri"  # 先设西文字体，顺带确保rPr/rFonts存在
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), "宋体")

    # ---- 封面主标题 ----
    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if meta_lines:
        for line in meta_lines:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(line)
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    blocks = parse_markdown(md)
    for blk in blocks:
        _docx_block(doc, blk)

    # 字节流返回（无需临时文件）
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _docx_block(doc, blk: dict):
    """渲染单个块到 Word 文档"""
    from docx.shared import Pt, RGBColor
    from docx.oxml.ns import qn

    if blk["type"] == "heading":
        # Word 标题最多用到 level 4，5-6 级降为 4
        doc.add_heading(blk["text"], level=min(blk["level"], 4))

    elif blk["type"] == "paragraph":
        _docx_rich_paragraph(doc.add_paragraph(), blk["text"])

    elif blk["type"] == "list":
        for item in blk["items"]:
            _docx_rich_paragraph(doc.add_paragraph(style="List Bullet"), item)

    elif blk["type"] == "quote":
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm0(1.0)
        for style_name, seg in _parse_inline(blk["text"]):
            r = p.add_run(seg)
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
            if style_name == "bold":
                r.bold = True

    elif blk["type"] == "table":
        header, rows = blk["header"], blk["rows"]
        ncols = max([len(header or [])] + [len(r) for r in rows]) or 1
        table = doc.add_table(
            rows=1 + len(rows), cols=ncols) if header \
            else doc.add_table(rows=len(rows), cols=ncols)
        table.style = "Light Grid Accent 1"
        table.autofit = True
        if header:
            for c, txt in enumerate(header + [""] * (ncols - len(header))):
                _docx_cell(table.rows[0].cells[c], txt, bold=True)
        for ri, row in enumerate(rows):
            for c in range(ncols):
                txt = row[c] if c < len(row) else ""
                _docx_cell(table.rows[ri + (1 if header else 0)].cells[c], txt)
        doc.add_paragraph()  # 表后留白

    elif blk["type"] == "hr":
        p = doc.add_paragraph()
        pPr = p._p.get_or_add_pPr()
        pBdr = pPr.makeelement(qn("w:pBdr"), {})
        bottom = pBdr.makeelement(qn("w:bottom"), {
            qn("w:val"): "single", qn("w:sz"): "6",
            qn("w:space"): "1", qn("w:color"): "C7D2FE"})
        pBdr.append(bottom)
        pPr.append(pBdr)


def Cm0(v: float):
    """局部导入 Cm，避免模块级依赖 docx.shared（保持本文件仅函数内依赖docx）"""
    from docx.shared import Cm
    return Cm(v)


def _docx_cell(cell, text: str, bold: bool = False):
    """表格单元格: 长文本自动换行由 Word 原生处理; 单元格内换行压成空格"""
    from docx.shared import Pt
    text = re.sub(r"\s*\n\s*", " ", str(text)).strip()
    p = cell.paragraphs[0]
    for style_name, seg in _parse_inline(text):
        r = p.add_run(seg)
        r.font.size = Pt(9)
        if bold:
            r.bold = True


def _docx_rich_paragraph(p, text: str):
    """段落: 按行内标记（粗体/斜体/代码）逐片段写入 run"""
    from docx.shared import Pt, RGBColor
    for style_name, seg in _parse_inline(text):
        r = p.add_run(seg)
        if style_name == "bold":
            r.bold = True
        elif style_name == "italic":
            r.italic = True
        elif style_name == "code":
            r.font.name = "Consolas"
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0x9A, 0x34, 0x12)


# ================================================================
# 三、LaTeX + BibTeX 导出器
# ================================================================

# LaTeX 特殊字符转义（注意顺序: 反斜杠最先）
_LATEX_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
]


def _tex_escape(text: str) -> str:
    for ch, rep in _LATEX_ESCAPES:
        text = text.replace(ch, rep)
    return text


def _tex_inline(text: str, refs: dict[str, str]) -> str:
    """
    行内标记 -> LaTeX: 粗体\\textbf、斜体\\emph、代码\\texttt，
    以及 [arXiv:xxxx] 引用标记 -> \\cite{...}（多引用核心处理）
    """
    out = ""
    for style_name, seg in _parse_inline(text):
        esc = _tex_escape(seg)
        if style_name == "bold":
            out += r"\textbf{" + esc + "}"
        elif style_name == "italic":
            out += r"\emph{" + esc + "}"
        elif style_name == "code":
            out += r"\texttt{" + re.sub(r"\s+", " ", esc) + "}"
        else:
            out += esc
    # 引用标记: [arXiv:2302.00232] -> \cite{arxiv_2302.00232}
    # （bibkey 由 _bibkey 生成，与 refs 里的登记一一对应）
    def _sub(m):
        pid = m.group(1)
        key = _bibkey(pid)
        refs[key] = pid
        return r"\cite{" + key + "}"
    out = re.sub(r"\[arXiv:([0-9]{4}\.[0-9]{4,5})\]", _sub, out)
    return out


def _bibkey(pid: str) -> str:
    """
    任意文献标识 -> 合法 bibkey（字母数字点横线）:
      2302.00232  -> arxiv_2302.00232
      CN123456A   -> arxiv_CN123456A（专利号保留字母）
    """
    return "arxiv_" + re.sub(r"[^0-9A-Za-z.\-]", "", pid)


def build_bibtex(refs_meta: list[dict]) -> str:
    """
    文献元数据列表 -> BibTeX 字符串

    refs_meta 元素字段: id/title/authors/year/venue/doi/url/source_type
    （id 也兼容专利号等任意标识——bibkey 统一清洗）
    """
    entries = []
    seen = set()
    for m in refs_meta:
        pid = str(m.get("id", "")).strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        authors = m.get("authors") or []
        if not authors:
            author = "Unknown"
        else:
            author = " and ".join(str(a) for a in authors)
        year = m.get("year") or "n.d."
        title = m.get("title") or pid
        venue = m.get("venue") or ""
        doi = m.get("doi") or ""
        url = m.get("url") or ""

        lines = [f"@misc{{{_bibkey(pid)},"]
        lines.append(f"  title         = {{{_tex_escape(str(title))}}},")
        lines.append(f"  author        = {{{_tex_escape(str(author))}}},")
        lines.append(f"  year          = {{{year}}},")
        if venue:
            lines.append(f"  howpublished  = {{{_tex_escape(str(venue))}}},")
        if doi:
            lines.append(f"  doi           = {{{doi}}},")
        if url:
            lines.append(f"  url           = {{{url}}},")
        # source_type 附加说明（学位论文/专利等）
        stype = m.get("source_type") or ""
        if stype and stype != "预印本":
            lines.append(f"  note          = {{[{_tex_escape(str(stype))}]}},")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + "\n"


def build_latex_package(md: str, title: str,
                         refs_meta: list[dict],
                         meta_lines: list[str] | None = None) -> bytes:
    """
    Markdown -> 可直接编译的 LaTeX zip 包字节流

    包内容:
      main.tex   —— 文档正文（ctexart 中文支持 + longtable 长表格）
      refs.bib   —— BibTeX 文献库（含正文引用到的所有条目 + 元数据全量）
      README.txt —— 编译说明（pdflatex -> bibtex -> pdflatex x2）
    """
    refs: dict[str, str] = {}  # bibkey -> 原始id（_tex_inline 过程中登记）

    body_parts = []
    for blk in parse_markdown(md):
        _latex_block(blk, body_parts, refs)
    body = "\n\n".join(body_parts)

    # refs.bib: 正文实际引用的条目优先；未引用的元数据也附上（备引）
    # 长文本边界: 正文引用了元数据里不存在的 id 时（如报告引用了
    # 未收录的 arXiv 编号），生成占位条目保证 LaTeX 永远可编译通过
    meta_by_id = {str(m.get("id")): m for m in refs_meta or []}
    cited = []
    placeholders = []
    for pid in refs.values():
        if pid in meta_by_id:
            cited.append(meta_by_id[pid])
        else:
            placeholders.append({
                "id": pid, "title": f"(元数据未收录: arXiv:{pid})",
                "authors": [], "year": None, "venue": "arXiv",
                "doi": None, "url": f"https://arxiv.org/abs/{pid}",
                "source_type": "",
            })
    rest = [m for m in refs_meta or []
            if str(m.get("id")) not in refs.values()]
    bib = build_bibtex(cited + placeholders + rest)

    title_tex = _tex_escape(title)
    meta_tex = " \\\\ ".join(_tex_escape(l) for l in (meta_lines or []))
    meta_cmd = (r"\date{" + meta_tex + "}\n") if meta_tex else r"\date{}\n"

    main_tex = r"""\documentclass[11pt,a4paper]{ctexart}
% ==== 由"科研文献整理Agent"自动生成 ====
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}      % 三线表
\usepackage{longtable}     % 跨页长表格（长文本边界处理）
\usepackage{array}         % 定宽p列
\usepackage{hyperref}      % 引用可点击跳转
\usepackage{xcolor}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue,
            urlcolor=blue}
\usepackage[numbers]{natbib}

\title{""" + title_tex + r"""}
""" + meta_cmd + r"""\author{}

\begin{document}
\maketitle

""" + body + r"""

% ==== 参考文献 ====
\bibliographystyle{unsrtnat}
\bibliography{refs}

\end{document}
"""

    readme = (
        "LaTeX 编译包（自动生成）\n"
        "========================\n"
        "包含文件:\n"
        "  main.tex —— 报告正文（含所有表格与引用 \\cite 标记）\n"
        "  refs.bib —— BibTeX 文献库\n\n"
        "编译方法（任选其一）:\n"
        "  方式A: 使用 Overleaf —— 新建项目，上传本包全部文件，\n"
        "         编译器选 XeLaTeX（ctexart 中文文档需要）\n"
        "  方式B: 本地 TeX Live / MiKTeX —— 依次执行:\n"
        "         xelatex main.tex\n"
        "         bibtex main\n"
        "         xelatex main.tex\n"
        "         xelatex main.tex\n"
        "  说明: pdflatex 对 ctexart 支持有限，推荐 XeLaTeX。\n"
        "  引用图谱等 HTML 交互产物不随包导出（LaTeX 静态文档不适用）。\n"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", main_tex)
        zf.writestr("refs.bib", bib)
        zf.writestr("README.txt", readme)
    return buf.getvalue()


def _latex_block(blk: dict, out: list[str], refs: dict[str, str]):
    """渲染单个块为 LaTeX 源码（追加到 out）"""
    if blk["type"] == "heading":
        lv = min(blk["level"], 3)
        cmds = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection"}
        out.append(cmds[lv] + "{" + _tex_escape(blk["text"]) + "}")

    elif blk["type"] == "paragraph":
        out.append(_tex_inline(blk["text"], refs))

    elif blk["type"] == "list":
        out.append(r"\begin{itemize}" + "\n" + "\n".join(
            r"\item " + _tex_inline(it, refs) for it in blk["items"]
        ) + "\n" + r"\end{itemize}")

    elif blk["type"] == "quote":
        out.append(r"\begin{quote}" + "\n" + _tex_escape(blk["text"])
                   + "\n" + r"\end{quote}")

    elif blk["type"] == "hr":
        out.append(r"\noindent\rule{\linewidth}{0.4pt}")

    elif blk["type"] == "table":
        _latex_table(blk, out, refs)


def _latex_table(blk: dict, out: list[str], refs: dict[str, str]):
    """
    表格 -> longtable（跨页安全，长文本用定宽 p 列自动换行）

    列宽策略: 正文宽度按列数均分; 单元格内换行压成空格，
    长文本由 p{...} 列的自动折行兜底（长文本边界处理）。
    """
    header, rows = blk["header"], blk["rows"]
    ncols = max([len(header or [])] + [len(r) for r in rows]) or 1
    # 均分可用宽度（\\linewidth），最小0.9cm防过窄
    colw = max(0.9, 15.5 / ncols)

    col_spec = "".join(f"p{{{colw:.2f}cm}}" for _ in range(ncols))

    def esc_cell(c: str) -> str:
        c = re.sub(r"\s*\n\s*", " ", str(c)).strip()
        return _tex_inline(c, refs)

    lines = [r"\begin{longtable}{" + col_spec + "}",
             r"\toprule"]
    if header:
        cells = [esc_cell(h) for h in header]
        cells += [""] * (ncols - len(cells))
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        lines.append(r"\endfirsthead")   # 跨页续表头
        lines.append(r"\toprule")
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")
    for row in rows:
        cells = [esc_cell(c) for c in row]
        cells += [""] * (ncols - len(cells))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    out.append("\n".join(lines))


# ================================================================
# 四、元数据适配层（把两处来源统一成 refs_meta 中间格式）
# ================================================================

def meta_from_papers_json(papers_meta: list[dict]) -> list[dict]:
    """papers_meta.json 内容 -> refs_meta"""
    return [{
        "id": p.get("arxiv_id", ""),
        "title": p.get("title", ""),
        "authors": p.get("authors", []),
        "year": p.get("year"),
        "venue": "arXiv preprint" if not p.get("venue") else p.get("venue"),
        "doi": p.get("doi"),
        "url": p.get("pdf_url") or f"https://arxiv.org/abs/{p.get('arxiv_id','')}",
        "source_type": "预印本",
    } for p in papers_meta]


def meta_from_mode_hits(hits: list[dict]) -> list[dict]:
    """模式检索溯源清单 hits -> refs_meta"""
    return [{
        "id": h.get("doi") or h.get("patent_no") or h.get("arxiv_id")
             or h.get("title", "")[:60],
        "title": h.get("title", ""),
        "authors": h.get("authors", []),
        "year": h.get("year"),
        "venue": h.get("venue"),
        "doi": h.get("doi"),
        "url": h.get("url") or h.get("pdf_url"),
        "source_type": h.get("source_type", ""),
    } for h in hits]
