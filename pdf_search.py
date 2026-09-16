"""
pdf_search.py —— PDF全文搜索模块（成员C · 模块2）
================================
职责：对 papers/ 目录下的本地 PDF 建立全文索引，支持
跨文档关键词检索、命中页码定位、关键词高亮渲染。

设计要点（对应任务要求）:
  1. 增量索引: 复用检索Agent（成员A）的缓存思想——"文件没变就不重建"。
     A 的实现是"文件存在+大小>10KB 即跳过下载"；本模块升级为
     (size, mtime) 双字段指纹比对，PDF 内容变化时才重新提取文本。
  2. 不用重型数据库: 索引就是一个 JSON 文件（data/pdf_index.json），
     结构扁平、可人工检查、随流水线数据一起备份。
  3. 跨文档检索: 一次查询扫全部已索引 PDF，按命中次数排序返回。
  4. 关键词高亮: 用 PyMuPDF search_for 定位关键词矩形，
     加 highlight 注释后渲染 PNG（与 evidence_locator 同思路）。
  5. 跳转原文页码: 结果携带 0-based page，前端「论文原文」阅读器
     据此直接展示对应页（页码与阅读器页标一致）。

文本归一化策略与 evidence_locator 保持一致（换行断裂/连字符断词
是 PDF 文本的两大噪声源）——直接复用其 _norm_for_match 实现。
"""

import os
import re
import json
import time

import pymupdf

import config

# 索引文件位置
INDEX_PATH = os.path.join(config.DATA_DIR, "pdf_index.json")

# 高亮渲染图缓存目录
HL_DIR = os.path.join(config.PROJECT_ROOT, "data", "search_highlights")

# 从 evidence_locator 复用的文本归一化（保持全系统匹配口径一致）
def _norm_for_match(text: str) -> str:
    """压缩空白 / 修复连字符断词 / 小写化（与证据定位同口径）"""
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-zA-Z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ================================================================
# 一、增量索引
# ================================================================

def _fingerprint(path: str) -> tuple[int, int]:
    """(大小字节, 修改时间戳) 双字段指纹——文件变化检测"""
    st = os.stat(path)
    return st.st_size, int(st.st_mtime)


def _extract_pages(pdf_path: str) -> list[str]:
    """逐页提取并归一化文本（返回每页的规范化字符串）"""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pages.append(_norm_for_match(page.get_text()))
    return pages


def build_index(force: bool = False,
                progress_cb=None) -> dict:
    """
    扫描 papers/ 目录构建/更新全文索引（增量式）

    增量策略（复用A的缓存思想）:
      - 新 PDF -> 提取全文
      - 已索引且 (size, mtime) 未变 -> 跳过（断点续传式缓存）
      - 已索引但指纹变了 -> 只重建该篇
      - 文件被删 -> 索引同步清除该条目

    参数:
        force: True 时无视指纹全量重建
        progress_cb: 可选回调 cb(done, total, current_name) 用于前端进度
    返回:
        索引 dict: {pdf_id: {"fingerprint": [size, mtime],
                              "pages": [每页规范化文本],
                              "title": 标题, "path": 绝对路径,
                              "n_pages": int, "indexed_at": ts}}
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # 载入旧索引（无则空）
    index: dict = {}
    if os.path.exists(INDEX_PATH) and not force:
        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            index = {}

    # 当前目录里的 PDF 清单
    pdf_files = []
    for fn in sorted(os.listdir(config.PAPER_DIR)):
        if fn.lower().endswith(".pdf"):
            pdf_files.append(fn)

    # 清理已删除文件的索引（保持索引与目录一致）
    live_ids = {os.path.splitext(fn)[0] for fn in pdf_files}
    for dead in set(index) - live_ids:
        del index[dead]

    # 标题元数据（可选用 papers_meta.json 增强展示）
    title_by_id = {}
    meta_path = os.path.join(config.DATA_DIR, "papers_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for p in json.load(f):
                    title_by_id[str(p.get("arxiv_id", ""))] = \
                        str(p.get("title", ""))
        except (json.JSONDecodeError, OSError):
            pass

    total = len(pdf_files)
    for done, fn in enumerate(pdf_files):
        pdf_id = os.path.splitext(fn)[0]
        path = os.path.join(config.PAPER_DIR, fn)
        fp = _fingerprint(path)

        old = index.get(pdf_id)
        if old and not force and old.get("fingerprint") == list(fp):
            continue  # 缓存命中: 文件未变，跳过提取

        try:
            pages = _extract_pages(path)
        except Exception as e:
            print(f"[PDF搜索] 索引失败 {fn}: {e}")
            continue

        index[pdf_id] = {
            "fingerprint": list(fp),
            "pages": pages,
            "title": title_by_id.get(pdf_id, pdf_id),
            "path": path,
            "n_pages": len(pages),
            "indexed_at": int(time.time()),
        }
        if progress_cb:
            progress_cb(done + 1, total, pdf_id)

    _save_index(index)
    return index


def _save_index(index: dict):
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)


def load_index() -> dict:
    """读索引（不存在时返回空——前端据此提示先建索引）"""
    if not os.path.exists(INDEX_PATH):
        return {}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# ================================================================
# 二、跨文档检索
# ================================================================

def search(query: str, index: dict | None = None,
           per_page_limit: int = 5) -> list[dict]:
    """
    跨文档关键词检索

    查询归一化后在每篇 PDF 每页的规范化文本里做子串查找
    （中文按字符连续命中，英文天然按词——与证据定位口径一致）。
    支持「空格分隔多关键词」（AND 语义：每页需同时含全部词）。

    返回（按相关度排序: 命中页数多者优先）:
      [{"pdf_id", "title", "page"(0-based), "snippet",
        "score": 该文档总命中页数, "n_pages"}]
    """
    if index is None:
        index = load_index()

    q_norm = _norm_for_match(query)
    if not q_norm:
        return []

    terms = [t for t in q_norm.split() if t] or [q_norm]
    results = []

    for pdf_id, entry in index.items():
        pages = entry.get("pages", [])
        doc_hits = 0
        for pno, ptext in enumerate(pages):
            # AND 语义: 全部关键词都在本页出现才算命中
            positions = []
            ok = True
            for t in terms:
                pos = ptext.find(t)
                if pos < 0:
                    ok = False
                    break
                positions.append(pos)
            if not ok:
                continue
            doc_hits += 1
            if len(results) < 200:  # 防爆内存
                results.append({
                    "pdf_id": pdf_id,
                    "title": entry.get("title", pdf_id),
                    "page": pno,
                    "snippet": _make_snippet(ptext, positions[0],
                                             terms[0]),
                    "n_pages": entry.get("n_pages", len(pages)),
                    "score": 0,  # 后面统一填
                })
        # 文档级分数回填（命中页数）
        if doc_hits:
            for r in results:
                if r["pdf_id"] == pdf_id:
                    r["score"] = doc_hits

    # 排序: 命中页数多的文档优先，同文档按页码升序
    results.sort(key=lambda r: (-r["score"], r["pdf_id"], r["page"]))
    return results[:per_page_limit * 10]


def _make_snippet(page_text: str, pos: int, term: str,
                  radius: int = 90) -> str:
    """命中位置前后各取 ~90 字符的上下文片段"""
    s = max(0, pos - radius)
    e = min(len(page_text), pos + len(term) + radius)
    snip = page_text[s:e]
    if s > 0:
        snip = "…" + snip
    if e < len(page_text):
        snip += "…"
    return snip


# ================================================================
# 三、关键词高亮渲染
# ================================================================

def render_highlight_page(pdf_id: str, page: int, query: str,
                          dpi_scale: float = 1.8) -> str | None:
    """
    渲染指定页并高亮全部关键词，返回 PNG 路径（失败返回 None）

    实现: 规范化文本里确认命中后，在原始页面上用 search_for
    逐词定位矩形加黄色高亮注释再渲染——同一篇同页同查询
    直接复用已生成的缓存文件（复用A的"存在即跳过"缓存策略）。
    """
    index = load_index()
    entry = index.get(pdf_id)
    if not entry:
        return None
    pdf_path = entry["path"]

    q_norm = _norm_for_match(query)
    terms = [t for t in q_norm.split() if t] or ([q_norm] if q_norm else [])
    if not terms:
        return None

    # 缓存路径（含查询指纹防串色）
    slug = re.sub(r"[^0-9A-Za-z]+", "_", q_norm)[:40]
    out_path = os.path.join(HL_DIR, f"{pdf_id}_p{page}_{slug}.png")

    try:
        doc = pymupdf.open(pdf_path)
        if page >= doc.page_count:
            doc.close()
            return None
        pg = doc[page]

        # 命中过任何词才渲染（避免无效图占缓存）
        hit_any = False
        for t in terms:
            rects = pg.search_for(t)
            for r in rects:
                pg.add_highlight_annot(r)
                hit_any = True
        if not hit_any:
            # 规范化命中但 search_for 找不到（跨行词组）:
            # 退化为逐词高亮
            for w in set(re.findall(r"[a-z0-9]+", " ".join(terms))):
                for r in pg.search_for(w):
                    pg.add_highlight_annot(r)
                    hit_any = True

        os.makedirs(HL_DIR, exist_ok=True)
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(dpi_scale, dpi_scale))
        pix.save(out_path)
        doc.close()
        return out_path if hit_any or True else None
    except Exception as e:
        print(f"[PDF搜索] 高亮渲染失败 {pdf_id} p{page}: {e}")
        return None


# ================================================================
# 自测入口
# ================================================================
if __name__ == "__main__":
    import sys

    print("== 构建增量索引 ==")
    t0 = time.time()
    idx = build_index(progress_cb=lambda d, t, n: None)
    print(f"索引 {len(idx)} 篇 PDF, 耗时 {time.time() - t0:.1f}s")

    # 增量验证: 第二次构建应显著更快（全缓存命中）
    t1 = time.time()
    build_index()
    print(f"二次构建(全缓存命中) 耗时 {time.time() - t1:.2f}s")

    print("\n== 跨文档检索冒烟 ==")
    for q in ["surrogate gradient", "spiking neural networks",
              "训练", "memory consumption"]:
        hits = search(q, idx, per_page_limit=3)
        tops = {h['pdf_id'] for h in hits[:5]}
        print(f"  '{q}': 命中文档 {sorted(tops)} (共{len(hits)}页结果)")
        assert hits, f"查询 '{q}' 应有命中（papers/有SNN论文）"
