"""
evidence_locator.py —— 证据定位Agent（方案一：证据链穿透）
================================
职责：
  把审查通过的声明里的原文引用(quote)定位到PDF的具体页面，
  生成高亮标注的页面截图，供前端"证据穿透面板"展示。

【定位的难点与三级匹配策略】
  PDF提取的文本与quote的常见差异（实测总结）:
    1. 换行断裂: PDF里句子被行尾打断, 提取文本带\n
    2. 连字符断词: "perfor-\nmance" vs "performance"
    3. 合字/特殊空格: fi/fl合字, 不间断空格
  因此逐级放宽匹配:
    L1: 完整quote直接search_for()        (~60%命中)
    L2: quote前50字符再搜                 (~25%)
    L3: 最长连续词组(5词滑窗)逐个尝试      (~10%)
    兜底: 返回None, 前端显示"无法定位"

输入: (pdf_path, quote_text)
输出: EvidenceLocatorResult(页码, 高亮图片路径, 匹配级别)
"""

import os
import re

import fitz  # PyMuPDF

import config


class EvidenceHit:
    """一次定位成功的结果"""

    def __init__(self, page_num: int, image_path: str, match_level: int,
                 matched_text: str):
        self.page_num = page_num      # 0-based页码
        self.image_path = image_path  # 高亮页面截图的路径
        self.match_level = match_level  # 1/2/3 匹配级别
        self.matched_text = matched_text  # 实际匹配到的文本

    @property
    def match_level_desc(self) -> str:
        return {1: "完整匹配", 2: "前缀匹配", 3: "词组匹配"}.get(
            self.match_level, "?")


# ---------------------------------------------------------------
# 文本归一化（与quote对齐的关键预处理）
# ---------------------------------------------------------------
def _norm_for_match(text: str) -> str:
    """
    归一化文本用于匹配比较:
      - 压缩所有空白为单个空格（消除换行断裂）
      - 移除连字符断词的连字符（perfor-mance -> performance）
        注意: 只处理"字母-换行"模式，正常连字符词(high-order)保留
      - 统一小写
    """
    # 1. 连字符断词: 小写字母 + 可选连字符 + 换行 + 小写字母 -> 直接拼接
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-zA-Z])", r"\1\2", text)
    # 2. 所有空白(含\n)压成单空格
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ---------------------------------------------------------------
# 核心定位函数
# ---------------------------------------------------------------
def locate_quote(pdf_path: str, quote: str) -> tuple[int, str] | None:
    """
    在PDF中定位quote文本

    返回:
        (page_num_0based, 实际匹配的规范化文本) 或 None(定位失败)

    实现: 用PyMuPDF逐页提取文本 -> 规范化 -> 子串查找。
    （不用page.search_for()的原因: 它对跨行文本敏感，
      而规范化后的子串匹配对换行/断词天然免疫）
    """
    if not os.path.exists(pdf_path) or not quote:
        return None

    quote_norm = _norm_for_match(quote)
    if len(quote_norm) < 10:  # 太短的引用无法可靠定位
        return None

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    try:
        for page_num, page in enumerate(doc):
            # 页面文本与高亮渲染(_page_word_rects)用同一词序列构造,
            # 保证"定位成功=>词对齐高亮必然命中"
            page_text = " ".join(
                t for t, _ in _page_word_rects(page))
            if quote_norm in page_text:
                return page_num, quote_norm

            # L2: 前缀匹配(长引用可能跨页)。按词边界截前缀——
            # 若按字符截断会把尾词切成半截, 词级高亮对不齐
            if len(quote_norm) > 60:
                prefix_words = quote_norm.split()
                prefix = ""
                for w in prefix_words:
                    if len(prefix) + len(w) + 1 > 50 and prefix:
                        break
                    prefix = f"{prefix} {w}".strip()
                if len(prefix.split()) >= 4 and prefix in page_text:
                    return page_num, prefix

            # L3: 最长词组滑窗(从长到短尝试, 至少5词)
            words = quote_norm.split()
            for win in (8, 6, 5):
                if len(words) < win:
                    continue
                for start in range(0, len(words) - win + 1):
                    chunk = " ".join(words[start:start + win])
                    if chunk in page_text:
                        return page_num, chunk
    finally:
        doc.close()
    return None


# ---------------------------------------------------------------
# 高亮渲染（词矩形精确对齐）
# ---------------------------------------------------------------
def _page_word_rects(page) -> list[tuple[str, object]]:
    """
    页面词序列(阅读顺序): [(规范化词, 矩形), ...]

    连字符断词合并: "perfor-" + "mance" -> "performance"（矩形取并集），
    与_norm_for_match的断词修复口径一致。
    """
    raw = page.get_text("words")  # 每项: [x0,y0,x1,y1,word,block,line,no]
    items: list[tuple[str, object]] = []
    prev: tuple[str, object] | None = None  # 以'-'结尾的待合并断词
    for w in raw:
        word = w[4]
        rect = fitz.Rect(w[0], w[1], w[2], w[3])
        if prev is not None:
            # 前词以连字符结尾 + 当前词字母开头 -> 断词合并
            if prev[0].endswith("-") and word[:1].isalpha():
                items.append((prev[0][:-1] + word, prev[1] | rect))
                prev = None
                continue
            items.append(prev)
            prev = None
        if word.endswith("-") and len(word) > 1 and word[:-1].isalpha():
            prev = (word, rect)
        else:
            items.append((word, rect))
    if prev is not None:
        items.append(prev)
    # 与locate_quote同一规范化口径（单词无空白，仅小写；保留标点）
    return [(_norm_for_match(t), r) for t, r in items]


def _find_word_span(page_words: list, target_words: list) -> tuple[int, int] | None:
    """
    在页面词序列中找目标词序列的连续匹配区间

    返回:
        (start, end) 词下标区间 或 None
    """
    n, m = len(page_words), len(target_words)
    if m == 0 or n < m:
        return None
    first = target_words[0]
    for s in range(n - m + 1):
        if page_words[s][0] != first:
            continue
        if all(page_words[s + k][0] == target_words[k] for k in range(m)):
            return s, s + m
    return None


def render_evidence_page(pdf_path: str, page_num: int,
                         matched_text: str, out_path: str) -> bool:
    """
    渲染指定页并高亮匹配文本

    高亮实现（三级降级, 词矩形精确对齐优先, 融合cbc分支与
    review-agent-benchmark分支两版方案）:
      L1 精确词段对齐: _page_word_rects构建词流(含连字符断词合并
         perfor-⏎mance->performance, 矩形并集), _find_word_span定位
         matched_text对应的连续词段——高亮的必然就是locate_quote
         匹配到的那段原文, 物理上杜绝错位/误亮别处单词
      L2 模糊滑窗对齐: matched_text与词流有细微差异(OCR噪声等)
         导致L1失败时, 滑窗取匹配率>=0.6的最佳窗口, 只取命中词bbox
      L3 search_for滑窗+全词兜底: 虚词过滤/多现词放弃, 最后防线
      全部失败返回False(前端显示"无法定位", 绝不误亮)

    返回:
        True=渲染成功; False=词对齐失败(前端显示"无法定位", 绝不误亮)
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        target_words = matched_text.split()
        if not target_words:
            doc.close()
            return False

        page_words = _page_word_rects(page)
        span = _find_word_span(page_words, target_words)
        if span is not None:
            s, e = span
            rects = [page_words[i][1] for i in range(s, e)]
        else:
            # L2/L3: matched_text与词流有差异时的容错路径
            rects = _rects_by_word_alignment(page, target_words, page_words)
            if not rects:
                rects = _rects_by_search_window(page, target_words)
            if not rects:
                print(f"[证据定位] 词对齐失败 p{page_num}: "
                      f"{matched_text[:40]}...")
                doc.close()
                return False

        # 同行且水平间隙小的相邻矩形合并为一条高亮带（美观+少注释;
        # 双栏同行左右间距大, 不会误合）
        for rect in _merge_line_rects(rects):
            page.add_highlight_annot(rect)

        # 渲染为图片（2倍缩放保证文字清晰）
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pix.save(out_path)
        doc.close()
        return True
    except Exception as e:
        print(f"[证据定位] 渲染失败 {pdf_path} p{page_num}: {e}")
        return False


def _merge_line_rects(rects: list) -> list:
    """同行(y差<2)且水平间隙<=25pt的相邻矩形合并为高亮带

    按阅读顺序(y,x)排序后线性合并; 合并基于矩形几何位置,
    与矩形来源(L1词段/L2模糊对齐/L3兜底)无关。
    """
    rs = sorted(rects, key=lambda r: (round(r.y0, 1), r.x0))
    out: list = []
    cur = None
    for r in rs:
        if (cur is not None and abs(cur.y0 - r.y0) < 2
                and abs(cur.y1 - r.y1) < 2
                and 0 <= r.x0 - cur.x1 <= 25):
            cur |= r
        else:
            if cur is not None:
                out.append(cur)
            cur = fitz.Rect(r)
    if cur is not None:
        out.append(cur)
    return out


def _norm_word(w: str) -> str:
    """词级归一化: 剥标点+小写(标点会干扰词匹配)"""
    return w.strip(".,;:!?()[]\"'`").lower()


# 逐词兜底时跳过的虚词: 孤立搜索这些词毫无定位价值,
# 只会在全页产生大量无关高亮
_STOP_WORDS = frozenset("""
a an the of to in on for with and or is are was were be been being
we our us it its this that these those as at by from not no but if
then than so such can could may might will would should must have
has had do does did over under between within into onto about above
below out up down off
""".split())


def _rects_by_word_alignment(page, words: list[str],
                             page_words: list | None = None) -> list:
    """L2: 词序列滑窗模糊对齐, 取匹配率最高窗口的命中词bbox

    matched_text与词流存在细微差异(OCR噪声/断词合并遗漏等)导致
    L1精确对齐失败时, 按60%匹配率兜底。相比search_for滑窗:
      - 斜体/特殊字体词不再搜索失败(实测部分斜体词search_for
        返回0矩形, 导致高亮缺口)
      - 短词不再子串误命中(search_for("art")会命中"artificial",
        词序列对齐受上下文约束不会)
      - 每个命中词的bbox天然相邻, 整段高亮无缺口

    page_words: 预构建词流[(规范化词, 矩形)](来自_page_word_rects,
    含断词合并); None时自建简易词流。
    """
    if not words:
        return []
    q_words = [_norm_word(w) for w in words]  # 与words一一对应, 空串占位
    n = len(q_words)

    if page_words is None:
        raw = page.get_text("words")
        page_words = [(_norm_for_match(t[4]), fitz.Rect(t[:4])) for t in raw]
    p_words = [_norm_word(w) for w, _ in page_words]
    if len(p_words) < n:
        return []

    # 滑窗找匹配率最高的起点
    best_s, best_m = -1, 0
    for s in range(0, len(p_words) - n + 1):
        m = sum(1 for i in range(n)
                if q_words[i] and p_words[s + i] == q_words[i])
        if m > best_m:
            best_m, best_s = m, s
    if best_s < 0 or best_m < max(2, int(n * 0.6)):
        return []  # 匹配率过低视为未命中, 交由备用方案

    # 只取匹配成功词的bbox(失配词位置不可靠, 丢弃避免错位高亮)
    return [page_words[best_s + i][1] for i in range(n)
            if q_words[i] and p_words[best_s + i] == q_words[i]]


def _rects_by_search_window(page, words: list[str]) -> list:
    """备用方案: search_for滑窗分块搜索(处理词级对齐失败的情况)"""
    if not words:
        return []

    rects: list = []

    def _add_batch(found):
        for r in found:
            # 近似去重: 重叠滑窗块会产生同一词的细微差异矩形
            # (y相差约1pt), 精确相等去重无法消除
            if not any(abs(r.y0 - o.y0) < 2 and abs(r.x0 - o.x0) < 2
                       for o in rects):
                rects.append(r)

    def _add_word_by_word(chunk_words, y_min=None, y_max=None):
        """逐词兜底: 在页面词流中做全词精确匹配

        不用search_for逐词搜: 它是子串匹配, "rot"会命中"rotate"的
        前3字符, "of"会命中"profile"——产生无意义高亮。
        改为页面词流全词相等比较:
          - 虚词(of/the/an...)与过短词直接跳过
          - y_min/y_max约束时只收集锚矩形附近的命中(避免同词
            在页面其他位置的高亮混入)
          - 无锚定时, 页面出现>=4次的词区分度太低, 直接放弃
            (满页开花比高亮缺口更糟)
        """
        wanted = {_norm_word(w) for w in chunk_words}
        pw = page.get_text("words")
        p_norm = [_norm_word(t[4]) for t in pw]
        freq = {}
        for w in p_norm:
            freq[w] = freq.get(w, 0) + 1
        for t, w in zip(pw, p_norm):
            if len(w) < 2 or w in _STOP_WORDS or w not in wanted:
                continue
            if y_min is not None:
                if t[3] < y_min or t[1] > y_max:
                    continue
            elif freq[w] >= 4:
                continue
            r = fitz.Rect(t[:4])
            if not any(r == o for o in rects):
                rects.append(r)

    if len(words) <= 8:
        # 短文本: 词流全词匹配优先(无子串误命中, 虚词被过滤),
        # 未命中再整句search_for(处理断行连字符拆词等词流失效场景)
        _add_word_by_word(words)
        if not rects:
            _add_batch(page.search_for(" ".join(words)))
        return rects

    # 长文本: 滑窗分块
    chunk, step = 6, 3
    starts = list(range(0, len(words) - chunk + 1, step))
    if starts[-1] != len(words) - chunk:
        starts.append(len(words) - chunk)  # 末尾块对齐文末

    for s in starts:
        phrase = " ".join(words[s:s + chunk])
        found = page.search_for(phrase)
        if found:
            _add_batch(found)
        elif s == 0 or s == starts[-1]:
            # 首块/末块失败: 逐词兜底(避免整段开头/结尾丢失)。
            # 末块兜底时已有锚矩形, 限定在其附近±20pt(约一行)收集,
            # 防止具体词(如state)在页面其他出现处被误高亮
            if rects:
                ys0 = min(r.y0 for r in rects)
                ys1 = max(r.y1 for r in rects)
                _add_word_by_word(words[s:s + chunk], ys0 - 20, ys1 + 20)
            else:
                _add_word_by_word(words[s:s + chunk])
    return rects


# ---------------------------------------------------------------
# 对外主入口：批量生成所有声明的证据索引
# ---------------------------------------------------------------
def build_evidence_index(cards: list[dict],
                         review_results: list[dict] | None = None,
                         only_supported: bool = True) -> dict:
    """
    为（通过审查的）声明批量生成证据图片与索引

    参数:
        cards: paper_cards.json的内容
        review_results: review_results.json的内容(用于过滤已核验声明);
                        None时不过滤
        only_supported: 只为核验通过的声明生成证据

    返回:
        evidence_index: {声明key: {page, image, match_level, ...}}
        声明key格式: "{arxiv_id}#{claim_index}"
    """
    # 通过审查的声明集合
    supported_keys = None
    if review_results is not None and only_supported:
        supported_keys = {
            f"{r['arxiv_id']}#{r['claim_index']}"
            for r in review_results
            if r.get("verdict", {}).get("verdict") == "supported"
        }

    # PDF路径缓存目录
    ev_dir = os.path.join(config.PROJECT_ROOT, "evidence")
    os.makedirs(ev_dir, exist_ok=True)

    index = {}
    total = 0
    hit = 0
    for card in cards:
        arxiv_id = card["arxiv_id"]
        pdf_path = os.path.join(config.PAPER_DIR,
                                f"{arxiv_id.replace('/', '_')}.pdf")
        if not os.path.exists(pdf_path):
            continue

        for idx, claim in enumerate(card.get("claims", [])):
            key = f"{arxiv_id}#{idx}"
            if supported_keys is not None and key not in supported_keys:
                continue
            total += 1

            # 逐条quote尝试定位（任一quote命中即成功）
            located = None
            used_quote = ""
            for q in claim.get("quotes", []):
                located = locate_quote(pdf_path, q.get("text", ""))
                if located:
                    used_quote = q.get("text", "")
                    break

            if not located:
                index[key] = {"located": False}
                continue

            page_num, matched = located
            img_name = f"{arxiv_id.replace('/', '_')}_c{idx}_p{page_num}.png"
            img_path = os.path.join(ev_dir, img_name)
            ok = render_evidence_page(pdf_path, page_num, matched, img_path)
            if not ok:
                index[key] = {"located": False}
                continue

            hit += 1
            index[key] = {
                "located": True,
                "page": page_num,
                "image": img_name,
                "match_level": 1 if len(_norm_for_match(used_quote)) ==
                                len(matched) else 2,
                "quote": used_quote[:150],
                "pdf": os.path.basename(pdf_path),
            }

    # 落盘索引
    idx_path = os.path.join(config.DATA_DIR, "evidence_index.json")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    import json
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    print(f"[证据定位] 完成: {hit}/{total} 条声明定位成功 "
          f"({hit * 100 // max(total, 1)}%), 索引: {idx_path}")
    return index


if __name__ == "__main__":
    # 独立测试: 读取现有数据跑一遍完整流程
    import json

    cards = json.load(open(
        os.path.join(config.DATA_DIR, "paper_cards.json"), encoding="utf-8"))
    try:
        reviews = json.load(open(
            os.path.join(config.DATA_DIR, "review_results.json"),
            encoding="utf-8"))
    except FileNotFoundError:
        reviews = None

    build_evidence_index(cards, reviews)
