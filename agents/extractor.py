"""
extractor.py —— 提取Agent
================================
职责（两段式）：
  1. pdf_to_text() : PyMuPDF解析PDF → 全文纯文本（含简易清洗）
  2. extract_card(): LLM从全文抽取结构化"信息卡片"（PaperCard）

核心设计（防幻觉机制的根基）:
  - 提取的每条声明(Claim)都强制附带原文逐字引用(Quote)
  - models.py 中 Claim.quotes 有 min_length=1 约束，没引用直接校验失败
  - 这些引用将在第3步被审查Agent核验，伪造引用会被程序化对齐检查识破

输入: list[PaperMeta]（检索Agent的产出，已有本地PDF）
输出: list[PaperCard]，落盘到 data/paper_cards.json
"""

import os
import json
import re

import pymupdf

import llm_client
import config
import budget
from models import PaperMeta, PaperCard

# ---------------------------------------------------------------
# 第1段：PDF → 纯文本
# ---------------------------------------------------------------
def _clean_page_text(text: str) -> str:
    """单页文本清洗：去掉连续3个以上的换行（PDF排版残留）"""
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


# ---------------------------------------------------------------
# K3·A3优化：公式/乱码段清理（高符号密度块剔除）
# ---------------------------------------------------------------
# 公式块特征：短、符号密集（=+∑∫括号下标）、几乎没有英文正文虚词。
# 三重条件同时满足才剔除（宁漏勿误伤：数学论文正文段也含公式，
# 误剔会破坏声明引用的完整性）。开关 config.FORMULA_FILTER。
_FORMULA_STOPWORDS = frozenset(
    "the of and in to a is are for with that on as by an be this it"
    .split())


def _is_formula_like(text: str) -> bool:
    """保守识别公式/乱码块: 短 + 符号密度>0.30 + 正文虚词<3
    （希腊字母等Unicode字母在PDF公式块中大量出现且被isalnum
    认定为字母，故密度阈值取0.30；误伤防线主要是虚词计数）"""
    t = text.strip()
    if not t or len(t) >= 200:
        return False
    sym = sum(1 for c in t if not (c.isalnum() or c.isspace()))
    if sym / len(t) <= 0.30:
        return False
    words = re.findall(r"[a-zA-Z]+", t.lower())
    if sum(1 for w in words if w in _FORMULA_STOPWORDS) >= 3:
        return False
    return True


# ---------------------------------------------------------------
# K3·A3优化：版面感知解析（2026-11新增）
# ---------------------------------------------------------------
# 问题: PDF是"排版指令的集合"而非文本流。page.get_text("text")按
# 内部对象顺序吐字，双栏论文的左右栏会交错拼接（读左栏第1行后直接
# 跳到右栏第1行），提取Agent拿到"错序+残缺"的上下文，写出的声明
# 与原文对不上→下游引用对齐大面积失败→被误判成幻觉。
#       这是"解析问题伪装成幻觉问题"的典型。
# 方案: 用page.get_text("blocks")拿带坐标的文本块，识别左右双栏，
# 按人类阅读顺序（左栏自上而下→右栏自上而下，通栏标题/摘要作横向
# 分隔带）重组。识别不了单栏/异常时回退纯文本流（宁可退回旧行为，
# 不能比旧的更差）。
def _page_text_layout(page) -> str:
    """版面感知的单页文本重组（K3）。返回该页按阅读顺序拼接的文本。"""
    blocks = page.get_text("blocks")  # [x0, y0, x1, y1, text, block_no, type]
    text_blocks = [b for b in blocks if b[6] == 0 and (b[4] or "").strip()]
    # 公式/乱码段清理: 剔除高符号密度块；但保留量不足一半时视为
    # 公式密集型论文（整页数学推导），放弃剔除防止破坏正文完整性
    if config.FORMULA_FILTER:
        kept = [b for b in text_blocks if not _is_formula_like(b[4])]
        if kept and len(kept) * 2 >= len(text_blocks):
            text_blocks = kept
    # 块太少谈不上分栏（封面/纯图页），退回默认文本流
    if len(text_blocks) < 3:
        return page.get_text("text")

    pw = float(page.rect.width)
    left: list = []    # 完全落在左半栏的块
    right: list = []   # 完全落在右半栏的块
    full: list = []    # 横跨整页宽度的通栏块（标题/摘要/图注）
    for b in text_blocks:
        x0, _, x1, _ = b[0], b[1], b[2], b[3]
        w = x1 - x0
        if w > 0.75 * pw:
            full.append(b)
        elif x1 <= 0.58 * pw:
            left.append(b)
        elif x0 >= 0.42 * pw:
            right.append(b)
        else:
            full.append(b)  # 骑跨中线的块按通栏处理

    def _col(col, y_top, y_bot):
        """取一列中位于竖直区间(y_top, y_bot)内的块，按阅读顺序排好"""
        picked = [b for b in col if b[3] > y_top and b[1] < y_bot]
        return sorted(picked, key=lambda b: (round(b[1], 1), b[0]))

    # 双栏成立条件: 左右两栏都有足够块，且通栏块不占绝对多数
    # （学术论文正文页左右栏块各≥2很稳定；若不满足说明是单栏版式）
    if len(left) >= 2 and len(right) >= 2 and len(full) <= len(left) + len(right):
        ordered: list = []
        prev_y = -1e9
        # 通栏块（标题/摘要/跨栏图注）把页面切成若干横向条带，
        # 条带内先左栏后右栏——双栏论文的标准阅读顺序
        for fb in sorted(full, key=lambda b: b[1]):
            ordered.extend(_col(left, prev_y, fb[1]))
            ordered.extend(_col(right, prev_y, fb[1]))
            ordered.append(fb)
            prev_y = fb[3]
        ordered.extend(_col(left, prev_y, 1e18))
        ordered.extend(_col(right, prev_y, 1e18))
        return "\n".join(b[4].strip() for b in ordered)
    # 单栏: 简单按(纵坐标,横坐标)排序即可修正轻微乱序
    ordered = sorted(text_blocks, key=lambda b: (round(b[1], 1), b[0]))
    return "\n".join(b[4].strip() for b in ordered)


def _extract_page_text(page) -> str:
    """单页文本入口: 按config.LAYOUT_AWARE_PDF灰度开关选解析路径，
    版面感知路径出错时回退纯文本流（绝不因解析升级让流水线崩溃）"""
    if config.LAYOUT_AWARE_PDF:
        try:
            return _page_text_layout(page)
        except Exception as e:
            print(f"[提取Agent] 版面感知解析异常, 回退纯文本流: {e}")
    return page.get_text("text")


def pdf_to_pages(pdf_path: str, max_chars: int = 100_000) -> list[str]:
    """
    解析PDF为逐页文本列表（供引用定位按页归属使用）

    每页做与 pdf_to_text 相同的清洗（含页码噪声行去除），
    总量超过 max_chars 时停止追加（与 pdf_to_text 截断口径一致）。
    K3起默认走版面感知路径（双栏按阅读顺序重组），
    LAYOUT_AWARE_PDF=0 可回退纯文本流。

    返回:
        list[str]，第i个元素为第i页(0-based)的清洗后文本
    """
    doc = pymupdf.open(pdf_path)
    pages: list[str] = []
    total = 0
    for page in doc:
        text = _clean_page_text(_extract_page_text(page))
        # 页码噪声处理：把单独成行的数字（页码）去掉
        lines = [ln for ln in text.split("\n")
                 if not (ln.strip().isdigit() and len(ln.strip()) <= 4)]
        text = "\n".join(lines)
        total += len(text)
        if total > max_chars:
            print(f"[提取Agent] 论文过长(>{max_chars}字符)，"
                  f"在{len(pages)}页处截断")
            break
        pages.append(text)
    doc.close()
    return pages


def pdf_to_text(pdf_path: str, max_chars: int = 100_000) -> str:
    """
    解析PDF全文并做简易清洗

    参数:
        pdf_path: PDF文件路径
        max_chars: 截断上限（2026-09-14由60K提到100K——AutoSNN等
                   长论文78K字符也被截断，导致附录里的实验细节丢失。
                   100K字符英文≈2.5万token，距128K上下文模型的上限
                   仍有充足余量；deepseek 64K上下文同样安全）

    返回:
        清洗后的全文文本
    """
    full_text = "\n".join(pdf_to_pages(pdf_path, max_chars=max_chars))

    if len(full_text) > max_chars:
        print(f"[提取Agent] 论文过长({len(full_text)}字符)，截断到{max_chars}")
        full_text = full_text[:max_chars]
    return full_text


# ---------------------------------------------------------------
# 第2段：LLM信息抽取
# ---------------------------------------------------------------
# 提取Agent的系统提示词（本项目Prompt工程的核心之一）
EXTRACTOR_SYSTEM_PROMPT = """你是一位严谨的科研文献信息抽取专家。

你的任务：阅读论文全文，提取出结构化的"信息卡片"。

【最重要的规则——逐字引用】
你提取的每一条声明(claim)，都必须附带至少1条来自原文的逐字引用(quote)：
  - quote.text 必须是论文原文的连续片段，一字不差（保留原拼写和数字）
  - quote.section 标注出处（如 "Abstract"、"3.2节"、"Table 2"）
  - 引用是用来事后核验你提取内容的，任何改写、概括、编造都会被检出

【提取要求】
1. 每篇论文提取 4-8 条声明，覆盖三类：
   - method: 核心方法/算法思路（该文提出了什么做法）
   - result: 关键实验结果（必须有具体数字，如准确率、数据集名）
   - limitation: 作者承认的局限或失败案例（通常在 Discussion/Limitations 节）
2. 声明内容用中文撰写（方便后续生成中文综述），但quote必须保留英文原文
3. result类声明要保留原文的数字精度，不许四舍五入
4. method_category 字段：给论文的方法打一个分类标签
   （如"代理梯度训练"/"突触可塑性规则"/"混合编码方案"等，用中文短词）

【严禁】
- 不许输出原文中找不到依据的内容（哪怕是常识）
- 不许把多篇论文的内容混在一起
"""


def extract_card(paper: PaperMeta) -> PaperCard | None:
    """
    对单篇论文执行信息抽取

    参数:
        paper: 已下载PDF的论文元数据

    返回:
        PaperCard 实例；解析失败返回None
    """
    budget.check("提取Agent")  # 超预算立即中止（异常穿透线程池向上传播）
    print(f"[提取Agent] 正在处理: {paper.title[:50]}...")

    # 1. PDF → 全文
    if not paper.local_path or not os.path.exists(paper.local_path):
        print(f"[提取Agent] PDF不存在，跳过: {paper.arxiv_id}")
        return None
    full_text = pdf_to_text(paper.local_path)

    # 2. 全文 → LLM → PaperCard
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content":
            f"论文标题: {paper.title}\n"
            f"发表年份: {paper.year}\n"
            f"arXiv编号: {paper.arxiv_id}\n\n"
            f"论文全文:\n{full_text}"},
    ]
    try:
        # 提取任务要求忠实，用最低温度
        card = llm_client.chat_json(
            messages, schema_class=PaperCard,
            temperature=config.TEMP_EXTRACTOR,
        )
        # 校验卡片的arxiv_id与元数据一致（防止张冠李戴）
        card.arxiv_id = paper.arxiv_id
        card.title = paper.title
        card.year = paper.year

        n_claims = len(card.claims)
        n_quotes = sum(len(c.quotes) for c in card.claims)
        print(f"[提取Agent] 完成: 提取{n_claims}条声明, {n_quotes}条原文引用")
        return card
    except budget.BudgetExceededError:
        raise  # 预算耗尽必须上抛终止流水线，不能按"单篇失败"吞掉
    except Exception as e:
        print(f"[提取Agent] 提取失败(跳过该论文): {e}")
        return None


# ---------------------------------------------------------------
# 组合入口：批量提取
# ---------------------------------------------------------------
def run(papers: list[PaperMeta], max_workers: int = 10) -> list[PaperCard]:
    """
    批量提取所有论文的信息卡片，落盘到 data/paper_cards.json

    性能说明: 提取是LLM密集操作（每篇30-60秒），串行处理15篇要10分钟+。
    改用3线程并行（LLM API支持并发请求），整体耗时降到约1/3。
    线程安全性: extract_card内只读写各自的paper/card对象,无共享状态。

    参数:
        papers: 已下载PDF的论文元数据列表
        max_workers: 并行线程数（2026-10-15由3提到6: DeepSeek并发
                     完全可承受, 提取耗时约降一半; 配合LLM缓存,
                     命中时并行度不再影响耗时）

    返回:
        成功提取的PaperCard列表（保持输入顺序）
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map保持输入顺序; 失败的返回None
        outcomes = list(executor.map(extract_card, papers))
    cards = [c for c in outcomes if c]

    # 落盘（证据库的核心文件，第3步审查Agent从这里读取）
    out_path = os.path.join(config.DATA_DIR, "paper_cards.json")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in cards], f, ensure_ascii=False, indent=2)
    print(f"[提取Agent] 信息卡片已保存: {out_path} (共{len(cards)}篇)")
    return cards
