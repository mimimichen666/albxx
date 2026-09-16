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

import pymupdf

import llm_client
import config
from models import PaperMeta, PaperCard

# ---------------------------------------------------------------
# 第1段：PDF → 纯文本
# ---------------------------------------------------------------
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
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        # 简易清洗：去掉连续3个以上的换行（PDF排版残留）
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        pages.append(text)
    doc.close()

    full_text = "\n".join(pages)

    # 二页码噪声处理：把单独成行的数字（页码）去掉
    lines = [ln for ln in full_text.split("\n")
             if not (ln.strip().isdigit() and len(ln.strip()) <= 4)]
    full_text = "\n".join(lines)

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
    except Exception as e:
        print(f"[提取Agent] 提取失败(跳过该论文): {e}")
        return None


# ---------------------------------------------------------------
# 组合入口：批量提取
# ---------------------------------------------------------------
def run(papers: list[PaperMeta], max_workers: int = 3) -> list[PaperCard]:
    """
    批量提取所有论文的信息卡片，落盘到 data/paper_cards.json

    性能说明: 提取是LLM密集操作（每篇30-60秒），串行处理15篇要10分钟+。
    改用3线程并行（LLM API支持并发请求），整体耗时降到约1/3。
    线程安全性: extract_card内只读写各自的paper/card对象,无共享状态。

    参数:
        papers: 已下载PDF的论文元数据列表
        max_workers: 并行线程数（3为API并发安全值）

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
