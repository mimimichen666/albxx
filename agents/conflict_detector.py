"""
conflict_detector.py —— 矛盾检测Agent（方案三：学术争议发现）
================================
职责：
  在"已通过核验的声明"之间检测学术争议——
  论文A说X有效 vs 论文B说X无效 这类观点冲突，
  让系统从"信息聚合器"升维成"洞察生成器"。

【两阶段设计（控制成本的关键）】
  全量声明两两配对是 O(n²)（15篇论文60条声明=1770对），
  不能全部送LLM。因此:
    阶段1 程序预筛: 只保留"同类声明 + 关键词重叠≥2个领域术语"的候选对
            （矛盾只可能发生在讨论相近主题的声明之间）
    阶段2 LLM精判: 仅对候选对判定四种关系:
            contradict(直接对立) / tension(张力:条件不同导致的表象冲突)
            / consistent(一致) / unrelated(不可比)

【重要的质量约束】
  只在已核验声明之间检测——拿幻觉声明检测矛盾会产出"虚构的争议"。
  矛盾检测的质量上限由审查机制保证（两模块协同的证明）。

输入: data/paper_cards.json + review_results.json
输出: data/conflicts.json（争议列表，前端"⚡学术争议"标签页消费）
"""

import os
import json
import re
from itertools import combinations

import llm_client
import config
from pydantic import BaseModel, Field


class PairRelation(BaseModel):
    """一对声明的LLM判定结果"""
    relation: str = Field(
        description="两声明的语义关系，只能是以下之一: "
                    "contradict(直接矛盾: A说X, B说非X) | "
                    "tension(张力: 结论表面冲突但条件不同,如数据集/场景不同) | "
                    "consistent(一致或互补) | unrelated(话题不可比)"
    )
    explanation: str = Field(
        description="一句话说明冲突点（contradict/tension时）或为何一致/不可比"
    )
    topic: str = Field(
        description="这对声明共同讨论的主题，3-8个字（如'代理梯度精度上限'）"
    )
    severity: int = Field(
        description="争议的学术重要性1-5。5=领域核心争议, 1=细枝末节。"
                    "consistent/unrelated时填1"
    )


# ---------------------------------------------------------------
# 声明池（与qa_agent同源：只用已核验声明）
# ---------------------------------------------------------------
def _load_verified_claims() -> list[dict]:
    """加载已核验声明（附论文信息 + 核验结论/置信度/引用，阶段5证据链）"""
    cards_path = os.path.join(config.DATA_DIR, "paper_cards.json")
    review_path = os.path.join(config.DATA_DIR, "review_results.json")
    if not os.path.exists(cards_path) or not os.path.exists(review_path):
        return []

    cards = json.load(open(cards_path, encoding="utf-8"))
    reviews = json.load(open(review_path, encoding="utf-8"))

    # 核验通过的声明详情（key -> review记录，供携带置信度）
    review_by_key = {
        f"{r['arxiv_id']}#{r['claim_index']}": r for r in reviews
        if (r.get("verdict") or {}).get("verdict") == "supported"
    }

    claims = []
    for card in cards:
        for idx, claim in enumerate(card.get("claims", [])):
            key = f"{card['arxiv_id']}#{idx}"
            r = review_by_key.get(key)
            if r is None:
                continue
            quotes = claim.get("quotes", [])
            claims.append({
                "key": key,  # 证据ID，与evidence_index/问答池口径一致
                "arxiv_id": card["arxiv_id"],
                "paper_title": card.get("title", "")[:50],
                "year": card.get("year"),
                "type": claim.get("claim_type"),
                "content": claim.get("content", ""),
                "confidence": (r.get("verdict") or {}).get("confidence"),
                "quote": (quotes[0].get("text", "")[:120]
                          if quotes else ""),
            })
    return claims


# ---------------------------------------------------------------
# 阶段1: 程序预筛（关键词重叠）
# ---------------------------------------------------------------
# 英文停用词（声明多为中文，但术语是英文的也要保留比较）
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "for",
    "with", "is", "are", "was", "were", "by", "as", "at", "that",
    "this", "these", "those", "it", "its", "from", "be", "been",
    "我们", "提出", "通过", "以及", "并且", "但是", "对于", "可以",
    "能够", "一个", "这个", "具有", "在", "上", "中", "的", "了",
    "和", "与", "或", "等", "并", "对", "是", "其", "该", "此",
}


def _extract_terms(text: str) -> set[str]:
    """
    提取声明中的领域特征词（用于预筛判断两声明是否讨论相近主题）

    策略（实测教训: 中文连续串切分是随机片段，跨声明对不上）:
      - 英文: 完整单词（含连字符术语），小写化
      - 中文: bigram（相邻2字滑窗）——"卷积神经网络"切出
        {卷积,积神,神经,经网,网络}, 两句都谈卷积神经网络时
        必然共享多个bigram，重叠计数天然鲁棒
      - 去停用词后返回
    """
    text_l = text.lower()
    # 英文词（含连字符的术语如 high-order）
    en_words = set(re.findall(r"[a-z][a-z-]{2,}", text_l))
    # 中文bigram
    cn_chars = re.findall(r"[\u4e00-\u9fa5]", text_l)
    cn_bigrams = {cn_chars[i] + cn_chars[i + 1]
                  for i in range(len(cn_chars) - 1)}
    return (en_words | cn_bigrams) - _STOPWORDS


def generate_candidates(claims: list[dict], min_overlap: int = 2
                        ) -> list[tuple[dict, dict, set[str]]]:
    """
    生成候选声明对（预筛）

    筛选条件:
      1. 跨论文（同论文内的声明是同一作者的观点，不构成学术争议）
      2. 声明类型相同或相近（method↔method, result↔result, limitation可跨）
      3. 术语重叠 >= min_overlap 个（讨论相近主题才可能矛盾）

    返回:
        [(声明A, 声明B, 重叠术语集), ...]
    """
    # 声明类型分组: 同组才比对（limitation与result也常冲突，放一起）
    TYPE_GROUP = {"method": "core", "result": "core",
                  "limitation": "core"}  # 目前三类都算core组

    cands = []
    for a, b in combinations(claims, 2):
        # 条件1: 跨论文
        if a["arxiv_id"] == b["arxiv_id"]:
            continue
        # 条件2: 同类型组
        if TYPE_GROUP.get(a["type"]) != TYPE_GROUP.get(b["type"]):
            continue
        # 条件3: 术语重叠
        overlap = _extract_terms(a["content"]) & _extract_terms(b["content"])
        if len(overlap) >= min_overlap:
            cands.append((a, b, overlap))
    return cands


# ---------------------------------------------------------------
# 阶段2: LLM精判
# ---------------------------------------------------------------
def _judge_pair(a: dict, b: dict) -> PairRelation:
    """单对声明的LLM关系判定"""
    prompt = f"""你是文献分析专家，判断两条来自不同论文的声明的关系。

声明A [论文《{a['paper_title']}》{a['year']}年, arXiv:{a['arxiv_id']}]:
{a['content']}

声明B [论文《{b['paper_title']}》{b['year']}年, arXiv:{b['arxiv_id']}]:
{b['content']}

判定标准:
- contradict: 直接对立（A说X有效，B说X无效；A说存在，B说不存在）
- tension: 张力（结论表面冲突但实验条件不同——不同数据集/任务/规模，
  这种"表象冲突"是学术争议的常见形态，务必与contradict区分）
- consistent: 一致或互补（互相支持或讨论不同方面不冲突）
- unrelated: 话题不可比（虽然共享一些词但讨论的不是一个东西）"""

    return llm_client.chat_json(
        [{"role": "system",
          "content": "你是严谨的学术文献分析专家，只输出JSON。"},
         {"role": "user", "content": prompt}],
        PairRelation, temperature=0.1,
    )


# ---------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------
def run(max_pairs: int = 60) -> list[dict]:
    """
    完整的矛盾检测流水线

    参数:
        max_pairs: 送LLM精判的最大对数（控制成本上限）

    返回:
        争议列表 [{"a":..., "b":..., "relation":..., "explanation":...,
                 "topic":..., "severity":...}] 并落盘 conflicts.json
    """
    claims = _load_verified_claims()
    if len(claims) < 2:
        print("[矛盾检测] 已核验声明不足2条，跳过")
        return []

    # ---- 阶段1: 预筛 ----
    all_pairs = len(claims) * (len(claims) - 1) // 2
    cands = generate_candidates(claims)
    print(f"[矛盾检测] 预筛: {all_pairs}全量对 -> {len(cands)}候选对 "
          f"(压缩{100 - len(cands) * 100 // max(all_pairs, 1)}%)")

    # 候选对太多时截断（按重叠术语数降序——重叠越多越可能真冲突）
    if len(cands) > max_pairs:
        cands.sort(key=lambda x: len(x[2]), reverse=True)
        cands = cands[:max_pairs]
        print(f"[矛盾检测] 截断到前{max_pairs}对")

    # ---- 阶段2: LLM精判（3线程并行）----
    from concurrent.futures import ThreadPoolExecutor

    def _safe_judge(pair):
        a, b, _ = pair
        try:
            return _judge_pair(a, b)
        except Exception as e:
            print(f"[矛盾检测] 判定失败: {e}")
            return None

    # 2026-10-16: 6→10线程, DeepSeek并发余量充足
    with ThreadPoolExecutor(max_workers=10) as executor:
        verdicts = list(executor.map(_safe_judge, cands))

    # ---- 汇总: 只保留矛盾/张力 ----
    conflicts = []
    for (a, b, overlap), verdict in zip(cands, verdicts):
        if verdict is None:
            continue
        if verdict.relation in ("contradict", "tension"):
            conflicts.append({
                "a": {"key": a["key"], "paper_title": a["paper_title"],
                      "year": a["year"], "content": a["content"],
                      "arxiv_id": a["arxiv_id"], "claim_index":
                      int(a["key"].split("#")[1]),
                      "confidence": a.get("confidence"),
                      "quote": a.get("quote", "")},
                "b": {"key": b["key"], "paper_title": b["paper_title"],
                      "year": b["year"], "content": b["content"],
                      "arxiv_id": b["arxiv_id"], "claim_index":
                      int(b["key"].split("#")[1]),
                      "confidence": b.get("confidence"),
                      "quote": b.get("quote", "")},
                "relation": verdict.relation,
                "topic": verdict.topic,
                "explanation": verdict.explanation,
                "severity": verdict.severity,
            })

    # 按严重度降序
    conflicts.sort(key=lambda c: c["severity"], reverse=True)

    # ---- 落盘 ----
    out_path = os.path.join(config.DATA_DIR, "conflicts.json")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conflicts, f, ensure_ascii=False, indent=1)

    n_con = sum(1 for c in conflicts if c["relation"] == "contradict")
    n_ten = len(conflicts) - n_con
    print(f"[矛盾检测] 完成: 直接矛盾{n_con}处 + 张力{n_ten}处, "
          f"已保存 {out_path}")
    return conflicts


if __name__ == "__main__":
    # 独立测试
    results = run()
    for c in results:
        print(f"\n[{c['relation']}|sev{c['severity']}] {c['topic']}")
        print(f"  A({c['a']['year']}): {c['a']['content'][:50]}")
        print(f"  B({c['b']['year']}): {c['b']['content'][:50]}")
        print(f"  分析: {c['explanation']}")
