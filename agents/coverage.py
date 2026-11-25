"""
agents/coverage.py —— 覆盖度分析模块（K4 全覆盖反向校验 与 P0-1 盲区报告 共用）
================================================================================
只读现成产物（papers_meta / paper_cards / review_results / plan_last /
final_report），不发起任何检索调用、不做核验级 LLM 调用——
所有数字都能从文件中复现，保证"数据依据"可追溯（答辩可现场验证）。

K4 全覆盖反向校验:
  正向流程是"论文 → 报告"；反向校验反过来逐篇检查检索到的论文
  是否都被报告覆盖（正文含 [arXiv:ID] 标注，或标题片段出现），
  不允许任何论文被"静默丢弃"。

P0-1 盲区聚合（供 blindspot.py 生成审稿人视角报告）:
  - 单一来源区域: 某方法分类下仅 1 篇论文 → 该类结论全部单来源
  - 论文引用强度: 每篇论文在报告正文中被 [arXiv:ID] 标注的次数
  - 检索方向缺口: 检索计划关键词在最终论文集（标题+摘要）中的命中数
  - 弱声明: 暂缓判定(abstained) / 低置信 supported 的声明
"""

import json
import os
import re
import time

import config


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _norm(s: str) -> str:
    """小写并去掉空白/标点，用于标题匹配（容忍报告里的排版差异）"""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", (s or "").lower())


def _verdict_of(r: dict) -> tuple[str, float]:
    """兼容新旧 review_results.json：verdict 可能是 dict 或 str"""
    v = r.get("verdict")
    if isinstance(v, dict):
        return v.get("verdict", ""), float(v.get("confidence", 1.0) or 0)
    return (v or ""), 1.0


# ----------------------------------------------------------------
# 单次分析（纯函数，便于单测与前端复算）
# ----------------------------------------------------------------
def analyze(topic: str, papers_raw, cards_raw, reviews_raw,
            plan_raw, report_md: str) -> dict:
    papers = papers_raw or []
    cards = cards_raw or []
    reviews = reviews_raw or []
    report_md = report_md or ""
    rep_norm = _norm(report_md)

    # ---- 可信声明索引: "arxiv_id#claim_index" -> True ----
    supported_keys = set()
    dist: dict[str, int] = {}
    weak_claims = []
    cards_by_id = {c.get("arxiv_id"): c for c in cards}
    for r in reviews:
        try:
            vv, conf = _verdict_of(r)
        except Exception:
            continue
        dist[vv] = dist.get(vv, 0) + 1
        if vv == "supported":
            supported_keys.add(f"{r.get('arxiv_id')}#{r.get('claim_index')}")
        if vv == "abstained" or (vv == "supported" and conf < 0.75):
            card = cards_by_id.get(r.get("arxiv_id")) or {}
            cl = (card.get("claims") or [{}])[r.get("claim_index")] \
                if isinstance(card.get("claims"), list) and \
                len(card.get("claims")) > (r.get("claim_index") or 0) else {}
            weak_claims.append({
                "arxiv_id": r.get("arxiv_id"),
                "title": card.get("title", ""),
                "claim_index": r.get("claim_index"),
                "claim_content": (cl.get("content") or
                                  r.get("claim_content") or "")[:120],
                "verdict": vv,
                "confidence": round(conf, 2),
            })
    weak_claims.sort(key=lambda w: w["confidence"])
    weak_claims = weak_claims[:12]

    # ---- K4: 逐篇论文是否被报告覆盖 ----
    per_paper = []
    covered_ids = []
    uncovered = []
    for p in papers:
        aid, title = p.get("arxiv_id", ""), p.get("title", "")
        mentions = report_md.count(f"[arXiv:{aid}]")
        tkey = _norm(title)
        title_hit = len(tkey) >= 12 and tkey[:24] in rep_norm
        is_cov = mentions > 0 or title_hit
        entry = {"arxiv_id": aid, "title": title,
                 "mentions": mentions, "title_match": title_hit,
                 "covered": is_cov}
        per_paper.append(entry)
        if is_cov:
            covered_ids.append(aid)
        else:
            uncovered.append({
                "arxiv_id": aid, "title": title,
                "reason": "报告正文无 [arXiv:ID] 标注，标题也未出现",
            })
    total = len(papers)
    k4 = {
        "coverage_rate": (len(covered_ids) / total) if total else 0.0,
        "covered": covered_ids,
        "uncovered": uncovered,
        "checked_by": "[arXiv:ID] 标注或标题片段匹配（确定性检测，无LLM）",
    }

    # ---- 论文引用强度（报告中被标注次数排序）----
    citation_ranking = sorted(per_paper, key=lambda e: -e["mentions"])

    # ---- 单一来源区域: 方法分类下仅 1 篇论文 ----
    by_cat: dict[str, list] = {}
    for c in cards:
        by_cat.setdefault(c.get("method_category") or "未分类", []).append(c)
    single_source_areas = []
    for cat, cs in sorted(by_cat.items(), key=lambda kv: len(kv[1])):
        if len(cs) != 1:
            continue
        card = cs[0]
        sup = sum(1 for i in range(len(card.get("claims") or []))
                  if f"{card.get('arxiv_id')}#{i}" in supported_keys)
        single_source_areas.append({
            "category": cat,
            "papers": [{"arxiv_id": card.get("arxiv_id"),
                        "title": card.get("title", "")}],
            "supported_claims": sup,
        })

    # ---- 检索方向缺口: 计划关键词在论文集(标题+摘要)中的命中数 ----
    keyword_gaps = []
    for kw in (plan_raw or {}).get("keywords") or []:
        k = str(kw).lower()
        hits = sum(1 for p in papers
                   if k in ((p.get("title") or "") + " " +
                            (p.get("abstract") or "")).lower())
        keyword_gaps.append({"keyword": kw, "hits": hits})
    keyword_gaps.sort(key=lambda g: g["hits"])

    return {
        "topic": topic,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_papers": total,
        "total_claims": len(reviews),
        "k4": k4,
        "verdict_distribution": dist,
        "single_source_areas": single_source_areas,
        "keyword_gaps": keyword_gaps,
        "citation_ranking": citation_ranking,
        "weak_claims": weak_claims,
    }


# ----------------------------------------------------------------
# 流水线入口: 读共享产物 → 分析 → 落盘 coverage_check.json
# ----------------------------------------------------------------
def run(topic: str = "") -> dict:
    data = config.DATA_DIR
    cov = analyze(
        topic,
        _load(os.path.join(data, "papers_meta.json")),
        _load(os.path.join(data, "paper_cards.json")),
        _load(os.path.join(data, "review_results.json")),
        _load(os.path.join(data, "plan_last.json")),
        _read_report(),
    )
    save(cov)
    return cov


def _read_report() -> str:
    try:
        with open(os.path.join(config.OUTPUT_DIR, "final_report.md"),
                  encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def save(cov: dict) -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, "coverage_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cov, f, ensure_ascii=False, indent=1)
    return path
