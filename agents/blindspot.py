"""
agents/blindspot.py —— P0-1 审稿人视角盲区报告
================================================================================
综述生成后，系统反向批判自己的综述。输入是 coverage.py 聚合好的
统计数据（每个数字可复现），LLM 只负责解读，不允许发明数字。

条目结构（每个盲区都带"数据依据"原句，来自程序统计）:
  B1 单一来源区域 —— 方法分类仅 1 篇论文支撑
  B2 检索方向缺口 —— 检索词命中过少
  B3 弱引用论文   —— 报告正文从未 [arXiv:ID] 标注的论文
  B4 弱声明       —— 暂缓判定 / 低置信通过
  K4 全覆盖反向校验 —— 未覆盖论文清单

LLM 失败时自动降级为纯数据版报告，保证 output/blindspots.md 必然产出。
"""

import json
import os

import config

SYSTEM_PROMPT = (
    "你是一位苛刻但公正的学术审稿人。你只会使用用户给你的统计数据，"
    "绝不发明任何数字、论文或结论。每个盲区条目必须逐字引用给出的"
    "「数据依据」句（放在 > 引用块里），再附 1-2 句审稿人视角的解读。"
)


def _evidence_lines(cov: dict) -> list[str]:
    """从聚合数据生成数据依据句——报告里每个条目都会引用其中之一"""
    lines: list[str] = []
    k4 = cov.get("k4") or {}
    unc = k4.get("uncovered") or []
    for i, u in enumerate(unc[:8], 1):
        lines.append(
            f"[K4未覆盖{i}] 检索到的论文《{u['title']}》"
            f"(arXiv:{u['arxiv_id']}) 未在报告正文出现：{u['reason']}。"
            f"当前报告覆盖率 {k4.get('coverage_rate', 0):.0%}"
            f"（{len(k4.get('covered') or [])}/{cov.get('total_papers', 0)} 篇）。")
    for i, a in enumerate(cov.get("single_source_areas") or [], 1):
        ps = "; ".join(
            f"{p['title']}(arXiv:{p['arxiv_id']}, "
            f"可信声明{a['supported_claims']}条)"
            for p in a["papers"])
        lines.append(
            f"[单源区{i}] 方法分类「{a['category']}」下仅 1 篇论文：{ps}。"
            "该分类下所有结论均只有单一来源。")
    for i, g in enumerate(cov.get("keyword_gaps") or [], 1):
        if g["hits"] <= 1:
            lines.append(
                f"[检索缺口{i}] 检索计划关键词「{g['keyword']}」在最终论文集"
                f"（{cov.get('total_papers', 0)} 篇的标题+摘要）中仅命中 "
                f"{g['hits']} 篇——该研究方向可能覆盖不足。")
    for i, p in enumerate(cov.get("citation_ranking") or [], 1):
        if p["mentions"] == 0 and p["covered"]:
            lines.append(
                f"[弱引用{i}] 论文《{p['title']}》(arXiv:{p['arxiv_id']}) "
                "被报告收录（标题出现），但正文从未以 [arXiv:ID] 标注引用——"
                "其信息实际未被综述正文使用。")
    for i, w in enumerate(cov.get("weak_claims") or [], 1):
        lines.append(
            f"[弱声明{i}] 《{w['title']}》声明「{w['claim_content']}」"
            f"判定={w['verdict']}，置信度仅 {w['confidence']}。")
    return lines


def _fallback_md(topic: str, cov: dict, lines: list[str]) -> str:
    """纯程序化报告（LLM 失败/无盲区时的兜底，数字全部来自 coverage.py）"""
    k4 = cov.get("k4") or {}
    dist = cov.get("verdict_distribution") or {}
    head = (
        "# 🕵️ 审稿人视角盲区报告（数据版）\n\n"
        f"> 自动生成于 {cov.get('generated_at', '')} · 主题: {topic}\n"
        "> 本报告由程序统计生成，每个数字均可从任务快照文件复现。\n\n"
        "## 摘要\n\n"
        f"- 检索论文 **{cov.get('total_papers', 0)}** 篇，报告覆盖 "
        f"**{len(k4.get('covered') or [])}** 篇"
        f"（K4 覆盖率 {k4.get('coverage_rate', 0):.0%}）\n"
        f"- 送审声明 **{cov.get('total_claims', 0)}** 条，判定分布: "
        + "、".join(f"{k} {v}" for k, v in dist.items()) + "\n"
        f"- 盲区条目 **{len(lines)}** 条\n\n")
    if not lines:
        return head + (
            "## ✅ 未检出显著盲区\n\n"
            "本次运行未发现单一来源区域、检索缺口、未覆盖论文或弱声明。\n"
            "这不代表报告完美——仅表示当前统计阈值下各项指标健康。\n")
    body = "".join(f"- {ln}\n" for ln in lines)
    return head + (
        "## 盲区清单（每条附数据依据）\n\n" + body +
        "\n> 说明: 以上条目由程序从任务产物统计得出，"
        "「仅 1 篇支撑」「命中 0 篇」等数字可在 "
        "paper_cards.json / review_results.json / plan_last.json 中复核。\n")


def generate(topic: str, cov: dict) -> str:
    """生成盲区报告并写入 output/blindspots.md，返回报告全文"""
    lines = _evidence_lines(cov)
    fallback = _fallback_md(topic, cov, lines)
    md = fallback
    if lines:
        prompt = (
            "以下是本次文献综述的统计聚合数据（全部为程序统计，数字真实）。\n"
            f"研究主题: {topic}\n"
            "请生成一份 Markdown 格式的《审稿人视角盲区报告》，要求:\n\n"
            "1. 开头为一级标题「# 🕵️ 审稿人视角盲区报告」与一段 60 字内摘要\n"
            "2. 按以下四个章节组织（无相关条目的章节写「未检出」）:\n"
            "   - ## 单一来源结论（置信度低）\n"
            "   - ## 子领域覆盖不足\n"
            "   - ## 收录但未使用的论文\n"
            "   - ## 弱声明与暂缓判定\n"
            "3. 每个条目格式: ### 简短标题，正文 1-2 句审稿人解读，\n"
            "   然后另起一行放「> 📊 数据依据: …」——必须逐字引用\n"
            "   输入中的对应 [方括号] 依据句，不得改动数字\n"
            "4. 结尾附「## K4 全覆盖反向校验」: 说明覆盖率与未覆盖论文\n"
            "5. 不得发明任何未在数据依据中出现的数字或论文\n\n"
            "统计依据清单:\n" + "\n".join(lines) + "\n\n"
            "补充统计: 判定分布 " +
            json.dumps(cov.get("verdict_distribution") or {}, ensure_ascii=False) +
            f"；送审声明总数 {cov.get('total_claims', 0)}。")
        try:
            import llm_client
            out = llm_client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                temperature=0.3)
            if out and len(out.strip()) > 150:
                md = out.strip()
        except Exception:
            md = fallback            # LLM 失败 → 纯数据版兜底
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, "blindspots.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return md
