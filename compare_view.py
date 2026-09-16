"""
compare_view.py —— 对比阅读视图（成员C · 模块4）
================================
职责：双文献并排卡片对比，展示 背景/方法/结论/争议点 四维信息，
联动阅读笔记与全文检索跳转。

数据来源（全部本地文件，缺失自动降级）:
  papers_meta.json —— 标题/作者/年份/摘要（背景）
  paper_cards.json —— 声明卡片（method=方法 / result=结论 /
                      limitation=局限）流水线未跑完时无此文件
  conflicts.json   —— 矛盾检测产出（争议点）
  notes.json       —— 阅读笔记（模块3，联动展示）

本模块为纯数据层（不 import streamlit），渲染由 app.py 完成。
"""

import os
import json

import config


def _load(name: str, default):
    """读 data/ 下的 JSON，缺失/损坏返回 default（降级不崩）"""
    path = os.path.join(config.DATA_DIR, name)
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


# ================================================================
# 一、单篇论文画像（对比卡片的一侧）
# ================================================================

def build_paper_profile(paper_id: str) -> dict:
    """
    汇总一篇论文的对比展示数据

    返回:
        {"paper_id", "title", "year", "authors", "abstract",
         "methods": [str], "results": [str], "limitations": [str],
         "notes_count": int, "recent_notes": [dict],
         "has_card": bool, "cited_by": int}
    """
    metas = _load("papers_meta.json", [])
    meta = next((p for p in metas
                 if str(p.get("arxiv_id")) == paper_id), None)

    title = meta.get("title", paper_id) if meta else paper_id
    abstract = meta.get("abstract", "") if meta else ""
    authors = meta.get("authors", []) if meta else []

    # 声明卡片（可能不存在: 流水线只跑到一半）
    methods, results, limitations = [], [], []
    has_card = False
    cards = _load("paper_cards.json", [])
    card = next((c for c in cards
                 if str(c.get("arxiv_id")) == paper_id), None)
    if card:
        has_card = True
        for claim in card.get("claims", []):
            ctype = claim.get("claim_type", "")
            if ctype == "method":
                methods.append(claim.get("content", ""))
            elif ctype == "result":
                results.append(claim.get("content", ""))
            elif ctype == "limitation":
                limitations.append(claim.get("content", ""))

    # 笔记联动（模块3）
    notes_count, recent_notes = 0, []
    try:
        import notes_store
        all_notes = notes_store.list_notes(paper_id)
        notes_count = len(all_notes)
        recent_notes = all_notes[-3:]  # 最近3条（list已按时间升序）
    except Exception:
        pass

    return {
        "paper_id": paper_id,
        "title": title,
        "year": meta.get("year") if meta else None,
        "authors": authors,
        "abstract": abstract,
        "cited_by": meta.get("cited_by", 0) if meta else 0,
        "methods": methods,
        "results": results,
        "limitations": limitations,
        "notes_count": notes_count,
        "recent_notes": recent_notes,
        "has_card": has_card,
    }


# ================================================================
# 二、两篇之间的争议点
# ================================================================

def pair_conflicts(paper_id_a: str, paper_id_b: str,
                   limit: int = 6) -> list[dict]:
    """
    两篇论文之间的争议（conflicts.json 里 a/b 分属两边的条目）

    返回:
        [{"relation": "contradict"/"tension", "topic", "explanation",
          "a_content", "b_content", "severity"}] 按severity降序
    """
    conflicts = _load("conflicts.json", [])
    out = []
    for c in conflicts:
        ida = (c.get("a") or {}).get("arxiv_id")
        idb = (c.get("b") or {}).get("arxiv_id")
        pair = {ida, idb}
        if pair == {paper_id_a, paper_id_b}:
            out.append({
                "relation": c.get("relation", ""),
                "topic": c.get("topic", ""),
                "explanation": c.get("explanation", ""),
                "a_content": (c.get("a") or {}).get("content", ""),
                "b_content": (c.get("b") or {}).get("content", ""),
                "severity": c.get("severity", 0),
            })
    out.sort(key=lambda x: -x["severity"])
    return out[:limit]


# ================================================================
# 三、可选论文清单（下拉框数据）
# ================================================================

def selectable_papers() -> list[dict]:
    """
    可对比的论文清单: papers_meta.json 里的论文 ∪ 有本地PDF的论文
    （后者覆盖"元数据没收录但下载了PDF"的情况）

    返回: [{"id", "label"}] label 如 "标题 (2022) [id]"
    """
    items: dict[str, dict] = {}

    for p in _load("papers_meta.json", []):
        pid = str(p.get("arxiv_id", ""))
        if pid:
            items[pid] = {
                "id": pid,
                "title": p.get("title", pid),
                "year": p.get("year"),
            }

    paper_dir = config.PAPER_DIR
    if os.path.isdir(paper_dir):
        for fn in os.listdir(paper_dir):
            if fn.lower().endswith(".pdf"):
                pid = os.path.splitext(fn)[0]
                if pid not in items:
                    items[pid] = {"id": pid, "title": pid, "year": None}

    out = []
    for it in items.values():
        yr = f" ({it['year']})" if it.get("year") else ""
        out.append({"id": it["id"],
                    "label": f"{it['title'][:60]}{yr} [{it['id']}]"})
    return out


# ================================================================
# 自测
# ================================================================
if __name__ == "__main__":
    import shutil

    DATA = config.DATA_DIR
    os.makedirs(DATA, exist_ok=True)

    # 备份并造测试数据（cards + conflicts 真实路径）
    backups = {}
    test_files = {
        "paper_cards.json": [
            {"arxiv_id": "2210.04195", "title": "OTTT", "year": 2022,
             "method_category": "在线训练",
             "claims": [
                 {"claim_type": "method",
                  "content": "提出通过时间的在线训练",
                  "quotes": [{"section": "3", "text": "online"}]},
                 {"claim_type": "result",
                  "content": "CIFAR-10达93.1%",
                  "quotes": [{"section": "4", "text": "93.1"}]},
                 {"claim_type": "limitation",
                  "content": "仅适用于直接编码SNN",
                  "quotes": [{"section": "5", "text": "only"}]},
             ]},
            {"arxiv_id": "2202.11946", "title": "PaperB", "year": 2022,
             "method_category": "代理梯度",
             "claims": [
                 {"claim_type": "method",
                  "content": "系统化研究代理梯度设计空间",
                  "quotes": [{"section": "3", "text": "sg"}]},
                 {"claim_type": "result",
                  "content": "给出代理梯度选择准则",
                  "quotes": [{"section": "4", "text": "guide"}]},
             ]},
        ],
        "conflicts.json": [
            {"a": {"arxiv_id": "2210.04195", "content": "A声明",
                   "paper_title": "OTTT", "year": 2022, "key": "x#0",
                   "claim_index": 0},
             "b": {"arxiv_id": "2202.11946", "content": "B声明",
                   "paper_title": "PaperB", "year": 2022, "key": "y#0",
                   "claim_index": 0},
             "relation": "contradict", "topic": "训练范式",
             "explanation": "两者对训练方式的结论相反",
             "severity": 0.8},
            {"a": {"arxiv_id": "1111.11111", "content": "无关A",
                   "paper_title": "X", "year": 2020, "key": "x#0",
                   "claim_index": 0},
             "b": {"arxiv_id": "2210.04195", "content": "OTTT声明",
                   "paper_title": "OTTT", "year": 2022, "key": "y#0",
                   "claim_index": 0},
             "relation": "tension", "topic": "其他",
             "explanation": "无关争议", "severity": 0.5},
        ],
    }
    for name, content in test_files.items():
        fp = os.path.join(DATA, name)
        if os.path.exists(fp):
            backups[name] = fp + ".bak_cmp"
            shutil.copy(fp, backups[name])
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False)

    # 笔记备份（notes_store 自测可能留了空文件，无碍）
    notes_fp = os.path.join(DATA, "notes.json")
    had_notes = os.path.exists(notes_fp)
    if had_notes:
        shutil.copy(notes_fp, notes_fp + ".bak_cmp")

    try:
        import notes_store
        notes_store.add_note("2210.04195", 2, "引用文本", "测试笔记",
                             paper_title="OTTT")

        print("== 1. 论文画像 ==")
        prof = build_paper_profile("2210.04195")
        assert prof["title"] == "Online Training Through Time for Spiking Neural Networks"
        assert prof["year"] == 2022
        assert "Mingqing Xiao" in prof["authors"]
        assert "brain-inspired" in prof["abstract"][:120] or len(prof["abstract"]) > 50
        assert prof["has_card"] is True
        assert len(prof["methods"]) == 1
        assert len(prof["results"]) == 1
        assert len(prof["limitations"]) == 1
        assert prof["notes_count"] == 1
        assert prof["recent_notes"][0]["note"] == "测试笔记"
        print("  元数据/claims分类/笔记联动 正确")

        print("== 2. 争议点 ==")
        pcs = pair_conflicts("2210.04195", "2202.11946")
        assert len(pcs) == 1
        assert pcs[0]["relation"] == "contradict"
        assert pcs[0]["severity"] == 0.8
        assert pcs[0]["a_content"] == "A声明"
        # 无冲突对
        assert pair_conflicts("2210.04195", "9999.99999") == []
        print("  两篇间争议/无关争议过滤 正确")

        print("== 3. 可选清单 ==")
        sel = selectable_papers()
        ids = {s["id"] for s in sel}
        assert "2210.04195" in ids and "2202.11946" in ids
        # 本地PDF但元数据没有的也应在列（papers/里有9篇，meta只有5篇）
        pdf_ids = {f[:-4] for f in os.listdir(config.PAPER_DIR)
                   if f.endswith(".pdf")}
        assert pdf_ids <= ids, "本地PDF未全部进入可选清单"
        lab = next(s for s in sel if s["id"] == "2210.04195")
        assert "[2210.04195]" in lab["label"] and "2022" in lab["label"]
        print(f"  共{len(sel)}篇可选（含本地PDF兜底），标签格式正确")

        print("== 4. 降级路径 ==")
        # 造一个不在cards里的论文
        prof2 = build_paper_profile("2302.10685")
        if not prof2["has_card"]:
            assert prof2["methods"] == [] and prof2["results"] == []
            print("  无cards时方法/结论为空（前端提示降级） 正确")
        # 不存在的论文ID
        prof3 = build_paper_profile("0000.00000")
        assert prof3["title"] == "0000.00000" and prof3["abstract"] == ""
        assert prof3["notes_count"] == 0
        print("  未知论文ID安全兜底 正确")

        print("\n模块4数据层自测全部通过")
    finally:
        # 还原
        for name in test_files:
            fp = os.path.join(DATA, name)
            bak = fp + ".bak_cmp"
            if os.path.exists(fp):
                os.remove(fp)
            if os.path.exists(bak):
                shutil.copy(bak, fp)
                os.remove(bak)
        if had_notes and os.path.exists(notes_fp + ".bak_cmp"):
            shutil.copy(notes_fp + ".bak_cmp", notes_fp)
            os.remove(notes_fp + ".bak_cmp")
        else:
            # 没备份说明原本没有笔记，清掉自测写入的
            if os.path.exists(notes_fp):
                os.remove(notes_fp)
