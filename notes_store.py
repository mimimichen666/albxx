"""
notes_store.py —— 阅读笔记模块（成员C · 模块3）
================================
职责：为本地 PDF 提供页面级批注（阅读笔记），绑定文献ID+页码，
本地 JSON 持久化，支持增删改查。

设计要点（对应任务要求）:
  1. 绑定文献ID和页码: 每条笔记必填 paper_id(arxiv_id) 与
     page(0-based，与PDF阅读器页标-1一致，显示时+1)。
  2. 本地持久化: data/notes.json（与全系统数据存放口径一致，
     不引入数据库）。
  3. 选中段落批注: quote 字段承载用户选中的原文段落
     （PDF以图片渲染，"选中"通过复制原文粘贴实现——
     全文搜索的结果片段可一键带入）。
  4. 增删改: add_note / update_note / delete_note / list_notes /
     notes_for_page 五个原语，UI层直接组合。

本模块为纯数据层（不 import streamlit），可独立自测。
"""

import os
import json
import time
import uuid

import config

NOTES_PATH = os.path.join(config.DATA_DIR, "notes.json")


def _load() -> list[dict]:
    if not os.path.exists(NOTES_PATH):
        return []
    try:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        # 损坏时不崩: 返回空并备份损坏文件（防呆）
        try:
            os.replace(NOTES_PATH, NOTES_PATH + ".corrupt")
        except OSError:
            pass
        return []


def _save(notes: list[dict]):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp = NOTES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=1)
    os.replace(tmp, NOTES_PATH)  # 原子替换，避免写一半损坏


# ================================================================
# 五个数据原语
# ================================================================

def add_note(paper_id: str, page: int, quote: str, note: str,
             paper_title: str = "") -> dict:
    """
    添加一条笔记（绑定文献ID+页码）

    返回: 新建的笔记 dict（含 id/created_at）
    抛出: ValueError —— note 与 quote 全空（空笔记无意义）
    """
    quote = (quote or "").strip()
    note = (note or "").strip()
    if not quote and not note:
        raise ValueError("原文引用与批注内容不能同时为空")
    if page < 0:
        raise ValueError("页码不能为负")

    entry = {
        "id": uuid.uuid4().hex[:12],
        "paper_id": str(paper_id),
        "paper_title": str(paper_title or ""),  # 冗余标题便于跨模块展示
        "page": int(page),
        "quote": quote[:2000],      # 选中段落（长文本边界: 截断上限）
        "note": note[:4000],        # 批注内容
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    notes = _load()
    notes.append(entry)
    _save(notes)
    return entry


def update_note(note_id: str, quote: str | None = None,
                note: str | None = None,
                page: int | None = None) -> bool:
    """
    修改笔记（字段级: 只更新传入的字段，None=保持不变）

    返回: False 表示 note_id 不存在
    """
    notes = _load()
    for e in notes:
        if e["id"] == note_id:
            if quote is not None:
                e["quote"] = quote.strip()[:2000]
            if note is not None:
                e["note"] = note.strip()[:4000]
            if page is not None and page >= 0:
                e["page"] = int(page)
            e["updated_at"] = int(time.time())
            _save(notes)
            return True
    return False


def delete_note(note_id: str) -> bool:
    """删除笔记; 返回 False 表示不存在"""
    notes = _load()
    before = len(notes)
    notes = [e for e in notes if e["id"] != note_id]
    if len(notes) == before:
        return False
    _save(notes)
    return True


def list_notes(paper_id: str | None = None) -> list[dict]:
    """
    全部笔记（可按文献过滤），按 (paper_id, page, created_at) 排序
    —— 模块4对比视图与tab8笔记面板共用
    """
    notes = _load()
    if paper_id is not None:
        notes = [e for e in notes if e["paper_id"] == paper_id]
    return sorted(notes, key=lambda e: (e["paper_id"], e["page"],
                                         e["created_at"]))


def notes_for_page(paper_id: str, page: int) -> list[dict]:
    """指定文献指定页的笔记（阅读器逐页展示用）"""
    return [e for e in _load()
            if e["paper_id"] == paper_id and e["page"] == page]


def count_notes(paper_id: str) -> int:
    """某篇笔记数（模块4对比卡片联动用）"""
    return sum(1 for e in _load() if e["paper_id"] == paper_id)


# ================================================================
# 自测
# ================================================================
if __name__ == "__main__":
    import shutil

    # 备份已有笔记（自测用临时文件，不动真实数据）
    backup = None
    if os.path.exists(NOTES_PATH):
        backup = NOTES_PATH + ".bak_test"
        shutil.copy(NOTES_PATH, backup)

    try:
        # 清场
        if os.path.exists(NOTES_PATH):
            os.remove(NOTES_PATH)

        print("== 1. 增 ==")
        n1 = add_note("2210.04195", 3, "surrogate gradient is used",
                      "这就是代理梯度，作者在第三页开头讲的")
        assert n1["paper_id"] == "2210.04195" and n1["page"] == 3
        assert n1["id"] and n1["created_at"] > 0
        n2 = add_note("2210.04195", 7, "", "整页思路概括")
        n3 = add_note("2202.11946", 1, "intro 段落", "背景相关")
        print(f"  3条笔记已添加: {n1['id']}, {n2['id']}, {n3['id']}")

        print("== 2. 查 ==")
        all_notes = list_notes()
        assert len(all_notes) == 3
        pg = notes_for_page("2210.04195", 3)
        assert len(pg) == 1 and pg[0]["note"].startswith("这就是")
        assert count_notes("2210.04195") == 2
        assert count_notes("2202.11946") == 1
        # 排序: paper_id升序 -> 2202在前
        assert all_notes[0]["paper_id"] == "2202.11946"
        print("  过滤/计数/排序正确")

        print("== 3. 改 ==")
        assert update_note(n1["id"], note="修正后的理解")
        assert update_note(n1["id"], page=4)
        assert not update_note("nonexistent")
        pg2 = notes_for_page("2210.04195", 4)
        assert len(pg2) == 1 and pg2[0]["note"] == "修正后的理解"
        assert pg2[0]["updated_at"] >= n1["created_at"]
        print("  字段级更新/页码变更/不存在ID均正确")

        print("== 4. 删 ==")
        assert delete_note(n2["id"])
        assert not delete_note(n2["id"])  # 重复删
        assert count_notes("2210.04195") == 1
        print("  删除/重复删除防护正确")

        print("== 5. 边界 ==")
        try:
            add_note("x", 0, "", "")
            assert False, "空笔记应报错"
        except ValueError:
            pass
        try:
            add_note("x", -1, "q", "n")
            assert False, "负页码应报错"
        except ValueError:
            pass
        # 长文本截断
        long_note = add_note("x", 0, "q" * 9999, "n" * 9999)
        assert len(long_note["quote"]) == 2000
        assert len(long_note["note"]) == 4000
        # JSON持久化往返
        reread = list_notes("x")
        assert len(reread) == 1 and reread[0]["id"] == long_note["id"]
        print("  空值/负页码/长文本截断/持久化往返 正确")

        print("\n模块3自测全部通过")
    finally:
        # 还原现场
        if os.path.exists(NOTES_PATH):
            os.remove(NOTES_PATH)
        if backup and os.path.exists(backup):
            shutil.copy(backup, NOTES_PATH)
            os.remove(backup)
