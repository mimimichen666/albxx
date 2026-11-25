"""
layout_eval.py —— K3·A3 版面感知解析效果评测（十篇论文对比入册）
================================
验收要求的"抽10篇论文对比改造前后"：

  1. 版面识别统计: 每篇论文的双栏页占比、公式/乱码块剔除数
     （arXiv论文多为单栏，双栏页占比如实报告）
  2. 下游代理指标——引用对齐率: 用系统真实产出的声明quote
     （paper_cards.json，主目录+tasks/*），分别在
     LAYOUT_AWARE_PDF 开/关两种解析模式的全文上做程序化对齐，
     统计命中率差异——版面重组若有效，对齐率应提升或持平

结果打印入册表格 + 落盘 data/benchmark/layout_eval.json。

使用方法:
  python layout_eval.py            # 随机抽10篇（种子固定可复现）
  python layout_eval.py --n 20     # 指定篇数
  python layout_eval.py --seed 7   # 换一批论文
"""

import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime

import pymupdf

import config
from agents.extractor import pdf_to_pages, _is_formula_like
from agents.reviewer import quote_alignment, _normalize

OUT_PATH = os.path.join(config.DATA_DIR, "benchmark", "layout_eval.json")


def collect_quotes() -> dict[str, list[str]]:
    """收集真实声明quote: {arxiv_id: [quote_text, ...]}"""
    paths = [os.path.join(config.DATA_DIR, "paper_cards.json")]
    paths += sorted(glob.glob(os.path.join(
        config.DATA_DIR, "tasks", "*", "paper_cards.json")))
    quotes: dict[str, list[str]] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            cards = json.load(f)
        for card in cards:
            aid = card.get("arxiv_id", "")
            for c in card.get("claims", []):
                for q in c.get("quotes") or []:
                    t = (q.get("text") or "").strip()
                    if aid and len(t) >= 15:  # 过短quote无对齐意义
                        quotes.setdefault(aid, []).append(t)
    return quotes


def page_profile(page) -> tuple[bool, int]:
    """单页版面画像: (是否双栏页, 公式剔除块数)——与extractor
    内部判定逻辑一致（先公式过滤后分栏识别）"""
    blocks = [b for b in page.get_text("blocks")
              if b[6] == 0 and (b[4] or "").strip()]
    if config.FORMULA_FILTER:
        kept = [b for b in blocks if not _is_formula_like(b[4])]
        removed = len(blocks) - len(kept)
        if kept and len(kept) * 2 >= len(blocks):
            blocks = kept
    else:
        removed = 0
    pw = float(page.rect.width)
    left = right = full = 0
    for b in blocks:
        w = b[2] - b[0]
        if w > 0.75 * pw:
            full += 1
        elif b[2] <= 0.58 * pw:
            left += 1
        elif b[0] >= 0.42 * pw:
            right += 1
        else:
            full += 1
    return (left >= 2 and right >= 2 and full <= left + right), removed


def main() -> None:
    ap = argparse.ArgumentParser(description="K3版面感知解析评测")
    ap.add_argument("--n", type=int, default=10, help="论文篇数（默认10）")
    ap.add_argument("--seed", type=int, default=2026, help="抽样种子")
    args = ap.parse_args()

    pdfs = sorted(glob.glob(os.path.join(
        config.PROJECT_ROOT, "papers", "*.pdf")))
    if not pdfs:
        sys.exit("[layout_eval] papers/ 目录没有PDF")
    quotes = collect_quotes()
    random.Random(args.seed).shuffle(pdfs)
    # 定向包含有真实quote的论文（保证对齐率对比有数据），随机补齐
    pinned = [p for p in pdfs
              if os.path.splitext(os.path.basename(p))[0] in quotes]
    rest = [p for p in pdfs if p not in set(pinned)]
    random.Random(args.seed).shuffle(rest)
    picked = (pinned + rest)[:max(args.n, min(len(pinned), args.n))]
    print(f"[layout_eval] 抽样 {len(picked)} 篇 / 库存 {len(pdfs)} 篇 | "
          f"定向{len(min(pinned, picked))}篇有quote数据 | "
          f"对齐数据源: {sum(len(v) for v in quotes.values())} 条quote"
          f"（{len(quotes)}篇论文）")

    rows = []
    tot = {"pages": 0, "two_col": 0, "removed": 0,
           "q": 0, "hit_off": 0, "hit_on": 0}
    aware = config.LAYOUT_AWARE_PDF
    try:
        for i, path in enumerate(picked, 1):
            arxiv = os.path.splitext(os.path.basename(path))[0]

            # 双模式全文
            config.LAYOUT_AWARE_PDF = True
            pages_on = pdf_to_pages(path)
            config.LAYOUT_AWARE_PDF = False
            pages_off = pdf_to_pages(path)
            config.LAYOUT_AWARE_PDF = True

            # 版面统计
            doc = pymupdf.open(path)
            two_col = removed = 0
            for page in doc:
                tc, rm = page_profile(page)
                two_col += tc
                removed += rm
            n_pages = len(doc)
            doc.close()

            # 引用对齐率对比（该论文有quote时）
            qs = quotes.get(arxiv, [])
            text_on = _normalize("\n".join(pages_on))
            text_off = _normalize("\n".join(pages_off))
            hit_on = sum(quote_alignment(q, text_on) for q in qs)
            hit_off = sum(quote_alignment(q, text_off) for q in qs)

            rows.append({
                "arxiv_id": arxiv, "pages": n_pages,
                "two_col_pages": two_col, "formula_removed": removed,
                "quotes": len(qs), "hit_off": hit_off, "hit_on": hit_on,
            })
            for k, v in zip(tot, (n_pages, two_col, removed,
                                  len(qs), hit_off, hit_on)):
                tot[k] += v
            print(f"  [{i:>2}/{len(picked)}] {arxiv}: {n_pages}页 | "
                  f"双栏{two_col}页 | 剔公式{removed}块 | "
                  f"quote {hit_off}/{len(qs)}→{hit_on}/{len(qs)}")
    finally:
        config.LAYOUT_AWARE_PDF = aware  # 恢复全局开关

    # ---- 入册表格 ----
    print("\n" + "=" * 84)
    print("📐 K3·A3 版面感知解析评测（LAYOUT_AWARE_PDF 开/关对比）")
    print("=" * 84)
    print(f"{'论文':<14}{'页数':>5}{'双栏页':>7}{'剔公式块':>9}"
          f"{'quote数':>8}{'对齐(旧)':>9}{'对齐(新)':>9}")
    print("-" * 84)
    for r in rows:
        print(f"{r['arxiv_id']:<14}{r['pages']:>5}{r['two_col_pages']:>7}"
              f"{r['formula_removed']:>9}{r['quotes']:>8}"
              f"{r['hit_off']:>9}{r['hit_on']:>9}")
    print("-" * 84)
    print(f"{'合计':<14}{tot['pages']:>5}{tot['two_col']:>7}"
          f"{tot['removed']:>9}{tot['q']:>8}{tot['hit_off']:>9}"
          f"{tot['hit_on']:>9}")
    if tot["q"]:
        print(f"\n引用对齐率: 旧解析 {tot['hit_off']}/{tot['q']}="
              f"{tot['hit_off']/tot['q']:.1%} → 版面感知 "
              f"{tot['hit_on']}/{tot['q']}={tot['hit_on']/tot['q']:.1%}")
    print(f"双栏页占比: {tot['two_col']}/{tot['pages']}"
          f"（arXiv论文以单栏为主，双栏多为IEEE/ACM排版）")
    print("=" * 84)

    payload = {
        "_readme": "K3·A3版面感知解析效果评测（layout_eval.py自动生成）",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_size": len(rows), "seed": args.seed,
        "rows": rows, "totals": tot,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[layout_eval] 已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
