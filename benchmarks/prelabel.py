"""
prelabel.py —— K2·B3 标注底稿生成（半自动预标注）
================================
把"从零标 200 条"变成"复核 200 条"：从已有流水线产出
（data/paper_cards.json 与 data/tasks/*/paper_cards.json 的真实声明）
按协议配额（每篇论文最多 8 条，防单篇偏置）抽样生成标注底稿：

  benchmarks/labeling_docs/{arxiv_id}.txt   论文全文（版面感知解析）
  benchmarks/documents.json                 {arxiv_id: 文件名} 登记
  benchmarks/labeling_prelabel.csv          预填 claim/quote，gold 留空

人工只需复核每行的 gold_verdict / gold_quote_alignment 两列
（判定口径见 LABELING_PROTOCOL.md），大幅降低标注成本。
声明与quote均来自系统真实产出，天然覆盖"系统会犯的错"。

使用方法:
  python benchmarks/prelabel.py               # 抽60条底稿
  python benchmarks/prelabel.py --n 120       # 指定条数
  python benchmarks/prelabel.py --seed 7      # 换一批抽样
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from agents.extractor import pdf_to_pages  # noqa: E402

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BENCH_DIR, "labeling_docs")
DOCS_JSON = os.path.join(BENCH_DIR, "documents.json")
OUT_CSV = os.path.join(BENCH_DIR, "labeling_prelabel.csv")
MAX_PER_PAPER = 8  # LABELING_PROTOCOL.md 第2节: 每篇论文最多贡献8条


def collect_claims() -> list[dict]:
    """从全部paper_cards.json收集 {arxiv_id, claim, quote} 列表"""
    paths = [os.path.join(config.DATA_DIR, "paper_cards.json")]
    paths += sorted(glob.glob(
        os.path.join(config.DATA_DIR, "tasks", "*", "paper_cards.json")))
    items: list[dict] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            cards = json.load(f)
        for card in cards:
            for c in card.get("claims", []):
                quotes = c.get("quotes") or []
                items.append({
                    "arxiv_id": card.get("arxiv_id", ""),
                    "claim": (c.get("content") or "").strip(),
                    "quote": (quotes[0].get("text") or "").strip()
                    if quotes else "",
                })
    return [it for it in items if it["arxiv_id"] and it["claim"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="K2标注底稿生成")
    ap.add_argument("--n", type=int, default=60, help="底稿条数（默认60）")
    ap.add_argument("--seed", type=int, default=2026, help="抽样种子")
    args = ap.parse_args()

    items = collect_claims()
    if not items:
        sys.exit("[prelabel] 没有可用的声明数据——请先跑一次流水线")
    print(f"[prelabel] 收集到 {len(items)} 条真实声明")

    # 分组 + 每篇限8条
    by_paper: dict[str, list[dict]] = {}
    for it in items:
        by_paper.setdefault(it["arxiv_id"], []).append(it)
    quota: list[dict] = []
    for aid, group in by_paper.items():
        quota += group if len(group) <= MAX_PER_PAPER else \
            random.Random(args.seed).sample(group, MAX_PER_PAPER)
    n = min(args.n, len(quota))
    picked = random.Random(args.seed).sample(quota, n)
    print(f"[prelabel] {len(by_paper)} 篇论文（每篇≤{MAX_PER_PAPER}条）"
          f"，抽样 {n} 条")

    # 论文全文 → labeling_docs/ + documents.json 登记
    os.makedirs(DOCS_DIR, exist_ok=True)
    registry = {}
    if os.path.exists(DOCS_JSON):
        with open(DOCS_JSON, "r", encoding="utf-8-sig") as f:
            registry = json.load(f)
    used = sorted({it["arxiv_id"] for it in picked})
    for i, aid in enumerate(used, 1):
        pdf = os.path.join(config.PROJECT_ROOT, "papers", f"{aid}.pdf")
        fname = f"{aid}.txt"
        text_path = os.path.join(DOCS_DIR, fname)
        if not os.path.exists(text_path):
            if not os.path.exists(pdf):
                print(f"[warn] {aid}.pdf 不存在，跳过该篇全文")
                continue
            pages = pdf_to_pages(pdf)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(pages))
            print(f"  [{i}/{len(used)}] {aid} 全文已导出"
                  f"（{len(pages)}页）")
        registry.setdefault(aid, fname)
    with open(DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    # CSV底稿（gold列留空待人工复核）
    if os.path.exists(OUT_CSV):
        bak = OUT_CSV.replace(".csv", f"_bak_{datetime.now():%Y%m%d_%H%M%S}.csv")
        os.replace(OUT_CSV, bak)
        print(f"[prelabel] 旧底稿已备份: {os.path.basename(bak)}")
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "doc_key", "claim_content", "evidence_text",
                    "quote_text", "gold_verdict", "gold_quote_alignment",
                    "annotator", "notes"])
        for i, it in enumerate(picked, 1):
            w.writerow([
                f"p{i:03d}", it["arxiv_id"], it["claim"], "",
                it["quote"], "", "", "", "预标注底稿-请按协议复核gold两列",
            ])
    print(f"[prelabel] 底稿已生成: {OUT_CSV}")
    print(f"[prelabel] 进度: {n}/200+ 条（复核定成后运行 "
          f"build_cases.py 合成正式测试集）")
    print("[prelabel] 复核方法: 每行对照 labeling_docs 全文判定 "
          "gold_verdict(supported/unsupported/contradicted) 与 "
          "gold_quote_alignment(true/false)，填 annotator 缩写")


if __name__ == "__main__":
    main()
