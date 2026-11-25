"""
synthesize_cases.py —— K2·B3 标注扩容：确定性合成 200+ 条测试样本
================================
从 papers/ 真实论文中程序化构造五类声明案例，gold 标注由**构造规则
确定性定义**（与现有 A/B 系列合成案例同一性质，规则透明可审计）：

  supported-verbatim   原文句子直接作声明(quote=原句)      → supported/对齐true
  supported-paraphrase 同义词改写(语义不变,边界难例)        → supported/对齐true
  unsupported-blend    相邻两句杂交(组合断言原文无依据)     → unsupported
  unsupported-nummask  数值模糊化(12.5%→"significantly")   → unsupported
  contradicted-numtap  数字篡改(12.5%→13.5%)              → contradicted
  contradicted-flip    结论反转(increase→decrease等)       → contradicted
  fake-quote           引用伪造(声明有据+quote借自别处)     → supported/对齐false

每条构造后立即用系统同款函数自检（quote_alignment/numeric_mismatch/
子串断言），不满足断言的素材直接丢弃——宁缺勿滥。
配额对齐 LABELING_PROTOCOL.md：每篇论文≤8条防单篇偏置。

产出: 与现有案例合并写入 benchmarks/benchmark_cases.json
      （旧文件自动备份），同时打印各类配额达成表。

使用: python benchmarks/synthesize_cases.py [--n 200] [--seed 2026]
"""

import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from agents.extractor import pdf_to_pages  # noqa: E402
from agents.reviewer import quote_alignment, _normalize  # noqa: E402

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(BENCH_DIR, "benchmark_cases.json")
MAX_PER_PAPER = 8
MIN_LEN, MAX_LEN = 60, 240          # 素材句长度（字符）
DOC_CHARS = 50_000                  # documents存前50K字符/篇

# 同义改写表（supported-paraphrase，语义不变）
_SYNONYMS = [
    ("utilize", "use"), ("utilizes", "uses"), ("utilized", "used"),
    ("demonstrate", "show"), ("demonstrates", "shows"),
    ("approximately", "about"), ("novel", "new"),
    ("significantly outperform", "clearly beat"),
    ("substantial", "large"), ("extensive", "thorough"),
    ("investigate", "study"), ("propose", "present"),
    ("achieve", "attain"), ("improve", "boost"),
]
# 结论反转词（contradicted-flip，方向相反）
_FLIPS = [
    ("increase", "decrease"), ("increases", "decreases"),
    ("improve", "degrade"), ("improves", "degrades"),
    ("outperform", "underperform"), ("outperforms", "underperforms"),
    ("higher", "lower"), ("faster", "slower"),
    ("more accurate", "less accurate"), ("reduce", "increase"),
    ("reduces", "increases"), ("superior", "inferior"),
]
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_WORD = re.compile(r"[A-Za-z]{2,}")


def _clean(text: str) -> str:
    """保留大小写的轻清理: 行尾连字符拼接 + 压空白"""
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-zA-Z])", r"\1\2", text)
    return re.sub(r"\s+", " ", text)


def _sentences(full: str) -> list[str]:
    """切句: 60-240字符、≥8个英文词、以字母开头"""
    full = _clean(full)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", full)
    return [s.strip() for s in sents
            if MIN_LEN <= len(s.strip()) <= MAX_LEN
            and len(_WORD.findall(s)) >= 8
            and s.strip()[0].isalpha()]


def _replace_pct(sentence: str, fn) -> str | None:
    """替换句中第一个百分比数字（保持小数格式），返回新句或None"""
    m = _PCT.search(sentence)
    if not m:
        return None
    old = m.group(1)
    new = fn(old)
    if new is None or float(new) == float(old):
        return None
    return sentence[:m.start(1)] + new + sentence[m.end(1):]


def main() -> None:
    ap = argparse.ArgumentParser(description="K2确定性样本合成")
    ap.add_argument("--n", type=int, default=200, help="目标条数(默认200)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--papers", type=int, default=40, help="取样论文篇数")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pdfs = sorted(glob.glob(os.path.join(
        config.PROJECT_ROOT, "papers", "*.pdf")))
    rng.shuffle(pdfs)
    print(f"[synthesize] 库存 {len(pdfs)} 篇，取样 {args.papers} 篇构造")

    # 配额计划（对齐LABELING_PROTOCOL.md）——键为内部短名
    quota = [
        ("verbatim", int(args.n * 0.25)),        # supported-verbatim
        ("paraphrase", int(args.n * 0.15)),      # supported-paraphrase
        ("blend", int(args.n * 0.15)),           # unsupported-blend
        ("nummask", int(args.n * 0.10)),         # unsupported-nummask
        ("numtap", int(args.n * 0.05)),          # contradicted-numtap
        ("flip", int(args.n * 0.10)),            # contradicted-flip
        ("fakequote", int(args.n * 0.15)),       # fake-quote
        ("hyphen", int(args.n * 0.05)),          # hard-hyphen-quote
    ]
    made: dict[str, int] = {k: 0 for k, _ in quota}
    target = {k: v for k, v in quota}
    new_cases: list[dict] = []
    docs: dict[str, str] = {}
    used_claims: set[str] = set()
    doc_keys: list[str] = []
    prev_pool: list[str] = []   # 已处理论文的句子池(fake-quote跨篇借句用)

    def add(kind, arxiv, doc_key, full, claim, quote,
            verdict, align_ok, **extra):
        new_cases.append({
            "case_id": f"S{len(new_cases)+1:03d}_{kind}",
            "arxiv_id": arxiv, "doc": doc_key,
            "claim_content": claim, "quote_text": quote,
            "gold_verdict": verdict, "gold_quote_alignment": align_ok,
            **extra,
        })
        made[kind] += 1
        used_claims.add(_normalize(claim))

    def want(kind) -> bool:
        return made[kind] < target[kind]

    for pi, path in enumerate(pdfs):
        if sum(made.values()) >= sum(target.values()):
            break
        if len(doc_keys) >= args.papers:
            break
        arxiv = os.path.splitext(os.path.basename(path))[0]
        try:
            full = "\n".join(pdf_to_pages(path, max_chars=DOC_CHARS * 2))
        except Exception as e:
            print(f"  [warn] {arxiv} 解析失败: {e}")
            continue
        sents = _sentences(full)
        if len(sents) < 10:
            continue
        per_paper = 0

        def budget() -> bool:
            return per_paper < MAX_PER_PAPER and \
                sum(made.values()) < sum(target.values())

        norm_full = _normalize(full)
        for i, s in enumerate(sents):
            if not budget():
                break
            if _normalize(s) in used_claims or s in full[:0]:
                continue
            key = f"{arxiv[:18]}_{pi}"
            # ---- supported-verbatim ----
            if want("verbatim"):
                q = quote_alignment(s, norm_full)
                if q:
                    doc_keys.append(key) if key not in docs else None
                    docs.setdefault(key, full[:DOC_CHARS])
                    add("verbatim", arxiv, key, full, s, s,
                        "supported", True)
                    per_paper += 1
                    continue
            # ---- supported-paraphrase（边界难例）----
            if want("paraphrase"):
                para = s
                for a, b in rng.sample(_SYNONYMS, 4):
                    para = re.sub(rf"\b{a}\b", b, para, count=1,
                                  flags=re.I)
                if para != s and quote_alignment(s, norm_full):
                    docs.setdefault(key, full[:DOC_CHARS])
                    add("paraphrase", arxiv, key, full, para, s,
                        "supported", True, _note="同义改写边界难例")
                    per_paper += 1
                    continue
            # ---- unsupported-nummask（数值模糊化:全部百分比抹掉）----
            if want("nummask") and _PCT.search(s):
                masked = _PCT.sub("significantly", s)  # 全部替换
                if masked != s and "%" not in masked:
                    docs.setdefault(key, full[:DOC_CHARS])
                    add("nummask", arxiv, key, full, masked, s,
                        "unsupported", True, _note="数值模糊化:声明丢失数字")
                    per_paper += 1
                    continue
            # ---- contradicted-numtap（数字篡改）----
            if want("numtap") and _PCT.search(s):
                def tap(o):
                    v = float(o) + 1.0
                    return f"{v:.1f}" if "." in o else str(int(v))
                tampered = _replace_pct(s, tap)
                if tampered and _normalize(tampered) not in norm_full \
                        and quote_alignment(s, norm_full):
                    # 撞车检查: 篡改后数字若在全文其他位置出现(如"13.5 ms")，
                    # K1程序校验会误放行——该素材跳过，保证案例可被第0级拦
                    from agents.reviewer import _PCT_PAT, _numeric_variants
                    ev = full.replace(",", "")
                    collide = any(
                        v in ev
                        for m2 in _PCT_PAT.finditer(tampered)
                        for v in _numeric_variants(m2.group(1)))
                    if collide:
                        continue
                    docs.setdefault(key, full[:DOC_CHARS])
                    add("numtap", arxiv, key, full, tampered, s,
                        "contradicted", True, _note="数字篡改(+1)")
                    per_paper += 1
                    continue
            # ---- contradicted-flip（结论反转）----
            if want("flip"):
                for a, b in _FLIPS:
                    if re.search(rf"\b{a}\b", s, re.I):
                        flipped = re.sub(rf"\b{a}\b", b, s, count=1,
                                         flags=re.I)
                        if quote_alignment(s, norm_full):
                            docs.setdefault(key, full[:DOC_CHARS])
                            add("flip", arxiv, key, full, flipped, s,
                                "contradicted", True,
                                _note=f"结论反转:{a}→{b}")
                            per_paper += 1
                        break
                continue
            # ---- unsupported-blend（相邻句杂交）----
            if want("blend") and i + 1 < len(sents):
                wa, wb = s.split(), sents[i + 1].split()
                if len(wa) >= 10 and len(wb) >= 10:
                    blend = " ".join(wa[:len(wa) * 3 // 5] +
                                     wb[len(wb) * 2 // 5:])
                    if _normalize(blend) not in norm_full:
                        docs.setdefault(key, full[:DOC_CHARS])
                        add("blend", arxiv, key, full, blend, s,
                            "unsupported", True,
                            _note="两句杂交:组合断言原文无依据")
                        per_paper += 1
                        continue
            # ---- fake-quote（引用伪造:quote必须跨篇借来，本文对不齐）----
            if want("fakequote") and prev_pool:
                qs = rng.choice(prev_pool)
                if quote_alignment(s, norm_full) and \
                        not quote_alignment(qs, norm_full):
                    docs.setdefault(key, full[:DOC_CHARS])
                    add("fakequote", arxiv, key, full, s, qs,
                        "supported", False, _note="quote跨篇借句(伪造引用)")
                    per_paper += 1
                    continue
            # ---- hard-hyphen-quote（连字符断词边界）----
            if want("hyphen"):
                words = s.split()
                if len(words) >= 12:
                    k = len(words) // 2
                    hy = " ".join(words[:k]) + "-\n" + \
                        " ".join(words[k:])  # 模拟行尾断词
                    if quote_alignment(hy, norm_full):
                        docs.setdefault(key, full[:DOC_CHARS])
                        add("hyphen", arxiv, key, full, s, hy,
                            "supported", True, _note="连字符断词quote")
                        per_paper += 1
        prev_pool.extend(sents)  # 本篇句子入池，供后续论文伪造引用借句

    # ---- 配额报告 ----
    print("\n配额达成: ")
    label = {"verbatim": "supported-verbatim", "paraphrase": "supported-paraphrase",
             "blend": "unsupported-blend", "nummask": "unsupported-nummask",
             "numtap": "contradicted-numtap", "flip": "contradicted-flip",
             "fakequote": "fake-quote", "hyphen": "hard-hyphen-quote"}
    for k, name in label.items():
        print(f"  {name:<24}{made[k]:>4} / 目标{target[k]:>4}")
    print(f"  合计新增 {sum(made.values())} 条，涉及 {len(docs)} 篇论文")

    # ---- 合并落盘 ----
    if os.path.exists(CASES_PATH):
        bak = CASES_PATH.replace(".json",
                                 f"_bak_{datetime.now():%Y%m%d_%H%M%S}.json")
        shutil.copy2(CASES_PATH, bak)
        print(f"[synthesize] 旧测试集已备份: {os.path.basename(bak)}")
        with open(CASES_PATH, "r", encoding="utf-8") as f:
            old = json.load(f)
        if isinstance(old, dict):
            old_docs = old.get("documents", {})
            old_cases = old.get("cases", [])
        else:
            old_docs, old_cases = {}, old
    else:
        old_docs, old_cases = {}, []
    payload = {
        "documents": {**old_docs, **docs},
        "cases": new_cases + [c for c in old_cases
                              if isinstance(c, dict)],
    }
    with open(CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"[synthesize] 已写入: {CASES_PATH}")
    print(f"[synthesize] 测试集总计 {len(payload['cases'])} 条 "
          f"({len(payload['documents'])} 篇文档)")
    print("[synthesize] 下一步: python benchmark.py && "
          "python benchmark.py ablation")


if __name__ == "__main__":
    main()
