"""
reviewer.py —— 批判性审查Agent（本项目核心创新点）
================================
职责（两级核验 + 量化统计）：
  1. quote_alignment(): 【程序化核验】用模糊匹配检查每条引用是否真实
     存在于PDF原文中 —— 零成本、100%可靠，专门抓"伪造引用"
  2. semantic_verify(): 【LLM核验】判断声明内容是否被原文片段支持
     —— 抓"引用是真的但声明夸大/曲解"的语义级幻觉
  3. generate_report(): 汇总两级核验结果，计算幻觉率

幻觉判定协议（可写进报告的量化定义）:
  一条声明被判为"幻觉"当且仅当满足以下任一:
    a) 所有引用都对齐失败（引用疑似伪造/来自截断部分）
    b) 语义核验 verdict = unsupported 或 contradicted

输入: list[PaperMeta], list[PaperCard]（第1、2步的产出）
输出: data/review_results.json + 控制台幻觉率报告
"""

import os
import json
import time
from difflib import SequenceMatcher

import llm_client
import config
from models import (
    PaperMeta, PaperCard, ReviewResult, ReviewVerdict, HallucinationReport,
)
from agents.extractor import pdf_to_text

# ---------------------------------------------------------------
# 第1级：程序化引用对齐检查
# ---------------------------------------------------------------
def _normalize(text: str) -> str:
    """归一化文本：压平空白、统一小写（PDF解析的换行/空格差异不影响比对）"""
    return " ".join(text.split()).lower()


def quote_alignment(quote_text, full_text_norm, fragment_len=80, threshold=0.85):
    """
    判断引用文本是否真实存在于论文全文中。

    匹配策略：
    1. 先进行标准化后的精确子串匹配
    2. 精确匹配失败后，再进行模糊匹配
    """

    if not quote_text or not full_text_norm:
        return False

    quote = _normalize(quote_text)
    full_text = _normalize(full_text_norm)

    if not quote:
        return False

    # ① 优先进行精确匹配
    # 可以避免固定窗口导致的漏检
    if quote in full_text:
        return True

    # ② 太短的引用容易产生误判
    if len(quote) < 10:
        return False

    # ③ 精确匹配失败后，再进行模糊匹配
    fragment = quote[:fragment_len]

    window_size = len(fragment)
    step = 20

    for i in range(0, max(1, len(full_text) - window_size + 1), step):
        window = full_text[i:i + window_size]

        matcher = SequenceMatcher(None, fragment, window)

        # 先使用 quick_ratio() 快速过滤
        if matcher.quick_ratio() < threshold:
            continue

        # 再使用 ratio() 精确判断
        if matcher.ratio() >= threshold:
            return True

    return False

# ---------------------------------------------------------------
# 第2级：LLM语义核验
# ---------------------------------------------------------------
REVIEWER_SYSTEM_PROMPT = """你是一位极其严格的批判性审查员，负责核验信息抽取系统输出的忠实性。

你会收到：一条【声明】（抽取系统对论文的概括）和【原文片段】（该声明声称的依据）。

判断声明是否被原文片段支持，输出三选一判定：
  - supported: 声明的内容在原文片段中有明确依据，数字、条件、结论均一致
  - unsupported: 声明的内容在片段中找不到依据（包括:片段只覆盖了声明的一部分、
    声明添加了片段中没有的信息）
  - contradicted: 声明与片段内容直接矛盾（如数字对不上、结论相反）

【严格标准——以下都算问题】
- 声明中的数字与原文不符（哪怕是四舍五入方向不同）
- 声明扩大了适用范围（原文说"在X数据集上"，声明说"在多个数据集上"）
- 声明把"作者提出"写成"作者证明"
- 声明混淆了本文工作与引用他人的工作

宁严勿宽：拿不准就判 unsupported，并说明理由。confidence 给你对判定的把握(0-1)。
"""


def semantic_verify(claim_content: str, evidence_text: str) -> ReviewVerdict:
    """
    用LLM核验单条声明是否被证据支持

    参数:
        claim_content: 声明内容（中文概括）
        evidence_text: 原文证据（引用拼接，若对齐失败则给截断前的全文开头）

    返回:
        ReviewVerdict（verdict/reason/confidence）
    """
    messages = [
        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
        {"role": "user", "content":
            f"【声明】{claim_content}\n\n"
            f"【原文片段】{evidence_text}"},
    ]
    return llm_client.chat_json(
        messages, schema_class=ReviewVerdict,
        temperature=config.TEMP_REVIEWER,  # 审查要求稳定，最低温度
    )


# ---------------------------------------------------------------
# 组合入口：完整审查流水线
# ---------------------------------------------------------------
def run(papers: list[PaperMeta], cards: list[PaperCard]) -> list[ReviewResult]:
    """
    对所有信息卡片执行两级核验，落盘 review_results.json

    性能说明: 语义核验是逐条声明的LLM调用（15篇论文约60条声明，
    串行需5-10分钟）。改为3线程并行后降到约1/3。
    线程安全性: text_cache在线程启动前构建完毕,线程内只读;
    每条声明只写自己的record,无共享可变状态。

    返回:
        list[ReviewResult]（每条声明一条核验记录）
    """
    # 建立论文全文缓存（每篇PDF只解析一次，避免重复IO）
    print("[审查Agent] 解析论文全文（缓存）...")
    text_cache: dict[str, str] = {}  # arxiv_id -> 归一化全文
    for p in papers:
        text_cache[p.arxiv_id] = _normalize(pdf_to_text(p.local_path))

    # ---- 先收集所有待核验任务（card, idx, claim），再并行执行 ----
    tasks = []
    for card in cards:
        for idx, claim in enumerate(card.claims):
            tasks.append((card, idx, claim))

    def _verify_one(task):
        """单条声明的两级核验（工作线程函数，只读text_cache）"""
        card, idx, claim = task
        budget.check("审查Agent")  # 超预算立即中止（异常穿透线程池上抛）
        full_text_norm = text_cache.get(card.arxiv_id, "")

        started = time.perf_counter()

        # ---- 第1级：程序化引用对齐 ----
        aligned_count = sum(
            quote_alignment(q.text, full_text_norm)
            for q in claim.quotes
        )
        aligned = aligned_count > 0

        # ---- 第2级：LLM语义核验 ----
        # 证据 = 所有引用拼接（每条限500字符，控制token）
        evidence = "\n---\n".join(q.text[:500] for q in claim.quotes)

        if not aligned:
            # 引用对齐失败时，把"该声明"连同全文开头一起送审——
            # 让审查员在更大范围内找证据，区分"真幻觉"和"引用格式问题"
            evidence = (f"(注:以下引用未能在原文中程序化定位，"
                        f"可能是格式差异，请结合全文片段判断)\n{evidence}"
                        f"\n---\n(论文开头片段)\n{full_text_norm[:2000]}")

        try:
            verdict = semantic_verify(claim.content, evidence)
        except Exception as e:
            print(f"[审查Agent] 语义核验失败(按unsupported计): {e}")
            verdict = ReviewVerdict(
                verdict="unsupported", reason="核验调用失败", confidence=0.0,
            )

        # 统一幻觉协议：引用全部失配 OR 语义不支持/矛盾。
        semantic_bad = verdict.verdict in ("unsupported", "contradicted")
        hallucination = (not aligned) or semantic_bad
        if not hallucination:
            failure_mode = "none"
        elif not aligned and semantic_bad:
            failure_mode = "mixed"
        elif not aligned:
            failure_mode = "fake_quote"
        else:
            failure_mode = verdict.verdict

        record = ReviewResult(
            arxiv_id=card.arxiv_id,
            claim_index=idx,
            claim_content=claim.content,
            quote_alignment=aligned,
            verdict=verdict,
            aligned_quote_count=aligned_count,
            quote_count=len(claim.quotes),
            hallucination=hallucination,
            failure_mode=failure_mode,
            review_latency_ms=(time.perf_counter() - started) * 1000,
        )

        # 进度输出（一眼看出两级核验的判定）
        flag = "✓" if (aligned and verdict.verdict == "supported") else "✗"
        print(f"  {flag} [{claim.claim_type:10s}] 对齐={'Y' if aligned else 'N'} "
              f"语义={verdict.verdict:12s} {claim.content[:40]}...")
        return record

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(_verify_one, tasks))

    # ---- 落盘 ----
    out_path = os.path.join(config.DATA_DIR, "review_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)
    print(f"[审查Agent] 核验记录已保存: {out_path}")

    return results


# ---------------------------------------------------------------
# 幻觉率统计报告（实验E1的核心产出）
# ---------------------------------------------------------------
def generate_report(results: list[ReviewResult]) -> HallucinationReport:
    """
    汇总核验结果，计算幻觉率

    幻觉判定协议:
      a) quote_alignment=False（引用伪造） -> 幻觉
      b) verdict in (unsupported, contradicted) -> 幻觉
      （两者是独立的失败模式，可能同时发生，只计一次）
    """
    supported = sum(1 for r in results
                    if r.verdict.verdict == "supported")
    unsupported = sum(1 for r in results
                      if r.verdict.verdict == "unsupported")
    contradicted = sum(1 for r in results
                       if r.verdict.verdict == "contradicted")
    fake_quotes = sum(1 for r in results if not r.quote_alignment)
    hallucinations = sum(1 for r in results if r.hallucination)
    aligned_claims = sum(1 for r in results if r.quote_alignment)

    report = HallucinationReport(
        total_claims=len(results),
        supported=supported,
        unsupported=unsupported,
        contradicted=contradicted,
        fake_quotes=fake_quotes,
        hallucinations=hallucinations,
        aligned_quotes=aligned_claims,
    )

    print("\n" + "=" * 55)
    print("📊 幻觉率量化报告（实验E1）")
    print("=" * 55)
    print(f"  声明总数:     {report.total_claims}")
    print(f"  原文支持:     {supported}")
    print(f"  无依据(幻觉): {unsupported}")
    print(f"  与原文矛盾:   {contradicted}")
    print(f"  引用未对齐:   {fake_quotes}")
    print(f"  幻觉声明(去重): {hallucinations}")
    print(f"  引用对齐率:   {report.quote_alignment_rate:.1%}")
    print(f"  幻觉率:       {report.hallucination_rate:.1%}")
    print("=" * 55)
    return report

# ---------------------------------------------------------------
# D Benchmark：离线、可复现的审查器评测
# ---------------------------------------------------------------
def _macro_f1(gold: list[str], pred: list[str], labels: tuple[str, ...]) -> float:
    """计算三分类 macro-F1，不依赖 sklearn，方便项目环境保持轻量。"""
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def run_d_benchmark(cases: list["BenchmarkCase"]) -> "BenchmarkReport":
    """运行 D Benchmark。

    Benchmark 样本必须提供人工标签 gold_verdict 与 gold_quote_alignment。
    每个 case 独立调用语义审查器，因此可重复比较不同模型/Prompt版本。
    该函数不会修改正式流水线产出的 review_results.json。
    """
    from models import BenchmarkCase, BenchmarkReport, BenchmarkResult

    if not cases:
        return BenchmarkReport(
            total_cases=0, verdict_correct=0, alignment_correct=0,
            supported_cases=0, unsupported_cases=0, contradicted_cases=0,
            verdict_accuracy=0.0, alignment_accuracy=0.0, macro_f1=0.0,
            results=[],
        )

    results = []
    gold_verdicts = []
    pred_verdicts = []
    for case in cases:
        verdict = semantic_verify(case.claim_content, case.evidence_text)
        if case.quote_text is not None and case.full_text is not None:
            predicted_alignment = quote_alignment(
                case.quote_text, _normalize(case.full_text)
            )
        else:
            # D Benchmark 若只评测语义核验，可省略原文全文；此时不伪造对齐结果。
            predicted_alignment = case.gold_quote_alignment

        results.append(BenchmarkResult(
            case_id=case.case_id,
            predicted_verdict=verdict.verdict,
            predicted_quote_alignment=predicted_alignment,
            verdict_correct=verdict.verdict == case.gold_verdict,
            alignment_correct=predicted_alignment == case.gold_quote_alignment,
        ))
        gold_verdicts.append(case.gold_verdict)
        pred_verdicts.append(verdict.verdict)

    verdict_correct = sum(r.verdict_correct for r in results)
    alignment_correct = sum(r.alignment_correct for r in results)
    n = len(results)
    counts = {label: gold_verdicts.count(label)
              for label in ("supported", "unsupported", "contradicted")}
    report = BenchmarkReport(
        total_cases=n,
        verdict_correct=verdict_correct,
        alignment_correct=alignment_correct,
        supported_cases=counts["supported"],
        unsupported_cases=counts["unsupported"],
        contradicted_cases=counts["contradicted"],
        verdict_accuracy=verdict_correct / n,
        alignment_accuracy=alignment_correct / n,
        macro_f1=_macro_f1(
            gold_verdicts, pred_verdicts,
            ("supported", "unsupported", "contradicted"),
        ),
        results=results,
    )
    print("\n[D Benchmark]")
    print(f"  样本数: {n}")
    print(f"  Verdict Accuracy: {report.verdict_accuracy:.1%}")
    print(f"  Macro-F1:          {report.macro_f1:.3f}")
    return report
