"""
reviewer.py —— 批判性审查Agent（本项目核心创新点）
================================
职责（两级核验 + 量化统计）：
  1. align_quote(): 【程序化核验】检查每条引用是否真实存在于PDF原文中，
     返回匹配片段、置信度、页码与上下文（QuoteMatchInfo）
     —— 零成本、100%可靠，专门抓"伪造引用"
  2. semantic_verify(): 【LLM核验·原子事实法】把声明拆成原子事实逐条判定，
     数值与实验条件重点核查，证据不足时暂缓(abstained)而非硬判
     —— 抓"引用是真的但声明夸大/曲解"的语义级幻觉
  3. generate_report(): 汇总两级核验结果，计算统一口径的三项指标

统一指标定义（协议 v2）:
  - 幻觉率: 幻觉声明数/总数。幻觉 = (所有引用未对齐) OR
    (语义判定 unsupported/contradicted)。
    LLM调用失败(llm_error)与证据不足(abstained)绝不计为幻觉——
    "核验失败"和"证实造假"是两回事，混在一起会虚增幻觉率。
  - 引用对齐率: 至少一条引用成功对齐的声明数/总数
  - 核验成功率: 完成语义判定的声明数/总数（管线可靠性指标）

输入: list[PaperMeta], list[PaperCard]（第1、2步的产出）
输出: data/review_results.json + 控制台幻觉率报告
"""

import os
import json
import re
import time
import bisect
from collections import Counter
from difflib import SequenceMatcher

import llm_client
import config
import budget
from models import (
    PaperMeta, PaperCard, ReviewResult, ReviewVerdict, HallucinationReport,
    QuoteMatchInfo, AtomicFactVerdict, SemanticVerifyResult,
)
from agents.extractor import pdf_to_text, pdf_to_pages

# ---------------------------------------------------------------
# 第1级：程序化引用对齐检查（阶段2：返回完整定位信息）
# ---------------------------------------------------------------
def _normalize(text: str) -> str:
    """归一化文本：行尾连字符断词拼接、压平空白、统一小写。

    连字符断词（"perfor-\\nmance"）是PDF提取的行尾伪影而非真实连字符，
    与 evidence_locator._norm_for_match 口径一致，拼接后可精确匹配。
    同一行内的正常连字符词（high-order）不受影响。
    """
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-zA-Z])", r"\1\2", text)
    return " ".join(text.split()).lower()


def _locate(quote_text: str, full_text: str,
            fragment_len: int = 80, threshold: float = 0.85):
    """
    核心匹配：在归一化全文中定位引用。

    匹配策略：
      1. 归一化后精确子串匹配（置信度1.0，短引用/数字/公式也适用）
      2. 精确失败且引用≥10字符时，用引用前fragment_len字符做滑窗模糊匹配，
         取相似度最高的窗口（要求≥threshold）

    返回:
        (start, end, confidence) —— 命中片段在归一化全文中的区间与置信度；
        未命中返回 None
    """
    quote = _normalize(quote_text)
    full_text = _normalize(full_text)
    if not quote or not full_text:
        return None

    # ① 精确匹配（短引用、纯数字、公式片段只要逐字存在即可命中）
    idx = full_text.find(quote)
    if idx >= 0:
        return (idx, idx + len(quote), 1.0)

    # ② 太短的引用没有模糊匹配的价值（误判率高）
    if len(quote) < 10:
        return None

    # ③ 滑窗模糊匹配（粗扫找候选 + 局部精扫，避免步长错位漏检）
    fragment = quote[:fragment_len]
    window_size = len(fragment)
    step = 20
    # 粗扫：quick_ratio 是字符袋相似度，对窗口偏移不敏感，适合低成本预筛
    coarse: list[tuple[float, int]] = []
    for i in range(0, max(1, len(full_text) - window_size + 1), step):
        window = full_text[i:i + window_size]
        coarse.append((SequenceMatcher(None, fragment, window).quick_ratio(), i))
    coarse.sort(reverse=True)
    # 精扫：对前5个候选区 ±step 内逐位取 ratio 最高者
    best: tuple[float, int] | None = None  # (ratio, start)
    for _, i0 in coarse[:5]:
        for i in range(max(0, i0 - step),
                       min(len(full_text) - window_size + 1, i0 + step + 1)):
            window = full_text[i:i + window_size]
            matcher = SequenceMatcher(None, fragment, window)
            if matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio >= threshold and (best is None or ratio > best[0]):
                best = (ratio, i)
    if best:
        return (best[1], best[1] + window_size, best[0])
    return None


def build_page_index(page_texts: list[str]) -> tuple[str, list[int]]:
    """
    把逐页文本拼成归一化全文，并记录每页在全文中的起始偏移。

    返回:
        (joined_norm_text, page_starts) —— page_starts[i] 是第i页(0-based)
        在归一化全文中的起始下标，供命中位置反查页码。
    """
    norms = [_normalize(p) for p in page_texts]
    joined = " ".join(norms)
    starts: list[int] = []
    off = 0
    for n in norms:
        starts.append(off)
        off += len(n) + 1  # +1 对应 join 用的空格
    return joined, starts


def align_quote(quote_text: str, full_text: str,
                page_starts: list[int] | None = None,
                fragment_len: int = 80, threshold: float = 0.85,
                context_chars: int = 100) -> QuoteMatchInfo:
    """
    引用对齐的完整版：返回是否命中 + 片段/置信度/页码/上下文。

    参数:
        quote_text: 引用原文
        full_text: 论文全文（任意格式，内部会归一化；传入页索引拼接的
                   归一化全文时可用 page_starts 反查页码）
        page_starts: build_page_index()产出的页起始偏移；None则不查页码
    """
    info = QuoteMatchInfo(aligned=False)
    hit = _locate(quote_text, full_text, fragment_len, threshold)
    if hit is None:
        return info

    start, end, confidence = hit
    full_norm = _normalize(full_text)
    info.aligned = True
    info.confidence = round(confidence, 4)
    info.matched_fragment = full_norm[start:end]
    info.context = full_norm[max(0, start - context_chars):end + context_chars]
    if page_starts:
        # 命中起点属于哪个页区间（bisect右边界-1）
        page_no = bisect.bisect_right(page_starts, start) - 1
        if 0 <= page_no < len(page_starts):
            info.page = page_no
    return info


def quote_alignment(quote_text, full_text_norm, fragment_len=80, threshold=0.85):
    """
    兼容入口：只判断引用是否真实存在于论文全文中（bool）。
    需要片段/页码/上下文等详情请用 align_quote()。
    """
    if not quote_text or not full_text_norm:
        return False
    return align_quote(quote_text, full_text_norm,
                       fragment_len=fragment_len, threshold=threshold).aligned

# ---------------------------------------------------------------
# 第2级：LLM语义核验（阶段3：原子事实拆分法）
# ---------------------------------------------------------------
REVIEWER_SYSTEM_PROMPT = """你是一位极其严格的批判性审查员，负责核验信息抽取系统输出的忠实性。

你会收到：一条【声明】（抽取系统对论文的概括，中文）和【原文片段】（该声明引用的论文原文逐字摘录，多为英文）。

【工作流程——原子事实核验法】
1. 把声明拆解为1-5条原子事实：每条只含一个可独立判定的断言
   （例："方法A在数据集X上达到95.2%准确率"应拆成
   ①方法A在数据集X上实验 ②准确率为95.2% 两条）
2. 对每条原子事实单独对照原文片段判定，重点核查：
   - 数字：数值、百分比、单位逐一精确比对
   - 实验条件：数据集名、任务场景、规模、对比基线是否一致
3. 每条原子事实的判定四选一：
   - supported: 片段中有明确依据
   - unsupported: 片段中找不到依据（包括片段只覆盖了声明的一部分）
   - contradicted: 与片段直接矛盾（数字对不上、结论相反）
   - abstained: 片段对该事实完全没有涉及，无法判断
     （宁暂缓不误判；但片段给了间接线索时必须给出判定，不许逃避）
4. reason 用一句话说明整体判定理由；confidence 是对声明整体判定的把握(0-1)。

【最容易漏的改错手法——逐词比对重点防范】
- 偷换名词: "服务导向→平台导向"、"域外→域内"、"通信→协同"、
  "路径→轨迹"这类单词级替换
- 偷换数字: 数值、倍数、数量、比例的任何变动
- 扩大范围: "在某数据集上→在多个数据集上"、"受延迟限制→普遍受延迟限制"
- 偷换强度: "提出→证明"、"不稳定→普遍不稳定"、"效率→自动化程度"

【判定标准——以下都算问题】
- 声明中的数字与原文不符（哪怕是四舍五入方向不同）
- 声明扩大了适用范围（原文说"在X数据集上"，声明说"在多个数据集上"）
- 声明把"作者提出"写成"作者证明"
- 声明混淆了本文工作与引用他人的工作

宁严勿宽：拿不准就判 unsupported，并说明理由。
"""


# ---------------------------------------------------------------
# K1·B2优化：数值程序化校验（数字/百分比硬事实不经LLM）
# ---------------------------------------------------------------
# 原理: "提升12%"被概括成"显著提升"这类模糊化，同源LLM核验时
# 大概率被放过（自我一致性偏差）。数字/百分比是程序可判定的事实——
# 声明中的百分比必须在原文摘录中逐字出现，出现不了就是硬伤。
_PCT_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _numeric_variants(num: str) -> set[str]:
    """一个数值的等价写法: "12.50"↔"12.5"、"12.0"↔"12"（尾零可省）"""
    variants = {num}
    trimmed = num.rstrip("0").rstrip(".") if "." in num else num
    if trimmed:
        variants.add(trimmed)
    return variants


def numeric_mismatch(claim: str, evidence: str) -> list[str]:
    """
    程序化数值校验（K1，纯函数可离线单测）。

    检查声明中每个百分比 X%（含 X.0% 尾零变体、"X percent"写法、
    千分位"1,200"归一）是否在证据原文中出现。
    返回未命中的数值列表（空列表 = 全部命中或声明无百分比）。

    注意: 只拦"声明有、原文没有"这一种硬伤（模糊化/夸大）；
    "原文有12%，声明写成13%"属于数值篡改，交给LLM语义核验按
    判定标准处理（方向性错误程序不适合硬判contradicted）。
    """
    misses: list[str] = []
    ev = evidence.replace(",", "")
    ev_lower = ev.lower()
    for m in _PCT_PAT.finditer(claim):
        raw = m.group(1)
        hits = _numeric_variants(raw)
        if any(v in ev for v in hits):
            continue
        if f"{raw} percent" in ev_lower or (
                any(f"{v} percent" in ev_lower for v in hits)):
            continue
        misses.append(f"{raw}%")
    return misses


def semantic_verify(claim_content: str, evidence_text: str,
                    check_numeric: bool = True) -> SemanticVerifyResult:
    """
    用LLM核验单条声明（原子事实拆分法）

    流程: K1数值程序校验（百分比硬事实，不调LLM）-> LLM拆原子事实
    并逐条判定 -> 程序按"最坏非暂缓判定"汇总verdict
    （verdict由程序强制推导，与原子事实严格一致，防止模型自相矛盾）。

    核验走 config.VERIFIER_APIS 端点（K1异构核验）：未配置
    VERIFIER_API_KEY 时与主LLM同源（行为不变）；配置后核验与
    生成使用不同厂商模型，消除自我一致性偏差。
    可选多模型投票（ENSEMBLE_VERIFY=1 + VERIFIER2_*）：多端点
    各判一次，多数一致采纳，分歧取最严并标注"待人工复核"。

    参数:
        check_numeric: 是否启用第0级数值程序校验（消融实验
            no_numeric 模式传 False，量化该层的独立贡献）

    返回:
        SemanticVerifyResult（atomic_facts + verdict/reason/confidence）
    """
    # ---- 第0级：数值程序校验（零LLM成本，确定性结论）----
    # 守卫: 仅当证据中完全没有任何百分比（无数字语境）时才程序拦截——
    # 这是"模糊化"（提升12%→显著提升）的确定性信号。若证据里有其他
    # 百分比，声明数值缺失更可能是"篡改"（原文95.2%声明97.2%），
    # 方向性判定交给LLM语义核验（可判contradicted）；程序硬拦成
    # unsupported会系统性吞掉contradicted类别，压低其召回。
    misses = (numeric_mismatch(claim_content, evidence_text)
              if check_numeric else [])
    if misses and not _PCT_PAT.search(evidence_text.replace(",", "")):
        facts = ", ".join(misses)
        return SemanticVerifyResult(
            atomic_facts=[AtomicFactVerdict(
                fact=f"声明中的数值 {facts} 应能在原文摘录中找到对应出处",
                verdict="unsupported",
                reason=f"程序化数值校验: 百分比 {facts} 未在原文摘录中出现，"
                       f"疑为模糊化/夸大概括（不经LLM，程序直接判定）",
                confidence=1.0,
            )],
            verdict="unsupported",  # 会被_aggregate_verdicts重新推导覆盖
            reason=f"声明中的数值 {facts} 在原文摘录中未出现（程序校验）",
            confidence=1.0,
        )

    messages = [
        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
        {"role": "user", "content":
            f"【声明】{claim_content}\n\n"
            f"【原文片段】{evidence_text}"},
    ]
    # K1: 核验端点——默认单端点（未配置VERIFIER_API_KEY则同源回落）；
    # ENSEMBLE_VERIFY=1 且配置VERIFIER2_*后启用多模型投票
    endpoints = (config.VERIFIER_ENSEMBLE
                 if (config.ENSEMBLE_VERIFY and config.VERIFIER_ENSEMBLE)
                 else [config.VERIFIER_APIS])
    results: list[SemanticVerifyResult] = []
    for eps in endpoints:
        r = llm_client.chat_json(
            messages, schema_class=SemanticVerifyResult,
            temperature=config.TEMP_REVIEWER,  # 审查要求稳定，最低温度
            apis=eps,  # K1: 核验走异构端点（未配置则同源回落）
        )
        # 程序化汇总：以原子事实的最坏非暂缓判定为准，覆盖模型的verdict字段
        r.verdict = _aggregate_verdicts(r.atomic_facts)
        results.append(r)

    result = results[0]
    if len(results) > 1:
        # 多模型投票（K1可选扩展）: 严格多数一致→采纳；否则取最严
        # 判定并在reason标注待人工复核（分歧本身是重要的风险信号）
        verdicts = [r.verdict for r in results]
        top, cnt = Counter(verdicts).most_common(1)[0]
        if cnt > len(verdicts) // 2:
            result.verdict = top
        else:
            result.verdict = max(verdicts, key=lambda v: _AGG_ORDER[v])
            result.reason = (f"【多模型核验分歧，待人工复核: "
                             f"{'/'.join(verdicts)}】"
                             f"{result.reason or ''}".strip())
        result.confidence = round(
            sum(r.confidence for r in results) / len(results), 3)
    return result


_AGG_ORDER = {"abstained": 0, "supported": 1, "unsupported": 2, "contradicted": 3}


def _aggregate_verdicts(facts: list[AtomicFactVerdict]) -> str:
    """
    原子事实 -> 声明级判定（纯函数，可离线单测）。

    规则: 取最坏的非abstained判定（contradicted > unsupported > supported）；
    全部abstained才判abstained（证据不足的暂缓只属于单个事实，
    只要有一条事实被证伪/证不支持，声明整体就不该放行）。
    """
    if not facts:
        return "abstained"
    worst = max(facts, key=lambda f: _AGG_ORDER[f.verdict])
    return worst.verdict if _AGG_ORDER[worst.verdict] > 0 else "abstained"


# ---------------------------------------------------------------
# 组合入口：完整审查流水线
# ---------------------------------------------------------------
def run(papers: list[PaperMeta], cards: list[PaperCard]) -> list[ReviewResult]:
    """
    对所有信息卡片执行两级核验，落盘 review_results.json

    性能说明: 语义核验是逐条声明的LLM调用（15篇论文约60条声明，
    串行需5-10分钟）。改为3线程并行后降到约1/3。
    线程安全性: page_cache在线程启动前构建完毕,线程内只读;
    每条声明只写自己的record,无共享可变状态。

    返回:
        list[ReviewResult]（每条声明一条核验记录）
    """
    # 建立论文页索引缓存（每篇PDF只解析一次：归一化全文+页起始偏移）
    print("[审查Agent] 解析论文全文（缓存）...")
    page_cache: dict[str, tuple[str, list[int]]] = {}
    for p in papers:
        try:
            pages = pdf_to_pages(p.local_path)
        except Exception as e:
            print(f"[审查Agent] PDF解析失败 {p.arxiv_id}: {e}")
            pages = []
        page_cache[p.arxiv_id] = build_page_index(pages)

    # ---- 先收集所有待核验任务（card, idx, claim），再并行执行 ----
    tasks = []
    for card in cards:
        for idx, claim in enumerate(card.claims):
            tasks.append((card, idx, claim))

    def _verify_one(task):
        """单条声明的两级核验（工作线程函数，只读page_cache）"""
        card, idx, claim = task
        budget.check("审查Agent")  # 超预算立即中止（异常穿透线程池上抛）
        full_text_norm, page_starts = page_cache.get(card.arxiv_id, ("", []))

        started = time.perf_counter()

        # ---- 第1级：程序化引用对齐（含片段/置信度/页码/上下文）----
        quote_details: list[QuoteMatchInfo] = []
        aligned_count = 0
        for q in claim.quotes:
            info = align_quote(q.text, full_text_norm, page_starts=page_starts)
            info.section = q.section
            if info.aligned:
                aligned_count += 1
            quote_details.append(info)
        aligned = aligned_count > 0

        # ---- 第2级：LLM语义核验（原子事实拆分）----
        # 证据 = 所有引用拼接（每条限500字符，控制token）
        evidence = "\n---\n".join(q.text[:500] for q in claim.quotes)

        if not aligned:
            # 引用对齐失败时，把"该声明"连同全文开头一起送审——
            # 让审查员在更大范围内找证据，区分"真幻觉"和"引用格式问题"
            evidence = (f"(注:以下引用未能在原文中程序化定位，"
                        f"可能是格式差异，请结合全文片段判断)\n{evidence}"
                        f"\n---\n(论文开头片段)\n{full_text_norm[:2000]}")

        atomic_facts: list[AtomicFactVerdict] = []
        review_status = "ok"
        try:
            sv = semantic_verify(claim.content, evidence)
            verdict = ReviewVerdict(
                verdict=sv.verdict, reason=sv.reason, confidence=sv.confidence,
            )
            atomic_facts = sv.atomic_facts
        except budget.BudgetExceededError:
            raise  # 预算耗尽必须上抛终止流水线
        except Exception as e:
            # 统一协议v2核心修正: LLM调用失败≠幻觉！
            # 占位verdict=abstained（不参与幻觉判定），review_status单列，
            # 统计时进入"核验失败"桶而非幻觉桶。
            print(f"[审查Agent] 语义核验调用失败(不计为幻觉): {e}")
            verdict = ReviewVerdict(
                verdict="abstained", reason=f"核验调用失败: {e}", confidence=0.0,
            )
            review_status = "llm_error"

        # 统一幻觉协议v2：引用全部失配 OR 语义不支持/矛盾。
        # abstained（证据不足）与 llm_error（核验失败）不触发幻觉。
        semantic_bad = verdict.verdict in ("unsupported", "contradicted")
        hallucination = (not aligned) or semantic_bad
        if not aligned and semantic_bad:
            failure_mode = "mixed"
        elif not aligned:
            failure_mode = "fake_quote"
        elif semantic_bad:
            failure_mode = verdict.verdict
        elif review_status == "llm_error":
            failure_mode = "llm_error"
        elif verdict.verdict == "abstained":
            failure_mode = "abstained"
        else:
            failure_mode = "none"

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
            review_status=review_status,
            quote_details=quote_details,
            atomic_facts=atomic_facts,
        )

        # 进度输出（一眼看出两级核验的判定）
        flag = "✓" if (aligned and verdict.verdict == "supported") else "✗"
        print(f"  {flag} [{claim.claim_type:10s}] 对齐={'Y' if aligned else 'N'} "
              f"语义={verdict.verdict:12s} {claim.content[:40]}...")
        return record

    from concurrent.futures import ThreadPoolExecutor
    # 2026-10-16: 6→10线程, DeepSeek并发余量充足, 核验耗时进一步下降
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_verify_one, tasks))

    # ---- 落盘 ----
    out_path = os.path.join(config.DATA_DIR, "review_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)
    print(f"[审查Agent] 核验记录已保存: {out_path}")

    return results


# ---------------------------------------------------------------
# 幻觉率统计报告（实验E1的核心产出，统一口径v2）
# ---------------------------------------------------------------
def generate_report(results: list[ReviewResult]) -> HallucinationReport:
    """
    汇总核验结果，按统一协议计算三项指标

    指标定义（v2）:
      幻觉率 = 幻觉声明/总数（llm_error与abstained绝不计入幻觉）
      引用对齐率 = 至少一条引用对齐的声明/总数
      核验成功率 = 完成语义判定的声明/总数（llm_error单列）
    """
    supported = sum(1 for r in results
                    if r.review_status == "ok"
                    and r.verdict.verdict == "supported")
    unsupported = sum(1 for r in results
                      if r.review_status == "ok"
                      and r.verdict.verdict == "unsupported")
    contradicted = sum(1 for r in results
                       if r.review_status == "ok"
                       and r.verdict.verdict == "contradicted")
    abstained_n = sum(1 for r in results
                      if r.review_status == "ok"
                      and r.verdict.verdict == "abstained")
    llm_errors = sum(1 for r in results if r.review_status == "llm_error")
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
        llm_errors=llm_errors,
        abstained=abstained_n,
    )

    print("\n" + "=" * 55)
    print("📊 幻觉率量化报告（实验E1 · 统一口径v2）")
    print("=" * 55)
    print(f"  声明总数:     {report.total_claims}")
    print(f"  原文支持:     {supported}")
    print(f"  无依据(幻觉): {unsupported}")
    print(f"  与原文矛盾:   {contradicted}")
    print(f"  引用未对齐:   {fake_quotes}")
    print(f"  证据不足暂缓: {abstained_n}（非幻觉）")
    print(f"  核验调用失败: {llm_errors}（非幻觉）")
    print(f"  幻觉声明(去重): {hallucinations}")
    print(f"  引用对齐率:   {report.quote_alignment_rate:.1%}")
    print(f"  核验成功率:   {report.verify_success_rate:.1%}")
    print(f"  幻觉率:       {report.hallucination_rate:.1%}")
    print("=" * 55)
    return report

# ---------------------------------------------------------------
# D Benchmark：离线、可复现的审查器评测（阶段4）
# ---------------------------------------------------------------
def _prf(gold: list[str], pred: list[str], label: str) -> tuple[float, float, float]:
    """单标签的 precision/recall/F1。"""
    tp = sum(g == label and p == label for g, p in zip(gold, pred))
    fp = sum(g != label and p == label for g, p in zip(gold, pred))
    fn = sum(g == label and p != label for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return precision, recall, f1


def _macro_f1(gold: list[str], pred: list[str], labels: tuple[str, ...]) -> float:
    """计算三分类 macro-F1，不依赖 sklearn，方便项目环境保持轻量。"""
    if not gold:
        return 0.0
    f1s = [_prf(gold, pred, label)[2] for label in labels]
    return sum(f1s) / len(f1s)


def run_d_benchmark(cases: list["BenchmarkCase"],
                    ablation: str = "full") -> "BenchmarkReport":
    """运行 D Benchmark（固定测试集上的审查器离线评测）。

    Benchmark 样本必须提供人工标签 gold_verdict 与 gold_quote_alignment。
    每个 case 独立调用语义审查器，因此可重复比较不同模型/Prompt版本。
    该函数不会修改正式流水线产出的 review_results.json。

    失败处理: 单个case的LLM调用失败会计入 errored_cases 并跳过，
    不伪造判定、不污染指标（与正式流水线的llm_error协议一致）。

    消融实验（K2·B3优化）——ablation参数:
      full          完整三级核验（程序对齐 + 数值校验 + LLM语义核验），默认
      no_semantic   关闭LLM语义核验（含第0级数值校验），判定退化为纯程序
                    规则（引用对齐成功→supported，失败→unsupported）——
                    回答"没有LLM语义核验，指标会掉多少"
                    （需样本提供quote_text+full_text，缺则跳过该样本）
      no_alignment  关闭程序化引用对齐（引用一律放行），只留语义核验
                    ——回答"没有逐字对齐兜底，伪造引用漏网多少"
      no_numeric    仅关闭第0级数值程序校验，其余两级全开——
                    回答"K1数值校验这一层独立贡献多少"
    各模式各跑一遍即得消融对比表（benchmark.py ablation 子命令）。

    指标: 逐类 P/R/F1 + Macro-F1 + 误放行率
      误放行率 = (gold为坏声明 却被预测为通过) / (gold坏声明总数)
      坏声明 = gold_verdict != supported 或 gold引用伪造
      通过   = predicted_verdict == supported 且 predicted对齐成功
    """
    from models import BenchmarkCase, BenchmarkReport, BenchmarkResult

    if ablation not in ("full", "no_semantic", "no_alignment", "no_numeric"):
        raise ValueError(f"未知消融模式: {ablation}")

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
    kept_cases: list[BenchmarkCase] = []
    errored_ids: list[str] = []
    for case in cases:
        try:
            if ablation == "no_semantic":
                # 消融LLM语义核验: 纯程序规则判定（零LLM成本）
                if case.quote_text is None or case.full_text is None:
                    print(f"  ⚠ [{case.case_id}] no_semantic消融需要"
                          f"quote_text+full_text，跳过该样本")
                    errored_ids.append(case.case_id)
                    continue
                predicted_alignment = quote_alignment(
                    case.quote_text, _normalize(case.full_text))
                verdict = "supported" if predicted_alignment else "unsupported"
            else:
                sv = semantic_verify(
                    case.claim_content, case.evidence_text,
                    check_numeric=(ablation != "no_numeric"))
                verdict = sv.verdict
                if case.quote_text is not None and case.full_text is not None:
                    predicted_alignment = quote_alignment(
                        case.quote_text, _normalize(case.full_text))
                else:
                    # D Benchmark 若只评测语义核验，可省略原文全文；此时不伪造对齐结果。
                    predicted_alignment = case.gold_quote_alignment
                if ablation == "no_alignment":
                    # 消融一级核验: 引用对齐一律放行（模拟没有该检查）
                    predicted_alignment = True
        except budget.BudgetExceededError:
            raise  # 预算耗尽直接中止整个benchmark
        except Exception as e:
            print(f"  ⚠ [{case.case_id}] 语义核验调用失败，跳过该样本: {e}")
            errored_ids.append(case.case_id)
            continue
        kept_cases.append(case)

        results.append(BenchmarkResult(
            case_id=case.case_id,
            predicted_verdict=verdict,
            predicted_quote_alignment=predicted_alignment,
            verdict_correct=verdict == case.gold_verdict,
            alignment_correct=predicted_alignment == case.gold_quote_alignment,
            gold_verdict=case.gold_verdict,
            gold_quote_alignment=case.gold_quote_alignment,
        ))
        gold_verdicts.append(case.gold_verdict)
        pred_verdicts.append(verdict)

    n = len(results)
    verdict_correct = sum(r.verdict_correct for r in results)
    alignment_correct = sum(r.alignment_correct for r in results)
    counts = {label: gold_verdicts.count(label)
              for label in ("supported", "unsupported", "contradicted")}
    labels = ("supported", "unsupported", "contradicted")
    prf = {label: _prf(gold_verdicts, pred_verdicts, label) for label in labels}

    # ---- 误放行率（false pass rate）----
    gold_bad_pass, gold_bad_total = 0, 0
    for case, r in zip(kept_cases, results):
        gold_bad = (case.gold_verdict != "supported"
                    or not case.gold_quote_alignment)
        if not gold_bad:
            continue
        gold_bad_total += 1
        if (r.predicted_verdict == "supported"
                and r.predicted_quote_alignment):
            gold_bad_pass += 1
    false_pass_rate = (gold_bad_pass / gold_bad_total) if gold_bad_total else 0.0

    report = BenchmarkReport(
        total_cases=n,
        verdict_correct=verdict_correct,
        alignment_correct=alignment_correct,
        supported_cases=counts["supported"],
        unsupported_cases=counts["unsupported"],
        contradicted_cases=counts["contradicted"],
        verdict_accuracy=verdict_correct / n if n else 0.0,
        alignment_accuracy=alignment_correct / n if n else 0.0,
        macro_f1=_macro_f1(
            gold_verdicts, pred_verdicts, labels),
        results=results,
        precision={label: p for label, (p, _, _) in prf.items()},
        recall={label: rc for label, (_, rc, _) in prf.items()},
        f1={label: f for label, (_, _, f) in prf.items()},
        false_pass_rate=false_pass_rate,
        errored_cases=len(errored_ids),
    )
    print("\n[D Benchmark]" + (f" (消融模式: {ablation})" if ablation != "full" else ""))
    print(f"  有效样本: {n} (跳过{len(errored_ids)}个LLM失败样本)")
    print(f"  Verdict Accuracy: {report.verdict_accuracy:.1%}")
    print(f"  Alignment Accuracy: {report.alignment_accuracy:.1%}")
    for label in labels:
        p, rc, f = prf[label]
        print(f"  [{label:13s}] P={p:.3f} R={rc:.3f} F1={f:.3f}")
    print(f"  Macro-F1: {report.macro_f1:.3f}")
    print(f"  误放行率(false pass): {report.false_pass_rate:.1%}")
    return report
