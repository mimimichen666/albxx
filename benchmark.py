"""
benchmark.py —— 固定 Benchmark 评测入口（阶段4）
================================
职责：
  1. load_cases(): 加载固定测试集 benchmarks/benchmark_cases.json
     （支持 {documents, cases} 结构：case 通过 doc 字段引用文档，
      加载时展开为 full_text/evidence_text；也兼容逐条内联的旧格式）
  2. run(): 对固定测试集运行审查器（run_d_benchmark），产出
     Precision/Recall/F1（逐类+Macro）、误放行率等指标并落盘；
     K2起自动附带 bootstrap 95%置信区间
  3. compare(): 比较两次评测报告，量化"修改前后"的指标变化
  4. ablation(): 消融实验（K2）——完整核验 / 仅程序对齐 / 仅语义核验
     三种模式各跑一遍，量化两级核验各自的贡献

使用方法：
  python benchmark.py                    # 运行当前审查器并保存报告（含置信区间）
  python benchmark.py compare old.json new.json   # 比较两份历史报告
  python benchmark.py ablation           # 消融实验：两级核验各自贡献多少

指标口径（与 agents/reviewer.py 统一）：
  - Precision/Recall/F1: 按 supported/unsupported/contradicted 逐类计算
  - Macro-F1: 三类F1平均
  - 误放行率: gold坏声明(判定!=supported 或 引用伪造)被预测放行的比例
  - LLM调用失败的样本计入 errored_cases，跳过不评测（不伪造判定）
  - 置信区间: 对逐条结果有放回重采样2000次取2.5%/97.5%分位（K2·B3），
    回答"幻觉率8%和10%在统计上是否有差别"
"""

import os
import json
import sys
import random
from datetime import datetime

import config
from agents.reviewer import run_d_benchmark
from models import BenchmarkCase

# 固定测试集目录（随仓库版本控制）
BENCH_DIR = os.path.join(config.PROJECT_ROOT, "benchmarks")
CASES_PATH = os.path.join(BENCH_DIR, "benchmark_cases.json")
# 评测报告输出目录（运行时产物，不入仓库）
OUT_DIR = os.path.join(config.DATA_DIR, "benchmark")
REPORT_PATH = os.path.join(OUT_DIR, "benchmark_report.json")

_LABELS = ("supported", "unsupported", "contradicted")


def load_cases(path: str = CASES_PATH) -> list[BenchmarkCase]:
    """
    加载固定测试集

    兼容两种格式:
      A) {"documents": {key: text}, "cases": [{..., "doc": key}]}
         -> 展开 full_text=documents[doc]，evidence_text缺省也用documents[doc]
      B) [{case...}, ...] 逐条内联 full_text/evidence_text（旧格式/单文件测试）
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return [BenchmarkCase(**item) for item in raw]

    documents = raw.get("documents", {})
    cases: list[BenchmarkCase] = []
    for item in raw.get("cases", []):
        item = {k: v for k, v in item.items() if not k.startswith("_")}
        doc_key = item.pop("doc", None)
        if doc_key is not None:
            doc_text = documents[doc_key]
            item.setdefault("full_text", doc_text)
            item.setdefault("evidence_text", doc_text)
        cases.append(BenchmarkCase(**item))
    return cases


# ---------------------------------------------------------------
# K2·B3优化：bootstrap置信区间
# ---------------------------------------------------------------
def _prf(gold: list[str], pred: list[str], label: str) -> tuple[float, float, float]:
    """单标签 P/R/F1（与reviewer._prf口径一致，本地独立实现避免私有引用）"""
    tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
    fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
    fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def _metrics_from_sample(sample) -> dict:
    """从一组BenchmarkResult算四个核心指标（重采样工作马）"""
    n = len(sample)
    gold = [r.gold_verdict for r in sample]
    pred = [r.predicted_verdict for r in sample]
    macro_f1 = sum(_prf(gold, pred, lb)[2] for lb in _LABELS) / len(_LABELS)
    bad_total = bad_pass = 0
    for r in sample:
        if r.gold_verdict != "supported" or not r.gold_quote_alignment:
            bad_total += 1
            if r.predicted_verdict == "supported" and r.predicted_quote_alignment:
                bad_pass += 1
    return {
        "verdict_accuracy": sum(r.verdict_correct for r in sample) / n,
        "alignment_accuracy": sum(r.alignment_correct for r in sample) / n,
        "macro_f1": macro_f1,
        "false_pass_rate": (bad_pass / bad_total) if bad_total else 0.0,
    }


def bootstrap_ci(results: list, n_boot: int = 2000, seed: int = 2026) -> dict:
    """
    对Benchmark逐条结果做bootstrap重采样，返回各核心指标的95%置信区间。

    原理（K2·B3）: 几十条声明时一条就是约2个百分点，"幻觉率8%"和"10%"
    在统计上可能无差别。对样本有放回重采样n_boot次，每次重算指标，
    取2.5%/97.5%分位数——把"一个数字"变成"一个带误差范围的估计"。

    返回:
        {"verdict_accuracy": [lo, hi], "macro_f1": [lo, hi],
         "false_pass_rate": [lo, hi], "alignment_accuracy": [lo, hi],
         "n_boot": 2000, "note": "95% bootstrap CI (percentile)"}
    """
    n = len(results)
    if n == 0 or any(r.gold_verdict is None or r.gold_quote_alignment is None
                     for r in results):
        # 没有gold无法还原逐类指标与误放行率（旧版报告的results不带
        # gold字段），明确返回空而不是带偏计算
        return {}
    rng = random.Random(seed)
    buckets: dict[str, list[float]] = {
        "verdict_accuracy": [], "alignment_accuracy": [],
        "macro_f1": [], "false_pass_rate": []}
    for _ in range(n_boot):
        sample = [results[rng.randrange(n)] for _ in range(n)]
        for k, v in _metrics_from_sample(sample).items():
            buckets[k].append(v)

    def ci(xs: list[float]) -> list[float]:
        xs = sorted(xs)
        return [round(xs[int(0.025 * n_boot)], 4),
                round(xs[min(n_boot - 1, int(0.975 * n_boot))], 4)]

    out = {k: ci(v) for k, v in buckets.items()}
    out["n_boot"] = n_boot
    out["note"] = "95% bootstrap CI (percentile method)"
    return out


def run(save: bool = True) -> "BenchmarkReport":
    """运行固定Benchmark并打印完整指标，可选落盘报告。"""
    from models import BenchmarkReport  # noqa: F401 (类型提示用)

    cases = load_cases()
    print(f"[Benchmark] 加载固定测试集: {len(cases)} 例 ({CASES_PATH})")
    report = run_d_benchmark(cases)

    # K2: 自动附带bootstrap置信区间（逐条结果重采样）
    ci = bootstrap_ci(report.results)
    if ci:
        print("  95%置信区间(bootstrap):")
        for k in ("verdict_accuracy", "alignment_accuracy", "macro_f1",
                  "false_pass_rate"):
            if k in ci:
                lo, hi = ci[k]
                print(f"    {k:<20s} [{lo:.3f}, {hi:.3f}]")

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        payload = report.model_dump()
        if ci:
            payload["confidence_intervals"] = ci  # 置信区间一并落盘
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        # 带时间戳的存档（供compare做"修改前后"对比）
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = os.path.join(OUT_DIR, f"benchmark_report_{stamp}.json")
        with open(archive, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Benchmark] 报告已保存: {REPORT_PATH}")
        print(f"[Benchmark] 存档(供前后对比): {archive}")
    return report


def ablation(save: bool = True) -> None:
    """
    消融实验（K2·B3）：四级核验逐层关闭对比。

    回答答辩必问的"每层核验各自贡献多少"：
      - full vs no_alignment  → 一级程序化对齐的价值（拦伪造引用）
      - full vs no_semantic   → LLM语义核验+数值校验整体的价值
      - full vs no_numeric    → K1数值程序校验的独立价值（拦模糊化）
    结果落盘 benchmarks 侧的 ablation_results.json（data/benchmark/）。
    """
    cases = load_cases()
    print(f"[消融实验] 加载固定测试集: {len(cases)} 例")
    modes = ("full", "no_semantic", "no_alignment", "no_numeric")
    reports: dict[str, dict] = {}
    for mode in modes:
        report = run_d_benchmark(cases, ablation=mode)
        payload = report.model_dump()
        ci = bootstrap_ci(report.results)
        if ci:
            payload["confidence_intervals"] = ci
        reports[mode] = payload

    # ---- 打印对比表 ----
    print("\n" + "=" * 86)
    print("🔬 消融实验对比（数值越高越好；误放行率越低越好）")
    print("=" * 86)
    print(f"{'指标':<20}{'完整核验':>12}{'仅程序对齐':>13}"
          f"{'仅语义核验':>13}{'仅关数值校验':>14}")
    print("-" * 86)
    keys = [("verdict_accuracy", "Verdict Accuracy"),
            ("alignment_accuracy", "Alignment Accuracy"),
            ("macro_f1", "Macro-F1"),
            ("false_pass_rate", "误放行率(低好)")]
    for key, name in keys:
        vals = [reports[m].get(key, 0.0) for m in modes]
        print(f"{name:<20}{vals[0]:>12.3f}{vals[1]:>13.3f}"
              f"{vals[2]:>13.3f}{vals[3]:>14.3f}")
    for m in modes:
        ci = reports[m].get("confidence_intervals") or {}
        if ci.get("macro_f1"):
            lo, hi = ci["macro_f1"]
            print(f"  {m:<12} Macro-F1 95%CI [{lo:.3f}, {hi:.3f}]")
    print("=" * 86)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "ablation_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print(f"[消融实验] 结果已保存: {out_path}")


def _get(report: dict, key: str, default=0.0):
    return report.get(key, default) if isinstance(report, dict) else getattr(report, key, default)


def compare(old_path: str, new_path: str) -> None:
    """
    比较两份Benchmark报告，输出"修改前后"的指标变化表。

    比较维度: Verdict Accuracy / Alignment Accuracy / Macro-F1 /
    误放行率 / 逐类F1。
    """
    with open(old_path, "r", encoding="utf-8") as f:
        old = json.load(f)
    with open(new_path, "r", encoding="utf-8") as f:
        new = json.load(f)

    rows = [
        ("Verdict Accuracy", _get(old, "verdict_accuracy"), _get(new, "verdict_accuracy")),
        ("Alignment Accuracy", _get(old, "alignment_accuracy"), _get(new, "alignment_accuracy")),
        ("Macro-F1", _get(old, "macro_f1"), _get(new, "macro_f1")),
        ("误放行率(越低越好)", _get(old, "false_pass_rate"), _get(new, "false_pass_rate")),
    ]
    for label in ("supported", "unsupported", "contradicted"):
        old_f1 = (_get(old, "f1") or {}).get(label, 0.0)
        new_f1 = (_get(new, "f1") or {}).get(label, 0.0)
        rows.append((f"F1[{label}]", old_f1, new_f1))

    print("\n" + "=" * 60)
    print("📊 Benchmark 前后对比")
    print(f"  旧: {old_path}  (n={_get(old, 'total_cases', '?')})")
    print(f"  新: {new_path}  (n={_get(new, 'total_cases', '?')})")
    print("=" * 60)
    print(f"{'指标':<24}{'修改前':>10}{'修改后':>10}{'变化':>10}")
    print("-" * 60)
    for name, o, n in rows:
        delta = n - o
        sign = "+" if delta >= 0 else ""
        arrow = "↑" if delta > 1e-9 else ("↓" if delta < -1e-9 else "→")
        print(f"{name:<24}{o:>10.3f}{n:>10.3f}{sign}{delta:>8.3f} {arrow}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "ablation":
        ablation()
    else:
        run()
