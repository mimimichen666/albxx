# -*- coding: utf-8 -*-
"""
pipeline_worker.py —— 流水线子进程入口（方案二: 进程隔离+硬终止）
================================================================
由 app.py 以 subprocess 启动, 在独立Python进程中跑完
规划→检索→提取→审查→综合→证据定位→矛盾检测 全流程。

与UI的通信（全部走 data/ 下的文件, 无需复杂IPC）:
  - pipeline_status.json   : 当前状态 {state, stage, msg, topic...}
                             state ∈ running / done / error / cancelled
  - pipeline_progress.jsonl: 追加式进度日志(每行一个JSON事件)
  - pipeline_cancel.flag   : 取消标志文件; worker在阶段边界发现即
                             优雅退出(已完成的成果全部保留在data/)
  - pipeline_stdout.log    : 父进程重定向的stdout/stderr(诊断用)

退出码: 0=完成  1=出错  2=被用户取消

运行: python pipeline_worker.py --topic "主题" --top_k 5 --mode standard --years 2
"""
import argparse
import json
import os
import sys
import time
import traceback

import config
import budget

# ★Windows编码修复: 子进程stdout被重定向到文件时, Python默认按系统区域
# 编码(GBK)包装, 审查Agent打印"✗"(\u2717)等字符会UnicodeEncodeError直接
# 炸掉流水线, 且GBK字节写进UTF-8日志文件造成乱码。统一reconfigure为
# UTF-8+替换模式(不可编码字符降级为"?", 任何print都不再致命)。
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 确保data目录存在(config导入后即有DATA_DIR)
os.makedirs(config.DATA_DIR, exist_ok=True)

STATUS_FILE = os.path.join(config.DATA_DIR, "pipeline_status.json")
PROGRESS_FILE = os.path.join(config.DATA_DIR, "pipeline_progress.jsonl")
CANCEL_FLAG = os.path.join(config.DATA_DIR, "pipeline_cancel.flag")


class Cancelled(Exception):
    """用户请求终止(worker在阶段边界主动抛出, 成果已保存)"""


def write_status(state: str, stage: str = "", msg: str = "", **extra):
    """原子性写状态文件(先写临时文件再替换, 防止读到半个JSON)"""
    payload = {"state": state, "stage": stage, "msg": msg,
               "ts": time.strftime("%H:%M:%S"), **extra}
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATUS_FILE)


def log(msg: str, stage: str = "", state: str = "running", **extra):
    """进度事件: 追加到jsonl + 刷新状态文件 + 打印(进stdout日志)"""
    write_status(state, stage, msg, **extra)
    event = {"t": time.strftime("%H:%M:%S"), "stage": stage, "msg": msg}
    try:
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass
    print(f"[{event['t']}] [{stage or state}] {msg}", flush=True)


def check_cancel():
    """阶段边界检查: 取消标志文件存在 → 抛Cancelled(成果已落盘, 无需清理)"""
    if os.path.exists(CANCEL_FLAG):
        raise Cancelled()


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_pipeline(topic: str, top_k: int, mode: str, years: int,
                 fast: bool = False):
    """完整流水线（与原app.py内联版逻辑一致, UI替换为文件进度）"""
    from agents import planner, searcher, extractor, reviewer, synthesizer

    # ---- 1+2. 检索阶段（标准 / 三模式两条路径, 模式未命中降级标准）----
    use_mode = mode in ("mode1", "mode2", "mode3")
    mode_miss = False
    papers = []
    if use_mode:
        from agents import mode_assistant
        log("模式检索: 规划检索词+多库检索中...", stage="检索")
        papers = mode_assistant.retrieve_for_pipeline(
            topic, mode, years, top_k=top_k)
        if papers:
            log(f"模式检索命中 {len(papers)} 篇（带PDF直链），开始下载PDF...",
                stage="检索")
        else:
            log("模式检索未命中带PDF直链的文献，自动切换为标准检索继续...",
                stage="检索")
            use_mode = False
            mode_miss = True
    if not use_mode:
        check_cancel()
        log("规划Agent: 分解研究主题中...", stage="规划")
        plan = planner.make_plan(topic)
        save_json(os.path.join(config.DATA_DIR, "plan_last.json"),
                  plan.model_dump())
        log(f"规划Agent完成: {len(plan.keywords)}组检索词", stage="规划")

        check_cancel()
        log(f"检索Agent: 下载论文PDF中（检索词: {', '.join(plan.keywords)}）...",
            stage="检索")
        # fast=快速模式(2026-10-15双模式): 每词5条+跳过引用链/经典注入,
        # 检索段约减半; 深度模式(默认)=全信号召回
        papers = searcher.run(plan, results_per_query=8, top_k=top_k,
                              fast=fast)
        # 论文清单实时落盘(2026-10-15流式展示): 运行面板读此文件
        # 即时上屏检索成果, 用户不用等提取/审查/综合完成
        save_json(os.path.join(config.DATA_DIR, "pipeline_papers.json"),
                  [{"arxiv_id": p.arxiv_id, "title": p.title,
                    "year": p.year, "cited_by": p.cited_by,
                    "arxiv_only": getattr(p, "arxiv_only", True)}
                   for p in papers])
    else:
        check_cancel()
        papers = searcher.download(papers)
        # 与searcher.run落盘口径一致, 供提取/审查只处理有本地PDF的论文
        if papers:
            save_json(os.path.join(config.DATA_DIR, "papers_meta.json"),
                      [p.model_dump() for p in papers])

    if not papers:
        if mode_miss:
            log("模式检索未命中，流水线终止", stage="检索", state="error")
        else:
            log("检索失败: 没有获得可用论文", stage="检索", state="error")
        sys.exit(1)
    log(f"检索Agent完成: {len(papers)}篇论文, PDF已下载", stage="检索")

    # 模式检索附加: 核心名词速查表（供论文原文页对照）
    if use_mode or mode_miss:
        try:
            from agents import mode_assistant
            log("生成核心名词速查表...", stage="检索")
            g = mode_assistant.generate_glossary(papers, topic)
            if g:
                save_json(os.path.join(config.DATA_DIR, "glossary_last.json"),
                          {"topic": topic, "md": g})
        except Exception as e:
            log(f"速查表生成失败(不影响流水线): {e}", stage="检索")

    # ---- 3. 提取Agent ----
    check_cancel()
    log(f"提取Agent: 抽取{len(papers)}篇论文的信息卡片（每篇约1分钟）...",
        stage="提取")
    cards = extractor.run(papers)
    log(f"提取Agent完成: {len(cards)}张卡片, "
        f"{sum(len(c.claims) for c in cards)}条声明", stage="提取")

    # ---- 4. 审查Agent ----
    check_cancel()
    log("审查Agent: 两级核验中（程序对齐+语义审查）...", stage="审查")
    results = reviewer.run(papers, cards)
    report = reviewer.generate_report(results)
    log(f"审查Agent完成: 幻觉率 {report.hallucination_rate:.1%}", stage="审查")

    # ---- 5. 综合Agent ----
    check_cancel()
    log("综合Agent: 生成综述报告与引用图谱...", stage="综合")
    synthesizer.run(topic)
    log("综合Agent完成: 报告+表格+图谱已生成", stage="综合")

    # ---- 6. 证据定位（方案一: 证据链穿透）----
    check_cancel()
    log("证据定位: 定位引用到PDF页面并高亮...", stage="证据定位")
    from agents import evidence_locator
    cards_raw = json.load(open(
        os.path.join(config.DATA_DIR, "paper_cards.json"), encoding="utf-8"))
    reviews_raw = json.load(open(
        os.path.join(config.DATA_DIR, "review_results.json"),
        encoding="utf-8"))
    ev_index = evidence_locator.build_evidence_index(cards_raw, reviews_raw)
    located_n = sum(1 for v in ev_index.values() if v.get("located"))
    log(f"证据定位完成: {located_n}条声明可穿透到PDF原文", stage="证据定位")

    # ---- 7. 矛盾检测（方案三: 学术争议发现; 快速模式跳过省时）----
    if fast:
        log("⚡快速模式: 跳过矛盾检测", stage="矛盾检测")
        # 落盘空结果, 避免上一主题的conflicts.json残留误导下游展示
        save_json(os.path.join(config.DATA_DIR, "conflicts.json"), [])
    else:
        check_cancel()
        log("矛盾检测: 在已核验声明间寻找学术争议...", stage="矛盾检测")
        from agents import conflict_detector
        conflicts = conflict_detector.run()
        n_con = sum(1 for c in conflicts if c["relation"] == "contradict")
        log(f"矛盾检测完成: 直接矛盾{n_con}处 + 张力{len(conflicts) - n_con}处",
            stage="矛盾检测")

    # ---- 8. 综述总结自动生成（2026-11-09: 不再由UI按钮触发）----
    check_cancel()
    log("综述总结: 自动生成叙述性综述与AI观点...", stage="综述总结")
    try:
        import llm_client
        _sv_path = os.path.join(config.OUTPUT_DIR, "survey_summary.md")
        try:
            _papers_raw = json.load(open(
                os.path.join(config.DATA_DIR, "papers_meta.json"),
                encoding="utf-8"))
            _cards_raw = json.load(open(
                os.path.join(config.DATA_DIR, "paper_cards.json"),
                encoding="utf-8"))
            _reviews_raw = json.load(open(
                os.path.join(config.DATA_DIR, "review_results.json"),
                encoding="utf-8"))
        except Exception:
            _papers_raw, _cards_raw, _reviews_raw = [], [], []
        # 只采用通过两级核验的声明（verdict=supported），保持不编造基调
        _ok = set()
        for _r in (_reviews_raw or []):
            try:
                if _r.get("verdict", {}).get("verdict") == "supported":
                    _ok.add(f"{_r['arxiv_id']}#{_r['claim_index']}")
            except Exception:
                pass
        _cards_by_id = {c.get("arxiv_id"): c for c in (_cards_raw or [])}
        _blocks = []
        for _i, _p in enumerate(_papers_raw or [], 1):
            _card = _cards_by_id.get(_p.get("arxiv_id"))
            _ls = [f"[{_i}] 标题: {_p.get('title', '')}",
                   f"    年份: {_p.get('year', '')}  "
                   f"被引次数: {_p.get('cited_by', 0)}"]
            _ab = (_p.get("abstract") or "").strip()
            if _ab:
                _ls.append(f"    摘要: {_ab[:500]}")
            if _card:
                _kept = []
                for _ci, _cl in enumerate(_card.get("claims") or []):
                    if not _ok or f"{_p.get('arxiv_id')}#{_ci}" in _ok:
                        _kept.append(_cl.get("content", ""))
                for _c in _kept[:4]:
                    _ls.append(f"    · 关键信息: {_c[:220]}")
            _blocks.append("\n".join(_ls))
        if _blocks:
            _src = "\n\n".join(_blocks)
            _sv_prompt = (
                "你是资深学术综述撰稿人兼领域分析师。"
                f"研究主题: {topic}\n"
                "基于给定的论文清单（[编号]为引用标识），输出两部分内容。\n\n"
                "【第一部分: 叙述性综述】\n"
                "- 用连贯的叙述性文字（禁止表格、禁止罗列式堆砌）梳理研究图景\n"
                "- 按脉络组织: 研究背景与动机 → 现有工作按思路分派 → "
                "各派代表工作与核心思想 → 发展趋势\n"
                "- 只能使用给定论文的信息，不得编造未提供的文献、数据或结论；"
                "提及具体工作时标注[编号]（600-900字）\n\n"
                "【第二部分: AI观点（仅供参考）】\n"
                "- 对检索出的论文做跨文献归纳偏置分析，输出3-5条观点\n"
                "- 每条观点必须遵循如下句式骨架（用实际归纳内容填充XX）:\n"
                "  现有 XX 方法在 XX 场景下具备 XX 优势，但普遍存在 XX 局限，"
                "在 XX 条件下效果不足，目前针对该问题的相关研究较少，"
                "有待进一步探索\n"
                "- 每条观点后标注依据: (支撑: [编号])；无直接支撑的推断标 (推断)\n"
                "- 观点之间不重复，各自覆盖不同的方法路线或维度\n\n"
                "输出为markdown，结构如下:\n"
                "# 📖 叙述性综述\n（正文）\n"
                "# 💡 AI观点（仅供参考）\n1. ...\n2. ...\n"
                "（结尾附一行提示: 以上观点为AI归纳推断，仅供参考）\n\n"
                "论文清单如下:\n" + _src)
            _out_sv = llm_client.chat(
                [{"role": "system",
                  "content": "你是严谨的学术综述撰稿人，忠于给定材料。"},
                 {"role": "user", "content": _sv_prompt}],
                temperature=0.3)
            if _out_sv:
                with open(_sv_path, "w", encoding="utf-8") as _f:
                    _f.write(_out_sv)
                log("综述总结完成: 已保存 output/survey_summary.md",
                    stage="综述总结")
            else:
                log("综述总结生成返回为空，跳过", stage="综述总结")
        else:
            log("无可用论文数据，跳过综述总结生成", stage="综述总结")
    except Exception as _sv_err:
        log(f"综述总结自动生成失败(不影响流水线): {_sv_err}",
            stage="综述总结")

    return report


def main():
    ap = argparse.ArgumentParser(description="文献综述流水线子进程")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--mode", default="standard",
                    choices=["standard", "mode1", "mode2", "mode3"])
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--fast", action="store_true",
                    help="快速模式: 每词5条+跳过引用链/经典注入/矛盾检测")
    args = ap.parse_args()

    # 清理上一轮的取消标志(每次启动都是新任务)
    if os.path.exists(CANCEL_FLAG):
        os.remove(CANCEL_FLAG)
    # 进度日志从头开始
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    # 写PID文件: UI跨会话判断"是否已有worker在跑"。刷新浏览器后
    # session_state会清零、运行锁失效, 用户再点按钮会造成并发双跑,
    # 两个worker互相抢API限流额度(2026-10-14实际发生过)
    with open(os.path.join(config.DATA_DIR, "pipeline.pid"), "w") as f:
        f.write(str(os.getpid()))

    write_status("running", stage="初始化", msg="子进程已启动")
    # 全局时间预算(2026-10-15合并自review-agent-benchmark分支):
    # 各Agent循环边界budget.check, 超预算抛BudgetExceededError,
    # 防止限流退避叠加导致worker病态长跑(子进程架构下网页有终止
    # 按钮兜底, 预算再提供第二道保险+明确的失败阶段标注)
    budget.start(config.GLOBAL_BUDGET_MIN)
    try:
        report = run_pipeline(args.topic, args.top_k, args.mode,
                              args.years, fast=args.fast)
        import llm_client
        llm_client.print_usage()
        write_status("done", stage="完成", msg="全部阶段执行完毕",
                     topic=args.topic,
                     hallucination_rate=round(report.hallucination_rate, 4))
        print("[worker] 流水线完成, 退出码0", flush=True)
        sys.exit(0)
    except Cancelled:
        write_status("cancelled", stage="已终止",
                     msg="用户终止: 已完成阶段的成果已保留", topic=args.topic)
        print("[worker] 收到取消请求, 优雅退出(退出码2)", flush=True)
        sys.exit(2)
    except budget.BudgetExceededError as e:
        write_status("error", stage=e.stage,
                     msg=f"全局时间预算({config.GLOBAL_BUDGET_MIN:.0f}分钟)耗尽: "
                         "多为API限流退避叠加所致, 可在.env调大GLOBAL_BUDGET_MIN"
                         "或稍后重试(已完成阶段的成果已保留)",
                     topic=args.topic)
        print(f"[worker] {e}, 优雅退出(退出码3)", flush=True)
        sys.exit(3)
    except SystemExit:
        raise                      # 上面的sys.exit(1)直接透传
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        write_status("error", stage="出错", msg="流水线执行异常, 详见stdout日志",
                     traceback=err[-2000:])
        sys.exit(1)
    finally:
        # 无论正常完成/出错/取消, 退出前删除PID文件
        # (不删的话UI会一直误判"流水线仍在运行")
        budget.stop()
        try:
            os.remove(os.path.join(config.DATA_DIR, "pipeline.pid"))
        except OSError:
            pass


if __name__ == "__main__":
    main()
