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
    from concurrent.futures import ThreadPoolExecutor

    # ---- 提前启动提取池（2026-10-16优化1: 下载→提取流水线重叠）----
    # 旧流程: 全部PDF下载完才开始提取(串行两段); 新流程: 每篇PDF
    # 落盘立即投入提取线程池, 两段重叠执行——提取的等待时间被剩余
    # 下载时间吸收, 端到端省30-60秒。重复下载的论文若被最终去重
    # 淘汰, 其已完成的提取结果直接丢弃(LLM缓存下代价极小)
    extract_pool = ThreadPoolExecutor(max_workers=10,
                                      thread_name_prefix="extract")
    extract_futures = []  # [(paper, future), ...] 提交序=下载完成序

    def _on_paper_downloaded(p):
        """下载回调: 单篇PDF就绪即提交提取(在下载线程中执行;
        list.append线程安全; 提取自身的失败在收尾统一处理)"""
        if getattr(p, "local_path", None):
            extract_futures.append(
                (p, extract_pool.submit(extractor.extract_card, p)))

    try:
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
            # on_downloaded=下载→提取重叠(2026-10-16优化1)
            papers = searcher.run(plan, results_per_query=8, top_k=top_k,
                                  fast=fast, on_downloaded=_on_paper_downloaded)
            # 论文清单实时落盘(2026-10-15流式展示): 运行面板读此文件
            # 即时上屏检索成果, 用户不用等提取/审查/综合完成
            save_json(os.path.join(config.DATA_DIR, "pipeline_papers.json"),
                      [{"arxiv_id": p.arxiv_id, "title": p.title,
                        "year": p.year, "cited_by": p.cited_by,
                        "arxiv_only": getattr(p, "arxiv_only", True)}
                       for p in papers])
        else:
            check_cancel()
            papers = searcher.download(papers,
                                       on_downloaded=_on_paper_downloaded)
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

        # ---- 3. 提取Agent（与下载重叠收尾, 2026-10-16优化1）----
        # 下载阶段每篇PDF就绪即已提交提取, 这里只等剩余future收尾
        check_cancel()
        if extract_futures:
            done_n = sum(1 for _, f in extract_futures if f.done())
            log(f"提取Agent: {len(extract_futures)}篇已流水线式提交"
                f"（{done_n}篇已在下载期间完成），等待剩余提取收尾...",
                stage="提取")
        cards = []
        for p, f in extract_futures:
            try:
                c = f.result()
            except Exception as e:
                log(f"提取失败(跳过): {p.title[:40]} | {e}", stage="提取")
                continue
            if c:
                cards.append(c)
        # 按最终论文顺序重排(提交序=下载完成序≠最终排序序);
        # 落盘口径与extractor.run一致
        order = {p.arxiv_id: i for i, p in enumerate(papers)}
        cards.sort(key=lambda c: order.get(c.arxiv_id, 999))
        cards_path = os.path.join(config.DATA_DIR, "paper_cards.json")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(cards_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in cards], f,
                      ensure_ascii=False, indent=2)
        log(f"提取Agent完成: {len(cards)}张卡片, "
            f"{sum(len(c.claims) for c in cards)}条声明", stage="提取")

        # ---- 4. 审查Agent ----
        check_cancel()
        log("审查Agent: 两级核验中（程序对齐+语义审查）...", stage="审查")
        results = reviewer.run(papers, cards)
        report = reviewer.generate_report(results)
        log(f"审查Agent完成: 幻觉率 {report.hallucination_rate:.1%}", stage="审查")

        # ---- 5+7. 综合与矛盾检测（并行, 2026-10-16优化4）----
        # 两者都只依赖审查产物(paper_cards/review_results), 互不读取
        # 对方输出(已核实synthesizer不读conflicts.json)——原先串行
        # 纯属浪费, 深度档省30-60秒。快速模式仍跳过矛盾检测
        check_cancel()
        if fast:
            log("综合Agent: 生成综述报告与引用图谱...", stage="综合")
            synthesizer.run(topic)
            log("综合Agent完成: 报告+表格+图谱已生成", stage="综合")
            log("⚡快速模式: 跳过矛盾检测", stage="矛盾检测")
            # 落盘空结果, 避免上一主题的conflicts.json残留误导下游展示
            save_json(os.path.join(config.DATA_DIR, "conflicts.json"), [])
        else:
            from agents import conflict_detector
            log("综合Agent+矛盾检测: 并行执行中（两者互不依赖）...",
                stage="综合")
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_syn = ex.submit(synthesizer.run, topic)
                fut_con = ex.submit(conflict_detector.run)
                fut_syn.result()   # 任一失败异常照常上抛(与原串行一致)
                conflicts = fut_con.result()
            log("综合Agent完成: 报告+表格+图谱已生成", stage="综合")
            n_con = sum(1 for c in conflicts if c["relation"] == "contradict")
            log(f"矛盾检测完成: 直接矛盾{n_con}处 + "
                f"张力{len(conflicts) - n_con}处", stage="矛盾检测")

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

        return report
    finally:
        # 取消/异常时不再等队列里的提取任务(进行中的会自然跑完);
        # 正常路径此时futures已全部collect, shutdown是幂等收尾
        extract_pool.shutdown(wait=False, cancel_futures=True)


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
