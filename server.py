# -*- coding: utf-8 -*-
"""
server.py —— FastAPI 任务服务器（方案一：前端与执行彻底解耦）
================================================================
替代 Streamlit 的执行模型：
  - POST /api/run       提交任务，立即返回 task_id（页面不再被任务阻塞）
  - GET  /api/tasks     任务列表（历史可查回）
  - GET  /api/tasks/{id}/stream   SSE 实时推送进度（阶段/日志/终态）
  - POST /api/tasks/{id}/cancel   即时终止：terminate/kill worker 子进程，
                                  并置取消标志以便优雅退出的场合兜底
  - GET  /api/tasks/{id}/result   报告+论文+矛盾+核验结果
  - GET  /api/tasks/{id}/graph    引用图谱（HTML）

任务隔离（不改动任何 agents 代码的前提下）：
  - 流水线 worker 仍写共享的 data/pipeline_status.json（同一时刻只有一个
    活跃任务，第二个提交会收到 409 提示）；
  - 每个任务终态时把产物快照进 data/tasks/<id>/，刷新/重启后历史仍可查回；
  - SSE 只对活跃任务直播进度；历史任务直接返回快照终态。

鉴权：若配置 APP_PASSWORD，所有 /api/* 要求头 X-App-Password 或
查询参数 ?pw=（SSE 的 EventSource 无法自定义头，走 query 参数）。

运行: uvicorn server:app --host 0.0.0.0 --port 7860
(HF Spaces 要求监听 7860 端口；本地可用 PORT 环境变量覆盖)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import config

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = config.PROJECT_ROOT
DATA = config.DATA_DIR
OUTPUT = config.OUTPUT_DIR
STATIC = os.path.join(BASE, "static")
WORKER = os.path.join(BASE, "pipeline_worker.py")
TASKS_DIR = os.path.join(DATA, "tasks")          # 每任务快照目录
REGISTRY = os.path.join(DATA, "tasks.json")      # 任务注册表
STATUS_FILE = os.path.join(DATA, "pipeline_status.json")
PROGRESS_FILE = os.path.join(DATA, "pipeline_progress.jsonl")
CANCEL_FLAG = os.path.join(DATA, "pipeline_cancel.flag")

# 各阶段产物 → 快照时的目标名（None 表示该文件不存在时跳过）
_SNAPSHOT_FILES = [
    ("final_report.md", os.path.join(OUTPUT, "final_report.md")),
    ("review_table.md", os.path.join(OUTPUT, "review_table.md")),
    ("citation_graph.html", os.path.join(OUTPUT, "citation_graph.html")),
    ("papers.json", os.path.join(DATA, "pipeline_papers.json")),
    ("paper_cards.json", os.path.join(DATA, "paper_cards.json")),
    ("review_results.json", os.path.join(DATA, "review_results.json")),
    ("conflicts.json", os.path.join(DATA, "conflicts.json")),
    ("evidence_index.json", os.path.join(DATA, "evidence_index.json")),
    ("plan.json", os.path.join(DATA, "plan_last.json")),
    ("glossary.json", os.path.join(DATA, "glossary_last.json")),
    # 老页面功能合并新增：论文完整元数据 / 综述总结 / Benchmark 评测
    ("meta.json", os.path.join(DATA, "papers_meta.json")),
    ("survey_summary.md", os.path.join(OUTPUT, "survey_summary.md")),
    ("blindspots.md", os.path.join(OUTPUT, "blindspots.md")),
    ("coverage_check.json", os.path.join(OUTPUT, "coverage_check.json")),
    ("benchmark.json", os.path.join(DATA, "benchmark",
                                    "benchmark_report.json")),
]

_lock = threading.Lock()          # 保护注册表读写与任务提交互斥
_ACTIVE_PROCS: dict = {}          # tid -> Popen，取消接口据此即时终止进程

app = FastAPI(title="文献综述 Agent")


# ----------------------------------------------------------------
# 注册表（data/tasks.json：任务列表持久化，容器存活期内可查回）
# ----------------------------------------------------------------
def _load_tasks() -> list[dict]:
    if os.path.exists(REGISTRY):
        try:
            with open(REGISTRY, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_tasks(tasks: list[dict]) -> None:
    os.makedirs(DATA, exist_ok=True)
    tmp = REGISTRY + ".tmp"
    # 过滤内存态字段（_proc=Popen句柄等）。2026-11-22检索失败根因：
    # api_run 把 Popen 塞进 task 后直接 _save_tasks → json.dump 抛
    # "Object of type Popen is not JSON serializable" → 接口500，
    # pid 没落盘、watcher 没启动 → sweep 把活任务误判为"已崩溃"。
    # 约定：下划线开头的 key 一律是内存态，永不落盘。
    clean = [{k: v for k, v in t.items() if not str(k).startswith("_")}
             for t in tasks]
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=1)
    os.replace(tmp, REGISTRY)


def _active(tasks: list[dict]) -> dict | None:
    return next((t for t in tasks if t.get("state") == "running"), None)


def _find(tasks: list[dict], tid: str) -> dict | None:
    return next((t for t in tasks if t.get("id") == tid), None)


def _task_dir(tid: str) -> str:
    return os.path.join(TASKS_DIR, tid)


# ----------------------------------------------------------------
# 鉴权（APP_PASSWORD 可选）
# ----------------------------------------------------------------
def _check_auth(request: Request, pw: str = "") -> None:
    secret = getattr(config, "APP_PASSWORD", "")
    if not secret:
        return
    got = request.headers.get("X-App-Password") or request.query_params.get("pw", "")
    if got != secret:
        raise HTTPException(401, "需要访问密码（X-App-Password 头或 ?pw= 参数）")


# ----------------------------------------------------------------
# 任务生命周期
# ----------------------------------------------------------------
def _spawn(task: dict) -> subprocess.Popen:
    """启动 worker 子进程，stdout/stderr 重定向到任务的运行日志。
    返回 Popen 句柄（仅存于内存传给 watcher；注册表只记 pid）"""
    log_path = os.path.join(_task_dir(task["id"]), "run.log")
    logf = open(log_path, "w", encoding="utf-8")
    cmd = [sys.executable, WORKER,
           "--topic", task["topic"],
           "--top_k", str(task["top_k"]),
           "--mode", task["mode"],
           "--years", str(task["years"])]
    if task.get("fast"):
        cmd.append("--fast")
    proc = subprocess.Popen(cmd, cwd=BASE, stdout=logf, stderr=logf)
    task["pid"] = proc.pid
    return proc


def _snapshot(task: dict, state: str) -> None:
    """终态时把共享 data/output 的产物复制进 data/tasks/<id>/"""
    d = _task_dir(task["id"])
    os.makedirs(d, exist_ok=True)
    for name, path in _SNAPSHOT_FILES:
        try:
            if os.path.exists(path):
                shutil.copy2(path, os.path.join(d, name))
        except OSError:
            pass
    task["state"] = state
    task["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def _read_status() -> dict:
    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _pid_alive(pid: int | None) -> bool:
    """跨平台判活。Windows 用 OpenProcess+GetExitCodeProcess（os.kill(0) 在
    Windows 语义不同，误用会直接杀进程），POSIX 用信号0探测。"""
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32
            # 64位句柄必须用 c_void_p：默认 c_int 截断会导致句柄无效、
            # GetExitCodeProcess 静默失败 → 活进程被误判死亡
            k32.OpenProcess.restype = ctypes.c_void_p
            k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not k32.GetExitCodeProcess(h, ctypes.byref(exit_code)):
                    return False  # 查询失败按死亡处理（句柄失效=进程不存在）
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                k32.CloseHandle(h)
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _sweep_orphans() -> None:
    """死亡检测兜底：worker 进程已不存在但任务仍标 running → 立即落终态。

    watcher 线程只在"本服务器进程且其内存句柄存活"时有效——服务器重启、
    watcher 线程意外退出（如注册表读取瞬时异常触发提前 return）都会留下
    永远 running 的僵尸任务。此函数挂在轮询接口上，读注册表里的 pid 判活，
    不依赖任何内存状态。"""
    try:
        with _lock:
            tasks = _load_tasks()
            changed = False
            for t in tasks:
                if t.get("state") != "running":
                    continue
                pid = t.get("pid")
                if pid is None:  # 旧任务未持久化 pid → 退回 worker 写的 pid 文件
                    try:
                        pid = int(open(os.path.join(DATA, "pipeline.pid")).read().strip())
                    except Exception:
                        pid = None
                if pid:
                    if _pid_alive(pid):
                        continue
                else:
                    # 无 pid 可查（注册表没存 + 全局pid文件还没写）：不能立即判死。
                    # worker 从 spawn 到写入 pipeline.pid 有 1~3 秒窗口（解释器
                    # 启动+import），2026-11-22 实测提交后 2 秒就被误判"崩溃"。
                    # 90 秒宽限期内一律视为存活；超期仍无 pid 才按僵尸处理。
                    try:
                        age = time.time() - time.mktime(time.strptime(
                            t.get("submitted_at", ""), "%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        continue        # 时间都解析不了 → 保守跳过，绝不误杀
                    if age < 90:
                        continue
                was_cancel = bool(t.get("cancel_requested"))
                _snapshot(t, "cancelled" if was_cancel else "error")
                t["msg"] = ("已按用户请求终止" if was_cancel
                            else "worker 进程已退出且未写入终态（崩溃或服务器重启）")
                changed = True
            if changed:
                _save_tasks(tasks)
    except Exception:
        pass  # 兜底路径自身绝不能拖垮轮询接口


def _watcher(task: dict) -> None:
    """后台线程：跟踪活跃任务直到终态，然后落快照"""
    misses = 0
    while True:
        time.sleep(1.0)
        tasks = _load_tasks()
        cur = _find(tasks, task["id"])
        if cur is None:
            # 注册表瞬时读取失败/被清空 → 连续多次确认后再退出，
            # 避免一次文件竞态就永久杀死 watcher、留下僵尸任务
            misses += 1
            if misses >= 10:
                return
            continue
        misses = 0
        st = _read_status()
        state = st.get("state")
        proc = task.get("_proc")
        # 状态文件出现终态，或进程已退出 → 收尾
        terminal = {"done", "error", "cancelled"}
        if state in terminal:
            _ACTIVE_PROCS.pop(task["id"], None)
            with _lock:
                tasks = _load_tasks()
                cur = _find(tasks, task["id"])
                if cur and cur.get("state") == "running":
                    _snapshot(cur, state)
                    # 把 worker 写入状态文件的诊断信息带回注册表，
                    # 否则前端失败后只能看到 state，看不到原因
                    if st.get("stage"):
                        cur["stage"] = st.get("stage")
                    if st.get("msg"):
                        cur["msg"] = st.get("msg")
                    if st.get("hallucination_rate") is not None:
                        cur["hallucination_rate"] = st.get("hallucination_rate")
                    if st.get("traceback"):
                        cur["traceback"] = st.get("traceback")[-1500:]
                    _save_tasks(tasks)
            return
        if proc is not None and proc.poll() is not None:
            # 进程退出但状态文件没写成终态（如被强杀）→ 按是否请求过取消定性
            _ACTIVE_PROCS.pop(task["id"], None)
            with _lock:
                tasks = _load_tasks()
                cur = _find(tasks, task["id"])
                if cur and cur.get("state") == "running":
                    was_cancel = bool(cur.get("cancel_requested"))
                    _snapshot(cur, "cancelled" if was_cancel else "error")
                    if was_cancel:
                        cur["msg"] = "已按用户请求即时终止"
                    else:
                        cur["msg"] = f"worker 进程退出码 {proc.returncode}"
                    _save_tasks(tasks)
            return


@app.post("/api/run")
async def api_run(request: Request):
    body = await request.json()
    _check_auth(request)
    topic = (body.get("topic") or "").strip()
    if not topic:
        raise HTTPException(400, "topic 不能为空")
    mode = body.get("mode", "standard")
    if mode not in ("standard", "mode1", "mode2", "mode3"):
        raise HTTPException(400, f"不支持的检索模式: {mode}")

    _sweep_orphans()          # 先清掉历史僵尸，避免误报"已有任务在运行"
    # 智能引导选中的术语（中英对照 [{zh, en}]）：中转页用它做论文关键词命中分析
    raw_focus = body.get("focus_keywords") or []
    focus = []
    for k in raw_focus[:8]:
        if isinstance(k, dict):
            zh, en = str(k.get("zh") or "").strip(), str(k.get("en") or "").strip()
            if zh or en:
                focus.append({"zh": zh, "en": en})
    with _lock:
        tasks = _load_tasks()
        act = _active(tasks)
        if act:
            raise HTTPException(
                409, f"已有任务在运行（{act['topic'][:20]}…），请等待完成或先取消")
        tid = uuid.uuid4().hex[:8]
        task = {
            "id": tid,
            "topic": topic,
            "focus_keywords": focus,
            "mode": mode,
            "fast": bool(body.get("fast")),
            "top_k": int(body.get("top_k", 6)),
            "years": int(body.get("years", 2)),
            "state": "running",
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(_task_dir(tid), exist_ok=True)
        tasks.insert(0, task)
        _save_tasks(tasks)
        proc = _spawn(task)
        task["_proc"] = proc          # 仅内存，watcher 用于兜底判活
        task["pid"] = proc.pid        # 持久化 pid：服务器重启后 _sweep_orphans 仍能判活
        _save_tasks(tasks)
        _ACTIVE_PROCS[tid] = proc     # 取消接口据此即时终止
    threading.Thread(target=_watcher, args=(task,), daemon=True).start()
    return {"id": tid, "state": "running"}


@app.get("/api/tasks")
async def api_tasks(request: Request):
    _check_auth(request)
    _sweep_orphans()
    tasks = _load_tasks()
    return [{"id": t.get("id"), "topic": t.get("topic", ""),
             "mode": t.get("mode", "standard"),
             "fast": t.get("fast"), "state": t.get("state", "error"),
             "submitted_at": t.get("submitted_at"),
             "finished_at": t.get("finished_at"),
             "hallucination_rate": t.get("hallucination_rate"),
             "msg": t.get("msg")} for t in tasks[:50]]


@app.get("/api/tasks/{tid}")
async def api_task(tid: str, request: Request):
    _check_auth(request)
    _sweep_orphans()
    tasks = _load_tasks()
    t = _find(tasks, tid)
    if not t:
        raise HTTPException(404, "任务不存在")
    info = {k: t.get(k) for k in
            ("id", "topic", "focus_keywords", "mode", "fast", "state", "submitted_at",
             "finished_at", "hallucination_rate", "msg", "stage", "traceback")}
    if t["state"] == "running":
        info["live"] = _read_status()     # 活跃任务附带实时状态
    return info


@app.post("/api/tasks/{tid}/cancel")
async def api_cancel(tid: str, request: Request):
    _check_auth(request)
    tasks = _load_tasks()
    t = _find(tasks, tid)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t["state"] != "running":
        raise HTTPException(400, f"任务已是终态（{t['state']}），无需取消")
    # 1) 优雅信号：worker 若恰好处于阶段边界，可正常收尾退出
    open(CANCEL_FLAG, "w").close()
    # 2) 记录取消意图：worker 被强杀时，watcher 据此定性为 cancelled 而非 error
    with _lock:
        tasks = _load_tasks()
        cur = _find(tasks, tid)
        if cur:
            cur["cancel_requested"] = True
            _save_tasks(tasks)
    # 3) 即时终止：直接 terminate/kill worker 子进程
    proc = _ACTIVE_PROCS.get(tid)
    hard = False
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
                proc.wait(timeout=3)
            hard = True
        except Exception:
            hard = False                      # 进程句柄异常时退回优雅取消
    _ACTIVE_PROCS.pop(tid, None)
    if hard:
        # 强杀成功 → 立即落终态，不等 watcher（worker 死在阶段中间时
        # 状态文件不会有终态，只靠 watcher 会留下僵尸窗口）
        with _lock:
            tasks = _load_tasks()
            cur = _find(tasks, tid)
            if cur and cur.get("state") == "running":
                _snapshot(cur, "cancelled")
                cur["msg"] = "已按用户请求即时终止"
                _save_tasks(tasks)
        return {"ok": True, "msg": "已即时终止 worker 进程", "hard": True}
    return {"ok": True, "msg": "取消标志已置位，将在当前阶段结束后停止", "hard": False}


@app.get("/api/tasks/{tid}/stream")
async def api_stream(tid: str, request: Request, pw: str = ""):
    _check_auth(request, pw)
    tasks = _load_tasks()
    t = _find(tasks, tid)
    if not t:
        raise HTTPException(404, "任务不存在")

    def gen():
        offset = 0
        last_state = None
        import time as _t
        deadline_idle = 0
        while True:
            tasks = _load_tasks()
            cur = _find(tasks, tid)
            if cur is None:
                yield f"data: {json.dumps({'type': 'end', 'state': 'gone'})}\n\n"
                return
            # 1) 推送新追加的进度日志行
            try:
                with open(PROGRESS_FILE, encoding="utf-8") as f:
                    f.seek(offset)
                    lines = f.readlines()
                    offset = f.tell()
            except OSError:
                lines = []
            for ln in lines:
                try:
                    ev = json.loads(ln)
                    yield f"data: {json.dumps({'type': 'log', **ev}, ensure_ascii=False)}\n\n"
                except Exception:
                    pass
            # 2) 推送状态变化
            tasks = _load_tasks()
            cur = _find(tasks, tid)
            state = cur["state"]
            if state != last_state:
                payload = {"type": "state", "state": state,
                           "msg": cur.get("msg"),
                           "hallucination_rate": cur.get("hallucination_rate")}
                if state == "running":
                    st = _read_status()
                    payload.update({"stage": st.get("stage"),
                                    "status_msg": st.get("msg")})
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_state = state
            if state != "running":
                yield f"data: {json.dumps({'type': 'end', 'state': state})}\n\n"
                return
            # worker 进程死了但注册表还没更新（watcher 1s 内会收尾），最多等20轮
            if not lines:
                deadline_idle += 1
                if deadline_idle > 20:
                    yield f"data: {json.dumps({'type': 'state', 'state': 'running', 'stage': '…'})}\n\n"
            _t.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/tasks/{tid}/result")
async def api_result(tid: str, request: Request):
    _check_auth(request)
    d = _task_dir(tid)
    if not os.path.isdir(d):
        raise HTTPException(404, "任务无产物（尚未运行完成或快照丢失）")
    out: dict = {"id": tid}
    # 只有成功任务才回产物：失败/终止任务的快照目录里可能残留
    # 上一次运行留在共享 DATA/OUTPUT 的旧文件，回传会张冠李戴
    t = _find(_load_tasks(), tid)
    if t is None or t.get("state") != "done":
        return out
    files = {"report": "final_report.md", "review_table": "review_table.md",
             "papers": "papers.json", "cards": "paper_cards.json",
             "review": "review_results.json", "conflicts": "conflicts.json",
             "evidence": "evidence_index.json", "plan": "plan.json",
             "meta": "meta.json", "glossary": "glossary.json",
             "survey_md": "survey_summary.md", "benchmark": "benchmark.json",
             "blindspots": "blindspots.md", "coverage": "coverage_check.json"}
    for key, name in files.items():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                txt = f.read()
            out[key] = json.loads(txt) if name.endswith(".json") else txt
        except Exception:
            continue
    return out


def _trim_abstract(text: str, zh_chars: int = 180, en_words: int = 30):
    """按语种截断摘要：中文按字（180 字），英文按词（30 词）。

    背景：原先统一 `abstract[:180]`，对英文论文等于只给 180 个字母（约 30 个词
    且常在单词中间截断），既过短又不美观。英文改为按词数截断，词边界干净。
    返回 (截断文本, 单位标签)，单位标签供前端展示"前 180 字 / 前 30 词"。
    """
    s = (text or "").strip()
    if not s:
        return "", "字"
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    # 中文按字数：CJK 字符有一定数量，或占比过半
    if cjk >= 10 or cjk * 2 >= len(s):
        return s[:zh_chars], "字"
    words = s.split()
    return " ".join(words[:en_words]), "词"


@app.get("/api/tasks/{tid}/bridge")
def api_bridge(tid: str, request: Request):
    """结果中转页数据：智能引导在主题尾部注入「重点关注：…」关键词，
    此端点解析这些关键词并计算每篇论文的命中关系（标题/摘要），
    供前端中转页展示「引导词 ↔ 论文」关联并做分流选择"""
    _check_auth(request)
    t = _find(_load_tasks(), tid)
    if t is None:
        raise HTTPException(404, "任务不存在")
    if t.get("state") != "done":
        raise HTTPException(404, "任务尚未完成")
    d = _task_dir(tid)
    if not os.path.isdir(d):
        raise HTTPException(404, "任务无产物")

    # 1) 解析引导关键词：优先用任务记录里的中英对照（智能引导传入）；
    #    旧任务退回从主题串解析——注意主题可能堆叠多段「重点关注：…」（旧版
    #    向导重复使用会把上次拼接结果带回输入框），全部解析并去重
    topic = t.get("topic", "")
    raw_focus = t.get("focus_keywords") or []
    if raw_focus:
        kws = [{"zh": str(k.get("zh") or "").strip(),
                "en": str(k.get("en") or "").strip()}
               for k in raw_focus if isinstance(k, dict)]
    else:
        kws, seen = [], set()
        for part in re.findall(r"[（(]重点关注[:：](.+?)[）)]", topic):
            for k in re.split(r"[；;]", part):
                k = k.strip()
                if k and k not in seen:
                    seen.add(k)
                    kws.append({"zh": k, "en": ""})

    # 2) 读取任务产物（缺哪个文件就降级为空，不让中转页整体失败）
    def _load(name):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    meta = _load("meta.json") or []
    cards = _load("paper_cards.json") or []
    plan = _load("plan.json") or {}
    card_map = {c.get("arxiv_id"): c for c in cards if isinstance(c, dict)}
    # 声明文本（中文）：中文关键词即使命中不了英文标题/摘要，也可能命中声明
    claims_text = {pid: " ".join(str(c.get("content") or "") for c in card.get("claims") or [])
                   for pid, card in card_map.items()}

    papers, cov = [], {i: 0 for i in range(len(kws))}
    for p in meta:
        if not isinstance(p, dict):
            continue
        title, abstract = p.get("title") or "", p.get("abstract") or ""
        tl, al = title.lower(), abstract.lower()
        cl = claims_text.get(p.get("arxiv_id"), "")
        hits = []
        for i, kw in enumerate(kws):
            # 中英任一形式命中即算（中文关键词也能通过英文术语命中英文论文）
            zh_t = kw["zh"] and kw["zh"].lower() in tl
            zh_a = kw["zh"] and kw["zh"].lower() in al
            en_t = kw["en"] and kw["en"].lower() in tl
            en_a = kw["en"] and kw["en"].lower() in al
            in_t, in_a = zh_t or en_t, zh_a or en_a
            # 声明命中：关键词出现在该论文已提取的中文声明里（第3通道，不计分）
            in_c = bool(kw["zh"]) and kw["zh"].lower() in cl.lower()
            if in_t or in_a or in_c:
                hits.append({"kw": kw["zh"] or kw["en"], "en": kw["en"],
                             "in_title": in_t, "in_abstract": in_a,
                             "in_claims": in_c})
                cov[i] += 1
        card = card_map.get(p.get("arxiv_id"), {})
        abs_txt, abs_unit = _trim_abstract(abstract)   # 中文180字 / 英文30词
        papers.append({
            "arxiv_id": p.get("arxiv_id"),
            "title": title,
            "authors": (p.get("authors") or [])[:4],
            "year": p.get("year"),
            "cited_by": p.get("cited_by", 0),
            "category": card.get("method_category"),
            "claims_n": len(card.get("claims") or []),
            "matches": hits,
            # 命中得分：标题命中 3 分/个，摘要命中 1 分/个
            "score": sum(3 if h["in_title"] else 1 for h in hits),
            "abstract": abs_txt,
            # 前端据此显示"摘要（前 180 字 / 前 30 词）"，避免英文显示错单位
            "abstract_label": f"前 {180 if abs_unit == '字' else 30} {abs_unit}",
            # 仅真正截断时才追加省略号（短摘要保持原样）
            "abstract_truncated": len(abs_txt) < len(abstract.strip()),
        })
    papers.sort(key=lambda x: (-x["score"], -(x["cited_by"] or 0)))
    return {
        "topic": topic,
        "base_topic": re.sub(r"[（(]重点关注[:：].*?[）)]", "", topic).strip(),
        "focus_keywords": [k["zh"] or k["en"] for k in kws],
        "queries": plan.get("keywords") or [],
        "coverage": [{"kw": k["zh"] or k["en"], "en": k["en"],
                      "n": cov[i], "total": len(papers)}
                     for i, k in enumerate(kws)],
        "papers": papers,
    }


@app.get("/api/tasks/{tid}/log")
async def api_log(tid: str, request: Request, tail: int = 150):
    """返回任务 worker 的运行日志尾部（失败诊断用）"""
    _check_auth(request)
    p = os.path.join(_task_dir(tid), "run.log")
    if not os.path.exists(p):
        raise HTTPException(404, "无运行日志（容器重启后磁盘会重置）")
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        raise HTTPException(404, "日志读取失败")
    n = max(10, min(tail, 500))
    return {"log": "".join(lines[-n:]), "total_lines": len(lines)}


@app.get("/api/tasks/{tid}/graph")
async def api_graph(tid: str, request: Request):
    _check_auth(request)
    p = os.path.join(_task_dir(tid), "citation_graph.html")
    if not os.path.exists(p):
        raise HTTPException(404, "该任务没有引用图谱")
    return FileResponse(p, media_type="text/html")


@app.get("/api/pdf/{arxiv_id}")
async def api_pdf(arxiv_id: str, request: Request):
    _check_auth(request)
    safe = os.path.basename(arxiv_id).replace(".pdf", "")
    p = os.path.join(config.PAPER_DIR, safe + ".pdf")
    if not os.path.exists(p):
        raise HTTPException(404, "PDF 不存在（容器重启后 papers/ 会清空）")
    return FileResponse(p, media_type="application/pdf")


# ================================================================
# 老页面功能合并（app.py → FastAPI）：问答 / AI助手 / 综述总结 /
# 小白解读 / 证据穿透 / PDF全文搜索 / Word·LaTeX导出 / Benchmark
# ================================================================
EVIDENCE_DIR = os.path.join(BASE, "evidence")

# qa_agent 等模块固定读 DATA_DIR 共享文件；查看历史任务前先把该任务
# 的快照拷回 DATA_DIR（单活跃任务架构：有任务运行时禁止切换数据源）
def _no_running() -> str | None:
    act = _active(_load_tasks())
    return (f"已有任务在运行（{act['topic'][:20]}…），结束后才能切换"
            "历史任务数据" if act else None)


def _stage_task_files(tid: str) -> None:
    d = _task_dir(tid)
    for src, dst in (("paper_cards.json", "paper_cards.json"),
                     ("review_results.json", "review_results.json"),
                     ("evidence_index.json", "evidence_index.json"),
                     ("meta.json", "papers_meta.json")):
        p = os.path.join(d, src)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(DATA, dst))


def _read_json(p: str, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _read_text(p: str) -> str:
    try:
        with open(p, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _guard_task(tid: str) -> None:
    if not os.path.isdir(_task_dir(tid)):
        raise HTTPException(404, "任务不存在（或快照已被清理）")


# ---- 可信问答（Evidence-Bounded QA） ----
@app.get("/api/tasks/{tid}/pool")
async def api_pool(tid: str, request: Request):
    """声明池（问答的知识边界）。读取前把该任务快照拷回 DATA_DIR"""
    _check_auth(request)
    _guard_task(tid)
    blocked = _no_running()
    if blocked:
        raise HTTPException(409, blocked)
    _stage_task_files(tid)
    from agents import qa_agent
    return {"pool": qa_agent.build_claim_pool()}


@app.post("/api/tasks/{tid}/qa")
async def api_qa(tid: str, request: Request):
    _check_auth(request)
    _guard_task(tid)
    blocked = _no_running()
    if blocked:
        raise HTTPException(409, blocked)
    body = await request.json()
    q = (body.get("question") or "").strip()
    if not q:
        raise HTTPException(400, "question 不能为空")
    _stage_task_files(tid)
    from agents import qa_agent
    ans, pool = qa_agent.answer_question(q, body.get("history"))
    return {"can_answer": ans.can_answer, "answer": ans.answer,
            "used_claims": ans.used_claims, "missing": ans.missing,
            "pool": pool}


@app.post("/api/tasks/{tid}/refusal-test")
async def api_refusal(tid: str, request: Request):
    """拒答能力自检（逐条 NDJSON 流式返回，避免长请求被网关掐断）"""
    _check_auth(request)
    _guard_task(tid)
    blocked = _no_running()
    if blocked:
        raise HTTPException(409, blocked)
    _stage_task_files(tid)

    def gen():
        from agents import qa_agent
        pool = qa_agent.build_claim_pool()
        if not pool:
            yield json.dumps({"type": "error", "msg": "声明池为空，无法测试"},
                             ensure_ascii=False) + "\n"
            return
        cases = []
        for c in pool[:3]:
            cases.append({"q": f"关于{c['paper_title'][:30]}这篇论文，"
                               f"{c['content'][:50]}... 请具体说明它的相关情况",
                          "expect": True})
        method = next((c for c in pool if c["type"] == "method"), None)
        if method:
            short = method["paper_title"].split(":")[0].strip()
            cases.append({"q": f"{short}的核心方法是什么？", "expect": True})
        for q in ("Transformer的自注意力机制是怎么计算的？",
                  "CRISPR基因编辑技术的原理是什么？",
                  "量子计算中的qubit和经典bit的本质区别是什么？",
                  "美联储加息对汇率的影响机制是什么？",
                  "明朝郑和下西洋的路线经过哪些地区？",
                  "黑洞的信息悖论目前有哪些候选解决方案？",
                  "白癜风的主要临床治疗手段有哪些？",
                  "国际法中的领海宽度是如何确定的？",
                  "碳交易市场的定价机制是怎样运作的？",
                  "古希腊城邦制度的典型特征是什么？"):
            cases.append({"q": q, "expect": False})
        correct = 0
        details = []
        for t in cases:
            try:
                ans, _ = qa_agent.answer_question(t["q"])
            except Exception as e:
                d = {"question": t["q"][:60],
                     "expect_answerable": t["expect"],
                     "actual_answerable": None, "correct": False,
                     "answer_preview": f"调用失败: {e}"}
            else:
                ok = ans.can_answer == t["expect"]
                correct += ok
                d = {"question": t["q"][:60],
                     "expect_answerable": t["expect"],
                     "actual_answerable": ans.can_answer, "correct": ok,
                     "answer_preview": ans.answer[:80]}
            details.append(d)
            yield json.dumps({"type": "case", **d}, ensure_ascii=False) + "\n"
        na = max(sum(1 for t in cases if t["expect"]), 1)
        nr = max(len(cases) - na, 1)
        yield json.dumps({"type": "final",
                          "answerable_acc": sum(
                              1 for d in details
                              if d["expect_answerable"] and d["correct"]) / na,
                          "refusal_acc": sum(
                              1 for d in details
                              if not d["expect_answerable"] and d["correct"])
                          / nr}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---- 生成类（流式文本） ----
_PLAIN_SYS = """你是一位翻译高手，擅长把学术文献综述逐句转述成大白话——像把一篇文言文翻译成白话文，内容一字不增一字不减，只是语言变通俗。

核心原则（决定成败，逐条遵守）:
1. **忠实转译，不从零讲课**: 报告讲什么你就讲什么，按报告原有的结构和顺序走。禁止添加报告之外的学科背景、历史铺垫、入门教学——读者要的是'看懂这份报告'，不是'学这门课'
2. **每个艰深句子都要翻成人话**: 原报告里读者可能看不懂的句子（术语密集的、逻辑复杂的），必须换成日常语言重新表达，意思严格不变。
   - 例: 原文'该方法采用代理梯度替代不可微的脉冲激活函数' → 翻译'这个方法的做法是：因为脉冲这种'放电行为'没法直接算数学梯度，他们就找个行为相似的平滑函数来代替它算'
3. **术语处理要轻**: 术语首次出现时在括号里用一句话点明它指什么即可（如: 代理梯度（就是替脉冲放电行为找的'数学替身'）），不展开教学，不单独成段
4. **数据结论引用全保留**: 所有数字、结论、论文引用信息原样保留，一个都不能丢——这是报告的核心价值
5. **类比是调味料不是主菜**: 只在难懂的原理处用一句类比帮理解，禁止大段故事化铺垫
6. markdown输出，小节标题跟随原报告但可口语化；文末加一小段'一句话总结这份报告'"""

_SURVEY_SYS = """你是资深学术综述撰稿人兼领域分析师。{topic_line}基于给定的论文清单（[编号]为引用标识），输出两部分内容。

【第一部分: 叙述性综述】
- 用连贯的叙述性文字（禁止表格、禁止罗列式堆砌）梳理研究图景
- 按脉络组织: 研究背景与动机 → 现有工作按思路分派 → 各派代表工作与核心思想 → 发展趋势
- 只能使用给定论文的信息，不得编造未提供的文献、数据或结论；提及具体工作时标注[编号]（600-900字）

【第二部分: AI观点（仅供参考）】
- 对检索出的论文做跨文献归纳偏置分析，输出3-5条观点
- 每条观点必须遵循如下句式骨架（用实际归纳内容填充XX）:
  现有 XX 方法在 XX 场景下具备 XX 优势，但普遍存在 XX 局限，在 XX 条件下效果不足，目前针对该问题的相关研究较少，有待进一步探索
- 每条观点后标注依据: (支撑: [编号])；无直接支撑的推断标 (推断)
- 观点之间不重复，各自覆盖不同的方法路线或维度

输出为markdown，结构如下:
# 📖 叙述性综述
（正文）
# 💡 AI观点（仅供参考）
1. ...
2. ...
（结尾附一行提示: 以上观点为AI归纳推断，仅供参考）

论文清单如下:
{src}"""

_ASSIST_GUIDE = """【系统使用说明】
- 系统定位: 科研文献整理Agent。用户输入研究主题后自动执行五段流水线: 规划(生成检索词)→检索(下载论文PDF)→提取(信息卡片)→审查(两级防幻觉核验)→综合(综述报告)。另有证据定位(PDF原文高亮穿透)和矛盾检测(学术争议发现)两个后处理。
- 操作: 左侧输入研究主题（建议具体），选检索模式与目标篇数，点「开始综述」（约10-15分钟，运行中勿重复点击）；历史任务可从左侧任务列表随时回看。
- 🎯三模式检索(🌱入门综述/🚀前沿突破/🔀交叉领域): 通过「检索模式」选择，作为流水线的检索阶段运行；模式未命中自动降级标准检索，降级后仍未命中才终止流水线。模式检索还会为核心名词生成速查表。
- 标签页: 🔍检索结果(四种排序+徽章+四路信号分解) | 📝综述报告(可生成🍼小白解读，支持Word/LaTeX导出) | 💬可信问答(仅基于声明池、证据不足会拒答) | 🤖AI助手(本页) | 📚综述总结(叙述性综述+AI观点) | 📄信息卡片(每篇论文的声明+📎证据原文) | ✅核验明细(每条声明核验结果+Benchmark) | ⚡矛盾与张力 | 📊综述表格 | 🕸引用图谱 | 📚论文原文(PDF阅读+🔍全文搜索+📝阅读笔记+📖名词速查) | ⚔️对比阅读 | 📖新手指南。
- 关键指标: 声明=从论文摘出的可核实信息；通过核验=引用真实且无夸大；幻觉率=AI表述与论文原文不符比例（越低越可信）；可穿透证据=能定位到PDF原文高亮处的声明数。
- 徽章含义: 🔗引用链经典(被多篇种子论文共同引用) | 🔥高被引 | 🤖高度切题(LLM相关性≥4.5)。"""


def _survey_src(d: str) -> str:
    """组织综述总结的LLM输入: 论文元数据 + 通过核验的关键声明"""
    meta = _read_json(os.path.join(d, "meta.json"), []) or []
    cards = _read_json(os.path.join(d, "paper_cards.json"), []) or []
    reviews = _read_json(os.path.join(d, "review_results.json"), []) or []
    supported = {f"{r.get('arxiv_id')}#{r.get('claim_index')}" for r in reviews
                 if (r.get("verdict") or {}).get("verdict") == "supported"}
    cards_by = {c.get("arxiv_id"): c for c in cards}
    blocks = []
    for i, p in enumerate(meta, 1):
        card = cards_by.get(p.get("arxiv_id"))
        lines = [f"[{i}] 标题: {p.get('title', '')}",
                 f"    年份: {p.get('year')}  被引次数: {p.get('cited_by', 0)}"]
        ab = (p.get("abstract") or "").strip()
        if ab:
            lines.append(f"    摘要: {ab[:500]}")
        if card:
            kept = []
            for idx, cl in enumerate(card.get("claims") or []):
                key = f"{p.get('arxiv_id')}#{idx}"
                if (key in supported) if supported else True:
                    kept.append(cl.get("content", ""))
            for c in kept[:4]:
                lines.append(f"    · 关键信息: {c[:220]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _stream(gen):
    return StreamingResponse(gen, media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/tasks/{tid}/plain")
async def api_plain(tid: str, request: Request):
    """🍼 小白解读：把综述报告转述成大白话（流式）"""
    _check_auth(request)
    _guard_task(tid)
    report = _read_text(os.path.join(_task_dir(tid), "final_report.md"))
    if not report:
        raise HTTPException(404, "该任务没有综述报告")
    import llm_client

    def gen():
        yield from llm_client.chat_stream(
            [{"role": "system", "content": _PLAIN_SYS},
             {"role": "user",
              "content": "请把以下文献综述转述成大白话版（要求: 忠实转译"
                         "报告本身，不添加报告外的学科教学）:\n\n" + report}],
            temperature=0.3)

    return _stream(gen())


@app.post("/api/tasks/{tid}/survey")
async def api_survey(tid: str, request: Request):
    """✍️ 综述总结：叙述性综述 + AI观点（流式，结束后落盘到任务快照）"""
    _check_auth(request)
    _guard_task(tid)
    blocked = _no_running()
    if blocked:
        raise HTTPException(409, blocked)
    d = _task_dir(tid)
    src = _survey_src(d)
    if not src:
        raise HTTPException(404, "没有可用的论文数据（meta.json 缺失）")
    tasks = _load_tasks()
    topic = (_find(tasks, tid) or {}).get("topic", "")
    topic_line = f"研究主题: {topic}\n" if topic else ""
    import llm_client

    def gen():
        buf = []
        try:
            for chunk in llm_client.chat_stream(
                    [{"role": "system",
                      "content": "你是严谨的学术综述撰稿人，忠于给定材料。"},
                     {"role": "user",
                      "content": _SURVEY_SYS.format(topic_line=topic_line,
                                                    src=src)}],
                    temperature=0.3):
                buf.append(chunk)
                yield chunk
        finally:
            if buf:
                try:
                    with open(os.path.join(d, "survey_summary.md"), "w",
                              encoding="utf-8") as f:
                        f.write("".join(buf))
                except OSError:
                    pass

    return _stream(gen())


@app.post("/api/assistant")
async def api_assistant(request: Request):
    """🤖 AI助手：系统使用 + 论文内容 全能问答（流式）"""
    _check_auth(request)
    body = await request.json()
    tid = body.get("tid") or ""
    history = body.get("history") or []
    parts = [_ASSIST_GUIDE]
    if tid and os.path.isdir(_task_dir(tid)):
        blocked = _no_running()
        if blocked:
            raise HTTPException(409, blocked)
        _stage_task_files(tid)
        meta = _read_json(os.path.join(_task_dir(tid), "meta.json"),
                          []) or []
        if meta:
            lines = ["【当前结果概况】"]
            topic = (_find(_load_tasks(), tid) or {}).get("topic", "")
            if topic:
                lines.append(f"- 检索主题: {topic}")
            lines.append(f"- 收录论文{len(meta)}篇:")
            for p in sorted(meta, key=lambda x: x.get("final_score") or 0,
                            reverse=True)[:15]:
                lines.append(
                    f"  {p.get('title', '')[:70]} ({p.get('year')}, "
                    f"被引{p.get('cited_by', 0)}, "
                    f"相关性{p.get('relevance_score')}, "
                    f"arXiv:{p.get('arxiv_id')})")
            parts.append("\n".join(lines))
        try:
            from agents import qa_agent
            pool = qa_agent.build_claim_pool()
            if pool:
                lines = [f"【已核验声明池】(共{len(pool)}条，"
                         "回答论文内容问题时只能依据这些)"]
                for c in pool:
                    lines.append(f"[声明#{c['num']}] ({c['type']}, "
                                 f"《{c['paper_title']}》{c['year']}年)\n"
                                 f"{c['content']}")
                parts.append("\n".join(lines))
        except Exception:
            pass
        report = _read_text(os.path.join(_task_dir(tid), "final_report.md"))
        if report:
            parts.append(f"【最终综述报告全文】\n{report}")
    sys_prompt = (
        "你是'科研文献整理Agent'系统的内置AI助手。用户可能是初次使用者"
        "或想了解论文内容的研究者。\n回答规则:\n"
        "1. 【系统使用类问题】(怎么操作、指标含义、某个标签页是什么): "
        "根据下面的系统说明回答，简洁实用\n"
        "2. 【论文内容类问题】(方法、结果、结论): 只依据下面的"
        "'已核验声明池'和'最终综述报告'回答，引用标注[声明#N]；"
        "池里没有的不要编造，说明'当前结果中没有相关信息'\n"
        "3. 【两者混合】: 分别按对应规则处理\n"
        "4. 全程用中文，通俗但专业；回答末尾视情况提示用户去哪个标签页"
        "能看到更多\n\n" + "\n\n".join(parts))
    import llm_client

    def gen():
        yield from llm_client.chat_stream(
            [{"role": "system", "content": sys_prompt}]
            + [h for h in history if h.get("role") in ("user", "assistant")]
            [-6:], temperature=0.3)

    return _stream(gen())


# ---- 证据穿透 / PDF全文搜索 / 导出 ----
@app.get("/api/tasks/{tid}/evidence/{arxiv_id}/{claim_index}")
async def api_evidence(tid: str, arxiv_id: str, claim_index: int,
                       request: Request):
    """声明 -> 证据索引（页码/高亮截图URL）。图片缺失时优雅降级"""
    _check_auth(request)
    idx = _read_json(os.path.join(_task_dir(tid), "evidence_index.json"),
                     {}) or {}
    ev = idx.get(f"{arxiv_id}#{claim_index}")
    if not ev:
        return {"located": False}
    out = {"located": bool(ev.get("located")), "pdf": ev.get("pdf"),
           "page": (ev.get("page") + 1) if ev.get("page") is not None else None,
           "match_level": ev.get("match_level")}
    img = ev.get("image")
    if out["located"] and img:
        name = os.path.basename(str(img))
        if os.path.exists(os.path.join(EVIDENCE_DIR, name)):
            out["image_url"] = "/api/evidence-img/" + name
        else:
            out["image_missing"] = True
    return out


@app.get("/api/evidence-img/{name}")
async def api_evidence_img(name: str, request: Request):
    _check_auth(request)
    p = os.path.join(EVIDENCE_DIR, os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404, "证据图片不存在（容器重启后 evidence/ 会清空）")
    return FileResponse(p, media_type="image/png")


@app.get("/api/pdfs")
async def api_pdfs(request: Request):
    """本地已下载的论文PDF清单（论文原文阅读用）"""
    _check_auth(request)
    import glob as _glob
    ids = sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in _glob.glob(os.path.join(config.PAPER_DIR, "*.pdf")))
    return {"ids": ids}


@app.get("/api/search-pdfs")
async def api_search_pdfs(request: Request, q: str = "", tid: str = "",
                          limit: int = 40):
    """PDF全文搜索（PyMuPDF跨文档逐页检索，多词=同页同时出现）"""
    _check_auth(request)
    q = (q or "").strip()
    if not q:
        raise HTTPException(400, "关键词不能为空")
    try:
        import pymupdf as fitz
    except ImportError:
        raise HTTPException(501, "服务器未安装 PyMuPDF，全文搜索不可用")
    import glob as _glob
    pdfs = sorted(_glob.glob(os.path.join(config.PAPER_DIR, "*.pdf")))
    if not pdfs:
        raise HTTPException(404, "papers/ 目录下没有PDF（容器重启后会清空，"
                                 "需重新运行流水线）")
    titles = {}
    if tid:
        for p in (_read_json(os.path.join(_task_dir(tid), "meta.json"),
                             []) or []):
            titles[p.get("arxiv_id")] = p.get("title")
    terms = [t for t in q.split() if t]
    by_doc: dict = {}
    max_hits = max(10, min(limit, 80))
    for path in pdfs:
        pid = os.path.splitext(os.path.basename(path))[0]
        try:
            doc = fitz.open(path)
        except Exception:
            continue
        with doc:
            for pi in range(doc.page_count):
                text = doc[pi].get_text()
                tl = text.lower()
                if all(t.lower() in tl for t in terms):
                    pos = tl.find(terms[0].lower())
                    snippet = text[max(0, pos - 60):pos + 160].replace(
                        "\n", " ").strip()
                    by_doc.setdefault(pid, []).append(
                        {"arxiv_id": pid,
                         "title": titles.get(pid) or pid,
                         "page": pi, "snippet": snippet})
    hits = []
    for pid, hs in sorted(by_doc.items(), key=lambda kv: -len(kv[1])):
        for h in hs[:3]:
            h["doc_hits"] = len(hs)
            hits.append(h)
        if len(hits) >= max_hits:
            break
    return {"query": q, "total_docs": len(by_doc), "hits": hits[:max_hits]}


@app.get("/api/tasks/{tid}/export/{fmt}")
async def api_export(tid: str, fmt: str, request: Request):
    """导出综述报告：docx / latex(zip)。Markdown 由前端直接下载"""
    _check_auth(request)
    _guard_task(tid)
    d = _task_dir(tid)
    report = _read_text(os.path.join(d, "final_report.md"))
    if not report:
        raise HTTPException(404, "该任务没有综述报告")
    topic = (_find(_load_tasks(), tid) or {}).get("topic", "文献综述")
    meta = _read_json(os.path.join(d, "meta.json"), []) or []
    meta_lines = [f"主题: {topic}", f"收录论文: {len(meta)} 篇"]
    try:
        import exporter
        if fmt == "docx":
            buf = exporter.md_to_docx(report, title=f"研究综述: {topic}",
                                      meta_lines=meta_lines)
            return Response(buf, media_type="application/vnd.openxmlformats"
                            "-officedocument.wordprocessingml.document",
                            headers={"Content-Disposition":
                                     'attachment; filename="review.docx"'})
        if fmt == "latex":
            refs = exporter.meta_from_papers_json(meta)
            buf = exporter.build_latex_package(report, f"研究综述: {topic}",
                                               refs, meta_lines)
            return Response(buf, media_type="application/zip",
                            headers={"Content-Disposition":
                                     'attachment; filename="latex_package.zip"'})
    except Exception as e:
        raise HTTPException(500, f"导出失败: {e}")
    raise HTTPException(400, f"不支持的导出格式: {fmt}")


@app.get("/api/health")
async def api_health():
    return {"ok": True, "app_password": bool(getattr(config, "APP_PASSWORD", ""))}


class GlossaryReq(BaseModel):
    topic: str
    identity: str = "研究生"
    stage: str = "入门了解"


@app.post("/api/glossary")
def api_glossary(req: GlossaryReq, request: Request):
    """检索前引导：为主题生成词云术语卡（含中英名/相关度/类别/一句话释义）。

    复用 llm_client.chat_json 的文件缓存——同主题+身份第二次生成零成本零延迟。
    """
    _check_auth(request)
    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(400, "topic 不能为空")
    try:
        import models as _models
        import llm_client as _llm
    except Exception as e:
        raise HTTPException(500, f"LLM 客户端不可用: {e}")
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    style_map = {
        "本科生": "面向零基础本科生：优先使用生活化类比，避免未解释的行话",
        "硕士生": "面向研究生：可用标准学术词汇，概念需点到定义",
        "博士生": "面向领域博士：用高密度专业表述，直接给方法论脉络",
        "研究者": "面向研究者：精确使用领域惯用语与前沿术语",
        "工程师": "面向工程师：侧重实现与落地视角，如硬件、开销、部署",
    }
    style = style_map.get(req.identity, style_map["硕士生"])
    sys_prompt = (
        "你是学术导航助手。根据用户的研究主题与画像，生成一份主题词云术语集，帮助用户在正式检索前建立概念地图。"
        "要求：\n"
        "1. terms 提供 15~25 个术语，按四类覆盖：core(核心概念/它是什么)、method(方法技术/怎么做的)、"
        "application(应用场景/用在哪里)、background(背景知识/为什么重要)，每类都要有；\n"
        "2. weight 为该术语与主题的相关度 1~10，核心概念给 9~10，背景知识给 3~6；\n"
        "3. term_en 用领域标准英文术语，term_zh 用规范中文译名；\n"
        "4. explanation 一律填空字符串\"\"——解释由前端在用户点击术语时按需生成，"
        "此处严禁生成任何解释文本，以大幅缩短响应时间；\n"
        "5. 只输出术语本身，禁止输出任何文献引用、arXiv编号或作者名。\n"
        "术语选择的偏好视角：" + style
    )
    user_prompt = f"研究主题：{topic}\n用户阶段：{req.stage}"
    try:
        glossary = _llm.chat_json(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}],
            _models.TopicGlossary,
            temperature=0.5,
            # 词云是交互式功能（用户盯着按钮等结果），必须收紧重试上限。
            # 默认配置下最坏耗时 ≈ 3轮解析×3次重试×120秒超时+退避 ≈ 19分钟，
            # 前端会一直停在"生成中…"——正是词云"卡死"的根源。
            # 收紧后最坏 = 2轮 × (1次调用 120秒) ≈ 4分钟封顶。
            max_retries=1, parse_rounds=2,
        )
    except Exception as e:
        raise HTTPException(502, f"词云生成失败: {e}")
    return {
        "topic_en": glossary.topic_en,
        "terms": [t.model_dump() for t in glossary.terms],
    }


class TermExplainReq(BaseModel):
    topic: str
    identity: str = "硕士生"
    term_zh: str
    term_en: str = ""
    category: str = "core"


@app.post("/api/term-explain")
def api_term_explain(req: TermExplainReq, request: Request):
    """术语解释懒加载：词云请求不再生成 80 字解释（首屏 62s→约20s），
    用户点击术语时按需调用本端点生成单个解释（2~4 秒，带缓存秒回重复点击）。"""
    _check_auth(request)
    topic = (req.topic or "").strip()
    term_zh = (req.term_zh or "").strip()
    if not topic or not term_zh:
        raise HTTPException(400, "topic 与 term_zh 不能为空")
    try:
        import models as _models
        import llm_client as _llm
    except Exception as e:
        raise HTTPException(500, f"LLM 客户端不可用: {e}")
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    style_map = {
        "本科生": "面向零基础本科生：优先使用生活化类比，避免未解释的行话",
        "硕士生": "面向研究生：可用标准学术词汇，概念需点到定义",
        "博士生": "面向领域博士：用高密度专业表述，直接给方法论脉络",
        "研究者": "面向研究者：精确使用领域惯用语与前沿术语",
        "工程师": "面向工程师：侧重实现与落地视角，如硬件、开销、部署",
    }
    style = style_map.get(req.identity, style_map["硕士生"])
    sys_prompt = (
        "你是学术导航助手。用一句不超过80字的中文解释给定的学术术语，"
        "说明它是什么/怎么用，结合研究主题给出语境。" + style + "。"
        "禁止输出文献引用、arXiv编号或作者名。"
    )
    user_prompt = (f"研究主题：{topic}\n术语：{term_zh}"
                   + (f"（{req.term_en}）" if req.term_en else "")
                   + f"\n类别：{req.category}")
    try:
        r = _llm.chat_json(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}],
            _models.TermExplain,
            temperature=0.4,
            max_retries=1, parse_rounds=2,
        )
    except Exception as e:
        raise HTTPException(502, f"解释生成失败: {e}")
    return {"explanation": r.explanation}


# 静态前端（放在最后，避免吞掉 /api/*）
# 根路由显式接管并禁缓存：前端页面更新后，浏览器刷新即可拿到新版，
# 不会出现"服务已更新、页面还是旧 JS"的错位（本次中转页未显示的根因）
@app.get("/", include_in_schema=False)
async def index_no_cache():
    return FileResponse(os.path.join(STATIC, "index.html"),
                        headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.environ.get("PORT", "7860")))
