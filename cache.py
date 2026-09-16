"""
cache.py —— 检索请求缓存层（成员A·任务1）
================================
职责：把 OpenAlex / Semantic Scholar 的元数据请求结果缓存到
本地SQLite，同一请求7天内不再访问网络。

解决什么问题（2026-09实测教训）:
  - OpenAlex 免费额度 $0.1/天（约100次请求），引用链挖掘单轮
    就要发50-100个请求，当天跑第二次流水线必然烧穿额度429
  - S2 认证模式限流1 req/s，4个关键词+引用链反查一轮要2分钟+
  - 缓存命中后这些请求全部本地完成（毫秒级），且不消耗任何额度

设计要点:
  1. 缓存粒度 = 原始JSON响应（_oa_request/_s2_request的返回值），
     这两个函数是全部元数据请求的必经之路（关键词检索/引用链挖掘/
     ID反查都走它们），一处包裹全量生效
  2. 只缓存HTTP 200的成功响应；429/5xx/超时的结果绝不入库
     （否则会把"额度耗尽"的429响应体缓存住，熔断检测会失效）
  3. 请求参数顺序无关：参数dict排序后哈希，{a:1,b:2}和{b:2,a:1}
     命中同一条缓存
  4. 同时维护req_log请求日志表（为任务4"慢查询诊断"打地基）:
     每次get/put都记录 数据源/耗时/是否命中缓存/状态，跑完流水线
     一条SQL即可出诊断报告
  5. 线程安全：PDF下载用ThreadPoolExecutor并发，SQLite连接加锁

注意：LLM调用不经过这里（llm_client.py独立），缓存只覆盖检索元数据。
"""

import os
import json
import time
import sqlite3
import hashlib
import threading

# 数据库文件放在 data/ 目录（已在.gitignore中，不会提交到仓库）
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "search_cache.db")

TTL_SECONDS = 7 * 86400  # 缓存有效期：7天（论文元数据变化极慢）

_LOCK = threading.Lock()   # 保护连接的写操作（多线程并发安全）
_CONN: sqlite3.Connection | None = None  # 惰性初始化的连接


def _init() -> sqlite3.Connection:
    """惰性初始化数据库连接和表结构（首次调用时执行）"""
    global _CONN
    if _CONN is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        # check_same_thread=False: 允许多线程共用连接（写操作有锁保护）
        _CONN = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _CONN.execute("""CREATE TABLE IF NOT EXISTS cache(
            key     TEXT PRIMARY KEY,   -- 请求指纹(SHA1)
            ts      REAL,               -- 写入时间戳（判断过期）
            payload TEXT                -- 原始JSON响应（仅200成功响应）
        )""")
        # 请求日志表（任务4慢查询诊断的数据源）
        _CONN.execute("""CREATE TABLE IF NOT EXISTS req_log(
            ts         REAL,            -- 请求时间戳
            source     TEXT,            -- openalex / s2 / crossref...
            key_prefix TEXT,            -- 请求指纹前16位（排查用）
            ms         REAL,            -- 耗时（毫秒）
            status     TEXT,            -- ok / 429 / 5xx / error / expired
            from_cache INTEGER          -- 1=缓存命中 0=真实网络请求
        )""")
        _CONN.commit()
    return _CONN


# ---------------------------------------------------------------
# 缓存键生成
# ---------------------------------------------------------------
def make_key(source: str, params: dict) -> str:
    """
    生成与参数顺序无关的缓存键

    原理: sorted(params.items()) 保证 {"query":"snn","limit":10} 和
    {"limit":10,"query":"snn"} 产生相同哈希——requests库发请求时参数
    序列化顺序不保证稳定，顺序敏感的键会导致大量假未命中
    """
    raw = source + "|" + json.dumps(sorted(params.items()),
                                    ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------
# 读 / 写缓存（只有200成功响应才允许put！）
# ---------------------------------------------------------------
def get(key: str) -> dict | None:
    """
    读取缓存。命中返回缓存的dict；未命中或已过期返回None。

    过期行顺手删除（惰性清理，不需要后台任务）。
    每次读取都记一条req_log（from_cache=1命中/0未命中）。
    """
    conn = _init()
    t0 = time.perf_counter()
    row = conn.execute(
        "SELECT ts, payload FROM cache WHERE key=?", (key,)).fetchone()
    if row is not None:
        if time.time() - row[0] < TTL_SECONDS:
            _log(key, "cache", (time.perf_counter() - t0) * 1000,
                 "ok", from_cache=1)
            return json.loads(row[1])
        else:
            # 已过期: 删除过期行，视同未命中
            with _LOCK:
                conn.execute("DELETE FROM cache WHERE key=?", (key,))
                conn.commit()
            _log(key, "cache", (time.perf_counter() - t0) * 1000,
                 "expired", from_cache=0)
            return None
    _log(key, "cache", (time.perf_counter() - t0) * 1000,
         "miss", from_cache=0)
    return None


def put(key: str, payload: dict) -> None:
    """
    写入缓存。调用方必须保证只在HTTP 200成功响应后调用——
    这是本模块最重要的契约，429/5xx响应一旦入库，后续7天内
    所有相同请求都会拿到错误响应，熔断器形同虚设。
    """
    conn = _init()
    with _LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO cache(key, ts, payload) VALUES (?,?,?)",
            (key, time.time(),
             json.dumps(payload, ensure_ascii=False)))
        conn.commit()


# ---------------------------------------------------------------
# 请求日志（任务4慢查询诊断的地基）
# ---------------------------------------------------------------
def _log(key: str, source: str, ms: float, status: str,
         from_cache: int) -> None:
    """记录一次缓存查询行为（内部函数，不导出）"""
    conn = _init()
    with _LOCK:
        conn.execute(
            "INSERT INTO req_log(ts, source, key_prefix, ms, status,"
            " from_cache) VALUES (?,?,?,?,?,?)",
            (time.time(), source, key[:16], round(ms, 1), status,
             from_cache))
        conn.commit()


def log_request(source: str, key: str, ms: float, status: str) -> None:
    """
    记录一次真实网络请求（给 _oa_request/_s2_request 用）

    与_log的区别: 这里记的是"发出了网络请求"（from_cache恒为0），
    _log记的是"查了一次缓存"。诊断时两者合在一起算命中率。
    """
    _log(key, source, ms, status, from_cache=0)


# ---------------------------------------------------------------
# 诊断报告（任务4会扩展成前端面板，这里先给SQL基础版）
# ---------------------------------------------------------------
def diag_summary(days: int = 7) -> str:
    """
    返回最近N天的人话版诊断报告（markdown表格字符串）

    能回答三个问题:
      1. 哪个数据源请求最多、最慢（网络耗时）
      2. 缓存命中率高不高（省了多少请求）
      3. 失败/限流集中在哪
    """
    conn = _init()
    cutoff = time.time() - days * 86400
    rows = conn.execute("""
        SELECT source,
               COUNT(*)                          AS total,
               ROUND(AVG(ms), 0)                 AS avg_ms,
               SUM(from_cache)                   AS cache_hits,
               SUM(CASE WHEN status NOT IN
                    ('ok','miss') THEN 1 ELSE 0 END) AS failures
        FROM req_log WHERE ts > ?
        GROUP BY source ORDER BY total DESC
    """, (cutoff,)).fetchall()
    if not rows:
        return "暂无请求记录（还没跑过带缓存的流水线）"

    lines = ["| 数据源 | 请求数 | 平均耗时(ms) | 缓存命中 | 失败/限流 |",
             "|---|---|---|---|---|"]
    for source, total, avg_ms, hits, failures in rows:
        hit_n = hits or 0
        hit_rate = f"{hit_n / total:.0%}" if total else "-"
        lines.append(
            f"| {source} | {total} | {avg_ms} | "
            f"{hit_n} ({hit_rate}) | {failures or 0} |")
    return "\n".join(lines)


def stats() -> tuple[int, int]:
    """(缓存条数, 日志条数) —— 简单计数，调试用"""
    conn = _init()
    n_cache = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    n_log = conn.execute("SELECT COUNT(*) FROM req_log").fetchone()[0]
    return n_cache, n_log


# ---------------------------------------------------------------
# 单独运行的测试入口：验证缓存层自身行为
# ---------------------------------------------------------------
if __name__ == "__main__":
    # 快速自检: put -> get 命中；改key -> miss
    k1 = make_key("openalex", {"query": "snn", "limit": 10})
    k1b = make_key("openalex", {"limit": 10, "query": "snn"})  # 顺序不同
    k2 = make_key("openalex", {"query": "other", "limit": 10})
    assert k1 == k1b, "参数顺序不同应产生相同键！"

    put(k1, {"results": ["demo"]})
    assert get(k1) == {"results": ["demo"]}, "缓存读取应命中"
    assert get(k2) is None, "不同key应未命中"

    n_cache, n_log = stats()
    print(f"[缓存层自检] 通过 | 缓存{n_cache}条, 日志{n_log}条")
    print(f"[缓存层自检] 数据库位置: {_DB_PATH}")
