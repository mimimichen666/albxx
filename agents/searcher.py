"""
searcher.py —— 检索Agent
================================
职责（三段式管线）：
  1. search()      : 检索论文元数据（OpenAlex主源，arXiv/S2双兜底）
  2. filter_papers(): LLM按筛选标准给摘要打相关性分，粗筛
  3. download()    : 并行下载PDF到本地 papers/ 目录

【数据源架构（重要的工程决策记录）】
  实测本机网络环境（2026-09-03）:
    - api.semanticscholar.org —— 认证密钥已下发(2026-09-06)，
      1次/秒独享额度，稳定可达
    - api.openalex.org   —— 稳定可达但免费额度仅$0.1/天(约100次
      请求)，耗尽后429持续到UTC午夜，只适合低量请求
    - export.arxiv.org   —— 时通时断（间歇性阻断）
  因此采用【S2主源 + OpenAlex引用链专用 + arXiv兜底】架构
  （2026-09-06调整，此前是OpenAlex主源，但每天跑1-2次流水线
  就烧穿额度导致429卡死）:
    主源: Semantic Scholar认证模式（稳定1req/s，元数据质量高，
          自带citationCount被引数）
    专用: OpenAlex只承担引用链挖掘+经典注入（它的referenced_works
          引用图谱是独有能力，检索主任务不再依赖它；额度耗尽时
          熔断器自动跳过这两路，流水线不受阻）
    兜底: arXiv官方API（S2不可用时的最后防线）
    下载: 优先OA PDF直链，无直链时回落到 arxiv.org/pdf/{id}

【OpenAlex额度坑（2026-09-04实测教训，重要！）】
  OpenAlex免费额度远比想象的小: $0.1/天(约100次检索请求)，
  用尽后持续429直到UTC午夜(北京时间次日8点)才重置。
  引用链挖掘单次运行就要发50-100个请求，当天多跑几次流水线
  必然烧穿额度。额度耗尽型429的响应体特征:
    {"message":"Insufficient budget...Resets at midnight UTC"}
  对这种429做退避重试毫无意义(要等十几个小时)，必须熔断——
  识别后全局跳过OpenAlex，立即切换arXiv/S2兜底。

输入: SearchPlan（规划Agent的产出）
输出: list[PaperMeta]（已下载、已过筛的论文列表）
"""

import os
import re
import time
import json

import requests

import llm_client
import config
import cache
from models import SearchPlan, PaperMeta

# 相关性评分的输出结构（局部使用，放这里就够了，不污染models.py）
from pydantic import BaseModel, Field


class RelevanceScores(BaseModel):
    """LLM对一批论文摘要的批量打分结果"""
    scores: list[float] = Field(
        description="与每篇论文摘要对应的relevance得分列表(1-5分，浮点数)，"
                    "顺序与输入的论文列表一致，长度必须相等"
    )


# Semantic Scholar API 配置（主数据源）
# citationCount: 被引数（影响力加成信号，替代OpenAlex的cited_by_count）
S2_ROOT = "https://api.semanticscholar.org/graph/v1"
S2_API_BASE = f"{S2_ROOT}/paper/search"
# 引用链兜底用的另外两个端点（成员A·P1-2修复）:
#   search/match  标题->S2 paperId 精确解析
#   paper/{id}/references  参考文献列表（OpenAlex缺referenced_works时）
S2_MATCH_BASE = f"{S2_ROOT}/paper/search/match"
S2_FIELDS = "title,abstract,year,authors,externalIds,openAccessPdf,citationCount"

# arXiv官方API配置
ARXIV_API_BASE = "http://export.arxiv.org/api/query"

# OpenAlex API配置（主数据源）
OPENALEX_API_BASE = "https://api.openalex.org/works"
# 礼貌池: 在User-Agent里附上联系方式可进入优先队列（官方推荐做法）
OPENALEX_UA = "literature-agent/0.1 (mailto:course-project@example.com)"

# Crossref API配置（第二主源，成员A·任务2）
# 免密钥、无每日额度限制（礼貌池模式稳定直连），作用:
#   S2(认证限流)和OpenAlex($0.1/天额度)都不可用时接管检索，
#   分摊两个主源的压力。注意: arXiv的DOI走DataCite注册局，
#   Crossref里查不到arXiv预印本——检索到的是期刊版本，
#   PDF下载依赖流水线已有的arXiv标题救援机制补链。
CROSSREF_API_BASE = "https://api.crossref.org/works"
# Crossref礼貌池: UA附邮箱即可（与OpenAlex同款做法）
CROSSREF_UA = "literature-agent/0.1 (mailto:course-project@example.com)"


# ---------------------------------------------------------------
# 第1段-A：OpenAlex检索（主数据源）
# ---------------------------------------------------------------
# 熔断器全局状态: 当天OpenAlex额度耗尽时置True，本进程内所有
# OpenAlex请求直接短路返回None（上层自动切换arXiv/S2兜底），
# 避免对"要等十几小时才重置"的429做无意义的退避重试
_OA_QUOTA_EXHAUSTED = False
# 连续/累计429计数（不限原因的风暴熔断）:
# 2026-09-08实测: 引用链挖掘跳过后，经典注入等调用点仍会对OpenAlex
# 发请求，每个请求卡满4次退避重试(95秒)——即使不是"额度耗尽"型429
# (如共享限流池打满)，风暴本身已让 enrichment 路失去性价比。
# 累计429达到阈值即熔断，主检索(S2)不受影响。
_OA_429_COUNT = 0
_OA_429_TRIP_THRESHOLD = 6  # 累计6次429即熔断（约2-3个请求的重试消耗）

# OpenAlex风暴熔断（时间型，2026-09-14实测教训）:
# 旧设计把"瞬时429风暴"也焊死到进程结束（复用额度耗尽标志），
# 但共享限流池打满通常几分钟就恢复——只有"Insufficient budget"
# 额度耗尽才值得永久熔断。仿照S2加冷却式熔断: 到期自动放行试探。
_OA_BREAKER_UNTIL = 0.0     # 风暴熔断到期时间戳（当前时间<它则短路）
_OA_BREAKER_COOLDOWN = 300  # 熔断冷却5分钟（到期自动恢复）

# ---------------------------------------------------------------
# S2熔断器全局状态（2026-09-09实测教训）:
# S2共享基础设施过载时，即使认证模式+合规节奏也会持续429，
# 旧逻辑每个关键词都卡满3次退避(10/20/30秒)才放弃，4个关键词
# 最坏白等4分钟。仿照OpenAlex加风暴熔断: 连续429达到阈值后
# 短路一段时间（冷却后自动恢复，S2抖动通常几分钟内好转，
# 不能像OpenAlex额度那样熔断一整天）。
_S2_429_COUNT = 0          # 连续429计数（成功一次即清零）
_S2_BREAKER_UNTIL = 0.0    # 熔断到期时间戳（0=未熔断；当前时间<它则短路）
_S2_BREAKER_TRIP = 3       # 连续3次429即熔断
_S2_BREAKER_COOLDOWN = 300 # 熔断冷却5分钟（到期自动放行试探）


def _check_oa_quota_error(resp: requests.Response) -> bool:
    """
    识别"额度耗尽"型429（区别于普通的瞬时限流）

    判据（OpenAlex实测响应）:
      - 响应体含 "Insufficient budget" 或 "Resets at midnight UTC"
      - Retry-After头 > 3600秒（普通限流通常几十秒）
    返回True表示额度耗尽，应触发熔断
    """
    try:
        body = resp.text[:300]
    except Exception:
        return False
    if ("Insufficient budget" in body
            or "Resets at midnight UTC" in body):
        return True
    retry_after = resp.headers.get("Retry-After", "")
    if retry_after.isdigit() and int(retry_after) > 3600:
        return True
    return False


def _oa_request(params: dict, max_retries: int = 4) -> dict | None:
    """
    带重试的OpenAlex请求（429自适应退避 + 额度熔断）

    退避策略（2026-09-04实测教训）:
      引用链挖掘单轮发30-40个请求，触发OpenAlex 429限流后，
      固定2秒的重试不仅缓不过来还会加重限流。429/5xx按
      5→15→30→45秒指数退避，给限流窗口足够的冷却时间；
      4xx参数错误立即放弃（重试无意义）。

    熔断策略（2026-09-04晚实测教训）:
      OpenAlex免费额度仅$0.1/天(约100次请求)，耗尽后429持续到
      UTC午夜。这种429响应体带"Insufficient budget"，重试要等
      十几小时——识别后置全局熔断标志，后续所有OpenAlex请求
      直接短路，流水线立即切换arXiv/S2兜底继续跑。

    返回:
        解析后的JSON dict，彻底失败返回None
    """
    global _OA_QUOTA_EXHAUSTED, _OA_429_COUNT, _OA_BREAKER_UNTIL

    # ---- 缓存层（成员A·任务1）----
    # 先查缓存再谈额度/熔断：命中则完全绕过网络和OpenAlex计数器，
    # 这正是缓存的核心价值——额度耗尽/熔断当天，已缓存过的请求
    # 照样秒回，流水线第二次运行不再烧任何请求
    _ck = cache.make_key("openalex", params)
    _hit = cache.get(_ck)
    if _hit is not None:
        # 命中提示: 缓存本是静默生效的，不打印的话使用者无法感知
        # 它在工作（2026-09-09实测反馈"观感不明显"的根源）
        print(f"[检索Agent] 缓存命中(openalex): "
              f"{str(params.get('filter') or params)[:50]}")
        return _hit

    if _OA_QUOTA_EXHAUSTED:
        return None  # 熔断中：当天额度已耗尽，直接走兜底数据源

    # ---- 风暴熔断检查（冷却式: 到期自动恢复，2026-09-14）----
    if time.time() < _OA_BREAKER_UNTIL:
        remain = int(_OA_BREAKER_UNTIL - time.time())
        print(f"[检索Agent] OpenAlex风暴熔断中(累计429已达阈值，"
              f"冷却剩{remain}秒)，本请求跳过走兜底数据源")
        return None

    # 退避表(秒): 3→8→15→30（2026-09-14从5/15/30/45缩短——
    # S2熔断后OpenAlex接管主检索时，旧表单请求最坏卡95秒拖垮
    # 整个检索阶段；OpenAlex限流窗口实测几十秒内，首次3秒够用）
    backoffs = [3, 8, 15, 30]
    for attempt in range(max_retries):
        try:
            _t0 = time.perf_counter()
            resp = requests.get(
                OPENALEX_API_BASE, params=params,
                headers={"User-Agent": OPENALEX_UA}, timeout=30,
            )
            resp.raise_for_status()
            _OA_429_COUNT = 0  # 请求成功: 重置风暴计数
            payload = resp.json()
            # 只缓存200成功响应（cache.put的契约）——429/5xx走异常
            # 分支，永远到不了这里，熔断检测不会被污染
            cache.put(_ck, payload)
            cache.log_request("openalex", _ck,
                              (time.perf_counter() - _t0) * 1000, "ok")
            return payload
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 429 and e.response is not None and \
                    _check_oa_quota_error(e.response):
                # 额度耗尽型429: 熔断，不再重试（重试要等到UTC午夜）
                _OA_QUOTA_EXHAUSTED = True
                cache.log_request("openalex", _ck,
                                  (time.perf_counter() - _t0) * 1000,
                                  "429_quota")
                print("[检索Agent] OpenAlex当日额度已耗尽($0.1/天)，"
                      "熔断并切换arXiv/S2兜底（北京时间明早8点重置）")
                return None
            if code == 429 or code >= 500:
                if code == 429:
                    _OA_429_COUNT += 1  # 风暴计数
                    # ---- 计数达标立即熔断，不再睡完本次退避 ----
                    # （旧逻辑要等下一轮循环顶才检查，白等一次退避）
                    if _OA_429_COUNT >= _OA_429_TRIP_THRESHOLD:
                        _OA_BREAKER_UNTIL = (time.time()
                                             + _OA_BREAKER_COOLDOWN)
                        cache.log_request("openalex", _ck,
                                          (time.perf_counter() - _t0) * 1000,
                                          "429_storm")
                        print(f"[检索Agent] OpenAlex 429风暴(累计"
                              f"{_OA_429_COUNT}次)，冷却式熔断"
                              f"{_OA_BREAKER_COOLDOWN // 60}分钟并切换"
                              f"兜底（到期自动恢复）")
                        return None
                cache.log_request("openalex", _ck,
                                  (time.perf_counter() - _t0) * 1000,
                                  str(code))
                wait = backoffs[min(attempt, len(backoffs) - 1)]
                print(f"[检索Agent] OpenAlex限流/服务异常({code})，"
                      f"退避{wait}秒后重试(第{attempt + 1}次)")
                time.sleep(wait)
            else:
                # 400/403/404等参数类错误，重试无意义
                cache.log_request("openalex", _ck,
                                  (time.perf_counter() - _t0) * 1000,
                                  str(code))
                print(f"[检索Agent] OpenAlex请求错误({code})，放弃: "
                      f"{str(e)[:80]}")
                return None
        except Exception as e:
            cache.log_request("openalex", _ck,
                              (time.perf_counter() - _t0) * 1000, "error")
            print(f"[检索Agent] OpenAlex请求异常(第{attempt + 1}次): {e}")
            time.sleep(2)
    return None


def _openalex_search(kw: str, limit: int, year_from: int | None = None,
                     sort_mode: str = "cited",
                     collect_citations: bool = True
                     ) -> list[PaperMeta]:
    """
    调用OpenAlex检索单个关键词（支持双路排序）

    检索策略（与arXiv的关键差异）:
      - 用 title_and_abstract.search 收紧匹配（默认的search全文搜索太宽泛,
        "SNN训练"能匹配出AlphaFold这种毫不相关的论文）
      - open_access.is_oa=true 只留有OA全文的（后续步骤必须要PDF）
      - 只检索arXiv托管的论文（下载成功率保障，见run()的说明）
      - sort_mode 双路:
          "cited"     -> cited_by_count:desc  影响力路（经典论文）
          "relevance" -> relevance_score:desc 相关性路（主题精准匹配）
        两路由search()做RRF融合，单一被引排序会埋没新经典论文
      - 时间范围: 传入year_from时用from_publication_date过滤

    返回:
        PaperMeta列表(cited_by已填充); 网络失败返回空列表（由上层切换兜底数据源）
    """
    # 从筛选条件里解析年份下限（如"近5年"），解析失败不设限
    if year_from is None:
        year_from = 0  # 由上层调用处决定，这里默认不过滤

    filters = [
        # 关键词同样要清洗(P1-1漏网点, 2026-09-14实据: req_log出现
        # 4次400来自本处)——planner生成的搜索词若带逗号
        # ("ChatGPT, GPT-4"式)，逗号被解析成过滤器分隔符导致400，
        # 且双路排序对同一关键词请求2次 -> 400也成对出现。
        # _oa_title_query把逗号等分隔符替换为空格，词袋匹配不受影响
        f"title_and_abstract.search:{_oa_title_query(kw)}",
        "open_access.is_oa:true",
        # 只检索arXiv托管的论文（source ID=S4306400194）——
        # 数量保障的最终解(实测2026-09-03): 出版社直链403/HTML是
        # 论文数量流失的主因，限定arXiv来源后下载成功率接近100%，
        # "目标15篇只到手8篇"的问题从根源上解决。
        # 代价: 检索池缩小为预印本(正式版也多在arXiv有副本，CS领域损失很小)
        "primary_location.source.id:S4306400194",
    ]
    if year_from >= 2000:  # 合理的年份下限才加过滤
        filters.append(f"from_publication_date:{year_from}-01-01")

    sort = ("cited_by_count:desc" if sort_mode == "cited"
            else "relevance_score:desc")
    data = _oa_request({
        "filter": ",".join(filters),
        "per_page": min(limit, 50),
        "sort": sort,
    })
    if data is None:
        return []

    papers = []
    for work in data.get("results", []):
        title = work.get("title") or ""
        # OpenAlex的摘要是倒排索引格式,需要重组为正常文本
        abstract = _oa_abstract_to_text(work.get("abstract_inverted_index"))
        if not title or not abstract:
            continue

        # arXiv编号: 从locations的arxiv仓库或doi里提取
        arxiv_id = _oa_extract_arxiv_id(work)

        # ---- 下载地址选择策略（实测教训 2026-09-03）----
        # 出版社直链(science.org/IEEE等)对脚本请求普遍403/404反爬,
        # 而arxiv.org对脚本友好且稳定。因此:
        #   有arXiv副本的论文 -> 强制走 arxiv.org/pdf/{id}
        #   纯非arXiv论文   -> 才用OpenAlex提供的OA直链
        if _looks_like_arxiv_id(arxiv_id):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        else:
            oa_loc = work.get("best_oa_location") or {}
            pdf_url = oa_loc.get("pdf_url") or oa_loc.get("landing_page_url")
            if not pdf_url:
                continue  # 既无arXiv副本又无OA链接，无法进入后续流程

        authors = [a["author"]["display_name"]
                   for a in (work.get("authorships") or [])[:5]]

        papers.append(PaperMeta(
            arxiv_id=arxiv_id,  # 非arXiv论文是OpenAlex ID(W开头)
            title=title,
            authors=authors,
            year=work.get("publication_year") or 0,
            abstract=abstract,
            pdf_url=pdf_url,
            cited_by=(work.get("cited_by_count") or 0)
            if collect_citations else 0,
            openalex_id=(work.get("id") or "").split("/")[-1],  # 引用链挖掘用
        ))
    return papers


def _looks_like_arxiv_id(pid: str) -> bool:
    """
    判断ID是否是真arXiv编号（而非OpenAlex的W开头ID）

    arXiv编号两种格式:
      新式: 2302.00232 (YYMM.NNNNN)
      旧式: cs.NE/0309021 等
    """
    import re
    if not pid:
        return False
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", pid):
        return True
    # 旧式分类编号: 字母.字母/数字
    if re.fullmatch(r"[a-z-]+(\.[A-Z]{2})?/\d{7}", pid):
        return True
    return False


def _oa_abstract_to_text(inverted_index: dict | None) -> str:
    """
    OpenAlex的摘要以倒排索引存储（词 -> 出现位置列表），
    因版权原因不直接提供纯文本。这里重组为正常句子。

    格式示例: {"hello": [0], "world": [1]} -> "hello world"
    """
    if not inverted_index:
        return ""
    # 收集 (位置, 词) 对
    positions = []
    for word, idx_list in inverted_index.items():
        for idx in idx_list:
            positions.append((idx, word))
    # 按位置排序后拼接
    positions.sort()
    return " ".join(w for _, w in positions)


def _oa_extract_arxiv_id(work: dict) -> str:
    """
    从OpenAlex作品里提取arXiv编号

    查找顺序: locations里的arxiv仓库 -> doi字段
    找不到时返回OpenAlex短ID（形如"W203...")，保证每篇论文有唯一标识
    """
    # 途径1: locations里arxiv仓库的landing_page_url
    for loc in (work.get("locations") or []):
        source = loc.get("source") or {}
        landing = loc.get("landing_page_url") or ""
        if "arxiv.org" in landing:
            # URL可能是 /abs/2302.00232 或 /pdf/1903.06379 两种形式
            # （旧代码只处理/abs/，遇到/pdf/会把整个URL当ID——脏数据）
            part = landing.split("/abs/")[-1].split("/pdf/")[-1]
            if part.endswith(".pdf"):  # 部分记录以 .pdf 结尾的脏格式
                part = part[:-4]
            # 去掉版本号v2（仅当开头是数字时才可能是新式编号）
            arxiv_id = part.split("v")[0] if part[:1].isdigit() else part
            return arxiv_id
    # 途径2: doi形如 10.48550/arXiv.2302.00232
    doi = work.get("doi") or ""
    if "arXiv." in doi:
        return doi.split("arXiv.")[-1]
    # 兜底: OpenAlex作品ID（W开头的短ID）
    return (work.get("id") or "").split("/")[-1] or "unknown"


# ---------------------------------------------------------------
# 第1段-D：引用链挖掘（经典论文召回的杀手锏）
# ---------------------------------------------------------------
def _oa_get_work(openalex_id: str,
                 select: str = "id,referenced_works") -> dict | None:
    """
    获取单个OpenAlex作品的指定字段（引用链挖掘用）

    与_oa_request的区别: 那个查集合端点(/works?filter=...)，
    这个查单个作品端点(/works/W123)，用于拿某篇论文的参考文献列表

    同样受全局熔断器保护（额度耗尽时直接返回None）

    缓存层（成员A·任务1）: 引用链挖掘对同一批经典论文的
    referenced_works 反复查（每轮挖掘都要拿种子的参考文献列表），
    这是OpenAlex请求量的大头。命中缓存时完全不消耗每日额度，
    熔断当天引用链挖掘照样能从缓存跑通。
    """
    global _OA_QUOTA_EXHAUSTED, _OA_429_COUNT

    # ---- 缓存层：先查缓存，命中则绕过网络和熔断检查 ----
    _ck = cache.make_key("openalex_work",
                         {"id": openalex_id, "select": select})
    _hit = cache.get(_ck)
    if _hit is not None:
        print(f"[检索Agent] 缓存命中(openalex_work): {openalex_id}")
        return _hit

    if _OA_QUOTA_EXHAUSTED:
        return None
    # 风暴熔断（冷却式）: 与_oa_request共用同一个到期时间戳
    if time.time() < _OA_BREAKER_UNTIL:
        return None
    url = f"https://api.openalex.org/works/{openalex_id}"
    for attempt in range(3):
        try:
            _t0 = time.perf_counter()
            resp = requests.get(url, params={"select": select},
                                headers={"User-Agent": OPENALEX_UA},
                                timeout=30)
            resp.raise_for_status()
            _OA_429_COUNT = 0
            payload = resp.json()
            # 只缓存200成功响应（429/5xx走异常分支到不了这里）
            cache.put(_ck, payload)
            cache.log_request("openalex_work", _ck,
                              (time.perf_counter() - _t0) * 1000, "ok")
            return payload
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            cache.log_request("openalex_work", _ck,
                              (time.perf_counter() - _t0) * 1000, str(code))
            if code == 429 and e.response is not None and \
                    _check_oa_quota_error(e.response):
                _OA_QUOTA_EXHAUSTED = True
                print("[检索Agent] OpenAlex当日额度已耗尽(单作品查询)，"
                      "熔断并跳过引用链挖掘")
                return None
            if code == 429:
                _OA_429_COUNT += 1  # 纳入风暴计数（_oa_request同款逻辑）
            print(f"[检索Agent] 单作品查询错误({code})，放弃")
            return None
        except Exception as e:
            cache.log_request("openalex_work", _ck,
                              (time.perf_counter() - _t0) * 1000, "error")
            print(f"[检索Agent] 单作品查询异常(第{attempt + 1}次): {e}")
            time.sleep(1.5)
    return None


def _seed_references(seed: PaperMeta,
                     fallback_quota: list[int] | None = None) -> list[str]:
    """
    获取种子论文的参考文献ID列表（带重复记录兜底）

    OpenAlex数据质量坑（实测2026-09-04）: 同一篇论文在OpenAlex常
    有两条记录——高被引记录(来自MAG/arXiv导入)经常没有参考文献
    数据(referenced_works为空)，而低被引的重复记录(来自Crossref)
    反而有。典型: SLAYER有W2891530223(被引472,引用0条)和
    W2950415370(被引6,引用21条)两条记录。
    因此: 主记录无参考文献时，按标题反查同论文的其他记录，
    取第一条有参考文献数据的（标题必须归一化匹配，防止拿错）。

    fallback_quota: 兜底反查的剩余配额（单元素列表模拟引用传递）。
    引用链挖掘中缺引用数据的种子可能很多（实测"llm"主题12篇种子
    8篇缺失），逐个反查会放大请求量触发429限流，故限制总反查次数。
    """
    work = _oa_get_work(seed.openalex_id)
    if work is None:
        return []
    refs = work.get("referenced_works") or []
    if refs:
        return refs

    # ---- 配额检查: 兜底反查有次数上限（控制API请求量）----
    if fallback_quota is not None:
        if fallback_quota[0] <= 0:
            return []
        fallback_quota[0] -= 1

    # ---- 主记录无引用数据 -> 标题反查重复记录 ----
    data = _oa_request({
        "filter": f"title.search:{_oa_title_query(seed.title)}",
        "per_page": 10,
        "select": "id,title,referenced_works",
    })
    if data is None:
        return []
    from difflib import SequenceMatcher
    norm = _norm_title(seed.title)
    for cand in data.get("results", []):
        cand_id = (cand.get("id") or "").split("/")[-1]
        if cand_id == seed.openalex_id:
            continue
        cand_norm = _norm_title(cand.get("title") or "")
        # 归一化标题必须相同或高度相似（防止拿别的论文的参考文献）
        if cand_norm != norm and SequenceMatcher(
                None, norm, cand_norm).ratio() < 0.92:
            continue
        c_refs = cand.get("referenced_works") or []
        if c_refs:
            print(f"[检索Agent] 引用链兜底: 《{seed.title[:38]}...》"
                  f"主记录无引用数据，已切换到有引用的重复记录"
                  f"({cand_id})")
            return c_refs
    return []


def _s2_references_as_wids(seed: PaperMeta) -> list[str]:
    """
    S2引用链兜底（成员A·P1-2，2026-09-14）: OpenAlex缺参考文献时
    从Semantic Scholar取，让引用链挖掘在LLM主题复活

    【问题背景（2026-09-14日志实测）】
      ChatGPT主题13个种子11个"无参考文献数据"——2023+论文的
      OpenAlex记录普遍缺referenced_works（MAG停更后的数据黑洞，
      连兜底重复记录也救不回），引用链整段空转只剩LLM先验注入。

    【转化链路（探针实测75%覆盖率）】
      S2 references端点给的是citedPaper.externalIds，映射到
      OpenAlex W-ID分两路:
        1. MAG ID -> 直接拼W{mag}（OpenAlex继承自MAG编号）
           2023论文引用MAG覆盖低(实测4/44)
        2. ArXiv ID -> OpenAlex的doi过滤器批量映射。OpenAlex无
           ids.arxiv过滤器(400实测)，但arXiv论文在OpenAlex里
           的DOI固定为10.48550/arxiv.{id}（DataCite注册），
           doi过滤支持|批量（探针实测30个命中29个=97%）
      合计33/44≈75%的引用可参与共被引统计——足以复活引用链

    请求量: 已知arXiv编号的种子2-3次请求（references+DOI批量）；
    需标题解析的+1次match。S2限流严格，由调用方的配额控制总量。

    返回:
        短W-ID列表（可直接进ref_counts共被引统计），失败返回[]
    """
    if time.time() < _S2_BREAKER_UNTIL:
        return []  # S2熔断中（_s2_request内部也会拦，这里省日志）

    # ---- 第1步: 解析S2 paperId ----
    paper_id = None
    if _looks_like_arxiv_id(seed.arxiv_id):
        # 有真实arXiv编号: 用ARXIV:前缀直取（省一次match请求）
        paper_id = f"ARXIV:{seed.arxiv_id}"
    else:
        # 只有W-ID/标题: 走search/match精确解析
        data = _s2_request({"query": seed.title},
                           url=S2_MATCH_BASE, cache_ns="s2_match")
        if data is None:
            return []
        cands = data.get("data") or []
        if not cands:
            return []
        best = cands[0]
        # 相似度门限: match可能返回同领域相近标题的论文，
        # 拿错论文的参考文献会污染整个共被引统计
        from difflib import SequenceMatcher
        score = SequenceMatcher(
            None, _norm_title(seed.title),
            _norm_title(best.get("title") or "")).ratio()
        if score < 0.85:
            print(f"[检索Agent] S2引用链兜底: 《{seed.title[:38]}...》"
                  f"标题解析相似度{score:.2f}<0.85，放弃（防拿错论文）")
            return []
        paper_id = best.get("paperId")
    if not paper_id:
        return []

    # ---- 第2步: 拉参考文献（externalIds轻量化，500条一次拿全）----
    data = _s2_request(
        {"fields": "externalIds", "limit": 500},
        url=f"{S2_ROOT}/paper/{paper_id}/references",
        cache_ns="s2_refs")
    if data is None:
        return []
    mag_ids: list[str] = []
    arxiv_dois: list[str] = []
    for entry in data.get("data") or []:
        # citedPaper可能为null（S2库中被引记录悬空），兜底空dict
        ext = ((entry.get("citedPaper") or {}).get("externalIds") or {})
        mag = ext.get("MAG")
        if mag:
            mag_ids.append(f"W{mag}")
        elif ext.get("ArXiv"):
            # 版本号后缀去掉(2303.17580v1->2303.17580)，DOI才对得上
            aid = re.sub(r"v\d+$", "", str(ext["ArXiv"])).lower()
            arxiv_dois.append(f"10.48550/arxiv.{aid}")

    # ---- 第3步: arXiv-only引用批量映射W-ID（50/批）----
    w_ids = list(mag_ids)
    for i in range(0, len(arxiv_dois), 50):
        batch = arxiv_dois[i:i + 50]
        rd = _oa_request({
            "filter": "doi:" + "|".join(batch),
            "per_page": 50, "select": "id"})
        if rd:
            w_ids += [(w.get("id") or "").split("/")[-1]
                      for w in rd.get("results", [])]
    return w_ids


def _fetch_chain_papers(w_ids: list[str], ref_counts: dict[str, int],
                        dropped: list[str]) -> list[PaperMeta]:
    """
    按OpenAlex ID批量拉取引用链经典的元数据，构建PaperMeta列表

    过滤规则（分级设计）:
      - 无标题: 丢弃（无法识别论文）
      - 无摘要且共被引<3: 丢弃（要参加LLM粗筛，没摘要无法打分；
        共被引>=3的免检经典不需要摘要，PDF全文才是提取数据源）
      - 下载链接: 优先arXiv副本，其次OA直链——都没有时pdf_url留空，
        交给下载环节的arXiv标题反查救援(_arxiv_rescue)处理
        （老经典的OpenAlex记录常只有出版社DOI，但arXiv上有副本）
    """
    papers: list[PaperMeta] = []
    for i in range(0, len(w_ids), 50):
        batch_ids = w_ids[i:i + 50]
        data = _oa_request({
            "filter": "openalex_id:" + "|".join(batch_ids),
            "per_page": 50,
        })
        if data is None:
            continue
        for work in data.get("results", []):
            w_id = (work.get("id") or "").split("/")[-1]
            title = work.get("title") or ""
            abstract = _oa_abstract_to_text(
                work.get("abstract_inverted_index"))
            hits = ref_counts.get(w_id, 0)
            if not title:
                continue
            if not abstract and hits < 3:
                dropped.append(f"《{title[:35]}》无摘要")
                continue

            arxiv_id = _oa_extract_arxiv_id(work)
            if _looks_like_arxiv_id(arxiv_id):
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            else:
                oa_loc = work.get("best_oa_location") or {}
                # OA直链优先，没有就留空（下载时按标题走arXiv救援）
                pdf_url = (oa_loc.get("pdf_url")
                           or oa_loc.get("landing_page_url") or "")
                arxiv_id = w_id  # 用OpenAlex ID作唯一标识
                if not pdf_url:
                    dropped.append(f"《{title[:35]}》无PDF(靠arXiv救援)")
                    pdf_url = ""

            papers.append(PaperMeta(
                arxiv_id=arxiv_id,
                title=title,
                authors=[a["author"]["display_name"]
                         for a in (work.get("authorships") or [])[:5]],
                year=work.get("publication_year") or 0,
                abstract=abstract,
                pdf_url=pdf_url,
                cited_by=work.get("cited_by_count") or 0,
                openalex_id=w_id,
                chain_hits=hits,
            ))
    return papers


def _citation_chain_expand(seeds: list[PaperMeta], max_seeds: int = 12,
                           min_hits: int = 2, keep_top: int = 30
                           ) -> list[PaperMeta]:
    """
    引用链挖掘（citation chaining / 共被引分析）——召回关键词检索
    永远找不到的奠基经典论文。

    【问题背景（2026-09-04）】
      关键词检索的系统性盲区: 奠基论文的标题/摘要不含现代检索词。
      例如SpikeProp(1999)是SNN反向传播的开山之作，但它的摘要里
      绝不会出现"surrogate gradient"这种2019年才流行的术语——
      所以title_and_abstract.search永远检索不到它。
      但领域内每篇后续论文都会引用它！

    【算法（学术界发现经典的标准方法，Google Scholar的核心思想）】
      1. 取检索结果中的高影响力论文作为"种子"（按多关键词命中数+
         被引数排序，取前max_seeds篇）
      2. 拉取每篇种子的参考文献列表（OpenAlex referenced_works字段）
      3. 统计共被引次数: 被n篇种子共同引用的论文，领域经典度∝n
      4. 共被引>=min_hits的按次数降序截取keep_top篇，批量拉元数据
      5. 过滤: 有摘要 + 有可下载PDF（优先arXiv副本，其次OA直链）

    参数:
        seeds: 种子候选（已按经典度预排序的PaperMeta列表）
        max_seeds: 实际使用的种子数（实测约4成种子的OpenAlex记录
            缺参考文献数据，8篇种子才能保证4-5篇有效参与共被引统计）
        min_hits: 共被引次数下限（2=被至少2篇种子引用）
        keep_top: 最多保留的经典论文数（控制后续LLM粗筛成本）

    返回:
        按chain_hits降序的PaperMeta列表（chain_hits字段已填充）
    """
    seeds = [s for s in seeds if s.openalex_id][:max_seeds]
    if not seeds:
        return []

    def _count_refs(seed_like: PaperMeta):
        """把单篇种子的参考文献累加进ref_counts（短ID归一化）"""
        refs = _seed_references(seed_like, fallback_quota)
        if not refs:
            # ---- S2兜底（P1-2修复）----
            # OpenAlex主记录+重复记录都无参考文献（2023+论文的
            # 数据黑洞）: 从Semantic Scholar取references并映射
            # 回W-ID，让引用链在LLM主题复活。配额控制: S2限流
            # 严格，单次运行最多兜底6篇种子
            if s2_fallback_quota[0] > 0:
                s2_fallback_quota[0] -= 1
                refs = _s2_references_as_wids(seed_like)
                if refs:
                    print(f"[检索Agent] 引用链S2兜底: 《{seed_like.title[:38]}...》"
                          f"OpenAlex无参考文献，从Semantic Scholar取回"
                          f"{len(refs)}条(映射为W-ID)")
        if refs:
            for w_id in refs:
                # OpenAlex的referenced_works可能是完整URL也可能是短ID，
                # 统一截取末段短ID（否则后续按短ID查dict会全部miss）
                short = w_id.split("/")[-1]
                ref_counts[short] = ref_counts.get(short, 0) + 1
        return len(refs) if refs else 0

    # 兜底反查配额: 缺引用数据的种子很多时（如"llm"主题12篇缺8篇），
    # 逐个标题反查会放大请求量触发429限流，最多反查4篇
    fallback_quota = [4]
    # S2兜底配额（P1-2）: 每篇种子1次match+1次references+N/50次
    # DOI批量映射，最多6篇防止S2限流触发熔断拖垮主检索
    s2_fallback_quota = [6]

    # ---- 第1轮: 原始种子的参考文献共被引统计 ----
    ref_counts: dict[str, int] = {}  # 短W_id -> 被几篇种子引用
    for seed in seeds:
        n = _count_refs(seed)
        if n == 0:
            print(f"[检索Agent] 引用链: 《{seed.title[:40]}...》"
                  f"OpenAlex与S2均无参考文献数据，放弃该种子")
        else:
            print(f"[检索Agent] 引用链: 《{seed.title[:40]}...》"
                  f"的参考文献{n}条")
        time.sleep(0.3)  # 对OpenAlex留礼貌间隔

    # ---- 第2步: 共被引>=min_hits的按次数降序，截取keep_top ----
    strong = sorted(
        (w_id for w_id, c in ref_counts.items() if c >= min_hits),
        key=lambda w: -ref_counts[w],
    )[:keep_top]
    if not strong:
        print("[检索Agent] 引用链挖掘: 无共被引>=2的论文")
        return []

    # ---- 第3步: 批量拉取经典论文的元数据 ----
    dropped: list[str] = []  # 诊断信息: 丢弃的经典及原因
    chain_papers = _fetch_chain_papers(strong, ref_counts, dropped)

    # ---- 第2轮: 雪球扩展（backward snowballing，系统综述标准技巧）----
    # 动机: 最老的奠基论文（SpikeProp 1999/Tempotron 2005）发表太早，
    # 近年种子论文可能不引它们，但第一轮挖出的顶级经典（SLAYER/
    # Spatio-Temporal BP）的参考文献里一定有——用顶级经典作二级
    # 种子再挖一层，把"经典的经典"也捞上来
    snowball_seeds = [p for p in chain_papers if p.chain_hits >= 3][:4]
    if snowball_seeds:
        print(f"[检索Agent] 雪球扩展: 用{len(snowball_seeds)}篇顶级经典"
              f"作二级种子（"
              + "; ".join(f"《{p.title[:28]}》" for p in snowball_seeds)
              + "）...")
        for sp in snowball_seeds:
            _count_refs(sp)
            time.sleep(0.3)
        # 更新已收录论文的chain_hits（二级引用累加）
        for p in chain_papers:
            p.chain_hits = ref_counts.get(p.openalex_id, p.chain_hits)
        # 新过线的论文（原<2现在>=2）补充进来
        existing = ({p.openalex_id for p in chain_papers}
                    | {s.openalex_id for s in seeds})
        new_strong = sorted(
            (w for w, c in ref_counts.items()
             if c >= min_hits and w not in existing),
            key=lambda w: -ref_counts[w])[:10]
        if new_strong:
            chain_papers += _fetch_chain_papers(new_strong, ref_counts,
                                                dropped)

    chain_papers.sort(key=lambda p: -p.chain_hits)
    print(f"[检索Agent] 引用链挖掘完成: {len(seeds)}篇种子"
          f"(+{len(snowball_seeds)}篇二级) -> "
          f"{len({w for w, c in ref_counts.items() if c >= min_hits})}篇"
          f"共被引>={min_hits}的经典 -> {len(chain_papers)}篇可下载")
    if dropped:
        print(f"[检索Agent] 引用链丢弃{len(dropped)}篇: "
              + "; ".join(dropped[:5]))
    return chain_papers


def _inject_classics(classic_titles: list[str],
                     seen: dict[str, PaperMeta]) -> list[PaperMeta]:
    """
    LLM领域先验注入：按规划Agent给出的经典论文标题反查OpenAlex收录

    【为什么需要这条路（2026-09-04实测教训）】
      引用链挖掘也有数据墙: OpenAlex对最老论文的引用图谱残缺——
      SpikeProp(1999)真实被引2000+，但OpenAlex记录只显示144次，
      且SuperSpike等论文的参考文献列表里根本没有链接到原作记录
      （只链接了它的后续改进论文）。共被引统计在这堵墙前无解。
      LLM的领域知识是召回这类"图谱黑洞"经典的最后手段：
      规划Agent直接给出精确标题，这里按标题反查确认收录。

    信任分级（防LLM幻觉标题污染结果）:
      - 标题相似度>=0.85才收录（防同名不同文）
      - 被引>=300: chain_hits=3（进经典免检通道——LLM说它是经典
        + 学术界高被引确认，双重证据）
      - 被引<300: chain_hits=2（享受粗筛阈值补偿，仍需LLM粗筛把关）
      - OpenAlex查无此文: 丢弃并打印（LLM幻觉标题自然过滤）

    返回:
        成功注入的经典PaperMeta列表（已加入seen）
    """
    if not classic_titles:
        return []
    if _OA_QUOTA_EXHAUSTED:
        # 熔断中(额度耗尽或429风暴): OpenAlex反查必失败，整体跳过
        # （2026-09-08实测: 不检查就会每个标题白等4次退避重试）
        print("[检索Agent] OpenAlex熔断中，经典注入跳过")
        return []
    from difflib import SequenceMatcher

    injected = []
    for title in classic_titles:
        data = _oa_request({
            "filter": f"title.search:{_oa_title_query(title)}",
            "per_page": 5,
            "select": "id,title,publication_year,cited_by_count,"
                      "abstract_inverted_index,authorships,locations,doi,"
                      "best_oa_location,open_access",
        })
        if data is None:
            continue
        best = None
        best_score = 0.0
        for work in data.get("results", []):
            cand = (work.get("title") or "")
            score = SequenceMatcher(None, _norm_title(title),
                                    _norm_title(cand)).ratio()
            if score > best_score:
                best_score, best = score, work
        # 标题相似度门槛: 防止LLM幻觉标题匹配到不相干的论文
        if best is None or best_score < 0.85:
            print(f"[检索Agent] 经典注入: 未找到《{title[:45]}》"
                  f"(最佳匹配相似度{best_score:.2f}<0.85，疑似标题有误)")
            continue

        w_id = (best.get("id") or "").split("/")[-1]
        # 已收录的（关键词/引用链已召回）只增强信号
        dup = _find_duplicate_by_openalex(w_id, seen)
        if dup is not None:
            old = seen[dup]
            old.chain_hits = max(old.chain_hits, 2)
            print(f"[检索Agent] 经典注入: 《{old.title[:45]}》已收录，"
                  f"chain_hits增强至{old.chain_hits}")
            continue

        abstract = _oa_abstract_to_text(best.get("abstract_inverted_index"))
        cited = best.get("cited_by_count") or 0
        arxiv_id = _oa_extract_arxiv_id(best)
        if _looks_like_arxiv_id(arxiv_id):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        else:
            oa_loc = best.get("best_oa_location") or {}
            pdf_url = (oa_loc.get("pdf_url")
                       or oa_loc.get("landing_page_url") or "")
            arxiv_id = w_id
        if not abstract and cited < 300:
            continue  # 无摘要又不够格免检的，无法进入粗筛

        meta = PaperMeta(
            arxiv_id=arxiv_id,
            title=best.get("title") or title,
            authors=[a["author"]["display_name"]
                     for a in (best.get("authorships") or [])[:5]],
            year=best.get("publication_year") or 0,
            abstract=abstract,
            pdf_url=pdf_url,
            cited_by=cited,
            openalex_id=w_id,
            # 信任分级: 高被引确认的经典免检，否则只享受阈值补偿
            chain_hits=3 if cited >= 300 else 2,
        )
        seen[meta.arxiv_id] = meta
        injected.append(meta)
        print(f"[检索Agent] 经典注入: 《{meta.title[:45]}》"
              f"(被引{cited}, chain_hits={meta.chain_hits})")
    return injected


def _find_duplicate_by_openalex(w_id: str,
                                seen: dict[str, PaperMeta]) -> str | None:
    """按OpenAlex ID查重（经典注入与已收录论文的快速对齐）"""
    for pid, old in seen.items():
        if old.openalex_id == w_id:
            return pid
    return None


# ---------------------------------------------------------------
# 第1段-B：arXiv官方API检索（兜底1）
# ---------------------------------------------------------------
def _arxiv_request(params: dict) -> str | None:
    """
    带缓存的arXiv API查询（成员A·任务1），返回原始XML文本

    arXiv返回Atom XML而非JSON，故用{"_xml": 文本}包装后入缓存，
    与OpenAlex/S2的dict缓存共用同一张表和同一套TTL/日志机制。
    arXiv无密钥限流，但有间歇性封锁（2026-09实测），缓存后
    同关键词的兜底检索不再重复碰运气。
    """
    _ck = cache.make_key("arxiv", params)
    _hit = cache.get(_ck)
    if _hit is not None:
        print(f"[检索Agent] 缓存命中(arxiv): "
              f"{params.get('search_query', '')[:50]}")
        return _hit.get("_xml", "")
    try:
        _t0 = time.perf_counter()
        resp = requests.get(ARXIV_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
    except Exception:
        return None  # 失败不缓存（下次还会真实重试）
    cache.put(_ck, {"_xml": resp.text})
    cache.log_request("arxiv", _ck,
                      (time.perf_counter() - _t0) * 1000, "ok")
    return resp.text


def _arxiv_search(kw: str, limit: int) -> list[PaperMeta]:
    """
    调用arXiv官方API检索单个关键词

    返回:
        PaperMeta列表；网络失败返回空列表（由上层切换到S2兜底）
    """
    import xml.etree.ElementTree as ET  # arXiv返回Atom XML格式

    # arXiv查询语法: 前缀限定标题/摘要检索，比全文检索质量高
    query = " OR ".join(f"all:{w}" for w in kw.split())
    params = {
        "search_query": query,
        "max_results": limit,
        "sortBy": "relevance",
    }
    xml_text = _arxiv_request(params)
    if xml_text is None:
        print("[检索Agent] arXiv API不可达，将切换兜底源")
        return []

    # 解析Atom XML（命名空间处理是标准写法）
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", ns):
        # arXiv的entry.id形如 http://arxiv.org/abs/2401.12345v2
        raw_id = entry.findtext("a:id", "", ns).split("/abs/")[-1]
        arxiv_id = raw_id.split("v")[0] if raw_id[0].isdigit() else raw_id
        abstract = (entry.findtext("a:summary", "", ns) or "").strip()
        title = (entry.findtext("a:title", "", ns) or "").strip()
        if not arxiv_id or not abstract or not title:
            continue

        # 统一用 arxiv.org/pdf/{id} 作为下载地址（实测最稳的域名）
        pdf_url = f"https://arxiv.org/pdf/{raw_id}"
        papers.append(PaperMeta(
            arxiv_id=arxiv_id,
            title=title.replace("\n", " "),
            authors=[a.findtext("a:name", "", ns)
                     for a in entry.findall("a:author", ns)][:5],
            year=int(entry.findtext("a:published", "0000", ns)[:4]),
            abstract=abstract.replace("\n", " "),
            pdf_url=pdf_url,
        ))
    return papers


# ---------------------------------------------------------------
# 第1段-B2：Crossref检索（第二主源，成员A·任务2）
# ---------------------------------------------------------------
def _crossref_request(params: dict, max_retries: int = 3) -> dict | None:
    """
    带缓存和重试的Crossref请求

    为什么选Crossref兜底（2026-09-10调研实测）:
      - 免密钥、无每日额度，UA附邮箱进礼貌池后稳定直连
        （实测响应1-1.5秒，无429）——S2熔断和OpenAlex额度
        耗尽同时发生时的可靠退路
      - 自带is-referenced-by-count被引数（影响力排序信号）
      - 局限: arXiv预印本的DOI注册在DataCite而非Crossref，
        所以这里检索到的是期刊正式版（与arXiv预印本同一篇
        论文的重复，由search()的标题相似度去重统一处理）

    返回:
        message字段的dict（items列表所在层），失败返回None
    """
    _ck = cache.make_key("crossref", params)
    _hit = cache.get(_ck)
    if _hit is not None:
        print(f"[检索Agent] 缓存命中(crossref): "
              f"{str(params.get('query.bibliographic', ''))[:50]}")
        return _hit

    for attempt in range(max_retries):
        try:
            _t0 = time.perf_counter()
            resp = requests.get(CROSSREF_API_BASE, params=params,
                                headers={"User-Agent": CROSSREF_UA},
                                timeout=30)
            resp.raise_for_status()
            payload = resp.json().get("message", {})
            # 只缓存200成功响应（cache.put的契约）
            cache.put(_ck, payload)
            cache.log_request("crossref", _ck,
                              (time.perf_counter() - _t0) * 1000, "ok")
            # Crossref礼貌池建议节奏≤1次/2秒（无强制，宁慢勿堵）
            time.sleep(1)
            return payload
        except Exception as e:
            cache.log_request("crossref", _ck,
                              (time.perf_counter() - _t0) * 1000, "error")
            print(f"[检索Agent] Crossref请求异常(第{attempt + 1}次): {e}")
            time.sleep(2 * (attempt + 1))
    return None


def _crossref_jats_to_text(jats: str | None) -> str:
    """
    清洗Crossref的JATS XML摘要为纯文本

    Crossref的abstract是出版社上传的JATS格式，形如:
      <jats:p>Background...<jats:italic>key</jats:italic>...</jats:p>
    处理: 剥掉全部XML标签 + 反转义HTML实体 + 压缩空白
    """
    import re
    import html
    if not jats:
        return ""
    text = re.sub(r"<[^>]+>", " ", jats)     # 所有标签替换为空格
    text = html.unescape(text)               # &amp; &lt; 等实体还原
    return " ".join(text.split())            # 压缩连续空白


def _crossref_search(kw: str, limit: int,
                     year_from: int | None = None) -> list[PaperMeta]:
    """
    调用Crossref检索单个关键词（第二主源，双兜底之后接管）

    检索策略:
      - query.bibliographic 对标题+摘要做书目检索（相关性排序）
      - select只取需要的字段（响应更快）
      - year_from传入时用from-pub-date过滤（与OpenAlex同义）
      - 请求3倍行数再截断: Crossref只有约1/4记录带摘要
        （出版社选择性上传），rows=limit只能留下limit*25%，
        多取再筛保证有效产出量（实测16行只留4篇→48行留14篇）

    PaperMeta填充约定（与其他数据源的关键差异）:
      - arxiv_id: 填DOI（如10.3897/jucs.164737）——仅作唯一ID用，
        下载阶段若无直链会走arXiv标题救援，救援成功后更新为
        真实arXiv编号（与OpenAlex W-id的流动路径完全一致）
      - pdf_url: 留空——期刊PDF直链普遍403反爬（2026-09-03实测），
        交给救援机制从arXiv拿，成功率远高于硬啃出版社
      - cited_by: is-referenced-by-count（Crossref的被引统计）

    返回:
        PaperMeta列表；网络失败返回空列表（由上层切换arXiv兜底）
    """
    params = {
        "query.bibliographic": kw,
        "rows": min(limit * 3, 60),  # 3倍超采，弥补摘要覆盖率(~25%)
        "select": "DOI,title,abstract,author,issued,is-referenced-by-count",
    }
    if year_from and year_from >= 2000:
        params["filter"] = f"from-pub-date:{year_from}-01-01"

    data = _crossref_request(params)
    if data is None:
        return []

    papers = []
    for item in data.get("items", []):
        title = (item.get("title") or [""])[0].strip()
        abstract = _crossref_jats_to_text(item.get("abstract"))
        # 无标题或无摘要的记录没有 downstream 价值（粗筛要读摘要打分）
        if not title or len(abstract) < 50:
            continue
        doi = item.get("DOI") or ""
        if not doi:
            continue  # 无DOI的记录（极少）无法构建唯一ID

        year = ((item.get("issued") or {})
                .get("date-parts") or [[0]])[0][0] or 0
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip()
                   for a in (item.get("author") or [])[:5]]

        papers.append(PaperMeta(
            arxiv_id=doi,          # DOI作唯一ID（救援成功后会替换）
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            pdf_url="",            # 待arXiv标题救援补链
            cited_by=item.get("is-referenced-by-count") or 0,
        ))
        if len(papers) >= limit:  # 3倍超采后截断回目标量
            break
    return papers


# ---------------------------------------------------------------
# 第1段-C：Semantic Scholar 检索（兜底2）
# ---------------------------------------------------------------
def _s2_request(params: dict, max_retries: int = 3,
                url: str | None = None, cache_ns: str = "s2") -> dict | None:
    """
    带重试的Semantic Scholar请求

    认证方式:
        有密钥时通过 x-api-key 请求头认证（限流额度1次/秒独享，稳定）
        无密钥时匿名调用（全球共享池，高峰期429频繁）

    url/cache_ns（P1-2引用链兜底新增）:
        url默认搜索端点；引用链兜底复用本函数的重试/熔断/缓存逻辑，
        传自定义端点URL。cache_ns区分缓存命名空间——不同端点用
        相同params时防止缓存键串台（如search和match都有query字段）

    重试策略（2026-09-04实测教训）:
        旧配置10次×60秒=单关键词最多卡10分钟，纯粹拖慢流水线，
        快速放弃比死等更优——还有OpenAlex/arXiv兜底，缺一路候选
        不至于空手而归。

    429说明（2026-09-07实测）:
        S2官方文档确认: 即使遵守1 req/s，其共享基础设施负载高时
        也会对认证用户返回429（官方推荐处理就是退避重试）。
        实测认证模式429频率不低，因此: (1)尊重Retry-After头;
        (2)成功后强制2秒间隔（宁慢勿堵，4个关键词总共才多花6秒）。

    返回:
        解析后的JSON dict，彻底失败返回None
    """
    global _S2_429_COUNT, _S2_BREAKER_UNTIL
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}

    # ---- 缓存层（成员A·任务1）----
    # 缓存键只含查询参数不含API密钥（密钥在headers里，不进params，
    # 换密钥后缓存依然有效——S2返回的元数据与哪个密钥请求无关）。
    # _url并入缓存键: references端点的paperId在URL路径里而非params，
    # 不并入的话所有种子共用{"fields","limit"}会缓存串台（第一篇
    # 种子的参考文献被错误回放给后续所有种子）。
    # 命中后跳过下方1 req/s限速等待：没有网络请求就无所谓限速
    _ck = cache.make_key(cache_ns, {**params, "_url": url or S2_API_BASE})
    _hit = cache.get(_ck)
    if _hit is not None:
        # 命中提示: 同时说明跳过了1 req/s限速（缓存读无需排队）
        print(f"[检索Agent] 缓存命中(s2): "
              f"{str(params.get('query', ''))[:50]} (跳过限速等待)")
        return _hit

    # ---- S2熔断检查（连续429风暴时短路，冷却后自动恢复）----
    # 熔断期间不发请求不等待，上层立即切换OpenAlex/arXiv兜底
    if time.time() < _S2_BREAKER_UNTIL:
        remain = int(_S2_BREAKER_UNTIL - time.time())
        print(f"[检索Agent] S2熔断中(连续{_S2_429_COUNT}次429，"
              f"冷却剩{remain}秒)，本请求跳过走兜底数据源")
        return None

    for attempt in range(max_retries):
        try:
            _t0 = time.perf_counter()
            resp = requests.get(url or S2_API_BASE, params=params,
                                headers=headers, timeout=30)
            if resp.status_code == 429:
                # 优先尊重服务端给的Retry-After（秒），没有则用退避序列
                cache.log_request("s2", _ck,
                                  (time.perf_counter() - _t0) * 1000,
                                  "429")
                # ---- 风暴熔断计数（2026-09-09）----
                # 连续429达到阈值说明是S2服务端过载而非本地节奏问题，
                # 再等再试都是白费——熔断5分钟，本轮走兜底
                _S2_429_COUNT += 1
                if _S2_429_COUNT >= _S2_BREAKER_TRIP:
                    _S2_BREAKER_UNTIL = (time.time()
                                         + _S2_BREAKER_COOLDOWN)
                    print(f"[检索Agent] S2连续{_S2_429_COUNT}次429"
                          f"(服务端过载)，熔断{_S2_BREAKER_COOLDOWN // 60}"
                          f"分钟并切换OpenAlex/arXiv兜底（冷却后自动恢复）")
                    return None
                # ---- 退避序列: 3/8/15秒（旧值10/20/30过保守）----
                # 实测S2的429多是共享池秒级抖动，首次3秒通常已够；
                # Retry-After头有值时始终优先尊重服务端指示
                retry_after = resp.headers.get("Retry-After")
                if config.S2_API_KEY:
                    wait = (int(retry_after) + 1 if retry_after
                            else (3, 8, 15)[min(attempt, 2)])
                else:
                    wait = 30
                mode = "认证" if config.S2_API_KEY else "匿名"
                print(f"[检索Agent] S2限流(429, {mode}模式)，等待{wait}秒后重试"
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            # 请求成功: 连续429计数清零（风暴解除）
            _S2_429_COUNT = 0
            # 只缓存200成功响应（cache.put的契约）；429在上面continue
            # 分支已被拦截，到这里的必然是成功响应
            cache.put(_ck, payload)
            cache.log_request("s2", _ck,
                              (time.perf_counter() - _t0) * 1000, "ok")
            # 官方限流为所有端点合计1次/秒，且共享负载高时1秒间隔
            # 也会429（2026-09-07实测），成功后强制2秒间隔宁慢勿堵
            if config.S2_API_KEY:
                time.sleep(2)
            return payload
        except Exception as e:
            cache.log_request("s2", _ck,
                              (time.perf_counter() - _t0) * 1000, "error")
            print(f"[检索Agent] S2请求异常(第{attempt + 1}次): {e}")
            time.sleep(5)
    return None


def _s2_search(kw: str, limit: int, year_from: int | None = None) -> list[PaperMeta]:
    """
    调用Semantic Scholar检索单个关键词（主数据源，认证模式）

    认证密钥(2026-09-06下发)后1 req/s独享，稳定不再匿名429。
    citationCount被引数填充到cited_by（影响力加成+综合排序需要）。

    年份过滤: S2的year参数支持区间格式"2021-2026"，
    与OpenAlex的from_publication_date等价。

    只保留有arXiv编号且提供了摘要的论文
    （本项目后续步骤需要从PDF提取全文，无法下载的论文没有价值）

    返回:
        PaperMeta列表；网络失败返回空列表（由上层切换兜底数据源）
    """
    params = {
        "query": kw,
        "limit": limit,
        "fields": S2_FIELDS,
    }
    if year_from:
        import datetime
        params["year"] = (f"{year_from}-"
                          f"{datetime.date.today().year}")

    data = _s2_request(params)
    if data is None:
        return []

    papers = []
    for item in data.get("data", []):
        arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
        abstract = item.get("abstract")
        # 三项任一缺失都无法进入后续流程，直接跳过
        if not arxiv_id or not abstract:
            continue
        papers.append(PaperMeta(
            arxiv_id=arxiv_id,
            title=(item.get("title") or "").strip(),
            authors=[a["name"] for a in (item.get("authors") or [])[:5]],
            year=item.get("year") or 0,
            abstract=abstract.replace("\n", " ").strip(),
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",  # 仍从arxiv下载
            cited_by=item.get("citationCount") or 0,  # S2被引数（影响力信号）
        ))
    return papers


# ---------------------------------------------------------------
# 第1段：统一检索入口（S2主源 → OpenAlex → arXiv 三级容错）
# ---------------------------------------------------------------
def _resolve_openalex_ids(papers: list[PaperMeta], limit: int = 12) -> int:
    """
    为S2/arXiv检索结果补齐OpenAlex ID（引用链挖掘的种子前置条件）

    背景: 引用链挖掘依赖OpenAlex的referenced_works引用图谱，
    但S2主源的结果只有arXiv ID。按标题反查OpenAlex逐篇解析
    （每篇1个请求，limit封顶；OpenAlex额度耗尽时熔断器让
    _oa_request直接返回None，解析自动中断，流水线不受阻）。

    返回:
        成功解析出openalex_id的论文数
    """
    from difflib import SequenceMatcher

    resolved = 0
    for p in papers:
        if resolved >= limit:
            break
        if p.openalex_id:
            continue
        data = _oa_request({
            "filter": f"title.search:{_oa_title_query(p.title)}",
            "per_page": 3,
            "select": "id,title,cited_by_count",
        })
        if data is None:
            # 熔断中或查询失败: 放弃剩余解析（后续种子同样会失败）
            break
        best, best_score = None, 0.0
        norm = _norm_title(p.title)
        for work in data.get("results", []):
            score = SequenceMatcher(
                None, norm, _norm_title(work.get("title") or "")).ratio()
            if score > best_score:
                best, best_score = work, score
        # 标题相似度门槛: 防止把别的论文的引用图谱接错种子
        if best is not None and best_score >= 0.85:
            p.openalex_id = (best.get("id") or "").split("/")[-1]
            resolved += 1
    return resolved



def _parse_year_from(inclusion_criteria: str) -> int | None:
    """
    从筛选条件文本里解析年份下限

    示例: "近5年、英文、有实验" -> 2021（当前年份-5）
    解析不到则返回None（不做时间过滤）
    """
    import re
    m = re.search(r"近\s*(\d+)\s*年", inclusion_criteria or "")
    if m:
        import datetime
        return datetime.date.today().year - int(m.group(1)) + 1
    return None


def _find_duplicate(meta: PaperMeta,
                    seen: dict[str, PaperMeta]) -> str | None:
    """
    查找已收录论文中与meta同属一篇的重复项

    背景: 同一篇论文在OpenAlex/arXiv可能有两个记录——
      旧式arXiv编号(cs.NE/0309021)与新式编号(1611.05280)并存，
      arxiv_id不同但实为同一篇论文（实测SNN检索里"Technical
      report: supervised training..."重复出现了两次）。
    策略: 标题相似度>0.90视为同一篇（O(n)比对，候选池<200篇可接受）
    """
    from difflib import SequenceMatcher

    norm_new = _norm_title(meta.title)
    for pid, old in seen.items():
        if pid == meta.arxiv_id:
            return pid
        if _norm_title(old.title) == norm_new:
            return pid
        # 相似度检查（标题非空时）
        if norm_new and _norm_title(old.title):
            if SequenceMatcher(None, norm_new,
                               _norm_title(old.title)).ratio() > 0.90:
                return pid
    return None


def _norm_title(title: str) -> str:
    """标题归一化: 小写+压缩空白（用于重复检测的快速通道）"""
    return " ".join((title or "").lower().split())


def _oa_title_query(title: str) -> str:
    """
    清洗标题，使其能安全拼进OpenAlex的title.search过滤器

    根因（2026-09-14探针实测）: OpenAlex的filter语法用逗号连接
    多个过滤器——标题带逗号时（《ChatGPT, GPT-4, and ...》类
    LLM论文重灾区）逗号后的部分被解析成第二个过滤器名，
    非法名 -> HTTP 400。探针结论: 逗号=400，冒号/撇号/括号/
    斜杠/& 均无害。当天日志: 2次400硬错误发生在引用链反查
    与经典注入，均因标题含逗号。

    处理: 把逗号/分号/引号等分隔符替换为空格（title.search是
    词袋匹配，丢标点不影响召回），再压缩空白、限长200字符
    （超长标题会产生超长URL）。
    """
    cleaned = re.sub(r"[,;\"'()\[\]{}<>]", " ", title or "")
    return " ".join(cleaned.split())[:200]


def search(plan: SearchPlan, results_per_query: int = 10) -> list[PaperMeta]:
    """
    按检索计划逐个关键词检索，多路结果RRF融合排序。

    【多路检索 + RRF融合（检索质量的核心改进 2026-09-04）】
    问题背景: 单一"被引数排序"有系统性偏差——
      老论文被引自然高（先发优势），新经典论文被埋没；
      而单一"相关性排序"又会被灌水文刷屏（关键词堆砌）。
    解决方案: 每个检索词做【双路检索】:
      路1 cited_by_count:desc     影响力路（经典论文浮上来）
      路2 relevance_score:desc    相关性路（主题精准匹配浮上来）
      路3 引用链挖掘（仅OpenAlex主源）: 从高影响力种子的参考文献
          统计共被引次数，召回关键词检索不到的奠基经典
          （如SpikeProp(1999)摘要不含"surrogate gradient"，
          但被所有后续SNN训练论文引用）
    融合算法: Reciprocal Rank Fusion（信息检索标准方法）
      rrf(paper) = Σ_各路 1/(K + rank)   K=60（原论文推荐值）
      再叠加多关键词命中加成: 被 n 个检索词命中 ×(1 + 0.5*(n-1))
      ——多篇论文被多个不同角度的检索词共同命中，是强相关信号
    数据源容错（2026-09-06起S2为首选）: S2认证模式失败时降级
    OpenAlex -> arXiv 单路检索。引用链挖掘始终尝试走OpenAlex
    （S2结果需先解析OpenAlex ID），额度耗尽时熔断自动跳过。

    返回:
        按RRF融合分降序的PaperMeta列表（已去重，rrf_score/cited_by/
        chain_hits已填充；引用链论文可按chain_hits另行排序）
    """
    K = 60  # RRF平滑常数（Cormack et al. 2009 推荐值）
    seen = {}        # arxiv_id -> PaperMeta（去重合并）
    rank_lists = []  # 各路的[arxiv_id]有序列表（RRF的输入）
    kw_hit_count = {}  # arxiv_id -> 被几个关键词命中（跨路去重计数）
    id_alias = {}    # 重复论文的ID别名映射: 旧id -> 保留的规范id
    source = None    # 数据源状态: None=未决定
    year_from = _parse_year_from(plan.inclusion_criteria)

    for kw in plan.keywords:
        print(f"[检索Agent] 正在检索: {kw} ...")
        kw_hit_ids = set()  # 本关键词命中的论文（用于跨关键词加成）

        # ---- 数据源检索（每个关键词都要执行，含cited路!）----
        # 注意: 旧版bug是cited_res只在第一个关键词赋值，后续关键词
        # 复用kw1的旧结果导致cited路"从未真正执行"（Surrogate
        # Gradient被引149却显示被引0的根因，2026-09-04修复）
        # 数据源优先级(2026-09-10任务2起): S2认证(稳定1req/s) ->
        # OpenAlex($0.1/天额度易烧穿) -> Crossref(免密钥无限额) ->
        # arXiv(间歇性阻断)。Crossref插在OpenAlex之后: 期刊版论文
        # 下载要靠arXiv救援（成功率低于arXiv直检），但胜在无限额
        # 稳定——OpenAlex熔断当天Crossref照样能出候选
        if source is None or source == "s2":
            cited_res = _s2_search(kw, results_per_query, year_from)
            if cited_res:
                if source is None:
                    source = "s2"
                    print(f"[检索Agent] 数据源选定: Semantic Scholar"
                          f"(认证模式"
                          f"{f', 年份下限{year_from}' if year_from else ''})")
            else:
                # S2失败（首轮即不可用，或中途429熔断）: 降级OpenAlex双路。
                # 注意source=="s2"时也要走这里——否则熔断后剩余关键词
                # 会拿到空结果静默丢失（2026-09-09熔断器引入时修复）
                cited_res = _openalex_search(kw, results_per_query,
                                             year_from, sort_mode="cited")
                if cited_res:
                    source = "openalex"
                    print("[检索Agent] S2不可用, 数据源切换: "
                          "OpenAlex双路检索")
                else:
                    # OpenAlex也不可用（额度熔断/网络失败）: Crossref接管
                    cited_res = _crossref_search(kw, results_per_query,
                                                 year_from)
                    if cited_res:
                        source = "crossref"
                        print("[检索Agent] S2/OpenAlex均不可用, "
                              "数据源切换: Crossref(免密钥第二主源)")
                    else:
                        source = "arxiv"
                        print("[检索Agent] 所有元数据源均不可用, "
                              "数据源切换: arXiv官方API")
                        cited_res = _arxiv_search(kw, results_per_query)
        elif source == "openalex":
            cited_res = _openalex_search(kw, results_per_query, year_from,
                                         sort_mode="cited")
            if not cited_res:
                # OpenAlex中途失败（429风暴熔断/网络异常）: 降级
                # Crossref -> arXiv（与S2分支同款修复——否则熔断后
                # 剩余关键词静默拿空结果，2026-09-14修复）
                cited_res = _crossref_search(kw, results_per_query, year_from)
                if cited_res:
                    source = "crossref"
                    print("[检索Agent] OpenAlex不可用, 数据源切换: "
                          "Crossref(免密钥第二主源)")
                else:
                    source = "arxiv"
                    print("[检索Agent] OpenAlex/Crossref均不可用, "
                          "数据源切换: arXiv官方API")
                    cited_res = _arxiv_search(kw, results_per_query)
        elif source == "crossref":
            cited_res = _crossref_search(kw, results_per_query, year_from)
        else:
            cited_res = _arxiv_search(kw, results_per_query)

        if source == "openalex":
            # ---- 双路检索: 影响力路 + 相关性路 ----
            rel_res = _openalex_search(kw, results_per_query, year_from,
                                       sort_mode="relevance",
                                       collect_citations=False)
            routes = [("cited", cited_res), ("relevance", rel_res)]
        else:
            # S2/arXiv只有单路（其API自带relevance排序）
            routes = [("cited", cited_res)]

        for route_name, results in routes:
            if not results:
                continue
            rank_lists.append([p.arxiv_id for p in results])
            for meta in results:
                dup_id = _find_duplicate(meta, seen)
                if dup_id is None:
                    if meta.arxiv_id not in seen:
                        seen[meta.arxiv_id] = meta
                else:
                    # 同一论文的重复命中: 保留信息更全的版本（带被引数的优先）
                    old = seen[dup_id]
                    if meta.cited_by > old.cited_by:
                        meta.abstract = meta.abstract or old.abstract
                        seen[meta.arxiv_id] = meta
                        del seen[dup_id]  # 换新键
                        # 记录别名: 旧id的排名贡献要归并到新id（RRF用）
                        id_alias[dup_id] = meta.arxiv_id
                    else:
                        old.abstract = old.abstract or meta.abstract
                        # 新id作为旧id的别名（排名贡献归并到保留的旧id）
                        if meta.arxiv_id != dup_id:
                            id_alias[meta.arxiv_id] = dup_id
                kw_hit_ids.add(meta.arxiv_id)
                if dup_id is not None:
                    # 重复版本也算本关键词命中（加成计数用规范化的id）
                    kw_hit_ids.add(dup_id)

        # 本关键词命中数累计（跨路去重后计数）
        for pid in kw_hit_ids:
            kw_hit_count[pid] = kw_hit_count.get(pid, 0) + 1
        time.sleep(1)  # 检索词之间留礼貌间隔

    # ---- 引用链挖掘（第3路信号: 召回关键词检索不到的奠基经典）----
    # 奠基论文的标题/摘要不含现代检索词（如SpikeProp的摘要里没有
    # "surrogate gradient"），关键词检索存在系统性盲区。从高影响力
    # 种子的参考文献里做共被引统计，被多篇种子共同引用=领域经典。
    # 该路只能走OpenAlex（referenced_works引用图谱是独有能力）。
    # S2主源时种子无openalex_id，先按标题反查解析（12篇封顶）；
    # OpenAlex额度耗尽时熔断器让解析/挖掘自动整体跳过。
    if source in ("s2", "openalex"):
        seed_pool = sorted(
            [p for p in seen.values()],
            key=lambda p: (kw_hit_count.get(p.arxiv_id, 0), p.cited_by),
            reverse=True)
        if source == "s2":
            n_resolved = _resolve_openalex_ids(seed_pool)
            if n_resolved == 0:
                print("[检索Agent] OpenAlex不可用(熔断/失败)，"
                      "引用链挖掘与经典注入本轮跳过")
            seed_pool = [p for p in seed_pool if p.openalex_id]
        chain_papers = _citation_chain_expand(seed_pool)
        chain_added = []  # 新挖出的经典（进入chain路排名）
        for cp in chain_papers:
            # 已被检索词召回的论文: 只补充chain_hits信号，不重复收录
            dup_id = (cp.arxiv_id if cp.arxiv_id in seen
                      else _find_duplicate(cp, seen))
            if dup_id is not None:
                old = seen.get(dup_id)
                if old is not None and cp.chain_hits > old.chain_hits:
                    old.chain_hits = cp.chain_hits
                continue
            seen[cp.arxiv_id] = cp
            chain_added.append(cp)
        if chain_papers:
            # 引用链路作为第3路排名融入RRF（按共被引次数降序）
            rank_lists.append([p.arxiv_id for p in
                               sorted(chain_papers,
                                      key=lambda p: -p.chain_hits)])
            print(f"[检索Agent] 引用链新增{len(chain_added)}篇经典候选, "
                  f"{len(chain_papers) - len(chain_added)}篇已有信号增强")

    # ---- LLM领域先验注入（第4路信号: 引用链图谱黑洞的兜底）----
    # OpenAlex对最老论文（1990s-2000s）的引用图谱残缺，共被引统计
    # 召回不到它们（SpikeProp真实被引2000+但图谱只记144）。规划Agent
    # 的领域知识给出经典标题清单，这里按标题反查OpenAlex确认收录
    if plan.classic_titles:
        injected = _inject_classics(plan.classic_titles, seen)
        if injected:
            # 注入的经典作为第4路排名融入RRF（按被引数降序）
            rank_lists.append([p.arxiv_id for p in
                               sorted(injected,
                                      key=lambda p: -p.cited_by)])
            print(f"[检索Agent] LLM先验注入{len(injected)}篇经典")

    # ---- RRF融合分计算 ----
    # 别名解析: 排名列表里的旧ID（被合并的重复版本）归并到规范ID计分
    def _canonical(pid: str) -> str:
        return id_alias.get(pid, pid)

    papers = list(seen.values())
    import math
    max_cited = max((p.cited_by for p in papers), default=0)
    for meta in papers:
        rrf = 0.0
        for rank_list in rank_lists:
            for pos, pid in enumerate(rank_list):
                if _canonical(pid) == meta.arxiv_id:
                    rrf += 1.0 / (K + pos + 1)
        # 多关键词命中加成: n个关键词命中 ×(1 + 0.5*(n-1))
        n_kw = kw_hit_count.get(meta.arxiv_id, 1)
        rrf *= 1.0 + 0.5 * (n_kw - 1)

        # ---- 影响力加成（经典论文的独立信号）----
        # 问题背景: 只被单一关键词命中的经典论文（如SLAYER被引472
        # 但标题不含"training backpropagation"字样），RRF分天然偏低。
        # 被引数是与检索排名正交的"经典度先验"，log压缩后归一化，
        # 以0.25权重叠加（避免被引数完全主导，防老论文垄断头部）:
        #   impact = log(1+cited) / log(1+max_cited)   ∈ [0,1]
        if max_cited > 0:
            impact = math.log1p(meta.cited_by) / math.log1p(max_cited)
        else:
            impact = 0.0
        meta.rrf_score = round(rrf * (1.0 + 0.25 * impact), 6)

    # 按RRF分降序返回（经典+相关的论文排前面，粗筛的送审顺序更优）
    papers.sort(key=lambda p: p.rrf_score, reverse=True)
    print(f"[检索Agent] 检索完成: {len(papers)}篇候选(去重), "
          f"{len(rank_lists)}路排名已融合(RRF)")
    return papers


# ---------------------------------------------------------------
# 第2段：LLM相关性粗筛
# ---------------------------------------------------------------
def filter_papers(plan: SearchPlan, papers: list[PaperMeta],
                  threshold: float = 4.0) -> list[PaperMeta]:
    """
    用LLM按筛选标准给每篇论文的摘要打1-5分，保留>=threshold的

    【经典免检通道（2026-09-04）】
      被>=3篇种子论文共同引用的论文（chain_hits>=3）不送LLM打分，
      直接保留。原因: 奠基论文的摘要写于20-30年前，问题设定的措辞
      与现代检索主题词天然不匹配（如SpikeProp摘要强调"误差反向传播"
      而非"SNN训练"），LLM相关性打分对它们有系统性低估；而共被引>=3
      是该领域多篇代表作"用脚投票"的共识信号，比LLM的一次性判断
      更权威。免检论文的relevance_score保持None，排序时给中性分。

    说明:
        分批送审（每批最多8篇的摘要），避免单次请求超长。
        批量打分一次调用处理多篇，比逐篇调用省~8倍token。

    参数:
        plan: 检索计划（提供筛选标准文本）
        papers: 候选论文列表
        threshold: 及格线，默认4分

    返回:
        过筛的论文列表（PaperMeta.relevance_score 已填充，经典免检的为None）
    """
    if not papers:
        return []

    # 经典免检: 共被引>=3的直接保留，其余送LLM打分
    classics = [p for p in papers if p.chain_hits >= 3]
    to_score = [p for p in papers if p.chain_hits < 3]
    if classics:
        print(f"[检索Agent] 经典免检通道: {len(classics)}篇"
              f"(共被引>=3)跳过LLM粗筛: "
              + "; ".join(f"《{p.title[:30]}》(被引{p.chain_hits}篇种子)"
                          for p in classics[:5]))

    BATCH = 8
    for i in range(0, len(to_score), BATCH):
        batch = to_score[i:i + BATCH]
        # 拼接本批论文的"编号+标题+摘要"文本
        paper_list_text = "\n\n".join(
            f"【论文{j}】标题: {p.title}\n摘要: {p.abstract[:600]}"  # 摘要截断，控制长度
            for j, p in enumerate(batch)
        )
        messages = [
            {"role": "system", "content":
                "你是文献筛选助手。根据筛选标准给每篇论文的摘要打相关性分(1-5分)。\n"
                "5=完全切题且满足所有标准, 4=高度相关, 3=部分相关, "
                "2=勉强沾边, 1=不相关。"},
            {"role": "user", "content":
                f"筛选标准: {plan.inclusion_criteria}\n"
                f"研究主题: {plan.topic}\n\n"
                f"{paper_list_text}\n\n"
                f"请输出从【论文0】到【论文{len(batch) - 1}】共{len(batch)}个分数。"},
        ]
        try:
            result = llm_client.chat_json(
                messages, schema_class=RelevanceScores,
                temperature=config.TEMP_EXTRACTOR,  # 筛选要求稳定，低温
            )
            # 把分数写回PaperMeta（防御式：长度不符时按最低分处理）
            for j, p in enumerate(batch):
                p.relevance_score = (
                    result.scores[j] if j < len(result.scores) else 1.0
                )
        except Exception as e:
            print(f"[检索Agent] 第{i // BATCH}批打分失败(整批按1分处理): {e}")
            for p in batch:
                p.relevance_score = 1.0

    # 只保留: 达到及格线的 + 经典免检的（共被引>=3）+ 阈值补偿的
    # 阈值补偿: 共被引>=2的论文及格线降1分——共被引是学术界"用脚
    # 投票"的独立相关性信号，可部分对冲LLM对老论文摘要的系统性
    # 低估（如Diehl&Cook被引1463、共被引2，但LLM只打3.0分被4.0
    # 及格线拦住）。补偿后仍需>=threshold-1，防止纯背景引用混入
    kept = [p for p in papers if p.chain_hits >= 3
            or (p.relevance_score or 0) >= threshold
            or (p.chain_hits >= 2
                and (p.relevance_score or 0) >= threshold - 1.0)]
    score_strs = [
        f"{p.arxiv_id}:"
        + ("免检" if (p.chain_hits >= 3 and p.relevance_score is None)
           else str(p.relevance_score))
        for p in papers
    ]
    print(f"[检索Agent] 粗筛完成: {len(papers)}篇 -> 保留{len(kept)}篇 "
          f"(含经典免检{len(classics)}篇; 得分: "
          + ", ".join(score_strs) + ")")
    return kept


# ---------------------------------------------------------------
# 第3段：PDF下载（并行版）
# ---------------------------------------------------------------
def _download_one(p: PaperMeta) -> bool:
    """
    下载单篇论文PDF（工作线程函数）

    返回:
        True=下载成功或本地已存在(local_path已填充); False=失败
    """
    filename = f"{p.arxiv_id.replace('/', '_')}.pdf"
    local_path = os.path.join(config.PAPER_DIR, filename)

    # 已下载过的直接复用（断点续传效果，重复实验不浪费流量）
    if os.path.exists(local_path) and os.path.getsize(local_path) > 10_000:
        p.local_path = local_path
        print(f"[检索Agent] 已存在,跳过下载: {filename}")
        return True

    # 无下载链接的（引用链经典无OA记录）: 直接失败进入救援流程，
    # 不走重试循环（空URL会抛MissingSchema，重试6次纯浪费30秒）
    if not p.pdf_url:
        print(f"[检索Agent] 无直链(待arXiv标题救援): {p.title[:45]}")
        return False

    try:
        # arXiv的PDF下载经常被间歇性重置连接(错误10054)，需要多次重试
        # （重置是瞬时的，多试几次总能碰上畅通窗口）
        for attempt in range(6):
            try:
                resp = requests.get(
                    p.pdf_url,
                    headers={"User-Agent": "literature-agent/0.1 (course project)"},
                    timeout=60,
                )
                resp.raise_for_status()
                break  # 下载成功，跳出重试循环
            except requests.HTTPError as dl_err:
                # HTTP 4xx是永久性错误(403反爬/404失效),重试无意义直接放弃
                if 400 <= dl_err.response.status_code < 500:
                    print(f"[检索Agent] 链接拒绝(HTTP "
                          f"{dl_err.response.status_code},不重试): "
                          f"{p.title[:45]}")
                    return False
                raise  # 5xx等瞬时错误才进入重试
            except Exception as dl_err:
                if attempt == 5:
                    raise dl_err  # 最后一次也失败，交给外层处理
                # 短间隔重试: 连接重置(10054)是瞬时故障,立即重连通常就成功
                # （旧值5/10/15秒过于保守,实测重置后2秒重连即可）
                wait = 2 * (attempt + 1)
                print(f"[检索Agent] 下载中断(第{attempt + 1}次)，"
                      f"{wait}秒后重试: {dl_err}")
                time.sleep(wait)

        with open(local_path, "wb") as f:
            f.write(resp.content)

        # ---- PDF魔数校验（OpenAlex数据源引入的新风险）----
        # OpenAlex的OA链接有时是HTML落地页而非PDF文件本身，
        # PyMuPDF解析非PDF会报错，所以在源头就拦住
        if not resp.content[:5] == b"%PDF-":
            os.remove(local_path)  # 删掉无效文件，避免残留
            print(f"[检索Agent] 链接非PDF(疑似HTML落地页,已跳过): "
                  f"{p.title[:50]} | {p.pdf_url[:60]}")
            return False

        p.local_path = local_path
        print(f"[检索Agent] 下载成功: {filename} "
              f"({len(resp.content) / 1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"[检索Agent] 下载失败(跳过): {p.title} | {e}")
        return False


def _download_arxiv_by_id(p: PaperMeta, raw_id: str) -> bool:
    """
    按arXiv编号直接下载PDF（救援下载的公共路径，P2-2重构抽出）

    关键事实（2026-09-14日志实据）: export.arxiv.org的API被间歇
    阻断时，arxiv.org的PDF下载主机往往正常（当天API不可达的
    同一批运行里PDF下载全部成功）——两者是不同主机，API故障
    ≠PDF不可下，所以拿到编号就有救。
    """
    rescue_url = f"https://arxiv.org/pdf/{raw_id}"
    # 沿用主下载的重试逻辑（连接重置是常态）
    for attempt in range(6):
        try:
            r = requests.get(rescue_url, timeout=60,
                             headers={"User-Agent": "literature-agent/0.1"})
            r.raise_for_status()
            if r.content[:5] != b"%PDF-":
                return False  # 不是PDF，放弃
            # 救援成功: 把ID更新为真实arXiv编号——
            # 引用链经典常顶着OpenAlex的W-id进来，更新后
            # papers_meta.json里的ID更可读，PDF缓存也能复用
            p.arxiv_id = (raw_id.split("v")[0]
                          if raw_id[:1].isdigit() else raw_id)
            filename = f"{p.arxiv_id.replace('/', '_')}.pdf"
            local_path = os.path.join(config.PAPER_DIR, filename)
            with open(local_path, "wb") as f:
                f.write(r.content)
            p.local_path = local_path
            print(f"[检索Agent] 救援成功(arXiv:{p.arxiv_id}): "
                  f"{p.title[:50]}")
            return True
        except requests.HTTPError:
            return False  # 4xx/5xx不重试
        except Exception:
            time.sleep(2 * (attempt + 1))
    return False


def _s2_lookup_arxiv_id(p: PaperMeta) -> str | None:
    """
    S2反查arXiv编号（P2-2救援兜底通道，2026-09-14）

    使用场景: export.arxiv.org API不可达（单点故障）时，改走
    Semantic Scholar的search/match按标题解析论文，从externalIds.
    ArXiv字段直接拿编号——不依赖arXiv API。

    返回:
        arXiv编号字符串（已剥离版本号），S2也无此论文或无arXiv
        副本时返回None。响应进缓存（7天），重复救援零成本。
    """
    if time.time() < _S2_BREAKER_UNTIL:
        return None  # S2熔断中
    data = _s2_request(
        {"query": p.title, "fields": "title,externalIds"},
        url=S2_MATCH_BASE, cache_ns="s2_match")
    if data is None:
        return None
    cands = data.get("data") or []
    if not cands:
        return None
    best = cands[0]
    # 相似度门限（与引用链兜底一致0.85）: match可能返回相近标题的
    # 其他论文，拿错编号会下载成别的论文
    from difflib import SequenceMatcher
    if SequenceMatcher(None, _norm_title(p.title),
                       _norm_title(best.get("title") or "")).ratio() < 0.85:
        return None
    aid = (best.get("externalIds") or {}).get("ArXiv")
    return re.sub(r"v\d+$", "", str(aid)) if aid else None


def _arxiv_rescue(p: PaperMeta) -> bool:
    """
    下载失败后的"arXiv救援": 用论文标题反查arXiv预印本

    使用场景: OpenAlex给的出版社直链被403反爬或返回HTML时，
    该论文很可能在arXiv有预印本副本（CS/AI领域尤其普遍）。
    用 ti:"标题" 精确检索arXiv，命中则从arxiv.org下载。

    双通道（P2-2去单点）:
        主通道: arXiv API标题检索 -> 编号
        兜底通道: arXiv API不可达时，S2 search/match反查
        externalIds.ArXiv -> 编号（PDF从arxiv.org下，API故障
        不影响PDF主机，见_download_arxiv_by_id的说明）

    返回:
        True=救援成功(local_path已填充); False=arXiv上也没有
    """
    import xml.etree.ElementTree as ET

    # 标题清洗: 去掉干扰检索的标点（连字符/冒号/引号），
    # arXiv的ti:检索对特殊字符和长短语敏感
    clean_title = (p.title.replace('"', " ").replace(":", " ")
                   .replace("-", " ").replace("–", " ").strip())

    # 用前5个实词做AND组合查询（实测整句短语查询命中率极低，
    # 而单词AND召回后再用相似度阈值过滤更可靠）
    words = [w for w in clean_title.split() if len(w) > 2][:5]
    if not words:
        return False
    query = " AND ".join(f'ti:"{w}"' for w in words)
    params = {
        "search_query": query,
        "max_results": 3,  # 取前3条逐一比对标题
    }
    xml_text = _arxiv_request(params)  # 复用带缓存的arXiv查询入口
    if xml_text is None:
        # ---- 兜底通道: arXiv API单点故障 -> S2反查编号 ----
        print("[检索Agent] arXiv API不可达，切换S2反查编号兜底...")
        aid = _s2_lookup_arxiv_id(p)
        if aid:
            return _download_arxiv_by_id(p, aid)
        print("[检索Agent] 救援失败: arXiv API与S2反查均不可达")
        return False

    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return False

    from difflib import SequenceMatcher
    for entry in root.findall("a:entry", ns):
        entry_title = (entry.findtext("a:title", "", ns) or "").strip()
        # 标题相似度校验（防止检索到只是同词的其他论文）
        if SequenceMatcher(None, clean_title.lower(),
                           entry_title.lower()).ratio() < 0.85:
            continue
        raw_id = entry.findtext("a:id", "", ns).split("/abs/")[-1]
        return _download_arxiv_by_id(p, raw_id)
    return False


def download(papers: list[PaperMeta], max_workers: int = 3) -> list[PaperMeta]:
    """
    并行下载论文PDF到本地 papers/ 目录

    性能说明: 串行下载5篇PDF约需20-40秒(每篇含下载+限速间隔),
    3线程并行后缩短到约1/3。3个并发对arxiv.org是安全的
    (它按IP限速,3并发远低于触发阈值;重试机制兜底偶发重置)。

    参数:
        papers: 待下载论文列表
        max_workers: 并行线程数（3为实测安全值）

    返回:
        下载成功的论文列表（local_path已填充）
    """
    os.makedirs(config.PAPER_DIR, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor

    downloaded = []
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map保持输入顺序; 返回值对应每篇是否成功
        outcomes = list(executor.map(_download_one, papers))
        for p, ok in zip(papers, outcomes):
            (downloaded if ok else failed).append(p)

    # ---- 二次救援: 主链接失败的论文尝试arXiv标题反查 ----
    # 出版社403/HTML落地页的论文，很多在arXiv有预印本副本
    if failed:
        print(f"[检索Agent] 主链接失败{len(failed)}篇，尝试arXiv救援...")
        time.sleep(2)  # 给arXiv API留间隔
        for p in failed:
            if _arxiv_rescue(p):
                downloaded.append(p)

    print(f"[检索Agent] PDF下载完成: {len(downloaded)}/{len(papers)} 篇"
          f"（含救援成功）")
    return downloaded


# ---------------------------------------------------------------
# 组合入口：完整的检索Agent流水线
# ---------------------------------------------------------------
def _composite_sort(papers: list[PaperMeta]) -> list[PaperMeta]:
    """
    最终综合排序（用户实际看到的论文顺序），四路信号加权融合:

        final = 0.45*相关性 + 0.20*RRF共识 + 0.15*影响力 + 0.20*引用链

    - 相关性(0.45): LLM对摘要的主题切题度打分(1-5)。权重最高——
      再经典的论文，跑题也不该进综述。经典免检论文未打分，给
      中性分3.5("高度相关"下限)，其经典度由引用链/影响力两路补足
    - RRF共识(0.20): 多路检索的排名共识——被多个检索角度共同
      命中的论文是强相关信号（归一化到0-5）
    - 影响力(0.15): log压缩的被引数归一化——经典度先验，
      防新灌水文靠关键词堆砌刷屏（归一化到0-5）
    - 引用链(0.20): 共被引次数log归一化——被多篇种子论文共同
      引用的奠基经典专属信号，关键词检索排序无法体现（归一化到0-5）

    计算后写入 PaperMeta.final_score（随JSON落盘，前端展示排序依据）
    """
    import math
    if not papers:
        return papers
    max_rrf = max(p.rrf_score for p in papers) or 1.0
    max_cited = max(p.cited_by for p in papers)
    max_chain = max(p.chain_hits for p in papers)
    for p in papers:
        rrf_norm = p.rrf_score / max_rrf  # 0~1
        impact = (math.log1p(p.cited_by) / math.log1p(max_cited)
                  if max_cited > 0 else 0.0)  # 0~1
        chain_norm = (math.log1p(p.chain_hits) / math.log1p(max_chain)
                      if max_chain > 0 else 0.0)  # 0~1
        # 免检经典relevance_score为None -> 中性分3.5
        rel = p.relevance_score if p.relevance_score is not None else 3.5
        p.final_score = round(
            0.45 * rel
            + 0.20 * 5.0 * rrf_norm
            + 0.15 * 5.0 * impact
            + 0.20 * 5.0 * chain_norm, 4)
    papers.sort(key=lambda p: p.final_score, reverse=True)
    return papers


def run(plan: SearchPlan, results_per_query: int = 10,
        threshold: float = 4.0, top_k: int | None = None) -> list[PaperMeta]:
    """
    检索Agent完整流水线: 检索 -> 粗筛 -> 下载

    参数:
        plan: 检索计划
        top_k: 最终最多保留的论文数（None=不限制，默认取plan.target_count）

    返回:
        list[PaperMeta]，已下载到本地
    """
    top_k = top_k or plan.target_count

    # 检索量随目标动态放大: 每个检索词取 max(results_per_query, top_k*2) 条，
    # 保证粗筛淘汰一半、下载再损失一部分后仍够top_k篇
    per_query = max(results_per_query, top_k * 2)
    all_candidates = search(plan, per_query)
    papers = filter_papers(plan, all_candidates, threshold)

    # ---- 自适应及格线 ----
    # 宽泛主题（如"计算机科学与技术"）下LLM打分普遍偏低，
    # 4.0及格线可能只剩零星几篇。此时降档到3.5重筛一次，
    # 用"相关性略降"换"数量达标"（宁可多留，后续提取/审查还会把关）
    if len(papers) < top_k and threshold > 3.5:
        print(f"[检索Agent] 及格线{threshold}下仅{len(papers)}篇(<目标{top_k}),"
              f"降档到3.5重筛...")
        papers = filter_papers(plan, all_candidates, 3.5)

    # ---- 综合排序: 相关性 + RRF共识 + 影响力 + 引用链 四路加权 ----
    # （公式设计动机详见_composite_sort的docstring。经典论文的
    #   排序保障来自影响力+引用链两路信号，弥补单一关键词命中时
    #   RRF分的天花板，如SLAYER被引472但标题不含"training"字样）
    _composite_sort(papers)
    print("[检索Agent] 综合排序: 0.45*相关性 + 0.20*RRF "
          "+ 0.15*影响力 + 0.20*引用链")

    # ---- 下载策略：小步快跑（P2-1优化，2026-09-14）----
    # 旧策略: 一次下载top_k*2篇再截前top_k——那是出版社403时代的
    # 保险打法（403/HTML吞掉近半才需要双倍余量）。限定arXiv来源后
    # 下载成功率接近100%，双倍余量变成纯浪费（实测下载28篇只用15篇，
    # 白下13篇≈20MB带宽+数分钟重试等待）。
    # 新策略: 首批只下top_k+20%余量（余量供下载后去重损耗），
    # 不足时按综合序继续切片补下——papers已排好序，切片即最优补位
    buffer = top_k // 5 + 1
    downloaded = [p for p in download(papers[:top_k + buffer])
                  if p.local_path]
    idx = top_k + buffer
    while len(downloaded) < top_k and idx < len(papers):
        shortfall = top_k - len(downloaded)
        need = shortfall + shortfall // 5 + 1  # 补位批同样带20%余量
        more = [p for p in download(papers[idx:idx + need])
                if p.local_path]
        downloaded += more
        idx += need

    # ---- 下载后仍不足→降档及格线补位 ----
    # 注意顺序: 必须等下载完成再判断（筛选通过数>=目标不代表下载成功数
    # 也达标，403/HTML会吞掉近半），降档后把"新过线且未下载"的继续下载
    if len(downloaded) < top_k and threshold > 3.5:
        print(f"[检索Agent] 下载后仅{len(downloaded)}/{top_k}篇，"
              f"及格线降档到3.5补位...")
        relaxed = filter_papers(plan, all_candidates, 3.5)
        # 用arxiv_id识别，排除已经下载过的
        done_ids = {p.arxiv_id for p in downloaded}
        extra = [p for p in relaxed
                 if p.arxiv_id not in done_ids and not p.local_path]
        # 补位也用综合排序（相关性+RRF+影响力+引用链），与主排序一致
        extra = _composite_sort(extra)[:top_k]
        if extra:
            more = download(extra)
            downloaded += [p for p in more if p.local_path]

    papers = downloaded
    # ---- 下载后去重 ----
    # arXiv救援会把引用链经典的W-id更新为真实arXiv编号，若该编号
    # 的论文已被关键词检索收录（标题差异大时search阶段去重会漏），
    # 这里按arxiv_id最终去重（保留排名靠前的，dict保持插入序）
    unique = {}
    for p in papers:
        unique.setdefault(p.arxiv_id, p)
    if len(unique) < len(papers):
        print(f"[检索Agent] 下载后去重: {len(papers)} -> {len(unique)}篇")
    papers = list(unique.values())

    # ---- 最终全局重排 ----
    # 补位论文是追加到列表末尾的，其final_score可能高于主体论文
    # （如及格线3.5补位的4.5分相关新论文），必须重算一次保证
    # 输出严格按final_score降序（composite_sort会重算归一化基准）
    if len(papers) > 1:
        _composite_sort(papers)

    if len(papers) > top_k:
        # 截断时同样按综合排序（downloaded保持输入顺序即综合序）
        papers = papers[:top_k]
        print(f"[检索Agent] 下载超额，按综合得分截取前{top_k}篇")

    if len(papers) < top_k:
        print(f"[检索Agent] 注意: 最终只获得{len(papers)}/{top_k}篇"
              f"（候选不足或下载失败过多）")

    # 把结果落盘成JSON（证据库的第一部分，后续Agent从这里读取）
    out_path = os.path.join(config.DATA_DIR, "papers_meta.json")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in papers], f, ensure_ascii=False, indent=2)
    print(f"[检索Agent] 元数据已保存: {out_path}")

    return papers


if __name__ == "__main__":
    # 单独测试：用一个固定的简单计划
    demo_plan = SearchPlan(
        topic="spiking neural network training",
        keywords=["spiking neural network training", "SNN backpropagation"],
        inclusion_criteria="近5年、与SNN训练直接相关、有实验结果",
        target_count=5,
    )
    run(demo_plan)
    llm_client.print_usage()
