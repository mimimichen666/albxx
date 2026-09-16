"""
mode_assistant.py —— 三模式检索（作为流水线的检索阶段）
================================
基于常规学术文献库（多库互补，突破单一数据库局限），
支持三种差异化检索模式（由UI侧边栏「检索方式」明确指定，无需LLM路由）:

  模式1 入门综述学习 —— 经典综述/教程优先（零基础友好）
  模式2 前沿最新突破 —— 近1-5年顶刊/顶会/高关注预印本（时间+影响力双排序）
  模式3 交叉领域检索 —— 跨学科主题: A+B组合词精准定位融合研究

数据源（全部真实API，结果可溯源）:
  Semantic Scholar  期刊/综述/会议论文元数据（被引数、DOI、OA链接、发表日期）
  OpenAlex          学术文献（期刊/会议等类型过滤、年份过滤）
  arXiv API         预印本

防编造设计（信息真实性约束的落地）:
  1. 所有文献条目来自API真实返回，从机制上杜绝编造DOI和链接
  2. 某类源检索失败或0命中时如实标注，绝不虚构该源文献
  3. 模式检索未命中 → 由app.py降级为标准检索继续（仍无结果才终止）

对外接口（仅被app.py流水线调用）:
  retrieve_for_pipeline(query, mode, years, top_k) -> list[PaperMeta]
      三模式检索 + 转PaperMeta + 落盘papers_meta.json
  generate_glossary(papers, topic) -> str | None
      为命中论文生成数量无上限的核心名词速查表（供「论文原文」页旁对照）
"""

import os
import re
import json
import time
import datetime
import sys
import xml.etree.ElementTree as ET

# 直接运行本文件自检时，把项目根目录加入path（app.py引入时不需要）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import html as _htmllib
import requests
from pydantic import BaseModel, Field

import llm_client
import config

# ================================================================
# 一、数据结构（本模块局部使用，不污染models.py）
# ================================================================
class Hit(BaseModel):
    """一条真实检索命中的文献/专利元数据（溯源清单的最小单元）"""
    title: str = Field(description="标题")
    authors: list[str] = Field(default_factory=list, description="作者/发明人（前4位）")
    year: int = Field(default=0, description="发表年份")
    pub_date: str = Field(default="", description="精确发表日期(YYYY-MM-DD)，未知为空")
    source_type: str = Field(default="", description="来源类型: 学位论文/预印本/会议论文/"
                                                     "期刊论文/发明专利/综述文献/论文")
    venue: str = Field(default="", description="出处: 期刊/会议/学位授予单位/专利申请人")
    doi: str = Field(default="", description="DOI号（专利无DOI，arXiv用官方DOI）")
    patent_no: str = Field(default="", description="专利公开号（仅发明专利）")
    url: str = Field(default="", description="原文落地页链接")
    pdf_url: str = Field(default="", description="OA全文直链（非OA为空）")
    oa: bool = Field(default=False, description="是否开放获取")
    cited_by: int = Field(default=0, description="被引次数（专利为0）")
    abstract: str = Field(default="", description="摘要（LLM解读的唯一内容依据）")
    is_review: bool = Field(default=False, description="S2标记的Review类型（模式1筛选用）")
    badge: str = Field(default="", description="影响力徽章: 顶刊/顶会·高被引")


class SourceStatus(BaseModel):
    """单个文献源的检索状态（透明化，失败/0命中如实呈现）"""
    name: str = Field(description="文献源名称，如 学位论文/预印本(arXiv)")
    api: str = Field(description="底层API")
    ok: bool = Field(description="检索是否成功（False=网络/接口失败）")
    count: int = Field(default=0, description="命中条数")
    note: str = Field(default="", description="补充说明")


class ModeQueryPlan(BaseModel):
    """检索词规划结果"""
    keywords: list[str] = Field(
        description="3-4组英文标准术语检索词（每组2-4个空格分隔的词）",
        min_length=1, max_length=5,
    )
    combined_keywords: list[str] = Field(
        default=[],
        description="交叉模式专用: 2-3组组合英文检索词，每组格式为"
                   "「A领域英文短语+B领域英文短语」，用+分隔两个领域的短语",
        max_length=4,
    )
    domains: list[str] = Field(
        default=[],
        description="交叉模式专用: 识别出的两个领域名（中文），如 ['大模型','医疗']",
        max_length=2,
    )


# ================================================================
# 二、LLM提示词
# ================================================================
QUERY_PLAN_SYSTEM = """你是学术检索规划专家，把用户的中文/英文检索需求转成英文检索词。

规则:
1. keywords: 3-4组英文检索词，每组2-4个空格分隔的标准英文术语，覆盖主题不同角度。
   必须使用领域规范英文术语（如"脉冲神经网络"是spiking neural network而不是pulse neural network）。
2. combined_keywords（仅交叉模式需要）: 2-3组组合英文检索词，每组格式为
   「A领域英文短语+B领域英文短语」——用加号+分隔两个领域的核心短语，
   如"large language model+industrial control""retrieval augmented generation+
   operational technology security"。同时把识别出的两个领域写入domains（中文）。
3. 非交叉模式: combined_keywords和domains留空数组。"""

# ================================================================
# 三、检索工具（每源一个轻量封装，均带优雅降级）
# ================================================================
S2_API_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = ("title,abstract,year,authors,externalIds,openAccessPdf,"
             "citationCount,venue,publicationDate,publicationTypes,paperId")

OA_API_BASE = "https://api.openalex.org/works"
OA_HEADERS = {"User-Agent": "literature-agent-mode-assistant/0.1 "
                            "(mailto:course-project@example.com)"}

ARXIV_API_BASE = "http://export.arxiv.org/api/query"

# 影响力启发式（无JCR权限，用venue知名度做顶刊/顶会徽章）
TOP_VENUE_PAT = re.compile(
    r"nature|science\b|cell\b|lancet|new england journal|\bjama\b|\bbmj\b|pnas|"
    r"ieee trans|ieee journal|acm trans|physical review|"
    r"neurips|\bnips\b|icml|iclr|cvpr|iccv|eccv|aaai|ijcai|\bacl\b|emnlp|naacl|"
    r"\bkdd\b|sigir|miccai|icassp|interspeech|sigchi|usenix|icra|iros", re.I)

CONF_VENUE_PAT = re.compile(
    r"proceedings|conference|symposium|workshop|cvpr|iccv|eccv|neurips|\bnips\b|"
    r"icml|iclr|aaai|ijcai|\bacl\b|emnlp|naacl|\bkdd\b|sigir|miccai|icassp|"
    r"interspeech|sigchi|usenix|icra|iros", re.I)

SURVEY_TITLE_PAT = re.compile(
    r"survey|\breview\b|tutorial|overview|primer|state of the art|"
    r"state-of-the-art|\bintroduction to\b", re.I)


def _norm_title(title: str) -> str:
    """标题归一化（去重key）。注意: OpenAlex标题里常嵌字面"\\n"两字符，
    必须先替换成空格再去符号，否则同一论文的两个记录归一化结果不同"""
    t = (title or "").replace("\\n", " ").replace("\n", " ")
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", t.lower())


def _dedup(hits: list[Hit]) -> list[Hit]:
    """按归一化标题去重，保留先出现者（先出现的来自更靠前的源/排序）"""
    seen, out = set(), []
    for h in hits:
        key = _norm_title(h.title)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _apply_badges(hits: list[Hit], high_cited: int):
    """影响力徽章: 顶刊/顶会（venue启发式）+ 高被引"""
    for h in hits:
        badges = []
        if TOP_VENUE_PAT.search(h.venue):
            badges.append("顶刊/顶会")
        if h.cited_by >= high_cited:
            badges.append(f"高被引({h.cited_by})")
        h.badge = "·".join(badges)


def _abstract_from_inverted(inverted_index: dict | None) -> str:
    """OpenAlex倒排索引摘要 → 正常文本"""
    if not inverted_index:
        return ""
    positions = []
    for word, idx_list in inverted_index.items():
        for idx in idx_list:
            positions.append((idx, word))
    positions.sort()
    return " ".join(w for _, w in positions)


# ---- 3.1 OpenAlex（支持文献类型过滤，是学位论文/预印本/会议/期刊四类源的底座）----
def _oa_search(kw: str, limit: int, work_type: str | None = None,
               year_from: int | None = None, source_type: str = "论文",
               sort: str = "cited_by_count:desc") -> list[Hit]:
    filters = [f"title_and_abstract.search:{kw}"]
    if work_type:
        filters.append(f"type:{work_type}")
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    try:
        resp = requests.get(OA_API_BASE, params={
            "filter": ",".join(filters),
            "per_page": min(limit, 25),
            "sort": sort,
            "mailto": "course-project@example.com",  # 礼貌池
        }, headers=OA_HEADERS, timeout=25)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[模式检索] OpenAlex请求失败({kw}, type={work_type}): {e}")
        return []
    time.sleep(0.6)  # 礼貌间隔

    hits = []
    for w in data.get("results", []):
        title = (w.get("title") or "").replace("\\n", " ").strip()
        title = re.sub(r"\s+", " ", title)
        if not title:
            continue
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        loc = w.get("primary_location") or {}
        src = ((loc.get("source") or {}).get("display_name")) or ""
        oa_loc = w.get("best_oa_location") or {}
        url = loc.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else "")
        hits.append(Hit(
            title=title,
            authors=[a["author"]["display_name"]
                     for a in (w.get("authorships") or [])[:4]],
            year=w.get("publication_year") or 0,
            pub_date=w.get("publication_date") or "",
            source_type=source_type,
            venue=src,
            doi=doi,
            url=url,
            pdf_url=oa_loc.get("pdf_url") or "",
            oa=bool((w.get("open_access") or {}).get("is_oa")),
            cited_by=w.get("cited_by_count") or 0,
            abstract=_abstract_from_inverted(
                w.get("abstract_inverted_index"))[:1200],
        ))
    return hits


# ---- 3.2 Semantic Scholar（综述/期刊/会议元数据，被引数与发表日期质量高）----
def _s2_request(params: dict, max_retries: int = 2) -> dict | None:
    """带限流处理的S2请求（策略与searcher._s2_request一致: 宁慢勿堵）"""
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}
    for attempt in range(max_retries):
        try:
            resp = requests.get(S2_API_BASE, params=params,
                                headers=headers, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = (int(retry_after) + 2 if retry_after else 15 * (attempt + 1))
                print(f"[模式检索] S2限流429，等待{wait}秒({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(2)  # 官方限流1 req/s，成功后强制间隔
            return resp.json()
        except Exception as e:
            print(f"[模式检索] S2请求异常: {e}")
            time.sleep(4)
    return None


def _s2_search(kw: str, limit: int, year_from: int | None = None) -> list[Hit]:
    params = {"query": kw, "limit": min(limit, 20), "fields": S2_FIELDS}
    if year_from:
        params["year"] = f"{year_from}-{datetime.date.today().year}"
    data = _s2_request(params)
    if not data:
        return []
    hits = []
    for item in data.get("data", []):
        title = (item.get("title") or "").strip()
        if not title:
            continue
        ext = item.get("externalIds") or {}
        doi = ext.get("DOI") or ""
        oa = item.get("openAccessPdf") or {}
        venue = (item.get("venue") or "").strip()
        ptypes = item.get("publicationTypes") or []
        is_review = "Review" in ptypes
        if is_review:
            stype = "综述文献"
        elif CONF_VENUE_PAT.search(venue):
            stype = "会议论文"
        elif venue:
            stype = "期刊论文"
        else:
            stype = "论文"
        paper_id = item.get("paperId") or ""
        hits.append(Hit(
            title=title,
            authors=[a.get("name", "") for a in (item.get("authors") or [])[:4]],
            year=item.get("year") or 0,
            pub_date=item.get("publicationDate") or "",
            source_type=stype,
            venue=venue,
            doi=doi,
            url=(f"https://doi.org/{doi}" if doi else
                 (f"https://www.semanticscholar.org/paper/{paper_id}"
                  if paper_id else "")),
            pdf_url=oa.get("url") or "",
            oa=bool(oa.get("url")),
            cited_by=item.get("citationCount") or 0,
            abstract=(item.get("abstract") or "").replace("\n", " ")[:1200],
            is_review=is_review,
        ))
    return hits


# ---- 3.3 arXiv（预印本；该域名实测时通时断，加重试）----
def _arxiv_search_html(kw: str, limit: int) -> list[Hit]:
    """
    arXiv HTML搜索兜底: export.arxiv.org API被网络阻断（连接重置）时，
    改走 https://arxiv.org/search/ 页面检索——实测API阻断期间
    主站页面仍可正常访问（2026-09-09: API三变体全reset、主站200）。
    """
    try:
        resp = requests.get(
            "https://arxiv.org/search/",
            params={"query": kw, "searchtype": "all", "size": 50},
            timeout=30, headers={"User-Agent": "literature-agent/0.1"})
        resp.raise_for_status()
    except Exception as e:
        print(f"[模式检索] arXiv HTML兜底失败: {e}")
        return []
    time.sleep(3)  # arXiv官方礼貌限速

    items = re.findall(r'<li class="arxiv-result">(.*?)</li>',
                       resp.text, re.S)
    _months = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
               "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
               "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}

    def _clean(s: str) -> str:
        # 高亮span在词中间时（如 Spike</span>-Timing）直接去除，
        # 避免标题被拆成 "Spike -Timing"；其余标签替换为空白分隔
        s = re.sub(r"</?span[^>]*>", "", s)
        return " ".join(_htmllib.unescape(
            re.sub(r"<[^>]+>", " ", s)).split())

    hits = []
    for it in items[:limit]:
        m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}"
                      r"|[a-z\-]+/\d{7})", it)
        if not m:
            continue
        arxiv_id = m.group(1)
        tm = re.search(r'<p class="title is-5 mathjax">(.*?)</p>', it, re.S)
        title = _clean(tm.group(1)) if tm else ""
        if not title:
            continue
        am = re.search(r'<p class="authors">(.*?)</p>', it, re.S)
        authors = ([_clean(a) for a in
                    re.findall(r">([^<>]+)</a>", am.group(1))] if am else [])[:4]
        # 摘要: 取abstract-full到段落结束</p>——中途的内嵌高亮span
        # 会让"到</span>"的匹配提前截断（实测30字符就断了）
        bm = re.search(r'class="abstract-full[^"]*"[^>]*>(.*?)</p>',
                       it, re.S)
        abstract = _clean(bm.group(1)) if bm else ""
        abstract = re.sub(r"△ Less\s*$", "", abstract).strip()
        pub = ""
        # 日期: 实际格式为 "Submitted</span> 7 September, 2026"（标签分隔）
        dm = re.search(r"Submitted\s*(?:</span>)?\s*"
                       r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", it)
        if dm and dm.group(2)[:3].title() in _months:
            pub = (f"{dm.group(3)}-{_months[dm.group(2)[:3].title()]}"
                   f"-{int(dm.group(1)):02d}")
        hits.append(Hit(
            title=title,
            authors=[a for a in authors if a],
            year=int(pub[:4]) if pub[:4].isdigit() else 0,
            pub_date=pub,
            source_type="预印本",
            venue="arXiv预印本",
            doi=f"10.48550/arXiv.{arxiv_id}",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            oa=True,
            abstract=abstract[:1200],
        ))
    return hits


def _arxiv_search(kw: str, limit: int) -> list[Hit]:
    query = " OR ".join(f"all:{w}" for w in kw.split())
    resp = None
    for attempt in range(3):  # arXiv域名间歇性阻断（README 2026-09实测记录），快速重试
        try:
            resp = requests.get(ARXIV_API_BASE, params={
                "search_query": query,
                "max_results": min(limit, 15),
                "sortBy": "relevance",
            }, timeout=30)
            resp.raise_for_status()
            break
        except Exception as e:
            print(f"[模式检索] arXiv请求失败(第{attempt + 1}次): {e}")
            resp = None
            time.sleep(3)

    hits = []
    if resp is not None:
        time.sleep(3)  # arXiv官方礼貌限速
        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(resp.text)
        except Exception as e:
            print(f"[模式检索] arXiv XML解析失败: {e}")
            root = None
        if root is not None:
            for entry in root.findall("a:entry", ns):
                raw_id = (entry.findtext("a:id", "", ns) or "").rstrip("/")
                arxiv_id = raw_id.split("/abs/")[-1]
                arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                title = " ".join((entry.findtext("a:title", "", ns) or "").split())
                abstract = " ".join((entry.findtext("a:summary", "", ns) or "").split())
                if not arxiv_id or not title:
                    continue
                pub = (entry.findtext("a:published", "", ns) or "")[:10]
                authors = [a.findtext("a:name", "", ns)
                           for a in entry.findall("a:author", ns)][:4]
                doi = (entry.findtext("a:doi", "", ns) or "").strip()
                hits.append(Hit(
                    title=title,
                    authors=authors,
                    year=int(pub[:4]) if pub[:4].isdigit() else 0,
                    pub_date=pub,
                    source_type="预印本",
                    venue="arXiv预印本",
                    doi=doi or f"10.48550/arXiv.{arxiv_id}",
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    oa=True,
                    abstract=abstract[:1200],
                ))
    if not hits:
        # API阻断/解析失败/0命中 → 主站HTML搜索兜底
        print("[模式检索] arXiv API不可用，切换arxiv.org主站HTML搜索兜底")
        hits = _arxiv_search_html(kw, limit)
    return hits


# ================================================================
# 四、检索规划（模式由UI侧边栏明确指定，无需LLM路由）
# ================================================================
def build_query_plan(query: str, mode: str) -> ModeQueryPlan:
    """把检索需求转成英文检索词（交叉模式额外产出A+B组合词）"""
    desc = {
        "mode1": "入门综述学习模式（检索经典综述/教程类文献）",
        "mode2": "前沿突破检索模式（检索近1-3年顶刊顶会最新成果）",
        "mode3": "交叉领域检索模式（主题横跨两个领域，组合关键词精准定位融合研究）",
    }[mode]
    return llm_client.chat_json(
        [{"role": "system", "content": QUERY_PLAN_SYSTEM},
         {"role": "user", "content": f"检索需求: {query}\n检索模式: {desc}"}],
        schema_class=ModeQueryPlan, temperature=0.4,
    )


# ================================================================
# 五、三模式检索执行
# ================================================================
def _retrieve_mode1(plan: ModeQueryPlan) -> tuple[list[Hit], list[SourceStatus]]:
    """模式1: 经典综述/教程检索（S2 + OpenAlex，按被引排序=权威性优先）"""
    kws = plan.keywords[:2]
    s2_all, oa_all = [], []
    for kw in kws:
        for suffix in ("survey", "review"):
            s2_all.extend(_s2_search(f"{kw} {suffix}", 8))
    for kw in kws:
        for suffix in ("survey", "review"):
            oa_all.extend(_oa_search(f"{kw} {suffix}", 6, work_type="article",
                                     source_type="综述文献"))

    # 综述判定: 标题含综述词 或 S2标记为Review类型
    cand = [h for h in s2_all + oa_all
            if SURVEY_TITLE_PAT.search(h.title) or h.is_review]
    for h in cand:
        h.source_type = "综述文献"
    hits = _dedup(cand)

    # S2+OpenAlex双双不可达（429限流）→ arXiv应急（含大量综述预印本，
    # 其HTML兜底路径在API阻断期间仍稳定可达）
    arx_status = None
    if not hits:
        arx_all = []
        for kw in kws:
            for suffix in ("survey", "review"):
                arx_all.extend(_arxiv_search(f"{kw} {suffix}", 6))
        cand = [h for h in arx_all if SURVEY_TITLE_PAT.search(h.title)]
        for h in cand:
            h.source_type = "综述文献"
        hits = _dedup(cand)
        if hits:
            arx_status = SourceStatus(
                name="综述文献(应急)", api="arXiv(主站HTML兜底)",
                ok=True, count=len(hits),
                note="S2/OpenAlex限流期间的应急源")

    hits.sort(key=lambda h: h.cited_by, reverse=True)  # 经典综述=被引优先
    hits = hits[:10]
    _apply_badges(hits, high_cited=150)

    statuses = [
        SourceStatus(name="综述文献(主源)", api="Semantic Scholar",
                     ok=bool(s2_all), count=len(s2_all),
                     note="" if s2_all else "S2不可达或无命中"),
        SourceStatus(name="综述文献(补充)", api="OpenAlex",
                     ok=bool(oa_all), count=len(oa_all),
                     note="" if oa_all else "OpenAlex不可达或无命中"),
    ]
    if arx_status:
        statuses.append(arx_status)
    return hits, statuses


def _retrieve_mode2(plan: ModeQueryPlan,
                    years: int) -> tuple[list[Hit], list[SourceStatus]]:
    """模式2: 近N年最新突破（S2年份过滤 + OpenAlex，时间+影响力双排序）"""
    year_from = datetime.date.today().year - years + 1
    s2_all, oa_all = [], []
    for kw in plan.keywords[:3]:
        s2_all.extend(_s2_search(kw, 10, year_from=year_from))
    for kw in plan.keywords[:2]:
        oa_all.extend(_oa_search(kw, 10, year_from=year_from,
                                 source_type="期刊论文"))

    hits = _dedup([h for h in s2_all + oa_all if h.abstract])

    # S2+OpenAlex双双不可达（429限流）→ arXiv应急（预印本本就是
    # 最新突破的典型载体；年份近N年过滤，命中为空时放宽到全部年份）
    arx_status = None
    if not hits:
        arx_all = []
        for kw in plan.keywords[:2]:
            arx_all.extend(_arxiv_search(kw, 8))
        cand = [h for h in arx_all
                if (h.pub_date or str(h.year or ""))[:4] >= str(year_from)]
        hits = _dedup(cand or arx_all)
        if hits:
            arx_status = SourceStatus(
                name=f"预印本({years}年内,应急)", api="arXiv(主站HTML兜底)",
                ok=True, count=len(hits),
                note="S2/OpenAlex限流期间的应急源")

    # 「发表时间+学术影响力」双重排序: 先比精确发表日期，同日比被引
    hits.sort(key=lambda h: (h.pub_date or str(h.year or ""), h.cited_by),
              reverse=True)
    hits = hits[:12]
    _apply_badges(hits, high_cited=30)  # 近年成果被引门槛降低

    statuses = [
        SourceStatus(name=f"期刊/会议论文({years}年内)", api="Semantic Scholar",
                     ok=bool(s2_all), count=len(s2_all),
                     note="" if s2_all else "S2不可达或无命中"),
        SourceStatus(name=f"期刊/会议论文({years}年内,补充)", api="OpenAlex",
                     ok=bool(oa_all), count=len(oa_all),
                     note="" if oa_all else "OpenAlex不可达或无命中"),
    ]
    if arx_status:
        statuses.append(arx_status)
    return hits, statuses


def _retrieve_mode3(plan: ModeQueryPlan) -> tuple[list[Hit], list[SourceStatus]]:
    """
    模式3: 交叉领域检索（常规文献源，非五源异构）。
    检索源: Semantic Scholar + OpenAlex + arXiv（与模式1/2同源，
    突破单一数据库局限的多库互补），期刊/会议/预印本自然覆盖。
    交叉检索词策略: 「A领域短语+B领域短语」组合词，
    两个领域的词必须同时出现（剔除单一领域无关文献）。
    """
    # A+B组合词优先，不足时用通用词兜底
    kws = (plan.combined_keywords[:2] + plan.keywords[:1]) or plan.keywords[:2]

    # ---- 组合词转精确检索语法 ----
    # "large language model+industrial control" ->
    #   S2:       large language model industrial control（词组AND语义）
    #   OpenAlex:  "large language model" "industrial control"（短语AND）
    #   arXiv:     all:"large language model" AND all:"industrial control"
    def _split_cross(kw: str) -> list[str]:
        parts = [p.strip() for p in kw.split("+") if p.strip()]
        return parts if len(parts) >= 2 else []

    def _s2_kw(kw: str) -> str:
        parts = _split_cross(kw)
        return " ".join(parts) if parts else kw

    def _oa_kw(kw: str) -> str:
        parts = _split_cross(kw)
        if parts:
            return " ".join(f'"{p}"' for p in parts)
        return f'"{kw}"'  # 单领域词也加短语引号，提升精度

    def _arxiv_kw(kw: str) -> str:
        parts = _split_cross(kw)
        if parts:
            return " AND ".join(f'all:"{p}"' for p in parts)
        return " AND ".join(f"all:{w}" for w in kw.split())

    # ---- 源1: Semantic Scholar（期刊/会议论文，组合词AND语义）----
    s2_hits: list[Hit] = []
    for kw in kws[:2]:
        s2_hits.extend(_s2_search(_s2_kw(kw), 10))

    # ---- 源2: OpenAlex（期刊/会议论文补充，短语AND精确）----
    oa_hits: list[Hit] = []
    for kw in kws[:2]:
        oa_hits.extend(_oa_search(_oa_kw(kw), 8, work_type="article",
                                  source_type="期刊/会议论文",
                                  sort="relevance_score:desc"))
    # 按venue细分标注（与S2分类口径一致）
    for h in oa_hits:
        h.source_type = "会议论文" if CONF_VENUE_PAT.search(h.venue) \
            else "期刊论文"

    # ---- 源3: arXiv（预印本，交叉主题预印本覆盖率高）----
    ax_hits: list[Hit] = []
    for kw in kws[:2]:
        ax_hits.extend(_arxiv_search(_arxiv_kw(kw), 6))

    hits = _dedup(s2_hits + oa_hits + ax_hits)
    hits = hits[:20]
    _apply_badges(hits, high_cited=50)

    statuses = [
        SourceStatus(name="期刊/会议论文(主源)", api="Semantic Scholar",
                     ok=bool(s2_hits), count=len(s2_hits),
                     note="" if s2_hits else "S2不可达或无命中"),
        SourceStatus(name="期刊/会议论文(补充)", api="OpenAlex",
                     ok=bool(oa_hits), count=len(oa_hits),
                     note="" if oa_hits else "OpenAlex不可达或无命中"),
        SourceStatus(name="预印本", api="arXiv API",
                     ok=bool(ax_hits), count=len(ax_hits),
                     note="" if ax_hits else "arXiv不可达或无命中（域名间歇阻断时重试）"),
    ]
    return hits, statuses


# ================================================================
# 六、核心名词速查表（全模式通用，供「论文原文」页旁对照）
# ================================================================
def generate_glossary(papers: list, topic: str = "") -> str | None:
    """
    为流水线模式检索命中的论文独立生成「核心名词速查表」。

    场景: 侧边栏选三模式跑完整流水线时不生成模式报告，这里
    用检索命中的标题+摘要直接产出速查表，存入会话供
    「论文原文」页旁对照（与报告内速查表同一规格: 数量无上限）。

    返回: 表格markdown，失败返回None（不影响流水线其余环节）。
    """
    src = []
    for i, p in enumerate(papers[:12], 1):
        title = getattr(p, "title", "") or ""
        abstract = (getattr(p, "abstract", "") or "")[:600]
        src.append(f"[{i}] {title}\n{abstract}")
    if not src:
        return None

    sys_prompt = (
        "你是科普编辑。根据给定文献的标题与摘要，整理一份「核心名词速查表」。\n"
        "要求:\n"
        "1. 收录这些文献涉及的所有专业术语、缩写、关键概念，"
        "数量不设上限、宁多勿漏\n"
        "2. 每条用高中生能懂的通俗语言解释（可配生活化类比），"
        "禁止未解释的学术黑话\n"
        "3. 只输出两列markdown表格，不要输出任何其他文字:\n"
        "| 术语 | 通俗解释 |\n|---|---|"
    )
    try:
        out = llm_client.chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": "\n\n".join(src)}],
            temperature=0.2)
    except Exception as e:
        print(f"[模式检索] 速查表生成失败(不影响流水线): {e}")
        return None
    # 容错: 只保留表格行（LLM偶尔会在前后加说明文字）
    lines = [ln.strip() for ln in (out or "").splitlines()
             if ln.strip().startswith("|")]
    return "\n".join(lines) if len(lines) >= 3 else None


# ================================================================
# 七、完整流水线桥接（三模式检索作为流水线的检索阶段）
# ================================================================
_ARXIV_ID_PAT = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})")
_ARXIV_DOI_PAT = re.compile(r"10\.48550/arXiv\.([0-9]{4}\.[0-9]{4,5})")


def _hit_to_paper(h: Hit) -> "PaperMeta":
    """
    Hit -> PaperMeta（流水线全流程的文献主键是 arxiv_id，
    非arXiv命中用DOI/标题生成文件名安全的替代ID）
    """
    from models import PaperMeta

    # 替代ID生成: arXiv真id优先 > DOI清洗 > 标题归一
    pid = None
    m = _ARXIV_ID_PAT.search(h.url or "") or _ARXIV_ID_PAT.search(h.pdf_url or "")
    if not m:
        m = _ARXIV_DOI_PAT.search(h.doi or "")
    if m:
        pid = m.group(1)
    elif h.doi:
        pid = "doi_" + re.sub(r"[^0-9A-Za-z.\-]", "_", h.doi)
    else:
        pid = "hit_" + re.sub(r"[^0-9A-Za-z]+", "_", h.title)[:48] + \
            f"_{abs(hash(h.title)) % 10000}"

    return PaperMeta(
        arxiv_id=pid,
        title=h.title,
        authors=h.authors[:5],
        year=h.year or 0,
        abstract=h.abstract or "",
        pdf_url=h.pdf_url or "",
        cited_by=h.cited_by or 0,
    )


def retrieve_for_pipeline(query: str, mode: str, years: int = 2,
                          top_k: int = 5) -> list["PaperMeta"]:
    """
    三模式检索 → 流水线桥接: 用模式化检索词规划+多库检索拿到文献，
    转成 PaperMeta 并落盘 papers_meta.json，后续提取/审查/综合照常执行。

    mode由UI侧边栏明确指定（不走LLM模式路由）；
    只保留带PDF直链的命中（流水线的提取Agent与证据定位都依赖本地PDF）。

    返回: 已落盘的 PaperMeta 列表（尚未下载PDF，下载由searcher.download承担）
    """
    plan = build_query_plan(query, mode)
    print(f"[模式检索→流水线] 检索词: {plan.keywords} | "
          f"组合词: {plan.combined_keywords}")

    if mode == "mode1":
        hits, _ = _retrieve_mode1(plan)
    elif mode == "mode2":
        hits, _ = _retrieve_mode2(plan, years)
    else:
        hits, _ = _retrieve_mode3(plan)

    papers = []
    for h in hits:
        if not h.pdf_url:
            continue  # 无PDF直链的文献无法进入提取/证据定位环节
        try:
            papers.append(_hit_to_paper(h))
        except Exception as e:
            print(f"[模式检索→流水线] 转换失败跳过: {h.title[:40]} | {e}")
        if len(papers) >= top_k:
            break

    # 落盘 papers_meta.json（与searcher.run()的输出约定一致，
    # 后续extractor/reviewer/synthesizer从data/目录读取）
    out_path = os.path.join(config.DATA_DIR, "papers_meta.json")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in papers], f,
                  ensure_ascii=False, indent=2)
    print(f"[模式检索→流水线] 元数据已保存: {out_path} "
          f"({len(papers)}篇，PDF下载由检索Agent承担)")
    return papers


# ---------------------------------------------------------------
# 单独运行本文件的检索通路自检（不调用LLM）
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("== 自检1: OpenAlex 常规论文源（模式3用） ==")
    for h in _oa_search('"large language model" "industrial control"', 3,
                        work_type="article",
                        source_type="期刊/会议论文"):
        print(" -", h.title[:60], h.year, h.source_type)
    print("== 自检2: arXiv 预印本源 ==")
    for h in _arxiv_search("retrieval augmented generation", 3):
        print(" -", h.title[:60], h.pub_date, h.doi)
    print("== 自检3: S2 期刊/会议论文源 ==")
    for h in _s2_search("large language model industrial control", 3):
        print(" -", h.title[:60], h.year, h.source_type, h.cited_by)
