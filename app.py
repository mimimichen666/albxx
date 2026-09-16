"""
app.py —— Streamlit前端（Demo展示界面）
================================
启动方法:
  conda activate brain
  cd d:\trae项目\literature_agent
  streamlit run app.py

界面结构:
  侧边栏: 主题输入 + 目标篇数 + 启动按钮 + 说明
  主区域:
    - 运行状态区: 四个Agent的实时进度
    - 结果标签页: 📄信息卡片 | ✅审查明细 | 📊综述表格 | 🕸引用图谱 | 📝最终报告
  复用按钮: 已有数据时可直接展示产出（不用重新跑流水线）
"""

import os
import sys
import json
import time

import streamlit as st

# 项目根目录加入path（streamlit的工作目录可能不同）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------
# 云端 Secrets 同步到环境变量（兼容 Streamlit Cloud / HF Spaces）
# ---------------------------------------------------------------
# 本地开发用 .env 文件，云端部署用平台 Secrets（st.secrets 读取）。
# 这里把云端 secrets 值同步到环境变量，让 config.py 用同一套
# os.environ 读取逻辑就能拿到值，无需改 config.py
try:
    _keys_to_sync = ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                     "S2_API_KEY", "APP_PASSWORD"]
    for _k in _keys_to_sync:
        if _k not in os.environ and _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:
    # 没有 secrets 文件 / 本地开发 / 任何原因异常 → 静默跳过
    # 密码门会直接查 st.secrets 做兜底，同步失败不影响密码门
    pass

import config
from models import PaperMeta, PaperCard, ReviewResult

# 证据图片目录（方案一：证据链穿透）
EVIDENCE_DIR = os.path.join(config.PROJECT_ROOT, "evidence")

# ---------------------------------------------------------------
# 页面全局配置
# ---------------------------------------------------------------
st.set_page_config(
    page_title="科研文献整理Agent",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------
# 访问密码门（云端部署防护）
# ---------------------------------------------------------------
# 部署到公网时，陌生访客必须先输密码才能用——否则他们随手点一下
# 「完整运行」就会触发流水线，消耗你的 LLM API 额度（一次几十次调用）
# 密码来源（优先级从高到低）:
#   1. config.APP_PASSWORD（本地 .env 或环境变量）
#   2. st.secrets["APP_PASSWORD"]（Streamlit Cloud / HF Spaces 平台 Secrets）
try:
    _pwd = config.APP_PASSWORD or (
        st.secrets.get("APP_PASSWORD", "") if "APP_PASSWORD" in st.secrets else ""
    )
except Exception:
    _pwd = config.APP_PASSWORD

if _pwd:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        # 未登录: 只渲染一个简洁登录页，主应用所有内容都不要执行
        st.markdown("""
        <div style="
            max-width: 360px; margin: 8vh auto; padding: 32px 28px;
            background: #FFFFFF; border: 1px solid #E2E8F0;
            border-radius: 14px; box-shadow: 0 8px 24px rgba(15,23,42,.08);
            text-align: center;">
            <div style="font-size: 2rem;">📚</div>
            <div style="font-size: 1.3rem; font-weight: 800;
                        color: #4F46E5; margin: 8px 0 4px;">
                科研文献整理 Agent
            </div>
            <div style="color: #64748B; font-size: .9rem; margin-bottom: 18px;">
                请输入访问密码
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.form("login_form"):
            pwd = st.text_input("访问密码", type="password")
            submitted = st.form_submit_button("登录", type="primary",
                                              use_container_width=True)
            if submitted:
                if pwd == _pwd:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("密码错误，请重试")
        st.stop()

# ---------------------------------------------------------------
# 全局视觉设计系统（对齐主流SaaS产品观感）
# 设计语言: 靛紫渐变主色 + 卡片化布局 + 柔和阴影 + 圆角
# 只用系统字体栈（不依赖外网字体，国内网络环境稳定）
# ---------------------------------------------------------------
st.markdown("""
<style>
/* ===== 全局基础 ===== */
:root {
    --primary: #4F46E5;       /* 主色: 靛蓝 */
    --primary-dark: #4338CA;
    --accent: #7C3AED;        /* 强调色: 紫罗兰 */
    --bg-soft: #F8FAFC;       /* 页面浅底 */
    --card-border: #E2E8F0;
    --text-main: #0F172A;
}
html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "PingFang SC", "Microsoft YaHei", "Noto Sans SC",
                 sans-serif;
    color: var(--text-main);
}
.stApp { background: var(--bg-soft); }
/* 主区域内容最大宽度收拢（过宽降低阅读舒适度） */
.block-container { padding-top: 1.6rem; max-width: 1400px; }

/* ===== 侧边栏: 品牌化渐变头部 ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F5F3FF 100%);
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] h1 {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: .5px;
}
/* 侧边栏输入控件统一圆角 */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSlider > div {
    border-radius: 10px;
}

/* ===== 按钮: 渐变主按钮 + 圆角次按钮 ===== */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid var(--card-border);
    transition: all .18s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(79, 70, 229, .18);
}
button[kind="primary"], [data-testid="baseButton-primary"] {
    background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    color: #FFF !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(79, 70, 229, .35);
}

/* ===== 顶部统计卡片(metric): 白卡+悬浮上浮 ===== */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, .06);
    transition: all .18s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(79, 70, 229, .12);
    border-color: #C7D2FE;
}
[data-testid="stMetricValue"] {
    color: var(--primary);
    font-weight: 800;
}
[data-testid="stMetricLabel"] { font-weight: 600; color: #475569; }

/* ===== 标签页: 药丸风格 ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 2px solid #EEF2FF;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 6px 16px;
    font-weight: 600;
    color: #64748B;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #EEF2FF, #F5F3FF);
    color: var(--primary) !important;
    border-radius: 999px;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ===== 折叠面板(expander): 卡片化 ===== */
[data-testid="stExpander"] details {
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    background: #FFFFFF;
    overflow: hidden;
    transition: box-shadow .18s ease;
}
[data-testid="stExpander"] details:hover {
    box-shadow: 0 3px 12px rgba(15, 23, 42, .07);
}
[data-testid="stExpander"] summary { font-weight: 600; }

/* ===== 表单控件标签加粗 ===== */
[data-baseweb="radio"] span, .stSelectbox label,
.stTextInput label, .stSlider label { font-weight: 600; }

/* ===== Markdown链接与分隔线 ===== */
a { color: var(--primary); }
hr { border-color: #EEF2FF; }

/* ===== 滚动条 ===== */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb {
    background: #C7D2FE; border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: #A5B4FC; }

/* ===== 亮色主题防御（配合.streamlit/config.toml的theme锁定） =====
   背景: 用户浏览器端曾持久化深色主题，仅CSS改浅背景会白字白底。
   下方规则确保正文文字在任何残留主题下都是深色可读。
   注意: Hero横幅用内联样式白字，不受此影响（渐变深底正确显示） */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stExpander"], .stTabs, .stRadio, .stSelectbox,
.stTextInput, .stSlider, .stMetric {
    color: #0F172A !important;
}
</style>
""", unsafe_allow_html=True)

# 会话状态: 存放流水线的中间产出（跨标签页共享，避免重复计算）
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "run_outputs" not in st.session_state:
    st.session_state.run_outputs = {}


# ---------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(path: str, cls=None):
    """读取data/或output/下的JSON（缓存，切标签页不重读）"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if cls:
        return [cls(**item) for item in data]
    return data


@st.cache_data(show_spinner=False)
def load_text(path: str):
    """读取文本文件（报告/表格/图谱HTML）"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------
# 流水线执行（封装四Agent，带进度显示）
# ---------------------------------------------------------------
def run_pipeline(topic: str, target_count: int,
                 retrieval_mode: str = "standard", mode_years: int = 2):
    """
    执行 规划->检索->提取->审查->综合 完整流水线

    retrieval_mode:
        "standard"        标准检索（规划Agent生成检索词，S2主源三段管线）
        "mode1"/"mode2"/"mode3" 三模式检索作为检索阶段
        （检索词规划与多库检索由模式检索承担，后续提取/审查/综合不变）
    """
    from agents import planner, searcher, extractor, reviewer, synthesizer
    import llm_client

    # ---- 1+2. 检索阶段（标准 / 三模式 两条路径；
    # 模式未命中先降级标准检索，降级后仍未命中才终止）----
    _use_mode = retrieval_mode in ("mode1", "mode2", "mode3")
    _mode_miss = False  # 模式检索未命中 → 已降级为标准检索
    papers = []
    if _use_mode:
        from agents import mode_assistant

        _mode_names = {"mode1": "🌱 入门综述", "mode2": "🚀 前沿突破",
                       "mode3": "🔀 交叉领域"}
        status = st.status(
            f"🎯 {_mode_names[retrieval_mode]}模式检索: "
            "规划检索词+多库检索中...", expanded=True)
        papers = mode_assistant.retrieve_for_pipeline(
            topic, retrieval_mode, mode_years, top_k=target_count)
        if papers:
            st.write(f"模式检索命中 {len(papers)} 篇（带PDF直链），"
                     "开始下载PDF...")
            status.update(
                label=f"🎯 {_mode_names[retrieval_mode]}检索完成: "
                      f"{len(papers)}篇候选", state="complete")
        else:
            # 用户要求: 模式检索未命中 → 先降级为标准检索继续
            st.warning("🎯 模式检索未命中带PDF直链的文献，"
                       "已自动切换为**标准检索**继续...")
            status.update(label="⚠️ 模式检索未命中，已切换为标准检索",
                          state="complete")
            _use_mode = False
            _mode_miss = True
    if not _use_mode:
        # ---- 1. 规划Agent（标准路径）----
        status = st.status("🧠 规划Agent: 分解研究主题...", expanded=True)
        plan = planner.make_plan(topic)
        st.session_state.run_outputs["plan"] = plan
        status.update(label=f"🧠 规划Agent完成: {len(plan.keywords)}组检索词",
                      state="complete")

    # ---- 2. 检索Agent: PDF下载（两条路径共用下载与筛选收尾）----
    status = st.status("🔍 检索Agent: 下载论文PDF中（arXiv限速,约需1-3分钟）...",
                       expanded=True)
    if _use_mode:
        # 模式路径: papers已落盘元数据，这里只补PDF下载（含arXiv救援）
        papers = searcher.download(papers)
        # 下载后重写元数据（剔除下载失败项，与searcher.run落盘口径一致，
        # 后续extractor/reviewer只处理有本地PDF的论文）
        if papers:
            with open(os.path.join(config.DATA_DIR, "papers_meta.json"),
                      "w", encoding="utf-8") as f:
                json.dump([p.model_dump() for p in papers], f,
                          ensure_ascii=False, indent=2)
    else:
        st.write("检索词: " + ", ".join(plan.keywords))
        papers = searcher.run(plan, results_per_query=8, top_k=target_count)
    if len(papers) == 0:
        # 用户要求: 模式检索降级标准检索后仍未命中 → 终止流水线
        if _mode_miss:
            st.error("❌ 模式检索未命中，流水线终止")
            status.update(label="❌ 模式检索未命中，流水线终止",
                          state="error")
        else:
            status.update(label="❌ 检索失败: 没有获得可用论文", state="error")
        st.stop()

    # 模式检索附加（含降级标准检索后命中的情况）: 生成核心名词速查表
    # （无上限），存入会话+落盘，供「论文原文」页旁随时对照查看
    if _use_mode or _mode_miss:
        try:
            st.write("生成核心名词速查表（供论文原文页对照）...")
            _g = mode_assistant.generate_glossary(papers, topic)
            if _g:
                st.session_state.glossary_md = _g
                st.session_state.glossary_topic = topic
                with open(os.path.join(config.DATA_DIR, "glossary_last.json"),
                          "w", encoding="utf-8") as f:
                    json.dump({"topic": topic, "md": _g}, f,
                              ensure_ascii=False, indent=1)
        except Exception as _g_err:
            st.write(f"速查表生成失败(不影响流水线): {_g_err}")
    st.write(f"获得 {len(papers)} 篇论文, PDF已下载")
    status.update(label=f"🔍 检索Agent完成: {len(papers)}篇论文", state="complete")

    # ---- 3. 提取Agent ----
    status = st.status("📄 提取Agent: 抽取信息卡片中（每篇约1分钟）...",
                       expanded=True)
    cards = extractor.run(papers)
    status.update(label=f"📄 提取Agent完成: {len(cards)}张卡片, "
                        f"{sum(len(c.claims) for c in cards)}条声明",
                  state="complete")

    # ---- 4. 审查Agent ----
    status = st.status("✅ 审查Agent: 两级核验中（程序对齐+语义审查）...",
                       expanded=True)
    results = reviewer.run(papers, cards)
    report = reviewer.generate_report(results)
    status.update(label=f"✅ 审查Agent完成: 幻觉率 {report.hallucination_rate:.1%}",
                  state="complete")

    # ---- 5. 综合Agent ----
    status = st.status("📝 综合Agent: 生成综述报告与引用图谱...", expanded=True)
    outputs = synthesizer.run(topic)
    status.update(label="📝 综合Agent完成: 报告+表格+图谱已生成", state="complete")

    # ---- 6. 证据定位（方案一：证据链穿透）----
    status = st.status("📎 证据定位: 定位引用到PDF页面并高亮...", expanded=True)
    from agents import evidence_locator
    cards_raw = json.load(open(
        os.path.join(config.DATA_DIR, "paper_cards.json"), encoding="utf-8"))
    reviews_raw = json.load(open(
        os.path.join(config.DATA_DIR, "review_results.json"),
        encoding="utf-8"))
    ev_index = evidence_locator.build_evidence_index(cards_raw, reviews_raw)
    located_n = sum(1 for v in ev_index.values() if v.get("located"))
    status.update(label=f"📎 证据定位完成: {located_n}条声明可穿透到PDF原文",
                  state="complete")

    # ---- 7. 矛盾检测（方案三：学术争议发现）----
    status = st.status("⚡ 矛盾检测: 在已核验声明间寻找学术争议...",
                       expanded=True)
    from agents import conflict_detector
    conflicts = conflict_detector.run()
    n_con = sum(1 for c in conflicts if c["relation"] == "contradict")
    n_ten = len(conflicts) - n_con
    status.update(
        label=f"⚡ 矛盾检测完成: 直接矛盾{n_con}处 + 张力{n_ten}处",
        state="complete")

    # ★关键: 清除数据加载缓存, 否则标签页仍显示上一次主题的旧数据
    # （load_json/load_text按路径缓存, 流水线覆盖文件后缓存不会自动失效）
    load_json.clear()
    load_text.clear()

    st.session_state.pipeline_done = True
    st.session_state.run_outputs["topic"] = topic
    llm_client.print_usage()

    # ★关键: 强制页面重新执行。原因: 本脚本从上到下运行, 侧边栏按钮触发
    # 流水线时, 主区域的旧数据早已加载完毕; 不rerun的话本次显示仍是旧结果
    st.rerun()


# ---------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------
with st.sidebar:
    st.title("📚 科研文献整理Agent")
    st.caption("检索 → 提取 → 审查 → 综合\n带防幻觉核验的文献综述流水线")

    st.divider()
    topic = st.text_input(
        "研究主题",
        value="脉冲神经网络的高效训练方法",
        help="支持中文输入，系统会自动翻译成英文检索词",
    )
    target_count = st.slider("目标论文数", 3, 15, 5)

    # 检索方式: 标准流水线 / 三模式检索作为检索阶段（产出完整综述）
    retrieval_mode_label = st.radio(
        "检索方式",
        ["📌 标准（推荐）", "🌱 入门综述模式", "🚀 前沿突破模式",
         "🔀 交叉领域模式"],
        help="标准=规划Agent生成检索词+四路信号综合排序；\n"
             "三模式=模式化检索词规划+多库检索，检索结果同样进入"
             "提取→审查→综合全流程，最终产出带核验的完整综述；"
             "模式未命中自动改用标准检索，仍未命中才终止流水线",
        key="retrieval_mode_radio",
    )
    _retrieval_mode = {"📌 标准（推荐）": "standard",
                       "🌱 入门综述模式": "mode1",
                       "🚀 前沿突破模式": "mode2",
                       "🔀 交叉领域模式": "mode3"}[retrieval_mode_label]
    mode_years_sidebar = 2
    if _retrieval_mode == "mode2":
        mode_years_sidebar = st.slider("前沿时间范围（近N年）", 1, 5, 2)

    col1, col2 = st.columns(2)
    with col1:
        # ★运行锁: 防止并发流水线（2026-09-04实测教训）
        # searcher.run()内不调用st.*函数，Streamlit无法中断旧运行——
        # 流水线执行中再点按钮会并发起新运行，多个运行同时轰击
        # OpenAlex触发429限流，速度反而暴慢。必须显式加锁。
        running = st.session_state.get("pipeline_running", False)
        if st.button("🚀 完整运行", type="primary", use_container_width=True,
                         disabled=running,
                         help="运行中请耐心等待，重复点击会触发API限流"):
                st.session_state.pipeline_running = True
                try:
                    with st.spinner("流水线运行中..."):
                        run_pipeline(topic, target_count,
                                     _retrieval_mode, mode_years_sidebar)
                finally:
                    st.session_state.pipeline_running = False
        if running:
            st.info("⏳ 流水线正在执行中，请等待完成后再操作页面")
    with col2:
        if st.button("📂 载入已有结果", use_container_width=True,
                     help="直接展示data/和output/目录下的历史产出"):
            st.session_state.pipeline_done = True

    st.divider()
    st.subheader("项目说明")
    st.markdown(f"""
**四个Agent流水线**

1. 🧠 **规划** — 主题→检索计划
2. 🔍 **检索** — arXiv/S2双源+LLM粗筛
3. 📄 **提取** — PDF→带引用的信息卡片
4. ✅ **审查** — 两级核验防幻觉
5. 📝 **综合** — 只用已核验内容生成综述

**核心创新**: 每条声明强制附原文引用，
程序对齐+LLM语义双重核验，
幻觉率全程量化可审计。
**证据链穿透**: 通过核验的声明
可一键定位到PDF原文高亮位置。
**可信问答**: 只基于已核验声明回答，
证据不足明确拒答。
**矛盾检测**: 自动发现论文间的
观点冲突与学术张力。

*模型: {os.path.basename(config.MODEL_NAME)}*
""")

# ---------------------------------------------------------------
# 主区域: 结果展示标签页
# ---------------------------------------------------------------
# 显示本次结果对应的主题（从会话状态取; 载入历史结果时显示通用标题）
_result_topic = st.session_state.run_outputs.get("topic", "历史结果")
st.title(f"📚 研究综述: {_result_topic}")

if not st.session_state.pipeline_done:
    # 开始检索的页面: 检索一律通过侧边栏流水线发起（用户要求:
    # 删去无需运行流水线的独立检索入口）
    st.info("👋 欢迎！请在👈左侧边栏输入研究主题、选择检索方式"
            "（📌标准 / 🌱入门综述 / 🚀前沿突破 / 🔀交叉领域），"
            "点击「🚀 完整运行」开始检索并产出带核验的完整综述；"
            "已有历史产出可点「📂 载入已有结果」直接查看。")
    st.stop()

# 各标签页共用的数据加载
papers = load_json(os.path.join(config.DATA_DIR, "papers_meta.json"), PaperMeta)
cards = load_json(os.path.join(config.DATA_DIR, "paper_cards.json"), PaperCard)
reviews = load_json(os.path.join(config.DATA_DIR, "review_results.json"),
                    ReviewResult)
evidence_index = load_json(os.path.join(config.DATA_DIR,
                                        "evidence_index.json")) or {}
table_md = load_text(os.path.join(config.OUTPUT_DIR, "review_table.md"))
report_md = load_text(os.path.join(config.OUTPUT_DIR, "final_report.md"))
graph_html = load_text(os.path.join(config.OUTPUT_DIR, "citation_graph.html"))

if cards is None:
    st.warning("没有找到历史产出数据，请先完整运行一次流水线")
    st.stop()


# ---------------------------------------------------------------
# 证据链穿透面板（方案一核心UI组件）
# ---------------------------------------------------------------
def evidence_popover(label: str, arxiv_id: str, claim_index: int,
                     claim_content: str = "", quotes=None):
    """
    声明旁的"📎 证据"弹层按钮: 点击展开显示
    PDF高亮页面截图 + 引用原文 + 定位信息

    参数:
        label: 按钮文字
        arxiv_id/claim_index: 声明唯一标识(对应evidence_index的key)
        claim_content: 声明内容(面板顶部展示)
        quotes: 声明的引用列表(面板中部展示)
    """
    key = f"{arxiv_id}#{claim_index}"
    ev = evidence_index.get(key)

    with st.popover(label, use_container_width=False):
        st.markdown(f"**声明**: {claim_content or '(见上方)'}")

        if ev is None:
            st.info("该声明无证据索引（可能未通过核验，证据仅为核验通过的声明生成）")
            if quotes:
                st.markdown("**引用原文(未定位)**:")
                for q in quotes:
                    st.markdown(f"> `{q.section}` {q.text[:150]}")
            return

        if not ev.get("located"):
            st.warning("引用未能在PDF中定位（可能因文本截断或格式差异），"
                       "以下为提取时记录的引用原文")
            if quotes:
                for q in quotes:
                    st.markdown(f"> `{q.section}` {q.text[:150]}")
            return

        # 定位成功: 展示元信息+高亮截图
        page = ev["page"] + 1  # 显示为1-based页码
        st.markdown(
            f"出处: **{ev['pdf']}** 第{page}页 "
            f"| 匹配级别: {ev.get('match_level', '?')}"
        )
        img_path = os.path.join(EVIDENCE_DIR, ev["image"])
        if os.path.exists(img_path):
            st.image(img_path, caption=f"PDF第{page}页 · 黄色高亮=证据原文",
                     use_container_width=True)
        else:
            st.error(f"证据图片缺失: {ev['image']}")
        if quotes:
            st.markdown("**引用原文**:")
            for q in quotes:
                st.markdown(f"> `{q.section}` {q.text[:150]}")


# ---- 顶部Hero横幅（品牌化门面） ----
st.markdown("""
<div style="
    background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #A855F7 100%);
    border-radius: 18px; padding: 26px 32px; margin-bottom: 6px;
    color: #FFFFFF; box-shadow: 0 8px 24px rgba(79,70,229,.25);">
  <div style="font-size: 1.7rem; font-weight: 800; letter-spacing: .5px;">
    📚 科研文献整理 Agent
  </div>
  <div style="margin-top: 6px; opacity: .92; font-size: 1.0rem;">
    检索 → 提取 → 审查 → 综合 —— 每句话都有原文出处的可信文献综述
  </div>
  <div style="margin-top: 12px;">
    <span style="background:rgba(255,255,255,.18); border-radius:999px;
      padding:4px 14px; margin-right:8px; font-size:.85rem;">🛡 两级防幻觉核验</span>
    <span style="background:rgba(255,255,255,.18); border-radius:999px;
      padding:4px 14px; margin-right:8px; font-size:.85rem;">📎 证据链穿透到PDF原文</span>
    <span style="background:rgba(255,255,255,.18); border-radius:999px;
      padding:4px 14px; margin-right:8px; font-size:.85rem;">⚡ 学术争议发现</span>
    <span style="background:rgba(255,255,255,.18); border-radius:999px;
      padding:4px 14px; font-size:.85rem;">🕸 引用图谱</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- 顶部统计卡片 ----
if reviews:
    total = len(reviews)
    good = sum(1 for r in reviews
               if r.quote_alignment and r.verdict.verdict == "supported")
    ev_located = sum(1 for v in evidence_index.values() if v.get("located"))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("收录论文", f"{len(cards)} 篇")
    c2.metric("声明总数", f"{total} 条")
    c3.metric("通过核验", f"{good} 条")
    c4.metric("幻觉率", f"{(total - good) / total:.1%}")
    c5.metric("可穿透证据", f"{ev_located} 条")
    st.caption("💡 看不懂这些指标？——「声明」=从论文摘出的可核实信息；"
               "「通过核验」=原文引用真实且无夸大；「幻觉率」=AI表述与论文"
               "原文不符的比例（越低越可信）；「可穿透证据」=能点击📎直达"
               "PDF原文高亮处的声明数。详见「🗂 更多 → 📖 新手指南」。")

# ================================================================
# 分层导航设计（降低标签栏认知负担）
# 高频标签平铺在第一层（结果消费动线: 找论文→读报告→问答→问AI）;
# 低频标签收进「🗂 更多」里的第二层标签栏——老用户不受新手指南等
# 低频入口干扰，但功能一个不少（Streamlit支持嵌套tabs，内容天然隔离）。
# 检索一律由侧边栏流水线发起（用户要求: 无独立检索入口）。
# 变量语义与初版完全一致，下方各 with tabX: 渲染代码零改动:
#   tab0=检索结果 tab1=信息卡片 tab2=审查明细
#   tab3=可信问答 tab4=学术争议 tab5=综述表格 tab6=引用图谱
#   tab7=最终报告 tab8=论文原文 tab9=AI助手 tabG=新手指南
# ================================================================
tab0, tab7, tab3, tab9, tab_more = st.tabs(
    ["🔍 检索结果", "📝 最终报告", "💬 可信问答",
     "🤖 AI助手", "🗂 更多"])

with tab_more:
    st.caption("低频功能收纳于此，保持常用功能一触可达")
    tab1, tab2, tab4, tab5, tab6, tab8, tabC, tabG = st.tabs(
        ["📄 信息卡片", "✅ 审查明细", "⚡ 学术争议", "📊 综述表格",
         "🕸 引用图谱", "📚 论文原文", "⚔️ 对比阅读", "📖 新手指南"])

# ---- 标签页: 新手指南（核心知识解释，面向所有使用者）----
with tabG:
    st.subheader("欢迎使用本系统 —— 3分钟看懂它是做什么的")
    st.success(
        "**一句话介绍**: 你输入一个研究主题，系统自动帮你检索论文、精读内容、"
        "核实每一条信息，最后产出一份「每句话都有原文出处」的文献综述。"
        "你可以把它理解为 **一支不会说谎的AI科研助理团队**。"
    )

    with st.expander("🧠 这支AI团队是怎么分工的？（五个Agent = 五个员工）", expanded=True):
        st.markdown("""
| 员工 | 干什么 | 生活化类比 |
|---|---|---|
| 🧠 **规划Agent** | 把你的主题翻译成多组英文检索词和筛选标准 | 会做调研提纲的**课题组长** |
| 🔍 **检索Agent** | 去论文库里找论文，下载PDF，并按"经典+相关"排序 | 熟门熟路的**图书管理员** |
| 📄 **提取Agent** | 精读每篇PDF，摘出"声明"并附上原文引用 | 做读书笔记的**秘书** |
| ✅ **审查Agent** | 逐条核对：引用是真是假？有没有夸大？ | 较真的**事实核查编辑** |
| 📝 **综合Agent** | 只用通过核验的内容写综述报告 | 不添油加醋的**撰稿人** |

**为什么这样设计？** 大模型直接写综述会"一本正经地胡说八道"（业内叫**幻觉**）。
本系统的对策：提取的每条信息都必须附带论文原文引用，再经过程序+AI双重核查，
所以报告里的关键结论**可以点击📎按钮跳转到PDF原文高亮处验证**。
""")

    with st.expander("🗺️ 标签页导读（常用标签 + 「🗂 更多」里的低频功能）",
                     expanded=True):
        st.markdown("""
**第一层常用标签**（按使用动线排列）:

1. **🔍 检索结果** — 系统找到了哪些论文？为什么这篇排前面？
   每篇的徽章含义：`🔗引用链经典`=被多篇论文共同引用的奠基之作；
   `🔥高被引`=学术界公认有影响力；`🤖高度切题`=与你的主题最匹配。
2. **📝 最终报告** — 综述正文。**看不懂？点"🍼 生成小白解读"按钮**，
   系统会把报告转写成通俗易懂的版本。
3. **💬 可信问答** — 直接用中文提问！系统只根据已核验的内容回答，
   **证据不够时宁可拒绝回答也不编造**——这是和普通AI对话最大的区别。
4. **🤖 AI助手** — 任何疑问直接问：系统怎么用、指标什么含义、
   论文里某个方法的效果，它都能答（论文内容只基于已核验信息）。

**🎯 三模式检索**（侧边栏「检索方式」选择，作为流水线的检索阶段运行）——
🌱**入门综述**（经典综述优先，零基础也能懂）/
🚀**前沿突破**（近1-5年顶刊顶会最新成果，按时间+影响力排序）/
🔀**交叉领域**（S2+OpenAlex+arXiv多库互补，组合词精准定位跨学科研究）。
检索命中的文献照常进入提取→审查→综合，产出带核验的完整综述；
**模式未命中自动降级为标准检索**继续，降级后仍未命中才终止流水线。

**第二层「🗂 更多」**（低频功能，点开后可见）:

5. **📄 信息卡片** — 每篇论文被拆成了哪些知识点（声明）？
   🔵=方法、🟢=实验结果、🟠=局限性说明；每条声明旁有📎证据可跳PDF原文。
6. **✅ 审查明细** — 哪些声明通过了核实？哪些被剔除、为什么？
   （被剔除≠论文差，只是那条表述与原文对不上）
7. **⚡ 学术争议** — 论文之间打架了？谁和谁的结论冲突？
   🔴=正面对立，🟡=表面冲突（实验条件不同导致的）。
8. **📊 综述表格** — 各论文的方法横向对比表。
9. **🕸 引用图谱** — 论文之间谁引用谁的关系网（可拖拽交互）。
10. **📚 论文原文** — 想亲自读论文？选一篇就在网页里直接阅读PDF全文，
    核对系统说的和论文写的是否一致。顶部还有 **🔍 全文搜索**
    （一个关键词同时查所有本地论文，命中处高亮并可跳到对应页码）、
    **📝 阅读笔记**（给某页写批注，绑定论文与页码，本地保存可增删改）、
    **📖 核心名词速查对照**（阅读区右侧常驻最近模式检索报告的名词表，
    随读随查，可一键隐藏）。
11. **⚔️ 对比阅读** — 挑两篇论文左右并排对比：背景/方法/结论/争议点
    一屏看清，还联动显示各自的阅读笔记。
12. **📖 新手指南** — 就是本页。熟悉系统后不用再看，需要时来「更多」找。
""")

    with st.expander("📚 术语速查表（看不懂某个词就来这里查）", expanded=False):
        st.markdown("""
| 术语 | 通俗解释 |
|---|---|
| **声明 (Claim)** | 从论文里摘出的一条可核实的信息，如"该方法在CIFAR-10上达到95%准确率" |
| **原文引用 (Quote)** | 支撑该声明的论文原话（逐字摘录，不许改写） |
| **引用对齐** | 程序检查：声明的"原文引用"是否真的出现在PDF里 → 抓**伪造引用** |
| **语义核验** | AI检查：引用虽真实，但有没有断章取义/夸大 → 抓**曲解原文** |
| **幻觉率** | 被剔除声明占总声明的比例，即"AI胡说占比"，本系统全程量化它 |
| **证据链穿透** | 点击📎直达PDF原文高亮位置，眼见为实 |
| **RRF共识** | 多个检索角度都命中的论文排前面（多个证据源一致=更可信） |
| **引用链/共被引** | 被多篇种子论文的参考文献共同引用=领域公认经典 |
| **免检通道** | 共被引≥3的经典跳过AI粗筛（学术界共识比AI一次性打分更可靠） |
| **矛盾 vs 张力** | 矛盾=结论正面对立；张力=表面冲突，细看是实验条件不同 |
| **arXiv编号** | 论文在arXiv预印本库的身份证号，如 2401.12345 |
""")

    with st.expander("🚀 最短上手路径（三步）", expanded=False):
        st.markdown("""
1. 左侧边栏输入主题（**写具体些**，如"脉冲神经网络的高效训练方法"而不是"神经网络"）→ 点 **🚀 完整运行**
2. 等待10-15分钟（系统在检索+精读+核查，日志会显示进度）
3. 先看 **📝 最终报告**（可点"🍼 生成小白解读"），有疑问去 **💬 可信问答** 直接提问
""")

# ---- 标签0: 检索结果（论文排序输出面板） ----
with tab0:
    st.subheader("检索结果排序")
    st.caption(
        "排序依据四路信号加权: **相关性**(LLM判断主题切题度) + "
        "**RRF共识**(被多个检索角度共同命中) + **影响力**(被引数) + "
        "**引用链**(被种子论文参考文献共同引用的次数，奠基经典专属信号)。\n"
        "经典论文保障机制: 共被引≥3走**免检通道**(学术界共识优先于LLM打分)；"
        "共被引≥2享受粗筛阈值补偿；最老的经典(如SpikeProp)由规划Agent的"
        "**LLM领域先验**按标题注入，弥补引用图谱数据缺失。"
    )

    # 排序方式切换（同一份论文池，不同视角的输出方式）
    sort_mode = st.radio(
        "排序方式",
        ["🎯 综合推荐", "🤖 相关优先", "🔥 经典优先", "🆕 最新优先"],
        horizontal=True,
        help="综合推荐=四路信号加权(默认); 相关优先=LLM切题度优先; "
             "经典优先=被引数优先; 最新优先=发表年份优先",
    )
    if papers:
        if sort_mode == "🎯 综合推荐":
            shown = sorted(papers, key=lambda p: p.final_score or 0,
                           reverse=True)
        elif sort_mode == "🤖 相关优先":
            shown = sorted(papers, key=lambda p: p.relevance_score or 0,
                           reverse=True)
        elif sort_mode == "🔥 经典优先":
            shown = sorted(papers, key=lambda p: p.cited_by, reverse=True)
        else:
            shown = sorted(papers, key=lambda p: p.year, reverse=True)

        for rank, p in enumerate(shown, 1):
            # 徽章: 让用户一眼看懂"为什么这篇排在这里"
            badges = []
            if p.chain_hits >= 3:
                badges.append(f"🔗引用链经典(被{p.chain_hits}篇种子引用)")
            elif p.chain_hits > 0:
                badges.append(f"🔗引用链({p.chain_hits})")
            if p.cited_by >= 300:
                badges.append(f"🔥高被引({p.cited_by})")
            if (p.relevance_score or 0) >= 4.5:
                badges.append(f"🤖高度切题({p.relevance_score})")
            badge_str = " ".join(f"`{b}`" for b in badges)

            score_parts = (
                f"相关性 {p.relevance_score if p.relevance_score is not None else '免检'}"
                f" · 被引 {p.cited_by} · 引用链 {p.chain_hits}"
                + (f" · 综合分 {p.final_score}" if p.final_score else "")
            )
            with st.expander(
                f"{rank}. {p.title[:70]} ({p.year}) {badge_str}"
            ):
                st.caption(f"[{p.arxiv_id}] {score_parts}")
                st.markdown(
                    p.abstract[:400] + ("..." if len(p.abstract) > 400 else "")
                )
    else:
        st.info("暂无检索结果数据")

# ---- 标签: 论文原文（本地PDF在线阅读）----
with tab1:
    st.subheader("论文信息卡片（提取Agent产出）")
    st.caption("点击每条声明旁的 📎 按钮可穿透查看PDF原文高亮位置")
    for card in cards:
        with st.expander(f"📁 {card.title[:55]}... [{card.arxiv_id}] "
                         f"({card.year}) — {card.method_category}"):
            # 原文入口: 该论文有本地PDF时提示跳转（跨标签页联动）
            if os.path.exists(os.path.join(config.PAPER_DIR,
                                           f"{card.arxiv_id}.pdf")):
                st.caption("📚 想读整篇论文？到「论文原文」标签页选这篇即可在线阅读")
        # 展开显示每条声明及其原文引用
            for i, claim in enumerate(card.claims):
                color = {"method": "🔵", "result": "🟢",
                         "limitation": "🟠"}.get(claim.claim_type, "⚪")
                # 声明 + 证据弹层按钮同行布局
                c_col, e_col = st.columns([5, 1])
                with c_col:
                    st.markdown(f"{color} **[{claim.claim_type}]** "
                                f"{claim.content}")
                with e_col:
                    evidence_popover("📎 证据", card.arxiv_id, i,
                                     claim.content, claim.quotes)
                for q in claim.quotes:
                    st.markdown(
                        f"> `{q.section}` {q.text[:200]}..."
                        if len(q.text) > 200 else f"> `{q.section}` {q.text}"
                    )
                st.divider()

# ---- 标签2: 审查明细 ----
with tab2:
    st.subheader("两级核验明细（审查Agent产出）")
    st.caption("第1级: 程序化引用对齐(抓伪造引用) | "
               "第2级: LLM语义核验(抓夸大/曲解)")

    # 用表格展示全部核验结果
    rows = []
    for r in reviews:
        rows.append({
            "论文": r.arxiv_id,
            "#": r.claim_index,
            "声明": r.claim_content[:50] + "...",
            "引用对齐": "✓" if r.quote_alignment else "✗",
            "语义判定": r.verdict.verdict,
            "置信度": f"{r.verdict.confidence:.2f}",
            "理由": r.verdict.reason[:60],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # 通过核验的声明: 带证据穿透按钮
    good_list = [r for r in reviews
                 if r.quote_alignment and r.verdict.verdict == "supported"]
    st.markdown(f"### ✓ 通过核验的声明 ({len(good_list)}条)")
    st.caption("点击 📎 可穿透查看该声明在PDF原文中的高亮位置")
    # 声明内容需要从cards里取（reviews只有claim_content摘要）
    for r in good_list:
        c_col, e_col = st.columns([5, 1])
        with c_col:
            st.markdown(f"✅ **[{r.arxiv_id} #{r.claim_index}]** "
                        f"{r.claim_content[:70]}...")
        with e_col:
            # 找对应claim的quotes（从cards索引）
            quotes = None
            for card in cards:
                if card.arxiv_id == r.arxiv_id and r.claim_index < len(card.claims):
                    quotes = card.claims[r.claim_index].quotes
                    break
            evidence_popover("📎 证据", r.arxiv_id, r.claim_index,
                             r.claim_content, quotes)

    # 分组展示: 通过 vs 剔除
    bad = [r for r in reviews if not r.quote_alignment
           or r.verdict.verdict != "supported"]
    st.markdown(f"### 被剔除的声明 ({len(bad)}条)")
    for r in bad:
        with st.expander(f"❌ [{r.arxiv_id} #{r.claim_index}] "
                         f"{r.claim_content[:50]}..."):
            st.markdown(f"- 引用对齐: {'✓ 真实' if r.quote_alignment else '✗ 疑似伪造'}")
            st.markdown(f"- 语义判定: **{r.verdict.verdict}** "
                        f"(置信度 {r.verdict.confidence})")
            st.markdown(f"- 判定理由: {r.verdict.reason}")

# ---- 标签3: 可信问答（方案二）----
with tab3:
    st.subheader("💬 可信问答（Evidence-Bounded QA）")
    st.caption("只基于已通过两级核验的声明池回答，证据不足时明确拒答——"
               "宁可拒答，不可编造")

    from agents import qa_agent as qa_mod

    # 会话状态: 对话历史
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # 左右布局: 左侧对话区 + 右侧声明池浏览
    qa_col, pool_col = st.columns([3, 2])

    with qa_col:
        # 渲染历史对话
        for msg in st.session_state.qa_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # 回答消息附引用声明与拒答说明
                if msg["role"] == "assistant" and msg.get("used_claims"):
                    st.caption(f"依据: 声明 {msg['used_claims']}")
                if msg["role"] == "assistant" and msg.get("missing"):
                    st.warning(f"缺失证据: {msg['missing']}")

        # 输入框
        if q := st.chat_input("问点什么…（如: HorNet在ImageNet上的表现如何？）"):
            st.session_state.qa_history.append(
                {"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("检索声明池并生成有据回答..."):
                    try:
                        ans, _ = qa_mod.answer_question(
                            q, st.session_state.qa_history[:-1])
                        body = ans.answer
                        if not ans.can_answer:
                            body = (f"⚠️ {ans.answer}\n\n"
                                    f"**缺失的证据**: {ans.missing}")
                        st.markdown(body)
                        if ans.used_claims:
                            st.caption(f"依据: 声明 {ans.used_claims}")
                        st.session_state.qa_history.append({
                            "role": "assistant",
                            "content": body,
                            "used_claims": ans.used_claims,
                            "missing": ans.missing if not ans.can_answer else "",
                        })
                    except Exception as e:
                        st.error(f"问答失败: {e}")

    with pool_col:
        st.markdown("**📚 当前声明池**（问答的知识边界）")
        pool = qa_mod.build_claim_pool()
        if pool:
            st.caption(f"共{len(pool)}条已核验声明 · 回答只能引用这些内容")
            for c in pool:
                type_icon = {"method": "🔵", "result": "🟢",
                             "limitation": "🟠"}.get(c["type"], "⚪")
                st.markdown(
                    f"{type_icon} **[#{c['num']}]** {c['content'][:80]}..."
                    if len(c["content"]) > 80
                    else f"{type_icon} **[#{c['num']}]** {c['content']}"
                )
        else:
            st.info("声明池为空，请先运行完整流水线")

        # 拒答能力一键自检（E5实验入口）
        st.divider()
        if st.button("🧪 拒答能力自检", help="运行3个可答+3个知识库外问题，"
                     "验证系统的拒答可靠性"):
            with st.spinner("测试中（约1分钟）..."):
                r = qa_mod.run_refusal_test()
            if r:
                m1, m2 = st.columns(2)
                m1.metric("可答组正确率", f"{r['answerable_acc']:.0%}")
                m2.metric("拒答组正确率", f"{r['refusal_acc']:.0%}")
                with st.expander("测试明细"):
                    for d in r["details"]:
                        mark = "✓" if d["correct"] else "✗"
                        st.markdown(
                            f"{mark} {'应答' if d['expect_answerable'] else '应拒'}"
                            f"→{'答了' if d['actual_answerable'] else '拒答'} "
                            f"| {d['question']}")

# ---- 标签4: 学术争议（方案三）----
with tab4:
    st.subheader("⚡ 学术争议（矛盾检测Agent产出）")
    st.caption("在已核验声明间自动检测观点冲突——"
               "🔴 直接矛盾(结论对立) | 🟡 张力(条件不同导致的表象冲突)。"
               "每条争议声明可点击📎穿透到PDF原文")

    conflicts = load_json(os.path.join(config.DATA_DIR, "conflicts.json"))

    if not conflicts:
        st.info("当前声明池未检测到学术争议（论文主题较分散时属正常）。"
                "论文数更多、主题更聚焦时争议检出率更高。")
        # 手动触发按钮（历史数据没有conflicts.json时用）
        if st.button("⚡ 立即检测矛盾"):
            with st.spinner("检测中（预筛+LLM精判，约1-3分钟）..."):
                from agents import conflict_detector
                conflict_detector.run()
                load_json.clear()
                st.rerun()
    else:
        n_con = sum(1 for c in conflicts if c["relation"] == "contradict")
        n_ten = len(conflicts) - n_con
        m1, m2 = st.columns(2)
        m1.metric("直接矛盾", f"{n_con} 处")
        m2.metric("张力关系", f"{n_ten} 处")

        for c in conflicts:
            icon = "🔴" if c["relation"] == "contradict" else "🟡"
            sev = "★" * c["severity"] + "☆" * (5 - c["severity"])
            with st.expander(
                f"{icon} [{c['relation']}] {c['topic']} "
                f"(严重度 {sev}) — {c['explanation'][:40]}..."
            ):
                st.markdown(f"**分析**: {c['explanation']}")

                # 双方观点对照展示（各带证据穿透）
                for side, label in [("a", "观点A"), ("b", "观点B")]:
                    s = c[side]
                    st.markdown(f"**{label}**: "
                                f"《{s['paper_title'][:40]}》"
                                f"({s['year']}, arXiv:{s['arxiv_id']})")
                    s_col, e_col = st.columns([5, 1])
                    with s_col:
                        st.markdown(f"> {s['content']}")
                    with e_col:
                        evidence_popover(
                            "📎 证据", s["arxiv_id"], s["claim_index"],
                            s["content"])
                st.divider()

        # 重新检测按钮（数据更新后刷新）
        if st.button("🔄 重新检测"):
            with st.spinner("检测中..."):
                from agents import conflict_detector
                conflict_detector.run()
                load_json.clear()
                st.rerun()

# ---- 标签5: 综述表格 ----
with tab5:
    st.subheader("方法分类对比表（仅收录通过核验的声明）")
    if table_md:
        st.markdown(table_md)
    else:
        st.warning("综述表格未生成")

# ---- 标签6: 引用图谱 ----
with tab6:
    st.subheader("引用关系图谱（交互式）")
    st.caption("🟢 本次检索论文 | 🔴 内部互引箭头 | ⚪ 外部被引论文 | "
               "橙色=方法分类(降级模式)")
    if graph_html:
        # 用iframe嵌入pyvis生成的HTML（组件化渲染交互图）
        st.components.v1.html(graph_html, height=620)
    else:
        st.warning("引用图谱未生成")

# ---- 标签7: 最终报告 ----
with tab7:
    st.subheader("最终综述报告")
    # 报告导读（面向不熟悉文献综述的读者）
    with st.expander("📖 如何阅读这份报告？（第一次看请点开）"):
        st.markdown("""
**这是一份什么文档？** 文献综述=某个研究领域的"地图"：
它不发明新东西，而是把多篇论文的观点**系统梳理**给你看。

**典型结构与读法**:
1. **背景部分** — 这个领域在解决什么问题？为什么重要？
   （读不懂术语时查「🗂 更多 → 📖 新手指南」的术语速查表）
2. **方法分类** — 现有解决方案分成哪几派？各自的思路是什么？
3. **对比与结论** — 不同方法的效果差异、适用场景。
4. **局限性** — 现有研究还没解决什么？（往往是做研究选题的金矿）

**可信度说明**: 报告只使用了**通过两级核验的声明**（程序查引用真伪 +
AI查语义曲解），每条关键信息理论上都可回溯到PDF原文。
被剔除的内容不会出现在这里——这是它和"让ChatGPT直接写综述"的本质区别。
        """)
    if report_md:
        # 小白解读生成（把综述转写成带背景知识的通俗版）
        # 注意: 生成耗时2-4分钟属正常（输出几千字），流式显示让你
        # 看到逐字进度而不是干等转圈；切换标签页会中断，请停留本页
        if st.button("🍼 生成小白解读", help="把这份报告本身转述成大白话："
                     "术语随文轻注、难句翻成人话，不添加报告外的"
                     "学科教学（生成过程逐字显示，请勿切换标签页）"):
            with st.expander("🍼 小白版报告（通俗解读）", expanded=True):
                import llm_client
                st.caption("⏳ 正在生成... 下方文字会逐步出现，"
                           "请停留在本页（讲解详尽，可能需要数分钟）")
                plain = st.write_stream(llm_client.chat_stream(
                    [
                        {"role": "system", "content":
                         "你是一位翻译高手，擅长把学术文献综述逐句"
                         "转述成大白话——像把一篇文言文翻译成白话文，"
                         "内容一字不增一字不减，只是语言变通俗。\n\n"
                         "核心原则（决定成败，逐条遵守）:\n"
                         "1. **忠实转译，不从零讲课**: 报告讲什么你就讲"
                         "什么，按报告原有的结构和顺序走。禁止添加"
                         "报告之外的学科背景、历史铺垫、入门教学——"
                         "读者要的是'看懂这份报告'，不是'学这门课'\n"
                         "2. **每个艰深句子都要翻成人话**: 原报告里"
                         "读者可能看不懂的句子（术语密集的、逻辑复杂的），"
                         "必须换成日常语言重新表达，意思严格不变。\n"
                         "   - 例: 原文'该方法采用代理梯度替代不可微的"
                         "脉冲激活函数' → 翻译'这个方法的做法是：因为"
                         "脉冲这种'放电行为'没法直接算数学梯度，他们"
                         "就找个行为相似的平滑函数来代替它算'\n"
                         "3. **术语处理要轻**: 术语首次出现时在括号里用"
                         "一句话点明它指什么即可（如: 代理梯度（就是替"
                         "脉冲放电行为找的'数学替身'）），不展开教学，"
                         "不单独成段\n"
                         "4. **数据结论引用全保留**: 所有数字、结论、"
                         "论文引用信息原样保留，一个都不能丢——这是"
                         "报告的核心价值\n"
                         "5. **类比是调味料不是主菜**: 只在难懂的原理处"
                         "用一句类比帮理解，禁止大段故事化铺垫\n"
                         "6. markdown输出，小节标题跟随原报告但可口语化；"
                         "文末加一小段'一句话总结这份报告'"},
                        {"role": "user",
                         "content": f"请把以下文献综述转述成大白话版"
                                    f"（要求: 忠实转译报告本身，不添加"
                                    f"报告外的学科教学）:\n\n{report_md}"},
                    ],
                    temperature=0.3,
                ))
                st.session_state.plain_report = plain
                st.session_state._just_generated = True
                st.success("✅ 小白解读已生成，可再次点击按钮重新生成")
        if (st.session_state.get("plain_report")
                and not st.session_state.pop("_just_generated", False)):
            with st.expander("🍼 小白版报告（通俗解读）", expanded=True):
                st.markdown(st.session_state.plain_report)
        st.markdown(report_md)
        # 提供下载按钮（Markdown + 成员C模块1: Word/LaTeX双格式导出）
        try:
            import exporter
            _papers_raw = load_json(
                os.path.join(config.DATA_DIR, "papers_meta.json")) or []
            _refs_meta = exporter.meta_from_papers_json(_papers_raw)
            _topic_exp = st.session_state.run_outputs.get("topic", "文献综述")
            _meta_lines = [f"主题: {_topic_exp}",
                           f"收录论文: {len(_papers_raw)} 篇"]
            _d1, _d2, _d3 = st.columns(3)
            with _d1:
                st.download_button(
                    "⬇️ Markdown",
                    data=report_md,
                    file_name="literature_review.md",
                    mime="text/markdown",
                )
            with _d2:
                st.download_button(
                    "⬇️ Word (.docx)",
                    data=exporter.md_to_docx(
                        report_md, title=f"研究综述: {_topic_exp}",
                        meta_lines=_meta_lines),
                    file_name="literature_review.docx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".wordprocessingml.document",
                )
            with _d3:
                st.download_button(
                    "⬇️ LaTeX+BibTeX (.zip)",
                    data=exporter.build_latex_package(
                        report_md, f"研究综述: {_topic_exp}",
                        _refs_meta, _meta_lines),
                    file_name="latex_package.zip",
                    mime="application/zip",
                    help="含 main.tex + refs.bib + README（编译说明），"
                         "上传 Overleaf 选 XeLaTeX 即可编译",
                )
        except Exception as _exp_err:
            st.caption(f"Word/LaTeX 导出不可用: {_exp_err}（Markdown下载不受影响）")
            st.download_button(
                "⬇️ 下载完整报告 (Markdown)",
                data=report_md,
                file_name="literature_review.md",
                mime="text/markdown",
            )
    else:
        st.warning("最终报告未生成")

# ---- 标签8: 论文原文（PDF搜索 + 本地PDF在线阅读）----
with tab8:
    st.subheader("论文原文阅读")
    st.caption("流水线下载的PDF全文可在此逐页阅读。想核对系统声明的"
               "原文出处时用这里（📄信息卡片里的📎证据会给出页码）。")

    # 扫描papers/目录，列出所有本地已有PDF（含未进入最终收录的）
    import glob as _glob
    pdf_files = sorted(_glob.glob(os.path.join(config.PAPER_DIR, "*.pdf")))
    if not pdf_files:
        st.info("papers/ 目录下没有PDF。请先完整运行一次流水线"
                "（检索Agent会自动下载论文PDF到本地）。")
    else:
        # =============================================================
        # 成员C · 模块2: PDF全文搜索（增量索引 + 跨文档 + 高亮 + 跳页）
        # =============================================================
        with st.expander("🔍 PDF全文搜索（跨所有本地论文）", expanded=False):
            try:
                import pdf_search as _ps

                _s_col, _s_btn = st.columns([4, 1])
                with _s_col:
                    _ps_q = st.text_input(
                        "关键词（多词=同时出现，支持中英文）",
                        key="ps_q",
                        placeholder="如: surrogate gradient / 脉冲 训练")
                with _s_btn:
                    st.write("")  # 对齐输入框基线
                    _go_search = st.button("🔍 搜索", use_container_width=True)

                if _go_search and _ps_q.strip():
                    # 增量构建索引（已索引且未变的PDF秒级跳过）
                    with st.spinner("正在更新索引（首次较慢，之后秒开）..."):
                        _idx = _ps.build_index()
                    st.session_state.ps_results = _ps.search(
                        _ps_q.strip(), _idx)
                    st.session_state.ps_query = _ps_q.strip()

                if st.session_state.get("ps_results") is not None:
                    _hits = st.session_state.ps_results
                    _pq = st.session_state.get("ps_query", "")
                    if not _hits:
                        st.info(f"没有找到与「{_pq}」匹配的内容")
                    else:
                        _docs = sorted({h['pdf_id'] for h in _hits})
                        st.caption(f"命中 {len(_hits)} 页 · "
                                   f"跨 {len(_docs)} 篇论文"
                                   f"（按命中页数排序，点开展看高亮）")
                        for _hi, _h in enumerate(_hits[:20]):
                            _flag = "🟨" if _h["pdf_id"] == \
                                st.session_state.get("pdf_jump_id") else ""
                            with st.expander(
                                f"{_flag}《{_h['title'][:52]}》"
                                f"· 第{_h['page'] + 1}页 · "
                                f"命中{_h['score']}页"
                            ):
                                st.markdown(f"**摘要片段**: {_h['snippet']}")
                                _img = _ps.render_highlight_page(
                                    _h["pdf_id"], _h["page"], _pq)
                                if _img and os.path.exists(_img):
                                    st.image(_img,
                                             caption=f"第{_h['page'] + 1}页 · "
                                                     f"黄色高亮=关键词",
                                             use_container_width=True)
                                _j1, _j2 = st.columns([1, 2])
                                with _j1:
                                    if st.button("📄 跳转原文页",
                                                 key=f"ps_jump_{_hi}",
                                                 use_container_width=True):
                                        st.session_state.pdf_jump_id = \
                                            _h["pdf_id"]
                                        st.session_state.pdf_jump_page = \
                                            _h["page"]
                                        # 联动模块3: 摘要片段预填进笔记引用
                                        st.session_state.note_prefill = \
                                            _h["snippet"]
                                        st.rerun()
                                with _j2:
                                    if st.button("📝 引用此段做笔记",
                                                 key=f"ps_note_{_hi}",
                                                 use_container_width=True):
                                        st.session_state.pdf_jump_id = \
                                            _h["pdf_id"]
                                        st.session_state.pdf_jump_page = \
                                            _h["page"]
                                        st.session_state.note_prefill = \
                                            _h["snippet"]
                                        st.rerun()
            except Exception as _ps_err:
                st.caption(f"全文搜索暂不可用: {_ps_err}")

        # 用arxiv_id关联元数据
        meta_by_id = {p.arxiv_id: p for p in (papers or [])}
        all_ids = [os.path.splitext(os.path.basename(p))[0]
                   for p in pdf_files]
        id_path = {pid: path for pid, path in zip(all_ids, pdf_files)}

        # 本次检索到的论文（元数据里有且PDF已下载）置顶，其余本地PDF靠后
        retrieved = [p for p in (papers or []) if p.arxiv_id in id_path]
        other_ids = [pid for pid in all_ids
                     if pid not in {p.arxiv_id for p in retrieved}]

        # 排序方式与「检索结果」标签页一致（四路信号同口径）
        sort_mode8 = st.radio(
            "本次检索论文排序",
            ["🎯 综合推荐", "🤖 相关优先", "🔥 经典优先", "🆕 最新优先"],
            horizontal=True,
            key="sort_pdf",
            help="综合推荐=四路信号加权(默认); 相关优先=LLM切题度优先; "
                 "经典优先=被引数优先; 最新优先=发表年份优先。"
                 "排序只作用于本次检索到的论文，其他本地PDF附后",
        )
        if sort_mode8 == "🎯 综合推荐":
            retrieved.sort(key=lambda p: p.final_score or 0, reverse=True)
        elif sort_mode8 == "🤖 相关优先":
            retrieved.sort(key=lambda p: p.relevance_score or 0, reverse=True)
        elif sort_mode8 == "🔥 经典优先":
            retrieved.sort(key=lambda p: p.cited_by, reverse=True)
        else:
            retrieved.sort(key=lambda p: p.year, reverse=True)

        # 构造下拉选项: 置顶段带排名和徽章，其他段标"未入选本次检索"
        options = []
        for rank, p in enumerate(retrieved, 1):
            badges = []
            if p.chain_hits >= 3:
                badges.append("🔗引用链经典")
            if p.cited_by >= 300:
                badges.append(f"🔥{p.cited_by}被引")
            if (p.relevance_score or 0) >= 4.5:
                badges.append("🤖高度切题")
            badge_str = (" " + " ".join(badges)) if badges else ""
            options.append(
                f"{rank}. {p.title[:55]} ({p.year}){badge_str} [{p.arxiv_id}]"
            )
        for pid in other_ids:
            options.append(f"📖 其他本地PDF [{pid}]")

        # 搜索跳转联动: 有跳转目标时下拉默认选中该论文
        _jump_id = st.session_state.get("pdf_jump_id")
        _default_idx = 0
        if _jump_id:
            for _oi, _opt in enumerate(options):
                if _opt.endswith(f"[{_jump_id}]"):
                    _default_idx = _oi
                    break

        choice = st.selectbox(
            f"选择论文（本次检索{len(retrieved)}篇 · "
            f"其他本地{len(other_ids)}篇）",
            options,
            index=_default_idx,
        )
        # 从label反解arxiv_id（末尾方括号里）
        chosen_id = choice[choice.rfind("[") + 1:-1]
        pdf_path = id_path[chosen_id]

        m = meta_by_id.get(chosen_id)
        _paper_title_disp = m.title if m else chosen_id
        if m:
            st.markdown(f"**{m.title}** · {m.year} · "
                        f"被引{m.cited_by} · arXiv:{chosen_id}")
        else:
            st.caption(f"arXiv:{chosen_id}")

        # =============================================================
        # 成员C · 模块3: 阅读笔记（绑定文献ID+页码，本地持久化）
        # =============================================================
        try:
            import notes_store as _ns

            _notes = _ns.list_notes(chosen_id)
            with st.expander(f"📝 阅读笔记（本篇 {len(_notes)} 条）",
                             expanded=bool(st.session_state.get(
                                 "note_prefill")) or len(_notes) > 0):
                # ---- 添加/编辑表单 ----
                _editing = st.session_state.get("note_editing")
                _prefill_q = st.session_state.pop("note_prefill", "")
                if _editing:
                    _src = next((n for n in _notes
                                 if n["id"] == _editing), None)
                else:
                    _src = None
                with st.form("note_form", clear_on_submit=True):
                    st.markdown("**➕ 添加笔记**（批注绑定本篇文献与页码，"
                                "保存在本地）" if not _src
                               else f"**✏️ 编辑笔记** `{_editing}`")
                    _nf1, _nf2 = st.columns([1, 2])
                    with _nf1:
                        _n_page = st.number_input(
                            "页码", min_value=1, max_value=99,
                            value=(_src["page"] + 1) if _src else 1,
                            help="1-based页码，与下方页标一致")
                    with _nf2:
                        _n_quote = st.text_area(
                            "选中段落（可从PDF/搜索结果复制，可留空）",
                            value=(_src["quote"] if _src else _prefill_q),
                            height=68)
                    _n_text = st.text_area(
                        "批注内容",
                        value=(_src["note"] if _src else ""), height=68)
                    _sv, _cancel = st.columns(2)
                    with _sv:
                        _submitted = st.form_submit_button(
                            "💾 保存", type="primary", use_container_width=True)
                    with _cancel:
                        if _src and st.form_submit_button(
                                "取消编辑", use_container_width=True):
                            st.session_state.note_editing = None
                            st.rerun()
                # form渲染必须在with块内完成，提交处理放块外
                if _submitted:
                    try:
                        if _src:
                            _ns.update_note(
                                _src["id"], quote=_n_quote, note=_n_text,
                                page=int(_n_page) - 1)
                            st.session_state.note_editing = None
                            st.toast("✅ 笔记已更新")
                        else:
                            _ns.add_note(
                                chosen_id, int(_n_page) - 1,
                                _n_quote, _n_text,
                                paper_title=_paper_title_disp)
                            st.toast("✅ 笔记已保存")
                        st.rerun()
                    except ValueError as _ve:
                        st.error(f"保存失败: {_ve}")

                # ---- 笔记列表（按页码分组） ----
                if _notes:
                    st.divider()
                    for _nt in _notes:
                        _e1, _e2, _e3 = st.columns([6, 1, 1])
                        with _e1:
                            st.markdown(
                                f"💬 **第{_nt['page'] + 1}页** · "
                                + time.strftime(
                                    "%m-%d %H:%M",
                                    time.localtime(_nt["created_at"])))
                            if _nt["quote"]:
                                st.markdown(
                                    f"> {_nt['quote'][:160]}"
                                    + ("…" if len(_nt["quote"]) > 160 else ""))
                            if _nt["note"]:
                                st.markdown(_nt["note"][:400])
                            st.caption("")
                        with _e2:
                            if st.button("✏️", key=f"ne_{_nt['id']}",
                                         help="编辑"):
                                st.session_state.note_editing = _nt["id"]
                                st.rerun()
                        with _e3:
                            if st.button("🗑", key=f"nd_{_nt['id']}",
                                         help="删除"):
                                _ns.delete_note(_nt["id"])
                                st.toast("已删除")
                                st.rerun()
        except Exception as _ns_err:
            st.caption(f"笔记功能暂不可用: {_ns_err}")

        # ---- 按页渲染为图片显示（可靠方案）----
        # 背景: base64数据URI内嵌PDF会被Streamlit组件的沙箱iframe
        # 拦截（浏览器插件策略），实测无法渲染。改用PyMuPDF逐页
        # 渲染PNG——任何浏览器都支持，还能精确对应📎证据的页码。
        @st.cache_data(show_spinner="正在解析PDF...")
        def _pdf_page_count(path: str) -> int:
            import pymupdf
            with pymupdf.open(path) as doc:
                return doc.page_count

        @st.cache_data(show_spinner=False)
        def _render_page(path: str, page_idx: int) -> bytes:
            """渲染单页为PNG图片（约150DPI，清晰度与速度的平衡）"""
            import pymupdf
            with pymupdf.open(path) as doc:
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8))
                return pix.tobytes("png")

        try:
            n_pages = _pdf_page_count(pdf_path)
        except Exception as e:
            st.error(f"PDF解析失败: {e}")
            st.stop()

        # =============================================================
        # 核心名词速查对照面板（全模式通用，数量无上限，可隐藏）
        # 数据源: 最近一次三模式流水线生成的速查表
        # （run_pipeline模式路径写入会话+glossary_last.json）
        # =============================================================
        _gloss_md = st.session_state.get("glossary_md") or ""
        if not _gloss_md:
            _gpath = os.path.join(config.DATA_DIR, "glossary_last.json")
            if os.path.exists(_gpath):
                try:
                    with open(_gpath, "r", encoding="utf-8") as f:
                        _gd = json.load(f)
                    _gloss_md = _gd.get("md") or ""
                    if _gloss_md:
                        st.session_state.glossary_topic = \
                            _gd.get("topic", "")
                except Exception:
                    pass
        if _gloss_md:
            _show_g = st.toggle(
                "📖 核心名词速查对照（右侧随读随查，再点一次隐藏）",
                value=st.session_state.get("gloss_show", False),
                key="gloss_show",
                help="来自最近一次模式检索报告的核心名词速查表，"
                     "在阅读区右侧常驻对照；关闭后恢复全宽阅读")
            if _show_g:
                _rc, _gc = st.columns([2.9, 1.1], gap="small")
                with _gc:
                    st.markdown("##### 📖 核心名词速查")
                    _gt = st.session_state.get("glossary_topic", "")
                    if _gt:
                        st.caption(f"来自模式检索: {_gt[:40]}")
                    st.markdown(_gloss_md)
            else:
                _rc = st.container()
        else:
            _rc = st.container()

        # 页码选择器 + 跳转目标页置顶渲染（模块2: 跳转原文页码）
        _jump_page = st.session_state.get("pdf_jump_page")
        _jump_active = (_jump_id == chosen_id and _jump_page is not None
                        and 0 <= _jump_page < n_pages)

        with _rc:
            if _jump_active:
                st.success(f"⬆️ 已跳转: 第 {_jump_page + 1} 页"
                           f"（来自搜索结果，下方为该页，继续滚动可读全文）")
                st.image(
                    _render_page(pdf_path, _jump_page),
                    caption=f"★ 跳转目标 · 第 {_jump_page + 1} 页 / "
                            f"共 {n_pages} 页 ★",
                    use_container_width=True,
                )
                # 消费后清理跳转标记（刷新后不残留）
                del st.session_state.pdf_jump_id
                del st.session_state.pdf_jump_page

            # 连续滚动阅读: 全部页面纵向排列，滚轮从头读到尾，
            # 无需逐页点击（首次渲染整篇需数秒，之后有缓存秒开）
            st.caption(f"共 {n_pages} 页 · 滚动阅读 · 📎证据标注的页码与"
                       f"下方页码标对应 · 有💬标记的页含笔记")
            # 页码->笔记索引（模块3: 逐页笔记展示）
            try:
                import notes_store as _ns_pg
                _pg_notes: dict[int, list] = {}
                for _pn in _ns_pg.list_notes(chosen_id):
                    _pg_notes.setdefault(_pn["page"], []).append(_pn)
            except Exception:
                _pg_notes = {}

            prog = st.progress(0.0, text="正在渲染整篇论文...")
            for i in range(n_pages):
                _mark = f" · 💬笔记×{len(_pg_notes.get(i, []))}" \
                    if i in _pg_notes else ""
                st.image(
                    _render_page(pdf_path, i),
                    caption=f"—— 第 {i + 1} 页 / 共 {n_pages} 页{_mark} ——",
                    use_container_width=True,
                )
                # 该页笔记紧凑展示（引用+批注各一行）
                for _pn2 in _pg_notes.get(i, [])[:3]:
                    _q_txt = _pn2["quote"][:80] + \
                        ("…" if len(_pn2["quote"]) > 80 else "") \
                        if _pn2["quote"] else ""
                    st.markdown(
                        f"💬 *第{i + 1}页笔记*: "
                        + (f"「{_q_txt}」— " if _q_txt else "")
                        + _pn2["note"][:200])
                prog.progress((i + 1) / n_pages,
                              text=f"已渲染 {i + 1}/{n_pages} 页")
            prog.empty()

        # 下载兜底（需要原生PDF阅读器/离线细读时使用）
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ 下载该PDF",
                data=f.read(),
                file_name=f"{chosen_id}.pdf",
                mime="application/pdf",
            )

# ---- 标签C: 对比阅读（成员C · 模块4: 双文献并排卡片对比）----
with tabC:
    st.subheader("⚔️ 对比阅读")
    st.caption("任选两篇论文并排对比**背景 / 方法 / 结论 / 争议点**，"
               "联动阅读笔记与全文检索。信息来自已核验的声明卡片"
               "（流水线未跑完时方法/结论区会提示降级，背景摘要仍可用）。")

    try:
        import compare_view as _cv
        import notes_store as _ns_c

        _sel = _cv.selectable_papers()
        if len(_sel) < 2:
            st.info("本地论文不足2篇，无法对比。请先运行流水线"
                    "（或把PDF放入 papers/ 目录）。")
        else:
            _labels = [s["label"] for s in _sel]
            _id_by_label = {s["label"]: s["id"] for s in _sel}

            # ---- 选择区（等宽双列，防布局错乱的核心: 固定[1,1]比例）----
            _sa_col, _sb_col = st.columns([1, 1])
            with _sa_col:
                _la = st.selectbox("📜 论文 A", _labels, index=0,
                                   key="cmp_a")
            with _sb_col:
                _lb = st.selectbox("📜 论文 B", _labels, index=1
                                   if len(_labels) > 1 else 0, key="cmp_b")
            _ida, _idb = _id_by_label[_la], _id_by_label[_lb]

            if _ida == _idb:
                st.warning("请选择两篇不同的论文（A 与 B 当前相同）")
            else:
                _pa = _cv.build_paper_profile(_ida)
                _pb = _cv.build_paper_profile(_idb)
                _pcs = _cv.pair_conflicts(_ida, _idb)

                # ---- 双卡片并排渲染 ----
                _card_a, _card_b = st.columns([1, 1], gap="medium")

                def _render_cmp_card(col, prof):
                    """单侧对比卡片（模块4核心渲染，布局防错乱:
                    长文本一律截断+expander收纳，绝不让文本撑破列宽）"""
                    with col:
                        _yr = f" · {prof['year']}" if prof.get("year") else ""
                        st.markdown(
                            f"### 📄 {prof['title'][:48]}"
                            + ("…" if len(prof["title"]) > 48 else ""))
                        st.caption(
                            f"arXiv:{prof['paper_id']}{_yr} · "
                            f"被引{prof.get('cited_by', 0)} · "
                            f"作者: {', '.join(prof['authors'][:3])}"
                            + (" 等" if len(prof["authors"]) > 3 else ""))

                        with st.expander("🧭 背景（摘要）", expanded=True):
                            _ab = prof["abstract"]
                            st.markdown(_ab[:300]
                                        + ("…\n\n*展开「查看完整摘要」*"
                                           if len(_ab) > 300 else ""))
                            if len(_ab) > 300:
                                if st.toggle("查看完整摘要",
                                             key=f"ab_{prof['paper_id']}"):
                                    st.markdown(_ab)

                        if prof["has_card"]:
                            with st.expander(
                                    f"🔧 方法（{len(prof['methods'])}条已核验声明）",
                                    expanded=True):
                                for _cm in prof["methods"][:4]:
                                    st.markdown(f"- {_cm[:150]}"
                                                + ("…" if len(_cm) > 150
                                                   else ""))
                            with st.expander(
                                    f"🎯 结论（{len(prof['results'])}条）",
                                    expanded=True):
                                for _cr in prof["results"][:4]:
                                    st.markdown(f"- {_cr[:150]}"
                                                + ("…" if len(_cr) > 150
                                                   else ""))
                        else:
                            st.info("🔧 方法/结论: 需完整运行流水线生成"
                                    "声明卡片后展示（当前仅有背景摘要）",
                                    icon="ℹ️")

                        if prof["limitations"]:
                            with st.expander(
                                    f"⚠️ 局限性（{len(prof['limitations'])}条）"):
                                for _cl in prof["limitations"][:3]:
                                    st.markdown(f"- {_cl[:150]}"
                                                + ("…" if len(_cl) > 150
                                                   else ""))

                        # ---- 笔记联动（模块3 ↔ 模块4）----
                        with st.expander(
                                f"📝 笔记（{prof['notes_count']}条）"):
                            if prof["recent_notes"]:
                                for _rn in prof["recent_notes"]:
                                    st.markdown(
                                        f"💬 **第{_rn['page'] + 1}页** · "
                                        f"{_rn['note'][:120]}")
                            else:
                                st.caption("暂无笔记。到「📚 论文原文」"
                                           "标签页阅读时可添加")
                            if st.button("📖 去阅读这篇",
                                         key=f"rd_{prof['paper_id']}",
                                         use_container_width=True):
                                st.session_state.pdf_jump_id = \
                                    prof["paper_id"]
                                st.session_state.pdf_jump_page = 0
                                st.toast("已定位该论文，请切到「📚 论文原文」"
                                         "标签页查看")

                _render_cmp_card(_card_a, _pa)
                _render_cmp_card(_card_b, _pb)

                # ---- 争议点（两篇之间的冲突，居中单列避免双栏错乱）----
                st.divider()
                st.markdown("#### ⚡ 两篇之间的争议点")
                if _pcs:
                    for _pc in _pcs:
                        _icon = "🔴" if _pc["relation"] == "contradict" \
                            else "🟡"
                        _rel = "直接矛盾" if _pc["relation"] == "contradict" \
                            else "学术张力"
                        with st.expander(
                                f"{_icon} [{_rel}] {_pc['topic']} "
                                f"(严重度{_pc['severity']:.1f})",
                                expanded=True):
                            st.markdown(f"**A方**: {_pc['a_content']}")
                            st.markdown(f"**B方**: {_pc['b_content']}")
                            st.markdown(f"> 🧭 解读: {_pc['explanation']}")
                elif _pa["has_card"] and _pb["has_card"]:
                    st.success("两篇论文之间未发现直接矛盾或张力"
                               "（基于已核验声明的矛盾检测）")
                else:
                    st.caption("争议点需矛盾检测产出（完整运行流水线后"
                               "自动生成）")

                # ---- 全文检索联动提示（模块2 ↔ 模块4）----
                st.divider()
                st.caption("🔗 想深挖某篇? 到「📚 论文原文」标签页的"
                           "「🔍 PDF全文搜索」输入关键词，可跨全部本地论文"
                           "检索并高亮跳转；两篇论文的笔记与阅读进度"
                           "也已在上文卡片联动展示。")
    except Exception as _cv_err:
        st.caption(f"对比视图暂不可用: {_cv_err}")

# ---- 标签9: AI助手（系统使用 + 论文内容 全能问答）----
with tab9:
    st.subheader("🤖 AI助手")
    st.caption("关于**本系统怎么用**、关于**当前检索的论文内容**，"
               "都可以直接问。回答基于系统说明和当前结果数据生成，"
               "论文内容类回答仅使用已核验的信息。")

    # 会话状态: 助手对话历史
    if "assist_history" not in st.session_state:
        st.session_state.assist_history = []

    # ---- 构建助手的知识上下文（每次对话时按当前数据动态生成）----
    def _build_assistant_context() -> str:
        """
        汇总助手可用的全部背景资料:
        1. 系统使用说明（怎么跑流水线、各标签页看什么、指标含义）
        2. 当前结果概况（主题、论文清单、声明池摘要、报告全文）
        """
        parts = []

        # 1. 系统说明
        parts.append(
            "【系统使用说明】\n"
            "- 系统定位: 科研文献整理Agent。用户输入研究主题后自动执行"
            "五段流水线: 规划(生成检索词)→检索(下载论文PDF)→提取(信息卡片)"
            "→审查(两级防幻觉核验)→综合(综述报告)。另有证据定位(PDF原文"
            "高亮穿透)和矛盾检测(学术争议发现)两个后处理。\n"
            "- 操作: 左侧边栏输入研究主题（建议具体，如'脉冲神经网络的"
            "高效训练方法'），选目标论文数，点'🚀 完整运行'（约10-15分钟，"
            "运行中勿重复点击）；已有结果点'📂 载入已有结果'。"
            "侧边栏「检索方式」可选标准或三模式作为流水线检索阶段"
            "（模式未命中自动降级标准检索，降级后仍未命中才终止流水线）。\n"
            "- 🎯三模式检索(🌱入门综述/🚀前沿突破/🔀交叉领域): 通过侧边栏"
            "「检索方式」选择，作为流水线的检索阶段运行，命中文献"
            "进入提取→审查→综合全流程；流水线还会为命中论文生成"
            "数量无上限的核心名词速查表。\n"
            "- 标签页(两层导航): 第一层常用=🔍检索结果(四种排序+徽章) | "
            "📝最终报告(可生成🍼小白解读，支持Word/LaTeX导出) | "
            "💬可信问答(仅基于声明池、"
            "证据不足会拒答) | 🤖AI助手(本页面)；第二层🗂更多=📄信息卡片"
            "(每篇论文的声明+📎证据原文) | ✅审查明细(每条声明核验结果) | "
            "⚡学术争议(论文间矛盾) | 📊综述表格 | 🕸引用图谱 | "
            "📚论文原文(PDF滚动阅读+🔍全文搜索跨文档高亮跳页+📝页面级"
            "笔记增删改+📖核心名词速查对照面板可显示隐藏) | "
            "⚔️对比阅读(双文献并排卡片: 背景/方法/结论/"
            "争议点，联动笔记) | 📖新手指南(3分钟入门)。\n"
            "- 关键指标: 声明=从论文摘出的可核实信息；通过核验=引用真实"
            "且无夸大；幻觉率=AI表述与论文原文不符比例（越低越可信）；"
            "可穿透证据=能定位到PDF原文高亮处的声明数。\n"
            "- 徽章含义: 🔗引用链经典(被多篇种子论文共同引用) | "
            "🔥高被引 | 🤖高度切题(LLM相关性≥4.5)。"
        )

        # 2. 当前结果数据
        if papers:
            topic_cur = st.session_state.run_outputs.get("topic", "")
            lines = [f"【当前结果概况】"]
            if topic_cur:
                lines.append(f"- 检索主题: {topic_cur}")
            lines.append(f"- 收录论文{len(papers)}篇:")
            for p in sorted(papers, key=lambda x: x.final_score or 0,
                            reverse=True)[:15]:
                lines.append(
                    f"  {p.title[:70]} ({p.year}, 被引{p.cited_by}, "
                    f"相关性{p.relevance_score}, arXiv:{p.arxiv_id})"
                )
            parts.append("\n".join(lines))

        # 3. 声明池（论文内容的问答依据）
        try:
            from agents import qa_agent as _qa
            pool = _qa.build_claim_pool()
            if pool:
                lines = [f"【已核验声明池】(共{len(pool)}条，"
                         "回答论文内容问题时只能依据这些)"]
                for c in pool:
                    lines.append(
                        f"[声明#{c['num']}] ({c['type']}, "
                        f"《{c['paper_title']}》{c['year']}年)\n"
                        f"{c['content']}"
                    )
                parts.append("\n".join(lines))
        except Exception:
            pass

        # 4. 综述报告（全文）
        if report_md:
            parts.append(f"【最终综述报告全文】\n{report_md}")

        return "\n\n".join(parts)

    # 渲染历史对话
    for msg in st.session_state.assist_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框（快捷问题提示）
    st.caption("💡 例如: '这套系统怎么保证不说谎？' / '帮我讲讲代理梯度"
               "方法的效果' / '第3篇论文的贡献是什么？'")
    if q := st.chat_input("问系统使用或论文内容的任何问题…"):
        st.session_state.assist_history.append(
            {"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            import llm_client
            try:
                ctx = _build_assistant_context()
                sys_prompt = (
                    "你是'科研文献整理Agent'系统的内置AI助手。用户可能是"
                    "初次使用者或想了解论文内容的研究者。\n"
                    "回答规则:\n"
                    "1. 【系统使用类问题】(怎么操作、指标含义、某个标签页"
                    "是什么): 根据下面的系统说明回答，简洁实用\n"
                    "2. 【论文内容类问题】(方法、结果、结论): 只依据下面"
                    "的'已核验声明池'和'最终综述报告'回答，引用标注"
                    "[声明#N]；池里没有的不要编造，说明'当前结果中没有"
                    "相关信息'\n"
                    "3. 【两者混合】: 分别按对应规则处理\n"
                    "4. 全程用中文，通俗但专业；回答末尾视情况提示用户"
                    "去哪个标签页能看到更多（如'详见📎证据弹层'）\n\n"
                    f"{ctx}"
                )
                history_msgs = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.assist_history[:-1][-6:]
                ]
                ans = st.write_stream(llm_client.chat_stream(
                    [{"role": "system", "content": sys_prompt}]
                    + history_msgs
                    + [{"role": "user", "content": q}],
                    temperature=0.3,
                ))
                st.session_state.assist_history.append(
                    {"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"助手回答失败: {e}")

    # 清空对话按钮
    if st.button("🗑 清空对话"):
        st.session_state.assist_history = []
        st.rerun()
