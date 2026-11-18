"""
onboarding.py —— 检索前引导：画像 → 词云选词 → 术语卡 → 确认启动
================================================================
在流水线启动前增加一步引导，让系统了解用户的基本情况与阶段所需：
  1. 主题输入（预填侧边栏主题）
  2. 用户画像：身份 + 所处阶段 → 自动映射检索方式/深度/篇数预设
  3. 词云选词：LLM按主题生成15-25个领域术语（带权重+分类+一句话解释），
     每个术语渲染为可点开的"词卡"（st.popover），支持加入检索词与纠错
  4. 确认启动：汇总预览，参数可覆盖修改后发起流水线

设计原则:
  - 引导永不阻塞主流程: LLM失败可一键跳过，侧边栏快速通道原样保留
  - 术语解释标注"AI生成"，与系统防幻觉人设一致；用户纠错写入
    data/glossary_edits.json，优先级高于LLM版
  - 词云生成走 llm_client.chat_json（自带缓存），同主题二次进入零成本
"""

import json
import os
from datetime import datetime

import streamlit as st

import config
import llm_client
from models import TopicGlossary

EDITS_PATH = os.path.join(config.DATA_DIR, "glossary_edits.json")
PROFILE_PATH = os.path.join(config.DATA_DIR, "run_profile.json")

# 术语分类的展示顺序与说明
CATEGORY_LABELS = [
    ("core", "🧩 核心概念 · 了解它是什么"),
    ("method", "🔧 方法技术 · 怎么做的"),
    ("application", "🚀 应用场景 · 用在哪里"),
    ("background", "📚 背景知识 · 前置基础"),
]

IDENTITY_OPTIONS = ["本科生", "硕士生", "博士生", "研究者", "工程师"]
STAGE_OPTIONS = ["入门了解", "开题调研", "正式综述", "追前沿", "交叉探索"]

# 用户阶段 → 流水线参数预设（confirm页可覆盖）
STAGE_PRESETS = {
    "入门了解": {"retrieval_mode": "mode1", "fast": True, "target_count": 5},
    "开题调研": {"retrieval_mode": "standard", "fast": False, "target_count": 8},
    "正式综述": {"retrieval_mode": "standard", "fast": False, "target_count": 10},
    "追前沿": {"retrieval_mode": "mode2", "fast": False, "target_count": 5},
    "交叉探索": {"retrieval_mode": "mode3", "fast": False, "target_count": 8},
}

MODE_LABELS = {
    "standard": "📌 标准（推荐）",
    "mode1": "🌱 入门综述模式",
    "mode2": "🚀 前沿突破模式",
    "mode3": "🔀 交叉领域模式",
}
MODE_BY_LABEL = {v: k for k, v in MODE_LABELS.items()}

# 身份 → 术语解释的语言深度要求（注入prompt）
_IDENTITY_STYLE = {
    "本科生": "面向初学者：多用生活化类比，不出现公式，先讲直觉再讲术语",
    "硕士生": "面向有基础的研究生：可用领域惯用语，讲清机制与用途",
    "博士生": "面向研究者：精炼准确，突出技术脉络与开放问题",
    "研究者": "面向研究者：精炼准确，突出技术脉络与开放问题",
    "工程师": "面向工程落地视角：强调解决什么问题、代价与适用条件",
}

GLOSSARY_SYSTEM = (
    "你是学术文献调研的引导助手。用户即将启动一次文献综述流水线，"
    "你要为他的研究主题生成一个'引导词云'：列出该主题相关的专业术语，"
    "并对每个术语给出一句话中文解释，帮助用户（尤其是新手）在检索前"
    "建立概念地图。只输出JSON。"
)


# ---------------------------------------------------------------
# LLM词云生成（chat_json自带缓存，同主题第二次进入零成本）
# ---------------------------------------------------------------
def generate_glossary(topic: str, identity: str, stage: str) -> TopicGlossary:
    """按主题+用户画像生成术语词云。失败抛异常，由调用方回退"""
    style = _IDENTITY_STYLE.get(identity, _IDENTITY_STYLE["硕士生"])
    user_msg = (
        f"研究主题: {topic}\n"
        f"用户身份: {identity}\n"
        f"所处阶段: {stage}（{'入门新手，覆盖更多背景知识' if stage == '入门了解' else '需要较完整的专业术语覆盖'}）\n\n"
        "请生成该主题的引导词云：\n"
        "1. terms 给出15-25个相关术语，四类（core/method/application/"
        "background）尽量都有覆盖，避免只出近义词\n"
        "2. weight为与主题的相关度1-10，10=最核心\n"
        "3. explanation为不超过80字的中文一句话定义，语言风格要求: "
        f"{style}\n"
        "4. term_en用该领域标准的英文术语原文；term_zh为通用中文译名\n"
        "5. topic_en给出主题的准确英文翻译\n"
        "6. 禁止编造文献引用或arXiv编号"
    )
    return llm_client.chat_json(
        [{"role": "system", "content": GLOSSARY_SYSTEM},
         {"role": "user", "content": user_msg}],
        schema_class=TopicGlossary,
        temperature=config.TEMP_PLANNER,
    )


# ---------------------------------------------------------------
# 用户纠错词条（glossary_edits.json，优先级高于LLM版）
# ---------------------------------------------------------------
def load_edits() -> dict:
    try:
        with open(EDITS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_edit(term_en: str, explanation: str) -> None:
    edits = load_edits()
    edits[term_en] = {
        "explanation": explanation,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(EDITS_PATH, "w", encoding="utf-8") as f:
        json.dump(edits, f, ensure_ascii=False, indent=2)


def _write_profile(record: dict) -> None:
    """画像快照随本次运行持久化，'载入已有结果'时可展示"""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------
# 渲染辅助
# ---------------------------------------------------------------
def _progress_bar(cur: str) -> None:
    """步骤指示: 当前步高亮"""
    steps = [("topic", "①主题"), ("profile", "②画像"),
             ("cloud", "③选词"), ("confirm", "④确认")]
    parts = []
    for key, label in steps:
        if key == cur:
            parts.append(f"**{label}**")
        else:
            parts.append(label)
    st.caption(" → ".join(parts))


def _term_explanation(term, edits: dict) -> tuple[str, bool]:
    """返回(生效的解释, 是否用户修订版)"""
    edit = edits.get(term.term_en)
    if edit and edit.get("explanation"):
        return edit["explanation"], True
    return term.explanation, False


def _render_cloud(glossary: TopicGlossary, edits: dict) -> list[str]:
    """词云: 按分类分组的词卡网格，返回当前已选术语列表"""
    st.caption(f"主题英译将用于检索: `{glossary.topic_en}`")
    selected: list = st.session_state.get("ob_selected", [])
    for cat_key, cat_label in CATEGORY_LABELS:
        terms = sorted(
            [t for t in glossary.terms if t.category == cat_key],
            key=lambda t: -t.weight)
        if not terms:
            continue
        st.markdown(f"**{cat_label}**")
        cols = st.columns(3)
        for i, term in enumerate(terms):
            with cols[i % 3]:
                expl, edited = _term_explanation(term, edits)
                in_sel = term.term_zh in selected
                mark = "✅ " if in_sel else ""
                with st.popover(f"{mark}{term.term_zh}", use_container_width=True):
                    st.caption(f"{term.term_en} · 相关度 {term.weight}/10")
                    if edited:
                        st.info("此释义已由用户修订", icon="✏️")
                    st.write(expl)
                    if st.button("移出检索词" if in_sel else "✓ 加入检索词",
                                 key=f"ob_toggle_{term.term_en}",
                                 use_container_width=True):
                        if in_sel:
                            selected.remove(term.term_zh)
                        else:
                            selected.append(term.term_zh)
                        st.session_state.ob_selected = selected
                        st.rerun()
                    with st.expander("✏️ 纠错/补充释义"):
                        new_expl = st.text_area("修订释义", value=expl,
                                                key=f"ob_edit_{term.term_en}")
                        if st.button("保存修订", key=f"ob_save_{term.term_en}"):
                            if new_expl.strip():
                                _save_edit(term.term_en, new_expl.strip())
                                st.toast("释义已修订，下次展示生效", icon="✏️")
                            st.rerun()
    st.divider()
    st.caption(f"已选 **{len(selected)}** 个检索词: "
               + ("、".join(selected) if selected else "（未选，将仅按主题检索）"))
    return selected


# ---------------------------------------------------------------
# 主入口: 引导状态机（返回None=不启动; 返回dict=启动参数）
# ---------------------------------------------------------------
def render_onboarding(sidebar_topic: str) -> dict | None:
    """在欢迎页区域调用。引导完成后返回流水线启动参数字典"""
    ss = st.session_state
    stage = ss.get("ob_stage", "topic")

    st.subheader("🧭 检索前引导")
    st.caption("花30秒告诉我们你的情况，系统会生成领域词云帮你选准检索词，"
               "术语不懂可以直接点开看解释；也可直接用侧边栏快速通道跳过")
    _progress_bar(stage)

    # ---------- 步骤①: 主题 ----------
    if stage == "topic":
        ob_topic = st.text_input(
            "研究主题（一句话中文即可，系统自动翻译成英文检索词）",
            value=ss.get("ob_topic", sidebar_topic),
            key="ob_topic_input",
        )
        c1, c2 = st.columns([1, 1])
        if c1.button("下一步：选择画像 →", type="primary",
                     use_container_width=True):
            if not ob_topic.strip():
                st.warning("请先填写研究主题")
            else:
                ss.ob_topic = ob_topic.strip()
                ss.ob_stage = "profile"
                st.rerun()
        if c2.button("跳过引导，直接检索", use_container_width=True,
                     help="不生成词云，直接用侧边栏参数发起流水线"):
            ss.ob_stage = "confirm"
            ss.ob_glossary = None
            ss.ob_selected = []
            st.rerun()
        return None

    # ---------- 步骤②: 用户画像 ----------
    if stage == "profile":
        identity = st.radio("你的身份", IDENTITY_OPTIONS,
                            index=1, horizontal=True,
                            key="ob_identity",
                            help="决定术语解释的语言深度")
        stage_sel = st.radio(
            "你当前所处阶段", STAGE_OPTIONS, horizontal=True,
            help="系统将据此推荐检索方式、运行深度与目标篇数")
        focus_note = st.text_input(
            "额外关注点（可选）", value=ss.get("ob_focus_val", ""),
            placeholder="如：关注硬件实现 / 只看近两年的工作",
            key="ob_focus_input")
        c1, c2, _ = st.columns([1, 1, 2])
        if c1.button("← 上一步", use_container_width=True):
            ss.ob_stage = "topic"
            st.rerun()
        if c2.button("🪄 生成词云与术语卡", type="primary",
                     use_container_width=True):
            # 注意: 不能直接赋值 ss.ob_identity (与radio的key同名会触发
            # StreamlitAPIException); 且widget在后续步骤不再渲染时其
            # session_state会被清理, 因此复制到独立的非widget键长期保存
            ss.ob_identity_val = identity
            ss.ob_stage_sel = stage_sel
            ss.ob_focus_val = focus_note
            try:
                with st.spinner("正在生成领域词云与术语解释（首次约需十几秒）..."):
                    ss.ob_glossary = generate_glossary(
                        ss.ob_topic, identity, stage_sel)
                ss.ob_selected = []
                ss.ob_stage = "cloud"
            except Exception as e:
                st.error(f"词云生成失败: {e}")
                st.caption("可直接跳过引导检索，或返回重试")
                ss.ob_glossary = None
                ss.ob_selected = []
            st.rerun()
        return None

    # ---------- 步骤③: 词云选词 ----------
    if stage == "cloud":
        glossary = ss.get("ob_glossary")
        if glossary is None:
            st.info("词云未生成（可能上次生成失败），"
                    "可直接进入确认页用主题检索")
        else:
            _render_cloud(glossary, load_edits())
        c1, c2, _ = st.columns([1, 1, 2])
        if c1.button("← 重新生成", use_container_width=True):
            ss.ob_stage = "profile"
            st.rerun()
        if c2.button("下一步：确认启动 →", type="primary",
                     use_container_width=True):
            ss.ob_stage = "confirm"
            st.rerun()
        return None

    # ---------- 步骤④: 确认启动 ----------
    if stage == "confirm":
        preset_stage = ss.get("ob_stage_sel", "开题调研")
        preset = STAGE_PRESETS.get(preset_stage, STAGE_PRESETS["开题调研"])
        # 参数widget状态修复: radio/slider的session_state会跨渲染残留,
        # 阶段变更或会话重连时旧状态会覆盖新预设, 造成"显示值"与
        # "落盘值"不一致。首次进入确认页或阶段变更时, 重置参数控件
        # 为该阶段预设默认值; 同一确认页内的手动调整则照常保留
        if ss.get("ob_preset_stage") != preset_stage \
                or "ob_count_slider" not in ss:
            ss.ob_preset_stage = preset_stage
            for _k in ("ob_mode_radio", "ob_depth_radio",
                       "ob_count_slider", "ob_years_slider"):
                ss.pop(_k, None)
        glossary = ss.get("ob_glossary")
        selected = ss.get("ob_selected", [])
        topic = ss.get("ob_topic", sidebar_topic)
        final_topic = (f"{topic}（重点关注：{'；'.join(selected)}）"
                       if selected else topic)

        with st.container(border=True):
            st.markdown(f"**主题**: {final_topic}")
            if glossary is not None:
                st.caption(f"画像: {ss.get('ob_identity_val', '')} · "
                           f"{ss.get('ob_stage_sel', '')}"
                           + (f" · 关注: {ss.get('ob_focus_val', '')}"
                              if ss.get("ob_focus_val") else ""))
            # 预设参数（全部可覆盖）。radio选项必须用中文标签(与MODE_BY_LABEL
            # 的键一致), 若传模式代码会导致后续MODE_BY_LABEL[mode_label] KeyError
            _mode_options = list(MODE_LABELS.values())
            mode_label = st.radio(
                "检索方式", _mode_options,
                index=_mode_options.index(
                    MODE_LABELS[preset["retrieval_mode"]]),
                key="ob_mode_radio", horizontal=True)
            depth_label = st.radio(
                "运行深度", ["⚡ 快速（约2-3分钟）", "🔬 深度（约10分钟）"],
                index=0 if preset["fast"] else 1,
                key="ob_depth_radio", horizontal=True)
            target_count = st.slider("目标论文数", 3, 15,
                                     preset["target_count"],
                                     key="ob_count_slider")
            mode_years = 2
            if MODE_BY_LABEL[mode_label] == "mode2":
                mode_years = st.slider("前沿时间范围（近N年）", 1, 5, 2,
                                       key="ob_years_slider")
        c1, c2, _ = st.columns([1, 1, 2])
        if c1.button("← 返回修改", use_container_width=True):
            ss.ob_stage = "cloud" if glossary is not None else "profile"
            st.rerun()
        if c2.button("🚀 开始检索", type="primary", use_container_width=True):
            _write_profile({
                "topic": final_topic,
                "topic_en": glossary.topic_en if glossary else None,
                "identity": ss.get("ob_identity_val"),
                "stage": ss.get("ob_stage_sel"),
                "focus_note": ss.get("ob_focus_val") or None,
                "selected_terms": selected,
                "preset": {
                    "retrieval_mode": MODE_BY_LABEL[mode_label],
                    "fast": depth_label.startswith("⚡"),
                    "target_count": target_count,
                    "mode_years": mode_years,
                },
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })
            # 清理引导状态，避免下次进入残留
            for k in ("ob_stage", "ob_glossary", "ob_selected",
                      "ob_topic", "ob_identity_val", "ob_focus_val",
                      "ob_stage_sel", "ob_preset_stage"):
                ss.pop(k, None)
            return {
                "topic": final_topic,
                "target_count": target_count,
                "retrieval_mode": MODE_BY_LABEL[mode_label],
                "mode_years": mode_years,
                "fast": depth_label.startswith("⚡"),
            }
        return None

    return None
