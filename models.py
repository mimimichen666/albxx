"""
models.py —— Pydantic 数据结构定义
================================
作用：定义各个Agent之间传递数据的"契约"。
每个结构对应流程中的一个环节，是整个系统的数据骨架。

数据流:
  SearchPlan(规划) -> PaperMeta(检索) -> PaperCard(提取)
  -> ReviewResult(审查) -> 最终综述报告
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ===============================================================
# 1. 规划Agent的输出：检索计划
# ===============================================================
class SearchPlan(BaseModel):
    """规划Agent把用户输入的研究主题分解成的检索计划"""
    topic: str = Field(description="用户输入的原始研究主题")
    keywords: list[str] = Field(
        description="3-5组不同角度的英文检索词（arXiv是英文库）",
        min_length=1, max_length=8,
    )
    classic_titles: list[str] = Field(
        default=[],
        description="领域奠基/经典论文的精确英文标题清单（5-8篇）。"
                    "这些论文的标题不含现代检索词，关键词检索召不回，"
                    "由LLM领域知识直接给出，检索Agent按标题反查收录",
        max_length=12,
    )
    inclusion_criteria: str = Field(
        description="论文筛选标准，如：近5年内、与主题直接相关、有实验结果"
    )
    target_count: int = Field(
        description="目标收集的论文数量（建议5-20篇）",
        ge=3, le=30,
    )


# ===============================================================
# 2. 检索Agent的输出：论文元数据
# ===============================================================
class PaperMeta(BaseModel):
    """从arXiv检索到的单篇论文的元信息"""
    arxiv_id: str = Field(description="arXiv编号，如 2401.12345")
    title: str = Field(description="论文标题")
    authors: list[str] = Field(description="作者列表")
    year: int = Field(description="发表年份")
    abstract: str = Field(description="摘要原文")
    pdf_url: str = Field(description="PDF下载链接")
    relevance_score: float | None = Field(
        default=None,
        description="LLM按筛选标准打的 relevance 相关性得分(1-5)，None表示尚未评分",
    )
    rrf_score: float = Field(
        default=0.0,
        description="多路检索RRF融合分(Reciprocal Rank Fusion)，"
                    "综合'被引影响力路'与'相关性路'的排名信号，"
                    "越高代表越经典/越相关。用于LLM同分时的排序兜底",
    )
    cited_by: int = Field(
        default=0,
        description="OpenAlex的被引次数（影响力先验，0表示数据源未提供）",
    )
    openalex_id: str | None = Field(
        default=None,
        description="OpenAlex作品ID（W开头），引用链挖掘的数据源标识，"
                    "None表示该论文来自arXiv/S2兜底数据源",
    )
    chain_hits: int = Field(
        default=0,
        description="引用链共被引次数——被几篇种子论文的参考文献共同引用。"
                    ">=2是领域经典信号，>=3触发LLM粗筛免检通道",
    )
    final_score: float | None = Field(
        default=None,
        description="最终综合排序分: 相关性+RRF共识+影响力+引用链四路加权，"
                    "None表示尚未计算（前端展示排序依据用）",
    )
    local_path: str | None = Field(
        default=None,
        description="下载到本地的PDF路径，None表示尚未下载",
    )


# ===============================================================
# 3. 提取Agent的输出：论文信息卡片（★证据库的核心单元）
# ===============================================================
class Quote(BaseModel):
    """支撑某条声明的原文引用（防幻觉机制的基石）"""
    section: str = Field(description="出处位置，如 '3.2节' 或 'Abstract'")
    text: str = Field(description="原文逐字引用(verbatim)，不许改写")


class Claim(BaseModel):
    """一条可被审查核验的"声明"（提取出的信息以声明为单位送审）"""
    claim_type: str = Field(
        description="声明类别: method(方法) / result(结果) / limitation(局限)"
    )
    content: str = Field(description="声明内容，如'该方法在CIFAR-10上达到95.2%准确率'")
    quotes: list[Quote] = Field(
        description="支撑该声明的原文引用，至少1条",
        min_length=1,
    )


class PaperCard(BaseModel):
    """单篇论文的完整信息卡片"""
    arxiv_id: str = Field(description="arXiv编号，用于和PaperMeta对应")
    title: str = Field(description="论文标题")
    year: int = Field(description="发表年份")
    method_category: str = Field(
        description="方法分类标签，如'监督训练'/'生物可塑性'/'混合架构'（用于综述表格分组）"
    )
    claims: list[Claim] = Field(description="提取出的所有声明列表")


# ===============================================================
# 4. 审查Agent的输出：核验结果（★幻觉率量化实验的核心）
# ===============================================================
class ReviewVerdict(BaseModel):
    """对单条声明的核验判定

    字段顺序即生成顺序(2026-09-22实验③迭代): reason放在verdict之前,
    让模型"先写逐项核对的理由, 再下结论"——理由成为思维链。
    实测: 先结论后理由时, 审查员对中文声明的单词级偷换
    (如"服务导向→平台导向")不敏感, 拦截率仅~50%。

    abstained = 证据不足、暂缓判定（不等于幻觉，进入"暂缓"桶单独披露）。
    """
    reason: str = Field(
        description="判定理由: 先列出声明中的关键事实(数字/名称/范围词), "
                    "并逐项注明原文片段中对应的英文表述; 若有不符, 明确指出"
                    "是哪一项对不上"
    )

    @field_validator("reason", mode="before")
    @classmethod
    def _coerce_reason(cls, v):
        """宽容化: LLM按'关键事实清单'提示词常把reason输出成结构化列表
        (如[{'fact': 'PLIF神经元', 'quote': ...}])而非纯字符串, 之前会
        连续3次校验失败→整条声明误判为unsupported。这里把列表/字典
        拼接为可读文本, 语义不变, 兼容两种输出形态。"""
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, dict):
                    parts.append("；".join(f"{k}: {val}"
                                           for k, val in item.items()))
                else:
                    parts.append(str(item))
            return "；".join(parts)
        if isinstance(v, dict):
            return "；".join(f"{k}: {val}" for k, val in v.items())
        return v

    verdict: Literal[
        "supported", "unsupported", "contradicted", "abstained"
    ] = Field(
        description="supported=原文支持；unsupported=原文不支持；"
                    "contradicted=与原文矛盾；abstained=证据不足暂缓判定"
    )
    confidence: float = Field(
        description="判定置信度0-1", ge=0.0, le=1.0,
    )


class AtomicFactVerdict(BaseModel):
    """原子事实核验记录：把声明拆成最小可判定单元后逐条判定。"""
    fact: str = Field(description="从声明中拆出的单条原子事实")
    verdict: Literal[
        "supported", "unsupported", "contradicted", "abstained"
    ] = Field(description="该原子事实的判定（含义同 ReviewVerdict.verdict）")
    reason: str = Field(description="一句话判定理由")
    confidence: float = Field(
        description="判定置信度0-1", ge=0.0, le=1.0,
    )


class SemanticVerifyResult(BaseModel):
    """语义核验的完整输出（原子事实拆分 + 程序化汇总）。

    LLM只负责逐条原子事实判定；verdict 由 reviewer 按最坏非暂缓判定
    程序化推导，保证与原子事实严格一致（防止模型自相矛盾）。
    """
    atomic_facts: list[AtomicFactVerdict] = Field(
        description="从声明拆出的原子事实及逐条判定", min_length=1, max_length=9,
    )
    verdict: Literal[
        "supported", "unsupported", "contradicted", "abstained"
    ] = Field(description="汇总判定（程序会按原子事实重新推导覆盖此字段）")
    reason: str = Field(description="一句话判定理由（针对声明整体）")
    confidence: float = Field(
        description="对声明整体判定的置信度0-1", ge=0.0, le=1.0,
    )


class QuoteMatchInfo(BaseModel):
    """单条引用的对齐详情（阶段2：引用定位升级的产出）。

    page 为 0-based 页码（与 evidence_locator 口径一致），None=无法定位页。
    """
    section: str = Field(default="", description="引用声称的出处位置")
    aligned: bool = Field(description="该引用是否成功对齐到原文")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="对齐置信度：精确匹配=1.0，模糊匹配=相似度比值",
    )
    matched_fragment: str | None = Field(
        default=None, description="实际命中的原文片段（模糊匹配时与引用不同）")
    page: int | None = Field(
        default=None, ge=0, description="命中片段所在的0-based页码")
    context: str | None = Field(
        default=None, description="命中位置前后各约100字符的上下文")


class ReviewResult(BaseModel):
    """一次完整的核验记录。

    新增字段均有默认值，保证旧版 review_results.json 仍可加载。
    """
    arxiv_id: str = Field(description="所属论文")
    claim_index: int = Field(description="声明在PaperCard.claims中的下标")
    claim_content: str = Field(description="声明内容（冗余存储，方便人工标注）")
    quote_alignment: bool = Field(
        description="是否至少有一条引用与论文原文对齐"
    )
    verdict: ReviewVerdict = Field(description="LLM审查员的语义核验判定")
    aligned_quote_count: int = Field(
        default=0, ge=0, description="成功对齐的引用数量"
    )
    quote_count: int = Field(
        default=0, ge=0, description="该声明的引用总数"
    )
    hallucination: bool = Field(
        default=False, description="按统一协议，该声明是否判定为幻觉"
    )
    failure_mode: Literal[
        "none", "fake_quote", "unsupported", "contradicted",
        "mixed", "abstained", "llm_error",
    ] = Field(
        default="none",
        description="失败模式。abstained=证据不足暂缓(非幻觉)；"
                    "llm_error=核验调用失败(不计入幻觉)",
    )
    review_latency_ms: float | None = Field(
        default=None, ge=0, description="该声明两级核验耗时（毫秒）"
    )
    review_status: Literal["ok", "llm_error"] = Field(
        default="ok",
        description="核验过程状态。llm_error=LLM调用失败未完成语义判定，"
                    "此时verdict仅为占位，统计中单列，绝不计为幻觉",
    )
    quote_details: list[QuoteMatchInfo] = Field(
        default_factory=list,
        description="每条引用的对齐详情（片段/置信度/页码/上下文）",
    )
    atomic_facts: list[AtomicFactVerdict] = Field(
        default_factory=list,
        description="原子事实拆分核验记录（语义核验升级的中间产物）",
    )


class BenchmarkCase(BaseModel):
    """D Benchmark 的单条人工标注样本。

    gold_verdict 是人工/数据集标签，不参与模型判定，只用于离线评测。
    """
    case_id: str
    arxiv_id: str | None = None
    claim_content: str
    evidence_text: str
    quote_text: str | None = None
    full_text: str | None = None
    gold_verdict: Literal["supported", "unsupported", "contradicted"]
    gold_quote_alignment: bool


class BenchmarkResult(BaseModel):
    """D Benchmark 单条样本的预测结果。"""
    case_id: str
    predicted_verdict: Literal["supported", "unsupported", "contradicted"]
    predicted_quote_alignment: bool
    verdict_correct: bool
    alignment_correct: bool
    # K2·B3优化: 保留人工标注真值——bootstrap置信区间重采样时
    # 需要从逐条结果还原逐类P/R/F1与误放行率，没有gold算不了
    gold_verdict: str | None = Field(
        default=None, description="人工标注真值(与BenchmarkCase.gold_verdict一致)")
    gold_quote_alignment: bool | None = Field(
        default=None, description="人工标注引用对齐真值——bootstrap重采样"
        "还原误放行率时需要（坏声明=gold非supported或gold引用伪造）")


class BenchmarkReport(BaseModel):
    """D Benchmark 汇总指标。

    统一指标定义（阶段4）:
      - Precision/Recall/F1: 按三分类逐类计算，另给 Macro 平均
      - 误放行率 false_pass_rate: 人工标注为"坏声明"(verdict!=supported
        或 引用伪造)却被系统放行(verdict=supported且对齐成功)的比例
    """
    total_cases: int
    verdict_correct: int
    alignment_correct: int
    supported_cases: int
    unsupported_cases: int
    contradicted_cases: int
    verdict_accuracy: float
    alignment_accuracy: float
    macro_f1: float
    results: list[BenchmarkResult] = Field(default_factory=list)
    precision: dict[str, float] = Field(
        default_factory=dict, description="逐类精确率")
    recall: dict[str, float] = Field(
        default_factory=dict, description="逐类召回率")
    f1: dict[str, float] = Field(
        default_factory=dict, description="逐类F1")
    false_pass_rate: float = Field(
        default=0.0, description="误放行率：坏声明被判定为通过的比例")
    errored_cases: int = Field(
        default=0, ge=0, description="LLM调用失败被跳过的样本数(不计入指标)")


# ===============================================================
# 5. 幻觉率统计（实验E1的产出）
# ===============================================================
class HallucinationReport(BaseModel):
    """一轮实验的幻觉率统计报告。

    统一指标定义（阶段1协议 v2）:
      - 幻觉率 = hallucinations / total_claims
        幻觉 = (所有引用未对齐) OR (语义判定 unsupported/contradicted)；
        LLM调用失败与证据不足(abstained)的声明绝不计为幻觉。
      - 引用对齐率 = 至少一条引用成功对齐的声明数 / total_claims
      - 核验成功率 = 完成语义判定的声明数 / total_claims
                    （= 1 - llm_errors/total，衡量核验管线自身的可靠性）
    """
    total_claims: int = Field(description="送审的声明总数")
    supported: int = Field(description="语义判定为原文支持的条数")
    unsupported: int = Field(description="语义判定为原文不支持的条数")
    contradicted: int = Field(description="语义判定为与原文矛盾的条数")
    fake_quotes: int = Field(description="程序检测出的引用未对齐条数")
    hallucinations: int | None = Field(
        default=None, ge=0, description="按统一协议判定为幻觉的声明数（去重后）"
    )
    aligned_quotes: int | None = Field(
        default=None, ge=0, description="至少一条引用成功对齐的声明数"
    )
    llm_errors: int = Field(
        default=0, ge=0, description="LLM调用失败、未完成语义判定的声明数")
    abstained: int = Field(
        default=0, ge=0, description="证据不足暂缓判定的声明数（非幻觉）")

    @property
    def hallucination_count(self) -> int:
        if self.hallucinations is not None:
            return self.hallucinations
        return min(self.total_claims, self.unsupported + self.contradicted + self.fake_quotes)

    @property
    def hallucination_rate(self) -> float:
        return self.hallucination_count / self.total_claims if self.total_claims else 0.0

    @property
    def quote_alignment_rate(self) -> float:
        aligned = self.aligned_quotes
        if aligned is None:
            aligned = self.total_claims - self.fake_quotes
        return aligned / self.total_claims if self.total_claims else 0.0

    @property
    def verify_success_rate(self) -> float:
        """核验成功率：完成语义判定的声明占比（LLM故障单列，不冒充判定）。"""
        if not self.total_claims:
            return 0.0
        return (self.total_claims - self.llm_errors) / self.total_claims


# ----------------------------------------------------------------
# 检索前引导（词云）
# ----------------------------------------------------------------
class GlossaryTerm(BaseModel):
    """词云中的单个术语条目"""
    term_en: str
    term_zh: str
    weight: int = Field(ge=1, le=10)      # 与主题相关度，决定词云字号与排序
    category: Literal["core", "method", "application", "background"]
    explanation: str

    @field_validator("explanation")
    @classmethod
    def _trim_expl(cls, v: str) -> str:
        return (v or "").strip()[:100]


class TopicGlossary(BaseModel):
    """LLM 为主体生成的主题词云：术语集合 + 主题英译"""
    topic_en: str
    terms: list[GlossaryTerm] = Field(min_length=8, max_length=30)
