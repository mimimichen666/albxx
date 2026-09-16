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

from pydantic import BaseModel, Field


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
    """对单条声明的核验判定。"""
    verdict: Literal["supported", "unsupported", "contradicted"] = Field(
        description="supported=原文支持；unsupported=原文不支持；contradicted=与原文矛盾"
    )
    reason: str = Field(description="一句话判定理由")
    confidence: float = Field(
        description="判定置信度0-1", ge=0.0, le=1.0,
    )


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
    failure_mode: Literal["none", "fake_quote", "unsupported", "contradicted", "mixed"] = Field(
        default="none", description="失败模式，便于D Benchmark统计"
    )
    review_latency_ms: float | None = Field(
        default=None, ge=0, description="该声明两级核验耗时（毫秒）"
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


class BenchmarkReport(BaseModel):
    """D Benchmark 汇总指标。"""
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


# ===============================================================
# 5. 幻觉率统计（实验E1的产出）
# ===============================================================
class HallucinationReport(BaseModel):
    """一轮实验的幻觉率统计报告。

    hallucinations 使用“并集”计数：一条声明即使同时引用失败且语义失败，也只计一次。
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
