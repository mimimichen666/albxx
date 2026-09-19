"""
budget.py —— 全局时间预算（防流水线无限卡死）
================================
问题背景:
    检索限流退避、下载重试、LLM多重试叠加后，流水线可能出现
    "病态长跑"——网页转圈几小时不结束。各环节的单次超时/重试
    上限只保证单步有界，无法保证总时长有界。

机制:
    流水线入口调用 start(分钟) 设定硬截止时间；各Agent在
    循环边界（每个检索词/每篇论文/每条声明/每次请求前）调用
    check(阶段名)，超预算立即抛 BudgetExceededError 向上传播，
    由前端捕获并在网页标注失败阶段。

设计要点:
    - 用 time.monotonic() 计时（不受系统时钟跳变影响）
    - check() 开销为一次浮点比较，可安全放进高频循环与线程池工作函数
    - 未 start() 时 check() 是空操作（各Agent单独跑验证脚本不受影响）

使用:
    budget.start(config.GLOBAL_BUDGET_MIN)   # 流水线入口
    budget.check("检索Agent")                 # 循环边界
"""

import time


class BudgetExceededError(RuntimeError):
    """全局时间预算耗尽。stage 记录耗尽时所处的流水线阶段。"""

    def __init__(self, stage: str = ""):
        self.stage = stage or "未知"
        super().__init__(f"全局时间预算耗尽于【{self.stage}】阶段")


# 硬截止时间（monotonic时钟，None=未启用预算）
_deadline: float | None = None


def start(minutes: float) -> None:
    """启动全局预算：从现在起 minutes 分钟后视为超时。"""
    global _deadline
    _deadline = time.monotonic() + max(0.0, minutes) * 60.0


def stop() -> None:
    """清除预算（流水线正常结束后调用，避免影响同进程的后续独立运行）。"""
    global _deadline
    _deadline = None


def remaining_seconds() -> float | None:
    """剩余预算秒数；未启用时返回None。"""
    if _deadline is None:
        return None
    return _deadline - time.monotonic()


def exceeded() -> bool:
    """预算是否已耗尽。"""
    r = remaining_seconds()
    return r is not None and r <= 0


def check(stage: str = "") -> None:
    """预算耗尽则抛 BudgetExceededError（开销极小，可放高频循环）。"""
    if exceeded():
        raise BudgetExceededError(stage)
