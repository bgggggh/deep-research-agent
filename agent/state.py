"""
agent/state.py
LangGraph 全局状态 —— 所有 Agent 节点共享同一个 State 对象

使用 TypedDict 是 LangGraph 的惯用法：
- 节点函数接收/返回 dict
- 字段类型有静态检查
- 不需要 dataclass 那一套 __dict__ 操作
"""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    # ── 输入 ──────────────────────────────────────────────
    query: str                              # 用户原始问题

    # ── Planner 产出 ──────────────────────────────────────
    research_plan: list[str]                # 分解出的子问题列表
    current_step: int                       # 当前执行到第几个子问题

    # ── Searcher 产出 ─────────────────────────────────────
    search_results: list[dict]
    # 每条格式: {"sub_query": str, "url": str, "title": str, "snippet": str, "content": str}

    # ── Critic 产出 ───────────────────────────────────────
    critic_verdict: str                     # "continue" | "sufficient" | "replan"
    critic_feedback: str                    # Critic 的具体反馈

    # ── Writer 产出 ───────────────────────────────────────
    final_report: str                       # 最终报告（带引用）
    citations: list[dict]
    # 每条格式: {"claim": str, "source_url": str, "snippet": str}

    # ── 元信息 ────────────────────────────────────────────
    iteration_count: int                    # 防止死循环的计数器
    max_iterations: int                     # Critic 最多允许重搜几次
    token_usage: dict                       # cost tracking
    messages: Annotated[list, add_messages]