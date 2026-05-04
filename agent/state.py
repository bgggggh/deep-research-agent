"""
agent/state.py
LangGraph 全局状态 —— 所有 Agent 节点共享同一个 State 对象

Stage 3 新增字段：
- retrieved_chunks: Retriever 检索到的 chunks（按子问题分组）
- need_web_search: 哪些子问题需要走 web search（RAG 不足）
"""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    # ── 输入 ──────────────────────────────────────────────
    query: str                              # 用户原始问题

    # ── Planner 产出 ──────────────────────────────────────
    research_plan: list[str]                # 分解出的子问题列表
    current_step: int                       # 当前执行到第几个子问题

    # ── Retriever 产出（Stage 3 新增）──────────────────────
    retrieved_chunks: list[dict]
    # 每条格式: {"sub_query": str, "parent_text": str, "child_text": str,
    #           "namespace": str, "score": float, "metadata": dict}

    need_web_search: list[bool]
    # 每个子问题是否需要 web search（True=需要，False=RAG足够）
    # 长度与 research_plan 一致

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