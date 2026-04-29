"""
agent/graph.py
LangGraph 主图 —— 把四个 Agent 节点连接成状态机

流程：
  START
    → planner          # 分解问题
    → searcher         # 搜索当前子问题
    → critic           # 评估证据充分性
    → [sufficient] → writer → END
    → [continue]   → searcher (继续下一个子问题)
    → [replan]     → planner  (重新规划)
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import ResearchState
from agent.nodes.agents import (
    planner_node,
    searcher_node,
    critic_node,
    writer_node,
)


def route_after_critic(state) -> str:
    """
    Critic 之后的路由函数。
    LangGraph 的 conditional_edge 会调用这个函数决定下一个节点。
    state 是 dict-like，用 .get() 安全访问。
    """
    verdict = state.get("critic_verdict", "continue")
    current_step = state.get("current_step", 0)
    plan_length = len(state.get("research_plan", []))

    if verdict == "sufficient":
        return "writer"

    if verdict == "replan":
        return "planner"

    # "continue": 还有子问题未搜索就继续，否则进 writer
    if current_step < plan_length:
        return "searcher"
    else:
        return "writer"


def build_graph() -> StateGraph:
    """构建并编译 LangGraph 状态机"""
    builder = StateGraph(ResearchState)

    # ── 注册节点 ──────────────────────────────────────────
    builder.add_node("planner", planner_node)
    builder.add_node("searcher", searcher_node)
    builder.add_node("critic", critic_node)
    builder.add_node("writer", writer_node)

    # ── 定义边 ────────────────────────────────────────────
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "searcher")
    builder.add_edge("searcher", "critic")

    # Critic → 条件路由（这是 Multi-Agent 的核心）
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "writer": "writer",
            "searcher": "searcher",
            "planner": "planner",
        },
    )

    builder.add_edge("writer", END)

    # ── 编译（加 MemorySaver 支持短期记忆）────────────────
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ── 单例，直接 import 用 ───────────────────────────────────
research_graph = build_graph()