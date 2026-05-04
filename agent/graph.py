"""
agent/graph.py
LangGraph 主图 + 可配置工厂

新增 build_graph(config) 支持消融实验：
  - "baseline":   只有 single_search + writer（无 multi-agent）
  - "no_critic":  Planner + Searcher + Writer（不带 Critic 动态路由）
  - "no_rag":     全多智能体但无 RAG（Retriever 跳过）
  - "full":       完整系统（默认，与之前行为一致）

通过 LangGraph 的 RunnableConfig 把 config 透传到节点，
节点根据 config 决定行为（不需要改原有节点代码太多）
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
from agent.nodes.retriever import retriever_node
from agent.nodes.episodic_writer import episodic_writer_node


# ═════════════════════════════════════════════════════════
# 路由函数（full 配置使用）
# ═════════════════════════════════════════════════════════
def route_after_critic(state) -> str:
    verdict = state.get("critic_verdict", "continue")
    current_step = state.get("current_step", 0)
    plan_length = len(state.get("research_plan", []))

    if verdict == "sufficient":
        return "writer"
    if verdict == "replan":
        return "planner"
    if current_step < plan_length:
        return "searcher"
    return "writer"


# 没有 Critic 的简单路由：跑完所有子问题就直接写
def route_after_searcher_no_critic(state) -> str:
    current_step = state.get("current_step", 0)
    plan_length = len(state.get("research_plan", []))
    if current_step < plan_length:
        return "searcher"
    return "writer"


# ═════════════════════════════════════════════════════════
# Baseline: 完全不用 multi-agent，只有一次性搜索 + 写报告
# ═════════════════════════════════════════════════════════
def baseline_search_node(state: ResearchState) -> dict:
    """
    Baseline: 不分解问题，直接用原 query 做 1 次 web search，结果丢给 writer
    模拟"普通 LLM + 搜索"的最朴素 pipeline
    """
    from agent.skills.search import web_search
    from agent.skills.tools import web_fetch

    query = state["query"]
    raw = web_search.invoke({"query": query, "max_results": 4})

    enriched = []
    for r in raw[:2]:
        fetched = web_fetch.invoke({"url": r["url"]})
        enriched.append({
            "sub_query": query,
            "url": r["url"],
            "title": r["title"],
            "snippet": r["snippet"],
            "content": fetched["content"] if fetched["success"] else r["snippet"],
        })
    for r in raw[2:]:
        enriched.append({
            "sub_query": query,
            "url": r["url"],
            "title": r["title"],
            "snippet": r["snippet"],
            "content": r["snippet"],
        })

    return {
        "search_results": enriched,
        "research_plan": [query],   # 让 Writer 当"单一子问题"处理
        "current_step": 1,
    }


# ═════════════════════════════════════════════════════════
# 工厂函数
# ═════════════════════════════════════════════════════════
def build_graph(config: str = "full"):
    """
    构建并编译 LangGraph 状态机。

    Args:
        config: 实验配置名
            - "baseline":  baseline_search → writer (无 multi-agent)
            - "no_critic": planner → searcher (循环) → writer (无 Critic)
            - "no_rag":    planner → searcher → critic → writer (无 RAG/EpisodicWriter)
            - "full":      全套 (默认)
    """
    builder = StateGraph(ResearchState)

    if config == "baseline":
        # ── 最简单的对照组 ──────────────────────────────────
        builder.add_node("baseline_search", baseline_search_node)
        builder.add_node("writer", writer_node)

        builder.add_edge(START, "baseline_search")
        builder.add_edge("baseline_search", "writer")
        builder.add_edge("writer", END)

    elif config == "no_critic":
        # ── 有 multi-agent 但无 Critic 动态路由 ─────────────
        builder.add_node("planner", planner_node)
        builder.add_node("searcher", searcher_node)
        builder.add_node("writer", writer_node)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "searcher")
        # Searcher 自循环直到所有子问题完成，然后进 Writer
        builder.add_conditional_edges(
            "searcher",
            route_after_searcher_no_critic,
            {"searcher": "searcher", "writer": "writer"},
        )
        builder.add_edge("writer", END)

    elif config == "no_rag":
        # ── 有 multi-agent + Critic，但无 RAG/Episodic ──────
        builder.add_node("planner", planner_node)
        builder.add_node("searcher", searcher_node)
        builder.add_node("critic", critic_node)
        builder.add_node("writer", writer_node)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "searcher")
        builder.add_edge("searcher", "critic")
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

    else:  # "full"
        # ── 完整系统 ────────────────────────────────────────
        builder.add_node("planner", planner_node)
        builder.add_node("retriever", retriever_node)
        builder.add_node("searcher", searcher_node)
        builder.add_node("critic", critic_node)
        builder.add_node("writer", writer_node)
        builder.add_node("episodic_writer", episodic_writer_node)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "retriever")
        builder.add_edge("retriever", "searcher")
        builder.add_edge("searcher", "critic")
        builder.add_conditional_edges(
            "critic",
            route_after_critic,
            {
                "writer": "writer",
                "searcher": "searcher",
                "planner": "planner",
            },
        )
        builder.add_edge("writer", "episodic_writer")
        builder.add_edge("episodic_writer", END)

    return builder.compile(checkpointer=MemorySaver())


# ── 默认单例（向后兼容，main.py 还在用）────────────────────
research_graph = build_graph(config="full")