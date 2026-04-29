"""
agent/nodes/agents.py
四个 Agent 节点：Planner / Searcher / Critic / Writer

LangGraph 约定：
  - 节点函数接收 state（dict-like），通过 state["key"] 或 state.key 访问字段
  - 节点函数返回 dict，只包含本节点修改的字段，LangGraph 自动 merge 到 state
"""
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import ResearchState
from agent.skills.search import web_search
from agent.skills.tools import web_fetch, format_citations
from llm.client import default_llm


# ═════════════════════════════════════════════════════════
# Planner Agent ── 把用户问题分解成 3-5 个子问题
# ═════════════════════════════════════════════════════════
PLANNER_SYSTEM = """你是一个研究规划专家。
用户给你一个研究问题，你需要将其分解为 3-5 个具体的、可独立搜索的子问题。
每个子问题应该：
- 具体且可搜索（能直接作为搜索引擎 query）
- 覆盖原问题的不同维度
- 按逻辑顺序排列

只输出子问题列表，每行一个，不加编号和前缀。"""


def planner_node(state: ResearchState) -> dict:
    """Planner: 生成 research plan"""
    query = state["query"]

    resp = default_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"研究问题：{query}"),
    ])

    sub_questions = [
        line.strip()
        for line in resp.content.strip().split("\n")
        if line.strip()
    ]

    return {
        "research_plan": sub_questions,
        "current_step": 0,
    }


# ═════════════════════════════════════════════════════════
# Searcher Agent ── 执行当前子问题的搜索 + 抓取
# ═════════════════════════════════════════════════════════
def searcher_node(state: ResearchState) -> dict:
    """Searcher: 搜索当前子问题"""
    step = state["current_step"]
    plan = state["research_plan"]
    existing_results = state.get("search_results", [])

    if step >= len(plan):
        return {}  # 所有子问题已搜索完，不更新 state

    sub_query = plan[step]

    # 1. 搜索
    raw_results = web_search.invoke({"query": sub_query, "max_results": 4})

    # 2. 抓取前 2 个结果的正文（平衡质量与速度）
    enriched = []
    for r in raw_results[:2]:
        fetched = web_fetch.invoke({"url": r["url"]})
        enriched.append({
            "sub_query": sub_query,
            "url": r["url"],
            "title": r["title"],
            "snippet": r["snippet"],
            "content": fetched["content"] if fetched["success"] else r["snippet"],
        })

    # 剩余结果只保留 snippet
    for r in raw_results[2:]:
        enriched.append({
            "sub_query": sub_query,
            "url": r["url"],
            "title": r["title"],
            "snippet": r["snippet"],
            "content": r["snippet"],
        })

    return {
        "search_results": existing_results + enriched,
        "current_step": step + 1,
    }


# ═════════════════════════════════════════════════════════
# Critic Agent ── 判断证据是否足够，决定继续/停止/重规划
# ═════════════════════════════════════════════════════════
CRITIC_SYSTEM = """你是一个研究质量评审专家。
给定原始研究问题和已收集的搜索结果，判断信息是否足够支撑一份高质量报告。

请输出以下三选一：
- sufficient：信息充足，可以生成报告
- continue：还有未搜索的子问题，继续搜索
- replan：信息质量太差，需要重新规划搜索策略

只输出上述三个词之一，然后换行写一句简短的理由（不超过50字）。"""


def critic_node(state: ResearchState) -> dict:
    """Critic: 评估证据充分性"""
    query = state["query"]
    search_results = state.get("search_results", [])
    plan = state.get("research_plan", [])
    completed = state.get("current_step", 0)
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    # 汇总最近 10 条证据摘要给 Critic（避免 token 爆炸）
    evidence_summary = "\n".join([
        f"- [{r.get('sub_query', '')}] {r.get('snippet', '')[:200]}"
        for r in search_results[-10:]
    ])

    prompt = f"""原始问题：{query}

已完成子问题：{completed}/{len(plan)}
已收集证据摘要：
{evidence_summary}

当前迭代次数：{iteration}/{max_iter}"""

    resp = default_llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=prompt),
    ])

    lines = resp.content.strip().split("\n")
    verdict = lines[0].strip().lower()
    feedback = lines[1].strip() if len(lines) > 1 else ""

    # 防止死循环：超过最大迭代次数强制输出
    if iteration >= max_iter:
        verdict = "sufficient"
        feedback = "已达最大迭代次数，强制输出"

    # 容错：如果 LLM 返回的 verdict 不在三个合法值里，默认 continue
    if verdict not in ("sufficient", "continue", "replan"):
        verdict = "continue"

    return {
        "critic_verdict": verdict,
        "critic_feedback": feedback,
        "iteration_count": iteration + 1,
    }


# ═════════════════════════════════════════════════════════
# Writer Agent ── 综合搜索结果，生成带引用的最终报告
# ═════════════════════════════════════════════════════════
WRITER_SYSTEM = """你是一个专业研究报告撰写专家。
根据提供的研究问题和收集到的证据，撰写一份结构清晰、有引用支撑的研究报告。

要求：
1. 报告结构：摘要 → 正文（按逻辑分段）→ 结论
2. 每个关键论点后面用 [来源: URL] 标注引用
3. 客观呈现，不要编造信息
4. 中文输出，正文 500-800 字"""


def writer_node(state: ResearchState) -> dict:
    """Writer: 生成最终报告"""
    query = state["query"]
    search_results = state.get("search_results", [])

    # 把搜索结果打包给 Writer（截断每条 content 防 token 爆炸）
    evidence = "\n\n".join([
        f"来源: {r.get('url', '')}\n内容: {r.get('content', '')[:500]}"
        for r in search_results
    ])

    prompt = f"""研究问题：{query}

收集到的证据：
{evidence}

请撰写研究报告。"""

    resp = default_llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=prompt),
    ])

    # 构造 citations 列表
    citations = [
        {
            "claim": r.get("sub_query", ""),
            "source_url": r.get("url", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in search_results
    ]

    citation_text = format_citations.invoke({"citations": citations})

    return {
        "final_report": resp.content + citation_text,
        "citations": citations,
    }