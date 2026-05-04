"""
agent/nodes/agents.py
四个原始 Agent 节点：Planner / Searcher / Critic / Writer

Stage 3 修改：
- Searcher: 跳过 need_web_search=False 的子问题，并把抓取内容 ingest 到 web_cache
- Writer: 同时使用 search_results 和 retrieved_chunks 作为证据
"""
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import ResearchState
from agent.skills.search import web_search
from agent.skills.tools import web_fetch, format_citations
from llm.client import default_llm
from memory.retrieval import ingest_text
from memory.vector_store import Namespace


# ═════════════════════════════════════════════════════════
# Planner Agent
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
# Searcher Agent (Stage 3 升级版)
# ═════════════════════════════════════════════════════════
def searcher_node(state: ResearchState) -> dict:
    """
    Searcher: 搜索当前子问题。
    Stage 3 改动：
    1. 检查 need_web_search[step]，如果是 False（RAG 已足够），跳过
    2. 抓到的内容 ingest 到 web_cache namespace（同会话内可复用）
    """
    step = state["current_step"]
    plan = state["research_plan"]
    existing_results = state.get("search_results", [])
    need_web = state.get("need_web_search", [True] * len(plan))

    if step >= len(plan):
        return {}

    sub_query = plan[step]

    # ── Stage 3 新增：如果 RAG 已足够，跳过 web search ─────────
    if step < len(need_web) and not need_web[step]:
        print(f"⏭️  [Searcher] 子问题 {step+1} 已被 RAG 解决，跳过 web search")
        return {"current_step": step + 1}

    # ── 1. 搜索 ───────────────────────────────────────────────
    raw_results = web_search.invoke({"query": sub_query, "max_results": 4})

    # ── 2. 抓取前 2 个结果的正文 ──────────────────────────────
    enriched = []
    for r in raw_results[:2]:
        fetched = web_fetch.invoke({"url": r["url"]})
        content = fetched["content"] if fetched["success"] else r["snippet"]

        enriched.append({
            "sub_query": sub_query,
            "url": r["url"],
            "title": r["title"],
            "snippet": r["snippet"],
            "content": content,
        })

        # ── Stage 3 新增：ingest 到 web_cache ─────────────────
        if fetched["success"] and len(content) > 200:
            try:
                ingest_text(
                    text=content,
                    namespace=Namespace.WEB_CACHE,
                    source_metadata={
                        "url": r["url"],
                        "title": r["title"],
                        "sub_query": sub_query,
                    },
                )
            except Exception as e:
                # ingest 失败不影响主流程
                print(f"⚠️  [Searcher] web_cache ingest 失败: {e}")

    # ── 3. 剩余结果只保留 snippet ─────────────────────────────
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
# Critic Agent (保持不变)
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
    retrieved_chunks = state.get("retrieved_chunks", [])
    plan = state.get("research_plan", [])
    completed = state.get("current_step", 0)
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    # Stage 3：把 RAG 命中也算进证据
    evidence_lines = []
    for r in search_results[-8:]:
        evidence_lines.append(f"- [web|{r.get('sub_query', '')}] {r.get('snippet', '')[:200]}")
    for c in retrieved_chunks[-4:]:
        evidence_lines.append(f"- [{c['namespace']}|{c.get('sub_query', '')}] {c.get('child_text', '')[:200]}")
    evidence_summary = "\n".join(evidence_lines)

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

    if iteration >= max_iter:
        verdict = "sufficient"
        feedback = "已达最大迭代次数，强制输出"

    if verdict not in ("sufficient", "continue", "replan"):
        verdict = "continue"

    return {
        "critic_verdict": verdict,
        "critic_feedback": feedback,
        "iteration_count": iteration + 1,
    }


# ═════════════════════════════════════════════════════════
# Writer Agent (Stage 3 升级版：合并 RAG + web 证据)
# ═════════════════════════════════════════════════════════
WRITER_SYSTEM = """你是一个专业研究报告撰写专家。
根据提供的研究问题和收集到的证据（包括来自历史记忆和网络搜索），撰写一份结构清晰、有引用支撑的研究报告。

要求：
1. 报告结构：摘要 → 正文（按逻辑分段）→ 结论
2. 每个关键论点后面用 [来源: URL] 标注引用
3. 客观呈现，不要编造信息
4. 中文输出，正文 500-800 字"""


def writer_node(state: ResearchState) -> dict:
    """Writer: 生成最终报告（合并 RAG + web 两路证据）"""
    query = state["query"]
    search_results = state.get("search_results", [])
    retrieved_chunks = state.get("retrieved_chunks", [])

    evidence_parts = []

    # web 搜索证据
    for r in search_results:
        evidence_parts.append(
            f"[网络搜索] 来源: {r.get('url', '')}\n内容: {r.get('content', '')[:500]}"
        )

    # RAG 证据
    for c in retrieved_chunks:
        url = c.get("metadata", {}).get("url", "RAG-内部记忆")
        evidence_parts.append(
            f"[历史记忆 namespace={c['namespace']}] 来源: {url}\n内容: {c.get('parent_text', '')[:500]}"
        )

    evidence = "\n\n".join(evidence_parts)

    prompt = f"""研究问题：{query}

收集到的证据：
{evidence}

请撰写研究报告。"""

    resp = default_llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=prompt),
    ])

    # 构造 citations
    citations = []
    for r in search_results:
        citations.append({
            "claim": r.get("sub_query", ""),
            "source_url": r.get("url", ""),
            "snippet": r.get("snippet", ""),
        })
    for c in retrieved_chunks:
        url = c.get("metadata", {}).get("url", "")
        if url:
            citations.append({
                "claim": c.get("sub_query", ""),
                "source_url": url,
                "snippet": c.get("child_text", "")[:200],
            })

    citation_text = format_citations.invoke({"citations": citations})

    return {
        "final_report": resp.content + citation_text,
        "citations": citations,
    }