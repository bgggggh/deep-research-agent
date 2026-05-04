"""
agent/nodes/retriever.py
Retriever Agent 节点 ── Stage 3 修复版

修复：用启发式规则代替 LLM 做 sufficiency check
原因：
  1. 小模型（Qwen 14B）在 sufficiency 判断上保守倾向严重，几乎全判 insufficient
  2. 每个子问题多一次 LLM 调用 = 5 个子问题 × 10秒 = 多 50 秒延迟
  3. 启发式规则 + 分数阈值已经足够好

启发式规则：
  - 至少有 N 个高质量 chunk 命中（score > 阈值）
  - 命中分数最高那条达到 strong_threshold
  这两个条件之一满足就算 sufficient
"""
from agent.state import ResearchState
from memory.retrieval import HybridRetriever
from memory.vector_store import Namespace


# ═════════════════════════════════════════════════════════
# 充分性判断阈值（基于 RRF 分数）
# ═════════════════════════════════════════════════════════
# RRF 分数特征：
#   - 单路 rank=0 → 1/(60+0) ≈ 0.0167
#   - 两路 rank=0 → 0.0167×2 = 0.0333（强命中）
#   - 单路 rank=2 → 1/(60+2) ≈ 0.0161
STRONG_HIT_SCORE = 0.045     # 两路都靠前 = 强命中
WEAK_HIT_SCORE = 0.015       # 单路命中
MIN_GOOD_HITS = 3           # 至少要这么多个 chunk 才算 sufficient


def _is_sufficient(results: list) -> bool:
    """
    启发式：判断 RAG 检索结果是否足够回答某个子问题
    """
    if not results:
        return False

    # 规则 1：有强命中（两路都 rank 靠前）
    if any(r.score >= STRONG_HIT_SCORE for r in results):
        # 强命中 + 至少有 2 条相关结果 = 充分
        good_count = sum(1 for r in results if r.score >= WEAK_HIT_SCORE)
        if good_count >= MIN_GOOD_HITS:
            return True

    return False


# ═════════════════════════════════════════════════════════
# Retriever 单例（避免重复加载 BGE-M3）
# ═════════════════════════════════════════════════════════
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ═════════════════════════════════════════════════════════
# Retriever Node 主逻辑
# ═════════════════════════════════════════════════════════
def retriever_node(state: ResearchState) -> dict:
    """
    Retriever: 对每个子问题查 RAG，标记是否需要 web search

    输入: state["research_plan"]
    输出:
        - retrieved_chunks: 跨子问题汇总的命中片段
        - need_web_search: 与 plan 等长的 bool 列表
    """
    plan = state.get("research_plan", [])
    if not plan:
        return {}

    retriever = _get_retriever()

    all_chunks = []
    web_search_flags = []

    for sub_query in plan:
        results = retriever.retrieve(
            query=sub_query,
            namespaces=[Namespace.EPISODIC, Namespace.KNOWLEDGE, Namespace.WEB_CACHE],
            top_k=3,
        )

        # 收集命中的 chunks（无论是否充分，给 Writer 也是有用的素材）
        for r in results:
            all_chunks.append({
                "sub_query": sub_query,
                "parent_text": r.parent_text,
                "child_text": r.child_text,
                "namespace": r.namespace,
                "score": r.score,
                "metadata": r.metadata,
            })

        # 启发式判断
        sufficient = _is_sufficient(results)
        web_search_flags.append(not sufficient)

    return {
        "retrieved_chunks": all_chunks,
        "need_web_search": web_search_flags,
    }