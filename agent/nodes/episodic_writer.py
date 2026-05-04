"""
agent/nodes/episodic_writer.py
Episodic Memory Writer 节点

职责：在 Writer 输出报告后，把"被实际引用"的事实写入 episodic namespace。
这是 long-term memory 的关键 —— 下次类似问题就能命中历史研究结果。

为什么不写入所有搜索结果？
答：搜索结果包含大量噪声（广告、不相关页面）。只把 Writer 引用过的内容回写，
   相当于一个 self-curation 机制，保证 episodic memory 的信噪比。

为什么 ingest 而不是直接 add？
答：复用 chunking 逻辑，让 episodic memory 也支持 parent-child retrieval。
"""
from datetime import datetime, timezone

from agent.state import ResearchState
from memory.retrieval import ingest_text
from memory.vector_store import Namespace


def episodic_writer_node(state: ResearchState) -> dict:
    """
    Writer 输出后调用，把 citations 对应的内容回写到 episodic memory。
    返回空 dict（不修改 state）。
    """
    citations = state.get("citations", [])
    search_results = state.get("search_results", [])
    query = state.get("query", "")

    if not citations or not search_results:
        return {}

    # 1. 收集被引用过的 url 集合
    cited_urls = {c.get("source_url", "") for c in citations if c.get("source_url")}

    # 2. 从 search_results 里挑出对应的全文内容
    written_count = 0
    timestamp = datetime.now(timezone.utc).isoformat()

    for r in search_results:
        url = r.get("url", "")
        content = r.get("content", "")

        if url not in cited_urls or not content:
            continue
        if len(content) < 100:
            # 太短不值得入库（snippet 级别）
            continue

        # 3. ingest 到 episodic namespace，metadata 标记本次研究的上下文
        ingest_text(
            text=content,
            namespace=Namespace.EPISODIC,
            source_metadata={
                "url": url,
                "title": r.get("title", ""),
                "sub_query": r.get("sub_query", ""),
                "research_query": query,             # 标记是哪次研究产生的
                "timestamp": timestamp,
                "verified": "true",                  # 已被 Writer 引用过
            },
        )
        written_count += 1

    if written_count > 0:
        print(f"💾 [EpisodicWriter] 写入 {written_count} 条已验证事实到长期记忆")

    return {}   # 不修改 state，写入是副作用