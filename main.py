"""
main.py —— 入口文件
用法: python main.py "你的研究问题"

修复：累积所有节点的输出（之前只取最后一个节点导致拿不到 final_report）
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import uuid
from agent.graph import research_graph
from agent.state import ResearchState
from memory.vector_store import get_vector_store


def run_research(query: str, verbose: bool = True) -> str:
    initial_state = ResearchState(query=query)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"\n{'='*60}")
    print(f"🔍 研究问题: {query}")
    print(f"{'='*60}\n")

    store = get_vector_store()
    init_stats = store.stats()
    print(f"📦 当前记忆状态: {init_stats}\n")

    # ── 累积 state（关键修复）─────────────────────────────────
    # stream 模式每次只返回当前节点的局部更新，需要自己 merge 成完整 state
    accumulated_state: dict = {"query": query}

    for step in research_graph.stream(initial_state, config=config):
        node_name = list(step.keys())[0]
        node_output = list(step.values())[0]

        # ── merge 进累积 state ────────────────────────────────
        if isinstance(node_output, dict):
            for k, v in node_output.items():
                accumulated_state[k] = v

        # ── 打印进度 ──────────────────────────────────────────
        if verbose:
            if node_name == "planner":
                plan = node_output.get("research_plan", [])
                print(f"📋 [Planner] 研究计划:")
                for i, q in enumerate(plan, 1):
                    print(f"   {i}. {q}")

            elif node_name == "retriever":
                chunks = node_output.get("retrieved_chunks", [])
                flags = node_output.get("need_web_search", [])
                covered = sum(1 for f in flags if not f)
                print(f"\n🔮 [Retriever] RAG 命中 {len(chunks)} 个 chunks")
                print(f"   {covered}/{len(flags)} 个子问题已被 RAG 解决，"
                      f"{len(flags)-covered} 个需要 web search")

            elif node_name == "searcher":
                step_num = node_output.get("current_step", 0)
                results = node_output.get("search_results", [])
                print(f"\n🔎 [Searcher] 已完成 {step_num} 个子问题搜索")
                for r in results[-2:]:
                    title = r.get("title", "")[:40]
                    url = r.get("url", "")[:50]
                    print(f"   → {title} ({url}...)")

            elif node_name == "critic":
                verdict = node_output.get("critic_verdict", "")
                feedback = node_output.get("critic_feedback", "")
                emoji = {"sufficient": "✅", "continue": "🔄", "replan": "🔁"}.get(verdict, "❓")
                print(f"\n{emoji} [Critic] 判断: {verdict} — {feedback}")

            elif node_name == "writer":
                print(f"\n✍️  [Writer] 报告生成完成")

            elif node_name == "episodic_writer":
                pass  # 节点内部自己 print

    # ── 输出最终报告（从累积 state 里取）──────────────────────
    report = accumulated_state.get("final_report", "")
    if not report:
        print("\n❌ 生成失败：未找到 final_report")
        return ""

    print(f"\n{'='*60}")
    print("📄 最终报告")
    print(f"{'='*60}")
    print(report)

    print(f"\n{'='*60}")
    print(f"📊 本次统计:")
    print(f"   - 引用数:        {len(accumulated_state.get('citations', []))}")
    print(f"   - RAG 命中数:    {len(accumulated_state.get('retrieved_chunks', []))}")
    print(f"   - Web 搜索结果:  {len(accumulated_state.get('search_results', []))}")
    print(f"   - 迭代次数:      {accumulated_state.get('iteration_count', 0)}")

    final_stats = store.stats()
    print(f"\n📦 记忆状态对比:")
    for ns in init_stats:
        delta = final_stats[ns] - init_stats[ns]
        sign = f"+{delta}" if delta >= 0 else str(delta)
        print(f"   {ns}: {init_stats[ns]} → {final_stats[ns]} ({sign})")

    return report


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "分析2024年国产大模型的商业化进展"
    run_research(query)