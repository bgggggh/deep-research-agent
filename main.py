"""
main.py —— 入口文件
用法: python main.py "你的研究问题"
"""
from dotenv import load_dotenv
load_dotenv()  # 自动读取 .env 里的 GOOGLE_API_KEY 等

import sys
import uuid
from agent.graph import research_graph
from agent.state import ResearchState


def run_research(query: str, verbose: bool = True) -> str:
    """
    运行完整的 Deep Research Agent。
    返回最终报告字符串。
    """
    initial_state = ResearchState(query=query)

    # thread_id 用于区分不同会话（支持多轮对话记忆）
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"\n{'='*60}")
    print(f"🔍 研究问题: {query}")
    print(f"{'='*60}\n")

    # ── stream 模式：实时看到每个节点的输出 ──────────────────
    final_state = None
    for step in research_graph.stream(initial_state, config=config):
        node_name = list(step.keys())[0]
        node_output = list(step.values())[0]   # dict, 注意不是对象

        if verbose:
            if node_name == "planner":
                plan = node_output.get("research_plan", [])
                print(f"📋 [Planner] 研究计划:")
                for i, q in enumerate(plan, 1):
                    print(f"   {i}. {q}")

            elif node_name == "searcher":
                step_num = node_output.get("current_step", 0)
                results = node_output.get("search_results", [])
                # 这里 plan 长度从 final_state 累积里取不到，写一个简化版
                print(f"\n🔎 [Searcher] 已完成 {step_num} 个子问题搜索")
                # 显示最新 2 条结果
                for r in results[-2:]:
                    title = r.get("title", "")[:40]
                    url = r.get("url", "")[:50]
                    print(f"   → {title} ({url}...)")

            elif node_name == "critic":
                verdict = node_output.get("critic_verdict", "")
                feedback = node_output.get("critic_feedback", "")
                emoji = {
                    "sufficient": "✅",
                    "continue": "🔄",
                    "replan": "🔁",
                }.get(verdict, "❓")
                print(f"\n{emoji} [Critic] 判断: {verdict} — {feedback}")

            elif node_name == "writer":
                print(f"\n✍️  [Writer] 报告生成完成\n")

        final_state = node_output

    # ── 输出最终报告 ─────────────────────────────────────────
    if final_state is None:
        print("❌ 生成失败")
        return ""

    report = final_state.get("final_report", "生成失败")
    print(f"\n{'='*60}")
    print("📄 最终报告")
    print(f"{'='*60}")
    print(report)
    print(f"\n{'='*60}")
    print(f"📊 统计: 搜索结果 {len(final_state.get('search_results', []))} 条 | "
          f"迭代 {final_state.get('iteration_count', 0)} 次 | "
          f"引用 {len(final_state.get('citations', []))} 个")

    return report


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "分析2024年国产大模型的商业化进展"
    run_research(query)