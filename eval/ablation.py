"""
eval/ablation.py
消融实验 v3 ── 支持按难度过滤，验证 'Multi-Agent 在复杂任务上更优' 的假设
"""
from __future__ import annotations
import argparse
import json
import time
import uuid
from pathlib import Path
from datetime import datetime

from agent.graph import build_graph
from agent.state import ResearchState
from eval.metrics import ExperimentRecord, evaluate_record
from eval.runner import (
    TESTSET_PATH, RESULTS_DIR, save_results,
    question_to_record_args,
)


CONFIGS = [
    {"name": "baseline",   "graph_config": "baseline",   "description": "单 Agent + 一次搜索"},
    {"name": "no_critic",  "graph_config": "no_critic",  "description": "Multi-Agent 但无 Critic"},
    {"name": "no_rag",     "graph_config": "no_rag",     "description": "Multi-Agent 全套但无 RAG"},
    {"name": "full_cold",  "graph_config": "full",       "description": "完整系统（cold start）"},
]


def reset_all_state() -> None:
    from memory import vector_store as vs_module
    if vs_module._default_store is not None:
        try:
            for ns in vs_module.Namespace:
                vs_module._default_store.clear(ns)
        except Exception as e:
            print(f"⚠️  清空 namespace 失败（可忽略）: {e}")
    from memory import chunking as ck_module
    ck_module._default_parent_store = None
    from agent.nodes import retriever as rt_module
    rt_module._retriever = None
    print("🧹 已重置 VectorStore / ParentStore / Retriever")


def run_one_with_graph(question: dict, graph, config_name: str) -> ExperimentRecord:
    initial_state = ResearchState(query=question["question"])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    accumulated: dict = {"query": question["question"]}
    start = time.time()
    try:
        for step in graph.stream(initial_state, config=config):
            node_output = list(step.values())[0]
            if isinstance(node_output, dict):
                for k, v in node_output.items():
                    accumulated[k] = v
    except Exception as e:
        print(f"❌ {question['id']} 异常: {e}")
    latency = time.time() - start

    sub_queries_done = set()
    for r in accumulated.get("search_results", []):
        if r.get("sub_query"):
            sub_queries_done.add(r["sub_query"])

    return ExperimentRecord(
        **question_to_record_args(question),
        final_report=accumulated.get("final_report", ""),
        citations=accumulated.get("citations", []),
        search_results=accumulated.get("search_results", []),
        retrieved_chunks=accumulated.get("retrieved_chunks", []),
        web_search_count=len(sub_queries_done),
        rag_hit_count=len(accumulated.get("retrieved_chunks", [])),
        iteration_count=accumulated.get("iteration_count", 0),
        latency_seconds=latency,
        config_name=config_name,
    )


def run_ablation(
    limit: int | None = None,
    use_llm_judge: bool = True,
    difficulty: str | None = None,    # ⭐ 新增
) -> dict:
    with TESTSET_PATH.open("r", encoding="utf-8") as f:
        testset = json.load(f)
    questions = testset["questions"]

    # 难度过滤（核心新增）
    if difficulty:
        questions = [q for q in questions if q["difficulty"] == difficulty]

    if limit:
        questions = questions[:limit]

    n_q, n_c = len(questions), len(CONFIGS)
    print(f"\n{'='*60}")
    diff_label = f" [{difficulty}]" if difficulty else ""
    print(f"🧪 消融实验{diff_label}: {n_c} 配置 × {n_q} 题")
    print(f"{'='*60}")
    for c in CONFIGS:
        print(f"  - {c['name']}: {c['description']}")

    all_results = {}
    for cfg in CONFIGS:
        config_name = cfg["name"]
        print(f"\n\n{'#'*60}")
        print(f"# 配置: {config_name}")
        print(f"{'#'*60}")

        reset_all_state()
        graph = build_graph(config=cfg["graph_config"])

        results = []
        for i, q in enumerate(questions, 1):
            print(f"\n[{config_name} {i}/{n_q}] {q['id']} ({q['difficulty']}/{q['topic']})")
            print(f"  Q: {q['question'][:60]}...")

            record = run_one_with_graph(q, graph, config_name=config_name)
            print(f"  ⏱  {record.latency_seconds:.1f}s | web {record.web_search_count} | "
                  f"rag {record.rag_hit_count} | iter {record.iteration_count}")

            evaluation = evaluate_record(record, use_llm_judge=use_llm_judge)
            evaluation["timestamp"] = datetime.now().isoformat()
            results.append(evaluation)
            if "quality_avg" in evaluation:
                print(f"  📊 kw {evaluation['keyword_partial_score']:.0%} | "
                      f"quality {evaluation['quality_avg']:.1f}/5 | "
                      f"cite {evaluation['citation_faithfulness']:.0%}")

        all_results[config_name] = results
        save_results(results, f"{config_name}_{difficulty or 'all'}")

    return all_results


def print_comparison_table(all_results: dict, label: str = "") -> None:
    title = f"📊 消融实验对比表 {label}".strip()
    print(f"\n\n{'='*92}")
    print(title)
    print(f"{'='*92}\n")

    metrics_to_compare = [
        ("quality_avg",            "Quality 综合分（1-5）"),
        ("quality_relevance",      "  - Relevance"),
        ("quality_completeness",   "  - Completeness"),
        ("quality_support",        "  - Support"),
        ("keyword_partial_score",  "Keyword 部分覆盖"),
        ("citation_faithfulness",  "引用忠实度"),
        ("rag_hit_rate",           "RAG 命中率"),
        ("web_search_count",       "Web 搜索次数"),
        ("latency_seconds",        "延迟（秒）"),
        ("iteration_count",        "迭代次数"),
    ]

    config_names = list(all_results.keys())
    header = f"{'指标':<24}"
    for name in config_names:
        header += f"{name:>15}"
    print(header)
    print("-" * len(header))

    for metric_key, metric_label in metrics_to_compare:
        row = f"{metric_label:<24}"
        for cfg_name in config_names:
            results = all_results[cfg_name]
            vals = [r.get(metric_key, 0) for r in results
                    if isinstance(r.get(metric_key), (int, float))]
            avg = sum(vals) / len(vals) if vals else 0
            if "rate" in metric_key or "score" in metric_key or "faithfulness" in metric_key:
                cell = f"{avg:.1%}"
            elif "latency" in metric_key:
                cell = f"{avg:.1f}s"
            else:
                cell = f"{avg:.2f}"
            row += f"{cell:>15}"
        print(row)

    # Pass rate
    print()
    pass_row = f"{'Keyword Pass Rate':<24}"
    for cfg_name in config_names:
        results = all_results[cfg_name]
        passed = sum(1 for r in results if r.get("keyword_pass", False))
        rate = passed / len(results) if results else 0
        pass_row += f"{rate:.1%}".rjust(15)
    print(pass_row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=[None, "easy", "medium", "hard"],
                        help="只跑某个难度的题")
    args = parser.parse_args()

    all_results = run_ablation(
        limit=args.limit,
        use_llm_judge=not args.no_judge,
        difficulty=args.difficulty,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.difficulty or "all"
    combined_path = RESULTS_DIR / f"ablation_{suffix}_{timestamp}.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 合并: {combined_path.name}")

    label = f"({args.difficulty})" if args.difficulty else ""
    print_comparison_table(all_results, label=label)


if __name__ == "__main__":
    main()