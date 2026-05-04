"""
eval/runner.py
评测执行器（v2 - 支持 all_required 关键词 + 默认开 LLM judge）
"""
from __future__ import annotations
import argparse
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
import csv

from agent.graph import research_graph
from agent.state import ResearchState
from eval.metrics import ExperimentRecord, evaluate_record


TESTSET_PATH = Path(__file__).parent / "testset" / "questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def question_to_record_args(q: dict) -> dict:
    """从测试集题目提取构造 ExperimentRecord 所需字段"""
    return {
        "question_id": q["id"],
        "question": q["question"],
        "expected_keywords": q.get("expected_keywords", []),
        "difficulty": q["difficulty"],
        "topic": q["topic"],
        "all_required": q.get("all_required", False),
    }


def run_one_question(question: dict, config_name: str = "default") -> ExperimentRecord:
    initial_state = ResearchState(query=question["question"])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    accumulated: dict = {"query": question["question"]}
    start = time.time()

    try:
        for step in research_graph.stream(initial_state, config=config):
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


def run_eval(
    limit: int | None = None,
    use_llm_judge: bool = True,         # v2: 默认开
    config_name: str = "default",
    questions_filter: callable = None,
) -> list[dict]:
    with TESTSET_PATH.open("r", encoding="utf-8") as f:
        testset = json.load(f)

    questions = testset["questions"]
    if questions_filter:
        questions = [q for q in questions if questions_filter(q)]
    if limit:
        questions = questions[:limit]

    print(f"\n{'='*60}")
    print(f"🧪 评测 config={config_name}, n={len(questions)}, "
          f"judge={'on' if use_llm_judge else 'off'}")
    print(f"{'='*60}\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['id']} ({q['difficulty']}/{q['topic']})")
        print(f"  Q: {q['question'][:60]}...")

        record = run_one_question(q, config_name=config_name)
        print(f"  ⏱  {record.latency_seconds:.1f}s | web {record.web_search_count} | "
              f"rag {record.rag_hit_count} | iter {record.iteration_count}")

        evaluation = evaluate_record(record, use_llm_judge=use_llm_judge)
        evaluation["timestamp"] = datetime.now().isoformat()
        results.append(evaluation)

        # 打印评测分数
        if "quality_avg" in evaluation:
            print(f"  📊 keyword {evaluation['keyword_partial_score']:.0%} ({evaluation['keyword_hits']}/{evaluation['keyword_total']}) | "
                  f"quality {evaluation['quality_avg']:.1f}/5 | "
                  f"citation {evaluation['citation_faithfulness']:.0%}")

    return results


def save_results(results: list[dict], config_name: str) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{config_name}_{timestamp}"
    json_path = RESULTS_DIR / f"{base_name}.json"
    csv_path = RESULTS_DIR / f"{base_name}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if results:
        keys = list(results[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)

    print(f"\n💾 已保存: {json_path.name}")
    return json_path, csv_path


def print_summary(results: list[dict]) -> None:
    if not results:
        return
    n = len(results)

    def avg(key: str) -> float:
        vals = [r.get(key, 0) for r in results if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0

    pass_rate = sum(1 for r in results if r.get("keyword_pass", False)) / n

    print(f"\n{'='*60}")
    print(f"📊 评测汇总 (n={n})")
    print(f"{'='*60}")
    print(f"\n[准确性]")
    print(f"  Keyword Pass Rate:    {pass_rate:.1%}")
    print(f"  Keyword 部分覆盖:      {avg('keyword_partial_score'):.1%}")
    if "quality_avg" in results[0]:
        print(f"  Quality 综合分:        {avg('quality_avg'):.2f} / 5")
        print(f"    Relevance:           {avg('quality_relevance'):.2f}")
        print(f"    Completeness:        {avg('quality_completeness'):.2f}")
        print(f"    Support:             {avg('quality_support'):.2f}")
    print(f"\n[引用质量]")
    print(f"  Citation Faithfulness: {avg('citation_faithfulness'):.1%}")
    print(f"\n[效率]")
    print(f"  Web 搜索数:           {avg('web_search_count'):.2f}")
    print(f"  RAG 命中数:           {avg('rag_hit_count'):.2f}")
    print(f"  RAG 命中率:           {avg('rag_hit_rate'):.1%}")
    print(f"  迭代次数:             {avg('iteration_count'):.2f}")
    print(f"  平均延迟:             {avg('latency_seconds'):.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true", help="关闭 LLM judge（默认开）")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--difficulty", type=str, default=None)
    args = parser.parse_args()

    f = (lambda q: q["difficulty"] == args.difficulty) if args.difficulty else None

    results = run_eval(
        limit=args.limit,
        use_llm_judge=not args.no_judge,
        config_name=args.config,
        questions_filter=f,
    )
    save_results(results, args.config)
    print_summary(results)


if __name__ == "__main__":
    main()