"""
eval/metrics.py
评测指标 v2

新增 / 改进：
1. keyword_coverage 支持 all_required=True（要求全部关键词都命中才算通过）
   这避免了"中了一个通用词就算 100%"的偏差
2. 增加 partial_score（连续值，不是 0/1），更精细
3. evaluate_record 默认开启 LLM judge —— 因为 baseline 高分主要靠 LLM judge 才能识别
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re

from langchain_core.messages import HumanMessage, SystemMessage
from llm.client import default_llm


@dataclass
class ExperimentRecord:
    question_id: str
    question: str
    expected_keywords: list[str]
    difficulty: str
    topic: str
    all_required: bool = False           # 新增：是否要求全部关键词命中
    final_report: str = ""
    citations: list[dict] = None
    search_results: list[dict] = None
    retrieved_chunks: list[dict] = None
    web_search_count: int = 0
    rag_hit_count: int = 0
    iteration_count: int = 0
    latency_seconds: float = 0.0
    config_name: str = ""

    def __post_init__(self):
        self.citations = self.citations or []
        self.search_results = self.search_results or []
        self.retrieved_chunks = self.retrieved_chunks or []


# ═════════════════════════════════════════════════════════
# Metric 1: Keyword Coverage（v2 改进）
# ═════════════════════════════════════════════════════════
def keyword_coverage(record: ExperimentRecord) -> dict:
    """
    返回:
        - hit_count: 命中关键词数
        - total: 总关键词数
        - partial_score: 部分命中率（hit_count / total）
        - pass: 是否通过（all_required 模式下需要全部命中，否则只要 >0 就通过）
    """
    if not record.expected_keywords:
        return {"hit_count": 0, "total": 0, "partial_score": 1.0, "pass": True}

    report_lower = record.final_report.lower()
    hits = [kw for kw in record.expected_keywords if kw.lower() in report_lower]
    hit_count = len(hits)
    total = len(record.expected_keywords)
    partial = hit_count / total if total > 0 else 0

    if record.all_required:
        passed = hit_count == total
    else:
        # 只要命中至少一个就算 pass（但 partial_score 仍是部分命中率）
        passed = hit_count > 0

    return {
        "hit_count": hit_count,
        "total": total,
        "partial_score": partial,
        "pass": passed,
        "hits": hits,
    }


# ═════════════════════════════════════════════════════════
# Metric 2: LLM-as-judge
# ═════════════════════════════════════════════════════════
JUDGE_SYSTEM = """你是一个严格的研究报告评审专家。
给定一个研究问题和一份 AI 生成的报告，请从以下三个维度按 1-5 分评分：

1. **相关性 (Relevance)**：报告是否切题，是否回答了用户的核心问题？
2. **完整性 (Completeness)**：报告是否覆盖了问题的主要方面？是否遗漏关键信息？
3. **支撑性 (Support)**：核心论点是否有引用或具体数据支撑？

按以下严格格式输出，不要其他任何内容：
RELEVANCE: <1-5>
COMPLETENESS: <1-5>
SUPPORT: <1-5>"""


def parse_judge_score(text: str) -> dict:
    scores = {"relevance": 0, "completeness": 0, "support": 0}
    for line in text.strip().split("\n"):
        m = re.match(r"(RELEVANCE|COMPLETENESS|SUPPORT):\s*(\d)", line.strip(), re.IGNORECASE)
        if m:
            key = m.group(1).lower()
            score = int(m.group(2))
            if 1 <= score <= 5:
                scores[key] = score
    return scores


def answer_quality(record: ExperimentRecord, judge_llm=None) -> dict:
    if not record.final_report:
        return {"relevance": 0, "completeness": 0, "support": 0, "avg": 0.0}

    judge = judge_llm or default_llm
    prompt = f"""研究问题：{record.question}

待评估的报告：
{record.final_report[:3000]}

请按格式输出三个维度的评分。"""

    try:
        resp = judge.invoke([
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        scores = parse_judge_score(resp.content)
        scores["avg"] = sum([scores["relevance"], scores["completeness"], scores["support"]]) / 3
        return scores
    except Exception as e:
        print(f"⚠️  Judge LLM 调用失败: {e}")
        return {"relevance": 0, "completeness": 0, "support": 0, "avg": 0.0}


# ═════════════════════════════════════════════════════════
# Metric 3: Citation Faithfulness
# ═════════════════════════════════════════════════════════
def citation_faithfulness(record: ExperimentRecord) -> dict:
    url_pattern = r"https?://[^\s\)\]\>]+"
    cited_urls = set(re.findall(url_pattern, record.final_report))
    cited_urls = {u.rstrip(".,;:") for u in cited_urls}

    if not cited_urls:
        return {"cited_urls_in_report": 0, "sourced_count": 0, "faithfulness": 0.0}

    valid_urls = set()
    for r in record.search_results:
        if r.get("url"):
            valid_urls.add(r["url"])
    for c in record.retrieved_chunks:
        u = c.get("metadata", {}).get("url", "")
        if u:
            valid_urls.add(u)

    sourced = sum(1 for u in cited_urls if u in valid_urls)

    return {
        "cited_urls_in_report": len(cited_urls),
        "sourced_count": sourced,
        "faithfulness": sourced / len(cited_urls) if cited_urls else 0.0,
    }


# ═════════════════════════════════════════════════════════
# Metric 4: Efficiency
# ═════════════════════════════════════════════════════════
def efficiency_metrics(record: ExperimentRecord) -> dict:
    return {
        "web_search_count": record.web_search_count,
        "rag_hit_count": record.rag_hit_count,
        "rag_hit_rate": (
            record.rag_hit_count / (record.rag_hit_count + record.web_search_count)
            if (record.rag_hit_count + record.web_search_count) > 0
            else 0.0
        ),
        "iteration_count": record.iteration_count,
        "latency_seconds": record.latency_seconds,
        "report_length_chars": len(record.final_report),
        "citation_count": len(record.citations),
    }


# ═════════════════════════════════════════════════════════
# 主评估函数
# ═════════════════════════════════════════════════════════
def evaluate_record(
    record: ExperimentRecord,
    use_llm_judge: bool = True,
    judge_llm=None,
) -> dict:
    """对单条实验记录跑所有 metrics"""
    kw = keyword_coverage(record)

    result = {
        "question_id": record.question_id,
        "question": record.question,
        "difficulty": record.difficulty,
        "topic": record.topic,
        "config": record.config_name,
        # Keyword
        "keyword_partial_score": kw["partial_score"],
        "keyword_pass": kw["pass"],
        "keyword_hits": kw["hit_count"],
        "keyword_total": kw["total"],
        # 兼容旧字段
        "keyword_coverage": kw["partial_score"],
        # Efficiency
        **efficiency_metrics(record),
        # Citation
        **{f"citation_{k}": v for k, v in citation_faithfulness(record).items()},
    }

    if use_llm_judge:
        quality = answer_quality(record, judge_llm=judge_llm)
        result.update({f"quality_{k}": v for k, v in quality.items()})

    return result