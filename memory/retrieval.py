"""
memory/retrieval.py
混合检索：BM25 + 向量检索 + RRF 重排 + Parent-Child 上下文还原

设计要点：
1. 两路并行检索（语义路 + 关键词路），覆盖不同查询模式
2. RRF 融合，无需人工调权重
3. Child 命中后还原成 Parent，给 LLM 完整上下文
4. 支持 namespace 过滤、metadata 过滤
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
import re

from rank_bm25 import BM25Okapi

from memory.vector_store import VectorStore, Namespace, get_vector_store
from memory.chunking import (
    ParentChunk, ChildChunk, ChunkingResult,
    ParentChildChunker, ParentStore, get_parent_store,
)


# ═════════════════════════════════════════════════════════
# 检索结果数据结构
# ═════════════════════════════════════════════════════════
@dataclass
class RetrievalResult:
    """单条检索结果"""
    parent_id: str
    parent_text: str          # 完整大块，给 LLM 用
    child_text: str           # 命中的小块，用于解释
    metadata: dict
    score: float              # RRF 融合后的分数（越大越相关）
    namespace: str            # 来自哪个 namespace（可解释性）


# ═════════════════════════════════════════════════════════
# 简易中文分词（BM25 需要 token 列表）
# ═════════════════════════════════════════════════════════
def simple_tokenize(text: str) -> list[str]:
    """
    简化版中英文分词：
    - 英文按空格 + 标点切
    - 中文按单字切（粗暴但对短查询足够）

    生产环境可换 jieba：pip install jieba
    然后：return list(jieba.cut(text))
    """
    # 把英文连续字符提取出来
    eng_tokens = re.findall(r"[a-zA-Z0-9]+", text)
    # 中文逐字切
    chn_tokens = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    return [t.lower() for t in eng_tokens + chn_tokens]


# ═════════════════════════════════════════════════════════
# BM25 索引（in-memory，每次重新构建）
# ═════════════════════════════════════════════════════════
class BM25Index:
    """
    BM25 索引器。
    简化设计：每个 namespace 一个内存索引，初次查询时从 ChromaDB 拉全量数据构建。
    生产环境应该用 Elasticsearch 或 OpenSearch 做持久化倒排索引。
    """

    def __init__(self):
        # {namespace_value: {"bm25": BM25Okapi, "ids": [...], "texts": [...], "metadatas": [...]}}
        self._cache: dict[str, dict] = {}

    def build(self, namespace: Namespace, store: VectorStore) -> None:
        """
        从 ChromaDB 拉全量数据构建 BM25 索引。
        简单实现：每次查询前重建（小数据量场景没问题）。
        """
        coll = store.collections[namespace]
        if coll.count() == 0:
            self._cache[namespace.value] = None
            return

        # 拉所有数据
        result = coll.get()
        ids = result["ids"]
        texts = result["documents"]
        metadatas = result["metadatas"]

        tokenized = [simple_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)

        self._cache[namespace.value] = {
            "bm25": bm25,
            "ids": ids,
            "texts": texts,
            "metadatas": metadatas,
        }

    def query(
        self,
        namespace: Namespace,
        query_text: str,
        k: int = 20,
    ) -> list[dict]:
        """
        BM25 查询。返回格式与 vector_store.query 一致。
        """
        idx = self._cache.get(namespace.value)
        if idx is None:
            return []

        query_tokens = simple_tokenize(query_text)
        if not query_tokens:
            return []

        scores = idx["bm25"].get_scores(query_tokens)

        # 取 top-k
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

        return [
            {
                "id": idx["ids"][i],
                "text": idx["texts"][i],
                "metadata": idx["metadatas"][i],
                "bm25_score": scores[i],
            }
            for i in top_indices
            if scores[i] > 0    # BM25 分 0 = 完全不相关，过滤掉
        ]


# ═════════════════════════════════════════════════════════
# RRF 融合
# ═════════════════════════════════════════════════════════
def rrf_fuse(
    rankings: list[list[dict]],
    k: int = 60,
    id_field: str = "id",
) -> list[tuple[str, float, dict]]:
    """
    Reciprocal Rank Fusion.

    Args:
        rankings: 多路检索的结果列表（每路是一个 dict 列表）
        k: RRF 的平滑常数，论文推荐 60
        id_field: 文档唯一标识字段名

    Returns:
        [(doc_id, fused_score, original_doc), ...] 按分数降序
    """
    # 累加每个文档在各路检索中的 RRF 分数
    score_map: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc[id_field]
            score_map[doc_id] += 1.0 / (k + rank)
            # 任何一路出现过都记一下原始 doc（用于后续取 metadata）
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    # 按融合分数降序
    sorted_ids = sorted(score_map.items(), key=lambda x: -x[1])
    return [(doc_id, score, doc_map[doc_id]) for doc_id, score in sorted_ids]


# ═════════════════════════════════════════════════════════
# 主检索器
# ═════════════════════════════════════════════════════════
class HybridRetriever:
    """
    混合检索器主接口。
    流程: query → 双路检索 → RRF 融合 → child→parent 还原 → 返回
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        parent_store: Optional[ParentStore] = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.parent_store = parent_store or get_parent_store()
        self.bm25 = BM25Index()

    def retrieve(
        self,
        query: str,
        namespaces: Optional[list[Namespace]] = None,
        top_k: int = 5,
        candidate_pool: int = 20,
    ) -> list[RetrievalResult]:
        """
        混合检索主入口。

        Args:
            query: 查询文本
            namespaces: 在哪些 namespace 检索（默认全部）
            top_k: 最终返回多少条
            candidate_pool: 每路检索的候选池大小（影响召回率）

        Returns:
            按 RRF 分数降序排列的 RetrievalResult 列表
        """
        if namespaces is None:
            namespaces = list(Namespace)

        # 跨 namespace 收集所有候选
        all_results: list[RetrievalResult] = []

        for ns in namespaces:
            if self.vector_store.count(ns) == 0:
                continue

            # 1. 向量检索
            vec_hits = self.vector_store.query(ns, query, k=candidate_pool)

            # 2. BM25 检索
            self.bm25.build(ns, self.vector_store)
            bm25_hits = self.bm25.query(ns, query, k=candidate_pool)

            # 3. RRF 融合
            fused = rrf_fuse([vec_hits, bm25_hits])

            # 4. 取 top-k 并还原 parent
            for doc_id, score, doc in fused[:top_k]:
                metadata = doc["metadata"]
                parent_id = metadata.get("parent_id")

                if parent_id is None:
                    # 没有 parent 关联（比如直接存的整段），用 child 文本
                    parent_text = doc["text"]
                else:
                    parent = self.parent_store.get(parent_id)
                    parent_text = parent.text if parent else doc["text"]

                all_results.append(
                    RetrievalResult(
                        parent_id=parent_id or doc_id,
                        parent_text=parent_text,
                        child_text=doc["text"],
                        metadata=metadata,
                        score=score,
                        namespace=ns.value,
                    )
                )

        # 跨 namespace 二次排序，去重
        all_results.sort(key=lambda r: -r.score)
        deduped = self._dedupe_by_parent(all_results)
        return deduped[:top_k]

    @staticmethod
    def _dedupe_by_parent(results: list[RetrievalResult]) -> list[RetrievalResult]:
        """同一个 parent 只保留最高分的那条（避免重复上下文）"""
        seen_parents: set[str] = set()
        deduped = []
        for r in results:
            if r.parent_id in seen_parents:
                continue
            seen_parents.add(r.parent_id)
            deduped.append(r)
        return deduped


# ═════════════════════════════════════════════════════════
# 高层便捷函数：一次性 ingest 文本
# ═════════════════════════════════════════════════════════
def ingest_text(
    text: str,
    namespace: Namespace,
    source_metadata: Optional[dict] = None,
    chunker: Optional[ParentChildChunker] = None,
    vector_store: Optional[VectorStore] = None,
    parent_store: Optional[ParentStore] = None,
) -> int:
    """
    把一段文本 chunking 后，children 写入向量库，parents 写入 KV store。
    返回写入的 child 数量。
    """
    chunker = chunker or ParentChildChunker()
    vector_store = vector_store or get_vector_store()
    parent_store = parent_store or get_parent_store()

    result = chunker.split(text, source_metadata=source_metadata or {})

    # 写 parents
    parent_store.add(result.parents)

    # 写 children 到向量库
    if result.children:
        vector_store.add(
            namespace,
            texts=[c.text for c in result.children],
            metadatas=[c.metadata for c in result.children],
            ids=[c.id for c in result.children],
        )

    return len(result.children)


# ═════════════════════════════════════════════════════════
# 自测：python -m memory.retrieval
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== 准备数据 ===")

    # 几个测试文档
    docs = [
        {
            "text": "Llama 3.1 70B 是 Meta 在 2024 年发布的开源大模型，"
                    "上下文长度高达 128K tokens，在多项基准测试中接近 GPT-4 水平。"
                    "它支持多语言，包括英文、中文、西班牙语等。",
            "metadata": {"source": "model_card", "model": "llama-3.1"},
        },
        {
            "text": "DeepSeek V3 是 2024 年底发布的 MoE 模型，总参数 671B，"
                    "每次推理仅激活 37B 参数。其设计理念是用稀疏激活换取推理效率，"
                    "推理成本比 Llama 70B 还低。",
            "metadata": {"source": "tech_report", "model": "deepseek-v3"},
        },
        {
            "text": "Claude 3.5 Sonnet 由 Anthropic 开发，在 HumanEval 编程基准上"
                    "达到了 92% 的 pass@1，超越 GPT-4o。它特别擅长长文本理解和"
                    "复杂推理任务，但单次调用价格略高。",
            "metadata": {"source": "blog_post", "model": "claude-3.5-sonnet"},
        },
        {
            "text": "国产大模型在 2024 年实现了显著的商业化突破。"
                    "金融、教育、医疗等垂直行业开始大规模采用大模型，"
                    "API 调用单价普遍降至每百万 token 几元人民币。",
            "metadata": {"source": "industry_report", "topic": "commercialization"},
        },
    ]

    # 清空旧数据
    store = get_vector_store()
    store.clear(Namespace.KNOWLEDGE)
    parent_store = get_parent_store()

    # Ingest
    for doc in docs:
        n = ingest_text(
            doc["text"],
            namespace=Namespace.KNOWLEDGE,
            source_metadata=doc["metadata"],
        )
        print(f"  写入 {n} 个 child")

    print(f"\n向量库统计: {store.stats()}")
    print(f"ParentStore 数量: {parent_store.count()}")

    # ── 测试查询 ───────────────────────────────────────────
    retriever = HybridRetriever()

    test_queries = [
        "Llama 上下文长度",            # 测精确匹配
        "中国大模型变现",              # 测同义词改写（vs 文档里"商业化"）
        "MoE 稀疏激活",               # 测专有名词
        "编程能力最强的模型",          # 测语义理解
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 查询: {query}")
        print('='*60)
        results = retriever.retrieve(query, top_k=2)

        if not results:
            print("  (无结果)")
            continue

        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] score={r.score:.4f} | namespace={r.namespace}")
            print(f"      命中片段: {r.child_text[:80]}...")
            print(f"      metadata: {r.metadata.get('source', 'N/A')}")                             