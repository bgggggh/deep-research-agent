"""
memory/vector_store.py
基于 ChromaDB 的三 namespace 向量库封装。

设计要点：
1. 一个 ChromaDB persistent client，三个 collection 对应三种数据生命周期
2. 统一用 BGE-M3 做 embedding（中文最强开源之一，本地跑免费）
3. 提供 add / query / delete / count 等基础操作，下游 Retriever 在此之上做混合检索
"""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional
import uuid

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# ═════════════════════════════════════════════════════════
# Namespace 定义
# ═════════════════════════════════════════════════════════
class Namespace(str, Enum):
    """三种数据生命周期"""
    KNOWLEDGE = "knowledge"      # 用户预灌的领域知识，永久
    EPISODIC = "episodic"        # Agent 验证过的事实，永久（可加 TTL）
    WEB_CACHE = "web_cache"      # 临时抓取的网页正文，单会话


# ═════════════════════════════════════════════════════════
# 单例向量库管理器
# ═════════════════════════════════════════════════════════
class VectorStore:
    """
    ChromaDB 三 namespace 封装。
    用法：
        store = VectorStore()
        store.add(Namespace.KNOWLEDGE, texts=[...], metadatas=[...])
        results = store.query(Namespace.KNOWLEDGE, "查询文本", k=5)
    """

    # 类级缓存的 embedding model（避免每次实例化都加载）
    _embedding_fn = None

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        # 持久化客户端：数据写到磁盘，进程重启不丢
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # 懒加载 embedding model（启动慢，第一次用时才加载）
        if VectorStore._embedding_fn is None:
            VectorStore._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-m3",
                device="cpu",   # M 系列 Mac 改成 "mps" 加速
            )

        # 三个 collection 一次性创建好
        self.collections = {
            ns: self.client.get_or_create_collection(
                name=ns.value,
                embedding_function=VectorStore._embedding_fn,
                metadata={"hnsw:space": "cosine"},   # 余弦相似度
            )
            for ns in Namespace
        }

    # ── 写入 ──────────────────────────────────────────────
    def add(
        self,
        namespace: Namespace,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        添加文档到指定 namespace。
        返回插入的 ids（如果未指定，自动生成 uuid）。
        """
        if not texts:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # ChromaDB 要求 metadata 不能为空 dict，加一个占位字段
        for m in metadatas:
            if not m:
                m["_placeholder"] = ""

        self.collections[namespace].add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        return ids

    # ── 查询 ──────────────────────────────────────────────
    def query(
        self,
        namespace: Namespace,
        query_text: str,
        k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        在指定 namespace 内做向量检索。
        返回格式: [{"id": ..., "text": ..., "metadata": ..., "distance": ...}, ...]

        where 参数支持 metadata 过滤，例如:
            where={"source": "academic_paper"}
        """
        coll = self.collections[namespace]

        # ChromaDB 要求 n_results <= count
        n = min(k, coll.count())
        if n == 0:
            return []

        result = coll.query(
            query_texts=[query_text],
            n_results=n,
            where=where,
        )

        # ChromaDB 返回的是 list of list（支持 batch query），我们只查一个
        return [
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],   # 越小越相似
            }
            for i in range(len(result["ids"][0]))
        ]

    # ── 工具方法 ──────────────────────────────────────────
    def count(self, namespace: Namespace) -> int:
        """返回某 namespace 内的文档总数"""
        return self.collections[namespace].count()

    def clear(self, namespace: Namespace) -> None:
        """清空某 namespace（开发调试用）"""
        self.client.delete_collection(name=namespace.value)
        self.collections[namespace] = self.client.get_or_create_collection(
            name=namespace.value,
            embedding_function=VectorStore._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def stats(self) -> dict:
        """返回所有 namespace 的统计信息"""
        return {ns.value: self.count(ns) for ns in Namespace}


# ── 单例（全局共享一个实例）──────────────────────────────
_default_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """获取全局单例"""
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store


# ═════════════════════════════════════════════════════════
# 自测：python -m memory.vector_store
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    store = get_vector_store()
    print("初始化 VectorStore 成功")
    print(f"当前各 namespace 文档数: {store.stats()}")

    # 写入测试
    print("\n=== 写入测试 ===")
    ids = store.add(
        Namespace.KNOWLEDGE,
        texts=[
            "Llama 3.1 70B 的上下文长度是 128K tokens",
            "DeepSeek V3 是中国开源的 MoE 模型",
            "Anthropic 的 Claude 4.5 在编程任务上表现突出",
        ],
        metadatas=[
            {"source": "test", "topic": "llm"},
            {"source": "test", "topic": "llm"},
            {"source": "test", "topic": "llm"},
        ],
    )
    print(f"插入 {len(ids)} 条记录")
    print(f"现在 KNOWLEDGE namespace 有 {store.count(Namespace.KNOWLEDGE)} 条")

    # 查询测试
    print("\n=== 查询测试 ===")
    results = store.query(Namespace.KNOWLEDGE, "Llama 上下文长度", k=2)
    for r in results:
        print(f"  [distance={r['distance']:.4f}] {r['text']}")

    # 清理
    print("\n=== 清理测试数据 ===")
    store.clear(Namespace.KNOWLEDGE)
    print(f"清空后: {store.stats()}")