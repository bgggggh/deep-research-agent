"""
memory/chunking.py
父子文档分块器 (Parent-Child Chunking)

核心思想：
  - 小 chunk 索引（向量更聚焦，检索更准）
  - 大 chunk 提供给 LLM（上下文完整，不会断章取义）
  - 通过 metadata["parent_id"] 把两层关联起来

这是解决 RAG 核心矛盾的工程方案：
  小 chunk 检索精准 ↔ 大 chunk 上下文完整
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ═════════════════════════════════════════════════════════
# 数据结构
# ═════════════════════════════════════════════════════════
@dataclass
class ParentChunk:
    """大块——给 LLM 看的，包含完整上下文"""
    id: str
    text: str
    metadata: dict


@dataclass
class ChildChunk:
    """小块——做向量索引的，更聚焦"""
    id: str
    text: str
    parent_id: str       # 指向所属的 parent
    metadata: dict


@dataclass
class ChunkingResult:
    """分块结果：parents 存 KV store，children 喂向量库"""
    parents: list[ParentChunk]
    children: list[ChildChunk]


# ═════════════════════════════════════════════════════════
# 分块器
# ═════════════════════════════════════════════════════════
class ParentChildChunker:
    """
    两层分块：先切 parent（大块），每个 parent 再切成 children（小块）。

    用法：
        chunker = ParentChildChunker()
        result = chunker.split(text="...", source_metadata={"url": "..."})
        # 把 result.children 写入向量库（作检索用）
        # 把 result.parents 写入 KV store（作上下文返回用）
    """

    def __init__(
        self,
        parent_size: int = 1500,
        parent_overlap: int = 100,
        child_size: int = 200,
        child_overlap: int = 30,
    ):
        # ── parent 用大窗口 + 较小 overlap（保证语义完整）──────
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )

        # ── child 用小窗口 + 较大 overlap 比例（保证检索粒度细）──
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\n", "。", "！", "？", ".", "!", "?", "，", ",", " ", ""],
        )

    def split(
        self,
        text: str,
        source_metadata: Optional[dict] = None,
    ) -> ChunkingResult:
        """
        将一段文本切成 parent + child 两层。
        source_metadata 会被复制到所有 parent 和 child 上（如 url, source, timestamp）。
        """
        if source_metadata is None:
            source_metadata = {}

        # 1. 切 parent
        parent_texts = self.parent_splitter.split_text(text)
        parents = [
            ParentChunk(
                id=str(uuid.uuid4()),
                text=pt,
                metadata={**source_metadata, "level": "parent"},
            )
            for pt in parent_texts
        ]

        # 2. 每个 parent 再切 children
        children = []
        for parent in parents:
            child_texts = self.child_splitter.split_text(parent.text)
            for ct in child_texts:
                children.append(
                    ChildChunk(
                        id=str(uuid.uuid4()),
                        text=ct,
                        parent_id=parent.id,
                        metadata={
                            **source_metadata,
                            "level": "child",
                            "parent_id": parent.id,
                        },
                    )
                )

        return ChunkingResult(parents=parents, children=children)


# ═════════════════════════════════════════════════════════
# Parent KV Store —— 存 parent_id → parent 的映射
# ═════════════════════════════════════════════════════════
# 简单实现：用 ChromaDB 的同一个 client 存（不索引向量，只做 KV）
# 也可以用 sqlite / json 文件 / Redis，看场景

class ParentStore:
    """
    Parent chunk 的 KV 存储。
    检索流程: 查到 child_id → 拿到 parent_id → store.get(parent_id) → 喂给 LLM
    """

    def __init__(self):
        self._store: dict[str, ParentChunk] = {}

    def add(self, parents: list[ParentChunk]) -> None:
        for p in parents:
            self._store[p.id] = p

    def get(self, parent_id: str) -> Optional[ParentChunk]:
        return self._store.get(parent_id)

    def get_many(self, parent_ids: list[str]) -> list[ParentChunk]:
        return [p for pid in parent_ids if (p := self._store.get(pid)) is not None]

    def count(self) -> int:
        return len(self._store)


# 全局单例
_default_parent_store: Optional[ParentStore] = None


def get_parent_store() -> ParentStore:
    global _default_parent_store
    if _default_parent_store is None:
        _default_parent_store = ParentStore()
    return _default_parent_store


# ═════════════════════════════════════════════════════════
# 自测：python -m memory.chunking
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    sample_text = """
人工智能（AI）大模型在2024年取得了显著进展。OpenAI 发布了 GPT-4o，
全面增强了多模态能力。Anthropic 推出了 Claude 3.5 Sonnet，在编程
任务上达到了新的 SOTA。

国内方面，DeepSeek 推出了 V3 版本的 MoE 模型，671B 总参数但每次
推理只激活 37B，大幅降低了推理成本。阿里通义千问 2.5 系列开源，
在中文理解任务上表现优异。

商业化方面，2024 年是大模型落地的关键年。金融、教育、医疗等
垂直行业开始大规模采用大模型，企业 SaaS 厂商也纷纷集成 LLM 能力。
然而，幻觉问题、数据隐私、推理成本仍是落地的主要挑战。
""".strip()

    chunker = ParentChildChunker(
        parent_size=200,    # 测试用小窗口看效果
        parent_overlap=20,
        child_size=80,
        child_overlap=15,
    )

    result = chunker.split(
        sample_text,
        source_metadata={"source": "test_doc", "url": "https://example.com"},
    )

    print(f"=== 分块结果 ===")
    print(f"parent 数量: {len(result.parents)}")
    print(f"child 数量: {len(result.children)}")
    print(f"平均每个 parent 包含 {len(result.children)/len(result.parents):.1f} 个 child")

    print(f"\n=== Parent #1 ({len(result.parents[0].text)} 字) ===")
    print(result.parents[0].text)

    print(f"\n=== 该 Parent 下的 Children ===")
    children_of_p1 = [c for c in result.children if c.parent_id == result.parents[0].id]
    for i, c in enumerate(children_of_p1, 1):
        print(f"  Child {i} ({len(c.text)} 字): {c.text[:60]}...")

    # 测试 ParentStore
    print(f"\n=== ParentStore 测试 ===")
    store = get_parent_store()
    store.add(result.parents)
    print(f"存入 {store.count()} 个 parent")
    p = store.get(result.parents[0].id)
    print(f"按 ID 取回: {p.text[:50]}...")