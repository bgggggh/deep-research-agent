# Deep Research Agent

> 基于 LangGraph 的多智能体深度研究系统，含 RAG、长期记忆与完整消融实验

[![Status](https://img.shields.io/badge/status-Week_2_完成-brightgreen)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

输入一个复杂问题，系统自主完成 **规划 → 检索 → 审核 → 撰写**，输出带引用的研究报告。全栈零成本：本地 Qwen 2.5 14B + 免费 Gemini API + DuckDuckGo + ChromaDB。

---

## 📊 核心实验结果

通过 4 配置消融实验（baseline / no_critic / no_rag / full_cold）在自建 50 题测试集上验证，**实测数据揭示了一个反直觉的工程结论**：

> **Multi-Agent 不是普适更优，而是复杂任务的最优解。**

### Easy 题（n=5，单一事实查询）

| 指标 | baseline | no_critic | no_rag | full_cold |
|---|---:|---:|---:|---:|
| Quality 综合分（1-5） | 4.40 | 4.47 | 4.20 | 4.33 |
| Keyword 部分覆盖 | **76.7%** | 70.0% | 70.0% | 70.0% |
| Keyword Pass Rate | **100%** | 80% | 80% | 80% |
| RAG 命中率 | 0% | 0% | 0% | **60.6%** |
| 端到端延迟 | **23s** | 62s | 67s | 94s |

简单查询场景下 baseline 凭"一题一搜"达到接近的质量，且只用 1/4 延迟。

### Hard 题（n=5，复杂综合任务）

| 指标 | baseline | no_critic | no_rag | full_cold |
|---|---:|---:|---:|---:|
| Quality 综合分（1-5） | 3.67 | **4.13** | 3.60 | 3.87 |
|   - Support 维度 | 3.00 | **3.80** | 3.00 | **3.80** |
| Keyword 部分覆盖 | 31.7% | **58.3%** | 58.3% | 53.3% |
| Keyword Pass Rate | 80% | **100%** | 100% | 100% |
| RAG 命中率 | 0% | 0% | 0% | **63.5%** |
| 端到端延迟 | 28s | 86s | 90s | 126s |

复杂任务场景下 baseline 关键词覆盖率从 76.7% 崩到 31.7%，完整系统稳定在 53.3%（**+68% 相对提升**）。RAG 跨会话命中率 63.5%，节省 ~4 次 web 搜索。

观察到 no_critic 在 Qwen 14B 上可能优于 full_cold，揭示 Critic 动态路由对底层模型 prompt 遵循能力的依赖。

---

## 🏗 系统架构

```
                            用户问题
                                │
                                ▼
                         ┌─────────────┐
                         │   Planner   │  ◀──┐
                         └─────────────┘     │
                                │            │ replan
                                ▼            │
                         ┌─────────────┐     │
                         │  Retriever  │     │
                         │  ┌─────┐    │     │
                         │  │ RAG │ 三层向量库
                         │  └─────┘    │     │
                         └─────────────┘     │
                                │            │
                                ▼            │
                         ┌─────────────┐     │
              ┌────────▶ │  Searcher   │     │
              │          └─────────────┘     │
              │                │             │
              │ continue       ▼             │
              │          ┌─────────────┐     │
              └──────────│   Critic    │ ────┘
                         └─────────────┘
                                │ sufficient
                                ▼
                         ┌─────────────┐
                         │   Writer    │
                         └─────────────┘
                                │
                                ▼
                ┌──────────────────────────────┐
                │   Episodic Writer (回写)     │
                └──────────────────────────────┘
                                │
                                ▼
                       报告 + Citations
```

所有节点共享 `ResearchState`（TypedDict），通过状态读写通信。Critic 的条件路由是整个系统自主性的来源。

---

## 🛠 技术栈

| 层级 | 技术 |
|---|---|
| 编排框架 | LangGraph 0.2 |
| LLM | Gemini 2.5 Flash / Qwen 2.5 14B (Ollama) |
| Embedding | BGE-M3 (智源) |
| 向量库 | ChromaDB (3 namespaces) |
| 关键词检索 | rank-bm25 |
| 检索融合 | RRF (k=60) |
| 分块 | Parent-Child Chunking |
| 网页抓取 | httpx + BeautifulSoup |
| 搜索 API | DuckDuckGo (`ddgs`) |
| 评估 | LLM-as-judge + 自建 metrics |

---

## 📁 项目结构

```
deep-research-agent/
├── agent/
│   ├── state.py               # ResearchState TypedDict
│   ├── graph.py               # 可配置 LangGraph 工厂（4 ablation 配置）
│   ├── nodes/
│   │   ├── agents.py          # Planner / Searcher / Critic / Writer
│   │   ├── retriever.py       # RAG 检索节点
│   │   └── episodic_writer.py # 已验证事实回写
│   └── skills/
│       ├── search.py          # web_search (DuckDuckGo)
│       └── tools.py           # web_fetch + format_citations
├── memory/
│   ├── vector_store.py        # ChromaDB 三 namespace 封装
│   ├── chunking.py            # Parent-Child Chunker
│   └── retrieval.py           # 混合检索 (BM25 + Dense + RRF)
├── llm/client.py              # 三模型可切换
├── eval/
│   ├── metrics.py             # 4 维度指标
│   ├── runner.py              # 单配置评测
│   ├── ablation.py            # 4 配置消融实验
│   ├── testset/questions.json # 50 题测试集
│   └── results/               # 实验输出
├── main.py
└── requirements.txt
```

---

## 🚀 快速开始

```bash
git clone https://github.com/bgggggh/deep-research-agent.git
cd deep-research-agent

conda create -n deep-research python=3.11 -y
conda activate deep-research
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env 填入 GOOGLE_API_KEY 或 GROQ_API_KEY

# 或本地模型（无需 API Key）
ollama pull qwen2.5:14b
# config.py 里改 DEFAULT_PROVIDER = LLMProvider.OLLAMA

python main.py "DeepSeek V3 的技术特点和商业化策略"
```

---

## 🧪 复现实验

```bash
rm -rf chroma_db/

# Hard 题消融
python -m eval.ablation --limit 5 --difficulty hard

# Easy 题消融
python -m eval.ablation --limit 5 --difficulty easy

# 单配置完整评测
python -m eval.runner --config full_50
```

输出自动包含 4 配置对比表。

---

## 🔑 关键设计决策

**LangGraph vs CrewAI**：LangGraph 显式暴露状态机和路由条件，调试时能精确定位问题。CrewAI 把图隐藏起来，黑盒程度太高。

**三 namespace 而非单一向量库**：用户上传 / Agent 验证 / 临时抓取的可信度不同，混在一起检索会污染结果。

**BM25 + 向量混合检索**：向量检索处理不了精确专有名词（embedding 压缩中被弱化），BM25 处理不了同义词改写。RRF 融合两者，无需手动调权重。

**Parent-Child Chunking**：解决"小 chunk 检索准但缺上下文 vs 大 chunk 上下文完整但检索糊"的矛盾。200 字小块索引、1500 字大块喂 LLM。

**Retriever 用启发式而非 LLM 判断 sufficiency**：最初用 LLM 判断"够不够"，发现单次调用 ~10s + 小模型保守倾向严重。改为基于 RRF 分数的启发式后速度提升 1000×。

---

## 🐛 Debug 故事

**ChromaDB rmtree 后 readonly**：`PersistentClient` 持有 SQLite 文件句柄和内存 schema cache，rmtree 删的是磁盘文件，client 不知道。改用 `delete_collection` API。

**LangGraph stream 模式只 yield 局部更新**：以为返回完整 state 导致拿不到 final_report。stream 是 yield-based 设计，需要调用方手动 merge。

**Qwen 14B sufficiency 判断保守**：几乎全判 insufficient 让 RAG 形同虚设。改成基于 RRF 分数的启发式规则。

---

## 📄 License

MIT