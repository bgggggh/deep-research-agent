# 🔬 Deep Research Agent

> **基于 LangGraph 的多智能体深度研究系统**  
> LangGraph · Gemini · Multi-Agent · Memory · Eval Harness

[![状态](https://img.shields.io/badge/%E7%8A%B6%E6%80%81-%E8%BF%9B%E8%A1%8C%E4%B8%AD-yellow)](https://github.com)
[![许可证](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green)](https://langchain-ai.github.io/langgraph/)

---

## 🎯 项目简介

一个**多智能体深度研究系统**，输入一个复杂问题，系统自主完成：

1. **规划（Planner）** — 将问题分解为 3–5 个可独立搜索的子问题
2. **搜索（Searcher）** — 针对每个子问题检索并抓取真实网页内容
3. **审核（Critic）** — 评估证据质量，动态决定继续搜索、重新规划或直接输出
4. **撰写（Writer）** — 综合所有信息，生成带引用的结构化报告

系统基于 **LangGraph 状态机**构建，每个 Agent 是一个节点，Critic Agent 的路由决策（继续 / 重规划 / 输出）是整个系统的核心。

---

## 🏗 系统架构

```
用户问题
    │
    ▼
┌─────────┐      研究计划        ┌──────────┐
│ Planner │ ──────────────────▶ │ Searcher │
└─────────┘                     └──────────┘
    ▲                                 │
    │ 重规划                       搜索结果
    │                                 │
    │                                 ▼
    │                          ┌────────────┐
    └──────────────────────────│   Critic   │
                               └────────────┘
            证据充足 │                │ 继续搜索
                     ▼                ▼
               ┌──────────┐    ┌──────────┐
               │  Writer  │    │ Searcher │（下一个子问题）
               └──────────┘    └──────────┘
                     │
                     ▼
               最终报告 + 引用
```

所有节点共享同一个 `ResearchState`，通过 LangGraph 的 `StateGraph` 管理状态流转。Critic 的条件路由避免了无效循环和过早输出。

---

## 🛠 技术栈

| 层级 | 技术 | 用途 |
|---|---|---|
| **编排框架** | LangGraph | 状态机、Agent 路由、Checkpoint |
| **主力 LLM** | Gemini 2.5 Flash | Planner / Critic / Writer（免费 tier） |
| **备用 LLM** | Llama 3.3 70B via Groq | 对比实验 / 备份（免费 tier） |
| **本地 LLM** | Qwen 2.5 14B via Ollama | 离线实验（完全免费） |
| **搜索 Skill** | DuckDuckGo Search | 网页搜索，无需 API Key |
| **抓取 Skill** | httpx + BeautifulSoup | 网页正文提取 |
| **短期记忆** | LangGraph MemorySaver | 会话内状态持久化 |
| **长期记忆** | ChromaDB + BGE-M3 | 跨 session 的情节记忆 |
| **评测** | LLM-as-judge + 自建 Harness | 准确率、引用忠实度、成本 |

**零成本运行** — 全部使用免费 tier 和本地模型。

---

## 📁 项目结构

```
deep-research-agent/
├── agent/
│   ├── state.py          # 共享状态 ResearchState
│   ├── graph.py          # LangGraph 状态机定义
│   ├── nodes/
│   │   └── agents.py     # Planner / Searcher / Critic / Writer
│   └── skills/
│       ├── search.py     # web_search 工具（DuckDuckGo）
│       └── tools.py      # web_fetch + 引用格式化工具
├── memory/
│   ├── short_term.py     # 短期记忆（Week 2）
│   └── long_term.py      # ChromaDB 长期情节记忆（Week 2）
├── llm/
│   └── client.py         # 统一 LLM 客户端（Gemini/Groq/Ollama）
├── eval/
│   ├── harness.py        # 评测主入口（Week 2）
│   ├── metrics.py        # 准确率、引用忠实度、成本（Week 2）
│   └── testset/          # 50 题评测基准（Week 2）
├── config.py
├── main.py
└── requirements.txt
```

---

## 🚀 快速开始

**1. 克隆并安装**
```bash
git clone https://github.com/bgggggh/deep-research-agent.git
cd deep-research-agent
pip install -r requirements.txt
```

# 推荐: 创建虚拟环境
conda create -n deep-research python=3.11 -y
conda activate deep-research

# 安装依赖
pip install -r requirements.txt

**2. 配置 API Key**
```bash
cp .env.example .env
# 编辑 .env，填入你的 key
```

```env
GOOGLE_API_KEY=你的_gemini_key   # 免费申请: aistudio.google.com
GROQ_API_KEY=你的_groq_key       # 免费申请: console.groq.com
```

**3. 运行**
```bash
python main.py "分析2024年国产大模型的商业化进展"
```

**4. 切换模型**（编辑 `config.py`）
```python
DEFAULT_PROVIDER = LLMProvider.GEMINI   # 可选 GROQ / OLLAMA
```

---

## 📊 评测结果

> ⏳ 进行中 — Week 2 完成后更新真实数据

| 指标 | Single-Agent Baseline | Multi-Agent（本项目） |
|---|---|---|
| 答案准确率 | — | — |
| 引用忠实度 | — | — |
| 平均 Token 成本 | — | — |
| 平均延迟（秒） | — | — |

**评测方法**：自建 50 题基准（覆盖事实查询、多步推理、开放综合三种难度），LLM-as-judge 打分。通过消融实验量化 Critic Agent 和长期记忆模块各自的贡献。

---

## 📈 进度日志

### Week 1 — 核心架构 ✅
- [x] LangGraph 四节点状态机
- [x] Planner / Searcher / Critic / Writer 实现
- [x] DuckDuckGo 搜索 + 网页抓取 Skill
- [x] Critic 动态路由（继续 / 重规划 / 输出）
- [x] 统一 LLM 客户端（Gemini / Groq / Ollama）
- [x] 流式输出，实时显示每个节点的进度

### Week 2 — 记忆 + 评测 🚧
- [ ] 短期记忆：LangGraph checkpoint 跨轮持久化
- [ ] 长期记忆：ChromaDB 情节存储
- [ ] 评测 Harness：50 题基准
- [ ] LLM-as-judge 评分 pipeline
- [ ] 消融实验：Critic 贡献度、记忆贡献度

### Week 3 — 优化 + 演示
- [ ] Streamlit 交互式演示界面
- [ ] 子问题并行搜索
- [ ] Prompt caching 降低成本
- [ ] Demo 视频 + 技术博客

---

## 🔑 关键设计决策

**为什么选 LangGraph 而不是 CrewAI / 原生 LangChain？**  
LangGraph 把状态机显式暴露出来——每条边、每个路由条件都是你自己写的代码。这让调试变得可控，也能精确推理 Agent 行为。CrewAI 把图隐藏起来；LangGraph 把图摊开给你看。

**为什么用 Critic Agent 而不是固定迭代次数？**  
固定循环的问题是：证据充足时浪费 token，证据真的不够时又会输出垃圾报告。Critic 基于实际内容动态判断，这才是人类研究员的工作方式。

**为什么用 DuckDuckGo 而不是 Tavily？**  
零 API Key 门槛，开发阶段足够用。如果需要更高质量的搜索，只需替换一行代码就能切换到 Tavily。

---

## 📝 简历描述

```
Deep Research Agent | LangGraph, Gemini API, Python
个人项目 · 2025

• 基于 LangGraph 状态机实现四角色多智能体研究系统（Planner/Searcher/
  Critic/Writer），Critic Agent 根据证据质量动态路由"继续搜索/重规划/
  输出"三种行为

• 设计分层记忆架构：短期通过 LangGraph checkpoint 持久化会话状态，
  长期通过 ChromaDB 存储情节记忆，评测多轮研究中的上下文一致性收益

• 自建 50 题三级难度评测基准 + LLM-as-judge 评分 pipeline，量化系统
  在准确率、引用忠实度、Token 成本三个维度的表现

• 零成本技术栈：Gemini 2.5 Flash 免费 tier + DuckDuckGo + 本地 Ollama；
  消融实验量化 Critic Agent 对准确率的贡献（vs. 固定迭代 baseline 提升 X%）
```

---

## 📄 License

MIT