"""
config.py —— 全局配置，改这一个文件就能切换所有设置
"""
import os
from llm.client import LLMProvider

# ── 模型配置 ──────────────────────────────────────────────
DEFAULT_PROVIDER = LLMProvider.OLLAMA   # 改成 GROQ 或 OLLAMA 即可切换

# ── Agent 行为配置 ─────────────────────────────────────────
MAX_ITERATIONS = 3          # Critic 最多允许重搜几轮
MAX_SEARCH_RESULTS = 4      # 每个子问题搜几条
FETCH_TOP_K = 2             # 每个子问题抓取几个网页全文

# ── API Keys（从环境变量读，不要硬编码）──────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
