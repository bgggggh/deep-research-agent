"""
llm/client.py
统一 LLM 客户端 —— 切换模型只改 config，其余代码不动
支持: Gemini 2.5 Flash (免费) / Llama 3.3 70B via Groq (免费) / Qwen 本地 Ollama (免费)
"""
from enum import Enum
from langchain_core.language_models import BaseChatModel


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"


def get_llm(provider: LLMProvider = LLMProvider.GEMINI, **kwargs) -> BaseChatModel:
    """
    返回 LangChain 兼容的 LLM 实例。
    LangGraph 节点直接调用 llm.invoke() 或 llm.stream()。
    """
    if provider == LLMProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            **kwargs,
        )

    elif provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            **kwargs,
        )

    elif provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="qwen2.5:14b",
            temperature=0.1,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── 默认实例（直接 import 用）─────────────────────────────
# 可在 config.py 里统一切换
from config import DEFAULT_PROVIDER
default_llm = get_llm(DEFAULT_PROVIDER)
