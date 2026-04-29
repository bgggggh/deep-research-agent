"""
agent/skills/search.py
Web 搜索 skill —— 用 DuckDuckGo（完全免费，无需 API key）
"""
from langchain_core.tools import tool
from ddgs import DDGS


@tool
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """搜索网页，返回标题、URL、摘要列表。"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in results
    ]
