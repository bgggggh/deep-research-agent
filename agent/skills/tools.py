"""
agent/skills/fetch.py
网页抓取 skill —— 抓取 URL 正文，截断到 max_chars 防止 token 爆炸
"""
import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
def web_fetch(url: str, max_chars: int = 3000) -> dict:
    """
    抓取指定 URL 的正文内容。
    返回 {"url": str, "content": str, "success": bool}
    """
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        # 去掉 script / style / nav
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return {"url": url, "content": text[:max_chars], "success": True}
    except Exception as e:
        return {"url": url, "content": str(e), "success": False}


# ─────────────────────────────────────────────────────────

"""
agent/skills/citation.py
引用管理 skill —— 把 claim 和来源 URL 绑定，生成带编号的引用列表
"""
from langchain_core.tools import tool


@tool
def format_citations(citations: list[dict]) -> str:
    """
    输入: [{"claim": str, "source_url": str, "snippet": str}, ...]
    输出: Markdown 格式的引用列表
    """
    if not citations:
        return ""
    lines = ["\n\n---\n**References**\n"]
    for i, c in enumerate(citations, 1):
        lines.append(f"[{i}] {c.get('source_url', 'N/A')}  \n> {c.get('snippet', '')[:200]}")
    return "\n".join(lines)
