from __future__ import annotations

from typing import Any

from backend.functions.news import get_f1_news


# ============================================================
# CONFIG
# ============================================================

DEFAULT_LIMIT = 8


# ============================================================
# FORMAT
# ============================================================

def _format_articles(
    articles: list[dict[str, Any]],
) -> str:

    if not articles:
        return "I couldn't find any recent Formula 1 news."

    lines: list[str] = []

    for index, article in enumerate(articles, start=1):

        title = article.get("title", "Untitled")
        source = article.get("source", "Unknown source")
        published = article.get("published_at", "")
        url = article.get("url", "")
        description = article.get("description", "")

        lines.append(f"{index}. {title}")
        lines.append(f"   Source: {source}")

        if published:
            lines.append(f"   Published: {published}")

        if description:
            lines.append(f"   {description}")

        if url:
            lines.append(f"   URL: {url}")

        lines.append("")

    return "\n".join(lines).strip()


# ============================================================
# AGENT
# ============================================================

def run_f1_news_agent(
    query: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:

    query = query.strip() if query else None

    try:

        articles = get_f1_news(
            query=query,
            limit=limit,
        )

        return {
            "success": bool(articles),
            "agent": "f1_news",
            "query": query,
            "count": len(articles),
            "articles": articles,
            "message": _format_articles(articles),
        }

    except Exception as exc:

        return {
            "success": False,
            "agent": "f1_news",
            "query": query,
            "count": 0,
            "articles": [],
            "message": f"F1 news retrieval failed: {exc}",
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ZOE F1 NEWS AGENT")
    print("=" * 60)
    print()
    print("Query-driven Formula 1 news retrieval.")
    print("Examples:")
    print("  latest F1 news")
    print("  McLaren latest news")
    print("  Ferrari latest news")
    print("  Lando Norris latest news")
    print("  Max Verstappen latest news")
    print()
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:

        try:
            query = input("F1 News > ").strip()

        except (KeyboardInterrupt, EOFError):
            print()
            break

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        result = run_f1_news_agent(query=query)

        print()
        print(result["message"])
        print()