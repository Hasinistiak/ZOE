from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

MODEL = "openai/gpt-oss-120b"
API_KEY = os.getenv("GROQ_API_KEY4") 

MAX_TOKENS = 2048


# ============================================================
# CLIENT
# ============================================================

client = Groq(api_key=API_KEY)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's web search agent.

Your job is to answer the user's question using the web when current,
recent, factual, or externally verifiable information is needed.

Search the web yourself when appropriate.

Rules:

- Answer exactly what the user asked.
- Do not dump search results.
- Do not describe your search process.
- Do not say "I searched the web".
- Do not return raw snippets.
- Do not return unnecessary URLs unless the user asks for them.
- Prefer current and reliable sources.
- When information is time-sensitive, use recent sources.
- If sources disagree, explain the disagreement briefly.
- Never invent information.
- If the question is simple, keep the answer simple.
- If the user asks for a list, return a list.
- If the user asks for an explanation, explain it.
- If the user asks for a person, give the relevant facts about that person.
- If the user asks about current events, provide the current information.
- If the user asks "who is", answer who they are rather than giving a
  generic news summary.
- If the user asks "what happened", summarize what happened.
- If the user asks "latest", prioritize recent information.
- If the answer cannot be reliably determined, say so.

You are an answer engine, not a news-feed formatter.
"""


# ============================================================
# AGENT
# ============================================================

def run_web_search_agent(
    query: str,
) -> dict[str, Any]:

    query = query.strip()

    if not query:
        return {
            "success": False,
            "agent": "web_search",
            "query": query,
            "answer": "No search query was provided.",
        }

    try:
        response = client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],

            tools=[
                {
                    "type": "browser_search",
                }
            ],

            # Let GPT-OSS decide how much reasoning/search is needed.
            reasoning_effort="low",

            max_completion_tokens=MAX_TOKENS,

            temperature=0.2,
        )

        message = response.choices[0].message
        answer = message.content or ""

        return {
            "success": bool(answer.strip()),
            "agent": "web_search",
            "query": query,
            "answer": answer.strip(),
        }

    except Exception as exc:

        return {
            "success": False,
            "agent": "web_search",
            "query": query,
            "answer": "",
            "error": str(exc),
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ZOE WEB SEARCH AGENT")
    print("=" * 60)
    print()
    print("Ask anything.")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:

        try:
            query = input("Web > ").strip()

        except (KeyboardInterrupt, EOFError):
            print()
            break

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        result = run_web_search_agent(query)

        print()

        if result["success"]:
            print(result["answer"])
        else:
            print(f"Web search failed: {result.get('error', 'Unknown error')}")

        print()