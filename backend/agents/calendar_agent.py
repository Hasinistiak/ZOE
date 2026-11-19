"""
ZOE Personal Calendar Agent, MCP edition.

Tools live in backend.mcp_servers.calendar_server. This file keeps what is
agent-specific and should NOT live in a tool server:

  - the date table / current-time prompt
  - the seen-ID delete guard (the model may only delete IDs it was shown)
  - deterministic summaries for the main brain
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from groq import AsyncGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

MODEL = os.getenv("ZOE_CALENDAR_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY3")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY3 is not set.")

client = AsyncGroq(api_key=GROQ_API_KEY)

TIMEZONE = "Asia/Dhaka"
ZOE_TIMEZONE = ZoneInfo(TIMEZONE)

MAX_AGENT_STEPS = int(os.getenv("ZOE_CALENDAR_MAX_STEPS", "8"))
MAX_TOOL_CALLS = int(os.getenv("ZOE_CALENDAR_MAX_TOOL_CALLS", "20"))
MAX_HISTORY_MESSAGES = 6
DATE_TABLE_DAYS = 15
MAX_COMPLETION_TOKENS = int(os.getenv("ZOE_CALENDAR_MAX_TOKENS", "1600"))
DEBUG = os.getenv("ZOE_CALENDAR_DEBUG", "").lower() in {"1", "true", "yes"}

READ_TOOLS = {"get_today_events", "get_events"}

SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "backend.mcp.calendar_server"],
    env={**os.environ},
)

# ============================================================
# TIME
# ============================================================


def get_current_datetime() -> datetime:
    return datetime.now(ZOE_TIMEZONE)


def build_date_table(now: datetime, days: int = DATE_TABLE_DAYS) -> str:
    lines = []
    for offset in range(days):
        d = (now + timedelta(days=offset)).date()
        tag = "  <- TODAY" if offset == 0 else "  <- TOMORROW" if offset == 1 else ""
        lines.append(f"{d.strftime('%A')}, {d.isoformat()}{tag}")
    return "\n".join(lines)


# ============================================================
# DISPLAY HELPERS
# ============================================================


def parse_dt(value: Any) -> tuple[datetime | None, bool]:
    """ISO string -> (Dhaka datetime | None, is_all_day)."""
    if not value:
        return None, False
    text = str(value).strip()
    try:
        if len(text) == 10:
            return datetime.fromisoformat(text).replace(tzinfo=ZOE_TIMEZONE), True
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None, False
    dt = dt.replace(tzinfo=ZOE_TIMEZONE) if dt.tzinfo is None else dt.astimezone(ZOE_TIMEZONE)
    return dt, False


def fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def fmt_day(d: date) -> str:
    return f"{d.strftime('%a %b')} {d.day}"


def day_label(d: date, today: date) -> str:
    if d == today:
        return f"Today ({fmt_day(d)})"
    if d == today + timedelta(days=1):
        return f"Tomorrow ({fmt_day(d)})"
    if d == today - timedelta(days=1):
        return f"Yesterday ({fmt_day(d)})"
    return fmt_day(d)


def event_start(event: dict[str, Any]) -> tuple[datetime | None, bool]:
    return parse_dt(event.get("datetime") or event.get("start_datetime") or event.get("start"))


def event_end(event: dict[str, Any]) -> datetime | None:
    dt, _ = parse_dt(event.get("end_datetime") or event.get("end"))
    return dt


def format_event_line(event: dict[str, Any]) -> str:
    title = str(event.get("title") or "Untitled event")
    start, all_day = event_start(event)
    if start is None:
        fallback = event.get("time")
        return f"{fallback} {title}".strip() if fallback else title
    if all_day:
        return f"All day: {title}"
    end = event_end(event)
    if end and end > start:
        return f"{fmt_time(start)}-{fmt_time(end)} {title}"
    return f"{fmt_time(start)} {title}"


def format_events_by_day(events: list[dict[str, Any]], today: date) -> str:
    buckets: dict[date, list[tuple[datetime, dict[str, Any]]]] = {}
    undated: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        start, _ = event_start(event)
        if start is None:
            undated.append(event)
        else:
            buckets.setdefault(start.date(), []).append((start, event))

    parts = []
    for d in sorted(buckets):
        items = sorted(buckets[d], key=lambda pair: pair[0])
        parts.append(f"{day_label(d, today)}: " + "; ".join(format_event_line(e) for _, e in items))
    if undated:
        parts.append("; ".join(format_event_line(e) for e in undated))
    return "\n".join(parts)


# ============================================================
# EVENT HELPERS
# ============================================================


def get_event_id(event: dict[str, Any]) -> str | None:
    for key in ("event_id", "id", "google_event_id"):
        if event.get(key):
            return str(event[key])
    return None


def get_event_link(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    for key in ("htmlLink", "html_link", "url", "link"):
        if result.get(key) and str(result[key]).strip():
            return str(result[key]).strip()
    for key in ("event", "result", "data"):
        link = get_event_link(result.get(key))
        if link:
            return link
    return None


# ============================================================
# SEEN-ID GUARD
#
# The model may only delete IDs returned by a lookup during THIS run.
# Enforced here (client side), not in the server, so it holds no matter
# how long the server process lives. Hallucinated IDs are harmless.
# ============================================================


def remember_events(result: Any, seen_ids: dict[str, str]) -> None:
    if not isinstance(result, dict):
        return
    for event in result.get("events") or []:
        if isinstance(event, dict):
            event_id = get_event_id(event)
            if event_id:
                seen_ids[event_id] = str(event.get("title") or "Untitled event")


async def guarded_delete(
    session: ClientSession,
    args: dict[str, Any],
    seen_ids: dict[str, str],
) -> dict[str, Any]:
    raw_ids = args.get("event_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("event_ids must be a non-empty list.")

    ids = list(dict.fromkeys(str(i).strip() for i in raw_ids if str(i).strip()))
    allowed = [i for i in ids if i in seen_ids]
    rejected = [
        {
            "event_id": i,
            "deleted": False,
            "error": "ID was not returned by a calendar lookup in this request.",
        }
        for i in ids
        if i not in seen_ids
    ]

    results: list[dict[str, Any]] = []
    if allowed:
        mcp_result = await session.call_tool("delete_events", {"event_ids": allowed})
        parsed = _parse(_result_text(mcp_result))
        if mcp_result.isError or not isinstance(parsed, dict):
            return {"success": False, "error": str(parsed), "results": rejected}
        if "results" not in parsed:  # server-level refusal (e.g. bulk cap)
            return {**parsed, "results": rejected}
        results = parsed["results"]

    for row in results:  # the server doesn't know titles; we do
        row["title"] = seen_ids.get(row.get("event_id"), row.get("title"))

    results += rejected
    return {"success": bool(results) and all(r.get("deleted") for r in results), "results": results}


# ============================================================
# MCP HELPERS
# ============================================================


def _to_groq_tool(tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def _result_text(result) -> str:
    return "\n".join(b.text for b in result.content if getattr(b, "type", "") == "text")


def _parse(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


async def execute_tool(
    session: ClientSession,
    name: str,
    raw_arguments: Any,
    seen_ids: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (operation record, payload for the model)."""
    try:
        if isinstance(raw_arguments, dict):
            args = raw_arguments
        elif raw_arguments and str(raw_arguments).strip():
            args = json.loads(raw_arguments)
        else:
            args = {}
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")

        if name == "delete_events":
            result = await guarded_delete(session, args, seen_ids)
        else:
            mcp_result = await session.call_tool(name, args)
            text = _result_text(mcp_result)
            result = _parse(text)
            if mcp_result.isError or not isinstance(result, dict):
                result = {"success": False, "error": text}

        if name in READ_TOOLS:
            remember_events(result, seen_ids)

        operation: dict[str, Any] = {
            "tool": name,
            "arguments": args,
            "success": result.get("success") is not False,
            "result": result,
        }
        link = get_event_link(result)
        if link:
            operation["url"] = link
        return operation, result

    except Exception as exc:
        error = {"success": False, "error": str(exc)}
        return (
            {"tool": name, "arguments": raw_arguments, "success": False, "error": str(exc)},
            error,
        )


# ============================================================
# SUMMARIES (what the MAIN BRAIN sees; deterministic, no LLM)
# ============================================================


def summarize_operation(op: dict[str, Any], today: date) -> str:
    tool = op.get("tool")
    args = op.get("arguments") if isinstance(op.get("arguments"), dict) else {}
    result = op.get("result") if isinstance(op.get("result"), dict) else {}

    if "error" in op and not result:
        return f"Failed ({tool}): {op['error']}"

    if tool == "get_today_events":
        events = result.get("events") or []
        if not events:
            return f"No events today ({fmt_day(today)})."
        return f"{len(events)} event(s) today ({fmt_day(today)}):\n" + format_events_by_day(events, today)

    if tool == "get_events":
        events = result.get("events") or []
        start, _ = parse_dt(args.get("start_datetime"))
        end, _ = parse_dt(args.get("end_datetime"))
        if start and end:
            last_day = (end - timedelta(seconds=1)).date()
            span = (
                day_label(start.date(), today)
                if start.date() == last_day
                else f"{fmt_day(start.date())} to {fmt_day(last_day)}"
            )
        else:
            span = "that range"
        if not events:
            return f"No events on {span}."
        return f"{len(events)} event(s) on {span}:\n" + format_events_by_day(events, today)

    if tool == "add_event":
        title = args.get("title", "Event")
        if not op.get("success", False):
            err = result.get("error") or op.get("error") or "unknown error"
            return f"Could not add '{title}': {err}"
        start, _ = parse_dt(args.get("start_datetime"))
        if start:
            return f"Added '{title}' on {day_label(start.date(), today)} at {fmt_time(start)}."
        return f"Added '{title}'."

    if tool == "delete_events":
        rows = result.get("results") or []
        done = [r.get("title") or r.get("event_id") for r in rows if r.get("deleted")]
        failed = [r for r in rows if not r.get("deleted")]
        parts = []
        if done:
            parts.append(f"Deleted {len(done)} event(s): " + ", ".join(map(str, done)) + ".")
        if failed:
            reasons = "; ".join(
                f"{r.get('title') or r.get('event_id')}: {r.get('error', 'not deleted')}"
                for r in failed
            )
            parts.append(f"Not deleted: {reasons}")
        if not parts:
            parts.append(result.get("error") or "Nothing was deleted.")
        return " ".join(parts)

    return f"{tool}: done." if op.get("success") else f"{tool}: failed."


def build_summary(operations: list[dict[str, Any]]) -> str:
    today = get_current_datetime().date()
    has_mutation = any(op.get("tool") not in READ_TOOLS for op in operations)
    lines = []
    for op in operations:
        if has_mutation and op.get("tool") in READ_TOOLS:
            continue  # lookups done only to find IDs are noise
        text = summarize_operation(op, today)
        if text:
            lines.append(text)
    return "\n".join(lines)


def collect_operation_links(operations: list[dict[str, Any]]) -> list[str]:
    links: list[str] = []
    for op in operations:
        link = op.get("url") or get_event_link(op.get("result"))
        if link and link not in links:
            links.append(str(link))
    return links


def build_result(operations: list[dict[str, Any]], answer: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": bool(operations) and all(op.get("success", False) for op in operations),
        "summary": answer or build_summary(operations),
    }
    links = collect_operation_links(operations)
    if len(links) == 1:
        result["url"] = links[0]
    elif len(links) > 1:
        result["urls"] = links
    if DEBUG:
        result["operations"] = operations
    return result


def build_error(
    message: str,
    operations: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    operations = operations or []
    parts = []
    if operations:
        done = build_summary(operations)
        if done:
            parts.append(done)
    parts.append(message)
    result: dict[str, Any] = {"success": False, "summary": "\n".join(parts), **extra}
    if DEBUG:
        result["operations"] = operations
    return result


# ============================================================
# SYSTEM PROMPT
# NOTE: this string goes through str.format(); no literal curly braces.
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's Personal Calendar Action Agent.
You control ONLY the PERSONAL Google Calendar.
You are an EXECUTION AGENT, not a conversational assistant.

============================================================
CURRENT TIME
============================================================

Current datetime: {current_datetime}
Current date: {current_date}
Current time: {current_time}
Current day of week: {current_day}
Timezone: Asia/Dhaka / UTC+06:00

============================================================
DATE LOOKUP TABLE (authoritative)
============================================================

Use this table to resolve every weekday name and day offset.
Never compute weekdays yourself; read them from here.

{date_table}

Rules:
- "today" and "tomorrow" are marked in the table.
- A bare weekday ("Friday", "this Friday") means the nearest matching row, counting today if it matches.
- "next <weekday>" means the matching row in the week AFTER the nearest one (7 days later).
- "next week" means the 7 days starting the coming Monday.
- "this weekend" means the coming Saturday and Sunday unless the user says otherwise.
- A bare date like "the 10th" means the nearest upcoming row with that day number.
- If a date is outside the table, compute it carefully from the table's last row.
- Always use +06:00 in tool arguments.

============================================================
SCOPE
============================================================

Personal calendar only. Football and F1 calendars have separate agents.
Earlier messages in this conversation may be supplied. Use them to resolve
"those", "them", "that one". If a reference still cannot be resolved to
specific events, delete nothing.

============================================================
QUERYING
============================================================

- Whole day: from 00:00:00 of that day to 00:00:00 of the following day.
- "today": use get_today_events.
- "this week" / "next week" / any multi-day range: ONE get_events call for the full range.
- "next <thing>" / "when is my <thing>": ONE get_events call from the CURRENT
  datetime to 14 days later. If nothing matches, ONE more call for the following 30 days.

============================================================
ANSWERING READ-ONLY QUESTIONS
============================================================

After looking up events for a question (not add/delete), reply with a SHORT
plain-text answer to exactly what was asked.
- Answer only the question. Do not list unrelated events.
- Give weekday, date and time, e.g. Tue Oct 13, 4:00 PM.
- "Next X": the first matching event, plus the one after only if clearly useful.
- Match by meaning: a biology class includes titles containing Biology.
  An exam is not a class unless the user said exam.
- If nothing matches, say so in one sentence and say what range you searched.
- One to three sentences. No markdown, no preamble.

============================================================
CREATION
============================================================

Preserve the user's title. Never invent attendees, location, description,
duration or end time. "Meeting tomorrow at 5" -> title Meeting, start tomorrow 5 PM.
Complete the whole objective: "Add physics at 5 and math at 7" creates both.
If one independent operation fails, continue with the others.

============================================================
DELETION
============================================================

For ANY deletion (today or another date, one event or many):
1. look the events up (get_today_events, or get_events for the exact range;
   once per date if dates differ)
2. decide which returned events the user clearly referred to
3. call delete_events with those exact IDs

Only delete IDs that appeared in a lookup result.
"Classes" or "lessons" means the class-type events in that date's lookup.
If the user referred to a whole date without naming events and it is not
clear which they mean, delete nothing.
Never delete an event the user did not clearly refer to.
If a match is ambiguous or weak, do not guess; ask which one.

============================================================
NO INVENTION / NO FILLER
============================================================

Never invent event IDs, titles, dates, times, URLs or calendar state.
For add/delete requests, use tools, finish the operation, then stop with no
conversational text.
"""


def build_system_prompt() -> str:
    now = get_current_datetime()
    return SYSTEM_PROMPT.format(
        current_datetime=now.isoformat(timespec="seconds"),
        current_date=now.strftime("%Y-%m-%d"),
        current_time=now.strftime("%H:%M:%S"),
        current_day=now.strftime("%A"),
        date_table=build_date_table(now),
    )


def clean_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Keep only plain user/assistant text turns, most recent last."""
    cleaned: list[dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    return cleaned[-MAX_HISTORY_MESSAGES:]


def build_messages(
    user_request: str,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        *clean_history(history),
        {"role": "user", "content": user_request.strip()},
    ]


# ============================================================
# AGENT LOOP
# ============================================================


async def _run(user_request: str, history: list[dict[str, Any]] | None) -> dict[str, Any]:
    messages = build_messages(user_request, history)
    operations: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}  # per run, never global
    total_calls = 0

    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            groq_tools = [_to_groq_tool(t) for t in (await session.list_tools()).tools]

            for step in range(1, MAX_AGENT_STEPS + 1):
                print(f"[Calendar Agent] Step {step}/{MAX_AGENT_STEPS}")

                try:
                    response = await client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=groq_tools,
                        tool_choice="auto",
                        temperature=0,
                        max_completion_tokens=MAX_COMPLETION_TOKENS,
                    )
                except Exception as exc:
                    return build_error(f"Calendar model request failed: {exc}", operations)

                if not response.choices:
                    return build_error("Invalid model response.", operations)

                message = response.choices[0].message
                calls = message.tool_calls or []

                # ---- model finished ----
                if not calls:
                    text = (message.content or "").strip()
                    if operations:
                        read_only = all(op.get("tool") in READ_TOOLS for op in operations)
                        # Read-only question: the model's short answer wins;
                        # falls back to the deterministic summary if empty.
                        return build_result(operations, answer=text if (read_only and text) else None)
                    return build_error(
                        "Calendar agent did not perform any action."
                        + (f" Model said: {text}" if text else "")
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.function.name,
                                    "arguments": c.function.arguments,
                                },
                            }
                            for c in calls
                        ],
                    }
                )

                for call in calls:
                    if total_calls >= MAX_TOOL_CALLS:
                        return build_error("Maximum tool calls reached.", operations)
                    total_calls += 1

                    name = call.function.name
                    print(f"[Calendar Agent] Tool #{total_calls}: {name}")

                    operation, payload = await execute_tool(
                        session, name, call.function.arguments, seen_ids
                    )
                    operations.append(operation)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(payload, ensure_ascii=False, default=str),
                        }
                    )

    return build_error("Calendar agent reached the maximum number of steps.", operations)


# ============================================================
# ZOE INTERFACE (sync, same signatures as before)
# ============================================================


def run_calendar_agent(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Returns {"success", "summary", ["needs_clarification"], ["url"/"urls"]}.
    Store result["summary"] as the assistant turn in history so references
    like "those" resolve on the next request.
    """
    if not query or not query.strip():
        return build_error("Calendar request cannot be empty.")

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run(query, history))

    box: dict[str, Any] = {}
    t = threading.Thread(target=lambda: box.update(r=asyncio.run(_run(query, history))))
    t.start()
    t.join()
    return box["r"]


def run_calendar_agent_text(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    return run_calendar_agent(query, history=history).get("summary", "")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    now = get_current_datetime()
    print("ZOE Personal Calendar Agent (MCP)")
    print(f"Now: {now.strftime('%A, %Y-%m-%d %H:%M')} ({TIMEZONE})")

    chat_history: list[dict[str, str]] = []
    while True:
        try:
            q = input("\nYou > ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue

            out = run_calendar_agent(q, history=chat_history)
            chat_history.append({"role": "user", "content": q})
            chat_history.append({"role": "assistant", "content": out.get("summary", "")})

            print(out.get("summary", ""))
            if DEBUG:
                print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"ERROR: {exc}")