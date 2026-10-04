from __future__ import annotations

import json
import os
import re
import traceback
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from groq import Groq

from backend.tools.spotify import (
    PLAYLISTS,
    get_configured_playlists,
    get_current_song,
    get_devices,
    normalize_text,
    spotify_next,
    spotify_pause,
    spotify_play_album,
    spotify_play_artist,
    spotify_play_playlist,
    spotify_play_track,
    spotify_previous,
    spotify_resume,
    spotify_search,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

MODEL = os.getenv(
    "ZOE_SPOTIFY_MODEL",
    "openai/gpt-oss-120b",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY1")

# Created lazily. Raising at import time means a missing key kills
# whatever imports this module, and pause/next/etc. don't need Groq.
_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client

    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY1 is not configured.")

        _client = Groq(api_key=GROQ_API_KEY)

    return _client


# ============================================================
# SYSTEM PROMPT
#
# Built dynamically so the numbered playlists always match
# PLAYLISTS in spotify.py. spotify.py expects the router to
# choose the playlist NUMBER; this prompt is what makes that work.
# ============================================================

def _build_system_prompt() -> str:
    playlist_lines = "\n".join(
        f"    {p['number']}. {p['name']}"
        for p in get_configured_playlists()
    )

    return f"""
You are ZOE's Spotify command router.

Your ONLY task is to choose exactly ONE Spotify function.

Never answer conversationally.
Never explain.
Never perform the Spotify action yourself.
Never invent music information.

CONFIGURED PLAYLISTS (use these numbers with spotify_play_playlist):
{playlist_lines}

FUNCTIONS:

spotify_play_playlist(number)
    Play one of the configured playlists above, by number.

spotify_play_track(name, artist?)
    Play a single song.

spotify_play_artist(name)
    Play an artist.

spotify_play_album(name, artist?)
    Play an album.

spotify_search(query, search_type?)
    Search Spotify without playing. search_type is one of
    track, artist, album, playlist (default track).

spotify_pause()
spotify_resume()
spotify_next()
spotify_previous()
spotify_current()
    Playback controls / what is playing.

spotify_playlists()
    List the configured playlists.

spotify_devices()
    List available Spotify devices.

RULES:

1. Always call exactly ONE function.

2. If the user names a configured playlist (by name, by number,
   or as "my <name> playlist"), call spotify_play_playlist with
   that playlist's number. Only use numbers listed above. (Olivia Rodrigo => Just OR, Arctic Monkeys => Just AM, Tony Stark/Ironamn => Even dead I am the hero, spiderman/Peter Parker => Property of Peter Parker)

3. "play X by Y" means spotify_play_track(name="X", artist="Y").
   Do not rewrite or "fix" song titles. Titles may contain the
   word "by" (e.g. "Stand by Me"); use judgement.

4. If the user gives only an artist name, call spotify_play_artist.
   If it is not clearly an artist, treat it as a song title.

5. Use spotify_play_album only when the user says "album".

6. Do not treat an artist name as a playlist unless the user says
   "playlist" or it matches a configured playlist name.

7. "search X" / "find X" means spotify_search.

8. "what's playing", "current song", "what song is this"
   means spotify_current.

9. "pause", "stop" means spotify_pause.
   "resume", "continue" means spotify_resume.
   "next", "skip" means spotify_next.
   "previous", "back" means spotify_previous.

10. "my playlists", "show playlists" means spotify_playlists.
    "devices", "spotify devices" means spotify_devices.

Examples:

User: play Do I Wanna Know by Arctic Monkeys
-> spotify_play_track(name="Do I Wanna Know", artist="Arctic Monkeys")

User: play Olivia Rodrigo
-> spotify_play_artist(name="Olivia Rodrigo")

User: play the album AM by Arctic Monkeys
-> spotify_play_album(name="AM", artist="Arctic Monkeys")

User: play playlist 2
-> spotify_play_playlist(number=2)

User: search Arctic Monkeys artist
-> spotify_search(query="Arctic Monkeys", search_type="artist")
"""


# ============================================================
# RESULT HELPERS
# ============================================================

def _base_result(
    ok: bool,
    action: str,
    *,
    message: str,
    data: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Internal result contract for ZOE's runtime.
    `message` is the user-facing text.
    """
    result: dict[str, Any] = {
        "ok": ok,
        "action": action,
        "message": message,
    }

    if data is not None:
        result["data"] = data

    if error is not None:
        result["error"] = error

    return result


def _success(action: str, raw: Any) -> dict[str, Any]:
    return _base_result(
        True,
        action,
        message=_build_message(action, raw),
        data=raw,
    )


def _failure(action: str, error: str) -> dict[str, Any]:
    # spotify.py already returns user-readable error strings
    # (see _clean_error), so surface them as-is.
    error = str(error).strip() or f"Spotify couldn't {action}."

    return _base_result(
        False,
        action,
        message=error,
        error=error,
    )


def _run(
    action: str,
    fn: Callable[..., Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Call a spotify.py function and normalize the outcome.

    spotify.py has two failure styles:
      - playback functions RETURN {"success": False, "error": ...}
      - search/devices/current RAISE RuntimeError
    Both are handled here, so a failed command is never
    reported to the user as a success.
    """
    try:
        raw = fn(**kwargs)
    except (Exception, SystemExit) as exc:
        # SystemExit is caught on purpose: if any library calls
        # sys.exit() during auth, it must not take ZOE down with it.
        print(f"[Spotify Agent] {action} failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return _failure(action, str(exc) or type(exc).__name__)

    if isinstance(raw, dict) and raw.get("success") is False:
        return _failure(
            action,
            raw.get("error") or f"Spotify couldn't {action}.",
        )

    return _success(action, raw)


# ============================================================
# HUMAN-READABLE MESSAGES
# ============================================================

def _item_label(item: dict[str, Any]) -> str:
    name = item.get("name") or "Unknown"
    artist = item.get("artist")

    return f'"{name}" by {artist}' if artist else f'"{name}"'


def _build_message(action: str, raw: Any) -> str:
    # ---------------- PLAY ----------------
    if action == "play_track":
        track = raw.get("track")
        artist = raw.get("artist")

        if track and artist:
            return f'Now playing "{track}" by {artist}.'
        if track:
            return f'Now playing "{track}".'
        return "Spotify playback started."

    if action == "play_artist":
        artist = raw.get("artist")
        return f"Now playing {artist}." if artist else "Spotify playback started."

    if action == "play_album":
        album = raw.get("album")
        artist = raw.get("artist")

        if album and artist:
            return f'Now playing the album "{album}" by {artist}.'
        if album:
            return f'Now playing the album "{album}".'
        return "Spotify playback started."

    if action == "play_playlist":
        name = raw.get("playlist_name")
        return (
            f'Now playing your "{name}" playlist.'
            if name
            else "Spotify playback started."
        )

    # ---------------- CONTROLS ----------------
    if action == "pause":
        return "Spotify playback paused."

    if action == "resume":
        return "Spotify playback resumed."

    if action == "next":
        return "Skipped to the next track."

    if action == "previous":
        return "Returned to the previous track."

    # ---------------- CURRENT ----------------
    if action == "current":
        if not raw or not raw.get("name"):
            return "Nothing is playing on Spotify right now."

        label = _item_label(raw)
        album = raw.get("album")
        suffix = f" from {album}" if album else ""

        if raw.get("is_playing"):
            return f"Currently playing {label}{suffix}."

        return f"Paused on {label}{suffix}."

    # ---------------- SEARCH ----------------
    if action == "search":
        results = raw or []

        if not results:
            return "I couldn't find any matching Spotify results."

        top = "; ".join(
            f"{i}. {_item_label(item)}"
            for i, item in enumerate(results[:3], start=1)
        )

        return f"I found {len(results)} results. Top matches: {top}."

    # ---------------- PLAYLISTS ----------------
    if action == "playlists":
        playlists = raw or []

        if not playlists:
            return "No playlists are configured."

        listing = ", ".join(
            f"{p['number']}. {p['name']}"
            for p in playlists
        )

        return f"You have {len(playlists)} playlists: {listing}."

    # ---------------- DEVICES ----------------
    if action == "devices":
        devices = raw or []

        if not devices:
            return (
                "I couldn't find any Spotify devices. "
                "Open Spotify on a device and try again."
            )

        names = ", ".join(
            f"{d.get('name', 'Unknown')}"
            f"{' (active)' if d.get('is_active') else ''}"
            for d in devices
        )

        plural = "s" if len(devices) != 1 else ""

        return f"I found {len(devices)} Spotify device{plural}: {names}."

    return "Spotify operation completed."


# ============================================================
# TOOL IMPLEMENTATIONS
#
# Each is a thin wrapper over exactly one spotify.py function.
# ============================================================

def spotify_play_track_tool(
    name: str,
    artist: Optional[str] = None,
) -> dict[str, Any]:
    return _run(
        "play_track",
        spotify_play_track,
        name=str(name or "").strip(),
        artist=(artist or None),
    )


def spotify_play_artist_tool(name: str) -> dict[str, Any]:
    return _run(
        "play_artist",
        spotify_play_artist,
        name=str(name or "").strip(),
    )


def spotify_play_album_tool(
    name: str,
    artist: Optional[str] = None,
) -> dict[str, Any]:
    return _run(
        "play_album",
        spotify_play_album,
        name=str(name or "").strip(),
        artist=(artist or None),
    )


def spotify_play_playlist_tool(number: int | str) -> dict[str, Any]:
    return _run(
        "play_playlist",
        spotify_play_playlist,
        number=number,
    )


def spotify_pause_tool() -> dict[str, Any]:
    return _run("pause", spotify_pause)


def spotify_resume_tool() -> dict[str, Any]:
    return _run("resume", spotify_resume)


def spotify_next_tool() -> dict[str, Any]:
    return _run("next", spotify_next)


def spotify_previous_tool() -> dict[str, Any]:
    return _run("previous", spotify_previous)


def spotify_current_tool() -> dict[str, Any]:
    return _run("current", get_current_song)


def spotify_search_tool(
    query: str,
    search_type: str = "track",
) -> dict[str, Any]:
    query = str(query or "").strip()

    if not query:
        return _failure("search", "No search query provided.")

    return _run(
        "search",
        spotify_search,
        query=query,
        search_type=search_type or "track",
    )


def spotify_playlists_tool() -> dict[str, Any]:
    return _run("playlists", get_configured_playlists)


def spotify_devices_tool() -> dict[str, Any]:
    return _run("devices", get_devices)


# ============================================================
# GROQ TOOL DECLARATIONS
# ============================================================

def _tool(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


spotify_tools = [
    _tool(
        "spotify_play_playlist",
        "Play one of ZOE's configured playlists by its number.",
        {
            "number": {
                "type": "integer",
                "enum": list(PLAYLISTS.keys()),
                "description": "Configured playlist number.",
            },
        },
        ["number"],
    ),
    _tool(
        "spotify_play_track",
        "Play a single song.",
        {
            "name": {"type": "string", "description": "Song title."},
            "artist": {"type": "string", "description": "Artist, if given."},
        },
        ["name"],
    ),
    _tool(
        "spotify_play_artist",
        "Play an artist.",
        {"name": {"type": "string", "description": "Artist name."}},
        ["name"],
    ),
    _tool(
        "spotify_play_album",
        "Play an album.",
        {
            "name": {"type": "string", "description": "Album title."},
            "artist": {"type": "string", "description": "Artist, if given."},
        },
        ["name"],
    ),
    _tool(
        "spotify_search",
        "Search Spotify without starting playback.",
        {
            "query": {"type": "string", "description": "Search text."},
            "search_type": {
                "type": "string",
                "enum": ["track", "artist", "album", "playlist"],
                "description": "What to search for. Default track.",
            },
        },
        ["query"],
    ),
    _tool("spotify_pause", "Pause Spotify playback."),
    _tool("spotify_resume", "Resume Spotify playback."),
    _tool("spotify_next", "Skip to the next track."),
    _tool("spotify_previous", "Go to the previous track."),
    _tool("spotify_current", "Return the currently playing track."),
    _tool("spotify_playlists", "List ZOE's configured playlists."),
    _tool("spotify_devices", "List available Spotify devices."),
]


# ============================================================
# HANDLERS
# ============================================================

TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "spotify_play_playlist": spotify_play_playlist_tool,
    "spotify_play_track": spotify_play_track_tool,
    "spotify_play_artist": spotify_play_artist_tool,
    "spotify_play_album": spotify_play_album_tool,
    "spotify_pause": spotify_pause_tool,
    "spotify_resume": spotify_resume_tool,
    "spotify_next": spotify_next_tool,
    "spotify_previous": spotify_previous_tool,
    "spotify_current": spotify_current_tool,
    "spotify_search": spotify_search_tool,
    "spotify_playlists": spotify_playlists_tool,
    "spotify_devices": spotify_devices_tool,
}


# ============================================================
# FAST LOCAL ROUTER
#
# Handles only what is unambiguous. Anything that needs to
# decide song vs. artist vs. album goes to Groq.
# ============================================================

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

_PLAY_PREFIXES = (
    "start playing",
    "play me",
    "put on",
    "listen to",
    "queue up",
    "play",
)

_SEARCH_PREFIXES = (
    "look up",
    "look for",
    "search",
    "find",
)


def _extract_after_prefix(
    query: str,
    prefixes: tuple[str, ...],
) -> Optional[str]:
    # Longest first, so "play me X" isn't swallowed by "play".
    for prefix in sorted(prefixes, key=len, reverse=True):
        match = re.match(
            rf"^{re.escape(prefix)}\s+(.+)$",
            query.strip(),
            re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return None


def _match_configured_playlist(text: str) -> Optional[int]:
    """
    Match "Just OR", "my Just OR playlist", "playlist 2",
    "number two", "#2" against the configured playlists.
    """
    t = normalize_text(text)
    t = re.sub(r"^(?:my|the)\s+", "", t)
    t = re.sub(r"\s+playlist$", "", t)

    match = re.fullmatch(r"(?:playlist|number|no|#)\s*(\d+|[a-z]+)", t)

    if match:
        token = match.group(1)
        number = int(token) if token.isdigit() else _NUMBER_WORDS.get(token)

        if number in PLAYLISTS:
            return number

    for number, data in PLAYLISTS.items():
        if normalize_text(data["name"]) == t:
            return number

    return None


def _fast_route(
    query: str,
) -> Optional[tuple[str, dict[str, Any]]]:
    normalized = normalize_text(query)

    if not normalized:
        return None

    if normalized in {"pause", "stop", "pause spotify", "stop spotify"}:
        return "spotify_pause", {}

    if normalized in {
        "resume", "continue", "resume spotify", "continue spotify", "play",
    }:
        return "spotify_resume", {}

    if normalized in {
        "next", "skip", "skip song", "skip this", "next song",
        "next track", "skip track",
    }:
        return "spotify_next", {}

    if normalized in {
        "previous", "prev", "back", "previous song",
        "previous track", "go back",
    }:
        return "spotify_previous", {}

    if normalized in {
        "current", "current song", "what is playing", "whats playing",
        "what's playing", "what is playing now", "what song is playing",
        "what song is this", "what track is playing",
        "what am i listening to", "what am i listening to right now",
    }:
        return "spotify_current", {}

    if normalized in {
        "playlists", "my playlists", "show playlists",
        "show my playlists", "list playlists", "list my playlists",
    }:
        return "spotify_playlists", {}

    if normalized in {
        "devices", "spotify devices", "available devices",
        "spotify available devices", "show devices",
        "show spotify devices",
    }:
        return "spotify_devices", {}

    search_query = _extract_after_prefix(query, _SEARCH_PREFIXES)

    if search_query:
        return "spotify_search", {"query": search_query}

    play_query = _extract_after_prefix(query, _PLAY_PREFIXES)

    if play_query:
        number = _match_configured_playlist(play_query)

        if number is not None:
            return "spotify_play_playlist", {"number": number}

    return None


# ============================================================
# GROQ ROUTER
# ============================================================

def _groq_route(query: str) -> tuple[str, dict[str, Any]]:
    """
    Use Groq tool calling when the local router can't resolve
    the command. Exactly one function call is required.
    """
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": query},
        ],
        tools=spotify_tools,
        tool_choice="required",
        temperature=0.0,
        # gpt-oss is a reasoning model: reasoning tokens count against
        # this limit, and 64 can truncate before the tool call is emitted.
        max_tokens=512,
    )

    if not response.choices:
        raise RuntimeError("Groq returned no choices.")

    tool_calls = response.choices[0].message.tool_calls or []

    if len(tool_calls) != 1:
        raise RuntimeError(
            f"Expected exactly one Spotify function call, got {len(tool_calls)}."
        )

    function = tool_calls[0].function

    if not function or function.name not in TOOL_HANDLERS:
        raise RuntimeError(
            f"Unsupported Spotify function: {getattr(function, 'name', None)}"
        )

    try:
        arguments = json.loads(function.arguments or "{}")
    except Exception as exc:
        raise RuntimeError("Invalid Spotify function arguments.") from exc

    if not isinstance(arguments, dict):
        raise RuntimeError("Invalid Spotify function arguments.")

    return function.name, arguments


# ============================================================
# AGENT
# ============================================================

def run_spotify_agent(
    query: str,
    context: Any = None,
) -> str:
    """
    Execute exactly one Spotify operation.

    Returns only the final user-facing response string.
    """
    query = str(query or "").strip()

    if not query:
        return "I didn't receive a Spotify request."

    # STEP 1 - fast local routing
    route = _fast_route(query)

    # STEP 2 - Groq only if necessary
    if route is None:
        try:
            route = _groq_route(query)
        except (Exception, SystemExit):
            traceback.print_exc()
            return "I couldn't determine which Spotify action you wanted."

    name, arguments = route

    print(f"[Spotify Agent] Route: {name} {arguments}")

    handler = TOOL_HANDLERS.get(name)

    if handler is None:
        return "I couldn't execute that Spotify command."

    # STEP 3 - execute exactly one tool
    try:
        result = handler(**arguments)
    except (Exception, SystemExit):
        traceback.print_exc()
        return "Something went wrong while controlling Spotify."

    message = result.get("message")

    if isinstance(message, str) and message.strip():
        return message.strip()

    error = result.get("error")

    if isinstance(error, str) and error.strip():
        return error.strip()

    return "Spotify operation completed."


# ============================================================
# METADATA
# ============================================================

AGENT_NAME = "spotify"

AGENT_DESCRIPTION = (
    "Controls Spotify playback, search, configured playlists, "
    "current playback, and devices."
)

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
    },
    "required": ["query"],
}


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    cli_query = " ".join(sys.argv[1:]).strip()

    if not cli_query:
        cli_query = input("Spotify > ").strip()

    print()
    print(f"ZOE: {run_spotify_agent(cli_query)}")
    print()