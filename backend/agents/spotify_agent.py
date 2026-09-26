from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq

from backend.functions.spotify import (
    get_current_song,
    get_devices,
    get_playlists,
    normalize_text,
    spotify_next,
    spotify_play,
    spotify_play_pause,
    spotify_previous,
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

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY1"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY1 is not configured."
    )

client = Groq(
    api_key=GROQ_API_KEY,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's Spotify command router.

Your ONLY task is to choose exactly ONE Spotify function.

Never answer conversationally.
Never explain.
Never perform the Spotify action yourself.
Never invent music information.

FUNCTIONS:

spotify_play(query)
    Play a song, artist, album, or playlist.

spotify_pause()
    Pause playback.

spotify_resume()
    Resume playback.

spotify_next()
    Skip to the next track.

spotify_previous()
    Go to the previous track.

spotify_current()
    Return the currently playing track.

spotify_search(query)
    Search Spotify without playing anything.

spotify_playlists()
    Return the user's playlists.

spotify_devices()
    Return available Spotify devices.

RULES:

1. Always call exactly ONE function.

2. For play requests, preserve the user's actual
   music query. Do not rewrite song titles.

3. "play X" means spotify_play(query="X").

4. "play X by Y" must preserve the entire phrase:
   "X by Y".

5. "search X" or "find X" means spotify_search.

6. "what is playing", "what's playing",
   "current song", "what song is this"
   means spotify_current.

7. "pause", "stop" means spotify_pause.

8. "resume", "continue" means spotify_resume.

9. "next", "skip", "next song" means spotify_next.

10. "previous", "back", "previous song"
    means spotify_previous.

11. "my playlists", "show my playlists",
    "show playlists" means spotify_playlists.

12. "devices", "spotify devices",
    "available devices" means spotify_devices.

13. Do not interpret an artist name as a playlist
    unless the user explicitly says playlist.

Examples:

User: play Do I Wanna Know by Arctic Monkeys
-> spotify_play("Do I Wanna Know by Arctic Monkeys")

User: play Olivia Rodrigo
-> spotify_play("Olivia Rodrigo")

User: play my Olivia Rodrigo playlist
-> spotify_play("my Olivia Rodrigo playlist")

User: play Just OR
-> spotify_play("Just OR")

User: play number two
-> spotify_play("number two")

User: search Arctic Monkeys
-> spotify_search("Arctic Monkeys")

User: what song is playing
-> spotify_current()

User: pause
-> spotify_pause()

User: continue
-> spotify_resume()

User: skip this
-> spotify_next()
"""


# ============================================================
# RESULT HELPERS
# ============================================================

def _base_result(
    ok: bool,
    action: str,
    *,
    message: str | None = None,
    data: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Internal result contract.

    The result is structured for ZOE's runtime/Brain.

    IMPORTANT:
    This dictionary is NOT intended to be shown directly
    to the user. The user-facing response is `message`.
    """

    result: dict[str, Any] = {
        "ok": ok,
        "action": action,
    }

    if message is not None:
        result["message"] = message

    if data is not None:
        result["data"] = data

    if error is not None:
        result["error"] = error

    return result


def _success(
    action: str,
    result: Any = None,
) -> dict[str, Any]:
    """
    Convert a raw Spotify result into the normalized
    internal Spotify response.
    """

    data = _normalize_result(
        action,
        result,
    )

    message = _build_message(
        action,
        data,
    )

    return _base_result(
        True,
        action,
        message=message,
        data=data,
    )


def _failure(
    action: str,
    error: str,
) -> dict[str, Any]:
    return _base_result(
        False,
        action,
        error=str(error),
        message=f"Spotify couldn't {action}.",
    )


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def _normalize_result(
    action: str,
    result: Any,
) -> dict[str, Any]:
    """
    Normalize different Spotify function responses into
    one predictable internal format.
    """

    if result is None:
        return {}

    if not isinstance(result, dict):
        return {
            "value": result,
        }

    # --------------------------------------------------------
    # MUSIC
    # --------------------------------------------------------

    if action in {
        "play",
        "current",
        "search",
    }:
        return _normalize_music_result(
            action,
            result,
        )

    # --------------------------------------------------------
    # PLAYLISTS
    # --------------------------------------------------------

    if action == "playlists":
        return _normalize_playlists(
            result
        )

    # --------------------------------------------------------
    # DEVICES
    # --------------------------------------------------------

    if action == "devices":
        return _normalize_devices(
            result
        )

    # --------------------------------------------------------
    # PLAYBACK
    # --------------------------------------------------------

    if action in {
        "pause",
        "resume",
        "next",
        "previous",
    }:
        return _normalize_playback_result(
            action,
            result,
        )

    return dict(result)


# ============================================================
# MUSIC NORMALIZATION
# ============================================================

def _normalize_music_result(
    action: str,
    result: dict[str, Any],
) -> dict[str, Any]:

    # --------------------------------------------------------
    # SEARCH RESULT
    # --------------------------------------------------------

    if action == "search":

        return _normalize_search_result(
            result
        )

    # --------------------------------------------------------
    # TRACK RESULT
    # --------------------------------------------------------

    data: dict[str, Any] = {}

    item = result.get("item")

    # Some implementations return:
    #
    # {
    #     "name": "...",
    #     "artist": "...",
    # }
    #
    # while Spotify itself may return:
    #
    # {
    #     "item": {...}
    # }

    source = (
        item
        if isinstance(item, dict)
        else result
    )

    # --------------------------------------------------------
    # Basic fields
    # --------------------------------------------------------

    if source.get("type"):
        data["type"] = source["type"]

    if source.get("name"):
        data["name"] = source["name"]

    if source.get("uri"):
        data["uri"] = source["uri"]

    if source.get("url"):
        data["url"] = source["url"]

    if source.get("duration_ms") is not None:
        data["duration_ms"] = source[
            "duration_ms"
        ]

    # --------------------------------------------------------
    # Artist
    # --------------------------------------------------------

    artist_names: list[str] = []

    artists = source.get("artists")

    if isinstance(artists, list):

        for artist in artists:

            if isinstance(artist, dict):

                name = artist.get("name")

                if name:
                    artist_names.append(
                        str(name)
                    )

            elif isinstance(artist, str):
                artist_names.append(
                    artist
                )

    if artist_names:

        data["artists"] = artist_names

        data["artist"] = ", ".join(
            artist_names
        )

    elif source.get("artist"):

        data["artist"] = str(
            source["artist"]
        )

    # --------------------------------------------------------
    # Album
    # --------------------------------------------------------

    album = source.get("album")

    if isinstance(album, dict):

        if album.get("name"):
            data["album"] = album[
                "name"
            ]

        if album.get("uri"):
            data["album_uri"] = album[
                "uri"
            ]

    elif isinstance(album, str):

        data["album"] = album

    elif source.get("album_name"):

        data["album"] = str(
            source["album_name"]
        )

    # --------------------------------------------------------
    # Playback state
    # --------------------------------------------------------

    if result.get("is_playing") is not None:
        data["is_playing"] = bool(
            result["is_playing"]
        )

    if result.get("device") is not None:
        data["device"] = result[
            "device"
        ]

    # --------------------------------------------------------
    # Preserve useful metadata
    # --------------------------------------------------------

    for key in (
        "success",
        "reason",
    ):

        if key in result:
            data[key] = result[key]

    return data


# ============================================================
# SEARCH NORMALIZATION
# ============================================================

def _normalize_search_result(
    result: dict[str, Any],
) -> dict[str, Any]:

    raw_results = (
        result.get("results")
        or result.get("items")
        or result.get("tracks")
        or result.get("data")
    )

    if isinstance(raw_results, dict):

        raw_results = (
            raw_results.get("items")
            or raw_results.get("results")
            or []
        )

    if not isinstance(
        raw_results,
        list,
    ):
        raw_results = []

    normalized: list[dict[str, Any]] = []

    for item in raw_results:

        if not isinstance(
            item,
            dict,
        ):
            continue

        track = _normalize_track(
            item
        )

        if track:
            normalized.append(
                track
            )

    return {
        "count": len(normalized),
        "results": normalized,
    }


# ============================================================
# TRACK NORMALIZATION
# ============================================================

def _normalize_track(
    track: dict[str, Any],
) -> dict[str, Any]:

    item: dict[str, Any] = {}

    if track.get("name"):
        item["name"] = track[
            "name"
        ]

    if track.get("uri"):
        item["uri"] = track[
            "uri"
        ]

    if track.get("url"):
        item["url"] = track[
            "url"
        ]

    artists = track.get(
        "artists"
    )

    artist_names: list[str] = []

    if isinstance(
        artists,
        list,
    ):

        for artist in artists:

            if isinstance(
                artist,
                dict,
            ):

                if artist.get("name"):
                    artist_names.append(
                        str(
                            artist["name"]
                        )
                    )

            elif isinstance(
                artist,
                str,
            ):
                artist_names.append(
                    artist
                )

    if artist_names:

        item["artists"] = artist_names
        item["artist"] = ", ".join(
            artist_names
        )

    elif track.get("artist"):

        item["artist"] = str(
            track["artist"]
        )

    album = track.get(
        "album"
    )

    if isinstance(
        album,
        dict,
    ):

        if album.get("name"):
            item["album"] = album[
                "name"
            ]

    elif album:

        item["album"] = str(
            album
        )

    return item


# ============================================================
# PLAYLIST NORMALIZATION
# ============================================================

def _normalize_playlists(
    result: dict[str, Any],
) -> dict[str, Any]:

    playlists = (
        result.get("playlists")
        or result.get("items")
        or result.get("data")
    )

    if isinstance(
        playlists,
        dict,
    ):
        playlists = (
            playlists.get("items")
            or playlists.get("playlists")
            or []
        )

    if not isinstance(
        playlists,
        list,
    ):
        playlists = []

    normalized: list[dict[str, Any]] = []

    for playlist in playlists:

        if not isinstance(
            playlist,
            dict,
        ):
            continue

        item: dict[str, Any] = {}

        if playlist.get("name"):
            item["name"] = playlist[
                "name"
            ]

        if playlist.get("uri"):
            item["uri"] = playlist[
                "uri"
            ]

        if playlist.get("url"):
            item["url"] = playlist[
                "url"
            ]

        if playlist.get("id"):
            item["id"] = playlist[
                "id"
            ]

        tracks = playlist.get(
            "tracks"
        )

        if isinstance(
            tracks,
            dict,
        ):

            if tracks.get("total") is not None:
                item["tracks"] = tracks[
                    "total"
                ]

        elif isinstance(
            tracks,
            int,
        ):
            item["tracks"] = tracks

        if item:
            normalized.append(
                item
            )

    return {
        "count": len(normalized),
        "playlists": normalized,
    }


# ============================================================
# DEVICE NORMALIZATION
# ============================================================

def _normalize_devices(
    result: dict[str, Any],
) -> dict[str, Any]:

    devices = (
        result.get("devices")
        or result.get("items")
        or result.get("data")
    )

    if isinstance(
        devices,
        dict,
    ):
        devices = (
            devices.get("devices")
            or devices.get("items")
            or []
        )

    if not isinstance(
        devices,
        list,
    ):
        devices = []

    normalized: list[dict[str, Any]] = []

    for device in devices:

        if not isinstance(
            device,
            dict,
        ):
            continue

        item: dict[str, Any] = {}

        for key in (
            "id",
            "name",
            "type",
            "volume_percent",
            "is_active",
        ):

            if key in device:
                item[key] = device[
                    key
                ]

        if item:
            normalized.append(
                item
            )

    return {
        "count": len(normalized),
        "devices": normalized,
    }


# ============================================================
# PLAYBACK NORMALIZATION
# ============================================================

def _normalize_playback_result(
    action: str,
    result: dict[str, Any],
) -> dict[str, Any]:

    data: dict[str, Any] = {}

    # --------------------------------------------------------
    # Explicit state
    # --------------------------------------------------------

    if result.get("state") is not None:
        data["state"] = result[
            "state"
        ]

    if result.get("is_playing") is not None:
        data["is_playing"] = bool(
            result["is_playing"]
        )

    if result.get("device") is not None:
        data["device"] = result[
            "device"
        ]

    # --------------------------------------------------------
    # Returned track
    # --------------------------------------------------------

    track = result.get(
        "track"
    )

    if isinstance(
        track,
        dict,
    ):

        normalized_track = _normalize_track(
            track
        )

        if normalized_track:
            data["track"] = normalized_track

    # --------------------------------------------------------
    # Some implementations return track
    # fields directly.
    # --------------------------------------------------------

    if (
        action in {
            "next",
            "previous",
        }
        and "track" not in data
        and (
            result.get("name")
            or result.get("artist")
        )
    ):

        track_data = _normalize_track(
            result
        )

        if track_data:
            data["track"] = track_data

    # --------------------------------------------------------
    # Fallback state
    # --------------------------------------------------------

    if not data:

        fallback_states = {
            "pause": "paused",
            "resume": "playing",
            "next": "skipped",
            "previous": "previous",
        }

        data["state"] = fallback_states.get(
            action,
            action,
        )

    return data


# ============================================================
# HUMAN-READABLE MESSAGES
# ============================================================

def _track_description(
    track: dict[str, Any],
) -> str:

    name = track.get(
        "name"
    )

    artist = track.get(
        "artist"
    )

    album = track.get(
        "album"
    )

    if name and artist and album:
        return (
            f'"{name}" by {artist} '
            f'from {album}'
        )

    if name and artist:
        return (
            f'"{name}" by {artist}'
        )

    if name:
        return f'"{name}"'

    return "the selected track"


def _build_message(
    action: str,
    data: dict[str, Any],
) -> str:

    # --------------------------------------------------------
    # PLAY
    # --------------------------------------------------------

    if action == "play":

        name = data.get(
            "name"
        )

        artist = data.get(
            "artist"
        )

        album = data.get(
            "album"
        )

        if name and artist and album:
            return (
                f'Now playing "{name}" by '
                f'{artist} from {album}.'
            )

        if name and artist:
            return (
                f'Now playing "{name}" by '
                f'{artist}.'
            )

        if name:
            return (
                f'Now playing "{name}".'
            )

        return "Spotify playback started."

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

    if action == "current":

        name = data.get(
            "name"
        )

        artist = data.get(
            "artist"
        )

        album = data.get(
            "album"
        )

        if name and artist and album:
            return (
                f'Currently playing "{name}" by '
                f'{artist} from {album}.'
            )

        if name and artist:
            return (
                f'Currently playing "{name}" by '
                f'{artist}.'
            )

        if name:
            return (
                f'Currently playing "{name}".'
            )

        return (
            "Nothing useful is currently "
            "playing on Spotify."
        )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    if action == "search":

        count = data.get(
            "count",
            0,
        )

        if count == 0:
            return (
                "I couldn't find any matching "
                "Spotify results."
            )

        if count == 1:
            result = data[
                "results"
            ][0]

            description = _track_description(
                result
            )

            return (
                f'I found one match: '
                f'{description}.'
            )

        return (
            f'I found {count} matching '
            f'Spotify results.'
        )

    # --------------------------------------------------------
    # PLAYLISTS
    # --------------------------------------------------------

    if action == "playlists":

        count = data.get(
            "count",
            0,
        )

        playlists = data.get(
            "playlists",
            [],
        )

        if count == 0:
            return (
                "You don't appear to have "
                "any Spotify playlists available."
            )

        if count == 1:
            name = (
                playlists[0].get("name")
                if playlists
                else None
            )

            if name:
                return (
                    f'You have one playlist: '
                    f'"{name}".'
                )

            return (
                "You have one Spotify playlist."
            )

        names = [
            item.get("name")
            for item in playlists
            if isinstance(item, dict)
            and item.get("name")
        ]

        if 0 < len(names) <= 5:
            formatted = ", ".join(
                f'"{name}"'
                for name in names
            )

            return (
                f'You have {count} Spotify playlists: '
                f'{formatted}.'
            )

        return (
            f'You have {count} Spotify playlists.'
        )

    # --------------------------------------------------------
    # DEVICES
    # --------------------------------------------------------

    if action == "devices":

        count = data.get(
            "count",
            0,
        )

        devices = data.get(
            "devices",
            [],
        )

        if count == 0:
            return (
                "I couldn't find any available "
                "Spotify devices."
            )

        active = [
            device
            for device in devices
            if isinstance(device, dict)
            and device.get("is_active")
        ]

        if active:

            active_names = [
                device.get("name")
                for device in active
                if device.get("name")
            ]

            if active_names:
                return (
                    f'I found {count} available '
                    f'Spotify device'
                    f'{"s" if count != 1 else ""}. '
                    f'{" and ".join(active_names)} '
                    f'{"is" if len(active_names) == 1 else "are"} '
                    f'currently active.'
                )

        if count == 1:

            name = (
                devices[0].get("name")
                if devices
                else None
            )

            if name:
                return (
                    f'I found one available '
                    f'Spotify device: "{name}".'
                )

            return (
                "I found one available "
                "Spotify device."
            )

        return (
            f'I found {count} available '
            f'Spotify devices.'
        )

    # --------------------------------------------------------
    # PAUSE
    # --------------------------------------------------------

    if action == "pause":
        return "Spotify playback paused."

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if action == "resume":

        track = data.get(
            "track"
        )

        if isinstance(
            track,
            dict,
        ):

            description = _track_description(
                track
            )

            return (
                f'Resumed playback of '
                f'{description}.'
            )

        return "Spotify playback resumed."

    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    if action == "next":

        track = data.get(
            "track"
        )

        if isinstance(
            track,
            dict,
        ):

            description = _track_description(
                track
            )

            return (
                f'Skipped to {description}.'
            )

        return (
            "Skipped to the next track."
        )

    # --------------------------------------------------------
    # PREVIOUS
    # --------------------------------------------------------

    if action == "previous":

        track = data.get(
            "track"
        )

        if isinstance(
            track,
            dict,
        ):

            description = _track_description(
                track
            )

            return (
                f'Returned to {description}.'
            )

        return (
            "Returned to the previous track."
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return (
        "Spotify operation completed."
    )


# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def spotify_play_tool(
    query: str,
) -> dict[str, Any]:

    query = str(
        query or ""
    ).strip()

    if not query:
        return _failure(
            "play",
            "No music query provided.",
        )

    try:

        result = spotify_play(
            query
        )

        return _success(
            "play",
            result,
        )

    except Exception as exc:

        return _failure(
            "play",
            str(exc),
        )


def spotify_pause_tool() -> dict[str, Any]:

    try:

        result = spotify_play_pause(
            "pause"
        )

        return _success(
            "pause",
            result,
        )

    except Exception as exc:

        return _failure(
            "pause",
            str(exc),
        )


def spotify_resume_tool() -> dict[str, Any]:

    try:

        result = spotify_play_pause(
            "resume"
        )

        return _success(
            "resume",
            result,
        )

    except Exception as exc:

        return _failure(
            "resume",
            str(exc),
        )


def spotify_next_tool() -> dict[str, Any]:

    try:

        result = spotify_next()

        return _success(
            "next",
            result,
        )

    except Exception as exc:

        return _failure(
            "next",
            str(exc),
        )


def spotify_previous_tool() -> dict[str, Any]:

    try:

        result = spotify_previous()

        return _success(
            "previous",
            result,
        )

    except Exception as exc:

        return _failure(
            "previous",
            str(exc),
        )


def spotify_current_tool() -> dict[str, Any]:

    try:

        result = get_current_song()

        return _success(
            "current",
            result,
        )

    except Exception as exc:

        return _failure(
            "current",
            str(exc),
        )


def spotify_search_tool(
    query: str,
) -> dict[str, Any]:

    query = str(
        query or ""
    ).strip()

    if not query:
        return _failure(
            "search",
            "No search query provided.",
        )

    try:

        result = spotify_search(
            query
        )

        return _success(
            "search",
            result,
        )

    except Exception as exc:

        return _failure(
            "search",
            str(exc),
        )


def spotify_playlists_tool() -> dict[str, Any]:

    try:

        result = get_playlists()

        return _success(
            "playlists",
            result,
        )

    except Exception as exc:

        return _failure(
            "playlists",
            str(exc),
        )


def spotify_devices_tool() -> dict[str, Any]:

    try:

        result = get_devices()

        return _success(
            "devices",
            result,
        )

    except Exception as exc:

        return _failure(
            "devices",
            str(exc),
        )


# ============================================================
# GROQ TOOL DECLARATIONS
# ============================================================

spotify_tools = [
    {
        "type": "function",
        "function": {
            "name": "spotify_play",
            "description": (
                "Play a Spotify song, artist, album, or playlist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The user's exact music request."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_pause",
            "description": "Pause Spotify playback.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_resume",
            "description": "Resume Spotify playback.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_next",
            "description": "Skip to the next Spotify track.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_previous",
            "description": "Play the previous Spotify track.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_current",
            "description": (
                "Return the currently playing Spotify track."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_search",
            "description": (
                "Search Spotify without starting playback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The exact search request."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_playlists",
            "description": (
                "Return the user's Spotify playlists."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_devices",
            "description": (
                "Return available Spotify playback devices."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


# ============================================================
# HANDLERS
# ============================================================

TOOL_HANDLERS = {
    "spotify_play": spotify_play_tool,
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
# ============================================================

def _extract_after_prefix(
    query: str,
    prefixes: tuple[str, ...],
) -> Optional[str]:

    normalized = normalize_text(
        query
    )

    for prefix in prefixes:

        if normalized.startswith(
            prefix + " "
        ):

            return query[
                len(prefix):
            ].strip()

    return None


def _fast_route(
    query: str,
) -> Optional[tuple[str, dict[str, Any]]]:
    """
    Resolve obvious commands without Groq.

    Conservative by design. If the request is ambiguous,
    Groq handles the routing.
    """

    normalized = normalize_text(
        query
    )

    if not normalized:
        return None

    # ========================================================
    # PAUSE
    # ========================================================

    if normalized in {
        "pause",
        "stop",
        "pause spotify",
        "stop spotify",
    }:

        return (
            "spotify_pause",
            {},
        )

    # ========================================================
    # RESUME
    # ========================================================

    if normalized in {
        "resume",
        "continue",
        "resume spotify",
        "continue spotify",
        "play",
    }:

        return (
            "spotify_resume",
            {},
        )

    # ========================================================
    # NEXT
    # ========================================================

    if normalized in {
        "next",
        "skip",
        "skip song",
        "skip this",
        "next song",
        "next track",
        "skip track",
    }:

        return (
            "spotify_next",
            {},
        )

    # ========================================================
    # PREVIOUS
    # ========================================================

    if normalized in {
        "previous",
        "prev",
        "back",
        "previous song",
        "previous track",
        "go back",
    }:

        return (
            "spotify_previous",
            {},
        )

    # ========================================================
    # CURRENT
    # ========================================================

    if normalized in {
        "current",
        "current song",
        "what is playing",
        "whats playing",
        "what's playing",
        "what is playing now",
        "what song is playing",
        "what song is this",
        "what track is playing",
        "what am i listening to",
        "what am i listening to right now",
    }:

        return (
            "spotify_current",
            {},
        )

    # ========================================================
    # PLAYLISTS
    # ========================================================

    if normalized in {
        "playlists",
        "my playlists",
        "show playlists",
        "show my playlists",
        "list playlists",
        "list my playlists",
    }:

        return (
            "spotify_playlists",
            {},
        )

    # ========================================================
    # DEVICES
    # ========================================================

    if normalized in {
        "devices",
        "spotify devices",
        "available devices",
        "spotify available devices",
        "show devices",
        "show spotify devices",
    }:

        return (
            "spotify_devices",
            {},
        )

    # ========================================================
    # SEARCH
    # ========================================================

    search_query = _extract_after_prefix(
        query,
        (
            "search",
            "find",
            "look up",
            "look for",
        ),
    )

    if search_query:

        return (
            "spotify_search",
            {
                "query": search_query,
            },
        )

    # ========================================================
    # PLAY
    # ========================================================

    play_query = _extract_after_prefix(
        query,
        (
            "play",
            "play me",
            "put on",
            "start playing",
            "listen to",
            "queue up",
        ),
    )

    if play_query:

        return (
            "spotify_play",
            {
                "query": play_query,
            },
        )

    return None


# ============================================================
# GROQ ROUTER
# ============================================================

def _groq_route(
    query: str,
) -> tuple[str, dict[str, Any]]:
    """
    Use Groq tool calling only when the local router
    cannot confidently resolve the Spotify command.

    Exactly one Spotify function call is required.
    """

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
        tools=spotify_tools,
        tool_choice="required",
        temperature=0.0,
        max_tokens=64,
    )

    if not response.choices:
        raise RuntimeError(
            "Groq returned no choices."
        )

    message = response.choices[0].message

    tool_calls = message.tool_calls or []

    if not tool_calls:
        raise RuntimeError(
            "Groq did not return a Spotify function call."
        )

    # We explicitly require exactly one operation.
    if len(tool_calls) != 1:
        raise RuntimeError(
            "Groq returned multiple Spotify function calls."
        )

    tool_call = tool_calls[0]

    if not tool_call.function:
        raise RuntimeError(
            "Groq returned an invalid Spotify function call."
        )

    name = tool_call.function.name

    if name not in TOOL_HANDLERS:
        raise RuntimeError(
            f"Groq returned unsupported Spotify "
            f"function: {name}"
        )

    arguments_raw = (
        tool_call.function.arguments
        or "{}"
    )

    try:
        import json

        arguments = json.loads(
            arguments_raw
        )

    except Exception as exc:
        raise RuntimeError(
            "Groq returned invalid Spotify "
            "function arguments."
        ) from exc

    if not isinstance(arguments, dict):
        raise RuntimeError(
            "Groq returned invalid Spotify "
            "function arguments."
        )

    return (
        name,
        arguments,
    )


# ============================================================
# AGENT
# ============================================================

def run_spotify_agent(
    query: str,
    context: Any = None,
) -> str:
    """
    Execute exactly one Spotify operation.

    Public agent contract:
        Returns only the final user-facing response string.

    The runtime should receive:
        "Now playing ..."
        "Spotify playback paused."
        "I found 5 matching Spotify results."

    Internal routing/metadata remains inside this function.
    """

    query = str(
        query or ""
    ).strip()

    if not query:
        return (
            "I didn't receive a Spotify request."
        )

    # ========================================================
    # STEP 1 — FAST LOCAL ROUTING
    # ========================================================

    route = _fast_route(
        query
    )

    # ========================================================
    # STEP 2 — GROQ ONLY IF NECESSARY
    # ========================================================

    if route is None:

        try:

            route = _groq_route(
                query
            )

        except Exception:
            return (
                "I couldn't determine which "
                "Spotify action you wanted."
            )

    name, arguments = route

    # ========================================================
    # STEP 3 — RESOLVE HANDLER
    # ========================================================

    handler = TOOL_HANDLERS.get(
        name
    )

    if handler is None:
        return (
            "I couldn't execute that Spotify command."
        )

    # ========================================================
    # STEP 4 — EXECUTE EXACTLY ONE TOOL
    # ========================================================

    try:

        result = handler(
            **arguments
        )

    except Exception:
        return (
            "Something went wrong while "
            "controlling Spotify."
        )

    # ========================================================
    # STEP 5 — RETURN SAME CONTRACT AS OTHER AGENTS
    # ========================================================

    if not isinstance(result, dict):
        return str(result)

    message = result.get(
        "message"
    )

    if isinstance(
        message,
        str,
    ) and message.strip():

        return message.strip()

    error = result.get(
        "error"
    )

    if isinstance(
        error,
        str,
    ) and error.strip():

        return error.strip()

    return (
        "Spotify operation completed."
    )


# ============================================================
# METADATA
# ============================================================

AGENT_NAME = "spotify"

AGENT_DESCRIPTION = (
    "Controls Spotify playback, search, playlists, "
    "current playback, and devices."
)

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
        }
    },
    "required": [
        "query"
    ],
}


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import sys

    query = " ".join(
        sys.argv[1:]
    ).strip()

    if not query:
        query = input(
            "Spotify > "
        ).strip()

    result = run_spotify_agent(
        query
    )

    print()

    # --------------------------------------------------------
    # USER-FACING OUTPUT
    # --------------------------------------------------------

    print(
        f"ZOE: {result}"
    )

    print()