from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from difflib import SequenceMatcher
from typing import Any, Optional

import spotipy
from dotenv import load_dotenv
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI",
    "http://127.0.0.1:8888/callback",
)

SPOTIFY_CACHE_PATH = os.getenv(
    "SPOTIFY_CACHE_PATH",
    ".spotify_cache",
)

# Spotify search is capped at 10 per request in current
# Development Mode API behavior.
MAX_SPOTIFY_SEARCH_LIMIT = 10

# Maximum results remembered for "play number X".
MAX_SEARCH_RESULTS = 20

# How long to wait after launching Spotify.
SPOTIFY_STARTUP_WAIT = float(
    os.getenv(
        "SPOTIFY_STARTUP_WAIT",
        "2.0",
    )
)

# Maximum time to wait for Spotify to expose a playback device.
SPOTIFY_DEVICE_WAIT = float(
    os.getenv(
        "SPOTIFY_DEVICE_WAIT",
        "8.0",
    )
)

# Poll interval while waiting for Spotify.
SPOTIFY_DEVICE_POLL = float(
    os.getenv(
        "SPOTIFY_DEVICE_POLL",
        "0.5",
    )
)


# ============================================================
# OAUTH
# ============================================================

SCOPE = (
    "user-read-playback-state "
    "user-read-currently-playing "
    "user-modify-playback-state "
    "user-library-read"
)

_sp: Optional[spotipy.Spotify] = None
_sp_lock = threading.RLock()


def get_spotify() -> spotipy.Spotify:
    """
    Lazily initialize Spotify.

    OAuth is never triggered merely by importing this module.
    """

    global _sp

    with _sp_lock:
        if _sp is not None:
            return _sp

        if not SPOTIFY_CLIENT_ID:
            raise RuntimeError(
                "SPOTIFY_CLIENT_ID is not configured."
            )

        if not SPOTIFY_CLIENT_SECRET:
            raise RuntimeError(
                "SPOTIFY_CLIENT_SECRET is not configured."
            )

        auth_manager = SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=SPOTIFY_REDIRECT_URI,
            scope=SCOPE,
            cache_path=SPOTIFY_CACHE_PATH,
            open_browser=True,
        )

        _sp = spotipy.Spotify(
            auth_manager=auth_manager,
        )

        return _sp


# ============================================================
# SEARCH STATE
# ============================================================

_SEARCH_STATE_LOCK = threading.RLock()

LAST_SEARCH_RESULTS: dict[int, dict[str, Any]] = {}
LAST_SEARCH_QUERY: Optional[str] = None


def _store_search_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    global LAST_SEARCH_RESULTS
    global LAST_SEARCH_QUERY

    with _SEARCH_STATE_LOCK:
        limited = results[:MAX_SEARCH_RESULTS]

        LAST_SEARCH_QUERY = query

        LAST_SEARCH_RESULTS = {
            index + 1: result
            for index, result in enumerate(limited)
        }


def get_last_search_query() -> Optional[str]:
    with _SEARCH_STATE_LOCK:
        return LAST_SEARCH_QUERY


# ============================================================
# SPOTIFY APP MANAGEMENT
# ============================================================

_SPOTIFY_APP_LOCK = threading.RLock()
_spotify_process: Optional[subprocess.Popen[Any]] = None


def is_spotify_open() -> bool:
    """
    Check whether the Spotify desktop application appears
    to be running.

    Works with normal Linux Spotify installations and also
    catches Flatpak-style Spotify processes.
    """

    try:
        result = subprocess.run(
            [
                "pgrep",
                "-f",
                r"(^|/)(spotify)( |$)|com\.spotify\.Client",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        return result.returncode == 0

    except FileNotFoundError:
        # pgrep should exist on normal Linux systems.
        # Fall back to checking process information.
        try:
            result = subprocess.run(
                ["ps", "-A", "-o", "comm="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )

            processes = [
                line.strip().lower()
                for line in result.stdout.splitlines()
            ]

            return any(
                process == "spotify"
                or "spotify" in process
                for process in processes
            )

        except Exception:
            return False

    except Exception:
        return False


def _launch_spotify_process() -> bool:
    """
    Launch the Spotify desktop application.

    Tries the native Spotify command first and then Flatpak.
    """

    global _spotify_process

    commands = [
        ["spotify"],
        ["flatpak", "run", "com.spotify.Client"],
    ]

    for command in commands:

        executable = command[0]

        # ----------------------------------------------------
        # Check whether executable exists.
        # ----------------------------------------------------

        if executable == "spotify":
            executable_exists = (
                subprocess.run(
                    ["which", "spotify"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                == 0
            )

        else:
            executable_exists = (
                subprocess.run(
                    ["which", "flatpak"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                == 0
            )

        if not executable_exists:
            continue

        try:
            _spotify_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            return True

        except Exception:
            continue

    return False


def ensure_spotify_open(
    wait_for_device: bool = False,
) -> dict[str, Any]:
    """
    Ensure the Spotify desktop application is running.

    If Spotify is already open:
        -> do nothing.

    If Spotify is closed:
        -> launch it.

    If wait_for_device=True:
        -> also wait until Spotify exposes a playback device
           through the Spotify Web API.
    """

    with _SPOTIFY_APP_LOCK:

        already_open = is_spotify_open()

        if already_open:
            result: dict[str, Any] = {
                "success": True,
                "opened": False,
                "already_open": True,
            }

        else:

            launched = _launch_spotify_process()

            if not launched:
                return {
                    "success": False,
                    "opened": False,
                    "already_open": False,
                    "message": (
                        "Spotify is not running and could "
                        "not be launched. Make sure Spotify "
                        "is installed."
                    ),
                }

            # ------------------------------------------------
            # Give the desktop application time to start.
            # ------------------------------------------------

            deadline = (
                time.monotonic()
                + SPOTIFY_STARTUP_WAIT
            )

            while time.monotonic() < deadline:

                if is_spotify_open():
                    break

                time.sleep(0.2)

            result = {
                "success": True,
                "opened": True,
                "already_open": False,
            }

    # ========================================================
    # Wait for Spotify API device if requested.
    # ========================================================

    if wait_for_device:

        deadline = (
            time.monotonic()
            + SPOTIFY_DEVICE_WAIT
        )

        while time.monotonic() < deadline:

            try:
                sp = get_spotify()

                response = sp.devices()

                devices = response.get(
                    "devices",
                    [],
                )

                if devices:
                    result["device_ready"] = True
                    return result

            except Exception:
                pass

            time.sleep(
                SPOTIFY_DEVICE_POLL
            )

        result["device_ready"] = False

    return result


def open_spotify() -> dict[str, Any]:
    """
    Public function for explicitly opening Spotify.

    If Spotify is already running, this does nothing.
    """

    try:
        result = ensure_spotify_open(
            wait_for_device=False
        )

        if not result.get("success"):
            return result

        if result.get("already_open"):
            return {
                "success": True,
                "message": "Spotify is already open.",
            }

        return {
            "success": True,
            "message": "Spotify opened.",
        }

    except Exception as exc:
        return {
            "success": False,
            "message": (
                f"Could not open Spotify: {exc}"
            ),
        }


# ============================================================
# CONFIGURED PLAYLISTS
# ============================================================

PLAYLISTS = {

    "1": "spotify:playlist:0SPRVgVOz5ujsm1NVXy00y",

    "2": "spotify:playlist:26827blLrCb7QC4I6qEZzd",

    "3": "spotify:playlist:6fSw4Sd2hPR1LjP0nRjUyP",

    "4": "spotify:playlist:1VsFBx9TrL8dmCd6jTxMAB",

    "5": "spotify:playlist:6IGubGbQPkDZIF7BL6y00N",

    "6": "spotify:playlist:0xtXuuhYKFpOzQ5qi2hGZY",

}


PLAYLIST_NAMES = {

    "1": "Just AM",

    "2": "Just OR",

    "3": "how come we never even dated",

    "4": "Slower Turner",

    "5": "Property Of Peter Parker",

    "6": "Even Dead I Am The Hero",

}


PLAYLIST_ALIASES = {

    "1": [
        "just am",
        "arctic monkeys playlist",
        "arctic monkey playlist",
        "playlist 1",
        "playlist one",
        "number 1",
        "number one",
    ],

    "2": [
        "just or",
        "olivia rodrigo playlist",
        "playlist 2",
        "playlist two",
        "number 2",
        "number two",
    ],

    "3": [
        "how come we never even dated",
        "never even dated",
        "playlist 3",
        "playlist three",
        "number 3",
        "number three",
    ],

    "4": [
        "slower turner",
        "playlist 4",
        "playlist four",
        "number 4",
        "number four",
    ],

    "5": [
        "property of peter parker",
        "peter parker playlist",
        "spider-man playlist",
        "spider man playlist",
        "spiderman playlist",
        "playlist 5",
        "playlist five",
        "number 5",
        "number five",
    ],

    "6": [
        "even dead i am the hero",
        "even dead i am the hero playlist",
        "even dead i am the hero playlist",
        "tony stark playlist",
        "tony stark",
        "iron man playlist",
        "ironman playlist",
        "iron man",
        "ironman",
        "playlist 6",
        "playlist six",
        "number 6",
        "number six",
    ],

}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()

    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(
        r"[^\w\s'-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.92

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def token_similarity(a: str, b: str) -> float:
    a_tokens = normalize_text(a).split()
    b_tokens = normalize_text(b).split()

    if not a_tokens or not b_tokens:
        return 0.0

    a_set = set(a_tokens)
    b_set = set(b_tokens)

    union = len(a_set | b_set)

    jaccard = (
        len(a_set & b_set) / union
        if union
        else 0.0
    )

    sequence = SequenceMatcher(
        None,
        " ".join(a_tokens),
        " ".join(b_tokens),
    ).ratio()

    return (
        jaccard * 0.45
        + sequence * 0.55
    )


def combined_similarity(
    a: str,
    b: str,
) -> float:
    return max(
        similarity(a, b),
        token_similarity(a, b),
    )


# ============================================================
# QUERY CLEANING
# ============================================================

def clean_play_query(query: str) -> str:
    """
    Remove conversational playback prefixes.

    Does NOT remove meaningful music words.
    """

    query = str(query or "").strip()

    if not query:
        return ""

    patterns = [
        r"^\s*can you please play\s+",
        r"^\s*could you please play\s+",
        r"^\s*would you please play\s+",
        r"^\s*can you play\s+",
        r"^\s*could you play\s+",
        r"^\s*would you play\s+",
        r"^\s*please play\s+",
        r"^\s*start playing\s+",
        r"^\s*put on\s+",
        r"^\s*play\s+",
        r"^\s*listen to\s+",
        r"^\s*i want to listen to\s+",
        r"^\s*i want to hear\s+",
        r"^\s*let me hear\s+",
        r"^\s*queue up\s+",
    ]

    for pattern in patterns:
        query = re.sub(
            pattern,
            "",
            query,
            count=1,
            flags=re.IGNORECASE,
        )

    return query.strip()


# ============================================================
# ERROR HANDLING
# ============================================================

def spotify_error_message(
    exc: Exception,
) -> str:

    if isinstance(exc, SpotifyException):

        status = getattr(
            exc,
            "http_status",
            None,
        )

        if status == 401:
            return (
                "Spotify authentication expired. "
                "Please reconnect Spotify."
            )

        if status == 403:
            return (
                "Spotify denied that request. "
                "Playback control normally requires "
                "a Spotify Premium account."
            )

        if status == 404:
            return (
                "Spotify could not find the requested "
                "resource or playback device."
            )

        if status == 429:

            reason = getattr(
                exc,
                "headers",
                {},
            ) or {}

            retry_after = reason.get(
                "Retry-After"
            )

            if retry_after:
                return (
                    "Spotify rate-limited the request. "
                    f"Retry after {retry_after} seconds."
                )

            return (
                "Spotify rate-limited the request. "
                "Please try again shortly."
            )

        reason = getattr(
            exc,
            "reason",
            None,
        )

        if reason:
            return f"Spotify error: {reason}"

    return str(exc)


# ============================================================
# DEVICE MANAGEMENT
# ============================================================

def get_active_device() -> Optional[dict[str, Any]]:
    """
    Device priority:

    1. Current playback device
    2. Active device
    3. Computer
    4. First available device

    If Spotify is not open, automatically launch it.
    """

    try:

        # ====================================================
        # IMPORTANT:
        # Make sure desktop Spotify is running first.
        # ====================================================

        startup = ensure_spotify_open(
            wait_for_device=True
        )

        if not startup.get("success"):
            print(
                "[SPOTIFY] "
                f"{startup.get('message')}"
            )

            return None

        sp = get_spotify()

        # ----------------------------------------------------
        # First ask Spotify what is actually playing.
        # ----------------------------------------------------

        try:

            current = sp.current_playback()

            if current:

                current_device = current.get(
                    "device"
                )

                if (
                    current_device
                    and current_device.get("id")
                ):
                    return current_device

        except Exception:
            pass

        # ----------------------------------------------------
        # Fall back to available devices.
        # ----------------------------------------------------

        response = sp.devices()

        devices = response.get(
            "devices",
            [],
        )

        if not devices:
            return None

        # ----------------------------------------------------
        # Active device.
        # ----------------------------------------------------

        for device in devices:

            if device.get("is_active"):
                return device

        # ----------------------------------------------------
        # Prefer computer.
        # ----------------------------------------------------

        for device in devices:

            if str(
                device.get("type", "")
            ).lower() == "computer":
                return device

        # ----------------------------------------------------
        # Otherwise first device.
        # ----------------------------------------------------

        return devices[0]

    except Exception as exc:

        print(
            "[SPOTIFY] Device lookup failed: "
            f"{spotify_error_message(exc)}"
        )

        return None


def get_devices() -> dict[str, Any]:

    try:

        # Automatically open Spotify if necessary.
        startup = ensure_spotify_open(
            wait_for_device=True
        )

        if not startup.get("success"):
            return {
                "success": False,
                "devices": [],
                "message": startup.get(
                    "message",
                    "Could not open Spotify.",
                ),
            }

        sp = get_spotify()

        response = sp.devices()

        devices = []

        for device in response.get(
            "devices",
            [],
        ):

            devices.append({
                "id": device.get("id"),
                "name": device.get("name"),
                "type": device.get("type"),
                "is_active": device.get(
                    "is_active"
                ),
                "volume_percent": device.get(
                    "volume_percent"
                ),
                "supports_volume": device.get(
                    "supports_volume"
                ),
            })

        return {
            "success": True,
            "devices": devices,
        }

    except Exception as exc:

        return {
            "success": False,
            "devices": [],
            "message": (
                f"Could not get devices: "
                f"{spotify_error_message(exc)}"
            ),
        }


# ============================================================
# PLAYLIST RESOLUTION
# ============================================================

def resolve_playlist(
    query: str,
) -> Optional[dict[str, Any]]:

    normalized = normalize_text(query)

    if not normalized:
        return None

    # --------------------------------------------------------
    # Numeric playlist
    # --------------------------------------------------------

    match = re.fullmatch(
        r"(?:playlist\s+|number\s+)?(\d+)",
        normalized,
    )

    if match:

        number = match.group(1)

        if number in PLAYLISTS:
            return {
                "type": "playlist",
                "id": number,
                "name": PLAYLIST_NAMES[number],
                "uri": PLAYLISTS[number],
            }

    # --------------------------------------------------------
    # Word-number playlist
    # --------------------------------------------------------

    word_numbers = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
    }

    match = re.fullmatch(
        r"(?:playlist\s+|number\s+)?"
        r"(one|two|three|four|five)",
        normalized,
    )

    if match:

        number = word_numbers[
            match.group(1)
        ]

        return {
            "type": "playlist",
            "id": number,
            "name": PLAYLIST_NAMES[number],
            "uri": PLAYLISTS[number],
        }

    # --------------------------------------------------------
    # Exact configured playlist aliases
    # --------------------------------------------------------

    for number, aliases in PLAYLIST_ALIASES.items():

        for alias in aliases:

            if normalized == normalize_text(alias):

                return {
                    "type": "playlist",
                    "id": number,
                    "name": PLAYLIST_NAMES[number],
                    "uri": PLAYLISTS[number],
                }

    return None


def resolve_playlist_from_phrase(
    query: str,
) -> Optional[dict[str, Any]]:
    """
    ONLY resolve configured playlists when playlist intent
    is explicit.

    Important:
        "play Olivia Rodrigo"
        -> artist

        "play my Olivia Rodrigo playlist"
        -> Just OR
    """

    normalized = normalize_text(query)

    explicit_playlist = bool(
        re.search(
            r"\b(?:playlist|playlists|list)\b",
            normalized,
        )
    )

    numeric_playlist = bool(
        re.fullmatch(
            r"(?:playlist\s+|number\s+)?"
            r"(?:\d+|one|two|three|four|five)",
            normalized,
        )
    )

    if not explicit_playlist and not numeric_playlist:
        return None

    return resolve_playlist(
        normalized
    )


# ============================================================
# MUSIC QUERY PARSER
# ============================================================

def parse_music_query(
    query: str,
) -> dict[str, Any]:

    query = clean_play_query(query)

    if not query:
        return {
            "type": "unknown",
            "query": "",
        }

    # --------------------------------------------------------
    # Artist requests
    # --------------------------------------------------------

    artist_patterns = [
        r"^something\s+by\s+(.+)$",
        r"^songs?\s+by\s+(.+)$",
        r"^music\s+by\s+(.+)$",
        r"^tracks?\s+by\s+(.+)$",
        r"^anything\s+by\s+(.+)$",
        r"^anything\s+from\s+(.+)$",
        r"^music\s+from\s+(.+)$",
    ]

    for pattern in artist_patterns:

        match = re.match(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:

            return {
                "type": "artist",
                "query": match.group(1).strip(),
            }

    # --------------------------------------------------------
    # TITLE by ARTIST
    # --------------------------------------------------------

    match = re.match(
        r"^(.+?)\s+by\s+(.+)$",
        query,
        flags=re.IGNORECASE,
    )

    if match:

        return {
            "type": "track",
            "query": match.group(1).strip(),
            "artist": match.group(2).strip(),
        }

    # --------------------------------------------------------
    # Explicit track
    # --------------------------------------------------------

    track_patterns = [
        r"^(?:the\s+)?track\s+(.+)$",
        r"^(?:the\s+)?song\s+(.+)$",
    ]

    for pattern in track_patterns:

        match = re.match(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:

            return {
                "type": "track",
                "query": match.group(1).strip(),
            }

    # --------------------------------------------------------
    # Explicit album
    # --------------------------------------------------------

    album_patterns = [
        r"^(?:the\s+)?album\s+(.+)$",
        r"^(?:the\s+)?record\s+(.+)$",
    ]

    for pattern in album_patterns:

        match = re.match(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:

            return {
                "type": "album",
                "query": match.group(1).strip(),
            }

    # --------------------------------------------------------
    # Explicit artist
    # --------------------------------------------------------

    match = re.match(
        r"^(?:the\s+)?artist\s+(.+)$",
        query,
        flags=re.IGNORECASE,
    )

    if match:

        return {
            "type": "artist",
            "query": match.group(1).strip(),
        }

    return {
        "type": "unknown",
        "query": query,
    }


# ============================================================
# ENTITY BUILDERS
# ============================================================

def _track_to_entity(
    item: dict[str, Any],
) -> dict[str, Any]:

    artists = [
        artist.get("name", "")
        for artist in item.get(
            "artists",
            [],
        )
        if artist
    ]

    return {
        "type": "track",
        "id": item.get("id"),
        "name": item.get("name"),
        "artist": ", ".join(artists),
        "artists": artists,
        "album": item.get(
            "album",
            {},
        ).get("name"),
        "uri": item.get("uri"),
        "url": item.get(
            "external_urls",
            {},
        ).get("spotify"),
        "popularity": item.get(
            "popularity",
            0,
        ),
        "is_playable": item.get(
            "is_playable",
            True,
        ),
    }


def _artist_to_entity(
    item: dict[str, Any],
) -> dict[str, Any]:

    return {
        "type": "artist",
        "id": item.get("id"),
        "name": item.get("name"),
        "artist": item.get("name"),
        "uri": item.get("uri"),
        "url": item.get(
            "external_urls",
            {},
        ).get("spotify"),
        "popularity": item.get(
            "popularity",
            0,
        ),
    }


def _album_to_entity(
    item: dict[str, Any],
) -> dict[str, Any]:

    artists = [
        artist.get("name", "")
        for artist in item.get(
            "artists",
            [],
        )
        if artist
    ]

    return {
        "type": "album",
        "id": item.get("id"),
        "name": item.get("name"),
        "artist": ", ".join(artists),
        "artists": artists,
        "uri": item.get("uri"),
        "url": item.get(
            "external_urls",
            {},
        ).get("spotify"),
    }


def _playlist_to_entity(
    item: dict[str, Any],
) -> dict[str, Any]:

    return {
        "type": "playlist",
        "id": item.get("id"),
        "name": item.get("name"),
        "description": item.get(
            "description"
        ),
        "uri": item.get("uri"),
        "url": item.get(
            "external_urls",
            {},
        ).get("spotify"),
        "tracks": item.get(
            "tracks",
            {},
        ).get("total", 0),
    }


# ============================================================
# SPOTIFY SEARCH
# ============================================================

def spotify_search(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:

    query = str(query or "").strip()

    if not query:
        return {
            "success": False,
            "message": "Empty Spotify search query.",
            "results": [],
        }

    limit = max(
        1,
        min(
            int(limit),
            MAX_SPOTIFY_SEARCH_LIMIT,
        ),
    )

    try:

        sp = get_spotify()

        response = sp.search(
            q=query,
            type="track,artist,album,playlist",
            limit=limit,
        )

        results: list[dict[str, Any]] = []

        for item in response.get(
            "tracks",
            {},
        ).get(
            "items",
            [],
        ):

            if item:
                results.append(
                    _track_to_entity(item)
                )

        for item in response.get(
            "artists",
            {},
        ).get(
            "items",
            [],
        ):

            if item:
                results.append(
                    _artist_to_entity(item)
                )

        for item in response.get(
            "albums",
            {},
        ).get(
            "items",
            [],
        ):

            if item:
                results.append(
                    _album_to_entity(item)
                )

        for item in response.get(
            "playlists",
            {},
        ).get(
            "items",
            [],
        ):

            if item:
                results.append(
                    _playlist_to_entity(item)
                )

        _store_search_results(
            query,
            results,
        )

        return {
            "success": True,
            "query": query,
            "results": results,
        }

    except Exception as exc:

        print(
            "[SPOTIFY] Search failed: "
            f"{spotify_error_message(exc)}"
        )

        return {
            "success": False,
            "query": query,
            "results": [],
            "message": (
                "Spotify search failed: "
                f"{spotify_error_message(exc)}"
            ),
        }


# ============================================================
# RESULT RANKING
# ============================================================

def _artist_score(
    requested: str,
    item: dict[str, Any],
) -> float:

    candidates = item.get(
        "artists",
        [],
    )

    if not candidates:

        candidate = item.get(
            "artist",
            "",
        )

        candidates = (
            [candidate]
            if candidate
            else []
        )

    return max(
        (
            combined_similarity(
                requested,
                candidate,
            )
            for candidate in candidates
            if candidate
        ),
        default=0.0,
    )


def rank_search_results(
    query: str,
    results: list[dict[str, Any]],
    preferred_type: Optional[str] = None,
    artist: Optional[str] = None,
) -> list[dict[str, Any]]:

    normalized_query = normalize_text(query)

    normalized_artist = normalize_text(
        artist or ""
    )

    ranked: list[
        tuple[float, dict[str, Any]]
    ] = []

    for item in results:

        item = dict(item)

        item_type = item.get(
            "type"
        )

        name = item.get(
            "name",
            "",
        )

        normalized_name = normalize_text(
            name
        )

        score = combined_similarity(
            query,
            name,
        )

        if normalized_name == normalized_query:
            score += 0.40

        elif (
            normalized_query
            and normalized_query in normalized_name
        ):
            score += 0.12

        if preferred_type:

            if item_type == preferred_type:
                score += 0.25

            else:
                score -= 0.20

        if normalized_artist:

            artist_score = _artist_score(
                artist or "",
                item,
            )

            score += (
                artist_score * 0.45
            )

            if artist_score >= 0.98:
                score += 0.25

        if item_type == "track":

            if item.get(
                "is_playable",
                True,
            ):
                score += 0.05

            else:
                score -= 0.30

        popularity = (
            float(
                item.get(
                    "popularity",
                    0,
                )
                or 0
            )
            / 100.0
        )

        score += (
            popularity * 0.025
        )

        item["_score"] = score

        ranked.append(
            (
                score,
                item,
            )
        )

    ranked.sort(
        key=lambda pair: pair[0],
        reverse=True,
    )

    return [
        item
        for _, item in ranked
    ]


# ============================================================
# SEARCH RESULT NUMBER
# ============================================================

def resolve_search_number(
    query: str,
) -> Optional[dict[str, Any]]:

    normalized = normalize_text(query)

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }

    match = re.fullmatch(
        r"(?:number\s+|result\s+)?(\d+)",
        normalized,
    )

    if match:

        number = int(
            match.group(1)
        )

        with _SEARCH_STATE_LOCK:

            return LAST_SEARCH_RESULTS.get(
                number
            )

    match = re.fullmatch(
        r"(?:number\s+|result\s+)?"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
        r"seventeen|eighteen|nineteen|twenty)",
        normalized,
    )

    if match:

        number = number_words[
            match.group(1)
        ]

        with _SEARCH_STATE_LOCK:

            return LAST_SEARCH_RESULTS.get(
                number
            )

    return None


# ============================================================
# TRACK RESOLUTION
# ============================================================

def resolve_track(
    query: str,
    artist: Optional[str] = None,
) -> Optional[dict[str, Any]]:

    query = clean_play_query(query)

    if not query:
        return None

    try:

        sp = get_spotify()

        if artist:

            search_query = (
                f'track:"{query}" '
                f'artist:"{artist}"'
            )

        else:

            search_query = (
                f'track:"{query}"'
            )

        response = sp.search(
            q=search_query,
            type="track",
            limit=10,
        )

        tracks = response.get(
            "tracks",
            {},
        ).get(
            "items",
            [],
        )

        if not tracks:
            return None

        entities = [
            _track_to_entity(item)
            for item in tracks
            if item
        ]

        ranked = rank_search_results(
            query=query,
            results=entities,
            preferred_type="track",
            artist=artist,
        )

        if not ranked:
            return None

        best = ranked[0]

        score = float(
            best.get(
                "_score",
                0.0,
            )
        )

        threshold = (
            1.05
            if artist
            else 0.82
        )

        if score < threshold:
            return None

        best = dict(best)

        best["score"] = score

        best.pop(
            "_score",
            None,
        )

        return best

    except Exception as exc:

        print(
            "[SPOTIFY] Track resolution failed: "
            f"{spotify_error_message(exc)}"
        )

        return None


# ============================================================
# ARTIST RESOLUTION
# ============================================================

def resolve_artist(
    query: str,
) -> Optional[dict[str, Any]]:

    query = clean_play_query(query)

    if not query:
        return None

    normalized_query = normalize_text(
        query
    )

    try:

        sp = get_spotify()

        response = sp.search(
            q=f'artist:"{query}"',
            type="artist",
            limit=10,
        )

        artists = response.get(
            "artists",
            {},
        ).get(
            "items",
            [],
        )

        if not artists:
            return None

        entities = [
            _artist_to_entity(item)
            for item in artists
            if item
        ]

        ranked = rank_search_results(
            query=query,
            results=entities,
            preferred_type="artist",
        )

        if not ranked:
            return None

        best = ranked[0]

        best_name = normalize_text(
            best.get(
                "name",
                "",
            )
        )

        score = float(
            best.get(
                "_score",
                0.0,
            )
        )

        if (
            best_name != normalized_query
            and score < 0.88
        ):
            return None

        best = dict(best)

        best["score"] = score

        best.pop(
            "_score",
            None,
        )

        return best

    except Exception as exc:

        print(
            "[SPOTIFY] Artist resolution failed: "
            f"{spotify_error_message(exc)}"
        )

        return None


# ============================================================
# ALBUM RESOLUTION
# ============================================================

def resolve_album(
    query: str,
    artist: Optional[str] = None,
) -> Optional[dict[str, Any]]:

    query = clean_play_query(query)

    if not query:
        return None

    try:

        sp = get_spotify()

        if artist:

            search_query = (
                f'album:"{query}" '
                f'artist:"{artist}"'
            )

        else:

            search_query = (
                f'album:"{query}"'
            )

        response = sp.search(
            q=search_query,
            type="album",
            limit=10,
        )

        albums = response.get(
            "albums",
            {},
        ).get(
            "items",
            [],
        )

        if not albums:
            return None

        entities = [
            _album_to_entity(item)
            for item in albums
            if item
        ]

        ranked = rank_search_results(
            query=query,
            results=entities,
            preferred_type="album",
            artist=artist,
        )

        if not ranked:
            return None

        best = ranked[0]

        score = float(
            best.get(
                "_score",
                0.0,
            )
        )

        threshold = (
            1.00
            if artist
            else 0.82
        )

        if score < threshold:
            return None

        best = dict(best)

        best["score"] = score

        best.pop(
            "_score",
            None,
        )

        return best

    except Exception as exc:

        print(
            "[SPOTIFY] Album resolution failed: "
            f"{spotify_error_message(exc)}"
        )

        return None


# ============================================================
# MUSIC RESOLUTION
# ============================================================

def resolve_music_query(
    query: str,
) -> Optional[dict[str, Any]]:

    query = clean_play_query(query)

    if not query:
        return None

    numbered = resolve_search_number(
        query
    )

    if numbered:
        return numbered

    parsed = parse_music_query(
        query
    )

    parsed_type = parsed.get(
        "type"
    )

    parsed_query = parsed.get(
        "query",
        "",
    )

    if parsed_type == "track":

        return resolve_track(
            parsed_query,
            parsed.get("artist"),
        )

    if parsed_type == "artist":

        return resolve_artist(
            parsed_query
        )

    if parsed_type == "album":

        return resolve_album(
            parsed_query,
            parsed.get("artist"),
        )

    search_result = spotify_search(
        query,
        limit=10,
    )

    results = search_result.get(
        "results",
        [],
    )

    if not results:
        return None

    ranked = rank_search_results(
        query=query,
        results=results,
        preferred_type="track",
    )

    if not ranked:
        return None

    best = ranked[0]

    score = float(
        best.get(
            "_score",
            0.0,
        )
    )

    if score < 0.78:
        return None

    best = dict(best)

    best["score"] = score

    best.pop(
        "_score",
        None,
    )

    return best


# ============================================================
# PLAYLIST PLAYBACK
# ============================================================

def spotify_play_playlist(
    playlist: dict[str, Any] | str,
) -> dict[str, Any]:

    if isinstance(
        playlist,
        str,
    ):

        resolved = resolve_playlist(
            playlist
        )

        if not resolved:

            return {
                "success": False,
                "message": (
                    f"Playlist '{playlist}' "
                    "was not found."
                ),
            }

        playlist = resolved

    device = get_active_device()

    if not device:

        return {
            "success": False,
            "message": (
                "No available Spotify device was found."
            ),
        }

    try:

        sp = get_spotify()

        sp.start_playback(
            device_id=device["id"],
            context_uri=playlist["uri"],
        )

        return {
            "success": True,
            "type": "playlist",
            "name": playlist.get("name"),
            "uri": playlist["uri"],
            "device": device.get("name"),
            "message": (
                f"Playing {playlist.get('name')}."
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "message": (
                "Could not start playlist playback: "
                f"{spotify_error_message(exc)}"
            ),
        }


# ============================================================
# ENTITY PLAYBACK
# ============================================================

def play_resolved_entity(
    entity: dict[str, Any],
) -> dict[str, Any]:

    if not entity:

        return {
            "success": False,
            "message": "Nothing was resolved.",
        }

    entity_type = entity.get(
        "type"
    )

    if entity_type == "playlist":

        return spotify_play_playlist(
            entity
        )

    device = get_active_device()

    if not device:

        return {
            "success": False,
            "message": (
                "No available Spotify device was found."
            ),
        }

    try:

        sp = get_spotify()

        # ----------------------------------------------------
        # TRACK
        # ----------------------------------------------------

        if entity_type == "track":

            uri = entity.get(
                "uri"
            )

            if not uri:

                return {
                    "success": False,
                    "message": (
                        "The track has no Spotify URI."
                    ),
                }

            if entity.get(
                "is_playable",
                True,
            ) is False:

                return {
                    "success": False,
                    "message": (
                        f"{entity.get('name')} is not "
                        "playable on this Spotify account."
                    ),
                }

            sp.start_playback(
                device_id=device["id"],
                uris=[uri],
            )

            return {
                "success": True,
                "type": "track",
                "name": entity.get("name"),
                "artist": entity.get("artist"),
                "album": entity.get("album"),
                "uri": uri,
                "device": device.get("name"),
                "message": (
                    f"Playing {entity.get('name')} "
                    f"by {entity.get('artist')}."
                ),
            }

        # ----------------------------------------------------
        # ARTIST
        # ----------------------------------------------------

        if entity_type == "artist":

            uri = entity.get(
                "uri"
            )

            if not uri:

                return {
                    "success": False,
                    "message": (
                        "The artist has no Spotify URI."
                    ),
                }

            sp.start_playback(
                device_id=device["id"],
                context_uri=uri,
            )

            return {
                "success": True,
                "type": "artist",
                "name": entity.get("name"),
                "uri": uri,
                "device": device.get("name"),
                "message": (
                    f"Playing {entity.get('name')}."
                ),
            }

        # ----------------------------------------------------
        # ALBUM
        # ----------------------------------------------------

        if entity_type == "album":

            uri = entity.get(
                "uri"
            )

            if not uri:

                return {
                    "success": False,
                    "message": (
                        "The album has no Spotify URI."
                    ),
                }

            sp.start_playback(
                device_id=device["id"],
                context_uri=uri,
            )

            return {
                "success": True,
                "type": "album",
                "name": entity.get("name"),
                "artist": entity.get("artist"),
                "uri": uri,
                "device": device.get("name"),
                "message": (
                    f"Playing {entity.get('name')} "
                    f"by {entity.get('artist')}."
                ),
            }

        return {
            "success": False,
            "message": (
                "Unsupported Spotify entity type: "
                f"{entity_type}"
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "message": (
                "Spotify playback failed: "
                f"{spotify_error_message(exc)}"
            ),
        }


# ============================================================
# MAIN PLAY
# ============================================================

def spotify_play(
    query: str,
) -> dict[str, Any]:

    query = clean_play_query(
        query
    )

    if not query:

        return {
            "success": False,
            "message": "What should I play?",
        }

    # --------------------------------------------------------
    # Search-result number
    # --------------------------------------------------------

    numbered = resolve_search_number(
        query
    )

    if numbered:

        return play_resolved_entity(
            numbered
        )

    # --------------------------------------------------------
    # Explicit playlist intent
    # --------------------------------------------------------

    playlist = resolve_playlist_from_phrase(
        query
    )

    if playlist:

        return spotify_play_playlist(
            playlist
        )

    # --------------------------------------------------------
    # Explicit music syntax
    # --------------------------------------------------------

    parsed = parse_music_query(
        query
    )

    parsed_type = parsed.get(
        "type"
    )

    if parsed_type == "track":

        entity = resolve_track(
            parsed.get(
                "query",
                "",
            ),
            parsed.get("artist"),
        )

        if entity:

            return play_resolved_entity(
                entity
            )

        artist_text = (
            f" by {parsed.get('artist')}"
            if parsed.get("artist")
            else ""
        )

        return {
            "success": False,
            "message": (
                f"I couldn't find the track "
                f"'{parsed.get('query', '')}'"
                f"{artist_text} on Spotify."
            ),
        }

    if parsed_type == "artist":

        entity = resolve_artist(
            parsed.get(
                "query",
                "",
            )
        )

        if entity:

            return play_resolved_entity(
                entity
            )

        return {
            "success": False,
            "message": (
                f"I couldn't find the artist "
                f"'{parsed.get('query', '')}' "
                "on Spotify."
            ),
        }

    if parsed_type == "album":

        entity = resolve_album(
            parsed.get(
                "query",
                "",
            ),
            parsed.get("artist"),
        )

        if entity:

            return play_resolved_entity(
                entity
            )

        return {
            "success": False,
            "message": (
                f"I couldn't find the album "
                f"'{parsed.get('query', '')}' "
                "on Spotify."
            ),
        }

    # --------------------------------------------------------
    # Bare music request
    # --------------------------------------------------------

    entity = resolve_music_query(
        query
    )

    if not entity:

        return {
            "success": False,
            "message": (
                f"I couldn't find anything on Spotify "
                f"for '{query}'."
            ),
        }

    return play_resolved_entity(
        entity
    )


# ============================================================
# PLAY / PAUSE / RESUME
# ============================================================

def spotify_play_pause(
    action: Optional[str] = None,
) -> dict[str, Any]:

    device = get_active_device()

    if not device:

        return {
            "success": False,
            "message": (
                "No available Spotify device was found."
            ),
        }

    try:

        sp = get_spotify()

        current = sp.current_playback()

        is_playing = bool(
            current
            and current.get(
                "is_playing"
            )
        )

        normalized_action = normalize_text(
            action or ""
        )

        if normalized_action in {
            "pause",
            "stop",
        }:

            sp.pause_playback(
                device_id=device["id"]
            )

            return {
                "success": True,
                "playing": False,
                "message": "Spotify paused.",
            }

        if normalized_action in {
            "play",
            "resume",
            "continue",
        }:

            sp.start_playback(
                device_id=device["id"]
            )

            return {
                "success": True,
                "playing": True,
                "message": "Spotify resumed.",
            }

        if is_playing:

            sp.pause_playback(
                device_id=device["id"]
            )

            return {
                "success": True,
                "playing": False,
                "message": "Spotify paused.",
            }

        sp.start_playback(
            device_id=device["id"]
        )

        return {
            "success": True,
            "playing": True,
            "message": "Spotify resumed.",
        }

    except Exception as exc:

        return {
            "success": False,
            "message": (
                "Spotify playback control failed: "
                f"{spotify_error_message(exc)}"
            ),
        }


# ============================================================
# CURRENT SONG
# ============================================================

def get_current_song() -> dict[str, Any]:

    try:

        # Automatically launch Spotify if necessary.
        startup = ensure_spotify_open(
            wait_for_device=True
        )

        if not startup.get("success"):

            return {
                "success": False,
                "message": startup.get(
                    "message",
                    "Could not open Spotify.",
                ),
            }

        sp = get_spotify()

        current = sp.current_playback()

        if not current:

            return {
                "success": True,
                "playing": False,
                "message": (
                    "Nothing is currently playing."
                ),
            }

        item = current.get(
            "item"
        )

        if not item:

            return {
                "success": True,
                "playing": bool(
                    current.get(
                        "is_playing",
                        False,
                    )
                ),
                "message": (
                    "Nothing is currently playing."
                ),
            }

        item_type = item.get(
            "type",
            "track",
        )

        if item_type != "track":

            return {
                "success": True,
                "playing": bool(
                    current.get(
                        "is_playing",
                        False,
                    )
                ),
                "type": item_type,
                "name": item.get("name"),
                "message": (
                    f"Currently playing "
                    f"{item.get('name')}."
                ),
            }

        artists = [
            artist.get(
                "name",
                "",
            )
            for artist in item.get(
                "artists",
                [],
            )
            if artist
        ]

        artist_text = ", ".join(
            artists
        )

        device = current.get(
            "device",
            {},
        ) or {}

        return {
            "success": True,
            "playing": bool(
                current.get(
                    "is_playing",
                    False,
                )
            ),
            "name": item.get("name"),
            "artist": artist_text,
            "album": item.get(
                "album",
                {},
            ).get("name"),
            "uri": item.get("uri"),
            "device": device.get("name"),
            "device_id": device.get("id"),
            "progress_ms": current.get(
                "progress_ms"
            ),
            "duration_ms": item.get(
                "duration_ms"
            ),
            "shuffle": current.get(
                "shuffle_state"
            ),
            "repeat": current.get(
                "repeat_state"
            ),
            "message": (
                f"{item.get('name')} by "
                f"{artist_text}"
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "message": (
                f"Could not get current song: "
                f"{spotify_error_message(exc)}"
            ),
        }


# ============================================================
# NEXT / PREVIOUS
# ============================================================

def spotify_next() -> dict[str, Any]:

    device = get_active_device()

    if not device:

        return {
            "success": False,
            "message": (
                "No available Spotify device was found."
            ),
        }

    try:

        sp = get_spotify()

        sp.next_track(
            device_id=device["id"]
        )

        return {
            "success": True,
            "message": (
                "Skipped to the next track."
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "message": (
                f"Could not skip track: "
                f"{spotify_error_message(exc)}"
            ),
        }


def spotify_previous() -> dict[str, Any]:

    device = get_active_device()

    if not device:

        return {
            "success": False,
            "message": (
                "No available Spotify device was found."
            ),
        }

    try:

        sp = get_spotify()

        sp.previous_track(
            device_id=device["id"]
        )

        return {
            "success": True,
            "message": (
                "Went to the previous track."
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "message": (
                f"Could not go to previous track: "
                f"{spotify_error_message(exc)}"
            ),
        }


# ============================================================
# USER PLAYLISTS
# ============================================================

def get_playlists(
    limit: int = 50,
) -> dict[str, Any]:

    try:

        sp = get_spotify()

        limit = max(
            1,
            int(limit),
        )

        playlists = []
        offset = 0

        while len(playlists) < limit:

            batch_size = min(
                50,
                limit - len(playlists),
            )

            response = sp.current_user_playlists(
                limit=batch_size,
                offset=offset,
            )

            items = response.get(
                "items",
                [],
            )

            if not items:
                break

            for item in items:

                if not item:
                    continue

                playlists.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "uri": item.get("uri"),
                    "url": item.get(
                        "external_urls",
                        {},
                    ).get("spotify"),
                    "tracks": item.get(
                        "tracks",
                        {},
                    ).get("total", 0),
                })

            if not response.get(
                "next"
            ):
                break

            offset += len(items)

        return {
            "success": True,
            "playlists": playlists[:limit],
        }

    except Exception as exc:

        return {
            "success": False,
            "playlists": [],
            "message": (
                f"Could not get playlists: "
                f"{spotify_error_message(exc)}"
            ),
        }


# ============================================================
# CLI
# ============================================================

def run_cli() -> None:

    print("=" * 60)
    print("ZOE SPOTIFY")
    print("=" * 60)
    print()

    print("Commands:")
    print("  play <song / artist / album / playlist>")
    print("  pause")
    print("  resume")
    print("  next")
    print("  previous")
    print("  current")
    print("  search <query>")
    print("  playlists")
    print("  devices")
    print("  open")
    print("  quit")
    print()

    while True:

        try:

            command = input(
                "Spotify > "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print()
            break

        if not command:
            continue

        normalized = normalize_text(
            command
        )

        if normalized in {
            "quit",
            "exit",
            "q",
        }:
            break

        # ----------------------------------------------------
        # Open Spotify
        # ----------------------------------------------------

        if normalized in {
            "open",
            "open spotify",
            "start spotify",
        }:

            result = open_spotify()

            print(
                result.get(
                    "message",
                    result,
                )
            )

            continue

        # ----------------------------------------------------
        # Play
        # ----------------------------------------------------

        if normalized.startswith(
            "play "
        ):

            result = spotify_play(
                command[5:].strip()
            )

            print(
                result.get(
                    "message",
                    result,
                )
            )

            continue

        # ----------------------------------------------------
        # Pause
        # ----------------------------------------------------

        if normalized in {
            "pause",
            "spotify pause",
        }:

            result = spotify_play_pause(
                "pause"
            )

            print(
                result.get(
                    "message",
                    result,
                )
            )

            continue

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        if normalized in {
            "resume",
            "continue",
            "play",
            "spotify resume",
        }:

            result = spotify_play_pause(
                "resume"
            )

            print(
                result.get(
                    "message",
                    result,
                )
            )

            continue

        # ----------------------------------------------------
        # Toggle
        # ----------------------------------------------------

        if normalized in {
            "toggle",
            "play pause",
            "pause play",
        }:

            result = spotify_play_pause()

            print(
                result.get(
                    "message",
                    result,
                )
            )

            continue

        # ----------------------------------------------------
        # Next
        # ----------------------------------------------------

        if normalized in {
            "next",
            "skip",
            "skip next",
            "next track",
        }:

            result = spotify_next()

            print(
                result.get(
                    "message",
                    result,
                )
            )

            continue

        # ----------------------------------------------------
        # Previous
        # ----------------------------------------------------

        if normalized in {
            "previous",
            "prev",
            "back",
            "previous track",
        }:

            result = spotify_previous()

            print(
                result.get(
                    "message",
                    result,
                )
            )

            continue

        # ----------------------------------------------------
        # Current
        # ----------------------------------------------------

        if normalized in {
            "current",
            "current song",
            "what is playing",
            "whats playing",
            "what's playing",
        }:

            result = get_current_song()

            print(
                result.get(
                    "message",
                    result,
                )
            )

            continue

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        if normalized.startswith(
            "search "
        ):

            result = spotify_search(
                command[7:].strip()
            )

            print()
            print(
                format_search_results(
                    result
                )
            )
            print()

            continue

        # ----------------------------------------------------
        # Playlists
        # ----------------------------------------------------

        if normalized in {
            "playlists",
            "my playlists",
        }:

            result = get_playlists()

            if not result.get(
                "success"
            ):

                print(
                    result.get(
                        "message",
                        result,
                    )
                )

                continue

            for playlist in result[
                "playlists"
            ]:

                print(
                    f"- {playlist.get('name')} "
                    f"({playlist.get('tracks', 0)} tracks)"
                )

            continue

        # ----------------------------------------------------
        # Devices
        # ----------------------------------------------------

        if normalized in {
            "devices",
            "spotify devices",
        }:

            result = get_devices()

            if not result.get(
                "success"
            ):

                print(
                    result.get(
                        "message",
                        result,
                    )
                )

                continue

            for device in result[
                "devices"
            ]:

                active = (
                    "ACTIVE"
                    if device.get(
                        "is_active"
                    )
                    else ""
                )

                print(
                    f"- {device.get('name')} "
                    f"[{device.get('type')}] "
                    f"{active}"
                )

            continue

        print(
            "Unknown command. Try: "
            "play <song/artist/album>, pause, "
            "resume, next, previous, current, "
            "search <query>"
        )


# ============================================================
# SEARCH RESULT FORMATTER
# ============================================================

def format_search_results(
    search_result: dict[str, Any],
) -> str:

    results = search_result.get(
        "results",
        [],
    )

    if not results:

        return (
            "I couldn't find anything on Spotify "
            f"for '{search_result.get('query', '')}'."
        )

    lines = []

    for index, item in enumerate(
        results,
        start=1,
    ):

        item_type = item.get(
            "type",
            "unknown",
        )

        name = item.get(
            "name",
            "Unknown",
        )

        if item_type == "track":

            lines.append(
                f"{index}. {name} — "
                f"{item.get('artist', 'Unknown artist')}"
            )

        elif item_type == "artist":

            lines.append(
                f"{index}. {name} — Artist"
            )

        elif item_type == "album":

            lines.append(
                f"{index}. {name} — "
                f"{item.get('artist', 'Unknown artist')} "
                f"— Album"
            )

        elif item_type == "playlist":

            lines.append(
                f"{index}. {name} — Playlist"
            )

        else:

            lines.append(
                f"{index}. {name}"
            )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_cli()