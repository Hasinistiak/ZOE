from __future__ import annotations

import os
import platform
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()

SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI",
    "http://127.0.0.1:8888/callback",
).strip()

SPOTIFY_CACHE_PATH = os.getenv(
    "SPOTIFY_CACHE_PATH",
    ".spotify_cache",
).strip()

SPOTIFY_DEVICE_WAIT = float(
    os.getenv("SPOTIFY_DEVICE_WAIT", "8")
)

SPOTIFY_DEVICE_POLL = float(
    os.getenv("SPOTIFY_DEVICE_POLL", "0.5")
)

SPOTIFY_SEARCH_LIMIT = max(
    1,
    min(
        int(os.getenv("SPOTIFY_SEARCH_LIMIT", "10")),
        50,
    ),
)

SCOPE = (
    "user-read-playback-state "
    "user-read-currently-playing "
    "user-modify-playback-state "
    "user-library-read"
)


# ============================================================================
# CONFIGURED PLAYLISTS
#
# IMPORTANT:
# The Spotify agent chooses the number.
# This module simply executes that number.
# ============================================================================

PLAYLISTS: dict[int, dict[str, str]] = {
    1: {
        "name": "Just AM",
        "uri": "spotify:playlist:0SPRVgVOz5ujsm1NVXy00y",
    },
    2: {
        "name": "Just OR",
        "uri": "spotify:playlist:26827blLrCb7QC4I6qEZzd",
    },
    3: {
        "name": "how come we never even dated",
        "uri": "spotify:playlist:6fSw4Sd2hPR1LjP0nRjUyP",
    },
    4: {
        "name": "Slower Turner",
        "uri": "spotify:playlist:1VsFBx9TrL8dmCd6jTxMAB",
    },
    5: {
        "name": "Property Of Peter Parker",
        "uri": "spotify:playlist:6IGubGbQPkDZIF7BL6y00N",
    },
    6: {
        "name": "Even Dead I Am The Hero",
        "uri": "spotify:playlist:0xtXuuhYKFpOzQ5qi2hGZY",
    },
}


# ============================================================================
# STATE
# ============================================================================

_spotify: Optional[spotipy.Spotify] = None
_last_search_results: list[dict[str, Any]] = []


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def normalize_text(value: Any) -> str:
    """
    Lightweight text normalization.

    This is intentionally NOT used to interpret user intent.
    It is only useful for comparisons / formatting.
    """
    if value is None:
        return ""

    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


def _clean_error(error: Exception) -> str:
    message = str(error).strip()

    if not message:
        return "Spotify returned an unknown error."

    lowered = message.lower()

    if "premium" in lowered:
        return "Spotify Premium is required for playback control."

    if "no active device" in lowered:
        return "No active Spotify playback device was found."

    if "403" in lowered:
        return "Spotify denied the playback request."

    if "401" in lowered:
        return "Spotify authentication has expired. Please reconnect Spotify."

    if "404" in lowered:
        return "Spotify could not find the requested item."

    if "429" in lowered:
        return "Spotify rate limit reached. Try again shortly."

    return f"Spotify error: {message}"


def _as_playlist_number(value: int | str) -> Optional[int]:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None

    return number if number in PLAYLISTS else None


# ============================================================================
# SPOTIFY CLIENT
# ============================================================================

def get_spotify() -> spotipy.Spotify:
    """
    Return the lazily-created Spotify client.
    """
    global _spotify

    if _spotify is not None:
        return _spotify

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

    _spotify = spotipy.Spotify(
        auth_manager=auth_manager,
        requests_timeout=10,
        retries=2,
    )

    return _spotify


# ============================================================================
# SPOTIFY DESKTOP
# ============================================================================

def is_spotify_running() -> bool:
    """
    Check whether the Spotify desktop process appears to be running.
    """

    system = platform.system()

    try:
        if system == "Windows":
            result = subprocess.run(
                [
                    "tasklist",
                    "/FI",
                    "IMAGENAME eq Spotify.exe",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )

            return "Spotify.exe" in result.stdout

        if system == "Darwin":
            result = subprocess.run(
                [
                    "pgrep",
                    "-x",
                    "Spotify",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )

            return result.returncode == 0

        result = subprocess.run(
            ["pgrep", "-x", "spotify"],
            capture_output=True,
            text=True,
            timeout=3,
        )

        return result.returncode == 0

    except Exception:
        return False


def open_spotify() -> bool:
    """
    Launch Spotify desktop if possible.
    """

    system = platform.system()

    try:
        if system == "Windows":
            candidates = [
                Path(os.getenv("APPDATA", ""))
                / "Spotify"
                / "Spotify.exe",
                Path(os.getenv("LOCALAPPDATA", ""))
                / "Spotify"
                / "Spotify.exe",
            ]

            for executable in candidates:
                if executable.exists():
                    subprocess.Popen(
                        [str(executable)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True

            subprocess.Popen(
                ["spotify:"],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        if system == "Darwin":
            subprocess.Popen(
                ["open", "-a", "Spotify"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True

        subprocess.Popen(
            ["spotify"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return True

    except Exception:
        try:
            webbrowser.open("spotify:")
            return True
        except Exception:
            return False


def ensure_spotify_open(
    wait_for_device: bool = False,
) -> bool:
    """
    Ensure Spotify desktop is running.

    When wait_for_device=True, launch Spotify if necessary and
    poll Spotify Connect until a device appears or the timeout expires.
    """
    if not is_spotify_running():
        open_spotify()

    if not wait_for_device:
        return True

    deadline = time.monotonic() + SPOTIFY_DEVICE_WAIT

    while time.monotonic() < deadline:
        try:
            if get_devices():
                return True
        except Exception:
            pass

        # If Spotify failed to start or exited, try launching it again.
        if not is_spotify_running():
            open_spotify()

        time.sleep(min(SPOTIFY_DEVICE_POLL, max(
            0.0, deadline - time.monotonic()
        )))

    return False


def _ensure_playback_device() -> str:
    """
    Find a usable Spotify device.

    If none is available, launch Spotify desktop on the PC,
    wait for Spotify Connect, and retry device discovery.
    """
    device = get_active_device()

    if device and device.get("id"):
        return device["id"]

    # No device: explicitly launch Spotify on the PC.
    open_spotify()

    deadline = time.monotonic() + SPOTIFY_DEVICE_WAIT

    while time.monotonic() < deadline:
        try:
            devices = get_devices()

            if devices:
                # Prefer the active device, then the PC, then any device.
                for candidate in devices:
                    if candidate.get("is_active") and candidate.get("id"):
                        return candidate["id"]

                for candidate in devices:
                    if (
                        candidate.get("type", "").lower() == "computer"
                        and candidate.get("id")
                    ):
                        return candidate["id"]

                for candidate in devices:
                    if candidate.get("id"):
                        return candidate["id"]

        except Exception:
            pass

        time.sleep(SPOTIFY_DEVICE_POLL)

    raise RuntimeError(
        "No Spotify playback device became available. "
        "Spotify was opened on the PC, but Connect did not expose "
        "a device in time. Open Spotify, sign in, and try again."
    )


def open_spotify() -> bool:
    """
    Launch Spotify desktop on Windows, macOS, or Linux.
    """
    system = platform.system()

    try:
        if system == "Windows":
            # Try the registered Spotify URI handler first.
            try:
                os.startfile("spotify:")
                return True
            except OSError:
                pass

            candidates = [
                Path(os.getenv("APPDATA", ""))
                / "Spotify" / "Spotify.exe",
                Path(os.getenv("LOCALAPPDATA", ""))
                / "Spotify" / "Spotify.exe",
            ]

            for executable in candidates:
                if executable.is_file():
                    subprocess.Popen(
                        [str(executable)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True

            # Fallback for installations registered with Windows.
            subprocess.Popen(
                ["cmd", "/c", "start", "", "spotify:"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True

        if system == "Darwin":
            subprocess.Popen(
                ["open", "-a", "Spotify"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True

        subprocess.Popen(
            ["spotify"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True

    except (OSError, subprocess.SubprocessError):
        try:
            return bool(webbrowser.open("spotify:"))
        except Exception:
            return False


# ============================================================================
# DEVICES
# ============================================================================

def get_devices() -> list[dict[str, Any]]:
    """
    Return available Spotify devices.
    """

    try:
        spotify = get_spotify()
        response = spotify.devices()

        return response.get("devices", [])

    except Exception as error:
        raise RuntimeError(_clean_error(error)) from error


def get_active_device() -> Optional[dict[str, Any]]:
    """
    Select the best playback device.

    Priority:
      1. Current active device
      2. Computer device
      3. First available device
    """

    devices = get_devices()

    if not devices:
        return None

    for device in devices:
        if device.get("is_active"):
            return device

    for device in devices:
        if device.get("type", "").lower() == "computer":
            return device

    return devices[0]


def _get_device_id() -> Optional[str]:
    device = get_active_device()

    if not device:
        return None

    return device.get("id")


def _ensure_playback_device() -> str:
    """
    Make sure Spotify has a usable device and return its ID.
    """

    device = get_active_device()

    if device:
        device_id = device.get("id")

        if device_id:
            return device_id

    ensure_spotify_open(wait_for_device=True)

    device = get_active_device()

    if not device:
        raise RuntimeError(
            "No Spotify playback device is available. "
            "Open Spotify and try again."
        )

    device_id = device.get("id")

    if not device_id:
        raise RuntimeError(
            "Spotify reported a device without a usable device ID."
        )

    return device_id


# ============================================================================
# PLAYLISTS
# ============================================================================

def get_configured_playlists() -> list[dict[str, Any]]:
    """
    Return the numbered playlists that the Spotify agent can choose from.
    """

    return [
        {
            "number": number,
            "name": data["name"],
            "uri": data["uri"],
        }
        for number, data in PLAYLISTS.items()
    ]


def get_playlist(number: int | str) -> Optional[dict[str, Any]]:
    """
    Get one configured playlist by number.

    Example:
        get_playlist(5)
    """

    playlist_number = _as_playlist_number(number)

    if playlist_number is None:
        return None

    data = PLAYLISTS[playlist_number]

    return {
        "number": playlist_number,
        "name": data["name"],
        "uri": data["uri"],
    }


def spotify_play_playlist(number: int | str) -> dict[str, Any]:
    """
    Play one of the configured playlists.

    The caller MUST provide the playlist number.

    Example:
        spotify_play_playlist(5)
    """

    playlist_number = _as_playlist_number(number)

    if playlist_number is None:
        valid = ", ".join(
            str(number)
            for number in PLAYLISTS
        )

        return {
            "success": False,
            "action": "play_playlist",
            "error": f"Invalid playlist number. Valid numbers: {valid}.",
        }

    playlist = PLAYLISTS[playlist_number]

    try:
        spotify = get_spotify()
        device_id = _ensure_playback_device()

        spotify.start_playback(
            device_id=device_id,
            context_uri=playlist["uri"],
        )

        return {
            "success": True,
            "action": "play_playlist",
            "playlist_number": playlist_number,
            "playlist_name": playlist["name"],
            "uri": playlist["uri"],
        }

    except Exception as error:
        return {
            "success": False,
            "action": "play_playlist",
            "playlist_number": playlist_number,
            "playlist_name": playlist["name"],
            "error": _clean_error(error),
        }


# ============================================================================
# SEARCH
# ============================================================================

def spotify_search(
    query: str,
    search_type: str = "track",
    limit: int = SPOTIFY_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    """
    Search Spotify.

    search_type:
        track
        artist
        album
        playlist

    This function performs API search only.
    It does not interpret the user's natural-language request.
    """

    global _last_search_results

    query = str(query).strip()

    if not query:
        return []

    allowed_types = {
        "track",
        "artist",
        "album",
        "playlist",
    }

    search_type = search_type.lower().strip()

    if search_type not in allowed_types:
        raise ValueError(
            f"Invalid search type: {search_type}. "
            f"Use one of: {', '.join(sorted(allowed_types))}."
        )

    limit = max(1, min(int(limit), 50))

    try:
        spotify = get_spotify()

        response = spotify.search(
            q=query,
            type=search_type,
            limit=limit,
        )

        items = response.get(
            f"{search_type}s",
            {},
        ).get("items", [])

        results: list[dict[str, Any]] = []

        for index, item in enumerate(items, start=1):
            if search_type == "track":
                results.append(
                    {
                        "number": index,
                        "type": "track",
                        "id": item.get("id"),
                        "uri": item.get("uri"),
                        "name": item.get("name"),
                        "artist": ", ".join(
                            artist.get("name", "")
                            for artist in item.get("artists", [])
                        ),
                        "album": (
                            item.get("album", {}) or {}
                        ).get("name"),
                        "duration_ms": item.get(
                            "duration_ms"
                        ),
                        "is_playable": item.get(
                            "is_playable",
                            True,
                        ),
                    }
                )

            elif search_type == "artist":
                results.append(
                    {
                        "number": index,
                        "type": "artist",
                        "id": item.get("id"),
                        "uri": item.get("uri"),
                        "name": item.get("name"),
                        "genres": item.get("genres", []),
                        "followers": (
                            item.get("followers", {}) or {}
                        ).get("total"),
                    }
                )

            elif search_type == "album":
                results.append(
                    {
                        "number": index,
                        "type": "album",
                        "id": item.get("id"),
                        "uri": item.get("uri"),
                        "name": item.get("name"),
                        "artist": ", ".join(
                            artist.get("name", "")
                            for artist in item.get("artists", [])
                        ),
                        "release_date": item.get(
                            "release_date"
                        ),
                    }
                )

            elif search_type == "playlist":
                results.append(
                    {
                        "number": index,
                        "type": "playlist",
                        "id": item.get("id"),
                        "uri": item.get("uri"),
                        "name": item.get("name"),
                        "owner": (
                            item.get("owner", {}) or {}
                        ).get("display_name"),
                        "tracks_total": (
                            item.get("tracks", {}) or {}
                        ).get("total"),
                    }
                )

        _last_search_results = results

        return results

    except Exception as error:
        raise RuntimeError(_clean_error(error)) from error


def search_tracks(
    query: str,
    limit: int = SPOTIFY_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    return spotify_search(
        query,
        search_type="track",
        limit=limit,
    )


def search_artists(
    query: str,
    limit: int = SPOTIFY_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    return spotify_search(
        query,
        search_type="artist",
        limit=limit,
    )


def search_albums(
    query: str,
    limit: int = SPOTIFY_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    return spotify_search(
        query,
        search_type="album",
        limit=limit,
    )


def search_playlists(
    query: str,
    limit: int = SPOTIFY_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    return spotify_search(
        query,
        search_type="playlist",
        limit=limit,
    )


def resolve_search_number(
    number: int | str,
) -> Optional[dict[str, Any]]:
    """
    Resolve a numbered result from the most recent search.

    Useful for:
        search_tracks("blinding lights")
        resolve_search_number(1)
    """

    try:
        index = int(str(number).strip())
    except (TypeError, ValueError):
        return None

    if index < 1 or index > len(_last_search_results):
        return None

    return _last_search_results[index - 1]


# ============================================================================
# TRACK RESOLUTION
# ============================================================================

def resolve_track(
    name: str,
    artist: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Resolve a track from an explicit track name.

    The agent supplies the track name and optional artist.
    """

    name = str(name).strip()

    if not name:
        return None

    query = name

    if artist:
        query = f"track:{name} artist:{artist}"

    results = search_tracks(
        query,
        limit=SPOTIFY_SEARCH_LIMIT,
    )

    if not results:
        return None

    if artist:
        target_artist = normalize_text(artist)

        for result in results:
            result_artist = normalize_text(
                result.get("artist", "")
            )

            if target_artist in result_artist:
                return result

    target_name = normalize_text(name)

    for result in results:
        if normalize_text(result.get("name")) == target_name:
            return result

    return results[0]


def resolve_artist(
    name: str,
) -> Optional[dict[str, Any]]:
    """
    Resolve an artist from an explicit artist name.
    """

    name = str(name).strip()

    if not name:
        return None

    results = search_artists(
        name,
        limit=SPOTIFY_SEARCH_LIMIT,
    )

    if not results:
        return None

    target = normalize_text(name)

    for result in results:
        if normalize_text(result.get("name")) == target:
            return result

    return results[0]


def resolve_album(
    name: str,
    artist: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Resolve an album from an explicit album name.
    """

    name = str(name).strip()

    if not name:
        return None

    query = name

    if artist:
        query = f"album:{name} artist:{artist}"

    results = search_albums(
        query,
        limit=SPOTIFY_SEARCH_LIMIT,
    )

    if not results:
        return None

    if artist:
        target_artist = normalize_text(artist)

        for result in results:
            result_artist = normalize_text(
                result.get("artist", "")
            )

            if target_artist in result_artist:
                return result

    target_name = normalize_text(name)

    for result in results:
        if normalize_text(result.get("name")) == target_name:
            return result

    return results[0]


# ============================================================================
# PLAYBACK
# ============================================================================

def spotify_play_uri(
    uri: str,
    *,
    context: bool = False,
) -> dict[str, Any]:
    """
    Low-level playback function.

    uri can be:
        spotify:track:...
        spotify:artist:...
        spotify:album:...
        spotify:playlist:...
    """

    uri = str(uri).strip()

    if not uri:
        return {
            "success": False,
            "error": "No Spotify URI was supplied.",
        }

    try:
        spotify = get_spotify()
        device_id = _ensure_playback_device()

        if context:
            spotify.start_playback(
                device_id=device_id,
                context_uri=uri,
            )
        else:
            spotify.start_playback(
                device_id=device_id,
                uris=[uri],
            )

        return {
            "success": True,
            "uri": uri,
        }

    except Exception as error:
        return {
            "success": False,
            "uri": uri,
            "error": _clean_error(error),
        }


def spotify_play_track(
    name: str,
    artist: Optional[str] = None,
) -> dict[str, Any]:
    """
    Play an explicit track.

    Example:
        spotify_play_track("vampire", "Olivia Rodrigo")
        spotify_play_track("Blinding Lights")
    """

    track = resolve_track(
        name=name,
        artist=artist,
    )

    if not track:
        return {
            "success": False,
            "action": "play_track",
            "track": name,
            "artist": artist,
            "error": "Track not found.",
        }

    result = spotify_play_uri(
        track["uri"],
        context=False,
    )

    result.update(
        {
            "action": "play_track",
            "track": track.get("name"),
            "artist": track.get("artist"),
            "album": track.get("album"),
        }
    )

    return result


def spotify_play_artist(
    name: str,
) -> dict[str, Any]:
    """
    Start an artist's playback.
    """

    artist = resolve_artist(name)

    if not artist:
        return {
            "success": False,
            "action": "play_artist",
            "artist": name,
            "error": "Artist not found.",
        }

    result = spotify_play_uri(
        artist["uri"],
        context=True,
    )

    result.update(
        {
            "action": "play_artist",
            "artist": artist.get("name"),
        }
    )

    return result


def spotify_play_album(
    name: str,
    artist: Optional[str] = None,
) -> dict[str, Any]:
    """
    Start an album.
    """

    album = resolve_album(
        name=name,
        artist=artist,
    )

    if not album:
        return {
            "success": False,
            "action": "play_album",
            "album": name,
            "artist": artist,
            "error": "Album not found.",
        }

    result = spotify_play_uri(
        album["uri"],
        context=True,
    )

    result.update(
        {
            "action": "play_album",
            "album": album.get("name"),
            "artist": album.get("artist"),
        }
    )

    return result


# ============================================================================
# PLAYBACK CONTROLS
# ============================================================================

def spotify_pause() -> dict[str, Any]:
    try:
        spotify = get_spotify()
        device_id = _get_device_id()

        spotify.pause_playback(
            device_id=device_id,
        )

        return {
            "success": True,
            "action": "pause",
        }

    except Exception as error:
        return {
            "success": False,
            "action": "pause",
            "error": _clean_error(error),
        }


def spotify_resume() -> dict[str, Any]:
    try:
        spotify = get_spotify()
        device_id = _ensure_playback_device()

        spotify.start_playback(
            device_id=device_id,
        )

        return {
            "success": True,
            "action": "resume",
        }

    except Exception as error:
        return {
            "success": False,
            "action": "resume",
            "error": _clean_error(error),
        }


def spotify_play_pause(
    action: str,
) -> dict[str, Any]:
    """
    Explicit playback control.

    action:
        play / resume
        pause
    """

    action = normalize_text(action)

    if action in {"pause", "stop"}:
        return spotify_pause()

    if action in {"play", "resume"}:
        return spotify_resume()

    return {
        "success": False,
        "error": (
            "Invalid playback action. "
            "Use 'play', 'resume', or 'pause'."
        ),
    }


def spotify_next() -> dict[str, Any]:
    try:
        spotify = get_spotify()
        device_id = _get_device_id()

        spotify.next_track(
            device_id=device_id,
        )

        return {
            "success": True,
            "action": "next",
        }

    except Exception as error:
        return {
            "success": False,
            "action": "next",
            "error": _clean_error(error),
        }


def spotify_previous() -> dict[str, Any]:
    try:
        spotify = get_spotify()
        device_id = _get_device_id()

        spotify.previous_track(
            device_id=device_id,
        )

        return {
            "success": True,
            "action": "previous",
        }

    except Exception as error:
        return {
            "success": False,
            "action": "previous",
            "error": _clean_error(error),
        }


# ============================================================================
# CURRENT PLAYBACK
# ============================================================================

def get_current_song() -> Optional[dict[str, Any]]:
    """
    Return the currently playing track.
    """

    try:
        spotify = get_spotify()

        current = spotify.current_playback()

        if not current:
            return None

        item = current.get("item")

        if not item:
            return None

        artists = item.get("artists", [])

        return {
            "is_playing": current.get("is_playing", False),
            "progress_ms": current.get("progress_ms"),
            "duration_ms": item.get("duration_ms"),
            "name": item.get("name"),
            "artist": ", ".join(
                artist.get("name", "")
                for artist in artists
            ),
            "album": (
                item.get("album", {}) or {}
            ).get("name"),
            "uri": item.get("uri"),
            "device": (
                current.get("device", {}) or {}
            ).get("name"),
        }

    except Exception as error:
        raise RuntimeError(_clean_error(error)) from error


# ============================================================================
# USER PLAYLISTS
# ============================================================================

def get_playlists(
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Return the user's Spotify playlists.

    This is separate from the numbered configured playlists.
    """

    try:
        spotify = get_spotify()

        response = spotify.current_user_playlists(
            limit=max(1, min(int(limit), 50)),
        )

        results = []

        for playlist in response.get("items", []):
            results.append(
                {
                    "id": playlist.get("id"),
                    "uri": playlist.get("uri"),
                    "name": playlist.get("name"),
                    "owner": (
                        playlist.get("owner", {}) or {}
                    ).get("display_name"),
                    "tracks_total": (
                        playlist.get("tracks", {}) or {}
                    ).get("total"),
                    "public": playlist.get("public"),
                }
            )

        return results

    except Exception as error:
        raise RuntimeError(_clean_error(error)) from error


# ============================================================================
# DIRECT PLAYLIST URI
# ============================================================================

def spotify_play_playlist_uri(
    uri: str,
    name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Play any explicit Spotify playlist URI.

    This is useful if the agent searches Spotify and gets a playlist URI.
    """

    result = spotify_play_uri(
        uri,
        context=True,
    )

    result.update(
        {
            "action": "play_playlist_uri",
            "playlist_name": name,
        }
    )

    return result


# ============================================================================
# SIMPLE AGENT-FACING API
# ============================================================================

def spotify_status() -> dict[str, Any]:
    """
    Compact status payload suitable for the ZOE agent.
    """

    try:
        current = get_current_song()

        return {
            "success": True,
            "playing": bool(
                current and current.get("is_playing")
            ),
            "current": current,
        }

    except Exception as error:
        return {
            "success": False,
            "error": _clean_error(error),
        }


# ============================================================================
# CLI
# ============================================================================

def _print_result(result: Any) -> None:
    if result is None:
        print("No result.")
        return

    if isinstance(result, list):
        if not result:
            print("No results.")
            return

        for item in result:
            number = item.get("number", "")
            name = item.get("name", "")
            artist = item.get("artist")

            if artist:
                print(
                    f"{number}. {name} — {artist}"
                )
            else:
                print(
                    f"{number}. {name}"
                )

        return

    print(result)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ZOE Spotify tools"
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # ---------------------------------------------------------------------
    # playlists
    # ---------------------------------------------------------------------

    subparsers.add_parser(
        "playlists",
        help="Show configured playlists",
    )

    # ---------------------------------------------------------------------
    # play-playlist
    # ---------------------------------------------------------------------

    playlist_parser = subparsers.add_parser(
        "play-playlist",
        help="Play configured playlist by number",
    )

    playlist_parser.add_argument(
        "number",
        type=int,
    )

    # ---------------------------------------------------------------------
    # search
    # ---------------------------------------------------------------------

    search_parser = subparsers.add_parser(
        "search",
        help="Search Spotify",
    )

    search_parser.add_argument(
        "query",
    )

    search_parser.add_argument(
        "--type",
        choices=[
            "track",
            "artist",
            "album",
            "playlist",
        ],
        default="track",
    )

    # ---------------------------------------------------------------------
    # play-track
    # ---------------------------------------------------------------------

    track_parser = subparsers.add_parser(
        "play-track",
        help="Play a track",
    )

    track_parser.add_argument(
        "name",
    )

    track_parser.add_argument(
        "--artist",
        default=None,
    )

    # ---------------------------------------------------------------------
    # play-artist
    # ---------------------------------------------------------------------

    artist_parser = subparsers.add_parser(
        "play-artist",
        help="Play an artist",
    )

    artist_parser.add_argument(
        "name",
    )

    # ---------------------------------------------------------------------
    # play-album
    # ---------------------------------------------------------------------

    album_parser = subparsers.add_parser(
        "play-album",
        help="Play an album",
    )

    album_parser.add_argument(
        "name",
    )

    album_parser.add_argument(
        "--artist",
        default=None,
    )

    # ---------------------------------------------------------------------
    # pause
    # ---------------------------------------------------------------------

    subparsers.add_parser(
        "pause",
        help="Pause playback",
    )

    # ---------------------------------------------------------------------
    # resume
    # ---------------------------------------------------------------------

    subparsers.add_parser(
        "resume",
        help="Resume playback",
    )

    # ---------------------------------------------------------------------
    # next
    # ---------------------------------------------------------------------

    subparsers.add_parser(
        "next",
        help="Skip to next track",
    )

    # ---------------------------------------------------------------------
    # previous
    # ---------------------------------------------------------------------

    subparsers.add_parser(
        "previous",
        help="Go to previous track",
    )

    # ---------------------------------------------------------------------
    # current
    # ---------------------------------------------------------------------

    subparsers.add_parser(
        "current",
        help="Show current song",
    )

    # ---------------------------------------------------------------------
    # devices
    # ---------------------------------------------------------------------

    subparsers.add_parser(
        "devices",
        help="Show Spotify devices",
    )

    args = parser.parse_args()

    if args.command == "playlists":
        _print_result(
            get_configured_playlists()
        )
        return

    if args.command == "play-playlist":
        _print_result(
            spotify_play_playlist(args.number)
        )
        return

    if args.command == "search":
        _print_result(
            spotify_search(
                args.query,
                args.type,
            )
        )
        return

    if args.command == "play-track":
        _print_result(
            spotify_play_track(
                args.name,
                args.artist,
            )
        )
        return

    if args.command == "play-artist":
        _print_result(
            spotify_play_artist(args.name)
        )
        return

    if args.command == "play-album":
        _print_result(
            spotify_play_album(
                args.name,
                args.artist,
            )
        )
        return

    if args.command == "pause":
        _print_result(
            spotify_pause()
        )
        return

    if args.command == "resume":
        _print_result(
            spotify_resume()
        )
        return

    if args.command == "next":
        _print_result(
            spotify_next()
        )
        return

    if args.command == "previous":
        _print_result(
            spotify_previous()
        )
        return

    if args.command == "current":
        _print_result(
            get_current_song()
        )
        return

    if args.command == "devices":
        _print_result(
            get_devices()
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()