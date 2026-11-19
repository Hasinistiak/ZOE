from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

# ============================================================
# ZOE BROWSER (Linux, Helium-first)
#
# Opens links directly in the machine's browser. The runtime
# calls open_links() with whatever URLs the brain returned;
# nothing is sent to the frontend.
#
# Env vars (all optional):
#   ZOE_BROWSER                    Browser command, optionally with args.
#                                  e.g. "helium-browser",
#                                  "/home/pi/Applications/helium.AppImage",
#                                  "chromium --ozone-platform=wayland".
#                                  Empty = auto-detect (Helium first).
#   ZOE_BROWSER_NEW                0 = reuse window, 1 = new window,
#                                  2 = new tab (default).
#   ZOE_BROWSER_OPEN_DELAY_SECONDS Pause between links (default 0.4).
#   ZOE_BROWSER_MAX_LINKS          Max links per call, 0 = unlimited.
#   ZOE_DISPLAY                    X display fallback (default ":0").
# ============================================================

logger = logging.getLogger("zoe.browser")

# Checked in order. Helium is Chromium-based, so Chromium flags apply.
# Debian/Pi OS Bookworm ships "chromium"; older releases use "chromium-browser".
AUTO_BROWSERS = (
    "helium-browser",
    "helium",
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "firefox",
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


BROWSER_CMD = os.getenv("ZOE_BROWSER", "").strip()
OPEN_NEW = _env_int("ZOE_BROWSER_NEW", 2)
OPEN_DELAY_SECONDS = _env_float("ZOE_BROWSER_OPEN_DELAY_SECONDS", 0.4)
MAX_LINKS_PER_CALL = _env_int("ZOE_BROWSER_MAX_LINKS", 0)


def _prepare_session_env() -> None:
    """Make a service/SSH-launched process able to reach the desktop session."""
    if not sys.platform.startswith("linux"):
        return

    uid = os.getuid()
    runtime = Path(os.getenv("XDG_RUNTIME_DIR") or f"/run/user/{uid}")

    if runtime.is_dir():
        os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime))

        bus = runtime / "bus"
        if bus.exists():
            os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus}")

    if not os.getenv("WAYLAND_DISPLAY"):
        for sock in ("wayland-0", "wayland-1"):
            if (runtime / sock).exists():
                os.environ["WAYLAND_DISPLAY"] = sock
                break

    if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
        os.environ["DISPLAY"] = os.getenv("ZOE_DISPLAY", ":0")

    xauth = Path.home() / ".Xauthority"
    if not os.getenv("XAUTHORITY") and xauth.exists():
        os.environ["XAUTHORITY"] = str(xauth)


_prepare_session_env()

# One worker thread: links open strictly in request order, and a
# slow browser never blocks the runtime.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zoe-browser")
_shutdown = threading.Event()


def normalize_urls(urls: Iterable[str] | None) -> list[str]:
    """Strip, validate (absolute http/https with a host), de-duplicate, cap."""
    if not urls:
        return []

    clean: list[str] = []
    seen: set[str] = set()

    for raw in urls:
        url = str(raw or "").strip()
        if not url:
            continue
        try:
            parsed = urlparse(url)
        except ValueError:
            logger.warning("[ZOE BROWSER] Ignoring malformed URL: %r", url)
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            logger.warning("[ZOE BROWSER] Ignoring non-http(s) URL: %r", url)
            continue
        if url in seen:
            continue
        seen.add(url)
        clean.append(url)

    if MAX_LINKS_PER_CALL > 0 and len(clean) > MAX_LINKS_PER_CALL:
        logger.warning(
            "[ZOE BROWSER] %s links requested; opening only the first %s.",
            len(clean), MAX_LINKS_PER_CALL,
        )
        clean = clean[:MAX_LINKS_PER_CALL]

    return clean


def _resolve_browser() -> list[str] | None:
    """Return the browser argv prefix, or None to fall back to webbrowser."""
    if BROWSER_CMD:
        try:
            argv = shlex.split(BROWSER_CMD)
        except ValueError as exc:
            logger.warning("[ZOE BROWSER] Bad ZOE_BROWSER value (%s); auto-detecting.", exc)
            argv = []
        if argv and shutil.which(argv[0]):
            return argv
        logger.warning("[ZOE BROWSER] %r not found; auto-detecting.", BROWSER_CMD)

    for name in AUTO_BROWSERS:
        if shutil.which(name):
            return [name]
    return None


def _build_argv(base: list[str], url: str) -> list[str]:
    name = Path(base[0]).name.lower()
    argv = list(base)
    if OPEN_NEW == 1:
        argv.append("--new-window")
    elif OPEN_NEW == 2 and "firefox" in name:
        argv.append("--new-tab")
    # Chromium-based browsers (Helium, Chrome, Chromium) open a new tab
    # by default when handed a URL.
    argv.append(url)
    return argv


def _launch(base: list[str], url: str) -> bool:
    try:
        # start_new_session: the browser outlives this process and
        # ignores our SIGINT.
        subprocess.Popen(
            _build_argv(base, url),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=os.environ.copy(),
        )
        return True
    except OSError as exc:
        logger.error("[ZOE BROWSER] Launch failed for %s: %s", url, exc)
        return False


def _open_worker(urls: list[str]) -> None:
    base = _resolve_browser()

    if base is None:
        logger.warning("[ZOE BROWSER] No known browser binary; using webbrowser module.")
    else:
        logger.info("[ZOE BROWSER] Using browser: %s", " ".join(base))

    for index, url in enumerate(urls):
        if _shutdown.is_set():
            return

        try:
            if base is not None:
                ok = _launch(base, url)
            else:
                ok = webbrowser.open(url, new=OPEN_NEW)

            if ok:
                logger.info("[ZOE BROWSER] Opened %s", url)
            else:
                logger.warning("[ZOE BROWSER] Failed to open %s (no display?).", url)
        except Exception as exc:
            logger.error(
                "[ZOE BROWSER] Failed to open %s: %s: %s",
                url, type(exc).__name__, exc,
            )

        if index < len(urls) - 1 and OPEN_DELAY_SECONDS > 0:
            time.sleep(OPEN_DELAY_SECONDS)


def open_links(urls: Iterable[str] | None) -> list[str]:
    """
    Queue links to be opened in the browser. Returns the list that was
    actually accepted (validated, de-duplicated, capped). Never blocks
    and never raises.
    """
    if _shutdown.is_set():
        return []

    accepted = normalize_urls(urls)
    if not accepted:
        return []

    try:
        _executor.submit(_open_worker, accepted)
    except RuntimeError:
        # Executor already shut down.
        return []

    return accepted


def shutdown_browser() -> None:
    _shutdown.set()
    _executor.shutdown(wait=False)