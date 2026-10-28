from __future__ import annotations

import logging
import os
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable
from urllib.parse import urlparse


# ============================================================
# ZOE BROWSER
#
# Opens links directly in the machine's browser. The runtime
# calls open_links() with whatever URLs the brain returned;
# nothing is sent to the frontend.
#
# Env vars (all optional):
#
#   ZOE_BROWSER                 Browser name/command passed to
#                               webbrowser.get(), e.g.
#                               "chromium-browser", "firefox",
#                               "chrome". Empty = system default.
#   ZOE_BROWSER_NEW             0 = same window, 1 = new window,
#                               2 = new tab (default).
#   ZOE_BROWSER_OPEN_DELAY_SECONDS
#                               Pause between links so the
#                               browser keeps them in order.
#                               Default 0.4.
#   ZOE_BROWSER_MAX_LINKS       Max links opened per response.
#                               0 = unlimited (default).
#   ZOE_DISPLAY                 X display to use on Linux when
#                               neither DISPLAY nor
#                               WAYLAND_DISPLAY is set
#                               (default ":0", for a Pi started
#                               from systemd/SSH).
# ============================================================

logger = logging.getLogger("zoe.browser")


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


BROWSER_NAME = os.getenv(
    "ZOE_BROWSER",
    "",
).strip()

OPEN_NEW = _env_int(
    "ZOE_BROWSER_NEW",
    2,
)

OPEN_DELAY_SECONDS = _env_float(
    "ZOE_BROWSER_OPEN_DELAY_SECONDS",
    0.4,
)

MAX_LINKS_PER_CALL = _env_int(
    "ZOE_BROWSER_MAX_LINKS",
    0,
)


# ------------------------------------------------------------
# Headless-launch fix: a backend started from systemd or SSH
# has no DISPLAY, so the browser silently never opens.
# ------------------------------------------------------------

if (
    sys.platform.startswith("linux")
    and not os.getenv("DISPLAY")
    and not os.getenv("WAYLAND_DISPLAY")
):

    os.environ["DISPLAY"] = os.getenv(
        "ZOE_DISPLAY",
        ":0",
    )


# ------------------------------------------------------------
# One worker thread: links open strictly in the order they
# were requested, and a slow browser never blocks the runtime.
# ------------------------------------------------------------

_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="zoe-browser",
)

_shutdown = threading.Event()


def normalize_urls(
    urls: Iterable[str] | None,
) -> list[str]:

    """
    Strip, validate (absolute http/https with a host), and
    de-duplicate while preserving order.
    """

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

            logger.warning(
                "[ZOE BROWSER] Ignoring malformed URL: %r",
                url,
            )

            continue

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
        ):

            logger.warning(
                "[ZOE BROWSER] Ignoring non-http(s) URL: %r",
                url,
            )

            continue

        if url in seen:

            continue

        seen.add(url)

        clean.append(url)

    if (
        MAX_LINKS_PER_CALL > 0
        and len(clean) > MAX_LINKS_PER_CALL
    ):

        logger.warning(
            "[ZOE BROWSER] %s links requested; opening only "
            "the first %s.",
            len(clean),
            MAX_LINKS_PER_CALL,
        )

        clean = clean[:MAX_LINKS_PER_CALL]

    return clean


def _get_browser() -> webbrowser.BaseBrowser | None:

    if not BROWSER_NAME:

        return None

    try:

        return webbrowser.get(
            BROWSER_NAME
        )

    except webbrowser.Error as exc:

        logger.warning(
            "[ZOE BROWSER] Browser %r not found (%s); "
            "falling back to the system default.",
            BROWSER_NAME,
            exc,
        )

        return None


def _open_worker(
    urls: list[str],
) -> None:

    browser = _get_browser()

    for index, url in enumerate(urls):

        if _shutdown.is_set():

            return

        try:

            if browser is not None:

                opened = browser.open(
                    url,
                    new=OPEN_NEW,
                )

            else:

                opened = webbrowser.open(
                    url,
                    new=OPEN_NEW,
                )

            if opened:

                logger.info(
                    "[ZOE BROWSER] Opened %s",
                    url,
                )

            else:

                logger.warning(
                    "[ZOE BROWSER] Browser reported failure "
                    "opening %s (no browser / no display?).",
                    url,
                )

        except Exception as exc:

            logger.error(
                "[ZOE BROWSER] Failed to open %s: %s: %s",
                url,
                type(exc).__name__,
                exc,
            )

        if (
            index < len(urls) - 1
            and OPEN_DELAY_SECONDS > 0
        ):

            time.sleep(
                OPEN_DELAY_SECONDS
            )


def open_links(
    urls: Iterable[str] | None,
) -> list[str]:

    """
    Queue links to be opened in the browser. Returns the list
    that was actually accepted (validated, de-duplicated,
    capped). Never blocks and never raises.
    """

    if _shutdown.is_set():

        return []

    accepted = normalize_urls(
        urls
    )

    if not accepted:

        return []

    try:

        _executor.submit(
            _open_worker,
            accepted,
        )

    except RuntimeError:

        # Executor already shut down.
        return []

    return accepted


def shutdown_browser() -> None:

    _shutdown.set()

    _executor.shutdown(
        wait=False,
    )