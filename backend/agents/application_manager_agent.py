"""
ZOE window + app agent for Linux (Arch).

Window control is compositor-specific on Linux, so this file talks to the
session through a small backend layer:

  * Hyprland  -> hyprctl            (native Wayland, full support)
  * Sway      -> swaymsg            (native Wayland, full support)
  * X11       -> python-xlib / EWMH (any EWMH window manager: i3, bspwm,
                                     KDE/GNOME on X11, XFCE, ...)

GNOME and KDE *Wayland* sessions expose no window-control API to normal
programs, so they are not supported. Use an X11 session or Hyprland/Sway.

Dependencies:
    pacman -S python-xlib python-dotenv
    pip install groq            (inside a venv; not in the official repos)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.getenv("ZOE_WINDOW_MODEL", "openai/gpt-oss-120b")
MAX_TOOL_CALLS = int(os.getenv("ZOE_WINDOW_MAX_TOOL_CALLS", "12"))
MAX_COMPLETION_TOKENS = int(os.getenv("ZOE_WINDOW_MAX_TOKENS", "600"))
REQUEST_TIMEOUT_S = float(os.getenv("ZOE_WINDOW_TIMEOUT", "15"))
MAX_HISTORY_MESSAGES = 6

# Force a backend: "hyprland", "sway" or "x11". Empty = auto-detect.
BACKEND_OVERRIDE = os.getenv("ZOE_WINDOW_BACKEND", "").strip().lower()

# App launcher
APP_CACHE_TTL_S = 300.0
APP_LAUNCH_WAIT_S = float(os.getenv("ZOE_APP_LAUNCH_WAIT", "3"))
APP_SCAN_TIMEOUT_S = float(os.getenv("ZOE_APP_SCAN_TIMEOUT", "6"))
APP_SCAN_MAX_DEPTH = 4
APP_SCAN_MISS_TTL_S = 60.0
APP_FUZZY_CUTOFF = 0.8
APP_DEEP_SCAN = os.getenv("ZOE_APP_DEEP_SCAN", "1").lower() in {"1", "true", "yes"}
APP_PREWARM = os.getenv("ZOE_APP_PREWARM", "1").lower() in {"1", "true", "yes"}

# Off by default on purpose: a voice command must not be able to run
# arbitrary PATH binaries ("reboot", "poweroff", ...). Only apps with a
# .desktop entry, plus the bounded portable-app scan, are launchable.
APP_ALLOW_PATH = os.getenv("ZOE_APP_ALLOW_PATH", "0").lower() in {"1", "true", "yes"}

DEBUG = os.getenv("ZOE_WINDOW_DEBUG", "").lower() in {"1", "true", "yes"}
FAST_PATH = os.getenv("ZOE_WINDOW_FAST_PATH", "1").lower() in {"1", "true", "yes"}

# Spoken shortcuts -> names that appear in window classes and .desktop files.
# Keys must be in normalized form. File-manager words are resolved
# dynamically (see _file_manager_key).
APP_ALIASES = {
    "vs code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "code oss": "code",
}

FILE_MANAGER_WORDS = {
    "files",
    "file",
    "file manager",
    "file explorer",
    "explorer",
}

logger = logging.getLogger("zoe.window_agent")
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)


# ============================================================
# LAZY GROQ CLIENT (importing this module never crashes ZOE)
# ============================================================

_client: Any = None
_client_lock = threading.Lock()


def _get_client() -> Any:
    global _client

    with _client_lock:
        if _client is None:
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY3") or os.getenv("GROQ_API_KEY")

            if not api_key:
                raise RuntimeError("GROQ_API_KEY3 is not set.")

            _client = Groq(
                api_key=api_key,
                timeout=REQUEST_TIMEOUT_S,
                max_retries=1,
            )

        return _client


# ============================================================
# SHARED HELPERS
# ============================================================

class BackendError(RuntimeError):
    """A window backend could not do what was asked."""


def _run(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        done = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise BackendError(f"'{cmd[0]}' is not installed.") from exc
    except subprocess.SubprocessError as exc:
        raise BackendError(f"'{cmd[0]}' failed: {exc}") from exc

    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()
        raise BackendError(detail or f"'{cmd[0]}' exited with {done.returncode}.")

    return done.stdout


def _process_name(pid: int) -> str:
    if pid <= 0:
        return ""

    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _spawn(argv: list[str], cwd: str | None = None) -> str | None:
    """Start a fully detached process. Returns an error string or None."""

    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd if cwd and os.path.isdir(cwd) else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except FileNotFoundError:
        return f"'{argv[0]}' was not found."
    except OSError as exc:
        return exc.strerror or str(exc)

    # Reap the child when it exits so a long-running ZOE leaves no zombies.
    threading.Thread(target=proc.wait, daemon=True).start()
    return None


# ============================================================
# WINDOW INFORMATION
# ============================================================

@dataclass
class WindowInfo:
    wid: str
    title: str
    process_id: int
    process_name: str
    app_id: str
    visible: bool
    minimized: bool
    maximized: bool
    active: bool
    workspace: str = ""


# ============================================================
# WINDOW BACKENDS
# ============================================================

class WindowBackend:
    """Interface every compositor / window-system backend implements."""

    name = "base"

    def list_windows(self, include_hidden: bool = False) -> list[WindowInfo]:
        raise NotImplementedError

    def get(self, wid: str) -> WindowInfo | None:
        for window in self.list_windows(include_hidden=True):
            if window.wid == wid:
                return window
        return None

    def focus(self, wid: str) -> None:
        raise NotImplementedError

    def minimize(self, wid: str) -> None:
        raise NotImplementedError

    def maximize(self, wid: str) -> None:
        raise NotImplementedError

    def restore(self, wid: str) -> None:
        raise NotImplementedError

    def close(self, wid: str) -> None:
        raise NotImplementedError


# --------------------------------------------------------
# Hyprland
# --------------------------------------------------------

class HyprlandBackend(WindowBackend):
    """
    Hyprland has no minimize, so "minimized" means parked on the special
    workspace `special:minimized`, which is never shown.

    Hyprland 0.55+ replaced `hyprctl dispatch <name> <args>` with a single
    Lua expression (`hl.dsp.*`). Every action is sent in Lua first and
    falls back to the legacy syntax, and the working mode is remembered.
    """

    name = "hyprland"
    MINIMIZED_WS = "special:minimized"

    def __init__(self) -> None:
        self._mode: str | None = None  # "lua" or "legacy" once known

    @staticmethod
    def _json(*args: str) -> Any:
        out = _run(["hyprctl", "-j", *args])

        try:
            return json.loads(out) if out.strip() else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _q(value: str) -> str:
        """Quote a string for embedding in a Lua expression."""
        return json.dumps(value)

    def _dispatch(self, lua: str, *legacy: str) -> None:
        attempts = [("lua", ["hyprctl", "dispatch", lua])]

        if legacy:
            attempts.append(("legacy", ["hyprctl", "dispatch", *legacy]))

        if self._mode == "legacy":
            attempts.reverse()

        errors: list[str] = []

        for mode, cmd in attempts:
            try:
                out = _run(cmd).strip()
            except BackendError as exc:
                errors.append(f"{mode}: {exc}")
                continue

            # hyprctl can exit 0 and still print an error.
            if out.lower().startswith("error") or "invalid" in out.lower():
                errors.append(f"{mode}: {out}")
                continue

            self._mode = mode
            return

        raise BackendError("; ".join(errors) or "hyprctl dispatch failed.")

    def _clients(self) -> list[dict[str, Any]]:
        data = self._json("clients")
        return data if isinstance(data, list) else []

    def _client(self, wid: str) -> dict[str, Any] | None:
        return next((c for c in self._clients() if c.get("address") == wid), None)

    def list_windows(self, include_hidden: bool = False) -> list[WindowInfo]:
        active = self._json("activewindow")
        active_addr = active.get("address") if isinstance(active, dict) else None
        windows: list[WindowInfo] = []

        for client in self._clients():
            if not client.get("mapped", True) and not include_hidden:
                continue

            workspace = (client.get("workspace") or {}).get("name", "")
            pid = int(client.get("pid") or 0)
            app_id = client.get("class") or ""
            address = client.get("address", "")

            windows.append(
                WindowInfo(
                    wid=address,
                    title=(client.get("title") or "").strip(),
                    process_id=pid,
                    process_name=_process_name(pid) or app_id,
                    app_id=app_id,
                    visible=bool(client.get("mapped", True)),
                    minimized=workspace == self.MINIMIZED_WS,
                    maximized=int(client.get("fullscreen") or 0) > 0,
                    active=address == active_addr,
                    workspace=workspace,
                )
            )

        return windows

    # -- dispatcher wrappers (Lua first, legacy fallback) ----------

    def _focus_window(self, wid: str) -> None:
        sel = f"address:{wid}"
        self._dispatch(
            f"hl.dsp.focus({{ window = {self._q(sel)} }})",
            "focuswindow",
            sel,
        )

    def _move_to(self, wid: str, workspace: str) -> None:
        sel = f"address:{wid}"
        self._dispatch(
            f"hl.dsp.window.move({{ workspace = {self._q(workspace)}, "
            f"follow = false, window = {self._q(sel)} }})",
            "movetoworkspacesilent",
            f"{workspace},{sel}",
        )

    def _unminimize(self, wid: str) -> None:
        workspace = self._json("activeworkspace")
        target = str(workspace.get("id")) if isinstance(workspace, dict) else "e+0"
        self._move_to(wid, target)

    @staticmethod
    def _is_minimized(client: dict[str, Any]) -> bool:
        return (client.get("workspace") or {}).get("name") == HyprlandBackend.MINIMIZED_WS

    # -- actions ---------------------------------------------------

    def focus(self, wid: str) -> None:
        client = self._client(wid)

        if client is None:
            raise BackendError("Window no longer exists.")

        if self._is_minimized(client):
            self._unminimize(wid)

        self._focus_window(wid)

    def minimize(self, wid: str) -> None:
        self._move_to(wid, self.MINIMIZED_WS)

    def maximize(self, wid: str) -> None:
        self.focus(wid)
        client = self._client(wid)

        if client is not None and int(client.get("fullscreen") or 0) == 0:
            sel = f"address:{wid}"
            # Legacy `fullscreen 1` toggles the focused window (mode 1 = maximize).
            self._dispatch(
                'hl.dsp.window.fullscreen({ mode = "maximized", action = "set", '
                f"window = {self._q(sel)} }})",
                "fullscreen",
                "1",
            )

    def restore(self, wid: str) -> None:
        client = self._client(wid)

        if client is None:
            raise BackendError("Window no longer exists.")

        if self._is_minimized(client):
            self._unminimize(wid)

        mode = int(client.get("fullscreen") or 0)

        if mode > 0:
            sel = f"address:{wid}"
            lua_mode = "maximized" if mode == 1 else "fullscreen"

            if self._mode == "legacy":
                self._focus_window(wid)

            self._dispatch(
                f'hl.dsp.window.fullscreen({{ mode = "{lua_mode}", action = "unset", '
                f"window = {self._q(sel)} }})",
                "fullscreen",
                "1" if mode == 1 else "0",
            )

    def close(self, wid: str) -> None:
        sel = f"address:{wid}"
        self._dispatch(
            f"hl.dsp.window.close({{ window = {self._q(sel)} }})",
            "closewindow",
            sel,
        )


# --------------------------------------------------------
# Sway
# --------------------------------------------------------

class SwayBackend(WindowBackend):
    """Minimized windows live in the scratchpad (`__i3_scratch`)."""

    name = "sway"
    SCRATCH_WS = "__i3_scratch"

    @staticmethod
    def _tree() -> dict[str, Any]:
        try:
            return json.loads(_run(["swaymsg", "-t", "get_tree"]))
        except json.JSONDecodeError as exc:
            raise BackendError("swaymsg returned invalid JSON.") from exc

    @classmethod
    def _walk(
        cls,
        node: dict[str, Any],
        workspace: str,
        out: list[tuple[dict[str, Any], str]],
    ) -> None:
        if node.get("type") == "workspace":
            workspace = node.get("name", "")

        children = (node.get("nodes") or []) + (node.get("floating_nodes") or [])

        if (
            node.get("pid")
            and not children
            and node.get("type") in {"con", "floating_con"}
        ):
            out.append((node, workspace))

        for child in children:
            cls._walk(child, workspace, out)

    def _command(self, wid: str, command: str) -> None:
        _run(["swaymsg", f"[con_id={wid}] {command}"])

    def list_windows(self, include_hidden: bool = False) -> list[WindowInfo]:
        found: list[tuple[dict[str, Any], str]] = []
        self._walk(self._tree(), "", found)
        windows: list[WindowInfo] = []

        for node, workspace in found:
            pid = int(node.get("pid") or 0)
            props = node.get("window_properties") or {}
            app_id = node.get("app_id") or props.get("class") or ""

            windows.append(
                WindowInfo(
                    wid=str(node["id"]),
                    title=(node.get("name") or "").strip(),
                    process_id=pid,
                    process_name=_process_name(pid) or app_id,
                    app_id=app_id,
                    visible=True,
                    minimized=workspace == self.SCRATCH_WS,
                    maximized=int(node.get("fullscreen_mode") or 0) > 0,
                    active=bool(node.get("focused")),
                    workspace=workspace,
                )
            )

        return windows

    def _unminimize(self, wid: str) -> None:
        self._command(wid, "scratchpad show")
        self._command(wid, "floating disable")

    def focus(self, wid: str) -> None:
        info = self.get(wid)

        if info is None:
            raise BackendError("Window no longer exists.")

        if info.minimized:
            self._unminimize(wid)

        self._command(wid, "focus")

    def minimize(self, wid: str) -> None:
        self._command(wid, "move scratchpad")

    def maximize(self, wid: str) -> None:
        self.focus(wid)
        self._command(wid, "fullscreen enable")

    def restore(self, wid: str) -> None:
        info = self.get(wid)

        if info is None:
            raise BackendError("Window no longer exists.")

        if info.minimized:
            self._unminimize(wid)

        if info.maximized:
            self._command(wid, "fullscreen disable")

    def close(self, wid: str) -> None:
        # `kill` on a Wayland client is a polite xdg_toplevel close request.
        self._command(wid, "kill")


# --------------------------------------------------------
# X11 (EWMH via python-xlib)
# --------------------------------------------------------

class X11Backend(WindowBackend):
    name = "x11"

    _SKIP_TYPES = (
        "_NET_WM_WINDOW_TYPE_DOCK",
        "_NET_WM_WINDOW_TYPE_DESKTOP",
        "_NET_WM_WINDOW_TYPE_TOOLBAR",
        "_NET_WM_WINDOW_TYPE_MENU",
        "_NET_WM_WINDOW_TYPE_DROPDOWN_MENU",
        "_NET_WM_WINDOW_TYPE_POPUP_MENU",
        "_NET_WM_WINDOW_TYPE_TOOLTIP",
        "_NET_WM_WINDOW_TYPE_NOTIFICATION",
        "_NET_WM_WINDOW_TYPE_COMBO",
        "_NET_WM_WINDOW_TYPE_DND",
        "_NET_WM_WINDOW_TYPE_SPLASH",
    )

    def __init__(self) -> None:
        try:
            from Xlib import X, display, error, protocol
        except ImportError as exc:
            raise BackendError(
                "python-xlib is required for the X11 backend "
                "(pacman -S python-xlib)."
            ) from exc

        self._X = X
        self._error = error
        self._protocol = protocol

        try:
            self._display = display.Display()
        except Exception as exc:
            raise BackendError(f"Cannot open the X display: {exc}") from exc

        self._root = self._display.screen().root
        self._atoms: dict[str, int] = {}
        self._lock = threading.RLock()

    # -- low level ------------------------------------------------

    def _atom(self, name: str) -> int:
        if name not in self._atoms:
            self._atoms[name] = self._display.intern_atom(name)

        return self._atoms[name]

    def _window(self, wid: str) -> Any:
        try:
            return self._display.create_resource_object("window", int(wid))
        except ValueError as exc:
            raise BackendError(f"Invalid window id: {wid!r}") from exc

    def _prop(self, window: Any, name: str) -> Any:
        try:
            prop = window.get_full_property(self._atom(name), self._X.AnyPropertyType)
        except self._error.XError:
            return None

        return None if prop is None else prop.value

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        return value if isinstance(value, str) else ""

    def _send(self, window: Any, message: str, data: list[int]) -> None:
        data = (data + [0] * 5)[:5]

        event = self._protocol.event.ClientMessage(
            window=window,
            client_type=self._atom(message),
            data=(32, data),
        )
        mask = self._X.SubstructureRedirectMask | self._X.SubstructureNotifyMask

        with self._lock:
            self._root.send_event(event, event_mask=mask)
            self._display.flush()

    # -- listing --------------------------------------------------

    def list_windows(self, include_hidden: bool = False) -> list[WindowInfo]:
        try:
            with self._lock:
                return self._list(include_hidden)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"X11 query failed: {exc}") from exc

    def _list(self, include_hidden: bool) -> list[WindowInfo]:
        ids = self._prop(self._root, "_NET_CLIENT_LIST") or []
        active_val = self._prop(self._root, "_NET_ACTIVE_WINDOW")
        active_id = int(active_val[0]) if active_val is not None and len(active_val) else 0

        skip_types = {self._atom(name) for name in self._SKIP_TYPES}
        a_hidden = self._atom("_NET_WM_STATE_HIDDEN")
        a_vert = self._atom("_NET_WM_STATE_MAXIMIZED_VERT")
        a_horz = self._atom("_NET_WM_STATE_MAXIMIZED_HORZ")
        a_skip = self._atom("_NET_WM_STATE_SKIP_TASKBAR")

        windows: list[WindowInfo] = []

        for raw_id in ids:
            window = self._display.create_resource_object("window", int(raw_id))

            try:
                state = {int(a) for a in (self._prop(window, "_NET_WM_STATE") or [])}
                types = {int(a) for a in (self._prop(window, "_NET_WM_WINDOW_TYPE") or [])}

                if not include_hidden and (types & skip_types or a_skip in state):
                    continue

                title = self._text(self._prop(window, "_NET_WM_NAME")).strip()

                if not title:
                    title = (window.get_wm_name() or "").strip()

                pid_val = self._prop(window, "_NET_WM_PID")
                pid = int(pid_val[0]) if pid_val is not None and len(pid_val) else 0

                wm_class = window.get_wm_class()
                app_id = wm_class[1] if wm_class else ""

            except self._error.XError:
                continue

            windows.append(
                WindowInfo(
                    wid=str(int(raw_id)),
                    title=title,
                    process_id=pid,
                    process_name=_process_name(pid) or app_id,
                    app_id=app_id,
                    visible=True,
                    minimized=a_hidden in state,
                    maximized=a_vert in state and a_horz in state,
                    active=int(raw_id) == active_id,
                    workspace="",
                )
            )

        return windows

    # -- actions --------------------------------------------------

    def focus(self, wid: str) -> None:
        # source=2 ("pager") so window managers do not treat it as a
        # focus-stealing request. This also un-minimizes the window.
        self._send(self._window(wid), "_NET_ACTIVE_WINDOW", [2, 0, 0])

    def minimize(self, wid: str) -> None:
        self._send(self._window(wid), "WM_CHANGE_STATE", [3])  # IconicState

    def _set_maximized(self, wid: str, enabled: bool) -> None:
        self._send(
            self._window(wid),
            "_NET_WM_STATE",
            [
                1 if enabled else 0,
                self._atom("_NET_WM_STATE_MAXIMIZED_VERT"),
                self._atom("_NET_WM_STATE_MAXIMIZED_HORZ"),
                2,
            ],
        )

    def maximize(self, wid: str) -> None:
        self.focus(wid)
        self._set_maximized(wid, True)

    def restore(self, wid: str) -> None:
        info = self.get(wid)

        if info is None:
            raise BackendError("Window no longer exists.")

        if info.minimized:
            self.focus(wid)

        if info.maximized:
            self._set_maximized(wid, False)

    def close(self, wid: str) -> None:
        self._send(self._window(wid), "_NET_CLOSE_WINDOW", [0, 2])


# --------------------------------------------------------
# Backend selection
# --------------------------------------------------------

_BACKENDS: dict[str, type[WindowBackend]] = {
    "hyprland": HyprlandBackend,
    "sway": SwayBackend,
    "x11": X11Backend,
}


def _detect_backend() -> WindowBackend:
    if BACKEND_OVERRIDE:
        factory = _BACKENDS.get(BACKEND_OVERRIDE)

        if factory is None:
            raise BackendError(
                f"Unknown ZOE_WINDOW_BACKEND '{BACKEND_OVERRIDE}'. "
                f"Use one of: {', '.join(_BACKENDS)}."
            )

        return factory()

    if os.getenv("HYPRLAND_INSTANCE_SIGNATURE") and shutil.which("hyprctl"):
        return HyprlandBackend()

    if os.getenv("SWAYSOCK") and shutil.which("swaymsg"):
        return SwayBackend()

    if os.getenv("DISPLAY"):
        if os.getenv("WAYLAND_DISPLAY"):
            logger.warning(
                "Wayland session without Hyprland/Sway: only XWayland "
                "windows will be visible to ZOE."
            )

        return X11Backend()

    raise BackendError(
        "No supported desktop session found. ZOE needs Hyprland, Sway, or an "
        "X11 session. If ZOE runs as a service, pass it DISPLAY / SWAYSOCK / "
        "HYPRLAND_INSTANCE_SIGNATURE."
    )


# ============================================================
# WINDOW MANAGER
# ============================================================

UNTITLED = "(untitled window)"
ACTIVE_ALIASES = {
    "active",
    "active window",
    "current",
    "current window",
    "this",
    "this window",
}
FILLER_WORDS = {"the", "a", "an", "window", "windows", "app", "application", "program"}


@lru_cache(maxsize=1)
def _file_manager_key() -> str | None:
    """Normalized name of the user's default file manager (e.g. 'nautilus')."""

    try:
        out = subprocess.run(
            ["xdg-mime", "query", "default", "inode/directory"],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None

    if not out:
        return None

    stem = out[:-8] if out.endswith(".desktop") else out
    return WindowManager._normalize(stem.rsplit(".", 1)[-1]) or None


class WindowManager:
    """Inspect and control top-level application windows."""

    def __init__(self, backend: WindowBackend | None = None) -> None:
        self._backend = backend
        self._backend_lock = threading.Lock()

    @property
    def backend(self) -> WindowBackend:
        # Lazy, so importing this module never fails outside a desktop session.
        with self._backend_lock:
            if self._backend is None:
                self._backend = _detect_backend()
                logger.debug("Window backend: %s", self._backend.name)

            return self._backend

    # --------------------------------------------------------
    # INSPECT WINDOWS
    # --------------------------------------------------------

    def list_windows(
        self,
        include_hidden: bool = False,
        include_untitled: bool = False,
    ) -> list[WindowInfo]:
        """List top-level application windows."""

        windows = []

        for window in self.backend.list_windows(include_hidden=include_hidden):
            if not window.title:
                if not include_untitled:
                    continue

                window.title = UNTITLED

            windows.append(window)

        return sorted(windows, key=lambda w: (not w.active, w.title.casefold()))

    def get_active_window(self) -> WindowInfo | None:
        return next(
            (w for w in self.backend.list_windows(include_hidden=True) if w.active),
            None,
        )

    # --------------------------------------------------------
    # RESOLVE A WINDOW FROM NATURAL-LANGUAGE IDENTIFIERS
    # --------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.casefold()
        # Dots and apostrophes vanish so "Z.O.E" matches "zoe".
        text = re.sub(r"[.']", "", text)
        # Everything else that isn't a word character becomes a space.
        text = re.sub(r"[^\w\s]|_", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _alias(text: str) -> str:
        """Map spoken shortcuts ("vs code", "files") to real names."""

        if text in FILE_MANAGER_WORDS:
            return _file_manager_key() or text

        return APP_ALIASES.get(text, text)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        tokens = set()

        for token in text.split():
            # Cheap singularization so "files" matches "file manager".
            if len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            tokens.add(token)

        return tokens

    @staticmethod
    def _strip_filler(text: str) -> str:
        return " ".join(t for t in text.split() if t not in FILLER_WORDS)

    @staticmethod
    def _app_key(window: WindowInfo) -> str:
        """'org.gnome.Nautilus' -> 'nautilus', 'firefox' -> 'firefox'."""

        app_id = window.app_id

        if app_id.count(".") >= 2:
            app_id = app_id.rsplit(".", 1)[-1]

        return WindowManager._normalize(app_id)

    def _score_matches(
        self,
        query: str,
        windows: list[WindowInfo],
    ) -> list[tuple[int, WindowInfo]]:
        query_tokens = self._tokens(query)
        matches: list[tuple[int, WindowInfo]] = []

        for window in windows:
            title = self._normalize(window.title)
            process = self._normalize(window.process_name)
            app = self._app_key(window)
            names = [n for n in (process, app) if n]

            if query == title or query in names:
                score = 100
            elif title.startswith(query):
                score = 90
            elif query in title:
                score = 80
            elif any(n.startswith(query) for n in names):
                score = 75
            elif any(query in n for n in names):
                score = 70
            elif query_tokens and query_tokens <= (
                self._tokens(title).union(*(self._tokens(n) for n in names))
            ):
                score = 60
            else:
                continue

            matches.append((score, window))

        return matches

    def find_window(self, query: str) -> dict[str, Any]:
        """
        Resolve a title or application name to one window.

        Returns a structured ambiguity response instead of guessing
        when multiple windows match equally well.
        """

        if not query or not query.strip():
            raise ValueError("A window title or application name is required.")

        normalized = self._normalize(query)

        if normalized in ACTIVE_ALIASES:
            active = self.get_active_window()

            if active is None:
                return {"success": False, "message": "No active window was found."}

            return {"success": True, "window": asdict(active)}

        windows = self.list_windows()
        matches = self._score_matches(self._alias(normalized), windows)

        if not matches:
            stripped = self._strip_filler(normalized)

            if stripped and stripped != normalized:
                matches = self._score_matches(self._alias(stripped), windows)

        if not matches:
            return {
                "success": False,
                "no_match": True,
                "message": f"No visible window matched '{query.strip()}'.",
            }

        matches.sort(key=lambda item: (-item[0], not item[1].active))

        best_score = matches[0][0]
        best = [w for score, w in matches if score == best_score]

        if len(best) > 1:
            return {
                "success": False,
                "ambiguous": True,
                "message": (
                    f"Multiple windows match '{query.strip()}'. "
                    "Specify a more precise window title."
                ),
                "matches": [
                    {
                        "title": w.title,
                        "process_name": w.process_name,
                        "wid": w.wid,
                        "active": w.active,
                    }
                    for w in best[:8]
                ],
            }

        return {"success": True, "window": asdict(best[0])}

    # --------------------------------------------------------
    # WINDOW ACTIONS
    # --------------------------------------------------------

    @staticmethod
    def _wait_until(predicate: Any, timeout: float = 0.5, step: float = 0.05) -> bool:
        deadline = time.monotonic() + timeout

        while True:
            try:
                if predicate():
                    return True
            except BackendError:
                pass

            if time.monotonic() >= deadline:
                return False

            time.sleep(step)

    def _state(self, wid: str) -> WindowInfo | None:
        return self.backend.get(wid)

    def _validate(self, wid: str) -> None:
        if not isinstance(wid, str) or not wid or self._state(wid) is None:
            raise ValueError(f"Invalid window handle: {wid}")

    def switch_to(self, wid: str) -> dict[str, Any]:
        self._validate(wid)
        self.backend.focus(wid)

        active = self._wait_until(
            lambda: (s := self._state(wid)) is not None and s.active and not s.minimized
        )

        return {
            "success": active,
            "message": (
                "Window is now active."
                if active
                else "The compositor did not grant focus."
            ),
        }

    def minimize(self, wid: str) -> dict[str, Any]:
        self._validate(wid)
        self.backend.minimize(wid)
        done = self._wait_until(lambda: (s := self._state(wid)) is not None and s.minimized)

        return {
            "success": done,
            "message": "Window minimized." if done else "Window could not be minimized.",
        }

    def maximize(self, wid: str) -> dict[str, Any]:
        self._validate(wid)
        self.backend.maximize(wid)
        done = self._wait_until(lambda: (s := self._state(wid)) is not None and s.maximized)

        return {
            "success": done,
            "message": "Window maximized." if done else "Window could not be maximized.",
        }

    def restore(self, wid: str) -> dict[str, Any]:
        self._validate(wid)
        self.backend.restore(wid)
        done = self._wait_until(
            lambda: (s := self._state(wid)) is not None
            and not s.minimized
            and not s.maximized
        )

        return {
            "success": done,
            "message": "Window restored." if done else "Window could not be restored.",
        }

    def toggle_maximize(self, wid: str) -> dict[str, Any]:
        self._validate(wid)

        state = self._state(wid)
        was_maximized = bool(state and state.maximized)
        result = self.restore(wid) if was_maximized else self.maximize(wid)
        result["action"] = "restore" if was_maximized else "maximize"

        return result

    def close(self, wid: str) -> dict[str, Any]:
        """
        Request a normal window close. Never force-terminates the
        application, which may prompt to save work or refuse to close.
        """

        self._validate(wid)

        try:
            self.backend.close(wid)
        except BackendError as exc:
            return {"success": False, "message": f"Close request failed: {exc}"}

        closed = self._wait_until(lambda: self._state(wid) is None, timeout=0.6)

        return {
            "success": True,
            "closed": closed,
            "message": (
                "Window closed."
                if closed
                else "Close request sent. The application may prompt to "
                "save work or refuse to close."
            ),
        }


# ============================================================
# SINGLETON WINDOW MANAGER
# ============================================================

window_manager = WindowManager()


# ============================================================
# APP LAUNCHER
# ============================================================

# Menu entries that match a name but are never what the user means.
APP_JUNK_WORDS = {
    "uninstall",
    "readme",
    "documentation",
    "release",
    "notes",
    "license",
    "manual",
    "help",
    "website",
    "changelog",
}

# Installer/helper executables ignored by the portable-app scan.
EXE_JUNK_PARTS = ("unins", "setup", "install", "updat", "crash", "helper", "redist")

SCAN_SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    "site-packages",
    "venv",
    "proc",
    "sys",
    "dev",
    "lost+found",
}

# Executable names that say nothing about which app a .desktop entry is.
WRAPPER_EXECS = {
    "flatpak", "env", "sh", "bash", "zsh", "python", "python3", "java",
    "xdg-open", "gtk-launch", "gio", "snap", "wine", "gamemoderun",
}

TERMINALS = (
    "kitty", "alacritty", "foot", "wezterm", "ghostty",
    "gnome-terminal", "konsole", "xfce4-terminal", "xterm",
)

_FIELD_CODE = re.compile(r"%[fFuUdDnNickvm]")


@dataclass(frozen=True)
class DesktopApp:
    name: str
    desktop_id: str
    path: str
    exec_line: str
    terminal: bool
    workdir: str
    keys: tuple[str, ...]


def _expand_exec(exec_line: str) -> list[str]:
    """Turn a .desktop Exec= value into an argv without field codes."""

    try:
        argv = shlex.split(exec_line)
    except ValueError:
        argv = exec_line.split()

    cleaned = []

    for arg in argv:
        arg = arg.replace("%%", "\0")
        arg = _FIELD_CODE.sub("", arg).replace("\0", "%")

        if arg:
            cleaned.append(arg)

    return cleaned


def _terminal_prefix() -> list[str] | None:
    candidates = [os.getenv("TERMINAL", "").strip(), *TERMINALS]

    for terminal in candidates:
        if terminal and shutil.which(terminal):
            name = os.path.basename(terminal)

            if name == "gnome-terminal":
                return [terminal, "--"]
            if name == "wezterm":
                return [terminal, "start", "--"]

            return [terminal, "-e"]

    return None


def _is_binary(path: str) -> bool:
    """True for ELF executables and AppImages."""

    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return False

    if path.lower().endswith(".appimage"):
        return True

    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


class AppLauncher:
    """
    Find and launch applications by name.

    Lookup order (fast to slow):
      0. Absolute path to a file or folder
      1. Desktop-entry index (every .desktop file: system, user, Flatpak)
      2. PATH (only when ZOE_APP_ALLOW_PATH=1)
      3. Fuzzy desktop-entry match (typos, partial names)
      4. Bounded scan for portable executables and AppImages
    """

    def __init__(self, windows: WindowManager) -> None:
        self._windows = windows
        self._apps_cache: tuple[list[DesktopApp], float] | None = None
        self._apps_lock = threading.Lock()
        self._scan_misses: dict[str, float] = {}
        self._scan_lock = threading.Lock()

    # --------------------------------------------------------
    # DESKTOP-ENTRY INDEX
    # --------------------------------------------------------

    @staticmethod
    def _data_dirs() -> list[Path]:
        home = Path.home()
        dirs = [Path(os.getenv("XDG_DATA_HOME") or home / ".local/share")]

        for entry in (os.getenv("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":"):
            if entry:
                dirs.append(Path(entry))

        dirs += [
            Path("/var/lib/flatpak/exports/share"),
            home / ".local/share/flatpak/exports/share",
        ]

        unique: list[Path] = []

        for directory in dirs:
            if directory not in unique:
                unique.append(directory)

        return unique

    @staticmethod
    def _parse_desktop_file(path: Path, desktop_id: str) -> DesktopApp | None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        entry: dict[str, str] = {}
        in_entry = False

        for line in text.splitlines():
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("["):
                if in_entry:
                    break

                in_entry = line == "[Desktop Entry]"
                continue

            if not in_entry or "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()

            if "[" in key:  # localized keys such as Name[de]
                continue

            entry.setdefault(key, value.strip())

        if entry.get("Type", "Application") != "Application":
            return None

        if entry.get("NoDisplay", "").lower() == "true":
            return None

        if entry.get("Hidden", "").lower() == "true":
            return None

        name = entry.get("Name", "")
        exec_line = entry.get("Exec", "")

        if not name or not exec_line:
            return None

        try_exec = entry.get("TryExec", "")

        if try_exec and not (shutil.which(try_exec) or os.path.isfile(try_exec)):
            return None

        norm = WindowManager._normalize
        keys: list[str] = [norm(name)]

        id_stem = desktop_id[:-8] if desktop_id.endswith(".desktop") else desktop_id
        keys.append(norm(id_stem.rsplit(".", 1)[-1]))

        argv = _expand_exec(exec_line)

        if argv:
            binary = os.path.basename(argv[0])
            extra = [a for a in argv[1:] if not a.startswith("-")]

            # Only a bare command identifies the app; "steam steam://run/42"
            # or "flatpak run org.x.App" must not claim "steam" / "flatpak".
            if binary not in WRAPPER_EXECS and not extra:
                keys.append(norm(binary))

        unique_keys = tuple(dict.fromkeys(k for k in keys if k))

        return DesktopApp(
            name=name,
            desktop_id=desktop_id,
            path=str(path),
            exec_line=exec_line,
            terminal=entry.get("Terminal", "").lower() == "true",
            workdir=entry.get("Path", ""),
            keys=unique_keys,
        )

    def _scan_desktop_files(self) -> list[DesktopApp]:
        apps: list[DesktopApp] = []
        seen_ids: set[str] = set()

        for data_dir in self._data_dirs():
            base = data_dir / "applications"

            if not base.is_dir():
                continue

            try:
                files = sorted(base.rglob("*.desktop"))
            except OSError:
                continue

            for path in files:
                desktop_id = str(path.relative_to(base)).replace(os.sep, "-")

                # Earlier directories (user) override later ones (system).
                if desktop_id in seen_ids:
                    continue

                seen_ids.add(desktop_id)
                app = self._parse_desktop_file(path, desktop_id)

                if app is not None:
                    apps.append(app)

        return apps

    def apps(self, force: bool = False) -> list[DesktopApp]:
        """Every launchable desktop entry (cached)."""

        with self._apps_lock:
            now = time.monotonic()
            cached = self._apps_cache

            if cached and not force and now - cached[1] < APP_CACHE_TTL_S:
                return cached[0]

            apps = self._scan_desktop_files()

            # Keep serving the old index if the scan came back empty.
            if not apps and cached:
                return cached[0]

            self._apps_cache = (apps, now)
            return apps

    def prewarm(self) -> None:
        try:
            self.apps()
        except Exception:
            logger.debug("App index prewarm failed", exc_info=True)

    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    @staticmethod
    def _is_junk(name: str, query: str) -> bool:
        return bool(set(name.split()) & APP_JUNK_WORDS) and not (
            set(query.split()) & APP_JUNK_WORDS
        )

    def _score(self, query: str, name: str, query_tokens: set[str]) -> int:
        if self._is_junk(name, query):
            return 0

        if name == query:
            return 100
        if f" {query} " in f" {name} ":
            return 85
        if name.startswith(query):
            return 75
        if query in name:
            return 70
        if query_tokens and query_tokens <= WindowManager._tokens(name):
            return 60

        return 0

    def _match_app(self, query: str) -> dict[str, Any] | None:
        """Exact / word / prefix / substring / token matching."""

        query_tokens = WindowManager._tokens(query)
        scored: list[tuple[int, DesktopApp]] = []

        for app in self.apps():
            score = max(
                (self._score(query, key, query_tokens) for key in app.keys),
                default=0,
            )

            if score:
                scored.append((score, app))

        if not scored:
            return None

        best_score = max(score for score, _ in scored)

        # Shortest display name wins ("Firefox" over "Firefox Developer Edition").
        best = sorted(
            (app for score, app in scored if score == best_score),
            key=lambda app: (len(app.name), app.name.casefold()),
        )

        if len(best) > 1 and len(best[1].name) == len(best[0].name):
            tied = [app for app in best if len(app.name) == len(best[0].name)]

            return {
                "success": False,
                "ambiguous": True,
                "message": (
                    f"Multiple applications match '{query}'. "
                    "Specify a more precise name."
                ),
                "matches": [{"title": app.name} for app in tied[:8]],
            }

        return {"success": True, "app": best[0]}

    def _fuzzy_app(self, query: str) -> tuple[DesktopApp | None, list[str]]:
        """Typo-tolerant match. Returns (match or None, closest names)."""

        if len(query) < 3:
            return None, []

        ranked: list[tuple[float, DesktopApp]] = []

        for app in self.apps():
            parts = {part for key in app.keys for part in (key, *key.split())}
            parts = {p for p in parts if not self._is_junk(p, query)}

            if not parts:
                continue

            ratio = max(SequenceMatcher(None, query, part).ratio() for part in parts)
            ranked.append((ratio, app))

        ranked.sort(key=lambda item: (-item[0], len(item[1].name)))
        suggestions = [app.name for ratio, app in ranked[:3] if ratio >= 0.5]

        if ranked and ranked[0][0] >= APP_FUZZY_CUTOFF:
            return ranked[0][1], suggestions

        return None, suggestions

    # --------------------------------------------------------
    # PORTABLE-APP SCAN (AppImages, unpacked tarballs, ~/bin)
    # --------------------------------------------------------

    @staticmethod
    def _scan_roots() -> list[str]:
        home = Path.home()
        candidates = [
            home / "Applications",
            home / "Downloads",
            home / "Desktop",
            home / "Documents",
            home / ".local/bin",
            home / "bin",
            Path("/opt"),
        ]

        return [str(p) for p in candidates if p.is_dir()]

    def _scan_for_exe(self, query: str) -> str | None:
        """Time-boxed, shallow-first search for a matching executable."""

        deadline = time.monotonic() + APP_SCAN_TIMEOUT_S
        query_is_junk = any(part in query for part in EXE_JUNK_PARTS)
        suffix_match = re.compile(
            rf"{re.escape(query)}[\s\d]*(?:\s*(?:x86 64|x64|x86|amd64|aarch64|linux|portable))*"
        )
        best: tuple[int, str] | None = None

        for root in self._scan_roots():
            queue = deque([(root, 0)])

            while queue:
                if time.monotonic() > deadline:
                    return best[1] if best else None

                directory, depth = queue.popleft()

                try:
                    with os.scandir(directory) as iterator:
                        entries = list(iterator)
                except OSError:
                    continue

                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if (
                                depth < APP_SCAN_MAX_DEPTH
                                and not entry.name.startswith(".")
                                and entry.name.casefold() not in SCAN_SKIP_DIRS
                            ):
                                queue.append((entry.path, depth + 1))
                            continue

                        if not entry.is_file():
                            continue

                    except OSError:
                        continue

                    stem = WindowManager._normalize(
                        re.sub(r"\.appimage$", "", entry.name, flags=re.IGNORECASE)
                    )

                    if not stem:
                        continue

                    if not query_is_junk and any(
                        part in stem for part in EXE_JUNK_PARTS
                    ):
                        continue

                    exact = stem == query

                    if not (
                        exact
                        or f" {query} " in f" {stem} "
                        or suffix_match.fullmatch(stem)
                    ):
                        continue

                    # Only now pay for the permission / magic-number check.
                    if not _is_binary(entry.path):
                        continue

                    if exact:
                        return entry.path

                    if best is None or len(entry.name) < best[0]:
                        best = (len(entry.name), entry.path)

        return best[1] if best else None

    def _deep_search(self, query: str) -> str | None:
        if not APP_DEEP_SCAN:
            return None

        with self._scan_lock:
            missed_at = self._scan_misses.get(query)

            if missed_at and time.monotonic() - missed_at < APP_SCAN_MISS_TTL_S:
                return None

        path = self._scan_for_exe(query)

        if path is None:
            with self._scan_lock:
                if len(self._scan_misses) > 64:
                    self._scan_misses.clear()

                self._scan_misses[query] = time.monotonic()

        return path

    # --------------------------------------------------------
    # LAUNCH
    # --------------------------------------------------------

    @staticmethod
    def _launch_app(app: DesktopApp) -> str | None:
        argv = _expand_exec(app.exec_line)

        if not argv:
            return "The desktop entry has no command."

        if app.terminal:
            prefix = _terminal_prefix()

            if prefix is None:
                return "This app needs a terminal and none was found."

            argv = [*prefix, *argv]

        return _spawn(argv, cwd=app.workdir or str(Path.home()))

    @staticmethod
    def _launch_path(path: str) -> str | None:
        if _is_binary(path):
            return _spawn([path], cwd=os.path.dirname(path))

        return _spawn(["xdg-open", path])

    def _wait_for_window(
        self,
        before: set[str],
        foreground_before: str,
    ) -> WindowInfo | None:
        """
        Wait for the launched app to show a titled window, so follow-up
        actions in the same request (maximize, move, close) can resolve it.
        """

        deadline = time.monotonic() + APP_LAUNCH_WAIT_S

        while True:
            try:
                for window in self._windows.list_windows():
                    if window.wid not in before:
                        return window

                # Single-instance apps raise their existing window instead.
                active = self._windows.get_active_window()

                if (
                    active is not None
                    and active.wid != foreground_before
                    and active.title
                ):
                    return active

            except BackendError:
                return None

            if time.monotonic() >= deadline:
                return None

            time.sleep(0.15)

    def _start(self, launch: Any, target: str, label: str) -> dict[str, Any]:
        try:
            before = {w.wid for w in self._windows.list_windows()}
            foreground = self._windows.get_active_window()
        except BackendError as exc:
            return {"success": False, "message": f"Window backend error: {exc}"}

        foreground_before = foreground.wid if foreground else ""

        error = launch()

        if error:
            return {
                "success": False,
                "message": f"Could not open '{label}': {error}",
            }

        window = self._wait_for_window(before, foreground_before)

        result: dict[str, Any] = {
            "success": True,
            "app": {"name": label, "target": target},
            "window_appeared": window is not None,
            "message": (
                f"Opened '{label}'."
                if window
                else f"Launched '{label}', but its window has not appeared yet."
            ),
        }

        if window is not None:
            result["window"] = {
                "title": window.title,
                "process_name": window.process_name,
                "wid": window.wid,
            }

        return result

    def _start_app(self, app: DesktopApp) -> dict[str, Any]:
        return self._start(lambda: self._launch_app(app), app.desktop_id, app.name)

    def _start_path(self, path: str, label: str) -> dict[str, Any]:
        return self._start(lambda: self._launch_path(path), path, label)

    # --------------------------------------------------------
    # PUBLIC ENTRY POINT
    # --------------------------------------------------------

    def open(self, query: str) -> dict[str, Any]:
        raw = (query or "").strip().strip('"')

        if not raw:
            return {"success": False, "message": "An application name is required."}

        # 0. Explicit absolute path (file or folder)
        expanded = os.path.expanduser(os.path.expandvars(raw))

        if os.path.isabs(expanded) and os.path.exists(expanded):
            label = os.path.basename(expanded.rstrip("/")) or expanded
            return self._start_path(expanded, label)

        name = raw[:-4] if raw.lower().endswith(".exe") else raw
        normalized = WindowManager._normalize(name)
        normalized = WindowManager._strip_filler(normalized) or normalized
        normalized = WindowManager._alias(normalized)

        # 1. Desktop entries: system, user and Flatpak apps
        match = self._match_app(normalized)

        if match is not None:
            if not match["success"]:
                return match

            return self._start_app(match["app"])

        # 2. PATH (opt-in)
        if APP_ALLOW_PATH and " " not in name:
            path = shutil.which(name)

            if path:
                return self._start_path(path, os.path.basename(path))

        # 3. Typos and partial names
        fuzzy, suggestions = self._fuzzy_app(normalized)

        if fuzzy is not None:
            return self._start_app(fuzzy)

        # 4. Portable / unregistered executables
        path = self._deep_search(normalized)

        if path:
            return self._start_path(path, os.path.basename(path))

        message = f"Could not find an application matching '{raw}'."

        if suggestions:
            message += f" Closest matches: {', '.join(suggestions)}."

        return {"success": False, "no_match": True, "message": message}


# ============================================================
# SINGLETON APP LAUNCHER
# ============================================================

app_launcher = AppLauncher(window_manager)

if APP_PREWARM:
    # Build the desktop-entry index in the background so the first
    # "open X" request does not pay the scan cost.
    threading.Thread(
        target=app_launcher.prewarm,
        name="zoe-app-index",
        daemon=True,
    ).start()


# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def tool_list_windows(include_hidden: bool = False) -> dict[str, Any]:
    windows = window_manager.list_windows(include_hidden=include_hidden)

    return {
        "success": True,
        "count": len(windows),
        "windows": [asdict(w) for w in windows],
    }


def tool_get_active_window() -> dict[str, Any]:
    window = window_manager.get_active_window()

    if window is None:
        return {"success": False, "message": "No active window was found."}

    return {"success": True, "window": asdict(window)}


def tool_open_app(query: str) -> dict[str, Any]:
    return app_launcher.open(query)


def _run_window_action(query: str, action: str) -> dict[str, Any]:
    resolved = window_manager.find_window(query)

    if not resolved.get("success"):
        return resolved

    window = resolved["window"]
    wid = window["wid"]

    actions = {
        "switch_to": window_manager.switch_to,
        "minimize": window_manager.minimize,
        "maximize": window_manager.maximize,
        "restore": window_manager.restore,
        "toggle_maximize": window_manager.toggle_maximize,
        "close_window": window_manager.close,
    }

    try:
        result = actions[action](wid)
    except (ValueError, BackendError) as exc:
        # The window vanished between resolve and act, or the backend failed.
        return {"success": False, "message": str(exc)}

    return {
        **result,
        "action": result.get("action", action),
        "window": {
            "title": window["title"],
            "process_name": window["process_name"],
            "wid": wid,
        },
    }


def _make_action_tool(action: str) -> Any:
    def tool(query: str) -> dict[str, Any]:
        return _run_window_action(query, action)

    tool.__name__ = f"tool_{action}"
    return tool


# ============================================================
# TOOL REGISTRY AND ARGUMENT SCHEMA
# ============================================================

TOOLS: dict[str, Any] = {
    "list_windows": tool_list_windows,
    "get_active_window": tool_get_active_window,
    "open_app": tool_open_app,
    **{
        name: _make_action_tool(name)
        for name in (
            "switch_to",
            "minimize",
            "maximize",
            "restore",
            "toggle_maximize",
            "close_window",
        )
    },
}

ALLOWED_ARGUMENTS: dict[str, set[str]] = {
    "list_windows": {"include_hidden"},
    "get_active_window": set(),
    "open_app": {"query"},
    "switch_to": {"query"},
    "minimize": {"query"},
    "maximize": {"query"},
    "restore": {"query"},
    "toggle_maximize": {"query"},
    "close_window": {"query"},
}

REQUIRED_ARGUMENTS: dict[str, set[str]] = {
    name: ({"query"} if "query" in allowed else set())
    for name, allowed in ALLOWED_ARGUMENTS.items()
}


# ============================================================
# GROQ TOOL DEFINITIONS
# ============================================================

def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


WINDOW_QUERY = {
    "query": {
        "type": "string",
        "description": (
            "Application name or distinctive words from the window title, "
            "e.g. 'firefox', 'visual studio code', 'files', or "
            "'active window'. Omit filler words like 'window' or 'app'."
        ),
    }
}

APP_QUERY = {
    "query": {
        "type": "string",
        "description": (
            "Application name as the user would say it, e.g. 'firefox', "
            "'spotify', 'kate', 'visual studio code', 'files'. "
            "Omit filler words like 'app' or 'open'. A full file or folder "
            "path also works."
        ),
    }
}

GROQ_TOOLS = [
    _tool(
        "list_windows",
        "List open top-level windows. Use when asked which apps or windows are open.",
        {
            "include_hidden": {
                "type": "boolean",
                "description": "Include hidden windows. Normally false.",
            }
        },
    ),
    _tool("get_active_window", "Identify the focused window.", {}),
    _tool(
        "open_app",
        "Launch an installed application (or open a file or folder path). "
        "Use for apps that are not already open; use switch_to for apps "
        "that already have a window.",
        APP_QUERY,
        ["query"],
    ),
    _tool(
        "switch_to",
        "Bring a window to the foreground, restoring it if minimized.",
        WINDOW_QUERY,
        ["query"],
    ),
    _tool("minimize", "Minimize a window.", WINDOW_QUERY, ["query"]),
    _tool("maximize", "Maximize a window.", WINDOW_QUERY, ["query"]),
    _tool(
        "restore",
        "Restore a minimized or maximized window to normal size.",
        WINDOW_QUERY,
        ["query"],
    ),
    _tool(
        "toggle_maximize",
        "Maximize a window, or restore it if already maximized.",
        WINDOW_QUERY,
        ["query"],
    ),
    _tool(
        "close_window",
        "Request a normal close of a window. Never force-kills.",
        WINDOW_QUERY,
        ["query"],
    ),
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's Linux desktop window-routing agent. You translate a request
into tool calls. Your tool results are summarized by code, not by you.

RULES
1. Emit ALL needed tool calls in a single response, in the order requested.
2. Use the application name or distinctive title words as the query.
   Drop filler words such as "window", "app", "the".
3. "this window" or "the active window" means query "active window".
4. If the user asks which apps or windows are open, call list_windows.
5. To launch an application (open, launch, start, run), call open_app with
   the application name. Use switch_to instead when the user wants an app
   that is already open. open_app waits for the new window to appear, so
   follow-up actions can come right after it, e.g. "open kate and
   maximize it" = open_app then maximize.
6. Only perform actions the user explicitly requested.
7. If the request is not about windows or launching applications, or a
   target is too vague to act on, reply with one short plain-text sentence
   and call no tools.
8. Plain text only, no markdown.
9. A request may contain several tasks joined by "and", "then", "also" or
   commas. Emit one separate tool call per task, in the order given. A verb
   applies to every item that follows it until a new verb appears, e.g.
   "open spotify and discord, close files also minimize vs code" =
   open_app spotify, open_app discord, close_window files, minimize vs code.
""".strip()


# ============================================================
# HISTORY AND MESSAGES
# ============================================================

def clean_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if not history:
        return []

    cleaned: list[dict[str, str]] = []

    for item in history:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        cleaned.append({"role": role, "content": content.strip()})

    return cleaned[-MAX_HISTORY_MESSAGES:]


def build_messages(
    user_request: str,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *clean_history(history),
        {"role": "user", "content": user_request.strip()},
    ]


# ============================================================
# ARGUMENT VALIDATION AND EXECUTION
# ============================================================

def normalize_arguments(name: str, arguments: Any) -> dict[str, Any]:
    if name not in ALLOWED_ARGUMENTS:
        raise ValueError(f"Unknown window tool: {name!r}")

    if arguments is None:
        arguments = {}

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON arguments for {name!r}.") from exc

    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for {name!r} must be a JSON object.")

    unknown = set(arguments) - ALLOWED_ARGUMENTS[name]

    if unknown:
        raise ValueError(
            f"Unknown arguments for {name!r}: {', '.join(sorted(unknown))}"
        )

    missing = [
        key
        for key in REQUIRED_ARGUMENTS[name]
        if arguments.get(key) is None
        or (isinstance(arguments[key], str) and not arguments[key].strip())
    ]

    if missing:
        raise ValueError(
            f"Missing required arguments for {name!r}: {', '.join(sorted(missing))}"
        )

    if "include_hidden" in arguments and not isinstance(
        arguments["include_hidden"], bool
    ):
        raise ValueError("include_hidden must be a boolean.")

    if "query" in arguments:
        if not isinstance(arguments["query"], str):
            raise ValueError("query must be a string.")

        arguments["query"] = arguments["query"].strip()

    return dict(arguments)


def execute_tool(name: str, arguments: Any) -> Any:
    return TOOLS[name](**normalize_arguments(name, arguments))


# ============================================================
# RESULT SUMMARIES (deterministic, no LLM)
# ============================================================

def summarize_result(name: str, result: dict[str, Any]) -> str:
    if not result.get("success", False):
        if result.get("ambiguous"):
            titles = ", ".join(
                str(item.get("title", "Unknown"))
                for item in result.get("matches", [])
            )

            return result.get("message", "Ambiguous window.") + (
                f" Matches: {titles}." if titles else ""
            )

        return str(result.get("message") or result.get("error") or f"{name} failed.")

    if name == "list_windows":
        windows = result.get("windows", [])

        if not windows:
            return "No matching windows are currently open."

        lines = [f"{len(windows)} window(s) found:"]

        for window in windows:
            flags = []

            if window.get("active"):
                flags.append("ACTIVE")
            if window.get("minimized"):
                flags.append("MINIMIZED")
            if window.get("maximized"):
                flags.append("MAXIMIZED")
            if not window.get("visible"):
                flags.append("HIDDEN")

            lines.append(
                f"{window['title']} "
                f"({window['process_name']}; {', '.join(flags) or 'NORMAL'})"
            )

        return "\n".join(lines)

    if name == "get_active_window":
        window = result["window"]
        return f"Active window: {window['title']} ({window['process_name']})."

    if name == "open_app":
        app_name = result.get("app", {}).get("name", "application")

        if result.get("window_appeared"):
            return f"Opened '{app_name}'."

        return f"Launched '{app_name}', but its window has not appeared yet."

    title = result.get("window", {}).get("title", "Window")

    if name == "close_window":
        if result.get("closed"):
            return f"Closed '{title}'."

        return (
            f"Close request sent to '{title}'. "
            "The application may still be running or asking to save."
        )

    if name == "toggle_maximize":
        past = "Restored" if result.get("action") == "restore" else "Maximized"
        return f"{past} '{title}'."

    messages = {
        "switch_to": f"Switched to '{title}'.",
        "minimize": f"Minimized '{title}'.",
        "maximize": f"Maximized '{title}'.",
        "restore": f"Restored '{title}'.",
    }

    return messages.get(name, str(result.get("message", f"{name} completed.")))


def _run_operation(name: str, raw_arguments: Any) -> dict[str, Any]:
    """Execute one tool call and return a uniform operation record."""

    try:
        arguments = normalize_arguments(name, raw_arguments)
        result = execute_tool(name, arguments)

        return {
            "tool": name,
            "arguments": arguments,
            "success": isinstance(result, dict) and result.get("success") is True,
            "result": result,
            "summary": summarize_result(name, result),
        }

    except Exception as exc:
        logger.debug("Tool %s failed", name, exc_info=True)

        return {
            "tool": name,
            "arguments": raw_arguments if isinstance(raw_arguments, str) else {},
            "success": False,
            "error": str(exc),
            "summary": f"{name} failed: {exc}",
        }


def _build_response(operations: list[dict[str, Any]]) -> dict[str, Any]:
    response: dict[str, Any] = {
        "success": bool(operations) and all(op["success"] for op in operations),
        "summary": "\n".join(op["summary"] for op in operations),
    }

    if DEBUG:
        response["operations"] = operations

    return response


# ============================================================
# FAST PATH (no LLM for simple single or bulk commands)
# ============================================================

_FAST_RE = re.compile(
    r"""^
    (?:(?:please|can\s+you|could\s+you)\s+)?
    (?P<verb>switch\s+to|go\s+to|bring\s+up|focus\s+on|focus|show|
             open|launch|start|run|
             minimi[sz]e|maximi[sz]e|restore|close)
    \s+(?:the\s+)?
    (?P<target>.+?)
    (?:\s+(?:window|app|application))?
    \s*[.!]?$""",
    re.IGNORECASE | re.VERBOSE,
)

# A segment with no verb of its own ("... and discord") reuses the last verb.
_FAST_BARE_RE = re.compile(
    r"""^
    (?:the\s+)?
    (?P<target>.+?)
    (?:\s+(?:window|app|application))?
    \s*[.!]?$""",
    re.IGNORECASE | re.VERBOSE,
)

_FAST_ACTIONS = {
    "switch to": "switch_to",
    "go to": "switch_to",
    "bring up": "switch_to",
    "focus on": "switch_to",
    "focus": "switch_to",
    "show": "switch_to",
    "open": "open_app",
    "launch": "open_app",
    "start": "open_app",
    "run": "open_app",
    "minimize": "minimize",
    "minimise": "minimize",
    "maximize": "maximize",
    "maximise": "maximize",
    "restore": "restore",
    "close": "close_window",
}

# Separators between tasks: commas, semicolons, "&", "and", "then", "also",
# and combinations such as "and then" / ", and".
_FAST_SPLIT_RE = re.compile(
    r"""
    \s*[,;]\s*(?:(?:and\s+then|and\s+also|and|then|also)\s+)?
    |\s+&\s+
    |\s+(?:and\s+then|and\s+also|and|then|also|plus)\s+
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FAST_REJECT_TARGETS = {
    "it", "that", "them", "those", "me", "all", "everything",
    "windows", "my windows", "open windows", "all windows",
}
_FAST_ACTIVE_TARGETS = {"this", "current", "active"}


def _parse_fast_segment(
    segment: str,
    last_action: str | None,
) -> tuple[str, str] | None:
    """
    Parse one command segment into (action, target).
    Returns None when the segment is not simple enough for the fast path.
    """

    segment = segment.strip()

    if not segment:
        return None

    match = _FAST_RE.match(segment)

    if match:
        verb = " ".join(match.group("verb").casefold().split())
        action = _FAST_ACTIONS.get(verb)
        target = match.group("target").strip()
    elif last_action is not None:
        bare = _FAST_BARE_RE.match(segment)

        if not bare:
            return None

        action = last_action
        target = bare.group("target").strip()
    else:
        return None

    if action is None or not target:
        return None

    target_key = target.casefold()

    if target_key in _FAST_REJECT_TARGETS or target_key.startswith("me "):
        return None

    # "open this" has no meaning for the launcher; let the model handle it.
    if action == "open_app" and target_key in _FAST_ACTIVE_TARGETS:
        return None

    if target_key in _FAST_ACTIVE_TARGETS:
        target = "active window"

    return action, target


def _try_fast_path(request: str) -> dict[str, Any] | None:
    """
    Handle unambiguous single or bulk commands without the model.

    Bulk example: "open spotify and discord, close files also minimize vs code".

    Returns None to defer to the LLM (complex phrasing, pronouns, or any
    segment that cannot be parsed). In bulk mode nothing is executed unless
    every segment parses, so a deferral never causes double execution.
    """

    if not FAST_PATH:
        return None

    segments = [s for s in _FAST_SPLIT_RE.split(request.strip()) if s and s.strip()]

    if not segments:
        return None

    # Parse everything first: all-or-nothing.
    plan: list[tuple[str, str]] = []
    last_action: str | None = None

    for segment in segments:
        parsed = _parse_fast_segment(segment, last_action)

        if parsed is None:
            return None

        plan.append(parsed)
        last_action = parsed[0]

    # Single command: unchanged behavior (retry via LLM if nothing matched).
    if len(plan) == 1:
        action, target = plan[0]
        operation = _run_operation(action, {"query": target})
        result = operation.get("result") or {}

        if result.get("no_match"):
            return None

        return _build_response([operation])

    # Bulk: run in order. A failed step is reported but does not stop the rest.
    # Re-running through the LLM here would repeat steps that already executed.
    operations = []

    for index, (action, target) in enumerate(plan):
        if index >= MAX_TOOL_CALLS:
            operations.append({
                "tool": "(skipped)",
                "arguments": {},
                "success": False,
                "summary": "Maximum window tool calls reached.",
            })
            break

        operations.append(_run_operation(action, {"query": target}))

    return _build_response(operations)


# ============================================================
# AGENT
# ============================================================

class WindowAgent:
    """Natural-language window management agent (single LLM routing call)."""

    def __init__(self, model: str = MODEL):
        self.model = model

    def run(
        self,
        user_request: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        request = (user_request or "").strip()

        if not request:
            return {"success": False, "summary": "Window request cannot be empty."}

        fast = _try_fast_path(request)

        if fast is not None:
            logger.debug("Handled by fast path")
            return fast

        return self._run_llm(request, history)

    def _run_llm(
        self,
        request: str,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(request, history),
            "tools": GROQ_TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }

        if "gpt-oss" in self.model:
            kwargs["extra_body"] = {"reasoning_effort": "low"}

        try:
            response = _get_client().chat.completions.create(**kwargs)
        except Exception as exc:
            return {
                "success": False,
                "summary": f"Window model request failed: {exc}",
            }

        if not response.choices:
            return {"success": False, "summary": "Invalid model response."}

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            text = (message.content or "").strip()

            return {
                "success": bool(text),
                "summary": text or "The window agent returned no result.",
            }

        operations: list[dict[str, Any]] = []

        for index, tool_call in enumerate(tool_calls):
            if index >= MAX_TOOL_CALLS:
                operations.append({
                    "tool": "(skipped)",
                    "arguments": {},
                    "success": False,
                    "summary": "Maximum window tool calls reached.",
                })
                break

            logger.debug("Tool #%d: %s", index + 1, tool_call.function.name)

            operations.append(
                _run_operation(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
            )

        return _build_response(operations)


# ============================================================
# SINGLETON
# ============================================================

window_agent = WindowAgent()


# ============================================================
# ZOE INTERFACE
# ============================================================

def run_application_agent(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    ZOE integration entry point.

    Returns {"success": bool, "summary": str}, plus "operations"
    when ZOE_WINDOW_DEBUG=1.
    """

    try:
        return window_agent.run(user_request=query, history=history)
    except Exception as exc:
        logger.exception("Window agent crashed")
        return {
            "success": False,
            "summary": f"Window agent error: {type(exc).__name__}: {exc}",
        }


def run_window_agent_text(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Return only the result summary."""

    return run_application_agent(query, history=history).get("summary", "")


# ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":
    print("ZOE Window Agent (Linux)")
    print("Type 'exit' to quit.")

    chat_history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("\nYou > ").strip()

            if user_input.casefold() in {"exit", "quit"}:
                break

            if not user_input:
                continue

            started = time.perf_counter()
            outcome = run_application_agent(user_input, history=chat_history)
            elapsed = time.perf_counter() - started

            chat_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": outcome.get("summary", "")},
            ])

            print(outcome.get("summary", ""))

            if DEBUG:
                print(f"[{elapsed:.2f}s]")
                print(json.dumps(outcome, indent=2, ensure_ascii=False, default=str))

        except KeyboardInterrupt:
            break

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")