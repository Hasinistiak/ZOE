from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import shutil
import string
import subprocess
import threading
import time
import winreg
from collections import deque
from ctypes import wintypes
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from dotenv import load_dotenv

import pywintypes
import win32api
import win32con
import win32gui
import win32process


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
PROCESS_CACHE_TTL_S = 60.0

# App launcher
APP_CACHE_TTL_S = 300.0
APP_LAUNCH_WAIT_S = float(os.getenv("ZOE_APP_LAUNCH_WAIT", "3"))
APP_SCAN_TIMEOUT_S = float(os.getenv("ZOE_APP_SCAN_TIMEOUT", "6"))
APP_SCAN_MAX_DEPTH = 5
APP_SCAN_MISS_TTL_S = 60.0
APP_FUZZY_CUTOFF = 0.8
APP_DEEP_SCAN = os.getenv("ZOE_APP_DEEP_SCAN", "1").lower() in {"1", "true", "yes"}
APP_PREWARM = os.getenv("ZOE_APP_PREWARM", "1").lower() in {"1", "true", "yes"}

DEBUG = os.getenv("ZOE_WINDOW_DEBUG", "").lower() in {"1", "true", "yes"}
FAST_PATH = os.getenv("ZOE_WINDOW_FAST_PATH", "1").lower() in {"1", "true", "yes"}

# Spoken shortcuts -> names that actually appear in window titles,
# process names and the Start Menu. Keys must be in normalized form.
APP_ALIASES = {
    "vs code": "visual studio code",
    "vscode": "visual studio code",
    "files": "explorer",
    "file": "explorer",
    "file explorer": "explorer",
    "windows explorer": "explorer",
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
# WIN32 CONSTANTS AND BINDINGS
# ============================================================

VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x0002
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_user32 = ctypes.WinDLL("user32", use_last_error=True) if os.name == "nt" else None
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None

if os.name == "nt":
    _user32.keybd_event.restype = None
    _user32.keybd_event.argtypes = [
        wintypes.BYTE,
        wintypes.BYTE,
        wintypes.DWORD,
        ctypes.c_size_t,
    ]
    _user32.AttachThreadInput.restype = wintypes.BOOL
    _user32.AttachThreadInput.argtypes = [
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.BOOL,
    ]

    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    _kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    _kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    try:
        _dwmapi = ctypes.WinDLL("dwmapi")
        _dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
        _dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
    except OSError:
        _dwmapi = None
else:
    _dwmapi = None

DWMWA_CLOAKED = 14


def _is_cloaked(hwnd: int) -> bool:
    """True for ghost UWP windows and windows hidden by the shell."""

    if _dwmapi is None:
        return False

    try:
        value = ctypes.c_int(0)
        hr = _dwmapi.DwmGetWindowAttribute(
            hwnd,
            DWMWA_CLOAKED,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )
        return hr == 0 and value.value != 0
    except Exception:
        return False


# ============================================================
# WINDOW INFORMATION
# ============================================================

@dataclass
class WindowInfo:
    hwnd: int
    title: str
    process_id: int
    process_name: str
    visible: bool
    minimized: bool
    maximized: bool
    active: bool


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


class WindowManager:
    """Inspect and control top-level Windows application windows."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("WindowManager requires Windows.")

        self._process_cache: dict[int, tuple[str, float]] = {}
        self._cache_lock = threading.Lock()

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _process_name(self, pid: int) -> str:
        now = time.monotonic()

        with self._cache_lock:
            cached = self._process_cache.get(pid)

            if cached and now - cached[1] < PROCESS_CACHE_TTL_S:
                return cached[0]

            # Opportunistic cleanup keeps the cache bounded.
            if len(self._process_cache) > 512:
                self._process_cache = {
                    p: v
                    for p, v in self._process_cache.items()
                    if now - v[1] < PROCESS_CACHE_TTL_S
                }

        name = "Unknown"
        handle = None

        try:
            handle = _kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                pid,
            )

            if handle:
                buffer = ctypes.create_unicode_buffer(32768)
                size = wintypes.DWORD(len(buffer))

                if _kernel32.QueryFullProcessImageNameW(
                    handle,
                    0,
                    buffer,
                    ctypes.byref(size),
                ):
                    name = os.path.basename(buffer.value)

        except (OSError, AttributeError):
            pass

        finally:
            if handle:
                _kernel32.CloseHandle(handle)

        with self._cache_lock:
            self._process_cache[pid] = (name, now)

        return name

    def _window_info(self, hwnd: int) -> WindowInfo | None:
        try:
            if not win32gui.IsWindow(hwnd):
                return None

            title = win32gui.GetWindowText(hwnd).strip()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            show_state = win32gui.GetWindowPlacement(hwnd)[1]

            return WindowInfo(
                hwnd=hwnd,
                title=title or UNTITLED,
                process_id=pid,
                process_name=self._process_name(pid),
                visible=bool(win32gui.IsWindowVisible(hwnd)),
                minimized=show_state == win32con.SW_SHOWMINIMIZED,
                maximized=show_state == win32con.SW_SHOWMAXIMIZED,
                active=hwnd == win32gui.GetForegroundWindow(),
            )

        except (pywintypes.error, OSError):
            return None

    @staticmethod
    def _validate(hwnd: int) -> None:
        if not isinstance(hwnd, int) or not win32gui.IsWindow(hwnd):
            raise ValueError(f"Invalid window handle: {hwnd}")

    @staticmethod
    def _is_maximized(hwnd: int) -> bool:
        # win32gui has no IsZoomed; read the show state instead.
        return win32gui.GetWindowPlacement(hwnd)[1] == win32con.SW_SHOWMAXIMIZED

    @staticmethod
    def _is_app_window(hwnd: int) -> bool:
        """Exclude tool windows (tooltips, floating palettes) from listings."""

        try:
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        except pywintypes.error:
            return False

        if ex_style & win32con.WS_EX_TOOLWINDOW and not (
            ex_style & win32con.WS_EX_APPWINDOW
        ):
            return False

        return True

    # --------------------------------------------------------
    # INSPECT WINDOWS
    # --------------------------------------------------------

    def list_windows(
        self,
        include_hidden: bool = False,
        include_untitled: bool = False,
    ) -> list[WindowInfo]:
        """List top-level application windows."""

        windows: list[WindowInfo] = []

        def callback(hwnd: int, _: Any) -> bool:
            try:
                if not include_hidden:
                    if not win32gui.IsWindowVisible(hwnd):
                        return True
                    if not self._is_app_window(hwnd):
                        return True
                    if _is_cloaked(hwnd):
                        return True

                info = self._window_info(hwnd)

                if info is None:
                    return True

                if not include_untitled and info.title == UNTITLED:
                    return True

                windows.append(info)

            except Exception:
                logger.debug("Skipping window %s", hwnd, exc_info=True)

            return True

        try:
            win32gui.EnumWindows(callback, None)
        except pywintypes.error:
            logger.debug("EnumWindows failed", exc_info=True)

        return sorted(
            windows,
            key=lambda w: (not w.active, w.title.casefold()),
        )

    def get_active_window(self) -> WindowInfo | None:
        hwnd = win32gui.GetForegroundWindow()
        return self._window_info(hwnd) if hwnd else None

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
        return APP_ALIASES.get(text, text)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        tokens = set()

        for token in text.split():
            # Cheap singularization so "files" matches "file explorer".
            if len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            tokens.add(token)

        return tokens

    @staticmethod
    def _strip_filler(text: str) -> str:
        return " ".join(t for t in text.split() if t not in FILLER_WORDS)

    def _score_matches(
        self,
        query: str,
        windows: list[WindowInfo],
    ) -> list[tuple[int, WindowInfo]]:
        query_tokens = self._tokens(query)
        matches: list[tuple[int, WindowInfo]] = []

        for window in windows:
            title = self._normalize(window.title)
            process = self._normalize(os.path.splitext(window.process_name)[0])

            if query == title or query == process:
                score = 100
            elif title.startswith(query):
                score = 90
            elif query in title:
                score = 80
            elif process.startswith(query):
                score = 75
            elif query in process:
                score = 70
            elif query_tokens and query_tokens <= (
                self._tokens(title) | self._tokens(process)
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
                        "hwnd": w.hwnd,
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
    def _wait_until(predicate: Any, timeout: float = 0.3, step: float = 0.03) -> bool:
        deadline = time.monotonic() + timeout

        while True:
            try:
                if predicate():
                    return True
            except pywintypes.error:
                pass

            if time.monotonic() >= deadline:
                return False

            time.sleep(step)

    def _try_set_foreground(self, hwnd: int, use_alt: bool) -> bool:
        """
        One attempt at taking the foreground. Windows blocks background
        processes from calling SetForegroundWindow, so we attach to the
        foreground thread's input queue and, as a last resort, tap Alt
        to lift the foreground lock.
        """

        foreground = win32gui.GetForegroundWindow()

        if foreground == hwnd:
            return True

        current_thread = win32api.GetCurrentThreadId()
        foreground_thread = (
            win32process.GetWindowThreadProcessId(foreground)[0]
            if foreground
            else 0
        )
        attached = False

        try:
            if use_alt:
                _user32.keybd_event(VK_MENU, 0, 0, 0)
                _user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

            if foreground_thread and foreground_thread != current_thread:
                attached = bool(
                    _user32.AttachThreadInput(
                        current_thread,
                        foreground_thread,
                        True,
                    )
                )

            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)

        except pywintypes.error:
            logger.debug("SetForegroundWindow raised", exc_info=True)

        finally:
            if attached:
                _user32.AttachThreadInput(current_thread, foreground_thread, False)

        return self._wait_until(lambda: win32gui.GetForegroundWindow() == hwnd)

    def switch_to(self, hwnd: int) -> dict[str, Any]:
        self._validate(hwnd)

        if not win32gui.IsWindowVisible(hwnd):
            return {"success": False, "message": "Cannot focus a hidden window."}

        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

        active = self._try_set_foreground(hwnd, use_alt=False) or (
            self._try_set_foreground(hwnd, use_alt=True)
        )

        return {
            "success": active,
            "message": (
                "Window is now active."
                if active
                else "Windows did not grant foreground focus."
            ),
        }

    def minimize(self, hwnd: int) -> dict[str, Any]:
        self._validate(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        done = self._wait_until(lambda: bool(win32gui.IsIconic(hwnd)))

        return {
            "success": done,
            "message": "Window minimized." if done else "Window could not be minimized.",
        }

    def maximize(self, hwnd: int) -> dict[str, Any]:
        self._validate(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        done = self._wait_until(lambda: self._is_maximized(hwnd))

        return {
            "success": done,
            "message": "Window maximized." if done else "Window could not be maximized.",
        }

    def restore(self, hwnd: int) -> dict[str, Any]:
        self._validate(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        done = self._wait_until(
            lambda: not win32gui.IsIconic(hwnd) and not self._is_maximized(hwnd)
        )

        return {
            "success": done,
            "message": "Window restored." if done else "Window could not be restored.",
        }

    def toggle_maximize(self, hwnd: int) -> dict[str, Any]:
        self._validate(hwnd)

        was_maximized = self._is_maximized(hwnd)
        result = self.restore(hwnd) if was_maximized else self.maximize(hwnd)
        result["action"] = "restore" if was_maximized else "maximize"

        return result

    def close(self, hwnd: int) -> dict[str, Any]:
        """
        Request a normal window close (WM_CLOSE).

        This never force-terminates the application. The application may
        prompt to save work or refuse to close.
        """

        self._validate(hwnd)

        # PostMessage returns None on success and raises on failure.
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except pywintypes.error as exc:
            return {
                "success": False,
                "message": f"Close request failed: {exc.strerror or exc}",
            }

        closed = self._wait_until(lambda: not win32gui.IsWindow(hwnd), timeout=0.4)

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

CREATE_NO_WINDOW = 0x08000000

# Start Menu entries that match a name but are never what the user means.
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

# Installer/helper executables ignored by the drive scan.
EXE_JUNK_PARTS = ("unins", "setup", "install", "updat", "crash", "helper", "redist")

SCAN_SKIP_DIRS = {
    "windows",
    "$recycle.bin",
    "system volume information",
    "node_modules",
    ".git",
    "__pycache__",
    "site-packages",
    "winsxs",
}


class AppLauncher:
    """
    Find and launch applications by name.

    Lookup order (fast to slow):
      0. Absolute path to a file, folder or shortcut
      1. Start Menu index (Win32 + Microsoft Store apps, any install location)
      2. "App Paths" registry, then PATH
      3. Fuzzy Start Menu match (typos, partial names)
      4. Bounded drive scan for portable / unregistered executables
    """

    def __init__(self, windows: WindowManager) -> None:
        self._windows = windows
        self._apps_cache: tuple[dict[str, str], float] | None = None
        self._apps_lock = threading.Lock()
        self._scan_misses: dict[str, float] = {}
        self._scan_lock = threading.Lock()

    # --------------------------------------------------------
    # START MENU INDEX
    # --------------------------------------------------------

    @staticmethod
    def _query_start_apps() -> dict[str, str]:
        command = (
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
            "Get-StartApps | ConvertTo-Json -Compress"
        )

        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
                creationflags=CREATE_NO_WINDOW,
            )
            raw = completed.stdout.strip()
            data = json.loads(raw) if raw else []

        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            logger.debug("Get-StartApps failed", exc_info=True)
            return {}

        if isinstance(data, dict):
            data = [data]

        return {
            item["Name"]: item["AppID"]
            for item in data
            if isinstance(item, dict) and item.get("Name") and item.get("AppID")
        }

    def start_apps(self, force: bool = False) -> dict[str, str]:
        """{display name: AppID} for everything in the Start Menu (cached)."""

        # The lock is held during the PowerShell call so concurrent callers
        # wait for one query instead of launching several.
        with self._apps_lock:
            now = time.monotonic()
            cached = self._apps_cache

            if cached and not force and now - cached[1] < APP_CACHE_TTL_S:
                return cached[0]

            apps = self._query_start_apps()

            # Keep serving the old index if PowerShell hiccups.
            if not apps and cached:
                return cached[0]

            self._apps_cache = (apps, now)
            return apps

    def prewarm(self) -> None:
        try:
            self.start_apps()
        except Exception:
            logger.debug("App index prewarm failed", exc_info=True)

    def _start_index(self) -> list[tuple[str, str, str]]:
        """[(display name, normalized name, AppID)]"""

        index = []

        for name, app_id in self.start_apps().items():
            normalized = WindowManager._normalize(name)

            if normalized:
                index.append((name, normalized, app_id))

        return index

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

    def _match_start_app(self, query: str) -> dict[str, Any] | None:
        """Exact / word / prefix / substring / token matching."""

        query_tokens = WindowManager._tokens(query)
        scored = []

        for name, normalized, app_id in self._start_index():
            score = self._score(query, normalized, query_tokens)

            if score:
                scored.append((score, name, app_id))

        if not scored:
            return None

        best_score = max(item[0] for item in scored)

        # Shortest display name wins ("Google Chrome" over "Chrome Remote Desktop").
        best = sorted(
            (item for item in scored if item[0] == best_score),
            key=lambda item: (len(item[1]), item[1].casefold()),
        )

        if len(best) > 1 and len(best[1][1]) == len(best[0][1]):
            tied = [item for item in best if len(item[1]) == len(best[0][1])]

            return {
                "success": False,
                "ambiguous": True,
                "message": (
                    f"Multiple applications match '{query}'. "
                    "Specify a more precise name."
                ),
                "matches": [{"title": item[1]} for item in tied[:8]],
            }

        return {"success": True, "name": best[0][1], "app_id": best[0][2]}

    def _fuzzy_start_app(
        self,
        query: str,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Typo-tolerant match. Returns (match or None, closest names)."""

        if len(query) < 3:
            return None, []

        ranked = []

        for name, normalized, app_id in self._start_index():
            if self._is_junk(normalized, query):
                continue

            ratio = max(
                SequenceMatcher(None, query, part).ratio()
                for part in {normalized, *normalized.split()}
            )
            ranked.append((ratio, name, app_id))

        ranked.sort(key=lambda item: (-item[0], len(item[1])))
        suggestions = [name for ratio, name, _ in ranked[:3] if ratio >= 0.5]

        if ranked and ranked[0][0] >= APP_FUZZY_CUTOFF:
            return (
                {"success": True, "name": ranked[0][1], "app_id": ranked[0][2]},
                suggestions,
            )

        return None, suggestions

    @staticmethod
    def _app_paths_lookup(name: str) -> str | None:
        """Executables registered under App Paths (what Win+R resolves)."""

        if not name or any(char in name for char in "\\/"):
            return None

        exe = name if name.lower().endswith(".exe") else f"{name}.exe"
        key_path = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{exe}"

        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                try:
                    with winreg.OpenKey(
                        hive,
                        key_path,
                        0,
                        winreg.KEY_READ | view,
                    ) as key:
                        value, _ = winreg.QueryValueEx(key, None)

                except OSError:
                    continue

                path = os.path.expandvars(str(value)).strip('"')

                if os.path.isfile(path):
                    return path

        return None

    # --------------------------------------------------------
    # DRIVE SCAN (portable / unregistered apps)
    # --------------------------------------------------------

    @staticmethod
    def _scan_roots() -> list[str]:
        env = os.environ
        home = os.path.expanduser("~")
        local_appdata = env.get("LOCALAPPDATA", "")

        candidates = [
            *(os.path.join(home, folder) for folder in ("Downloads", "Desktop", "Documents")),
            env.get("ProgramFiles", ""),
            env.get("ProgramFiles(x86)", ""),
            os.path.join(local_appdata, "Programs") if local_appdata else "",
        ]

        # Other drives are scanned from the root; the system drive is not,
        # because everything useful on it is covered above or by the Start Menu.
        system_drive = env.get("SystemDrive", "C:").upper()

        for letter in string.ascii_uppercase:
            drive = f"{letter}:"

            if drive != system_drive and os.path.isdir(f"{drive}\\"):
                candidates.append(f"{drive}\\")

        roots: list[str] = []
        seen: set[str] = set()

        for path in candidates:
            key = os.path.normcase(path)

            if path and key not in seen and os.path.isdir(path):
                seen.add(key)
                roots.append(path)

        return roots

    def _scan_for_exe(self, query: str) -> str | None:
        """Time-boxed, shallow-first search for a matching .exe."""

        deadline = time.monotonic() + APP_SCAN_TIMEOUT_S
        query_is_junk = any(part in query for part in EXE_JUNK_PARTS)
        suffix_match = re.compile(rf"{re.escape(query)}[\s\d]*(?:x64|x86|portable)?")
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
                                and entry.name.casefold() not in SCAN_SKIP_DIRS
                            ):
                                queue.append((entry.path, depth + 1))
                            continue

                        if not entry.name.lower().endswith(".exe"):
                            continue

                    except OSError:
                        continue

                    stem = WindowManager._normalize(entry.name[:-4])

                    if not stem:
                        continue

                    if not query_is_junk and any(
                        part in stem for part in EXE_JUNK_PARTS
                    ):
                        continue

                    if stem == query:
                        return entry.path

                    if f" {query} " in f" {stem} " or suffix_match.fullmatch(stem):
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
    def _launch(target: str) -> str | None:
        """ShellExecute the target. Returns an error string, or None on success."""

        try:
            # ShellExecute (not Popen) so UAC prompts, shortcuts, folders,
            # and shell:AppsFolder\\<AppID> targets all behave natively.
            os.startfile(target)
            return None

        except OSError as exc:
            logger.debug("Launch failed for %s", target, exc_info=True)
            return exc.strerror or str(exc)

    def _wait_for_window(
        self,
        before: set[int],
        foreground_before: int,
    ) -> WindowInfo | None:
        """
        Wait for the launched app to show a titled window, so follow-up
        actions in the same request (maximize, move, close) can resolve it.
        """

        deadline = time.monotonic() + APP_LAUNCH_WAIT_S

        while True:
            for window in self._windows.list_windows():
                if window.hwnd not in before:
                    return window

            # Single-instance apps raise their existing window instead.
            active = self._windows.get_active_window()

            if (
                active is not None
                and active.hwnd != foreground_before
                and active.title != UNTITLED
            ):
                return active

            if time.monotonic() >= deadline:
                return None

            time.sleep(0.15)

    def _start(self, target: str, label: str) -> dict[str, Any]:
        before = {w.hwnd for w in self._windows.list_windows()}
        foreground = self._windows.get_active_window()
        foreground_before = foreground.hwnd if foreground else 0

        error = self._launch(target)

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
                "hwnd": window.hwnd,
            }

        return result

    # --------------------------------------------------------
    # PUBLIC ENTRY POINT
    # --------------------------------------------------------

    def open(self, query: str) -> dict[str, Any]:
        raw = (query or "").strip().strip('"')

        if not raw:
            return {"success": False, "message": "An application name is required."}

        # 0. Explicit absolute path (file, folder, shortcut)
        expanded = os.path.expandvars(raw)

        if os.path.isabs(expanded) and os.path.exists(expanded):
            label = os.path.basename(expanded.rstrip("\\/")) or expanded
            return self._start(expanded, label)

        name = raw[:-4] if raw.lower().endswith(".exe") else raw
        normalized = WindowManager._normalize(name)
        normalized = WindowManager._strip_filler(normalized) or normalized
        normalized = WindowManager._alias(normalized)

        # 1. Start Menu: classic desktop apps and Store apps, wherever installed
        match = self._match_start_app(normalized)

        if match is not None:
            if not match["success"]:
                return match

            return self._start(f"shell:AppsFolder\\{match['app_id']}", match["name"])

        # 2. App Paths registry, then PATH
        path = self._app_paths_lookup(name) or shutil.which(name)

        if path:
            return self._start(path, os.path.splitext(os.path.basename(path))[0])

        # 3. Typos and partial names
        fuzzy, suggestions = self._fuzzy_start_app(normalized)

        if fuzzy is not None:
            return self._start(f"shell:AppsFolder\\{fuzzy['app_id']}", fuzzy["name"])

        # 4. Portable / unregistered executables
        path = self._deep_search(normalized)

        if path:
            return self._start(path, os.path.splitext(os.path.basename(path))[0])

        message = f"Could not find an application matching '{raw}'."

        if suggestions:
            message += f" Closest matches: {', '.join(suggestions)}."

        return {"success": False, "no_match": True, "message": message}


# ============================================================
# SINGLETON APP LAUNCHER
# ============================================================

app_launcher = AppLauncher(window_manager)

if APP_PREWARM:
    # Build the Start Menu index in the background so the first
    # "open X" request does not pay the PowerShell startup cost.
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
    hwnd = window["hwnd"]

    actions = {
        "switch_to": window_manager.switch_to,
        "minimize": window_manager.minimize,
        "maximize": window_manager.maximize,
        "restore": window_manager.restore,
        "toggle_maximize": window_manager.toggle_maximize,
        "close_window": window_manager.close,
    }

    try:
        result = actions[action](hwnd)
    except ValueError as exc:
        # The window vanished between resolve and act.
        return {"success": False, "message": str(exc)}

    return {
        **result,
        "action": result.get("action", action),
        "window": {
            "title": window["title"],
            "process_name": window["process_name"],
            "hwnd": hwnd,
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
            "e.g. 'chrome', 'visual studio code', 'file explorer', or "
            "'active window'. Omit filler words like 'window' or 'app'."
        ),
    }
}

APP_QUERY = {
    "query": {
        "type": "string",
        "description": (
            "Application name as the user would say it, e.g. 'chrome', "
            "'spotify', 'notepad', 'visual studio code', 'task manager'. "
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
    _tool("get_active_window", "Identify the foreground window.", {}),
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
        "Request a normal close (WM_CLOSE) of a window. Never force-kills.",
        WINDOW_QUERY,
        ["query"],
    ),
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's Windows window-routing agent. You translate a request into
tool calls. Your tool results are summarized by code, not by you.

RULES
1. Emit ALL needed tool calls in a single response, in the order requested.
2. Use the application name or distinctive title words as the query.
   Drop filler words such as "window", "app", "the".
3. "this window" or "the active window" means query "active window".
4. If the user asks which apps or windows are open, call list_windows.
5. To launch an application (open, launch, start, run), call open_app with
   the application name. Use switch_to instead when the user wants an app
   that is already open. open_app waits for the new window to appear, so
   follow-up actions can come right after it, e.g. "open notepad and
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
    """Natural-language Windows management agent (single LLM routing call)."""

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
    print("ZOE Windows Agent")
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