from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections import deque
from typing import Any

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.getenv(
    "ZOE_VITALS_MODEL",
    "openai/gpt-oss-20b",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY3")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY3 is not set.")

client = Groq(api_key=GROQ_API_KEY)

# Network transport to the ESP32 (replaces the USB serial port).
# Use the mDNS name, or the IP printed on the monitor's LCD at boot.
DEVICE_HOST = os.getenv("ZOE_VITALS_HOST", "zoe-vitals.local").strip()

DEVICE_PORT = int(os.getenv("ZOE_VITALS_TCP_PORT", "3333"))

CONNECT_TIMEOUT = float(os.getenv("ZOE_VITALS_CONNECT_TIMEOUT", "4"))

# Per-recv timeout; also the polling granularity of the read loop.
SOCKET_TIMEOUT = float(os.getenv("ZOE_VITALS_SOCKET_TIMEOUT", "0.25"))

DEFAULT_HR_DURATION = int(
    os.getenv("ZOE_VITALS_HR_SECONDS", "20")
)

DEFAULT_SPO2_TIMEOUT = int(
    os.getenv("ZOE_VITALS_SPO2_SECONDS", "60")
)

MAX_AGENT_STEPS = int(
    os.getenv("ZOE_VITALS_MAX_STEPS", "5")
)

MAX_TOOL_CALLS = int(
    os.getenv("ZOE_VITALS_MAX_TOOL_CALLS", "6")
)

MAX_CONTEXT_MESSAGES = int(
    os.getenv("ZOE_VITALS_MAX_CONTEXT_MESSAGES", "8")
)

MAX_SESSION_READINGS = int(
    os.getenv("ZOE_VITALS_MAX_SESSION_READINGS", "100")
)

MAX_SESSION_EVENTS = int(
    os.getenv("ZOE_VITALS_MAX_SESSION_EVENTS", "80")
)

client_lock = threading.RLock()


# ============================================================
# NETWORK DEVICE
# ============================================================

class VitalsDevice:
    """
    TCP interface for the ESP32 Vital Monitor.

    Commands (one per line):
        MODE HR
        MODE SPO2
        START
        STOP
        STATUS

    Telemetry:
        One JSON object per line.

    Non-JSON diagnostic lines are ignored by the telemetry
    collector. They may still be printed for troubleshooting.

    The firmware accepts one client at a time; a new connection
    replaces the previous one.
    """

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._rx = bytearray()
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # Connection
    # --------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        with self._lock:
            if self._sock is not None:
                return {
                    "success": True,
                    "connected": True,
                    "host": DEVICE_HOST,
                    "port": DEVICE_PORT,
                }

            try:
                sock = socket.create_connection(
                    (DEVICE_HOST, DEVICE_PORT),
                    timeout=CONNECT_TIMEOUT,
                )

                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.settimeout(SOCKET_TIMEOUT)

                self._sock = sock
                self._rx.clear()

                return {
                    "success": True,
                    "connected": True,
                    "host": DEVICE_HOST,
                    "port": DEVICE_PORT,
                }

            except OSError as exc:
                self._sock = None

                return {
                    "success": False,
                    "connected": False,
                    "error": (
                        f"Could not reach the vitals monitor at "
                        f"{DEVICE_HOST}:{DEVICE_PORT} ({exc}). Check that it "
                        f"is powered, on the same network, and that "
                        f"ZOE_VITALS_HOST is correct."
                    ),
                }

    def disconnect(self) -> None:
        with self._lock:
            self._drop()

    def _drop(self) -> None:
        sock, self._sock = self._sock, None
        self._rx.clear()

        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _ensure_connected(self) -> socket.socket:
        if self._sock is not None:
            return self._sock

        result = self.connect()

        if not result.get("success"):
            raise ConnectionError(result.get("error", "Connection failed."))

        if self._sock is None:
            raise ConnectionError("Connection was not established.")

        return self._sock

    def flush_input(self) -> None:
        """
        Discard anything the device sent while nobody was reading,
        so a new session never sees stale readings or results.
        """

        with self._lock:
            self._rx.clear()

            if self._sock is None:
                return

            sock = self._sock

            try:
                sock.setblocking(False)

                while True:
                    try:
                        chunk = sock.recv(4096)
                    except (BlockingIOError, InterruptedError):
                        break

                    if not chunk:
                        # Peer closed the connection.
                        self._drop()
                        return
            except OSError:
                self._drop()
                return
            finally:
                if self._sock is not None:
                    try:
                        self._sock.settimeout(SOCKET_TIMEOUT)
                    except OSError:
                        pass

    # --------------------------------------------------------
    # Command transmission
    # --------------------------------------------------------

    def send_command(self, command: str) -> dict[str, Any]:
        command = command.strip().upper()

        allowed = {
            "MODE HR",
            "MODE HEART_RATE",
            "MODE SPO2",
            "START",
            "STOP",
            "STATUS",
            "HELP",
        }

        if command not in allowed:
            return {
                "success": False,
                "error": f"Unsupported command: {command}",
            }

        payload = (command + "\n").encode("ascii")

        with self._lock:
            last_error = "Unknown error."

            # One retry on a fresh connection covers a stale socket
            # (monitor rebooted or WiFi dropped since the last use).
            for _ in range(2):
                try:
                    sock = self._ensure_connected()
                    sock.sendall(payload)

                    return {
                        "success": True,
                        "command": command,
                        "host": DEVICE_HOST,
                    }

                except (OSError, ConnectionError) as exc:
                    last_error = str(exc)
                    self._drop()

            return {
                "success": False,
                "error": last_error,
            }

    # --------------------------------------------------------
    # JSON Lines reader
    # --------------------------------------------------------

    def _read_line(self) -> str | None:
        """
        Return one decoded line, or None if nothing complete
        arrived within SOCKET_TIMEOUT.

        Raises ConnectionError if the link is lost.
        """

        sock = self._ensure_connected()

        while True:
            idx = self._rx.find(b"\n")

            if idx >= 0:
                raw = bytes(self._rx[:idx])
                del self._rx[: idx + 1]
                return raw.decode("utf-8", errors="replace").strip()

            if len(self._rx) > 65536:
                self._rx.clear()

            try:
                chunk = sock.recv(4096)

            except socket.timeout:
                return None

            except OSError as exc:
                self._drop()
                raise ConnectionError(str(exc)) from exc

            if not chunk:
                self._drop()
                raise ConnectionError("The monitor closed the connection.")

            self._rx.extend(chunk)

    def read_json_event(self) -> dict[str, Any] | None:
        """
        Read one line and return it only if it is a JSON object.

        Diagnostic text from the firmware is skipped.

        Raises ConnectionError if the link is lost.
        """

        with self._lock:
            line = self._read_line()

        if not line:
            return None

        if not line.startswith("{"):
            print(f"[Vitals Device] {line}")
            return None

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(f"[Vitals Device] Invalid JSON: {line}")
            return None

        if not isinstance(payload, dict):
            return None

        return payload

    # --------------------------------------------------------
    # Wait for matching event
    # --------------------------------------------------------

    def wait_for_event(
        self,
        event_type: str,
        timeout: float = 3.0,
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                event = self.read_json_event()
            except ConnectionError:
                return None

            if event and event.get("type") == event_type:
                return event

        return None

    # --------------------------------------------------------
    # Collect a complete measurement session
    # --------------------------------------------------------

    def collect_session(
        self,
        mode: str,
        duration_seconds: int,
    ) -> dict[str, Any]:
        """
        Switch to the requested mode, start monitoring, and
        collect intermediate JSON readings and events.

        SpO2 sessions stop early when a final result event
        arrives. Otherwise, collection ends at the timeout.
        """

        mode = mode.strip().upper()

        if mode not in {"HR", "SPO2"}:
            return {
                "success": False,
                "error": "Mode must be HR or SPO2.",
            }

        if not 3 <= duration_seconds <= 60:
            return {
                "success": False,
                "error": "Duration must be between 3 and 60 seconds.",
            }

        with self._lock:
            readings: deque[dict[str, Any]] = deque(
                maxlen=MAX_SESSION_READINGS
            )

            events: deque[dict[str, Any]] = deque(
                maxlen=MAX_SESSION_EVENTS
            )

            # Make sure we are connected, then drop anything stale
            # (old readings, a result from a previous timeout, etc.).
            try:
                self._ensure_connected()
            except ConnectionError as exc:
                return {"success": False, "error": str(exc)}

            self.flush_input()

            start_command = self.send_command(
                "MODE HR" if mode == "HR" else "MODE SPO2"
            )

            if not start_command.get("success"):
                return start_command

            # MODE resets the device to a fresh scan. START
            # explicitly enables acquisition.
            start_result = self.send_command("START")

            if not start_result.get("success"):
                return start_result

            started_at = time.monotonic()
            deadline = started_at + duration_seconds
            final_result: dict[str, Any] | None = None
            connection_error: str | None = None

            while time.monotonic() < deadline:
                try:
                    event = self.read_json_event()
                except ConnectionError as exc:
                    connection_error = str(exc)
                    break

                if event is None:
                    continue

                event_type = str(event.get("type", "")).lower()

                if event_type == "reading":
                    readings.append(event)

                elif event_type in {"event", "status", "result", "beat"}:
                    events.append(event)

                if event_type == "result":
                    final_result = event
                    break

                # A result may be embedded in an event depending
                # on the firmware protocol.
                if (
                    event.get("event") in {"spo2_result", "measurement_complete"}
                    and mode == "SPO2"
                ):
                    final_result = event
                    break

            elapsed = round(time.monotonic() - started_at, 2)

            if connection_error and final_result is None:
                return {
                    "success": False,
                    "error": (
                        "Connection to the monitor was lost during the "
                        f"measurement: {connection_error}"
                    ),
                    "mode": mode,
                    "duration_actual_seconds": elapsed,
                    "reading_count": len(readings),
                    "readings": list(readings),
                    "events": list(events),
                }

            return {
                "success": True,
                "mode": mode,
                "duration_requested_seconds": duration_seconds,
                "duration_actual_seconds": elapsed,
                "reading_count": len(readings),
                "readings": list(readings),
                "events": list(events),
                "final_result": final_result,
                "completed": final_result is not None,
                "note": (
                    "A timeout is not a failed medical measurement by itself. "
                    "Check the readings, state, and signal quality."
                ),
            }


device = VitalsDevice()


# ============================================================
# VITALS TOOLS
# ============================================================

def get_device_status() -> dict[str, Any]:
    """
    Request the ESP32's current mode and monitoring state.
    """

    device.flush_input()

    command_result = device.send_command("STATUS")

    if not command_result.get("success"):
        return command_result

    status = device.wait_for_event(
        "status",
        timeout=3.0,
    )

    if status is None:
        return {
            "success": False,
            "error": (
                "No JSON status response was received. "
                "Check the network connection and firmware protocol."
            ),
        }

    return {
        "success": True,
        "status": status,
    }


def set_monitor_mode(mode: str) -> dict[str, Any]:
    """
    Switch the ESP32 into heart-rate or oxygen measurement mode.
    """

    normalized = mode.strip().upper()

    if normalized in {"HR", "HEART RATE", "HEART_RATE", "HEARTRATE"}:
        command = "MODE HR"

    elif normalized in {"SPO2", "OXYGEN", "BLOOD OXYGEN", "BLOOD_OXYGEN"}:
        command = "MODE SPO2"

    else:
        return {
            "success": False,
            "error": "Unsupported mode. Choose HR or SPO2.",
        }

    result = device.send_command(command)

    if result.get("success"):
        result["selected_mode"] = normalized

    return result


def start_monitoring() -> dict[str, Any]:
    return device.send_command("START")


def stop_monitoring() -> dict[str, Any]:
    return device.send_command("STOP")


def collect_vitals_readings(
    mode: str,
    duration_seconds: int | None = None,
) -> dict[str, Any]:
    """
    Run a measurement and return all received JSON readings.

    Heart rate defaults to a shorter collection period.
    SpO2 is allowed a longer period for its measurement window.
    """

    normalized = mode.strip().upper()

    if normalized in {"HR", "HEART RATE", "HEART_RATE"}:
        normalized = "HR"
        duration = (
            duration_seconds
            if duration_seconds is not None
            else DEFAULT_HR_DURATION
        )

    elif normalized in {"SPO2", "OXYGEN", "BLOOD OXYGEN"}:
        normalized = "SPO2"
        duration = (
            duration_seconds
            if duration_seconds is not None
            else DEFAULT_SPO2_TIMEOUT
        )

    else:
        return {
            "success": False,
            "error": "Mode must be HR or SPO2.",
        }

    return device.collect_session(
        mode=normalized,
        duration_seconds=int(duration),
    )


TOOLS = {
    "get_device_status": get_device_status,
    "set_monitor_mode": set_monitor_mode,
    "start_monitoring": start_monitoring,
    "stop_monitoring": stop_monitoring,
    "collect_vitals_readings": collect_vitals_readings,
}


# ============================================================
# GROQ TOOL DEFINITIONS
# ============================================================

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_device_status",
            "description": (
                "Get the connected ESP32 Vital Monitor's current "
                "mode, state, and monitoring status. Use when the "
                "user asks whether the monitor is connected, active, "
                "stopped, or which mode is selected."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_monitor_mode",
            "description": (
                "Switch the ESP32 Vital Monitor mode without "
                "necessarily collecting a complete measurement. "
                "Use HR for heart rate or SPO2 for blood oxygen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["HR", "SPO2"],
                        "description": "Target measurement mode.",
                    },
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_monitoring",
            "description": (
                "Enable measurement acquisition on the ESP32. "
                "Use when the user asks to start or resume monitoring."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_monitoring",
            "description": (
                "Stop active measurement acquisition on the ESP32. "
                "Use when the user explicitly asks to stop monitoring."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "collect_vitals_readings",
            "description": (
                "Run a heart-rate or SpO2 measurement session and "
                "collect ALL JSON reading events received from the "
                "ESP32, not just the final result. Use this when the "
                "user asks to check, measure, or read their heart rate "
                "or blood oxygen. HR defaults to a shorter session; "
                "SpO2 uses a longer window. Do not claim a measurement "
                "was completed unless the returned data supports it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["HR", "SPO2"],
                        "description": "Measurement mode.",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "minimum": 3,
                        "maximum": 60,
                        "description": (
                            "Optional collection duration. Use around "
                            "10–20 seconds for HR and up to 60 seconds "
                            "for SpO2."
                        ),
                    },
                },
                "required": ["mode"],
            },
        },
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's specialized Vitals Monitor Agent.

You control a connected ESP32 Vital Monitor and interpret the
measurement data it returns. You are not the main ZOE personality.

Your available device measures heart rate and estimated blood
oxygen saturation (SpO2) using a MAX30102 sensor.

CORE RULES
----------

1. Never invent, interpolate, or fabricate readings.
2. Use the actual data returned by the tools.
3. If the user asks to check their heart rate, collect HR readings.
4. If the user asks to check blood oxygen, oxygen saturation, or
   SpO2, collect an SpO2 session.
5. Use the device status tool for status-only questions.
6. Use mode switching when the user explicitly requests a mode change.
7. Do not claim that a command succeeded if the tool reports failure.
8. Distinguish intermediate readings from the final result.
9. Explain when readings are unstable, invalid, missing, or low quality.
10. Do not treat a missing reading as zero.
11. Do not diagnose diseases or claim the device is medical-grade.
12. Never greet the user.

MEASUREMENT WORKFLOW
--------------------

For "check my heart rate":
- Collect HR readings.
- Review the received readings and signal quality.
- Summarize the latest credible reading and the observed stability.
- Mention if the device failed to produce a stable value.

For "check my blood oxygen":
- Collect SpO2 readings.
- Review the complete sequence of returned readings.
- Inspect the final result, candidate count, and signal quality.
- Explain whether the measurement completed or timed out.
- Never turn an invalid or low-quality result into a confident number.

For "switch to heart-rate mode" or "switch to oxygen mode":
- Call set_monitor_mode with the requested mode.
- Report success only if the tool confirms it.

For "stop monitoring":
- Call stop_monitoring.

For "start monitoring":
- Call start_monitoring.

INTERPRETING THE SESSION
------------------------

The collected data may contain:
- heart_rate_bpm
- heart_rate_valid
- spo2_percent
- spo2_valid
- pi_percent
- ratio_r
- signal_quality
- state
- elapsed_ms
- candidate_count
- IR/red signal values
- events and a final result

Use the fields actually present. Firmware versions may omit some fields.

When multiple readings exist:
- Describe the trend or variability across the readings.
- Identify how many readings were received.
- Prefer valid readings with acceptable signal quality.
- Do not average invalid values or assume every reported value is valid.
- Do not present a noisy transient as a stable result.
- State if the session timed out or no final result was returned.

ANSWER STYLE
------------

Be concise but informative. Give the result first, then the quality
and trend, then any relevant limitation.

Example:
"Heart rate is currently reading 78 BPM. The recent valid readings
were fairly consistent, although the sensor reported some unstable
windows. Keep your finger still and repeat the measurement if the
value continues to fluctuate."

This is only an example of style, not actual device data.

SAFETY
------

This hobbyist MAX30102 device is not a certified medical instrument.
Its SpO2 estimate can be inaccurate because of movement, poor contact,
perfusion, sensor placement, and firmware calibration.

Do not diagnose a condition from these readings. If the user reports
severe shortness of breath, chest pain, fainting, blue/grey lips, or
another emergency symptom, advise urgent medical help rather than
relying on the device.
"""


# ============================================================
# ARGUMENT VALIDATION
# ============================================================

def normalize_arguments(
    tool_name: str,
    arguments: Any,
) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON arguments for '{tool_name}'."
            ) from exc

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    if tool_name not in TOOLS:
        raise ValueError(f"Unknown vitals tool: {tool_name}")

    if tool_name in {
        "get_device_status",
        "start_monitoring",
        "stop_monitoring",
    }:
        return {}

    if tool_name == "set_monitor_mode":
        mode = str(arguments.get("mode", "")).strip().upper()

        if mode not in {"HR", "SPO2"}:
            raise ValueError("Mode must be HR or SPO2.")

        return {"mode": mode}

    if tool_name == "collect_vitals_readings":
        mode = str(arguments.get("mode", "")).strip().upper()

        if mode not in {"HR", "SPO2"}:
            raise ValueError("Mode must be HR or SPO2.")

        cleaned: dict[str, Any] = {"mode": mode}
        duration = arguments.get("duration_seconds")

        if duration is not None:
            if isinstance(duration, bool):
                raise ValueError("Duration must be an integer.")

            try:
                duration = int(duration)
            except (TypeError, ValueError) as exc:
                raise ValueError("Duration must be an integer.") from exc

            if not 3 <= duration <= 60:
                raise ValueError("Duration must be between 3 and 60 seconds.")

            cleaned["duration_seconds"] = duration

        return cleaned

    raise ValueError(f"No validator exists for '{tool_name}'.")


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: Any,
) -> Any:
    try:
        cleaned = normalize_arguments(tool_name, arguments)
        return TOOLS[tool_name](**cleaned)

    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def serialize_tool_result(result: Any) -> str:
    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return str(result)


# ============================================================
# CONTEXT
# ============================================================

def clean_context(
    context: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if not context:
        return []

    cleaned: list[dict[str, str]] = []

    for item in context[-MAX_CONTEXT_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if content:
            cleaned.append({
                "role": role,
                "content": content,
            })

    return cleaned


def build_messages(
    query: str,
    context: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *clean_context(context),
        {
            "role": "user",
            "content": query.strip(),
        },
    ]


# ============================================================
# VITALS MONITOR AGENT
# ============================================================

class VitalsMonitorAgent:
    """
    Specialized agent for the ESP32 Vital Monitor.

    It can:
    - switch measurement modes
    - start and stop monitoring
    - retrieve device status
    - collect all available session readings
    - analyze the collected readings
    """

    def __init__(self, model: str = MODEL) -> None:
        self.model = model

    def run(
        self,
        query: str,
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        if not query or not query.strip():
            return "Tell me whether to check heart rate, blood oxygen, or device status."

        messages = build_messages(query, context)
        tool_calls_used = 0

        for step in range(MAX_AGENT_STEPS):
            print(
                f"\n[Vitals Agent] Step {step + 1}/{MAX_AGENT_STEPS}"
            )

            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=GROQ_TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                    max_completion_tokens=1200,
                )

                message = response.choices[0].message

            except Exception as exc:
                print(f"[Vitals Agent] Model error: {exc}")
                return "I couldn't process the vital-monitor request."

            tool_calls = message.tool_calls or []

            if not tool_calls:
                content = (message.content or "").strip()

                return content or (
                    "I couldn't produce a reliable interpretation "
                    "from the available monitor data."
                )

            if tool_calls_used + len(tool_calls) > MAX_TOOL_CALLS:
                return (
                    "The vitals request reached its tool-call limit. "
                    "Try a more specific request."
                )

            # Preserve the assistant's original tool-call message.
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            })

            for call in tool_calls:
                tool_calls_used += 1
                name = call.function.name
                raw_arguments = call.function.arguments

                print(f"[Vitals Agent] Tool #{tool_calls_used}: {name}")

                try:
                    print(f"[Vitals Agent] Arguments: {raw_arguments}")
                except Exception:
                    pass

                result = execute_tool(name, raw_arguments)

                print(
                    "[Vitals Agent] Tool completed: "
                    f"{name}; success={result.get('success') if isinstance(result, dict) else 'unknown'}"
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": serialize_tool_result(result),
                })

        return (
            "I couldn't complete the vitals analysis within the "
            "configured reasoning limit."
        )


# ============================================================
# SINGLETON AND PUBLIC ZOE INTERFACE
# ============================================================

vitals_monitor_agent = VitalsMonitorAgent()


def run_vitals_monitor_agent(
    query: str,
    context: list[dict[str, Any]] | None = None,
) -> str:
    """
    Public interface for ZOE Runtime.
    """

    with client_lock:
        return vitals_monitor_agent.run(
            query=query,
            context=context,
        )


# ============================================================
# STANDALONE CLI
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 64)
    print("                 ZOE VITALS MONITOR AGENT")
    print("=" * 64)
    print(f"Model: {MODEL}")
    print(f"Device: {DEVICE_HOST}:{DEVICE_PORT}")
    print()
    print("Examples:")
    print("  Check my heart rate.")
    print("  Check my blood oxygen.")
    print("  Switch to heart-rate mode.")
    print("  Stop monitoring.")
    print("  Is the monitor connected?")
    print()
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            break

        try:
            answer = run_vitals_monitor_agent(query)
            print(f"ZOE: {answer}\n")
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}\n")

    device.disconnect()