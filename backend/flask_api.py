from __future__ import annotations

import atexit
import json
import queue
import threading
from typing import Generator

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from backend.zoe import (
    run_zoe,
    extract_response_text,

    # State
    get_zoe_state,
    get_zoe_snapshot,
    get_status_snapshot,
    is_zoe_muted,
    mute_zoe,
    unmute_zoe,

    # STT
    get_stt_data,
    start_stt,

    # Response
    get_last_response,
    set_last_response,

    # Speech
    stop_zoe_speaking,
    zoe_is_speaking,
    is_zoe_processing,

    # Runtime events
    subscribe_runtime_events,
    unsubscribe_runtime_events,
    get_runtime_snapshot,

    # Shutdown
    shutdown_zoe,
)

from backend.integrations.telegram_bot import (
    start_telegram_bot,
    stop_telegram_bot,
    is_telegram_running,
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


CORS(
    app,
    origins=[
        "https://zoe.local:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
)


# ============================================================
# CONFIG
# ============================================================

SSE_HEARTBEAT_SECONDS = 15

SERVICE_NAME = "ZOE MARK 31"

FRONTEND_URL = "https://zoe.local:5173"

BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000


# ============================================================
# LOCKS
# ============================================================

_response_lock = threading.Lock()

_shutdown_lock = threading.Lock()


# ============================================================
# SHUTDOWN STATE
# ============================================================

_shutdown_called = False


# ============================================================
# STARTUP STATE
# ============================================================

_startup_lock = threading.Lock()
_startup_complete = False


# ============================================================
# STARTUP
# ============================================================

def start_services() -> None:
    """
    Start all ZOE services exactly once.

    This keeps startup explicit instead of executing
    service initialization merely because this module
    was imported.
    """

    global _startup_complete

    with _startup_lock:

        if _startup_complete:
            return

        print(
            "[FLASK] Starting ZOE services..."
        )

        # ----------------------------------------------------
        # STT
        # ----------------------------------------------------

        try:

            print(
                "[FLASK] Starting microphone pipeline..."
            )

            start_stt()

            print(
                "[FLASK] Microphone pipeline ready."
            )

        except Exception as exc:

            print(
                "[FLASK] Microphone pipeline failed "
                f"to start: {exc}"
            )

        # ----------------------------------------------------
        # TELEGRAM
        # ----------------------------------------------------

        try:

            telegram_started = (
                start_telegram_bot()
            )

            if telegram_started:

                print(
                    "[FLASK] Telegram interface started."
                )

            else:

                print(
                    "[FLASK] Telegram interface disabled "
                    "or unavailable."
                )

        except Exception as exc:

            print(
                "[FLASK] Telegram failed "
                f"to start: {exc}"
            )

        _startup_complete = True


# ============================================================
# RESPONSE HELPERS
# ============================================================

def _get_safe_last_response() -> str:
    """
    Safely retrieve the latest ZOE response.
    """

    with _response_lock:

        try:

            response = (
                get_last_response()
                or ""
            )

        except Exception as exc:

            print(
                "[FLASK] Failed to read last response:",
                exc,
            )

            return ""

    return str(response)


def _set_safe_last_response(
    response: str,
) -> None:
    """
    Safely update the latest ZOE response.
    """

    with _response_lock:

        try:

            set_last_response(
                response
            )

        except Exception as exc:

            print(
                "[FLASK] Failed to update last response:",
                exc,
            )


# ============================================================
# SNAPSHOT HELPERS
# ============================================================

def _safe_runtime_snapshot() -> dict:
    """
    Retrieve the runtime snapshot safely.
    """

    try:

        snapshot = (
            get_runtime_snapshot()
        )

        if not isinstance(
            snapshot,
            dict,
        ):

            return {
                "online": True,
                "activity": [],
            }

        return snapshot

    except Exception as exc:

        print(
            "[FLASK] Runtime snapshot error:",
            exc,
        )

        return {
            "online": False,
            "activity": [],
            "error": str(exc),
        }


def _safe_status_snapshot() -> dict:
    """
    Retrieve status without allowing a status
    subsystem failure to break /api/state.
    """

    try:

        status = (
            get_status_snapshot()
        )

        if isinstance(
            status,
            dict,
        ):

            return status

        return {}

    except Exception as exc:

        print(
            "[FLASK] Status snapshot error:",
            exc,
        )

        return {
            "error": str(exc),
        }


def _build_fallback_state_snapshot(
    runtime_snapshot: dict,
) -> dict:
    """
    Build a state snapshot when the canonical
    get_zoe_snapshot() function is unavailable
    or returns an incomplete value.
    """

    try:

        current_state = get_zoe_state()

    except Exception:

        current_state = "offline"


    try:

        stt = get_stt_data()

        if not isinstance(
            stt,
            dict,
        ):

            stt = {}

    except Exception as exc:

        print(
            "[FLASK] STT snapshot error:",
            exc,
        )

        stt = {}


    try:

        muted = bool(
            is_zoe_muted()
        )

    except Exception:

        muted = False


    try:

        speaking = bool(
            zoe_is_speaking()
        )

    except Exception:

        speaking = False


    try:

        processing = bool(
            is_zoe_processing()
        )

    except Exception:

        processing = False


    activity = runtime_snapshot.get(
        "activity",
        [],
    )

    if not isinstance(
        activity,
        list,
    ):

        activity = []


    return {

        "online": True,

        # ------------------------------------------------
        # GLOBAL STATE
        # ------------------------------------------------

        "state": current_state,

        "speaking": (
            current_state == "speaking"
            or speaking
        ),

        "listening": (
            current_state == "listening"
        ),

        "thinking": (
            current_state == "thinking"
        ),

        "processing": processing,

        "muted": muted,

        # ------------------------------------------------
        # STT
        # ------------------------------------------------

        "stt": {

            "text": (
                stt.get(
                    "text",
                    "",
                )
                or ""
            ),

            "final_text": (
                stt.get(
                    "final_text",
                    "",
                )
                or ""
            ),

            "last_final_text": (
                stt.get(
                    "last_final_text",
                    "",
                )
                or ""
            ),

            "final": bool(
                stt.get(
                    "final",
                    False,
                )
            ),

            "speaking": bool(
                stt.get(
                    "speaking",
                    False,
                )
            ),

            "user_speaking": bool(
                stt.get(
                    "user_speaking",
                    False,
                )
            ),

        },

        # ------------------------------------------------
        # RESPONSE
        # ------------------------------------------------

        "response": (
            _get_safe_last_response()
        ),

        # ------------------------------------------------
        # RUNTIME
        # ------------------------------------------------

        "runtime": runtime_snapshot,

        "activity": activity,

        # ------------------------------------------------
        # STATUS
        # ------------------------------------------------

        "status": (
            _safe_status_snapshot()
        ),

        # ------------------------------------------------
        # TELEGRAM
        # ------------------------------------------------

        "telegram": {
            "running": is_telegram_running(),
        },

    }


def _build_state_snapshot() -> dict:
    """
    Build the complete frontend state.

    Preferred source:

        get_zoe_snapshot()

    The Flask layer enriches that snapshot with
    STT/runtime/status/Telegram information if necessary.
    """

    runtime_snapshot = (
        _safe_runtime_snapshot()
    )


    # ========================================================
    # CANONICAL ZOE SNAPSHOT
    # ========================================================

    try:

        canonical = (
            get_zoe_snapshot()
        )

    except Exception as exc:

        print(
            "[FLASK] Canonical ZOE snapshot error:",
            exc,
        )

        canonical = None


    if isinstance(
        canonical,
        dict,
    ):

        # Work on a copy so we never mutate
        # the backend's returned object.

        snapshot = dict(
            canonical
        )


        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        snapshot.setdefault(
            "runtime",
            runtime_snapshot,
        )

        runtime = snapshot.get(
            "runtime"
        )

        if not isinstance(
            runtime,
            dict,
        ):

            snapshot["runtime"] = (
                runtime_snapshot
            )

            runtime = runtime_snapshot


        # ----------------------------------------------------
        # Activity
        # ----------------------------------------------------

        activity = runtime.get(
            "activity",
            [],
        )

        if not isinstance(
            activity,
            list,
        ):

            activity = []

        snapshot["activity"] = activity


        # ----------------------------------------------------
        # STT
        # ----------------------------------------------------

        if not isinstance(
            snapshot.get("stt"),
            dict,
        ):

            try:

                stt = get_stt_data()

            except Exception:

                stt = {}

            snapshot["stt"] = stt


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        if "response" not in snapshot:

            snapshot["response"] = (
                _get_safe_last_response()
            )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if "status" not in snapshot:

            snapshot["status"] = (
                _safe_status_snapshot()
            )


        # ----------------------------------------------------
        # Telegram
        # ----------------------------------------------------

        snapshot["telegram"] = {
            "running": is_telegram_running(),
        }


        # ----------------------------------------------------
        # Online
        # ----------------------------------------------------

        snapshot.setdefault(
            "online",
            True,
        )


        return snapshot


    # ========================================================
    # BACKWARDS-COMPATIBLE FALLBACK
    # ========================================================

    return _build_fallback_state_snapshot(
        runtime_snapshot
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    snapshot = _build_state_snapshot()

    return jsonify({

        "ok": bool(
            snapshot.get(
                "online",
                False,
            )
        ),

        "service": SERVICE_NAME,

        "state": snapshot.get(
            "state",
            "offline",
        ),

    })


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def status_snapshot():

    status = _safe_status_snapshot()

    if "error" in status:

        return jsonify({

            "success": False,

            **status,

            "telegram": {
                "running": is_telegram_running(),
            },

        }), 500


    return jsonify({

        **status,

        "telegram": {
            "running": is_telegram_running(),
        },

    })


# ============================================================
# STATE
# ============================================================

@app.get("/api/state")
def state():

    snapshot = _build_state_snapshot()

    if not snapshot.get(
        "online",
        False,
    ):

        return jsonify(
            snapshot
        ), 500

    return jsonify(
        snapshot
    )


# ============================================================
# ACTIVITY
# ============================================================

@app.get("/api/activity")
def activity():

    try:

        runtime_snapshot = (
            _safe_runtime_snapshot()
        )

        activities = (
            runtime_snapshot.get(
                "activity",
                [],
            )
        )

        return jsonify({

            "success": True,

            "activity": activities,

        })

    except Exception as exc:

        print(
            "[FLASK] Activity error:",
            exc,
        )

        return jsonify({

            "success": False,

            "activity": [],

            "error": str(exc),

        }), 500


# ============================================================
# SSE ENCODER
# ============================================================

def _sse_encode(
    event: dict,
) -> str:
    """
    Convert a runtime event into an SSE message.

    Runtime event type remains inside:

        data.type
    """

    if not isinstance(
        event,
        dict,
    ):

        event = {
            "type": "unknown",
            "data": event,
        }


    event_id = str(
        event.get(
            "id",
            "",
        )
        or ""
    )


    try:

        payload = json.dumps(
            event,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    except Exception as exc:

        print(
            "[FLASK SSE] Event serialization error:",
            exc,
        )

        payload = json.dumps({

            "type": "serialization_error",

            "error": str(exc),

        })


    lines = []


    if event_id:

        lines.append(
            f"id: {event_id}"
        )


    for line in payload.splitlines():

        lines.append(
            f"data: {line}"
        )


    lines.append("")


    return "\n".join(
        lines
    ) + "\n"


# ============================================================
# SSE INITIAL SNAPSHOT
# ============================================================

def _sse_snapshot_event() -> str:
    """
    Send complete frontend state immediately
    after an SSE connection is established.
    """

    snapshot = (
        _build_state_snapshot()
    )


    event = {

        "id": "snapshot",

        "type": "state_snapshot",

        "timestamp": None,

        "data": snapshot,

    }


    return _sse_encode(
        event
    )


# ============================================================
# RUNTIME EVENTS
# ============================================================

@app.get("/api/events")
def runtime_events():

    try:

        subscriber_id, event_queue = (
            subscribe_runtime_events()
        )

    except Exception as exc:

        print(
            "[FLASK SSE] Subscription failed:",
            exc,
        )

        return jsonify({

            "success": False,

            "error": str(exc),

        }), 500


    @stream_with_context
    def generate() -> Generator[
        str,
        None,
        None,
    ]:

        try:

            # ------------------------------------------------
            # Initial hydration
            #
            # Subscription happens BEFORE snapshot.
            # ------------------------------------------------

            yield _sse_snapshot_event()


            # ------------------------------------------------
            # Live event loop
            # ------------------------------------------------

            while True:

                try:

                    event = event_queue.get(
                        timeout=SSE_HEARTBEAT_SECONDS
                    )

                except queue.Empty:

                    yield ": heartbeat\n\n"

                    continue


                # ------------------------------------------------
                # Runtime shutdown
                # ------------------------------------------------

                if event is None:

                    break


                # ------------------------------------------------
                # Forward runtime event
                # ------------------------------------------------

                yield _sse_encode(
                    event
                )


        except GeneratorExit:

            pass


        except (
            BrokenPipeError,
            ConnectionResetError,
        ):

            pass


        except Exception as exc:

            print(
                "[FLASK SSE] Stream error:",
                exc,
            )


        finally:

            try:

                unsubscribe_runtime_events(
                    subscriber_id
                )

            except Exception as exc:

                print(
                    "[FLASK SSE] Unsubscribe error:",
                    exc,
                )


    response = Response(
        generate(),
        mimetype="text/event-stream",
    )


    # ========================================================
    # SSE HEADERS
    # ========================================================

    response.headers[
        "Cache-Control"
    ] = "no-cache, no-store, must-revalidate"

    response.headers[
        "Pragma"
    ] = "no-cache"

    response.headers[
        "Expires"
    ] = "0"

    response.headers[
        "X-Accel-Buffering"
    ] = "no"

    response.headers[
        "Connection"
    ] = "keep-alive"


    return response


# ============================================================
# RESPONSE
# ============================================================

@app.get("/api/response")
def response():

    try:

        return jsonify({

            "success": True,

            "response": (
                _get_safe_last_response()
            ),

        })

    except Exception as exc:

        print(
            "[FLASK] Response error:",
            exc,
        )

        return jsonify({

            "success": False,

            "response": "",

            "error": str(exc),

        }), 500


# ============================================================
# MICROPHONE
# ============================================================

@app.get("/api/microphone")
def microphone_state():

    try:

        return jsonify({

            "success": True,

            "muted": bool(
                is_zoe_muted()
            ),

        })

    except Exception as exc:

        print(
            "[FLASK] Microphone state error:",
            exc,
        )

        return jsonify({

            "success": False,

            "muted": False,

            "error": str(exc),

        }), 500


# ============================================================
# MUTE
# ============================================================

@app.post("/api/microphone/mute")
def microphone_mute():

    try:

        mute_zoe()

        return jsonify({

            "success": True,

            "muted": True,

        })

    except Exception as exc:

        print(
            "[FLASK] Microphone mute error:",
            exc,
        )

        try:

            muted = bool(
                is_zoe_muted()
            )

        except Exception:

            muted = False


        return jsonify({

            "success": False,

            "muted": muted,

            "error": str(exc),

        }), 500


# ============================================================
# UNMUTE
# ============================================================

@app.post("/api/microphone/unmute")
def microphone_unmute():

    try:

        unmute_zoe()

        return jsonify({

            "success": True,

            "muted": False,

        })

    except Exception as exc:

        print(
            "[FLASK] Microphone unmute error:",
            exc,
        )

        try:

            muted = bool(
                is_zoe_muted()
            )

        except Exception:

            muted = False


        return jsonify({

            "success": False,

            "muted": muted,

            "error": str(exc),

        }), 500


# ============================================================
# REALTIME STT
# ============================================================

@app.get("/api/stt")
def stt():

    try:

        data = get_stt_data()

        if not isinstance(
            data,
            dict,
        ):

            data = {}


        return jsonify({

            "success": True,

            "text": (
                data.get(
                    "text",
                    "",
                )
                or ""
            ),

            "final_text": (
                data.get(
                    "final_text",
                    "",
                )
                or ""
            ),

            "last_final_text": (
                data.get(
                    "last_final_text",
                    "",
                )
                or ""
            ),

            "final": bool(
                data.get(
                    "final",
                    False,
                )
            ),

            "speaking": bool(
                data.get(
                    "speaking",
                    False,
                )
            ),

            "user_speaking": bool(
                data.get(
                    "user_speaking",
                    False,
                )
            ),

        })


    except Exception as exc:

        print(
            "[FLASK] STT error:",
            exc,
        )

        return jsonify({

            "success": False,

            "text": "",

            "final_text": "",

            "last_final_text": "",

            "final": False,

            "speaking": False,

            "user_speaking": False,

            "error": str(exc),

        }), 500

# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )


    message = data.get(
        "message",
        "",
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if not isinstance(
        message,
        str,
    ):

        return jsonify({

            "success": False,

            "error": (
                "message must be a string"
            ),

        }), 400


    message = message.strip()


    if not message:

        return jsonify({

            "success": False,

            "error": (
                "Message cannot be empty"
            ),

        }), 400


    # ========================================================
    # RUN ZOE
    # ========================================================

    try:

        _set_safe_last_response(
            ""
        )


        result = run_zoe(
            message,
            speak_output=True,
        )


        response_text = (
            extract_response_text(
                result
            )
        )


        if response_text is None:

            response_text = ""


        if not isinstance(
            response_text,
            str,
        ):

            response_text = str(
                response_text
            )


        response_text = (
            response_text.strip()
        )


        _set_safe_last_response(
            response_text
        )


        return jsonify({

            "success": True,

            "response": response_text,

            "result": response_text,

        })


    except Exception as exc:

        print(
            "[FLASK] Chat error:",
            exc,
        )


        return jsonify({

            "success": False,

            "response": "",

            "error": str(exc),

        }), 500


# ============================================================
# SPEECH STOP
# ============================================================

@app.post("/api/speech/stop")
def stop_speech():

    try:

        stop_zoe_speaking()

        return jsonify({

            "success": True,

        })


    except Exception as exc:

        print(
            "[FLASK] Speech stop error:",
            exc,
        )

        return jsonify({

            "success": False,

            "error": str(exc),

        }), 500


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown():

    global _shutdown_called

    with _shutdown_lock:

        if _shutdown_called:
            return

        _shutdown_called = True


    print(
        "\n[FLASK] Shutting down ZOE..."
    )


    # ========================================================
    # TELEGRAM FIRST
    # ========================================================

    try:

        stop_telegram_bot()

    except Exception as exc:

        print(
            "[FLASK] Telegram shutdown error:",
            exc,
        )


    # ========================================================
    # ZOE RUNTIME
    # ========================================================

    try:

        shutdown_zoe()

    except Exception as exc:

        print(
            "[FLASK] ZOE shutdown error:",
            exc,
        )


    print(
        "[FLASK] ZOE shutdown complete."
    )


atexit.register(
    shutdown
)


# ============================================================
# SERVER
# ============================================================

def main() -> None:

    start_services()

    print()

    print(
        "=" * 52
    )

    print(
        f"                 {SERVICE_NAME}"
    )

    print(
        "             FLASK API SERVER"
    )

    print(
        "=" * 52
    )

    print()

    print(
        "Frontend:"
    )

    print(
        f"  {FRONTEND_URL}"
    )

    print()

    print(
        "Backend:"
    )

    print(
        f"  http://127.0.0.1:{BACKEND_PORT}"
    )

    print()

    print(
        "Realtime STT:"
    )

    print(
        "  GET /api/stt"
    )

    print()

    print(
        "State:"
    )

    print(
        "  GET /api/state"
    )

    print()

    print(
        "Activity:"
    )

    print(
        "  GET /api/activity"
    )

    print()

    print(
        "Runtime events:"
    )

    print(
        "  GET /api/events"
    )

    print()

    print(
        "Response:"
    )

    print(
        "  GET /api/response"
    )

    print()

    print(
        "Chat:"
    )

    print(
        "  POST /api/chat"
    )

    print()

    print(
        "Microphone:"
    )

    print(
        "  GET  /api/microphone"
    )

    print(
        "  POST /api/microphone/mute"
    )

    print(
        "  POST /api/microphone/unmute"
    )

    print()

    print(
        "Speech:"
    )

    print(
        "  POST /api/speech/stop"
    )

    print()

    print(
        "Telegram:"
    )

    print(
        "  Enabled:",
        is_telegram_running(),
    )

    print()

    print(
        "[FLASK] Starting HTTP backend..."
    )

    print()


    app.run(
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        debug=False,
        threaded=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()