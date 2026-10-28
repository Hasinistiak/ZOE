from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from backend.execution.zoe_runtime import create_zoe_runtime
from backend.speech.tts import preload, speak_async, stop as stop_tts, is_speaking
from backend.speech.stt import start as stt_start, stop as stt_stop, set_speaking as stt_set_speaking, set_muted as stt_set_muted

BRAIN_MODEL = os.getenv("ZOE_MODEL", "").strip()
VOICE_QUEUE_SIZE = max(1, int(os.getenv("ZOE_VOICE_QUEUE_SIZE", "32")))
RUNTIME_EVENT_QUEUE_SIZE = max(1, int(os.getenv("ZOE_RUNTIME_EVENT_QUEUE_SIZE", "100")))
RUNTIME_RESPONSE_TIMEOUT = max(5.0, float(os.getenv("ZOE_RUNTIME_RESPONSE_TIMEOUT", "120")))
IDLE_MUTE_SECONDS = max(5.0, float(os.getenv("ZOE_IDLE_MUTE_SECONDS", "120")))

print()
print("=" * 60)
print("                     ZOE MARK 35")
print("=" * 60)
print()
print(f"[ZOE] Brain model: {BRAIN_MODEL or 'configured by runtime'}")

_state_lock = threading.RLock()
_stt_lock = threading.RLock()
_response_lock = threading.RLock()
_runtime_lock = threading.RLock()
_runtime_event_lock = threading.RLock()
_processing_lock = threading.RLock()
_runtime_run_lock = threading.RLock()
_tts_generation_lock = threading.RLock()
_voice_worker_lock = threading.RLock()
_wakeword_lock = threading.RLock()

_zoe_muted = False
_user_speaking = False
_zoe_thinking = False
_zoe_speaking = False
_auto_muted = False
_idle_baseline = time.monotonic()

_runtime_started = False
_runtime_shutdown = False
_stt_started = False
_stt_failed = False
_stt_error = ""

_current_stt_text = ""
_final_stt_text = ""
_last_final_stt_text = ""
_stt_final = False
_last_verified_stt_at: float | None = None
_last_response = ""

_processing_speech = False
_active_request_id: str | None = None
_tts_generation = 0
runtime = None
_wakeword = None

_idle_watchdog: threading.Thread | None = None
_idle_watchdog_stop = threading.Event()
_runtime_event_subscribers: dict[str, queue.Queue] = {}

@dataclass(frozen=True)
class VoiceRequest:
    request_id: str
    text: str

_voice_queue: queue.Queue[VoiceRequest] = queue.Queue(maxsize=VOICE_QUEUE_SIZE)
_voice_worker: threading.Thread | None = None
_voice_worker_stop = threading.Event()


def _safe_text(text: Any) -> str:
    return "" if text is None else str(text).strip()


def clean_response_text(text: Any) -> str:
    text = _safe_text(text)
    return text.replace("**", "").replace("*", "").strip() if text else ""


def extract_response_text(result) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        return clean_response_text(result) or None
    if hasattr(result, "answer"):
        return clean_response_text(getattr(result, "answer", "") or "") or None
    if isinstance(result, dict):
        return clean_response_text(result.get("text") or result.get("answer") or "") or None
    return clean_response_text(result) or None


def set_last_response(text: str) -> None:
    global _last_response
    with _response_lock:
        _last_response = clean_response_text(text)


def get_last_response() -> str:
    with _response_lock:
        return _last_response


def clear_last_response() -> None:
    set_last_response("")


def _next_tts_generation() -> int:
    global _tts_generation
    with _tts_generation_lock:
        _tts_generation += 1
        return _tts_generation


def _is_current_tts_generation(generation: int) -> bool:
    with _tts_generation_lock:
        return generation == _tts_generation


def set_zoe_speaking(speaking: bool) -> None:
    global _zoe_speaking
    with _state_lock:
        _zoe_speaking = bool(speaking)


def is_zoe_actually_speaking() -> bool:
    try:
        return bool(is_speaking())
    except Exception:
        with _state_lock:
            return _zoe_speaking


def zoe_is_speaking() -> bool:
    return is_zoe_actually_speaking()


def set_user_speaking(speaking: bool) -> None:
    global _user_speaking
    with _state_lock:
        _user_speaking = bool(speaking)


def is_user_speaking() -> bool:
    with _state_lock:
        return _user_speaking


def set_zoe_thinking(thinking: bool) -> None:
    global _zoe_thinking
    with _state_lock:
        _zoe_thinking = bool(thinking)


def is_zoe_thinking() -> bool:
    with _state_lock:
        return _zoe_thinking


def is_zoe_processing() -> bool:
    with _processing_lock:
        return _processing_speech


def get_active_request_id() -> str | None:
    with _processing_lock:
        return _active_request_id


def get_zoe_state() -> str:
    with _state_lock:
        muted = _zoe_muted
        user_speaking = _user_speaking
        thinking = _zoe_thinking
        internal_speaking = _zoe_speaking
    try:
        speaking = bool(is_speaking())
    except Exception:
        speaking = internal_speaking
    if user_speaking:
        return "listening"
    if thinking:
        return "thinking"
    if speaking:
        return "speaking"
    if muted:
        return "muted"
    return "idle"


def get_status_snapshot() -> dict:
    try:
        return {} if runtime is None else runtime.status_worker.snapshot()
    except Exception as exc:
        print("[ZOE STATUS] Snapshot error:", exc)
        return {}


def is_zoe_muted() -> bool:
    with _state_lock:
        return _zoe_muted


def is_zoe_auto_muted() -> bool:
    with _state_lock:
        return _auto_muted


def _reset_runtime_speaking() -> None:
    rt = runtime
    if rt is not None:
        try:
            rt.set_speaking(False)
        except Exception as exc:
            print("[ZOE] Runtime speaking reset error:", exc)


def set_zoe_muted(muted: bool) -> None:
    global _zoe_muted, _user_speaking, _zoe_thinking, _auto_muted

    muted = bool(muted)

    with _state_lock:
        state_changed = _zoe_muted != muted
        if state_changed:
            _zoe_muted = muted
            if muted:
                _user_speaking = False
                _zoe_thinking = False

    if muted:
        if state_changed:
            print("[ZOE] Muting...")
            clear_current_stt()
            _next_tts_generation()

            try:
                stt_set_muted(True)
            except Exception as exc:
                print("[ZOE STT] Mute error:", exc)


            with _state_lock:
                _auto_muted = False

        if not _arm_wakeword():
            print("[ZOE WAKE] Warning: wakeword could not be started.")

        if state_changed:
            print("[ZOE] MUTED. Wakeword monitoring enabled.")
        return

    if not state_changed:
        _disarm_wakeword()
        return

    print("[ZOE] Unmuting...")

    try:
        stt_set_muted(False)
    except Exception as exc:
        with _state_lock:
            _zoe_muted = True
        print("[ZOE STT] Unmute error:", exc)
        raise

    _disarm_wakeword()

    with _state_lock:
        _auto_muted = False

    _touch_activity()
    print("[ZOE] UNMUTED.")


def mute_zoe() -> None:
    set_zoe_muted(True)


def unmute_zoe() -> None:
    set_zoe_muted(False)


def _touch_activity() -> None:
    global _idle_baseline
    with _state_lock:
        _idle_baseline = time.monotonic()


def _seconds_idle() -> float:
    with _stt_lock:
        verified_at = _last_verified_stt_at
    with _state_lock:
        baseline = _idle_baseline
    return time.monotonic() - max(baseline, verified_at or 0.0)


def _is_busy() -> bool:
    return (
        is_user_speaking()
        or is_zoe_thinking()
        or is_zoe_processing()
        or is_zoe_actually_speaking()
        or not _voice_queue.empty()
    )


def _wake_up() -> None:
    if not is_zoe_muted():
        return
    print("[ZOE] Wakeword heard, waking up.")
    try:
        unmute_zoe()
    except Exception as exc:
        print("[ZOE] Wake-up unmute failed:", exc)


def _on_wakeword() -> None:
    threading.Thread(target=_wake_up, name="ZOE-WakeUp", daemon=True).start()


def _arm_wakeword() -> bool:
    global _wakeword
    with _wakeword_lock:
        try:
            if _wakeword is None:
                from backend.speech.wakeword import ZoeWakeWord
                _wakeword = ZoeWakeWord(on_wake=_on_wakeword)
            _wakeword.start()
            return True
        except Exception as exc:
            print("[ZOE WAKE] Could not arm wakeword:", exc)
            return False


def _disarm_wakeword() -> None:
    with _wakeword_lock:
        if _wakeword is not None:
            try:
                _wakeword.stop()
            except Exception as exc:
                print("[ZOE WAKE] Disarm error:", exc)


def _auto_mute() -> None:
    global _auto_muted

    if is_zoe_muted():
        return

    print(f"[ZOE] No verified speech for {IDLE_MUTE_SECONDS:.0f}s, going to sleep.")
    set_zoe_muted(True)

    if is_zoe_muted():
        with _state_lock:
            _auto_muted = True
        print("[ZOE] Muted. Say 'Hey Zoe' to wake.")


def _idle_watchdog_loop() -> None:
    while not _idle_watchdog_stop.wait(1.0):
        try:
            if _runtime_shutdown or not _stt_started or is_zoe_muted():
                continue
            if _is_busy():
                _touch_activity()
                continue
            if _seconds_idle() >= IDLE_MUTE_SECONDS:
                _auto_mute()
        except Exception as exc:
            print("[ZOE IDLE] Watchdog error:", exc)


def _start_idle_watchdog() -> None:
    global _idle_watchdog
    if _idle_watchdog is not None and _idle_watchdog.is_alive():
        return
    _idle_watchdog_stop.clear()
    _touch_activity()
    _idle_watchdog = threading.Thread(
        target=_idle_watchdog_loop,
        name="ZOE-Idle-Watchdog",
        daemon=True,
    )
    _idle_watchdog.start()


def get_current_stt() -> str:
    with _stt_lock:
        return _current_stt_text


def get_final_stt() -> str:
    with _stt_lock:
        return _final_stt_text


def get_last_final_stt() -> str:
    with _stt_lock:
        return _last_final_stt_text


def is_stt_final() -> bool:
    with _stt_lock:
        return _stt_final


def get_last_verified_stt_age() -> float | None:
    with _stt_lock:
        at = _last_verified_stt_at
    return None if at is None else time.monotonic() - at


def get_stt_data() -> dict:
    with _stt_lock:
        text = _current_stt_text
        final_text = _final_stt_text
        last_final_text = _last_final_stt_text
        final = _stt_final
    return {
        "text": text,
        "final_text": final_text,
        "last_final_text": last_final_text,
        "final": final,
        "user_speaking": is_user_speaking(),
        "speaking": is_user_speaking(),
    }


def clear_current_stt() -> None:
    global _current_stt_text, _final_stt_text, _stt_final
    with _stt_lock:
        _current_stt_text = ""
        _final_stt_text = ""
        _stt_final = False


def set_final_stt(text: str) -> None:
    global _current_stt_text, _final_stt_text, _last_final_stt_text
    global _last_verified_stt_at, _stt_final

    text = _safe_text(text)
    with _stt_lock:
        if text:
            _last_final_stt_text = text
            _last_verified_stt_at = time.monotonic()
        _current_stt_text = text
        _final_stt_text = text
        _stt_final = bool(text)


def on_user_speech_partial(text: str) -> None:
    global _current_stt_text, _stt_final
    if is_zoe_muted():
        return
    text = _safe_text(text)
    with _stt_lock:
        _current_stt_text = text
        _stt_final = False
    if text:
        print(f"\r[ZOE STT] {text:<100}", end="", flush=True)


def on_user_speech_start() -> None:
    if is_zoe_muted():
        return
    clear_current_stt()
    set_user_speaking(True)
    print()
    print("[ZOE STT] User speaking.")


def on_user_speech_end() -> None:
    if is_zoe_muted():
        return
    set_user_speaking(False)
    print("[ZOE STT] User speech ended.")


def clear_zoe_speaking_state() -> None:
    set_zoe_speaking(False)
    try:
        stt_set_speaking(False)
    except Exception:
        pass


def _tts_completed(
    generation: int,
    on_complete: Callable[[], None] | None,
) -> None:
    if not _is_current_tts_generation(generation):
        return
    clear_zoe_speaking_state()
    print("[ZOE TTS] Finished.")
    if on_complete is not None:
        try:
            on_complete()
        except Exception as exc:
            print("[ZOE TTS] Completion callback error:", exc)


def zoe_speak_async(
    text: str,
    on_complete: Callable[[], None] | None = None,
) -> None:
    text = clean_response_text(text)
    if not text or is_zoe_muted():
        if on_complete is not None:
            try:
                on_complete()
            except Exception as exc:
                print("[ZOE TTS] Completion callback error:", exc)
        return

    generation = _next_tts_generation()
    set_zoe_speaking(True)

    try:
        stt_set_speaking(True)
    except Exception:
        pass

    def completion_callback() -> None:
        _tts_completed(generation, on_complete)

    try:
        speak_async(text, on_complete=completion_callback)
    except Exception as exc:
        if _is_current_tts_generation(generation):
            clear_zoe_speaking_state()
        print("[ZOE TTS] Failed to start:", exc)
        raise


def interrupt_zoe() -> None:
    print()
    print("[ZOE] BARGE-IN")
    _next_tts_generation()
    try:
        stop_tts()
    except Exception as exc:
        print("[ZOE] TTS interruption error:", exc)
    clear_zoe_speaking_state()
    _reset_runtime_speaking()
    print("[ZOE] Speech interrupted.")


def _store_runtime_output(text: str) -> None:
    text = clean_response_text(text)
    if text:
        set_last_response(text)


def _speak_runtime_output(
    text: str,
    on_complete: Callable[[], None] | None = None,
) -> None:
    text = clean_response_text(text)
    if not text:
        if on_complete is not None:
            try:
                on_complete()
            except Exception as exc:
                print("[ZOE TTS] Completion callback error:", exc)
        return
    zoe_speak_async(text, on_complete=on_complete)


def _interrupt_runtime_output() -> None:
    interrupt_zoe()


def _publish_runtime_event(event: dict) -> None:
    if not isinstance(event, dict):
        return
    with _runtime_event_lock:
        subscribers = list(_runtime_event_subscribers.items())
    for subscriber_id, subscriber_queue in subscribers:
        try:
            subscriber_queue.put_nowait(event)
        except queue.Full:
            try:
                subscriber_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                subscriber_queue.put_nowait(event)
            except queue.Full:
                pass
        except Exception as exc:
            print(f"[ZOE EVENTS] Subscriber {subscriber_id} error:", exc)


def subscribe_runtime_events() -> tuple[str, queue.Queue]:
    subscriber_id = f"sub_{uuid.uuid4().hex}"
    subscriber_queue = queue.Queue(maxsize=RUNTIME_EVENT_QUEUE_SIZE)
    with _runtime_event_lock:
        _runtime_event_subscribers[subscriber_id] = subscriber_queue
    print("[ZOE EVENTS] Subscriber connected:", subscriber_id)
    return subscriber_id, subscriber_queue


def unsubscribe_runtime_events(subscriber_id: str) -> None:
    with _runtime_event_lock:
        removed = _runtime_event_subscribers.pop(subscriber_id, None)
    if removed is not None:
        print("[ZOE EVENTS] Subscriber disconnected:", subscriber_id)


def get_runtime_event_subscriber_count() -> int:
    with _runtime_event_lock:
        return len(_runtime_event_subscribers)


def get_runtime_snapshot() -> dict:
    if runtime is None:
        return {"online": False}
    try:
        return runtime.snapshot()
    except Exception as exc:
        print("[ZOE EVENTS] Runtime snapshot error:", exc)
        return {"online": False, "error": str(exc)}


def get_current_url() -> str:
    snapshot = get_runtime_snapshot()
    url = snapshot.get("url", "")
    return url.strip() if isinstance(url, str) else ""


def _initialize_runtime() -> None:
    global runtime, _runtime_started, _runtime_shutdown

    if _runtime_started:
        return

    with _runtime_lock:
        if _runtime_started:
            return
        if _runtime_shutdown:
            raise RuntimeError("ZOE runtime has already been shut down.")

        print("[ZOE] Initializing TTS...")
        preload()
        print("[ZOE] TTS ready.")
        print("[ZOE] Initializing Runtime...")

        runtime = create_zoe_runtime(
            on_output=_store_runtime_output,
            on_speech=_speak_runtime_output,
            on_interrupt=_interrupt_runtime_output,
            on_event=_publish_runtime_event,
        )

        _runtime_started = True
        print("[ZOE] Runtime ready.")


_initialize_runtime()


def _clear_voice_queue() -> None:
    while True:
        try:
            _voice_queue.get_nowait()
        except queue.Empty:
            break
        else:
            _voice_queue.task_done()


def _ensure_voice_worker() -> None:
    global _voice_worker
    with _voice_worker_lock:
        if _voice_worker is not None and _voice_worker.is_alive():
            return
        _voice_worker_stop.clear()
        _voice_worker = threading.Thread(
            target=_voice_request_worker,
            name="ZOE-Voice-Request-Worker",
            daemon=True,
        )
        _voice_worker.start()
        print("[ZOE] Voice request worker started.")


def _voice_request_worker() -> None:
    global _processing_speech, _active_request_id

    while not _voice_worker_stop.is_set():
        try:
            request = _voice_queue.get(timeout=0.25)
        except queue.Empty:
            continue

        try:
            if is_zoe_muted():
                continue

            with _processing_lock:
                _processing_speech = True
                _active_request_id = request.request_id

            set_zoe_thinking(True)
            print()
            print(f"[ZOE REQUEST] id={request.request_id}")
            print(f"user> {request.text}")

            try:
                result = run_zoe(
                    request.text,
                    speak_output=True,
                    wait_for_response=False,
                )
                response = extract_response_text(result)
                if response:
                    if get_last_response() != response:
                        set_last_response(response)
                    print("zoe>", response)
                else:
                    print("zoe> Background task started.")
            except Exception as exc:
                print()
                print("[ZOE ERROR]", exc)
                set_last_response("I ran into an error while processing that request.")
            finally:
                set_zoe_thinking(False)
                with _processing_lock:
                    _processing_speech = False
                    _active_request_id = None
                print()

        except Exception as exc:
            print("[ZOE VOICE WORKER] Unexpected error:", exc)
            set_zoe_thinking(False)
            with _processing_lock:
                _processing_speech = False
                _active_request_id = None
        finally:
            _voice_queue.task_done()


def _enqueue_voice_request(text: str) -> str | None:
    text = _safe_text(text)
    if not text or is_zoe_muted() or _runtime_shutdown:
        return None

    request_id = uuid.uuid4().hex
    request = VoiceRequest(request_id=request_id, text=text)

    try:
        _voice_queue.put_nowait(request)
    except queue.Full:
        print("[ZOE STT] Voice request queue full.")
        return None

    _ensure_voice_worker()
    print("[ZOE STT] Queued request", request_id)
    return request_id


def on_user_speech(text: str) -> None:
    text = _safe_text(text)
    set_user_speaking(False)
    if is_zoe_muted() or not text:
        return
    set_final_stt(text)
    if _enqueue_voice_request(text) is None:
        print("[ZOE STT] Request could not be queued.")


def run_zoe(
    message: str,
    speak_output: bool = True,
    wait_for_response: bool = True,
    timeout: float | None = None,
):
    message = _safe_text(message)
    if not message or _runtime_shutdown:
        return None

    _initialize_runtime()

    if timeout is None:
        timeout = RUNTIME_RESPONSE_TIMEOUT

    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = RUNTIME_RESPONSE_TIMEOUT

    timeout = max(1.0, timeout)

    with _runtime_run_lock:
        if _runtime_shutdown:
            raise RuntimeError("ZOE runtime is shut down.")

        clear_last_response()

        try:
            result = runtime.run(
                message=message,
                speak_output=speak_output,
                wait_for_completion=wait_for_response,
                timeout=timeout if wait_for_response else None,
            )
        except Exception as exc:
            set_last_response(f"ZOE error: {exc}")
            raise

        answer = extract_response_text(result)
        if answer:
            set_last_response(answer)
        return result


def start_stt() -> None:
    global _stt_started, _stt_failed, _stt_error

    with _runtime_lock:
        if _stt_started:
            print("[ZOE STT] Already running.")
            return
        if _runtime_shutdown:
            raise RuntimeError("Cannot start STT after ZOE shutdown.")

        print("[ZOE] Initializing STT...")

        try:
            stt_start(
                on_final=on_user_speech,
                on_interruption=interrupt_zoe,
                on_speech_start=on_user_speech_start,
                on_speech_end=on_user_speech_end,
                on_partial=on_user_speech_partial,
            )
        except Exception as exc:
            _stt_failed = True
            _stt_error = str(exc)
            print("[ZOE STT] Startup failed:")
            print(f"    {type(exc).__name__}: {exc}")
            raise

        _stt_started = True
        _stt_failed = False
        _stt_error = ""

        print("[ZOE] STT ready.")
        _ensure_voice_worker()
        _start_idle_watchdog()

        if is_zoe_muted():
            if not _arm_wakeword():
                print("[ZOE WAKE] Warning: wakeword could not be started.")


def get_stt_status() -> dict:
    with _runtime_lock:
        started = _stt_started
        failed = _stt_failed
        error = _stt_error
    return {"started": started, "failed": failed, "error": error}


def get_zoe_snapshot() -> dict:
    with _state_lock:
        muted = _zoe_muted
        auto_muted = _auto_muted
        user_speaking = _user_speaking
        thinking = _zoe_thinking
        internal_speaking = _zoe_speaking

    with _processing_lock:
        processing = _processing_speech
        active_request_id = _active_request_id

    stt = get_stt_data()
    speaking = is_zoe_actually_speaking()
    runtime_snapshot = get_runtime_snapshot()
    current_url = ""

    if isinstance(runtime_snapshot, dict):
        runtime_url = runtime_snapshot.get("url", "")
        if isinstance(runtime_url, str):
            current_url = runtime_url.strip()

    return {
        "online": _runtime_started and not _runtime_shutdown,
        "state": get_zoe_state(),
        "speaking": speaking,
        "listening": user_speaking,
        "thinking": thinking,
        "processing": processing,
        "muted": muted,
        "auto_muted": auto_muted,
        "user_speaking": user_speaking,
        "zoe_speaking": speaking,
        "internal_speaking": internal_speaking,
        "stt": stt,
        "response": get_last_response(),
        "url": current_url,
        "active_request_id": active_request_id,
        "stt_status": get_stt_status(),
        "queued_requests": _voice_queue.qsize(),
        "runtime": runtime_snapshot,
        "event_subscribers": get_runtime_event_subscriber_count(),
    }


def stop_zoe_speaking() -> None:
    interrupt_zoe()


def shutdown_zoe() -> None:
    global _runtime_shutdown, _runtime_started, _stt_started
    global _processing_speech, _active_request_id, _voice_worker

    with _runtime_lock:
        if _runtime_shutdown:
            return
        print()
        print("[ZOE] Shutting down...")
        _runtime_shutdown = True

    _idle_watchdog_stop.set()
    _disarm_wakeword()
    _next_tts_generation()
    _voice_worker_stop.set()
    _clear_voice_queue()

    with _processing_lock:
        _processing_speech = False
        _active_request_id = None

    try:
        stt_stop()
    except Exception as exc:
        print("[ZOE] STT shutdown error:", exc)

    _stt_started = False

    try:
        stop_tts()
    except Exception as exc:
        print("[ZOE] TTS shutdown error:", exc)

    clear_zoe_speaking_state()
    set_user_speaking(False)
    set_zoe_thinking(False)
    clear_current_stt()

    worker = _voice_worker
    if worker is not None and worker is not threading.current_thread():
        worker.join(timeout=5.0)
        if worker.is_alive():
            print("[ZOE] Voice request worker did not terminate in time.")

    _voice_worker = None

    if runtime is not None:
        try:
            runtime.shutdown()
        except Exception as exc:
            print("[ZOE] Runtime shutdown error:", exc)

    _runtime_started = False

    with _runtime_event_lock:
        for subscriber_queue in _runtime_event_subscribers.values():
            try:
                subscriber_queue.put_nowait(None)
            except queue.Full:
                try:
                    subscriber_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber_queue.put_nowait(None)
                except queue.Full:
                    pass
        _runtime_event_subscribers.clear()

    print("[ZOE] Shutdown complete.")


def main() -> None:
    try:
        start_stt()
        print()
        print("=" * 60)
        print("                    ZOE MARK 35 ONLINE")
        print("=" * 60)
        print()

        while True:
            try:
                message = input("you> ").strip()
            except EOFError:
                break

            if not message:
                continue
            if message.lower() in {"exit", "quit"}:
                break
            _enqueue_voice_request(message)

    except KeyboardInterrupt:
        print("\n[ZOE] Interrupted.")
    finally:
        shutdown_zoe()


if __name__ == "__main__":
    main()