from __future__ import annotations

import socket
import threading
from typing import Callable, List, Optional


ESP32_HOST = "0.0.0.0"
ESP32_PORT = 5000
ESP32_RECV_SIZE = 4096

AudioCallback = Callable[[bytes], None]


class ESP32AudioBridge:

    _instance: Optional["ESP32AudioBridge"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ESP32AudioBridge":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._socket_lock = threading.RLock()
        self._esp32_connected = False

        self._subscribers: List[AudioCallback] = []
        self._subscribers_lock = threading.RLock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._ref_count = 0
        self._ref_lock = threading.RLock()

    # --------------------------------------------------
    # SUBSCRIBE / UNSUBSCRIBE
    # --------------------------------------------------

    def subscribe(self, callback: AudioCallback):
        with self._subscribers_lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
        self._acquire()

    def unsubscribe(self, callback: AudioCallback):
        with self._subscribers_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
        self._release()

    def _acquire(self):
        with self._ref_lock:
            self._ref_count += 1
            if self._ref_count == 1:
                self._start()

    def _release(self):
        with self._ref_lock:
            self._ref_count = max(0, self._ref_count - 1)
            if self._ref_count == 0:
                self._stop()

    # --------------------------------------------------
    # START / STOP (only runs when ref_count crosses 0<->1)
    # --------------------------------------------------

    def _start(self):
        if self._running:
            return
        self._running = True

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((ESP32_HOST, ESP32_PORT))
        server.listen(1)
        server.settimeout(1.0)

        with self._socket_lock:
            self._server_socket = server

        self._thread = threading.Thread(
            target=self._loop, name="ESP32-Audio-Bridge", daemon=True
        )
        self._thread.start()

        print(f"[ESP32 BRIDGE] Listening on {ESP32_HOST}:{ESP32_PORT}")

    def _stop(self):
        if not self._running:
            return
        self._running = False

        with self._socket_lock:
            client = self._client_socket
            server = self._server_socket
            self._client_socket = None
            self._server_socket = None
            self._esp32_connected = False

        for sock in (client, server):
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

        print("[ESP32 BRIDGE] Stopped.")

    # --------------------------------------------------
    # ACCEPT + RECEIVE LOOP
    # --------------------------------------------------

    def _accept_client(self) -> Optional[socket.socket]:
        with self._socket_lock:
            server = self._server_socket
        if server is None:
            return None
        try:
            client, address = server.accept()
        except (socket.timeout, OSError):
            return None

        client.settimeout(1.0)
        with self._socket_lock:
            old = self._client_socket
            self._client_socket = client
            self._esp32_connected = True
        if old is not None:
            try:
                old.close()
            except Exception:
                pass

        print(f"[ESP32 BRIDGE] Microphone connected: {address[0]}:{address[1]}")
        return client

    def _loop(self):
        client = None
        while self._running:
            if client is None:
                client = self._accept_client()
                if client is None:
                    continue

            try:
                data = client.recv(ESP32_RECV_SIZE)
            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError,
                     ConnectionAbortedError, OSError):
                data = None
            except Exception as e:
                print(f"[ESP32 BRIDGE] Receive error: {e}")
                data = None

            if not data:
                try:
                    client.close()
                except Exception:
                    pass
                with self._socket_lock:
                    if self._client_socket is client:
                        self._client_socket = None
                    self._esp32_connected = False
                print("[ESP32 BRIDGE] Microphone disconnected.")
                client = None
                continue

            self._broadcast(data)

    def _broadcast(self, data: bytes):
        with self._subscribers_lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(data)
            except Exception as e:
                print(f"[ESP32 BRIDGE] Subscriber error: {e}")

    @property
    def is_esp32_connected(self) -> bool:
        with self._socket_lock:
            return self._esp32_connected


def get_bridge() -> ESP32AudioBridge:
    return ESP32AudioBridge.get_instance()