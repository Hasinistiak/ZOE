from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeCommand:
    name: str
    description: str
    handler_factory: Callable[[], Callable[[], Any]]


_RUNTIME_COMMANDS: dict[str, RuntimeCommand] = {}


def _register_command(
    name: str,
    description: str,
    handler_factory: Callable[[], Callable[[], Any]],
) -> None:
    if name in _RUNTIME_COMMANDS:
        raise ValueError(f"Runtime command already registered: {name}")

    _RUNTIME_COMMANDS[name] = RuntimeCommand(
        name=name,
        description=description,
        handler_factory=handler_factory,
    )



# ============================================================
# MUTE / STANDBY
# ============================================================

def _make_mute_handler_factory() -> Callable[[], Callable[[], Any]]:
    def factory() -> Callable[[], Any]:
        import requests

        def _handler() -> dict[str, Any]:
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/api/microphone/mute",
                    timeout=10,
                )
                response.raise_for_status()

                try:
                    result = response.json()
                except ValueError:
                    result = {}

                return {
                    "success": True,
                    "command": "mute",
                    "standby": True,
                    "result": result,
                }

            except requests.RequestException as exc:
                return {
                    "success": False,
                    "command": "mute",
                    "standby": False,
                    "error": str(exc),
                }

        return _handler

    return factory


_register_command(
    "mute",
    "Put ZOE into standby/sleep mode through the microphone mute API.",
    _make_mute_handler_factory(),
)

# ============================================================
# PUBLIC API
# ============================================================

def get_runtime_commands() -> dict[str, RuntimeCommand]:
    return dict(_RUNTIME_COMMANDS)


def get_runtime_command_descriptions() -> list[dict[str, str]]:
    return [
        {
            "name": cmd.name,
            "description": cmd.description,
        }
        for cmd in _RUNTIME_COMMANDS.values()
    ]


def execute_runtime_command(command_name: str) -> dict[str, Any]:
    cmd = _RUNTIME_COMMANDS.get(command_name)

    if cmd is None:
        raise ValueError(f"Unknown runtime command: {command_name}")

    handler = cmd.handler_factory()
    return handler()


def validate_runtime_command(command_name: str) -> bool:
    return command_name in _RUNTIME_COMMANDS