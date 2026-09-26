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
        from backend.zoe import mute_zoe

        def _handler() -> dict[str, Any]:
            mute_zoe()

            return {
                "success": True,
                "command": "mute",
                "standby": True,
            }

        return _handler

    return factory


_register_command(
    "mute",
    "Put ZOE into standby/sleep mode. Stop active participation and wait until ZOE is awakened again.",
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