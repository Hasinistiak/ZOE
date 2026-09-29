
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DEV_DIRECTORY = Path.home() / "Dev"


# ============================================================
# HELPERS
# ============================================================

def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def get_command(command: str) -> str:
    """
    Return the platform-appropriate executable.
    """

    if os.name == "nt":
        if command == "npm":
            return "npm.cmd"

        if command == "npx":
            return "npx.cmd"

        if command == "code":
            return "code.cmd"

    return command


def run_command(
    command: list[str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """
    Run an external command completely non-interactively.
    """

    env = os.environ.copy()

    # Tell npm / Tauri tooling this is an automated environment.
    env["CI"] = "true"

    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )


def write_file(
    path: Path,
    content: str,
) -> None:
    path.write_text(
        content,
        encoding="utf-8",
    )


# ============================================================
# TAURI PROJECT CREATOR
# ============================================================

def create_tauri_project(name: str) -> Path:
    """
    Create a Tauri + React + JavaScript project completely
    non-interactively.

    Git initialization and GitHub publishing are handled by
    the central project_creator.py.
    """

    # ========================================================
    # VALIDATE PROJECT NAME
    # ========================================================

    if not re.fullmatch(
        r"[A-Za-z0-9_-]+",
        name,
    ):
        raise ValueError(
            "Project name may only contain letters, "
            "numbers, underscores, and hyphens."
        )

    project_directory = DEV_DIRECTORY / name

    if project_directory.exists():
        raise FileExistsError(
            f"Project already exists:\n{project_directory}"
        )

    DEV_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # COMMANDS
    # ========================================================

    npm_command = get_command("npm")
    code_command = get_command("code")

    # ========================================================
    # CHECK REQUIRED TOOLS
    # ========================================================

    missing = []

    if not command_exists(npm_command):
        missing.append("Node.js / npm")

    if not command_exists("cargo"):
        missing.append("Rust / Cargo")

    if not command_exists("rustc"):
        missing.append("Rust compiler")

    if missing:
        raise RuntimeError(
            "Required tools are missing:\n\n"
            + "\n".join(
                f"• {item}"
                for item in missing
            )
            + "\n\n"
            "Install the required development tools first."
        )

    # ========================================================
    # APPLICATION IDENTIFIER
    # ========================================================

    identifier = f"com.{name.lower()}.app"

    identifier = re.sub(
        r"[^a-z0-9.-]",
        "",
        identifier,
    )

    # ========================================================
    # CREATE TAURI PROJECT
    # ========================================================

    print("[ZOE] Creating Tauri + React project...")

    command = [
        npm_command,
        "create",
        "tauri-app@latest",
        name,
        "--",
        "--template",
        "react",
        "--manager",
        "npm",
        "--flavor",
        "javascript",
        "--identifier",
        identifier,
    ]

    run_command(
        command,
        cwd=DEV_DIRECTORY,
    )

    # ========================================================
    # VERIFY PROJECT DIRECTORY
    # ========================================================

    if not project_directory.is_dir():
        raise RuntimeError(
            "Tauri reported successful project creation, "
            "but the project directory was not found:\n"
            f"{project_directory}"
        )

    # ========================================================
    # INSTALL FRONTEND DEPENDENCIES
    # ========================================================

    print("[ZOE] Installing frontend dependencies...")

    run_command(
        [
            npm_command,
            "install",
        ],
        cwd=project_directory,
    )

    # ========================================================
    # VERIFY TAURI CONFIGURATION
    # ========================================================

    tauri_directory = project_directory / "src-tauri"

    if not tauri_directory.is_dir():
        raise RuntimeError(
            "Tauri project was created, but src-tauri "
            "was not found."
        )

    tauri_config = (
        tauri_directory / "tauri.conf.json"
    )

    if not tauri_config.exists():
        # Tauri versions may use different configuration
        # formats, so don't fail solely on this filename.
        config_files = [
            tauri_directory / "tauri.conf.json",
            tauri_directory / "tauri.conf.json5",
            tauri_directory / "tauri.conf.toml",
        ]

        if not any(
            path.exists()
            for path in config_files
        ):
            raise RuntimeError(
                "Tauri project was created, but no "
                "Tauri configuration file was found."
            )

    # ========================================================
    # CONFIGURE .GITIGNORE
    # ========================================================

    gitignore = """node_modules/

dist/
build/

src-tauri/target/

.env
.env.local
.env.*.local

.vscode/
.idea/

.DS_Store
Thumbs.db
"""

    write_file(
        project_directory / ".gitignore",
        gitignore,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "[ZOE] Tauri + React project ready:\n"
        f"      {project_directory}"
    )

    # ========================================================
    # OPEN VS CODE
    # ========================================================

    if command_exists(code_command):
        print("[ZOE] Opening project in VS Code...")

        subprocess.Popen(
            [
                code_command,
                ".",
            ],
            cwd=project_directory,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return project_directory
