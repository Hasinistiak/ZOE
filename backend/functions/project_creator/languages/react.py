
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

    # Tell npm and other tools that this is automation.
    env["CI"] = "true"

    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )


def write_file(path: Path, content: str) -> None:
    path.write_text(
        content,
        encoding="utf-8",
    )


# ============================================================
# REACT + VITE PROJECT CREATOR
# ============================================================

def create_react_project(name: str) -> Path:
    """
    Create a React + Vite project without any user interaction.

    The function:
        1. Validates the project name
        2. Creates the Vite project
        3. Installs dependencies
        4. Configures .gitignore
        5. Verifies package.json
        6. Optionally opens VS Code

    Git initialization and GitHub publishing are intentionally
    handled by the central project_creator.py.
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
    # CHECK NODE / NPM
    # ========================================================

    if not command_exists(npm_command):
        raise RuntimeError(
            "Node.js / npm was not found.\n\n"
            "Install Node.js before creating a React project."
        )

    print("[ZOE] Creating React + Vite project...")

    # ========================================================
    # CREATE REACT + VITE
    # ========================================================

    run_command(
        [
            npm_command,
            "create",
            "vite@latest",
            name,
            "--",
            "--template",
            "react",
        ],
        cwd=DEV_DIRECTORY,
    )

    # ========================================================
    # VERIFY PROJECT DIRECTORY
    # ========================================================

    if not project_directory.is_dir():
        raise RuntimeError(
            "Vite reported successful project creation, "
            "but the project directory was not found:\n"
            f"{project_directory}"
        )

    # ========================================================
    # INSTALL DEPENDENCIES
    # ========================================================

    print("[ZOE] Installing React dependencies...")

    run_command(
        [
            npm_command,
            "install",
        ],
        cwd=project_directory,
    )

    # ========================================================
    # VERIFY PACKAGE.JSON
    # ========================================================

    package_json = project_directory / "package.json"

    if not package_json.is_file():
        raise RuntimeError(
            "React project was created, but package.json "
            "was not found."
        )

    # ========================================================
    # CONFIGURE .GITIGNORE
    # ========================================================

    gitignore = """node_modules/
dist/

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
        f"[ZOE] React + Vite project ready:\n"
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
