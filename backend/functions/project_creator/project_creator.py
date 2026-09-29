from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from backend.functions.project_creator.languages.python import create_py_project
from backend.functions.project_creator.languages.react import create_react_project
from backend.functions.project_creator.languages.rust import create_rust_project
from backend.functions.project_creator.languages.tauri import create_tauri_project
from backend.functions.project_creator.languages.cpp import create_cpp_project
from backend.functions.project_creator.languages.reactnative import (
    create_reactnative_project,
)


# ============================================================
# CONFIG
# ============================================================

DEV_DIRECTORY = Path.home() / "Dev"


LANGUAGES = {
    "python": create_py_project,
    "c++": create_cpp_project,
    "cpp": create_cpp_project,
    "react": create_react_project,
    "react native": create_reactnative_project,
    "reactnative": create_reactnative_project,
    "rust": create_rust_project,
    "tauri": create_tauri_project,
}


# ============================================================
# RESULT
# ============================================================

@dataclass
class ProjectResult:
    success: bool
    name: str
    language: str
    destination: str
    path: str | None = None
    github_repo: str | None = None
    message: str = ""


# ============================================================
# EXECUTABLE DISCOVERY
# ============================================================

def find_command(command: str) -> str | None:
    """
    Locate an executable in PATH and common Windows installation paths.
    """

    # --------------------------------------------------------
    # First: normal PATH lookup
    # --------------------------------------------------------

    found = shutil.which(command)

    if found:
        return found

    # --------------------------------------------------------
    # Windows fallback paths
    # --------------------------------------------------------

    if os.name == "nt":

        common_paths = {
            "gh": [
                Path(r"C:\Program Files\GitHub CLI\gh.exe"),
                Path.home()
                / r"AppData\Local\Programs\GitHub CLI\gh.exe",
            ],
            "git": [
                Path(r"C:\Program Files\Git\cmd\git.exe"),
                Path(r"C:\Program Files\Git\bin\git.exe"),
            ],
        }

        for path in common_paths.get(command, []):
            if path.exists():
                return str(path)

    return None


def executable(command: str) -> str:
    """
    Return the full path to an executable.

    Raises:
        RuntimeError: If the executable cannot be found.
    """

    path = find_command(command)

    if not path:
        raise RuntimeError(
            f"{command} is not installed or could not be found."
        )

    return path


def check_command(command: str) -> bool:
    """
    Check whether a command exists and can execute.
    """

    path = find_command(command)

    if not path:
        return False

    try:
        subprocess.run(
            [path, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        return True

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        OSError,
    ):
        return False


# ============================================================
# COMMAND UTILITIES
# ============================================================

def run_command(
    command: list[str],
    cwd: Path,
) -> subprocess.CompletedProcess:
    """
    Run a command and raise an error when it fails.
    """

    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def run_tool(
    tool: str,
    args: list[str],
    cwd: Path,
) -> subprocess.CompletedProcess:
    """
    Run an external tool using its discovered executable path.
    """

    return run_command(
        [executable(tool), *args],
        cwd,
    )


# ============================================================
# VALIDATION
# ============================================================

def valid_project_name(name: str) -> bool:
    """
    Validate a Windows-compatible project directory name.
    """

    if not name:
        return False

    invalid_chars = '<>:"/\\|?*'

    if any(char in name for char in invalid_chars):
        return False

    if name[-1] in (" ", "."):
        return False

    # Prevent names beginning with numbers.
    if not (name[0].isalpha() or name[0] == "_"):
        return False

    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    return name.upper() not in reserved


def normalize_language(language: str) -> str | None:
    """
    Normalize language/framework input.
    """

    if not language:
        return None

    return language.strip().lower()


def get_existing_projects() -> set[str]:
    """
    Return existing directories inside ~/Dev.
    """

    try:
        DEV_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        return {
            item.name.lower()
            for item in DEV_DIRECTORY.iterdir()
            if item.is_dir()
        }

    except OSError:
        return set()


# ============================================================
# GIT
# ============================================================

def run_git(
    args: list[str],
    cwd: Path,
) -> subprocess.CompletedProcess:
    """
    Run Git inside the project directory.
    """

    return run_tool(
        "git",
        args,
        cwd,
    )


def initialize_git(project_path: Path) -> None:
    """
    Initialize Git and create the initial commit.
    """

    # --------------------------------------------------------
    # Check Git
    # --------------------------------------------------------

    if not check_command("git"):
        raise RuntimeError(
            "Git is not installed or is not available."
        )

    git_dir = project_path / ".git"

    # --------------------------------------------------------
    # Initialize repository
    # --------------------------------------------------------

    if not git_dir.exists():
        print("[ZOE] Initializing Git...")

        run_git(
            ["init"],
            project_path,
        )

    else:
        print("[ZOE] Git repository already exists.")

    # --------------------------------------------------------
    # Stage project files
    # --------------------------------------------------------

    print("[ZOE] Staging project files...")

    run_git(
        ["add", "."],
        project_path,
    )

    # --------------------------------------------------------
    # Check staged files
    # --------------------------------------------------------

    status = run_git(
        [
            "status",
            "--porcelain",
        ],
        project_path,
    )

    # --------------------------------------------------------
    # Create initial commit
    # --------------------------------------------------------

    if status.stdout.strip():

        print("[ZOE] Creating initial commit...")

        run_git(
            [
                "-c",
                "user.name=ZOE",
                "-c",
                "user.email=zoe@localhost",
                "commit",
                "-m",
                "Initial project creation",
            ],
            project_path,
        )

    else:

        # ----------------------------------------------------
        # Verify whether a commit already exists
        # ----------------------------------------------------

        has_commit = subprocess.run(
            [
                executable("git"),
                "rev-parse",
                "--verify",
                "HEAD",
            ],
            cwd=project_path,
            capture_output=True,
            text=True,
        )

        if has_commit.returncode != 0:
            raise RuntimeError(
                "The project contains no files and no Git commit exists."
            )

        print("[ZOE] Existing Git commit detected.")


def get_git_remote(
    project_path: Path,
) -> str | None:
    """
    Return the origin URL if one exists.
    """

    result = subprocess.run(
        [
            executable("git"),
            "remote",
            "get-url",
            "origin",
        ],
        cwd=project_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    remote = result.stdout.strip()

    return remote or None


# ============================================================
# GITHUB
# ============================================================

def check_github_auth() -> None:
    """
    Verify GitHub CLI installation and authentication.
    """

    # --------------------------------------------------------
    # Check GitHub CLI
    # --------------------------------------------------------

    gh_path = find_command("gh")

    if not gh_path:
        raise RuntimeError(
            "GitHub CLI (gh) is installed, "
            "but could not be located.\n\n"
            "Expected location:\n"
            r"C:\Program Files\GitHub CLI\gh.exe"
        )

    print(f"[ZOE] GitHub CLI: {gh_path}")

    # --------------------------------------------------------
    # Check authentication
    # --------------------------------------------------------

    print("[ZOE] Checking GitHub authentication...")

    result = subprocess.run(
        [
            gh_path,
            "auth",
            "status",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        error = (
            result.stderr.strip()
            or result.stdout.strip()
            or "GitHub authentication failed."
        )

        raise RuntimeError(
            "GitHub CLI is not authenticated.\n"
            f"{error}\n\n"
            "Run:\n"
            "gh auth login"
        )


def get_github_username() -> str:
    """
    Get the currently authenticated GitHub username.
    """

    result = run_tool(
        "gh",
        [
            "api",
            "user",
            "--jq",
            ".login",
        ],
        Path.cwd(),
    )

    username = result.stdout.strip()

    if not username:
        raise RuntimeError(
            "Could not determine the authenticated GitHub username."
        )

    return username


def create_github_repository(
    project_path: Path,
    name: str,
    private: bool = True,
) -> str:
    """
    Create the GitHub repository and push the local project.
    """

    # --------------------------------------------------------
    # Verify GitHub
    # --------------------------------------------------------

    check_github_auth()

    # --------------------------------------------------------
    # Get authenticated account
    # --------------------------------------------------------

    print("[ZOE] Getting GitHub account...")

    username = get_github_username()

    print(f"[ZOE] GitHub account: {username}")

    # --------------------------------------------------------
    # Repository visibility
    # --------------------------------------------------------

    visibility = (
        "--private"
        if private
        else "--public"
    )

    print(
        f"[ZOE] Creating "
        f"{'private' if private else 'public'} "
        f"GitHub repository..."
    )

    # --------------------------------------------------------
    # Check existing remote
    # --------------------------------------------------------

    existing_remote = get_git_remote(
        project_path
    )

    if existing_remote:

        print(
            f"[ZOE] Existing origin detected: "
            f"{existing_remote}"
        )

        # ----------------------------------------------------
        # If an origin already exists, push to it.
        # ----------------------------------------------------

        print("[ZOE] Pushing existing repository...")

        run_git(
            [
                "branch",
                "-M",
                "main",
            ],
            project_path,
        )

        run_git(
            [
                "push",
                "-u",
                "origin",
                "main",
            ],
            project_path,
        )

        # ----------------------------------------------------
        # Verify repository URL
        # ----------------------------------------------------

        url_result = run_tool(
            "gh",
            [
                "repo",
                "view",
                "--json",
                "url",
                "--jq",
                ".url",
            ],
            project_path,
        )

        github_url = url_result.stdout.strip()

        if not github_url:
            raise RuntimeError(
                "Repository exists and was pushed, "
                "but GitHub URL could not be retrieved."
            )

        return github_url

    # --------------------------------------------------------
    # Full repository name
    # --------------------------------------------------------

    repository = f"{username}/{name}"

    print(
        f"[ZOE] Repository: {repository}"
    )

    # --------------------------------------------------------
    # Create repository
    # --------------------------------------------------------

    result = subprocess.run(
        [
            executable("gh"),
            "repo",
            "create",
            repository,
            visibility,
            "--source",
            str(project_path),
            "--remote",
            "origin",
        ],
        cwd=project_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        error = (
            result.stderr.strip()
            or result.stdout.strip()
            or "GitHub repository creation failed."
        )

        raise RuntimeError(error)

    # --------------------------------------------------------
    # Verify origin
    # --------------------------------------------------------

    remote = get_git_remote(
        project_path
    )

    if not remote:

        raise RuntimeError(
            "GitHub repository was created, "
            "but the origin remote was not configured."
        )

    print(
        f"[ZOE] Remote: {remote}"
    )

    # --------------------------------------------------------
    # Ensure main branch
    # --------------------------------------------------------

    print("[ZOE] Setting main branch...")

    run_git(
        [
            "branch",
            "-M",
            "main",
        ],
        project_path,
    )

    # --------------------------------------------------------
    # Push
    # --------------------------------------------------------

    print("[ZOE] Pushing project to GitHub...")

    run_git(
        [
            "push",
            "-u",
            "origin",
            "main",
        ],
        project_path,
    )

    # --------------------------------------------------------
    # Verify GitHub repository
    # --------------------------------------------------------

    print(
        "[ZOE] Verifying GitHub repository..."
    )

    url_result = run_tool(
        "gh",
        [
            "repo",
            "view",
            repository,
            "--json",
            "url",
            "--jq",
            ".url",
        ],
        project_path,
    )

    github_url = url_result.stdout.strip()

    if not github_url:

        raise RuntimeError(
            "GitHub repository was created "
            "but no repository URL was returned."
        )

    # --------------------------------------------------------
    # Verify remote repository has commits
    # --------------------------------------------------------

    print(
        "[ZOE] Verifying remote repository..."
    )

    verification = subprocess.run(
        [
            executable("git"),
            "ls-remote",
            "--heads",
            "origin",
            "main",
        ],
        cwd=project_path,
        capture_output=True,
        text=True,
    )

    if verification.returncode != 0:

        raise RuntimeError(
            "Repository exists, but the remote branch "
            "could not be verified.\n"
            f"{verification.stderr.strip()}"
        )

    if not verification.stdout.strip():

        raise RuntimeError(
            "GitHub repository exists, "
            "but main branch contains no pushed commit."
        )

    print(
        "[ZOE] Remote repository verified."
    )

    return github_url


# ============================================================
# PROJECT CREATION
# ============================================================

def create_project(
    name: str,
    language: str,
    destination: str = "local",
    github_private: bool = True,
) -> ProjectResult:

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    name = name.strip()

    language_key = normalize_language(
        language
    )

    destination = destination.strip().lower()

    # --------------------------------------------------------
    # Validate project name
    # --------------------------------------------------------

    if not valid_project_name(name):

        return ProjectResult(
            success=False,
            name=name,
            language=language,
            destination=destination,
            message="Invalid project name.",
        )

    # --------------------------------------------------------
    # Validate destination
    # --------------------------------------------------------

    if destination not in {
        "local",
        "github",
    }:

        return ProjectResult(
            success=False,
            name=name,
            language=language,
            destination=destination,
            message=(
                "Destination must be "
                "'local' or 'github'."
            ),
        )

    # --------------------------------------------------------
    # Validate language
    # --------------------------------------------------------

    creator = LANGUAGES.get(
        language_key
    )

    if creator is None:

        return ProjectResult(
            success=False,
            name=name,
            language=language,
            destination=destination,
            message=(
                "Unsupported language or framework: "
                f"{language}"
            ),
        )

    # --------------------------------------------------------
    # Check duplicate
    # --------------------------------------------------------

    if name.lower() in get_existing_projects():

        return ProjectResult(
            success=False,
            name=name,
            language=language,
            destination=destination,
            message=(
                f"Project '{name}' already exists."
            ),
        )

    # --------------------------------------------------------
    # Prepare Dev directory
    # --------------------------------------------------------

    try:

        DEV_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as error:

        return ProjectResult(
            success=False,
            name=name,
            language=language,
            destination=destination,
            message=(
                "Could not create Dev directory: "
                f"{error}"
            ),
        )

    project_path = (
        DEV_DIRECTORY / name
    )

    # ========================================================
    # CREATION
    # ========================================================

    try:

        print()
        print(
            "═════════════════════════════════════════════"
        )
        print(
            "             ZOE PROJECT CREATOR"
        )
        print(
            "═════════════════════════════════════════════"
        )
        print()

        print(
            "[ZOE] Starting project creation..."
        )

        print(
            f"[ZOE] Project     : {name}"
        )

        print(
            f"[ZOE] Language    : {language}"
        )

        print(
            f"[ZOE] Destination : {destination}"
        )

        print()

        # ----------------------------------------------------
        # Create language project
        # ----------------------------------------------------

        print(
            f"[ZOE] Running {language} project creator..."
        )

        creator(name)

        # ----------------------------------------------------
        # Verify project directory
        # ----------------------------------------------------

        if not project_path.exists():

            return ProjectResult(
                success=False,
                name=name,
                language=language,
                destination=destination,
                message=(
                    "Project creator completed, "
                    "but the expected project directory "
                    "was not found:\n"
                    f"{project_path}"
                ),
            )

        if not project_path.is_dir():

            return ProjectResult(
                success=False,
                name=name,
                language=language,
                destination=destination,
                path=str(project_path),
                message=(
                    "The expected project path exists "
                    "but is not a directory."
                ),
            )

        print(
            "[ZOE] Project created at:"
        )

        print(
            f"      {project_path}"
        )

        # ====================================================
        # LOCAL
        # ====================================================

        if destination == "local":

            return ProjectResult(
                success=True,
                name=name,
                language=language,
                destination="local",
                path=str(project_path),
                message=(
                    "Project created locally."
                ),
            )

        # ====================================================
        # GITHUB
        # ====================================================

        initialize_git(
            project_path
        )

        github_repo = create_github_repository(
            project_path=project_path,
            name=name,
            private=github_private,
        )

        # ----------------------------------------------------
        # Final verification
        # ----------------------------------------------------

        print()
        print(
            "[ZOE] GitHub publication verified."
        )

        return ProjectResult(
            success=True,
            name=name,
            language=language,
            destination="github",
            path=str(project_path),
            github_repo=github_repo,
            message=(
                "Project created locally "
                "and published to GitHub."
            ),
        )

    # ========================================================
    # GIT / SUBPROCESS ERROR
    # ========================================================

    except subprocess.CalledProcessError as error:

        stderr = (
            error.stderr.strip()
            if error.stderr
            else ""
        )

        stdout = (
            error.stdout.strip()
            if error.stdout
            else ""
        )

        message = (
            stderr
            or stdout
            or str(error)
        )

        return ProjectResult(
            success=False,
            name=name,
            language=language,
            destination=destination,
            path=str(project_path),
            message=message,
        )

    # ========================================================
    # EXPECTED RUNTIME ERROR
    # ========================================================

    except RuntimeError as error:

        return ProjectResult(
            success=False,
            name=name,
            language=language,
            destination=destination,
            path=str(project_path),
            message=str(error),
        )

    # ========================================================
    # UNKNOWN ERROR
    # ========================================================

    except Exception as error:

        return ProjectResult(
            success=False,
            name=name,
            language=language,
            destination=destination,
            path=str(project_path),
            message=(
                f"Unexpected error: {error}"
            ),
        )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="ZOE Project Creator",
    )

    # --------------------------------------------------------
    # Project name
    # --------------------------------------------------------

    parser.add_argument(
        "name",
        help="Project name",
    )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    parser.add_argument(
        "language",
        help=(
            "Project language/framework "
            "(python, cpp, react, reactnative, rust, tauri)"
        ),
    )

    # --------------------------------------------------------
    # Destination
    # --------------------------------------------------------

    parser.add_argument(
        "--destination",
        choices=[
            "local",
            "github",
        ],
        default="local",
        help=(
            "Project destination. "
            "Default: local"
        ),
    )

    # --------------------------------------------------------
    # GitHub visibility
    # --------------------------------------------------------

    parser.add_argument(
        "--public",
        action="store_true",
        help=(
            "Create a public GitHub repository. "
            "Default is private."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "═════════════════════════════════════════════"
    )
    print(
        "             ZOE PROJECT CREATOR"
    )
    print(
        "═════════════════════════════════════════════"
    )
    print()

    print(
        f"Project     : {args.name}"
    )

    print(
        f"Language    : {args.language}"
    )

    print(
        f"Destination : {args.destination}"
    )

    if args.destination == "github":

        visibility = (
            "public"
            if args.public
            else "private"
        )

        print(
            f"Visibility  : {visibility}"
        )

    print()

    # ========================================================
    # CREATE
    # ========================================================

    result = create_project(
        name=args.name,
        language=args.language,
        destination=args.destination,
        github_private=not args.public,
    )

    print()

    # ========================================================
    # SUCCESS
    # ========================================================

    if result.success:

        print(
            "═════════════════════════════════════════════"
        )

        print(
            "              ✓ PROJECT CREATED"
        )

        print(
            "═════════════════════════════════════════════"
        )

        print()

        print(
            f"Name        : {result.name}"
        )

        print(
            f"Language    : {result.language}"
        )

        print(
            f"Destination : {result.destination}"
        )

        print(
            f"Path        : {result.path}"
        )

        if result.github_repo:

            print(
                f"GitHub      : {result.github_repo}"
            )

        print()

        print(
            result.message
        )

        print()

        return

    # ========================================================
    # FAILURE
    # ========================================================

    print(
        "═════════════════════════════════════════════"
    )

    print(
        "           ✗ PROJECT CREATION FAILED"
    )

    print(
        "═════════════════════════════════════════════"
    )

    print()

    print(
        f"Project     : {result.name}"
    )

    print(
        f"Language    : {result.language}"
    )

    print(
        f"Destination : {result.destination}"
    )

    if result.path:

        print(
            f"Path        : {result.path}"
        )

    print()

    print(
        "Reason:"
    )

    print(
        result.message
    )

    print()

    raise SystemExit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()