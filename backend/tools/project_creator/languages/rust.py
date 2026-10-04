import os
import subprocess


# Windows Dev directory
DEV_DIRECTORY = os.path.join(os.path.expanduser("~"), "Dev")


def create_rust_project(name):
    project_directory = os.path.join(DEV_DIRECTORY, name)

    # Create Rust project
    subprocess.run(
        ["cargo", "new", name],
        cwd=DEV_DIRECTORY,
        check=True,
    )

    # Open project in VS Code
    code_command = "code.cmd" if os.name == "nt" else "code"

    subprocess.Popen(
        [code_command, "."],
        cwd=project_directory,
    )

    print(f"Created Rust project: {project_directory}")