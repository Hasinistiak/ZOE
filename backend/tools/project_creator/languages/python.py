import os
import subprocess


DEV_DIRECTORY = os.path.join(os.path.expanduser("~"), "Dev")


def create_py_project(name):
    project_directory = os.path.join(DEV_DIRECTORY, name)

    # Create project directory
    os.makedirs(project_directory, exist_ok=False)

    # Create virtual environment
    subprocess.run(
        ["python", "-m", "venv", "venv"],
        cwd=project_directory,
        check=True,
    )

    # Create main.py
    main_file = os.path.join(project_directory, "main.py")

    with open(main_file, "w", encoding="utf-8") as file:
        file.write("")

    # Open in VS Code
    code_command = "code.cmd" if os.name == "nt" else "code"

    subprocess.Popen(
        [code_command, "."],
        cwd=project_directory,
    )

    print(f"Created Python project: {project_directory}")