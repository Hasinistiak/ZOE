import os
import subprocess


# Windows Dev directory
DEV_DIRECTORY = os.path.join(os.path.expanduser("~"), "Dev")


def create_cpp_project(name):
    project_directory = os.path.join(DEV_DIRECTORY, name)

    # Create project directory
    os.makedirs(project_directory, exist_ok=False)

    # Create project folders
    os.makedirs(os.path.join(project_directory, "src"))
    os.makedirs(os.path.join(project_directory, "include"))
    os.makedirs(os.path.join(project_directory, "build"))

    # Create source files
    main_cpp = os.path.join(project_directory, "src", "main.cpp")
    header = os.path.join(project_directory, "include", "myheader.h")

    with open(main_cpp, "w", encoding="utf-8") as file:
        file.write(
            """#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
        )

    with open(header, "w", encoding="utf-8") as file:
        file.write(
            """#pragma once
"""
        )

    # Open project in VS Code
    code_command = "code.cmd" if os.name == "nt" else "code"

    subprocess.Popen(
        [code_command, "."],
        cwd=project_directory,
    )

    print(f"Created C++ project: {project_directory}")