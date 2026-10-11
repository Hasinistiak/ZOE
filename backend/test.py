
import difflib
import json
import os
import shutil
import string
import subprocess
import sys
import winreg


def start_apps() -> dict:
    """{display name: AppID} for everything Windows shows in Start."""
    cmd = (
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
        "Get-StartApps | ConvertTo-Json -Compress"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    if not out:
        return {}
    data = json.loads(out)
    if isinstance(data, dict):
        data = [data]
    return {a["Name"]: a["AppID"] for a in data}


def app_paths_lookup(name: str):
    """Look the exe up in the App Paths registry (HKLM and HKCU)."""
    exe = name if name.lower().endswith(".exe") else name + ".exe"
    key_path = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, key_path) as k:
                path, _ = winreg.QueryValueEx(k, None)
                if path and os.path.exists(path.strip('"')):
                    return path.strip('"')
        except OSError:
            continue
    return None


def deep_search(name: str):
    """Walk every drive looking for <name>.exe. Slow, last resort."""
    exe = (name if name.lower().endswith(".exe") else name + ".exe").lower()
    skip = {"$recycle.bin", "system volume information", "windows"}
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames if d.lower() not in skip]
            for f in filenames:
                if f.lower() == exe:
                    return os.path.join(dirpath, f)
    return None


def launch_start_app(app_id: str):
    # shell:AppsFolder works for both classic desktop apps and Store apps
    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])


def open_app(query: str, deep: bool = False) -> bool:
    q = query.strip()

    # 0. Direct file/folder/URL path: let Windows pick the handler
    if os.path.exists(q):
        os.startfile(q)
        return True

    # 1. Start Menu index (exact, then substring, then fuzzy)
    apps = start_apps()
    lower = {n.lower(): n for n in apps}
    match = lower.get(q.lower())
    if not match:
        subs = [n for l, n in lower.items() if q.lower() in l]
        match = min(subs, key=len) if subs else None
    if not match:
        close = difflib.get_close_matches(q.lower(), lower.keys(), n=1, cutoff=0.6)
        match = lower[close[0]] if close else None
    if match:
        print(f"Opening: {match}")
        launch_start_app(apps[match])
        return True

    # 2. App Paths registry
    path = app_paths_lookup(q)
    if path:
        print(f"Opening: {path}")
        subprocess.Popen([path])
        return True

    # 3. PATH
    path = shutil.which(q)
    if path:
        print(f"Opening: {path}")
        subprocess.Popen([path])
        return True

    # 4. Deep scan (opt-in)
    if deep:
        print("Scanning all drives, this can take a while...")
        path = deep_search(q)
        if path:
            print(f"Opening: {path}")
            subprocess.Popen([path])
            return True

    print(f"Couldn't find '{query}'. Try --list, or re-run with --deep.")
    return False


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    if args[0] == "--list":
        for n in sorted(start_apps(), key=str.lower):
            print(n)
        sys.exit(0)
    deep = "--deep" in args
    name = " ".join(a for a in args if a != "--deep")
    sys.exit(0 if open_app(name, deep) else 1)