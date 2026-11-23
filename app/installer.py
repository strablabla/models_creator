"""Second half of install.cmd, the same on Windows and Linux: once uv has
installed Python and the libraries, puts a "Models creator" icon on the
Desktop and in the applications menu, then offers to start the application.

    uv run python -m app.installer          (install.cmd runs it)
    uv run python -m app.installer --yes    no question: starts it at the end
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "Models creator"
COMMENT = "Annotate images, train and test U-Net segmentation models"


def venv_python(windowless=False):
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / ("pythonw.exe" if windowless else "python.exe")
    return ROOT / ".venv" / "bin" / "python"


def windows_shortcuts():
    """Desktop and Start menu shortcuts. They run launch.py with pythonw.exe,
    which opens no window at all."""
    script = rf"""
$Shell = New-Object -ComObject WScript.Shell
foreach ($Dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {{
    $Link = $Shell.CreateShortcut((Join-Path $Dir "{NAME}.lnk"))
    $Link.TargetPath       = "{venv_python(windowless=True)}"
    $Link.Arguments        = '"{ROOT / 'launch.py'}"'
    $Link.WorkingDirectory = "{ROOT}"
    $Link.Description      = "{COMMENT}"
    $Link.IconLocation     = "{ROOT / 'static' / 'icon.ico'},0"
    $Link.Save()
    Write-Output (Join-Path $Dir "{NAME}.lnk")
}}
"""
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()


def linux_desktop_dir():
    try:
        d = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True).stdout.strip()
        if d and Path(d).is_dir():
            return Path(d)
    except OSError:
        pass
    for name in ("Desktop", "Bureau"):
        if (Path.home() / name).is_dir():
            return Path.home() / name
    return None


def linux_shortcuts():
    """A .desktop file in the applications menu and on the Desktop."""
    entry = f"""[Desktop Entry]
Type=Application
Name={NAME}
Comment={COMMENT}
Exec="{venv_python()}" "{ROOT / 'launch.py'}"
Path={ROOT}
Icon={ROOT / 'static' / 'icon-256.png'}
Terminal=false
Categories=Science;Graphics;
"""
    made = []
    apps = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    targets = [apps / "models-creator.desktop"]
    desk = linux_desktop_dir()
    if desk:
        targets.append(desk / "models-creator.desktop")
    for t in targets:
        t.write_text(entry, encoding="utf-8")
        t.chmod(0o755)
        # GNOME shows Desktop launchers greyed out until they are marked trusted
        try:
            subprocess.run(["gio", "set", str(t), "metadata::trusted", "true"], capture_output=True)
        except OSError:  # no gio: not GNOME, nothing to do
            pass
        made.append(str(t))
    return made


def start():
    if os.name == "nt":
        subprocess.Popen([str(venv_python(windowless=True)), str(ROOT / "launch.py")], cwd=ROOT,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        subprocess.Popen([str(venv_python()), str(ROOT / "launch.py")], cwd=ROOT, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    print()
    try:
        made = windows_shortcuts() if os.name == "nt" else linux_shortcuts()
    except Exception as e:
        print(f"Could not create the shortcuts: {e}")
        print(f"The application can still be started with:  uv run launch.py   (in {ROOT})")
        return 1
    print(f"{NAME} is installed. Shortcuts:")
    for m in made:
        print("   ", m)
    print()
    if "--yes" in sys.argv:
        answer = "y"
    else:
        try:
            answer = input("Start it now? [Y/n] ").strip().lower()
        except EOFError:
            answer = "n"
    if answer in ("", "y", "yes", "o", "oui"):
        start()
        print("Starting: the page opens in your browser in a few seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
