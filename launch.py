"""Starts Models creator and opens it in the browser: what the Desktop
shortcut runs, on Windows and on Linux.

- If the application is already running, it only opens the page.
- Otherwise it starts the server in the background, with no window, writes
  its messages to logs/server.log, and opens the page once it answers.

    launch.py            what the shortcut does
    launch.py --console  runs the server in this terminal, messages on screen
                         (to look into a server that does not start)
"""
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST, PORT = "127.0.0.1", 8765
URL = f"http://{HOST}:{PORT}/"
SERVER = ["-m", "uvicorn", "app.main:app", "--host", HOST, "--port", str(PORT)]


def answers():
    try:
        with urllib.request.urlopen(URL + "api/info", timeout=1) as r:
            return r.status == 200
    except OSError:
        return False


def console_python():
    """The shortcut runs pythonw.exe (no window) on Windows; the server is
    started with python.exe in a hidden console instead, so that programs it
    starts (nvidia-smi...) use that console rather than opening their own."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return sys.executable


def error(msg):
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Models creator", msg)
        root.destroy()
    except Exception:
        print(msg, file=sys.stderr)


def main():
    os.chdir(ROOT)
    if "--console" in sys.argv:
        sys.exit(subprocess.call([console_python()] + SERVER))
    if answers():
        webbrowser.open(URL)
        return
    log_path, log = open_log()
    if os.name == "nt":
        opts = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        opts = {"start_new_session": True}  # survives the closing of the terminal or launcher
    proc = subprocess.Popen([console_python()] + SERVER, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, **opts)
    for _ in range(240):
        time.sleep(0.5)
        if answers():
            webbrowser.open(URL)
            return
        if proc.poll() is not None:
            break
    error("The server did not start.\n\nIts messages are in:\n" + str(log_path))
    sys.exit(1)


def open_log():
    """logs/server.log, started afresh. A server still starting (or stuck)
    keeps it locked on Windows: then a second file is used."""
    (ROOT / "logs").mkdir(exist_ok=True)
    for name in ("server.log", "server_2.log", "server_3.log"):
        path = ROOT / "logs" / name
        try:
            return path, open(path, "w", encoding="utf-8")
        except OSError:
            continue
    path = ROOT / "logs" / f"server_{os.getpid()}.log"
    return path, open(path, "w", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pythonw shows nothing otherwise
        error(f"Models creator could not start:\n\n{type(e).__name__}: {e}")
        sys.exit(1)
