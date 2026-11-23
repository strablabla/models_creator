:; # --- install.cmd --- installs Models creator. ONE file for Windows and Linux:
:; #   Windows: double-click it.
:; #   Linux:   sh install.cmd      (from this folder, or with its path)
:; # It installs uv if needed, which then installs Python and the libraries in
:; # .venv (a few GB, mostly PyTorch and TensorFlow), then puts a "Models creator"
:; # icon on the Desktop and in the applications menu.
:; #
:; # How one file serves both: the shell reads ":;" as "do nothing, then run the
:; # rest of the line", and stops at "exit" below; cmd reads ":" lines as labels,
:; # skips them, and runs the part after "@echo off". Keep LF line endings
:; # (.gitattributes) and no goto in the Windows part.
:; set -e
:; cd "$(dirname "$0")"
:; UV="$(command -v uv || true)"
:; [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
:; if [ -z "$UV" ]; then
:;     echo "Installing uv (https://docs.astral.sh/uv/)..."
:;     if command -v curl >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh
:;     else wget -qO- https://astral.sh/uv/install.sh | sh; fi
:;     UV="$HOME/.local/bin/uv"
:; fi
:; echo "Installing Python and the libraries (the first time: several minutes)..."
:; "$UV" sync
:; exec "$UV" run python -m app.installer "$@"
:; exit $?
@echo off
cd /d "%~dp0"
set "UV=uv"
where uv >nul 2>nul || set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not "%UV%"=="uv" if not exist "%UV%" (
    echo Installing uv ^(https://docs.astral.sh/uv/^)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
if not "%UV%"=="uv" if not exist "%UV%" (
    echo.
    echo uv could not be installed: see the messages above.
    pause
    exit /b 1
)
echo Installing Python and the libraries ^(the first time: several minutes^)...
"%UV%" sync
if errorlevel 1 (
    echo.
    echo The installation failed: see the messages above.
    pause
    exit /b 1
)
"%UV%" run python -m app.installer %*
pause
