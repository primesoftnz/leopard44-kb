@echo off
:: Leopard 44 KB setup launcher — Windows
:: Guided, idempotent, re-runnable installer (D-04).
:: All non-trivial logic lives in scripts\setup_core.py (unit-testable with mocks).
::
:: NOTE: this script deliberately uses flat `if errorlevel N` / `goto` control flow
:: rather than parenthesised `if (...)` blocks. Inside a parenthesised block CMD
:: expands %ERRORLEVEL% once at parse time, so a `choice`/`where` result set inside
:: the block is never actually read — that silently bypassed the install-consent
:: prompt. `if errorlevel N` is evaluated at runtime and is the correct idiom here.
setlocal

:: cd to the directory containing this .bat file (%~dp0 = drive+path of batch file)
cd /d "%~dp0"

:: Check if Ollama is installed (Windows PATH check). errorlevel 0 = found.
where ollama >nul 2>&1
if not errorlevel 1 goto :check_uv

echo Ollama not found on this system.
choice /C YN /M "Install Ollama automatically (Y/N)?"
:: choice sets ERRORLEVEL: Y=1, N=2. "if errorlevel 2" is true only when the user chose N.
if errorlevel 2 goto :check_uv

:: Primary: winget (Windows 10 1809+ with App Installer). errorlevel 1 = not found.
where winget >nul 2>&1
if errorlevel 1 goto :winget_missing
echo Installing Ollama via winget...
winget install -e --id Ollama.Ollama --accept-source-agreements
goto :check_uv

:winget_missing
:: Degrade: manual download message (OllamaSetup.exe /SILENT is unofficial — prefer manual)
echo winget not available. Please download and install Ollama manually:
echo   https://ollama.com/download/windows
echo Then re-run this script.
pause
exit /b 1

:check_uv
:: --- UV BOOTSTRAP (Codex HIGH 13-04 / PKG-07) ---
:: Bootstrap uv BEFORE the delegate call — a fresh clone has no prerequisites (PKG-07).
where uv >nul 2>&1
if not errorlevel 1 goto :uv_found

echo uv not found -- installing via the official PowerShell installer...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:: Second check after install attempt
where uv >nul 2>&1
if not errorlevel 1 goto :uv_found

:: WR-02: setup_core.py imports third-party packages (httpx, leopard44_kb) that only
:: exist AFTER `uv sync`, so it cannot bootstrap a bare clone. Emit an actionable
:: remedy and exit non-zero instead of dying on a confusing ModuleNotFoundError.
echo ERROR: uv is required but could not be installed automatically. 1>&2
echo Install uv manually (see https://docs.astral.sh/uv/getting-started/installation/), then re-run setup.bat. 1>&2
exit /b 1

:uv_found
:: Delegate to the Python installer core (all platform-independent logic lives there)
uv run python scripts\setup_core.py --skip-install %*
