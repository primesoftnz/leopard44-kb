@echo off
:: Leopard 44 KB Windows run launcher (start.bat)
:: Launches the server. Does NOT re-run setup — use setup.bat for that (D-05).
cd /d "%~dp0"
:: Prepend uv's default install location so the launcher finds uv after setup installed it.
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run l44 serve
