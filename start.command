#!/usr/bin/env bash
# Leopard 44 KB macOS / Linux run launcher (start.command)
# Launches the server. Does NOT re-run the installer — run ./setup first (D-05).
# macOS Terminal opens .command files with CWD set to $HOME — SCRIPT_DIR guard is mandatory.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Prepend uv's default install location so a Finder double-click (no shell profile) finds uv.
export PATH="$HOME/.local/bin:$PATH"
exec uv run l44 serve
