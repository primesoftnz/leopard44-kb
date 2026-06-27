#!/usr/bin/env bash
# Leopard 44 KB setup — Linux / macOS
# Guided, idempotent, re-runnable installer (D-04).
# All non-trivial logic lives in scripts/setup_core.py (unit-testable with mocks).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- UV BOOTSTRAP (Codex HIGH 13-04 / PKG-07) ---
# A fresh clone may have NO prerequisites — bootstrap uv before the installer runs.
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing via the official astral.sh installer..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        export PATH="$HOME/.local/bin:$PATH"  # uv's default install location
    fi
fi

# Second check: if uv is still absent (curl unavailable or install failed),
# fall back to the Python bootstrapper which installs uv itself before syncing.
if ! command -v uv >/dev/null 2>&1; then
    echo "uv still not available — falling back to python3 bootstrapper..."
    exec python3 scripts/setup_core.py "$@"
fi

exec uv run python scripts/setup_core.py "$@"
