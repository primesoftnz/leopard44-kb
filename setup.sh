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

# Second check: if uv is STILL absent (no curl, or the install failed) we cannot
# proceed. scripts/setup_core.py imports third-party packages (httpx, leopard44_kb)
# that only exist AFTER `uv sync`, so it cannot bootstrap a bare clone — running it
# now would die with a confusing ModuleNotFoundError (WR-02). Emit an actionable
# remedy and exit non-zero instead.
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required but could not be installed automatically." >&2
    echo "Install uv manually, then re-run ./setup.sh:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pipx install uv / brew install uv" >&2
    echo "  docs: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

exec uv run python scripts/setup_core.py "$@"
