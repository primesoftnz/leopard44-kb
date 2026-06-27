"""Leopard 44 KB installer core — detect hardware, tier, Ollama, pull models, migrate seed, smoke-test.

This module holds all non-trivial setup logic so it is unit-testable with mocks and not
triplicated across shell syntaxes. It is NOT a l44 CLI subcommand (D-15); the three
OS launchers (setup.sh, setup.bat, start.command) invoke it via `uv run python`.

Requirements: INSTALL-01, INSTALL-02, INSTALL-04, INSTALL-05, INSTALL-06
Security: T-06-01 (V5 allowlist), T-06-04 (dirty-tree guard), T-06-07 (consent-gated install)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from subprocess import DEVNULL

import httpx

# ---------------------------------------------------------------------------
# Re-export from leopard44_kb.config so tests can monkeypatch setup_core.write_config
# ---------------------------------------------------------------------------
from leopard44_kb.config import TIER_MODELS, write_config  # noqa: F401
from leopard44_kb.ingest.embedder import detect_ram_gb  # noqa: F401

# ---------------------------------------------------------------------------
# Repo root: derived from this file's location, never os.getcwd() (Pitfall 5)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPU_VRAM_THRESHOLD_MIB: int = 9000  # conservative — qwen2.5:14b is ~9 GB

# The 10 reference markdown files to migrate into shared/leopard44/
FILES_TO_MIGRATE = [
    "bluewater-capability.md",
    "competitive-comparison.md",
    "ex-charter-buying-guide.md",
    "interior-layout.md",
    "known-issues.md",
    "market-value.md",
    "onboard-systems.md",
    "sailing-performance.md",
    "technical-specifications.md",
    "upgrades-modifications.md",
]

ALLOWED_TIERS = {"8gb", "16gb", "gpu"}  # T-06-01 allowlist (V5 input validation)


# ---------------------------------------------------------------------------
# INSTALL-01: GPU detection
# ---------------------------------------------------------------------------


def detect_gpu_vram_mib() -> int | None:
    """Return total VRAM in MiB across all GPUs, or None when nvidia-smi absent/errors.

    Uses nvidia-smi --query-gpu=memory.total which returns one line per GPU.
    Returns the MAX across all GPUs so a multi-GPU system gets the best tier.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            text=True,
            stderr=DEVNULL,
            timeout=5,
        )
        # Output: "8192 MiB\n16376 MiB\n" — one line per GPU; take max
        mib_values = []
        for line in out.strip().splitlines():
            line = line.strip()
            if line:
                mib_values.append(int(line.replace(" MiB", "").strip()))
        if not mib_values:
            return None
        return max(mib_values)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# INSTALL-01: Tier recommendation + resolution
# ---------------------------------------------------------------------------


def recommend_tier() -> str:
    """Return the recommended tier string based on hardware.

    Priority: GPU VRAM (>=9000 MiB) > RAM (>=14 GB) > default (8gb).
    """
    vram = detect_gpu_vram_mib()
    if vram is not None and vram >= GPU_VRAM_THRESHOLD_MIB:
        return "gpu"
    if detect_ram_gb() >= 14:
        return "16gb"
    return "8gb"


def resolve_tier(cli_tier: str | None) -> str:
    """Return a validated tier string.

    Allowlist enforcement (T-06-01): {8gb, 16gb, gpu} case-sensitive.
    None -> recommend_tier().
    Invalid value -> print remedy to stderr and sys.exit(2).
    The raw value is NEVER forwarded to write_config or any shell command.
    """
    if cli_tier is None:
        return recommend_tier()
    if cli_tier in ALLOWED_TIERS:
        return cli_tier
    # Reject — bad value never reaches downstream (T-06-01)
    print(
        f"Error: --tier '{cli_tier}' is not valid. "
        f"Allowed values: 8gb, 16gb, gpu (case-sensitive).",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# INSTALL-02: Ollama daemon check (OLLAMA_HOST-aware)
# ---------------------------------------------------------------------------


def is_ollama_daemon_running() -> bool:
    """Return True iff the Ollama daemon is reachable and returns HTTP 200.

    Honours OLLAMA_HOST env var (read at call time, not import time), so tests
    can override with monkeypatch.setenv (Codex MEDIUM finding).
    """
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host}/api/version"
    try:
        r = httpx.get(url, timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# INSTALL-01/INSTALL-02: Idempotent model pull guard
# ---------------------------------------------------------------------------


def ensure_model_pulled(tag: str) -> None:
    """Pull the given Ollama model tag if not already present.

    Uses exact full-tag matching against `ollama list` output — no prefix/partial
    matching (Pattern 4). On pull failure, prints remedy to stderr and raises
    subprocess.CalledProcessError.
    """
    # Check if exact tag is already present in ollama list
    try:
        listing = subprocess.check_output(
            ["ollama", "list"],
            text=True,
            stderr=DEVNULL,
            timeout=30,
        )
    except Exception:
        # Includes subprocess.TimeoutExpired: a daemon that passes the HTTP health
        # check but hangs on the `list` RPC must not hang the installer forever.
        listing = ""

    # Exact match on the NAME column: each line starts with "tag  ID  ..."
    # We check if the exact full tag appears at the start of any non-header line.
    lines = listing.strip().splitlines()
    for line in lines:
        # Split on whitespace; column 0 is the full tag
        parts = line.split()
        if parts and parts[0] == tag:
            return  # already present, skip pull

    # Not present — pull it
    try:
        subprocess.check_call(["ollama", "pull", tag])
    except subprocess.CalledProcessError:
        print(
            f"Failed to pull {tag}. Check your internet connection and retry: "
            f"ollama pull {tag}",
            file=sys.stderr,
        )
        raise


# ---------------------------------------------------------------------------
# T-06-09 / D-04: Embedding-model mismatch guard
# ---------------------------------------------------------------------------


def assert_embedding_compatible(tier: str) -> None:
    """Raise if the existing store's embedding model mismatches the tier's embedding.

    A clean install (empty store, most_common_embedding_model returns None) is
    always compatible. A mismatch on an existing store means silently changing the
    embedding model would cause 384-dim violations (D-04 enforcement).
    """
    from leopard44_kb.retrieve import most_common_embedding_model
    from leopard44_kb.store import open_db

    try:
        conn = open_db()
        corpus_model = most_common_embedding_model(conn)
        conn.close()
    except Exception:
        # Cannot read store — treat as clean install (no violation)
        return

    if corpus_model is None:
        return  # empty store, fresh install — always OK

    expected_emb = TIER_MODELS[tier][1]  # (gen_model, emb_model, emb_ver)[1]
    if corpus_model != expected_emb:
        raise ValueError(
            f"Embedding model mismatch (D-04): the existing store uses "
            f"'{corpus_model}' but tier '{tier}' requires '{expected_emb}'. "
            f"Re-running setup with a different tier would silently corrupt the "
            f"vector store. To switch tiers, first wipe the store: "
            f"rm ~/.local/share/leopard44-kb/store.db and re-run setup."
        )


# ---------------------------------------------------------------------------
# INSTALL-04: Seed layout validation (pure filesystem check — NO git)
# ---------------------------------------------------------------------------


def validate_seed_layout(repo_root: Path) -> None:
    """Verify all 10 reference .md files are present in shared/leopard44/.

    Raises SystemExit (with the missing filename in the message) when any file
    is absent. Does NOT perform any git operations.

    Returns None on success.
    """
    leopard44 = repo_root / "shared" / "leopard44"
    for fname in FILES_TO_MIGRATE:
        dest = leopard44 / fname
        if not dest.exists():
            raise SystemExit(
                f"Seed layout error: missing {fname} in shared/leopard44/. "
                f"Run the seed migration or re-clone the repository."
            )


# ---------------------------------------------------------------------------
# INSTALL-04: Seed file migration (git mv — maintainer action, done once)
# ---------------------------------------------------------------------------


def migrate_seed_files(repo_root: Path) -> bool:
    """git mv the 10 reference files into shared/leopard44/. Idempotent.

    Checks git status --porcelain for a dirty tree and aborts with a clear message.
    Returns True if any files were moved (migration ran); False if already done.
    Commits the move in a single named commit (D-11, Open Q2).
    """
    # Check for dirty working tree (T-06-04 guard)
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            cwd=str(repo_root),
        ).strip()
    except subprocess.CalledProcessError:
        status = ""

    if status:
        print(
            "Seed migration aborted: working tree has uncommitted changes. "
            "Commit or stash changes before running migrate_seed_files().",
            file=sys.stderr,
        )
        sys.exit(1)

    dest_dir = repo_root / "shared" / "leopard44"
    moved: list[str] = []

    for fname in FILES_TO_MIGRATE:
        src = repo_root / fname
        dst = dest_dir / fname
        if dst.exists():
            continue  # already migrated — idempotent (D-08)
        if not src.exists():
            print(f"  WARNING: {fname} not found at repo root — skipping", file=sys.stderr)
            continue
        subprocess.check_call(
            ["git", "mv", str(src), str(dst)],
            cwd=str(repo_root),
        )
        moved.append(fname)

    if moved:
        # Remove .gitkeep now that the directory has real content (only leopard44)
        gitkeep = dest_dir / ".gitkeep"
        if gitkeep.exists():
            subprocess.check_call(
                ["git", "rm", str(gitkeep)],
                cwd=str(repo_root),
            )
        subprocess.check_call(
            ["git", "commit", "-m", "chore: migrate 10 L44 reference docs to shared/leopard44/"],
            cwd=str(repo_root),
        )

    return bool(moved)


# ---------------------------------------------------------------------------
# INSTALL-06: Seed ingest helper (factored out so main() tests can monkeypatch)
# ---------------------------------------------------------------------------


def _run_seed_ingest(repo_root: Path) -> None:
    """Run `l44 ingest shared/leopard44 --layer shared` via subprocess.check_call.

    Uses check_call so it blocks and raises CalledProcessError on failure (Pitfall 3).
    """
    subprocess.check_call(
        ["uv", "run", "l44", "ingest", "shared/leopard44", "--layer", "shared"],
        cwd=str(repo_root),
    )


# ---------------------------------------------------------------------------
# INSTALL-06: Smoke-test
# ---------------------------------------------------------------------------


def run_smoke_test(question: str) -> bool:
    """Return True iff l44 ask returns a non-empty answer with a shared: citation.

    Pass criterion (D-12, tightened per Codex HIGH fix):
      - returncode == 0
      - stdout is non-empty
      - stdout contains a '^[N] shared:' line within a Sources block

    Returns False for:
      - REFUSAL_MESSAGE output (no Sources block)
      - vessel-only citations
      - empty stdout
      - no Sources block
    """
    from leopard44_kb.answer import REFUSAL_MESSAGE

    # WR-05: bound the smoke query so a stalled Ollama generation cannot wedge the
    # installer indefinitely (every other stallable subprocess here is bounded too).
    try:
        result = subprocess.run(
            ["uv", "run", "l44", "ask", question],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False

    output = result.stdout
    if result.returncode != 0:
        return False
    if not output.strip():
        return False
    if REFUSAL_MESSAGE in output:
        return False

    # Require a Sources block AND a '[N] shared:' citation line within it
    # render_citation_block emits "\n---\nSources:\n[1] shared: <title>"
    if "Sources:" not in output:
        return False

    # Check for the [N] shared: pattern after the Sources: marker
    sources_section = output.split("Sources:")[-1]
    if re.search(r"^\s*\[\d+\]\s+shared:", sources_section, re.MULTILINE):
        return True

    return False


# ---------------------------------------------------------------------------
# main() orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Full install flow. Returns 0 on success, non-zero on failure.

    Flow (D-05/06/08/09/10/11/12):
    1. resolve tier (allowlist gate — argparse choices + resolve_tier)
    2. Ollama in PATH? if not, consent-gated install; if no consent, exit non-zero
    3. Daemon running? if not, hard-fail with remedy (D-06)
    4. assert_embedding_compatible (D-04)
    5. pull both models via ensure_model_pulled (idempotent)
    6. write_config (AFTER pulls succeed — atomic, D-06)
    7. uv sync --extra dev then pre-commit install (D-09, Open Q3 ordering)
    8. validate_seed_layout (pure fs check — no git)
    9. _run_seed_ingest (seed the shared layer)
    10. run_smoke_test (D-12 — failure is a hard install failure)
    """
    parser = argparse.ArgumentParser(
        prog="setup_core",
        description="Leopard 44 KB first-run installer",
    )
    parser.add_argument(
        "--tier",
        choices=list(ALLOWED_TIERS),
        default=None,
        help="Override the recommended model tier (8gb, 16gb, gpu)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip Ollama auto-install prompt (for testing / already-installed environments)",
    )
    args = parser.parse_args(argv)

    # ---- Step 1: resolve tier (allowlist gate) ----
    tier = resolve_tier(args.tier)

    # ---- Step 2: Ollama in PATH? ----
    import shutil

    ollama_present = shutil.which("ollama") is not None

    if not ollama_present:
        if args.skip_install:
            print(
                "Ollama not found. Install it first: https://ollama.com/download",
                file=sys.stderr,
            )
            return 1

        # D-05: consent-gated install
        answer = input("Ollama not found. Auto-install Ollama? [y/N] ").strip().lower()
        if answer != "y":
            print(
                "Ollama not found. Install it first: https://ollama.com/download",
                file=sys.stderr,
            )
            return 1

        # HTTPS-only, official, consent-gated (T-06-07)
        try:
            subprocess.check_call(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
            )
        except subprocess.CalledProcessError:
            print("Ollama auto-install failed. Install manually: https://ollama.com/download", file=sys.stderr)
            return 1

        # Start daemon after install if not already running
        if not is_ollama_daemon_running():
            subprocess.Popen(["ollama", "serve"], stdout=DEVNULL, stderr=DEVNULL)
            import time
            for _ in range(10):
                time.sleep(1)
                if is_ollama_daemon_running():
                    break

    # ---- Step 3: Daemon running? ----
    if not is_ollama_daemon_running():
        print(
            "Ollama is installed but not running. Start it with: ollama serve",
            file=sys.stderr,
        )
        return 1

    # ---- Step 4: D-04 embedding mismatch guard ----
    try:
        assert_embedding_compatible(tier)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # ---- Step 5: pull both models (idempotent) ----
    gen_model, emb_model, _emb_ver = TIER_MODELS[tier]
    try:
        ensure_model_pulled(gen_model)
        ensure_model_pulled(emb_model)
    except subprocess.CalledProcessError:
        return 1

    # ---- Step 6: write_config AFTER pulls succeed (atomic, D-06) ----
    write_config(tier)

    # ---- Step 7: uv sync + pre-commit install (D-09, Open Q3) ----
    try:
        subprocess.check_call(["uv", "sync", "--extra", "dev"])
        subprocess.check_call(["uv", "run", "pre-commit", "install"])
    except subprocess.CalledProcessError as exc:
        print(f"Setup step failed: {exc}. Check that uv and pre-commit are available.", file=sys.stderr)
        return 1

    # ---- Step 7b: voice setup — isolated .venv-stt + whisper weights (PKG-08, non-fatal) ----
    uv_path = shutil.which("uv")
    if uv_path:
        try:
            subprocess.check_call([uv_path, "run", "l44", "voice", "setup"])
        except subprocess.CalledProcessError:
            print(
                "Warning: voice setup failed — offline voice will not be available until resolved.\n"
                "  Re-run manually: uv run l44 voice setup",
                file=sys.stderr,
            )
    else:
        print(
            "Warning: uv not found on PATH after sync — skipping voice setup.\n"
            "  Re-run setup after uv is available, or manually: uv run l44 voice setup",
            file=sys.stderr,
        )

    # ---- Step 7c: ffmpeg advisory check (T-13-41, non-fatal) ----
    # faster-whisper uses PyAV's bundled FFmpeg so system ffmpeg is optional.
    if not shutil.which("ffmpeg"):
        print(
            "\nNote: system ffmpeg not found. Voice transcription works without it "
            "(faster-whisper uses PyAV's bundled FFmpeg — system ffmpeg is optional).\n"
            "For broader audio format support, install it:\n"
            "  Linux:   sudo apt install ffmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg",
        )

    # ---- Step 7d: schematic-render guidance ----
    print(
        "\nSchematic render: if you have the factory Owner's Manual PDF, generate "
        "schematic PNGs with:\n"
        "  l44 schematic render <your-manual.pdf> --pages 61-89",
    )

    # ---- Step 8: validate seed layout (pure fs check) ----
    repo_root = _REPO_ROOT
    try:
        validate_seed_layout(repo_root)
    except SystemExit as exc:
        if exc.code:
            print(str(exc.code), file=sys.stderr)
        return 1

    # ---- Step 9: seed ingest ----
    try:
        _run_seed_ingest(repo_root)
    except subprocess.CalledProcessError:
        print(
            "Seed ingest failed. Check that the shared/leopard44/ files are present "
            "and retry: l44 ingest shared/leopard44 --layer shared",
            file=sys.stderr,
        )
        return 1

    # ---- Step 10: smoke-test (D-12) ----
    smoke_q = "What are the known issues with the Leopard 44?"
    if not run_smoke_test(smoke_q):
        print(
            "Smoke-test failed: no shared-layer source cited. "
            "Re-run setup or manually run: l44 ingest shared/leopard44 --layer shared",
            file=sys.stderr,
        )
        return 1

    print(
        "\nSetup complete! Leopard 44 KB is ready to use.\n"
        "  Tier:       " + tier + "\n"
        "  Generation: " + gen_model + "\n"
        "  Embedding:  " + emb_model + "\n"
        "\nShared layer seeded and smoke-test passed (shared: citation confirmed).\n"
        "Now run ./start (or ./start.command on macOS / start.bat on Windows) to launch the server."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
