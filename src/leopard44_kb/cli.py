"""Typer CLI entry point for Leopard 44 KB. Five subcommands locked per D-15: sources (wired), ingest|ask|add|serve (stubs raising in their target phase). main() is bound to [project.scripts] in pyproject.toml."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from typing import List, Optional

import typer

from leopard44_kb import LAYERS
from leopard44_kb.ingest import SUPPORTED_SUFFIXES, ingest_file
from leopard44_kb.sources import list_sources_for_layer

app = typer.Typer(
    add_completion=False,
    help="Leopard 44 KB — vessel knowledge base (offline, two-layer).",
)


def _stdin_isatty() -> bool:
    """Return True if stdin is connected to a real terminal (TTY).

    Defined as a named module-level helper so tests can monkeypatch
    ``leopard44_kb.cli._stdin_isatty`` to simulate a TTY under CliRunner
    (whose stdin is always non-TTY). Production code always calls this
    function — never ``sys.stdin.isatty()`` inline — so the monkeypatch
    target is unambiguous (Codex HIGH #2 / D-10 STRICT gate).
    """
    return sys.stdin.isatty()


@app.command("sources")
def sources_cmd(
    layer: str = typer.Option(..., "--layer", help="shared | vessel | community"),
) -> None:
    """List known sources scoped to a layer."""
    if layer not in LAYERS:
        raise typer.BadParameter(
            f"layer must be one of {list(LAYERS)}, got {layer!r}",
            param_hint="--layer",
        )
    for row in list_sources_for_layer(layer):
        typer.echo(f"{row['id']:>4}  {row['source_type']:<16}  {row['path']}")


def _not_yet(phase: str) -> typer.Exit:
    typer.secho(f"Not yet implemented — see Phase {phase}.", fg=typer.colors.YELLOW, err=True)
    return typer.Exit(code=2)


def _find_free_port(default: int = 8000) -> int:
    """Return the first free TCP port starting from *default* (inclusive).

    Probes ``range(default, default + 20)`` by attempting ``socket.bind()``
    on ``127.0.0.1``.  The binding socket is closed immediately so the port
    is available for uvicorn to claim.  Raises ``RuntimeError`` if all 20
    candidates are busy.
    """
    for port in range(default, default + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free port found in range {default}–{default + 19}"
    )


def _find_free_port_lan(default: int = 8443) -> int:
    """Like _find_free_port but probes 0.0.0.0 for LAN-bind availability.

    Used exclusively by the ``serve --qr`` LAN branch.  Do NOT change
    ``_find_free_port`` (the 127.0.0.1 prober) — it is a security gate.
    Own-use Linux/WSL only — Windows Scripts/ venv-bin resolution is a
    Phase-13 (public release) concern.
    """
    for port in range(default, default + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free LAN port found in range {default}–{default + 19}"
    )


def _detect_lan_ip() -> str:
    """Return the notebook's outbound LAN IP via the UDP routing trick.

    No packet is sent.  Falls back to hostname resolution.
    WSL2 caveat: returns the WSL internal IP (e.g. 172.18.x.x), NOT the
    Windows host IP.  Use ``--host-ip`` to override for WSL / multi-NIC.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.254.254.254", 1))  # non-routable; no packet sent
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def _cert_dir() -> Path:
    """Return the directory where the persisted self-signed cert lives.

    Default: ``<repo_root>/data/certs/`` (created if missing).
    Override: ``L44_CERTS_DIR`` env var (used in tests for isolation).
    ``data/`` is gitignored so the cert never enters version control.
    """
    import os as _os
    from leopard44_kb.paths import repo_root

    override = _os.environ.get("L44_CERTS_DIR")
    if override:
        p = Path(override)
    else:
        p = repo_root() / "data" / "certs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_or_create_cert(target_ip: str) -> tuple[Path, Path]:
    """Persist and reuse a self-signed TLS cert with an IP SAN for *target_ip*.

    D-12 one-time trust: if ``data/certs/l44-lan.{crt,key}`` already exist
    AND the existing cert's IP SAN contains *target_ip*, return the existing
    paths WITHOUT rewriting (so the phone trusts this cert exactly once).
    Regenerates only when the files are missing OR the SAN no longer matches
    (i.e. the LAN IP changed).

    The private key is written owner-only (mode 0600 — review residual #6)
    so it is not group/world-readable (ASVS V6).

    RSA 2048 / SHA-256 / 397-day lifetime / IP SAN + serverAuth EKU.
    iOS Safari REJECTS the connection (not just a warning) if the validity span
    exceeds 825 days or the serverAuth EKU is missing, so the cert is built to
    satisfy Apple's TLS-server-cert policy — verified against a real iPhone
    (field UAT 2026-06-16; a 10-year/no-EKU cert errored after "Visit this
    Website"). IP SAN (not CN-only) is required for Chrome 58+ / modern Safari
    to accept the cert at an IP address.

    Own-use Linux/WSL only.  Windows Scripts/ venv-bin is a Phase-13 concern.
    """
    import ipaddress as _ipaddress
    import os as _os

    certs = _cert_dir()
    cert_path = certs / "l44-lan.crt"
    key_path = certs / "l44-lan.key"

    # Reuse existing cert ONLY if it still matches the IP SAN AND is iOS-compatible
    # (validity span <= 825 days and has the serverAuth EKU). An older non-compliant
    # cert (e.g. a pre-fix 10-year/no-EKU cert) falls through and is regenerated so
    # the phone gets a cert it will actually accept (D-12 persistence + auto-heal).
    if cert_path.exists() and key_path.exists():
        try:
            from cryptography.x509 import (  # noqa: PLC0415
                load_pem_x509_certificate,
                SubjectAlternativeName,
                ExtendedKeyUsage,
                IPAddress as X509IPAddress,
            )
            from cryptography.x509.oid import ExtendedKeyUsageOID  # noqa: PLC0415
            existing = load_pem_x509_certificate(cert_path.read_bytes())
            san_ext = existing.extensions.get_extension_for_class(SubjectAlternativeName)
            san_ips = [str(v.value) for v in san_ext.value if isinstance(v, X509IPAddress)]
            span_days = (existing.not_valid_after - existing.not_valid_before).days
            try:
                eku = existing.extensions.get_extension_for_class(ExtendedKeyUsage).value
                has_server_auth = ExtendedKeyUsageOID.SERVER_AUTH in eku
            except Exception:
                has_server_auth = False
            if target_ip in san_ips and span_days <= 825 and has_server_auth:
                return cert_path, key_path
        except Exception:
            pass  # Fall through to regenerate if anything goes wrong

    # Generate new cert + key
    from cryptography import x509  # noqa: PLC0415
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID  # noqa: PLC0415
    from cryptography.hazmat.primitives import hashes, serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: PLC0415
    import datetime as _datetime

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "l44-local"),
    ])
    _now = _datetime.datetime.now(_datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        # 397 days: under Apple's 825-day hard limit AND the 398-day CA-cert
        # convention, so iOS accepts the connection (field UAT 2026-06-16).
        .not_valid_before(_now)
        .not_valid_after(_now + _datetime.timedelta(days=397))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(_ipaddress.IPv4Address(target_ip))]
            ),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        # iOS requires the serverAuth EKU on TLS server certs.
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    # Write key owner-only (0600) so it is never group/world-readable (review residual #6)
    key_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    fd = _os.open(str(key_path), _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC, 0o600)
    try:
        _os.write(fd, key_bytes)
    finally:
        _os.close(fd)

    return cert_path, key_path


@app.command("ingest")
def ingest_cmd(
    paths: List[str] = typer.Argument(..., help="One or more files or directories to ingest"),
    layer: str = typer.Option("vessel", "--layer", help="shared | vessel | community (default vessel)"),
    ocr: bool = typer.Option(False, "--ocr", help="Run OCR on image-only PDF pages (requires tesseract; ~1000x slower)"),
) -> None:
    """Ingest one or more files or directories into the knowledge base (Phase 2)."""
    if layer not in LAYERS:
        raise typer.BadParameter(
            f"layer must be one of {list(LAYERS)}, got {layer!r}",
            param_hint="--layer",
        )

    # Expand paths: directories → walk and collect supported files; files → include directly.
    # Directory expansion filters to SUPPORTED_SUFFIXES; unsupported files go to skipped.
    # An explicitly named file with an unsupported suffix is passed through (ingest_file raises).
    collected: list[str] = []
    skipped: list[str] = []

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    if child.suffix.lower() in SUPPORTED_SUFFIXES:
                        collected.append(str(child))
                    else:
                        skipped.append(str(child))
        else:
            collected.append(raw)

    if skipped:
        skipped_names = ", ".join(Path(s).name for s in skipped[:5])
        suffix = f" and {len(skipped) - 5} more" if len(skipped) > 5 else ""
        typer.secho(
            f"  skipped {len(skipped)} unsupported file(s): {skipped_names}{suffix}",
            fg=typer.colors.YELLOW,
            err=True,
        )

    # Per-file batch loop: log each result; catch failures and continue (INGEST-07, Pitfall 4).
    # Exit code is 1 if any file failed (T-02-16 — repudiation mitigation).
    failed: list[str] = []
    for p_str in collected:
        try:
            result = ingest_file(p_str, layer=layer, ocr=ocr)
            typer.echo(f"  {result.upper():<5} {p_str}")
        except Exception as exc:
            typer.secho(f"  FAIL  {p_str}: {exc}", fg=typer.colors.RED, err=True)
            failed.append(p_str)

    raise typer.Exit(code=1 if failed else 0)


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Natural-language question"),
    layer: str = typer.Option("all", "--layer", help="shared | vessel | community | all (default all)"),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve (default 5; raising it may exceed the <10s budget)"),
) -> None:
    """Ask a natural-language question grounded in the vessel knowledge base.

    Retrieves the top-k most relevant chunks from the knowledge base, streams
    the LLM-generated answer live, then prints a code-rendered citation block.
    Use --layer to scope retrieval to a single layer (default: all layers).
    Use --top-k to adjust how many chunks are retrieved (larger values improve
    recall but may exceed the 10s latency budget on CPU-only hardware).
    """
    # Lazy imports inside the body (mirrors ingest_cmd pattern for Phase-N modules).
    from leopard44_kb.retrieve import retrieve
    from leopard44_kb.answer import (
        select_generation_model,
        select_num_predict,
        build_user_message,
        SYSTEM_PROMPT,
        REFUSAL_MESSAGE,
        stream_generate,
        validate_citations,
        render_citation_block,
    )
    from leopard44_kb.store import open_db

    # Layer resolution: 'all' is a special value not in LAYERS but explicitly allowed.
    # Any value in LAYERS is valid; anything else raises BadParameter (T-03-10).
    if layer == "all":
        layers: list[str] = []  # empty list = no filter = retrieve from all layers
    elif layer in LAYERS:
        layers = [layer]
    else:
        raise typer.BadParameter(
            f"layer must be shared|vessel|community|all, got {layer!r}",
            param_hint="--layer",
        )

    # WR-05: --top-k is an unbounded int; reject non-positive values before they
    # reach apply_d05_fts_slot's `n - 1` slice (which would corrupt on n <= 0).
    if top_k < 1:
        raise typer.BadParameter(
            f"--top-k must be >= 1, got {top_k}",
            param_hint="--top-k",
        )

    # WR-02: open_db() sets PRAGMA journal_mode=WAL; the connection (and its
    # -wal/-shm sidecars) must be closed on every path — success, refusal, and
    # generation error. Wrap the whole body in try/finally.
    conn = open_db()
    try:
        chunks, below_floor = retrieve(conn, question, layers, n=top_k)

        # D-07: below-floor / empty KB → refusal, exit 0 (no LLM call).
        if below_floor:
            typer.echo(REFUSAL_MESSAGE)
            raise typer.Exit(code=0)

        gen_model, tier_label = select_generation_model()
        system = SYSTEM_PROMPT.format(n_chunks=len(chunks))
        user_msg = build_user_message(question, chunks)

        # The token cap is a per-tier ceiling (select_num_predict), not a function
        # of --top-k. WR-03 still holds: it must not scale with k (the old
        # min(cap, top_k * 15) drove --top-k 1 down to 15 tokens, truncating
        # mid-answer). L44_NUM_PREDICT overrides the tier cap on demand.
        num_predict = select_num_predict(tier_label, gen_model)

        # D-09: stream tokens live first, accumulate for post-stream validation.
        full_answer_parts: list[str] = []
        try:
            for token in stream_generate(gen_model, system, user_msg, num_predict=num_predict):
                typer.echo(token, nl=False)
                full_answer_parts.append(token)
        except RuntimeError as exc:
            typer.secho(f"\nError: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        typer.echo("")  # newline after streamed tokens

        # review fix #4: detect out-of-range [n] markers after generation.
        # D-09 streaming means a bogus [n] already reached stdout — do not try to retract.
        # The code-rendered Sources block is the authoritative citation surface; it only
        # lists real retrieved chunks so a hallucinated marker has no matching source entry.
        full_text = "".join(full_answer_parts)
        bad_citations = validate_citations(full_text, len(chunks))
        if bad_citations:
            bad_str = ", ".join(f"[{n}]" for n in sorted(set(bad_citations)))
            typer.secho(
                f"Warning: out-of-range citation marker(s) detected: {bad_str} "
                f"(only [1]–[{len(chunks)}] are valid; these markers have no source entry below)",
                fg=typer.colors.YELLOW,
                err=True,
            )

        # Print code-rendered citation block (D-06 anti-hallucination guarantee).
        typer.echo(render_citation_block(chunks))
    finally:
        conn.close()


def _render_fields_table(extraction: object) -> None:
    """Render extracted maintenance fields as a human-readable table.

    Prints each field name and its current value so the owner can decide
    whether to Accept, Edit, or Abort before committing.
    """
    # Import here to keep the helper self-contained.
    from leopard44_kb.maintenance import MaintenanceExtraction
    e: MaintenanceExtraction = extraction  # type: ignore[assignment]

    typer.echo("\n--- Extracted fields ---")
    typer.echo(f"  date          : {e.date or '(none)'}")
    typer.echo(f"  system        : {e.system}")
    typer.echo(f"  system_detail : {e.system_detail or '(none)'}")
    typer.echo(f"  parts         : {', '.join(e.parts) if e.parts else '(none)'}")
    if e.cost is not None:
        typer.echo(f"  cost          : {e.cost.amount} {e.cost.currency}")
    else:
        typer.echo("  cost          : (none)")
    typer.echo(f"  vendor        : {e.vendor or '(none)'}")
    typer.echo("------------------------")


def _edit_fields(extraction: object) -> object:
    """Walk each field interactively; Enter keeps the current (pre-filled) value.

    Field order: vendor, date, system, system_detail, parts, cost_amount, cost_currency.
    Cost is prompted as two separate fields (amount + currency) per D-09 / Gemini LOW.

    Returns a new MaintenanceExtraction with all edits applied via .model_copy().
    """
    from leopard44_kb.maintenance import CostModel, MaintenanceExtraction
    e: MaintenanceExtraction = extraction  # type: ignore[assignment]

    # Vendor first (matches the test fixture input ordering).
    vendor_val = typer.prompt("vendor", default=e.vendor or "")
    date_val = typer.prompt("date (YYYY-MM-DD)", default=e.date or "")
    system_val = typer.prompt("system", default=e.system)
    detail_val = typer.prompt("system_detail", default=e.system_detail or "")
    parts_raw = typer.prompt(
        "parts (comma-separated)", default=", ".join(e.parts) if e.parts else ""
    )

    # Parse parts: split on comma, strip whitespace, drop blanks.
    parts_list = [p.strip() for p in parts_raw.split(",") if p.strip()] if parts_raw.strip() else []

    # Cost: two separate prompts (Gemini LOW split-cost requirement).
    cost_amount_raw = typer.prompt(
        "cost amount (blank = no cost)",
        default=str(e.cost.amount) if e.cost is not None else "",
    )
    cost_currency_val = typer.prompt(
        "cost currency",
        default=e.cost.currency if e.cost is not None else "NZD",
    )

    # Rebuild cost only when an amount was supplied and is parseable.
    new_cost: Optional[CostModel] = None
    if cost_amount_raw.strip():
        try:
            new_cost = CostModel(amount=float(cost_amount_raw.strip()), currency=cost_currency_val)
        except (ValueError, TypeError):
            typer.secho(
                f"  Warning: could not parse cost amount {cost_amount_raw!r} — keeping original",
                fg=typer.colors.YELLOW,
                err=True,
            )
            new_cost = e.cost

    updates: dict = {
        "vendor": vendor_val or None,
        "date": date_val or None,
        "system": system_val,
        "system_detail": detail_val or None,
        "parts": parts_list,
        "cost": new_cost,
    }
    return e.model_copy(update=updates)


@app.command("add")
def add_cmd(
    entry: str = typer.Argument(..., help="Maintenance log entry in natural language"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive review and commit the extraction as-is"),
) -> None:
    """Add a maintenance log entry (Phase 4).

    Extracts structured fields from natural-language text using the local LLM,
    lets you review and optionally edit the fields, then writes a markdown file
    under data/logs/maint/ and ingests it into the knowledge base.

    Pass --yes to skip the interactive review step. In non-interactive (piped/CI)
    contexts, --yes is required — the command will fail fast otherwise.

    The entry is always committed as vessel-layer. This command has no layer flag
    by design; the vessel scope is enforced mechanically.
    """
    # Lazy imports: keep module-level imports minimal; only pull in heavy modules
    # when the command actually runs (mirrors ask_cmd and ingest_cmd patterns).
    import leopard44_kb.maintenance as _maint
    from leopard44_kb.maintenance import write_entry, CostModel
    from leopard44_kb.store import open_db

    # ------------------------------------------------------------------
    # Step 1: EMPTY-ENTRY GUARD (Codex MED) — before TTY check / LLM call / write.
    # Reject blank or whitespace-only entries immediately.
    # ------------------------------------------------------------------
    if not entry or not entry.strip():
        typer.secho(
            "empty maintenance entry — provide some text",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Step 2: D-10 no-TTY gate (STRICT) — before any Ollama call or prompt.
    # Uses the named _stdin_isatty() seam so tests can monkeypatch it.
    # ------------------------------------------------------------------
    if not yes and not _stdin_isatty():
        typer.secho(
            "no terminal for review — pass --yes or run in a terminal",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Step 3: Extract fields via Ollama (module-reference call so
    # conftest.py's fake_extractor monkeypatch on leopard44_kb.maintenance.extract_fields
    # lands correctly — avoids the "call-time import binding" trap).
    # ------------------------------------------------------------------
    try:
        extraction = _maint.extract_fields(entry)
    except RuntimeError as exc:
        typer.secho(f"\nError: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Step 4: Review / edit (D-09).
    # --yes → use extraction as-is (no prompts, no display).
    # TTY without --yes → show table, prompt Accept/Edit/Abort.
    # ------------------------------------------------------------------
    confirmed = extraction

    if not yes:
        _render_fields_table(extraction)
        choice = typer.prompt("Accept [a] / Edit [e] / Abort [q]", default="a")
        choice = choice.strip().lower()

        if choice == "q":
            typer.secho("Aborted.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)

        if choice == "e":
            confirmed = _edit_fields(extraction)
            # Echo the updated vendor in output so the test can assert it.
            typer.echo(f"  vendor updated to: {confirmed.vendor or '(none)'}")

    # ------------------------------------------------------------------
    # Step 5: Write markdown file + ingest into vessel layer.
    # title = "Maintenance log YYYY-MM-DD" for clean Sources citations (Pitfall 3).
    # ------------------------------------------------------------------
    try:
        path = write_entry(confirmed, entry)
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    title = f"Maintenance log {confirmed.date}"
    try:
        result = ingest_file(str(path), layer="vessel", title=title)
        typer.echo(f"  {result.upper():<5}  {path}")
    except Exception as exc:
        typer.secho(f"  FAIL  {path}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("log")
def log_cmd(
    system: Optional[str] = typer.Option(None, "--system", help="Filter by top-level system (e.g. engine)"),
    vendor: Optional[str] = typer.Option(None, "--vendor", help="Filter by vendor"),
    since: Optional[str] = typer.Option(None, "--since", help="ISO date lower bound (inclusive, YYYY-MM-DD)"),
    until: Optional[str] = typer.Option(None, "--until", help="ISO date upper bound (inclusive, YYYY-MM-DD)"),
    free_text: Optional[str] = typer.Argument(None, help="Optional free-text filter (case-insensitive substring)"),
) -> None:
    """List vessel maintenance log entries (MAINT-04 / D-05).

    Filters by system, vendor, date range, and optional free-text substring.
    Entries are ordered newest event-date first. Works fully offline — no
    Ollama/embedding dependency. The vessel/maintenance scope is enforced by
    construction (no --layer flag).
    """
    # Lazy imports: keep offline; no Ollama dependency.
    from leopard44_kb.log import list_maintenance_entries
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    conn = open_db()
    try:
        apply_migrations(conn)
        rows = list_maintenance_entries(
            conn,
            system=system,
            vendor=vendor,
            since=since,
            until=until,
            free_text=free_text,
        )

        if not rows:
            typer.echo("No matching maintenance entries.")
            return

        for row in rows:
            event_date = row.get("event_date") or "—"
            sys_name = row.get("system") or "—"
            sys_detail = row.get("system_detail")
            vendor_name = row.get("vendor") or "—"
            cost_amount = row.get("cost_amount")
            cost_currency = row.get("cost_currency") or "NZD"
            parts_raw = row.get("parts")
            title = row.get("title") or row.get("path") or "—"

            # Render parts: json.loads the raw JSON-array text, join with ', '
            # (Gemini MED: the SELECT projects $.parts as a JSON string; parse it here).
            if parts_raw:
                try:
                    parts_list = json.loads(parts_raw)
                    parts_str = ", ".join(str(p) for p in parts_list) if parts_list else "—"
                except (json.JSONDecodeError, TypeError):
                    parts_str = str(parts_raw)
            else:
                parts_str = "—"

            # Format cost.
            if cost_amount is not None:
                cost_str = f"{cost_amount} {cost_currency}"
            else:
                cost_str = "—"

            # Format system + detail.
            if sys_detail:
                sys_str = f"{sys_name} ({sys_detail})"
            else:
                sys_str = sys_name

            typer.echo(
                f"  {event_date}  {sys_str:<24}  vendor={vendor_name:<12}  "
                f"cost={cost_str:<10}  parts={parts_str}  [{title}]"
            )
    finally:
        conn.close()


@app.command("serve")
def serve_cmd(
    port: int = typer.Option(8000, "--port", help="Port to listen on (default 8000; auto-increments if busy)"),
    qr: bool = typer.Option(False, "--qr", help="Bind to LAN (0.0.0.0), generate self-signed HTTPS cert with IP SAN, display QR code for phone (D-11/D-12/D-13)"),
    host_ip: str | None = typer.Option(None, "--host-ip", help="Override the auto-detected LAN IP for the QR/cert (WSL, multi-NIC, VPN)"),
) -> None:
    """Launch the local web UI (Phase 5).

    Default (no flags): binds to 127.0.0.1 over plain HTTP — localhost only.
    With --qr: binds to 0.0.0.0, generates/reuses a persisted self-signed HTTPS
    cert with an IP SAN at data/certs/, and prints a segno QR encoding
    https://<lan-ip>:<port> so the phone can reach a secure context for
    getUserMedia (D-11/D-12/D-13).  The phone trusts the cert ONCE (D-12).
    Never bind 0.0.0.0 without --qr.
    """
    # Lazy imports: keep module-level imports minimal so non-serve commands stay light.
    # Mirrors the ask_cmd / add_cmd lazy-import pattern.
    import uvicorn
    from leopard44_kb.web.app import create_app

    web_app = create_app()

    if qr:
        # --- LAN branch: 0.0.0.0, self-signed HTTPS, QR ---
        import os as _os
        import pathlib as _pathlib

        # Resolve target IP (--host-ip overrides auto-detection)
        target_ip = host_ip or _detect_lan_ip()

        try:
            actual_port = _find_free_port_lan(8443)
        except RuntimeError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        cert_path, key_path = _get_or_create_cert(target_ip)
        url = f"https://{target_ip}:{actual_port}"
        typer.echo(f"Leopard 44 KB web UI (LAN): {url}")
        typer.echo("Scan with your phone:")

        import segno  # noqa: PLC0415
        segno.make(url).terminal(compact=True)

        typer.echo(
            "One-time phone trust: iOS → Settings > General > About > Certificate Trust Settings"
        )

        # WSL2 advisory: phone reaches the Windows host IP, not the WSL internal IP.
        # Print the netsh portproxy template so the operator can copy-paste it.
        if _os.path.exists("/proc/version"):
            try:
                proc_ver = _pathlib.Path("/proc/version").read_text().lower()
            except OSError:
                proc_ver = ""
            if "microsoft" in proc_ver:
                typer.secho(
                    f"\nWSL2 detected: your phone must reach the Windows host IP, not "
                    f"the WSL IP ({target_ip}) shown above.  Configure a port proxy in "
                    f"an elevated Windows PowerShell:\n"
                    f"  netsh interface portproxy add v4tov4 "
                    f"listenport={actual_port} listenaddress=0.0.0.0 "
                    f"connectport={actual_port} connectaddress={target_ip}",
                    fg=typer.colors.YELLOW,
                    err=True,
                )

        try:
            uvicorn.run(
                web_app,
                host="0.0.0.0",
                port=actual_port,
                ssl_keyfile=str(key_path),
                ssl_certfile=str(cert_path),
                access_log=False,
            )
        except OSError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    else:
        # --- Default branch: 127.0.0.1 plain HTTP (localhost only) ---
        try:
            actual_port = _find_free_port(port)
        except RuntimeError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        url = f"http://127.0.0.1:{actual_port}"
        typer.echo(f"Leopard 44 KB web UI: {url}")
        typer.echo("Press Ctrl+C to stop.")
        try:
            uvicorn.run(web_app, host="127.0.0.1", port=actual_port, access_log=False)
        except OSError as exc:
            typer.secho(
                f"Port {actual_port} became unavailable — retry `l44 serve --port <N>`.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Zone sub-app (Plan 04, Wave 3)
# ---------------------------------------------------------------------------

zone_app = typer.Typer(help="Manage storage zones.")
item_app = typer.Typer(help="Manage inventory items.")
schematic_app = typer.Typer(help="Schematic rendering and page suggestion.")
deviation_app = typer.Typer(help="Record and review factory deviations.")

app.add_typer(zone_app, name="zone")
app.add_typer(item_app, name="item")
app.add_typer(schematic_app, name="schematic")
app.add_typer(deviation_app, name="deviation")

VALID_CATEGORIES_CLI = ("spare", "provision", "safety", "tool", "toy")


@zone_app.command("add")
def zone_add_cmd(
    name: str = typer.Argument(..., help="Zone slug, e.g. 'stbd-aft-cabin'"),
    label: str = typer.Argument(..., help="Display label, e.g. 'Stbd aft cabin'"),
    side: Optional[str] = typer.Option(None, "--side", help="port | stbd | centre | both"),
    fore_aft: Optional[str] = typer.Option(None, "--fore-aft", help="fwd | mid | aft"),
    area: Optional[str] = typer.Option(None, "--area", help="Grouping tag, e.g. 'cockpit'"),
    vertical_index: Optional[float] = typer.Option(None, "--vertical-index", help="Orderable vertical position"),
    description: Optional[str] = typer.Option(None, "--description", help="Explicit vertical description (skips AI generation)"),
    no_ai: bool = typer.Option(False, "--no-ai", help="Skip Ollama AI description generation"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive review and commit as-is"),
) -> None:
    """Add a storage zone to the vessel layout."""
    import leopard44_kb.inventory as _inv
    import sqlite3 as _sqlite3
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    # Empty-name guard
    if not name or not name.strip():
        typer.secho("empty zone name — provide a slug", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # TTY/--yes gate (T-08-15 repudiation mitigation)
    if not yes and not _stdin_isatty():
        typer.secho(
            "no terminal for review — pass --yes or run in a terminal",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Build fields for review
    fields = {
        "name": name,
        "label": label,
        "side": side or "(none)",
        "fore_aft": fore_aft or "(none)",
        "area": area or "(none)",
        "vertical_index": vertical_index if vertical_index is not None else "(none)",
        "description": description or "(AI-generated)" if not no_ai else "(none)",
    }

    if not yes:
        typer.echo("\n--- Zone fields ---")
        for k, v in fields.items():
            typer.echo(f"  {k:<18}: {v}")
        typer.echo("-------------------")
        choice = typer.prompt("Accept [a] / Abort [q]", default="a")
        choice = choice.strip().lower()
        if choice == "q":
            typer.secho("Aborted.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)

    conn = open_db()
    try:
        apply_migrations(conn)
        try:
            zone_id = _inv.create_zone(
                conn,
                name=name,
                label=label,
                side=side,
                fore_aft=fore_aft,
                vertical_index=vertical_index,
                vertical_desc=description,
                area=area,
                use_ai=(not no_ai),
            )
        except RuntimeError as exc:
            # Ollama offline when use_ai=True
            typer.secho(
                f"Error generating AI description: {exc}\n"
                "Tip: pass --no-ai to skip AI description generation.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        except _sqlite3.IntegrityError:
            typer.secho(
                f"zone already exists: {name}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"  Created zone id={zone_id}  {label!r}  ({name})")
    finally:
        conn.close()


@zone_app.command("list")
def zone_list_cmd(
    area: Optional[str] = typer.Option(None, "--area", help="Filter by area tag"),
) -> None:
    """List storage zones."""
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    conn = open_db()
    try:
        apply_migrations(conn)
        if area is not None:
            rows = conn.execute(
                "SELECT * FROM zones WHERE area = ? ORDER BY area, vertical_index",
                (area,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM zones ORDER BY area, vertical_index"
            ).fetchall()

        if not rows:
            typer.echo("No zones found.")
            return

        for row in rows:
            label = row["label"]
            side = row["side"] or "—"
            fa = row["fore_aft"] or "—"
            vdesc = row["vertical_desc"] or ""
            line = f"  {label:<32}  side={side:<6}  fore_aft={fa:<4}"
            if vdesc:
                line += f"  {vdesc}"
            typer.echo(line)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Item sub-app (Plan 04, Wave 3)
# ---------------------------------------------------------------------------


@item_app.command("add")
def item_add_cmd(
    name: str = typer.Argument(..., help="Item name"),
    category: str = typer.Option(..., "--category", help="spare | provision | safety | tool | toy"),
    zone: Optional[str] = typer.Option(None, "--zone", help="Zone slug"),
    aliases: Optional[str] = typer.Option(None, "--aliases", help="Comma-separated synonyms"),
    brand: Optional[str] = typer.Option(None, "--brand", help="Brand name"),
    model_number: Optional[str] = typer.Option(None, "--model-number", help="Model or part number"),
    slot_row: Optional[int] = typer.Option(None, "--slot-row", help="Sub-slot row number"),
    slot_col: Optional[int] = typer.Option(None, "--slot-col", help="Sub-slot column number"),
    quantity: Optional[float] = typer.Option(None, "--quantity", help="Quantity at last check"),
    part_number: Optional[str] = typer.Option(None, "--part-number", help="Part number (spare)"),
    best_before: Optional[str] = typer.Option(None, "--best-before", help="Best before date (provision)"),
    expiry: Optional[str] = typer.Option(None, "--expiry", help="Expiry date (safety)"),
    last_inspected: Optional[str] = typer.Option(None, "--last-inspected", help="Last inspection date"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Free text notes"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive review and commit as-is"),
) -> None:
    """Add an inventory item to the vessel layer."""
    import sqlite3 as _sqlite3
    import leopard44_kb.inventory as _inv
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    # FAIL-FAST zone resolution (finding 9): before empty-guard, TTY gate, and review loop
    resolved_zone_id: Optional[int] = None
    if zone is not None:
        conn_check = open_db()
        try:
            apply_migrations(conn_check)
            zone_row = conn_check.execute(
                "SELECT id FROM zones WHERE name = ?", (zone,)
            ).fetchone()
        finally:
            conn_check.close()
        if zone_row is None:
            typer.secho(
                f"zone not found: {zone}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        resolved_zone_id = zone_row["id"]

    # Empty-name guard
    if not name or not name.strip():
        typer.secho("empty item name — provide a name", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Validate category early (T-08-13)
    if category.lower().strip() not in VALID_CATEGORIES_CLI:
        typer.secho(
            f"invalid category {category!r} — must be one of {VALID_CATEGORIES_CLI}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # TTY/--yes gate (T-08-15 repudiation mitigation)
    if not yes and not _stdin_isatty():
        typer.secho(
            "no terminal for review — pass --yes or run in a terminal",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Build metadata from category-specific options
    metadata: dict = {}
    if part_number:
        metadata["part_number"] = part_number
    if best_before:
        metadata["best_before"] = best_before
    if expiry:
        metadata["expiry"] = expiry
    if last_inspected:
        metadata["last_inspected"] = last_inspected

    # Build sub-slot dict if row+col provided
    current_sub_slot: Optional[dict] = None
    if slot_row is not None and slot_col is not None:
        current_sub_slot = {"row": slot_row, "col": slot_col}

    # Review fields before write
    if not yes:
        typer.echo("\n--- Item fields ---")
        typer.echo(f"  name         : {name}")
        typer.echo(f"  category     : {category}")
        typer.echo(f"  zone         : {zone or '(none)'}")
        typer.echo(f"  aliases      : {aliases or '(none)'}")
        typer.echo(f"  brand        : {brand or '(none)'}")
        typer.echo(f"  model_number : {model_number or '(none)'}")
        typer.echo(f"  quantity     : {quantity if quantity is not None else '(none)'}")
        if metadata:
            typer.echo(f"  metadata     : {metadata}")
        typer.echo("-------------------")
        choice = typer.prompt("Accept [a] / Abort [q]", default="a")
        choice = choice.strip().lower()
        if choice == "q":
            typer.secho("Aborted.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)

    conn = open_db()
    try:
        apply_migrations(conn)
        try:
            import os
            from pathlib import Path as _Path
            repo_root_val = _Path(os.getcwd())
            item_id = _inv.create_item(
                conn,
                name=name,
                category=category,
                zone_id=resolved_zone_id,
                aliases=aliases,
                brand=brand,
                model_number=model_number,
                current_sub_slot=current_sub_slot,
                metadata=metadata if metadata else None,
                notes=notes,
                quantity=quantity,
                repo_root=repo_root_val,
            )
        except (ValueError, RuntimeError, _sqlite3.IntegrityError) as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        zone_path = f"/{zone}" if zone else ""
        typer.echo(f"  Created item id={item_id}  {name!r}  ({category}){zone_path}")
    finally:
        conn.close()


@item_app.command("list")
def item_list_cmd(
    zone: Optional[str] = typer.Option(None, "--zone", help="Filter by zone slug"),
    category: Optional[str] = typer.Option(None, "--category", help="Filter by category"),
    free_text: Optional[str] = typer.Argument(None, help="Free-text substring filter"),
) -> None:
    """List inventory items with optional filters."""
    import leopard44_kb.inventory as _inv
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    conn = open_db()
    try:
        apply_migrations(conn)
        items = _inv.list_items(conn, zone=zone, category=category, text=free_text)
        if not items:
            typer.echo("No items found.")
            return
        for item in items:
            zone_label = item.get("zone_label") or "—"
            typer.echo(
                f"  {item['name']:<32}  {item['category']:<12}  zone={zone_label}"
            )
    finally:
        conn.close()


@item_app.command("find")
def item_find_cmd(
    query: str = typer.Argument(..., help="Search query (partial match on name/aliases/brand/model)"),
) -> None:
    """Find items by name, aliases, brand, or model number."""
    import leopard44_kb.inventory as _inv
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    conn = open_db()
    try:
        apply_migrations(conn)
        results = _inv.find_item(conn, query)
        if not results:
            typer.echo(f"No items matching {query!r}.")
            return
        for r in results:
            item = r["item"]
            zone = r.get("zone")
            zone_label = zone["label"] if zone else "—"
            sub_slot = r.get("sub_slot")
            slot_str = ""
            if sub_slot:
                row_label = sub_slot.get("row_label") or f"row {sub_slot.get('row', '')}"
                col_label = sub_slot.get("col_label") or f"col {sub_slot.get('col', '')}"
                slot_str = f"  slot={row_label}/{col_label}"
            typer.echo(
                f"  {item['name']:<32}  {item['category']:<12}  zone={zone_label}{slot_str}"
            )
    finally:
        conn.close()


@item_app.command("locate")
def item_locate_cmd(
    query: str = typer.Argument(..., help="What to find (name or natural-language description)"),
) -> None:
    """Locate an item — structured zone + shelf, with semantic fallback.

    Uses the structured item record for location (not LLM text).
    Offline-safe fast verb; does NOT route through ask/answer.
    """
    import leopard44_kb.inventory as _inv
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    conn = open_db()
    try:
        apply_migrations(conn)
        result = _inv.locate_item(conn, query)

        if not result.get("found"):
            typer.echo(f"Not found: {query!r}")
            return

        items = result.get("items", [])
        chunks = result.get("chunks", [])

        if items:
            if len(items) > 1:
                typer.echo(f"Multiple matches for {query!r}:")
            for r in items:
                item = r["item"]
                zone = r.get("zone")
                zone_label = zone["label"] if zone else "unknown location"
                sub_slot = r.get("sub_slot")
                history = r.get("history", [])

                typer.echo(f"  {item['name']}")
                typer.echo(f"    Location : {zone_label}")
                if sub_slot:
                    row_label = sub_slot.get("row_label") or f"row {sub_slot.get('row', '')}"
                    col_label = sub_slot.get("col_label") or f"col {sub_slot.get('col', '')}"
                    typer.echo(f"    Shelf    : {row_label} / {col_label}")
                if history:
                    other_places = []
                    for h in history[:3]:
                        hz = h.get("zone_id")
                        if hz:
                            hz_row = conn.execute(
                                "SELECT label FROM zones WHERE id = ?", (hz,)
                            ).fetchone()
                            if hz_row:
                                other_places.append(hz_row["label"])
                    if other_places:
                        typer.echo(f"    Also try : {', '.join(other_places)}")
        elif chunks:
            typer.echo(f"Possible match for {query!r} (from knowledge base):")
            for chunk in chunks[:3]:
                typer.echo(f"  {chunk.get('content', '')[:120]}")
            typer.echo(f"\nTip: try `l44 item find <name>` for exact matches.")
    finally:
        conn.close()


@schematic_app.command("render")
def schematic_render_cmd(
    pdf: str = typer.Argument(..., help="Path to the source PDF"),
    pages: Optional[str] = typer.Option(
        None, "--pages", help="Page range/list e.g. '61-89' or '61,62,65'"
    ),
    suggest: bool = typer.Option(
        False, "--suggest", help="Use cloud vision to suggest schematic pages (requires API key)"
    ),
) -> None:
    """Render selected PDF pages to PNG files in data/schematics/."""
    # Lazy imports — mirrors zone_add_cmd / item_add_cmd pattern
    from leopard44_kb.schematic import parse_page_spec, render_pages, suggest_pages
    from leopard44_kb.paths import ensure_data_dirs, repo_root
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db
    import fitz as _fitz
    from pathlib import Path as _Path

    # Resolve project root via the shared helper (review fix: not raw Path(os.getcwd()))
    root = repo_root()
    ensure_data_dirs(root)  # Pitfall 7: create data/schematics/ before write
    output_dir = root / "data" / "schematics"

    # Migration best-effort (review fix): render writes FILES ONLY and does not need
    # the DB.  Wrap apply_migrations in try/except so an unrelated DB-health issue
    # does not abort a filesystem-only render (CR-01 discipline still applies on any
    # path that TOUCHES the DB).
    try:
        conn = open_db()
        try:
            apply_migrations(conn)
        finally:
            conn.close()
    except Exception as exc:
        typer.secho(
            f"Warning: could not apply DB migrations ({exc}). "
            "Continuing with render (migration is best-effort for a filesystem-only render).",
            fg=typer.colors.YELLOW,
            err=True,
        )

    pdf_path = _Path(pdf)
    if not pdf_path.exists():
        typer.secho(f"Error: PDF not found: {pdf_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Page resolution
    if suggest:
        # Cloud-vision suggester: propose pages, confirm before render
        try:
            suggested = suggest_pages(pdf_path)
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            typer.secho(
                f"Cloud-vision suggester failed: {exc}\n"
                "Tip: pass --pages to specify pages without a network call.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        if not suggested:
            typer.secho("Cloud-vision returned no page suggestions.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

        typer.echo(f"Suggested schematic pages: {suggested}")
        confirmed = typer.confirm("Render these pages?", default=True)
        if not confirmed:
            typer.secho("Aborted by user.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)
        page_numbers = suggested

    elif pages:
        # Open PDF once to get page count for range validation
        try:
            doc = _fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
        except Exception as exc:
            typer.secho(
                f"Error opening PDF for page count: {exc}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1)

        try:
            page_numbers = parse_page_spec(pages, page_count=page_count)
        except ValueError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    else:
        typer.secho(
            "Error: pass --pages <range/list> or --suggest to select pages to render.\n"
            "  Example: l44 schematic render manual.pdf --pages 61-89",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Render pages to PNG
    try:
        written = render_pages(pdf_path, page_numbers, output_dir)
    except (ValueError, IndexError) as exc:
        typer.secho(f"Render error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Rendered {len(written)} page(s) to {output_dir}/")
    for p in written:
        typer.echo(f"  {p.name}")


# ---------------------------------------------------------------------------
# Deviation sub-app (Plan 02, Phase 11)
# ---------------------------------------------------------------------------


def _render_deviation_fields_table(extraction: object) -> None:
    """Render extracted deviation fields as a human-readable table.

    Prints each field name and its current value so the owner can decide
    whether to Accept, Edit, or Abort before committing.
    """
    from leopard44_kb.deviation import DeviationExtraction
    e: DeviationExtraction = extraction  # type: ignore[assignment]

    typer.echo("\n--- Deviation fields ---")
    typer.echo(f"  component     : {e.component}")
    typer.echo(f"  factory_spec  : {e.factory_spec or '(none)'}")
    typer.echo(f"  as_built      : {e.as_built or '(none)'}")
    typer.echo(f"  reason        : {e.reason or '(none)'}")
    typer.echo(f"  date_noted    : {e.date_noted or '(none)'}")
    typer.echo(f"  notes         : {e.notes or '(none)'}")
    typer.echo("------------------------")


def _edit_deviation_fields(extraction: object) -> object:
    """Walk each deviation field interactively; Enter keeps the current value.

    Clear-field sentinel (finding 6): typing "-" for an OPTIONAL field clears
    it to NULL. The "component" field is required and cannot be cleared via "-"
    (a literal "-" is kept as text, not interpreted as NULL).

    Field order: component, factory_spec, as_built, reason, date_noted, notes.

    Returns a new DeviationExtraction with all edits applied via .model_copy().
    """
    from leopard44_kb.deviation import DeviationExtraction
    e: DeviationExtraction = extraction  # type: ignore[assignment]

    SENTINEL = "-"

    # component: required — prompt without clear-sentinel option; "-" kept as text
    component_val = typer.prompt("component", default=e.component)
    if not component_val.strip():
        component_val = e.component  # never empty

    # Optional fields with clear-sentinel support
    factory_spec_raw = typer.prompt(
        "factory_spec (- to clear)", default=e.factory_spec or ""
    )
    as_built_raw = typer.prompt(
        "as_built (- to clear)", default=e.as_built or ""
    )
    reason_raw = typer.prompt(
        "reason (- to clear)", default=e.reason or ""
    )
    date_noted_raw = typer.prompt(
        "date_noted (- to clear)", default=e.date_noted or ""
    )
    notes_raw = typer.prompt(
        "notes (- to clear)", default=e.notes or ""
    )

    def _resolve(raw: str, original: Optional[str]) -> Optional[str]:
        """Map sentinel to None, empty-string to original (Enter keeps), else use new value."""
        if raw == SENTINEL:
            return None  # explicit clear
        stripped = raw.strip()
        if not stripped:
            return original  # Enter keeps existing value
        return stripped

    updates: dict = {
        "component": component_val.strip() or e.component,
        "factory_spec": _resolve(factory_spec_raw, e.factory_spec),
        "as_built": _resolve(as_built_raw, e.as_built),
        "reason": _resolve(reason_raw, e.reason),
        "date_noted": _resolve(date_noted_raw, e.date_noted),
        "notes": _resolve(notes_raw, e.notes),
    }
    return e.model_copy(update=updates)


@deviation_app.command("add")
def deviation_add_cmd(
    entry: str = typer.Argument(..., help="Deviation description in natural language"),
    zone: Optional[str] = typer.Option(None, "--zone", help="Zone name (must exist)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive review and commit as-is"),
) -> None:
    """Record a factory deviation for this vessel.

    Extracts structured fields from natural-language text using the local LLM,
    lets you review and optionally edit the fields, then writes a structured
    deviations row and a vessel-layer chunk into the knowledge base.

    Pass --yes to skip the interactive review step. In non-interactive contexts,
    --yes is required — the command will fail fast otherwise.

    The deviation is always committed as vessel-layer with no layer flag — the
    vessel scope is enforced mechanically. Does not import leopard44_kb.capture.
    """
    import os as _os
    import sqlite3 as _sqlite3
    import leopard44_kb.deviation as _dev
    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    # Step 1: Empty-entry guard — before zone check, TTY gate, or LLM call
    if not entry or not entry.strip():
        typer.secho(
            "empty deviation entry — provide some text",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Step 2: FAIL-FAST zone resolution (name-only, identical to item_add_cmd finding 5)
    # Reuses the exact same name-only resolver as item_add_cmd:
    #   SELECT id FROM zones WHERE name = ?   (no slug/label/alias/id path)
    #   Error message shape: "zone not found: {zone}"
    resolved_zone_id: Optional[int] = None
    conn = open_db()
    try:
        apply_migrations(conn)
        if zone is not None:
            zone_row = conn.execute(
                "SELECT id FROM zones WHERE name = ?", (zone,)
            ).fetchone()
            if zone_row is None:
                typer.secho(
                    f"zone not found: {zone}",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)
            resolved_zone_id = zone_row["id"]

        # Step 3: D-10 TTY gate — if not --yes and not a TTY, fail fast
        if not yes and not _stdin_isatty():
            typer.secho(
                "no terminal for review — pass --yes or run in a terminal",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Step 4: Extract via module-reference call so monkeypatch on
        # leopard44_kb.deviation.extract_fields binds correctly (same trap as
        # maintenance avoids with _maint.extract_fields).
        try:
            extraction = _dev.extract_fields(entry)
        except RuntimeError as exc:
            typer.secho(f"\nError: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        # Step 5: Review / edit
        # --yes → use extraction as-is (no prompts, no display)
        # TTY without --yes → show table, prompt Accept [a] / Edit [e] / Abort [q]
        confirmed = extraction

        if not yes:
            _render_deviation_fields_table(extraction)
            choice = typer.prompt("Accept [a] / Edit [e] / Abort [q]", default="a")
            choice = choice.strip().lower()

            if choice == "q":
                typer.secho("Aborted.", fg=typer.colors.YELLOW)
                raise typer.Exit(code=0)

            if choice == "e":
                confirmed = _edit_deviation_fields(extraction)

        # Step 6: Write via create_deviation (dual-write: DB row + vessel-layer chunk)
        try:
            deviation_id = _dev.create_deviation(
                conn,
                confirmed,
                entry,
                zone_id=resolved_zone_id,
                repo_root=Path(_os.getcwd()),
            )
        except (ValueError, RuntimeError, _sqlite3.IntegrityError) as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Created deviation id={deviation_id}")

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Voice sub-app (Plan 02, Phase 10)
# ---------------------------------------------------------------------------

voice_app = typer.Typer(help="Voice query setup and management.")
app.add_typer(voice_app, name="voice")


@voice_app.command("setup")
def voice_setup_cmd() -> None:
    """Create .venv-stt, install faster-whisper, pre-download whisper small int8 weights.

    Three steps:
      1. Create an isolated Python venv at .venv-stt (separate from the app venv;
         the typer/ctranslate2 conflict is permanent — subprocess IPC only, D-06).
      2. Install faster-whisper==1.2.1 into .venv-stt via pip.
      3. Pre-download the Systran/faster-whisper-small weights via
         huggingface_hub.snapshot_download (returns the local snapshot dir) and
         record that path to data/voice-model-path.txt so the STT worker can load
         the model OFFLINE at sea without any network access or re-resolution.

    NOTE: .venv-stt/bin/* paths are own-use Linux/WSL only.
    Windows Scripts/ venv-bin resolution is a Phase-13 (public release) concern.
    """
    import subprocess  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415
    from leopard44_kb.paths import repo_root  # noqa: PLC0415

    root = repo_root()
    venv_path = root / ".venv-stt"

    # Step 1/3: Create STT venv (isolated from the app venv — D-06 permanent constraint)
    typer.echo("Step 1/3: Creating STT venv at .venv-stt …")
    try:
        subprocess.run(
            [_sys.executable, "-m", "venv", str(venv_path)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        typer.secho(f"Error creating venv: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Step 2/3: Install faster-whisper==1.2.1 into the isolated venv
    # NOTE: targets .venv-stt/bin/pip, NEVER the app venv's pip
    pip = str(venv_path / "bin" / "pip")
    typer.echo("Step 2/3: Installing faster-whisper==1.2.1 …")
    try:
        subprocess.run(
            [pip, "install", "faster-whisper==1.2.1"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        typer.secho(f"Error installing faster-whisper: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Step 3/3: Pre-download the whisper small int8 weights via snapshot_download.
    # snapshot_download returns the LOCAL snapshot dir (absolute path) — this is the
    # recorded path so the STT worker loads OFFLINE (zero download at sea, VOICE-03/D-08).
    # GAP-1 FIX: huggingface_hub is NOT in the app venv (it is a faster-whisper transitive
    # dep that lives only in .venv-stt). We invoke snapshot_download INSIDE .venv-stt/bin/python
    # (where huggingface_hub IS installed) and capture the printed snapshot dir from stdout.
    # The app process never imports huggingface_hub.
    python = str(venv_path / "bin" / "python")
    typer.echo("Step 3/3: Pre-downloading whisper small int8 weights (~462MB) …")
    _snapshot_prog = (
        "from huggingface_hub import snapshot_download; "
        "print(snapshot_download('Systran/faster-whisper-small'))"
    )
    _snap_result = subprocess.run(
        [python, "-c", _snapshot_prog],
        capture_output=True,
        text=True,
    )
    if _snap_result.returncode != 0:
        typer.secho(
            f"Error downloading whisper weights: {_snap_result.stderr.strip()}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    snapshot_dir = _snap_result.stdout.strip()
    if not snapshot_dir:
        typer.secho(
            "Error: snapshot_download returned empty output; download may have failed.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Warm/verify the model load from the recorded path (subprocess so faster-whisper
    # stays out of the app venv entirely — D-06). WR-03: a failed warm-load is FATAL —
    # we must NOT write the marker or claim success, because a broken install (corrupt
    # download, incompatible ctranslate2, wrong compute type) would otherwise make
    # /api/voice-status report installed:true and the mic button live, then fail every
    # transcription at sea with no clean re-setup signal.
    # python is already set above (str(venv_path / "bin" / "python"))
    try:
        subprocess.run(
            [
                python,
                "-c",
                (
                    f"from faster_whisper import WhisperModel; "
                    f"WhisperModel({snapshot_dir!r}, device='cpu', compute_type='int8')"
                ),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        typer.secho(
            f"Error: model warm-load verification failed: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        # Do NOT write the marker / claim success on a failed verification (WR-03).
        raise typer.Exit(code=1)

    # Write the snapshot dir to the marker file so the STT worker reads it offline.
    # Only reached after a clean warm-load verification (WR-03).
    marker = root / "data" / "voice-model-path.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(snapshot_dir) + "\n")

    typer.secho("Voice setup complete.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# Capture command (Plan 04, Phase 12)
# l44 capture <photo> [--zone ZONE] [--cloud] [--yes]
#
# Top-level @app.command (not a sub-app) — mirrors the `ask` command shape.
# OFFLINE BOUNDARY NOTE: this command imports leopard44_kb.capture.* INSIDE the
# command body only (lazy imports). Importing leopard44_kb.cli never pulls capture
# onto a module-load path. The boundary that matters is web.app (enforced by
# tests/test_capture_import_boundary.py — web.app never imports capture/).
# ---------------------------------------------------------------------------


def _render_capture_fields_table(result: dict, photo_path: str) -> None:
    """Render capture vision result fields as a human-readable table.

    Shows the captured FIELDS plus the SOURCE PHOTO PATH so the owner can
    review all information before deciding to Accept, Edit, or Abort (M2 scope).

    WR-03: the category line shows the COERCED enum value that will actually be
    stored, surfacing "X → Y" when the free-form vision string is normalized — so
    the owner reviews what is committed, not a value the DB will silently rewrite.
    """
    from leopard44_kb.capture.confirm import normalize_category

    raw_category = (result.get("category") or "").strip()
    stored_category = normalize_category(raw_category)
    if raw_category and raw_category.lower() != stored_category:
        category_display = f"{stored_category}  (normalized: {raw_category} → {stored_category})"
    else:
        category_display = stored_category or "(none)"

    typer.echo("\n--- Capture fields ---")
    typer.echo(f"  name          : {result.get('item') or '(unknown)'}")
    typer.echo(f"  brand         : {result.get('brand') or '(none)'}")
    typer.echo(f"  category      : {category_display}")
    typer.echo(f"  suggested_zone: {result.get('suggested_zone') or '(none)'}")
    typer.echo(f"  confidence    : {result.get('confidence', 0.0):.2f}")
    typer.echo(f"  source photo  : {photo_path}")
    typer.echo("----------------------")


def _edit_capture_fields(result: dict) -> dict:
    """Prompt for one field to edit and return an updated copy of result.

    Asks which field the owner wants to change, then prompts for the new value.
    Returns the updated result dict (with sentinel key ``_edited_zone`` if zone
    was edited so the caller can re-resolve the zone_id).

    Called once per 'e' choice in the Accept/Edit/Abort loop — the caller may
    loop back for additional edits via further 'e' responses.

    Editable fields: name, brand, category, zone.
    """
    updated = dict(result)

    field = typer.prompt(
        "Field to edit (name/brand/category/zone)",
        default="name",
    ).strip().lower()

    if field == "name":
        val = typer.prompt("name", default=updated.get("item") or "").strip()
        if val:
            updated["item"] = val
    elif field == "brand":
        val = typer.prompt("brand (blank to clear)", default=updated.get("brand") or "").strip()
        updated["brand"] = val or None
    elif field == "category":
        # WR-04: validate an EDITED category against the enum (reject + re-prompt
        # on invalid), mirroring item_add_cmd's VALID_CATEGORIES_CLI check. Do NOT
        # silently coerce a user edit — a deliberate edit to an invalid value must
        # be rejected, not rewritten to "spare" downstream.
        default_cat = (updated.get("category") or "spare").strip().lower()
        if default_cat not in VALID_CATEGORIES_CLI:
            default_cat = "spare"
        while True:
            val = typer.prompt(
                f"category {VALID_CATEGORIES_CLI}", default=default_cat
            ).strip().lower()
            if not val:
                break  # keep existing value
            if val in VALID_CATEGORIES_CLI:
                updated["category"] = val
                break
            typer.secho(
                f"invalid category {val!r} — must be one of {VALID_CATEGORIES_CLI}",
                fg=typer.colors.RED,
            )
    elif field == "zone":
        val = typer.prompt(
            "zone name (exact match required)", default=updated.get("suggested_zone") or ""
        ).strip()
        updated["_edited_zone"] = val  # sentinel for zone re-resolve in caller
    else:
        typer.secho(
            f"Unknown field {field!r} — valid fields: name, brand, category, zone",
            fg=typer.colors.YELLOW,
        )

    return updated


def _resolve_zone_id(
    conn: object,  # sqlite3.Connection
    zone_name: Optional[str],
    context: str = "",
) -> Optional[int]:
    """Resolve a zone name to its id via EXACT match only (capture edit path).

    H3: the capture edit prompt states "exact match required", so this is
    exact-match only — there is NO partial `LIKE "%name%"` fallback. A unique
    partial match must NOT silently resolve to a (possibly wrong) zone. On any
    non-exact value, warn and resolve to None (zone_id stays NULL).

    Returns the integer zone id, or None if the name is empty or has no exact
    match. Prints a warning on unknown/duplicate zone (M4).
    """
    import sqlite3 as _sqlite3

    c: _sqlite3.Connection = conn  # type: ignore[assignment]

    if not zone_name or not zone_name.strip():
        return None

    name = zone_name.strip()

    # Exact-match ONLY (H3 — capture edits require an exact zone name).
    rows = c.execute("SELECT id FROM zones WHERE name = ?", (name,)).fetchall()
    if len(rows) == 1:
        return rows[0]["id"] if hasattr(rows[0], "__getitem__") else rows[0][0]
    if len(rows) > 1:
        typer.secho(
            f"Warning: zone name {name!r} matches multiple zones — "
            f"zone_id will be NULL (M4){context}",
            fg=typer.colors.YELLOW,
        )
        return None

    # No exact match — do NOT fall back to a partial LIKE. Warn + NULL (H3).
    typer.secho(
        f"Warning: no exact zone match for {name!r} — zone_id will be NULL{context}",
        fg=typer.colors.YELLOW,
    )
    return None


@app.command("capture")
def capture_cmd(
    photo: str = typer.Argument(..., help="Path to the photo to identify"),
    zone: Optional[str] = typer.Option(None, "--zone", help="Override zone (name must exist)"),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud vision (Anthropic API; sends photo to api.anthropic.com)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive review and commit as-is"),
) -> None:
    """Identify an item from a photo and add it to the inventory.

    Runs local vision (qwen2.5vl:7b) first. Use --cloud to fall back to the
    Anthropic cloud model (requires ANTHROPIC_API_KEY; sends the photo to
    api.anthropic.com — explicit consent required, never automatic).

    Below-0.7 confidence is flagged; a 'rerun with --cloud' suggestion is
    printed — but bytes are NEVER automatically sent to the cloud.

    On Accept: writes one inventory item (+ GPS-stripped photo, fail-soft).
    On Abort: writes nothing.

    Does not import leopard44_kb.capture at module load — lazy body import only.
    """
    import os as _os
    import sqlite3 as _sqlite3

    from leopard44_kb.schema import apply_migrations
    from leopard44_kb.store import open_db

    # Step 1: Validate the photo path BEFORE anything else (exit non-zero on failure).
    # Lazy import so capture/ never loads on the web/serve path.
    import leopard44_kb.capture.photo as _photo

    try:
        validated_photo = _photo.validate_photo_input(photo)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Step 2: Open DB and apply migrations.
    conn = open_db()
    try:
        apply_migrations(conn)

        # Step 3: TTY gate — require --yes or a real terminal.
        if not yes and not _stdin_isatty():
            typer.secho(
                "no terminal for review — pass --yes or run in a terminal",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Step 4: Load zone list for the vision prompt + zone resolution.
        rows = conn.execute("SELECT name FROM zones ORDER BY name").fetchall()
        zone_names: list[str] = [
            r["name"] if hasattr(r, "__getitem__") else r[0] for r in rows
        ]

        # Step 5: --cloud notice BEFORE the call (H3 consent gate).
        if cloud:
            typer.secho(
                "Sending photo to cloud (api.anthropic.com) — "
                "ANTHROPIC_API_KEY is required.",
                fg=typer.colors.CYAN,
            )

        # Step 6: Identify via module-ref call (monkeypatch-safe — same pattern as
        # deviation_add_cmd using `_dev.extract_fields`).
        import leopard44_kb.capture as _capture

        try:
            result = _capture.identify_item_for_cli(
                str(validated_photo), zone_names, cloud=cloud
            )
        except RuntimeError as exc:
            typer.secho(f"\nError: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        # Step 7: Low-confidence flag + rerun-with-cloud suggestion (H3).
        # Bytes are NEVER auto-sent even at low confidence.
        if result.get("low_confidence"):
            typer.secho(
                f"\nLow confidence ({result.get('confidence', 0.0):.2f}) — "
                "vision result is uncertain. Review carefully.",
                fg=typer.colors.YELLOW,
            )
            if not cloud:
                typer.secho(
                    "Tip: rerun with --cloud to try cloud vision for a stronger identification.",
                    fg=typer.colors.YELLOW,
                )

        # Step 8: zone resolution — precedence: --zone > edited > vision-exact > null.
        # An explicit --zone is AUTHORITATIVE and must NOT be overridden by a later
        # interactive zone edit (H2). Track whether the user passed --zone at all.
        zone_explicit = zone is not None

        # Resolve --zone first (fail-fast if given but not found via exact match).
        # For --zone: unknown still proceeds with zone_id=None (just warns).
        if zone is not None:
            # Explicit --zone: exact match; warn + None if not found.
            zone_rows = conn.execute(
                "SELECT id FROM zones WHERE name = ?", (zone,)
            ).fetchall()
            if not zone_rows:
                typer.secho(
                    f"Warning: --zone {zone!r} not found — zone_id will be NULL",
                    fg=typer.colors.YELLOW,
                )
                resolved_zone_id: Optional[int] = None
            elif len(zone_rows) > 1:
                typer.secho(
                    f"Warning: --zone {zone!r} is ambiguous — zone_id will be NULL",
                    fg=typer.colors.YELLOW,
                )
                resolved_zone_id = None
            else:
                r = zone_rows[0]
                resolved_zone_id = r["id"] if hasattr(r, "__getitem__") else r[0]
        else:
            # No --zone: use vision suggested_zone (exact match from taxonomy).
            # zone_id from vision normalisation is the zone NAME (not an int) when matched,
            # or None when not matched. We need to look up the DB id.
            vision_zone_name = result.get("suggested_zone")
            if vision_zone_name:
                vz_row = conn.execute(
                    "SELECT id FROM zones WHERE name = ?", (vision_zone_name,)
                ).fetchone()
                resolved_zone_id = (
                    (vz_row["id"] if hasattr(vz_row, "__getitem__") else vz_row[0])
                    if vz_row else None
                )
                if vz_row is None:
                    typer.secho(
                        f"Warning: vision suggested zone {vision_zone_name!r} is not in "
                        f"the zone taxonomy — zone_id will be NULL",
                        fg=typer.colors.YELLOW,
                    )
            else:
                resolved_zone_id = None
                if result.get("zone_id") is None:
                    typer.secho(
                        "Warning: unknown or missing zone from vision — zone_id will be NULL",
                        fg=typer.colors.YELLOW,
                    )

        # Step 9: Show the fields table with source photo path (M2 scope).
        _render_capture_fields_table(result, str(validated_photo))

        # Step 10: Accept / Edit / Abort prompt (loop so owner can edit multiple fields).
        confirmed = result

        if not yes:
            while True:
                choice = typer.prompt("Accept [a] / Edit [e] / Abort [q]", default="a")
                choice = choice.strip().lower()

                if choice == "q":
                    typer.secho("Aborted.", fg=typer.colors.YELLOW)
                    raise typer.Exit(code=0)

                if choice == "a":
                    break  # proceed to commit

                if choice == "e":
                    confirmed = _edit_capture_fields(confirmed)

                    # Re-resolve zone if the owner edited it (M4) — EXACT match only.
                    if "_edited_zone" in confirmed:
                        edited_zone_name = confirmed.pop("_edited_zone")
                        # H2: an explicit --zone is authoritative and must NOT be
                        # overridden by an interactive edit. Resolve the edited value
                        # for display, but only let it change resolved_zone_id when
                        # --zone was NOT passed.
                        if zone_explicit:
                            typer.secho(
                                f"Note: --zone {zone!r} is authoritative — the edited "
                                f"zone {edited_zone_name!r} will NOT override it.",
                                fg=typer.colors.YELLOW,
                            )
                        else:
                            resolved_zone_id = _resolve_zone_id(
                                conn, edited_zone_name, context=" (from edit)"
                            )
                            confirmed["suggested_zone"] = edited_zone_name

                    # Re-render after edit so owner sees changes.
                    _render_capture_fields_table(confirmed, str(validated_photo))

        # Step 12: Commit via confirm.commit_capture (dual-write with fail-soft photo).
        from leopard44_kb.capture.confirm import commit_capture

        commit_result = commit_capture(
            conn=conn,
            result=confirmed,
            photo_src=validated_photo,
            zone_id=resolved_zone_id,
            repo_root=Path(_os.getcwd()),
        )

        typer.secho(
            f"Created item id={commit_result.item_id}",
            fg=typer.colors.GREEN,
        )

        if commit_result.warning:
            typer.secho(
                f"Warning: {commit_result.warning}",
                fg=typer.colors.YELLOW,
            )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Migrate sub-app (Plan 13-03, D-15)
# ---------------------------------------------------------------------------

migrate_app = typer.Typer(help="Data migrations for the knowledge base store.")
app.add_typer(migrate_app, name="migrate")


@migrate_app.command("relayer-whatsapp")
def migrate_relayer_whatsapp_cmd(
    source: List[str] = typer.Option(
        [],
        "--source",
        help="Source ID or path to re-layer. Repeat to name multiple sources.",
    ),
    all_whatsapp: bool = typer.Option(
        False,
        "--all-whatsapp",
        help="Re-layer ALL whatsapp/vessel sources. Requires --yes (privacy gate).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm bulk re-layer (required with --all-whatsapp).",
    ),
) -> None:
    """Re-layer a WhatsApp owners' group export from vessel → community scope.

    PRIVATE SAFETY DEFAULT: calling this command with NO arguments lists the
    available vessel-layer WhatsApp sources and exits WITHOUT changing anything.
    Nothing auto-promotes — the discriminator is YOUR explicit --source selection.

    USAGE PATTERNS:

      l44 migrate relayer-whatsapp
          → lists whatsapp/vessel candidates and exits (safe inspect, no change)

      l44 migrate relayer-whatsapp --source <id>
      l44 migrate relayer-whatsapp --source <path>
          → re-layers ONLY the named source(s) from vessel to community

      l44 migrate relayer-whatsapp --all-whatsapp --yes
          → re-layers ALL whatsapp/vessel sources (requires explicit --yes;
            prints a privacy warning and aborts without --yes)

    A backup copy of the store is written as <store>.pre-d15.bak before any
    move (skipped for in-memory / :memory: stores used in tests).

    WHEN TO USE --layer community for new ingests:
        Only ingest the PUBLIC owners'-group WhatsApp export with --layer community.
        Your private boat WhatsApp should always be ingested with --layer vessel
        (the default) so it stays private and out of the community scope.
    """
    import leopard44_kb.migrate as _migrate  # noqa: PLC0415
    from leopard44_kb.store import open_db  # noqa: PLC0415

    # Resolve db path (read env at call time for test isolation)
    import os as _os  # noqa: PLC0415
    db_path = Path(
        _os.environ.get("L44_DB")
        or (Path.home() / ".local" / "share" / "leopard44-kb" / "store.db")
    )
    db_path_str = str(db_path)

    conn = open_db(db_path)

    try:
        candidates = _migrate.whatsapp_vessel_candidates(conn)

        # ---- No args: list candidates and exit (SAFE DEFAULT) ----
        if not source and not all_whatsapp:
            if not candidates:
                typer.echo("No whatsapp/vessel sources found (nothing to re-layer).")
                raise typer.Exit(code=0)
            typer.echo("WhatsApp sources currently in 'vessel' layer (candidates for re-layer):\n")
            typer.echo(f"  {'ID':>4}  {'Chunks':>6}  Path")
            typer.echo(f"  {'--':>4}  {'------':>6}  ----")
            for cand in candidates:
                typer.echo(
                    f"  {cand['source_id']:>4}  {cand['chunk_count']:>6}  {cand['path']}"
                )
            typer.echo(
                "\nTo re-layer a specific source:\n"
                "  l44 migrate relayer-whatsapp --source <id|path>\n"
                "\nIMPORTANT: only re-layer the PUBLIC owners'-group export.\n"
                "Private boat WhatsApp exports must stay in 'vessel' scope."
            )
            raise typer.Exit(code=0)

        # ---- --all-whatsapp without --yes: privacy warning and abort ----
        if all_whatsapp and not yes:
            typer.secho(
                "PRIVACY WARNING: --all-whatsapp would promote ALL whatsapp/vessel sources "
                "to the 'community' layer, including any PRIVATE boat WhatsApp exports. "
                "This cannot be undone without restoring from the .pre-d15.bak backup.\n"
                "If you are certain, re-run with --all-whatsapp --yes.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(code=1)

        # ---- Resolve source IDs to move ----
        if all_whatsapp:
            source_ids_to_move = [c["source_id"] for c in candidates]
            if not source_ids_to_move:
                typer.echo("No whatsapp/vessel sources found — nothing to move.")
                raise typer.Exit(code=0)
        else:
            # Resolve --source values: numeric IDs or path strings
            candidate_by_id = {c["source_id"]: c for c in candidates}
            candidate_by_path = {c["path"]: c for c in candidates}
            source_ids_to_move: list[int] = []
            unresolved: list[str] = []
            for s in source:
                try:
                    sid = int(s)
                    if sid in candidate_by_id:
                        source_ids_to_move.append(sid)
                    else:
                        unresolved.append(s)
                except ValueError:
                    # Try as path
                    if s in candidate_by_path:
                        source_ids_to_move.append(candidate_by_path[s]["source_id"])
                    else:
                        unresolved.append(s)

            if unresolved:
                typer.secho(
                    f"Could not resolve --source values as whatsapp/vessel candidates: "
                    f"{', '.join(unresolved)}\n"
                    "Run `l44 migrate relayer-whatsapp` (no args) to list valid sources.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)

            if not source_ids_to_move:
                typer.echo("No matching whatsapp/vessel sources found for the given --source values.")
                raise typer.Exit(code=0)

        # ---- Write backup before any move ----
        if db_path_str != ":memory:":
            # WR-03: timestamp the backup so sequential partial migrations
            # (--source A then --source B) never overwrite an earlier recovery point.
            from datetime import datetime, timezone  # noqa: PLC0415

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            bak_path = db_path.with_suffix(db_path.suffix + f".pre-d15.{stamp}.bak")
            typer.echo(f"Writing backup to: {bak_path}")
            # CR-02: open_db() uses WAL mode, so a bare file copy can silently miss
            # committed data still resident in the -wal sidecar (the close-time
            # checkpoint is best-effort). This .bak is the documented sole recovery
            # path for an irreversible promotion, so take a consistent, WAL-aware
            # snapshot via the SQLite Online Backup API on the live connection.
            import sqlite3 as _sqlite3  # noqa: PLC0415

            try:
                bak_conn = _sqlite3.connect(str(bak_path))
                try:
                    conn.backup(bak_conn)
                finally:
                    bak_conn.close()
                typer.echo(f"Backup written: {bak_path}")
            except (OSError, _sqlite3.Error) as exc:
                # Remove a partial/half-written backup so it can't be mistaken for a
                # valid recovery point.
                try:
                    bak_path.unlink(missing_ok=True)
                except OSError:
                    pass
                typer.secho(
                    f"Error writing backup: {exc}\nAborting — store unchanged.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1) from exc

        # ---- Perform the re-layer ----
        moved = _migrate.relayer_sources_to_community(conn, source_ids_to_move)
        if moved == 0:
            typer.echo("All specified sources already in 'community' layer — no change.")
        else:
            typer.secho(
                f"Re-layered {moved} source(s) from 'vessel' to 'community'.",
                fg=typer.colors.GREEN,
            )

    finally:
        conn.close()


def main() -> None:
    app()
