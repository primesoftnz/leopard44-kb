# RED state until Plan 03 ships the real serve_cmd (see VALIDATION.md).
# Imports from leopard44_kb.cli are safe at module top level (cli.app already exists).
# uvicorn.run is monkeypatched so it never actually blocks.
"""Tests for UI-01: l44 serve command — serve_cmd unit tests via typer CliRunner."""
from __future__ import annotations

import socket

import pytest
from typer.testing import CliRunner

from leopard44_kb.cli import app

# mix_stderr is not supported in typer 0.26 — stderr merged into result.output.
runner = CliRunner()


def _patch_uvicorn_noop(monkeypatch):
    """Patch uvicorn.run to a no-op that captures call arguments."""
    captured: list[dict] = []

    def _noop_run(app_obj, **kwargs):
        captured.append(kwargs)

    import leopard44_kb.cli as cli_module

    # The serve_cmd does lazy import of uvicorn inside its body.
    # We patch the uvicorn module object directly so the lazy import picks it up.
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", _noop_run)
    return captured


def test_serve_no_longer_a_stub(monkeypatch, tmp_path):
    """serve command is real in Phase 5 — does NOT say 'Not yet implemented'.

    Mirrors test_ingest_no_longer_a_stub from test_sources_cli.py.
    Exit code must NOT be 2 (the stub code); 0 is expected after real replacement.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    _patch_uvicorn_noop(monkeypatch)

    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 2, (
        f"serve still behaves as a stub (exit 2); got: {result.output!r}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Not yet implemented" not in combined, (
        f"Stub message still present after Phase 5 replacement: {combined!r}"
    )


def test_host_is_localhost(monkeypatch, tmp_path):
    """Security: l44 serve (no --qr) must bind to 127.0.0.1, never 0.0.0.0.

    This is the security-critical localhost-bind assertion. The default serve MUST
    stay on 127.0.0.1; LAN bind requires explicit --qr opt-in (D-11).
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    captured = _patch_uvicorn_noop(monkeypatch)

    result = runner.invoke(app, ["serve"])  # NO --qr flag
    assert result.exit_code != 2, (
        f"serve without --qr is still a stub (exit 2): {result.output!r}"
    )
    assert len(captured) == 1, (
        f"Expected uvicorn.run to be called exactly once; captured: {captured!r}"
    )
    assert captured[0].get("host") == "127.0.0.1", (
        f"serve without --qr must bind 127.0.0.1; uvicorn.run kwargs: {captured[0]!r}"
    )


def test_qr_binds_lan(monkeypatch, tmp_path):
    """RED — l44 serve --qr must bind 0.0.0.0 (LAN, explicit opt-in) + pass ssl kwargs.

    Fails today because --qr, _detect_lan_ip, and _get_or_create_cert do not exist yet.
    Plan 02 implements these.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    captured = _patch_uvicorn_noop(monkeypatch)

    import leopard44_kb.cli as cli_mod

    # Patch IP detection and cert generation so no real network/crypto happens
    monkeypatch.setattr(cli_mod, "_detect_lan_ip", lambda: "192.168.1.42")

    # Write placeholder cert/key files into tmp_path for the patched cert helper
    cert_file = tmp_path / "c.crt"
    key_file = tmp_path / "k.key"
    cert_file.write_text("CERT")
    key_file.write_text("KEY")
    monkeypatch.setattr(
        cli_mod,
        "_get_or_create_cert",
        lambda ip: (cert_file, key_file),
    )

    # Patch segno.make so no real QR is rendered to stdout
    import segno
    monkeypatch.setattr(
        segno,
        "make",
        lambda url: type("Q", (), {"terminal": lambda self, **kw: None})(),
    )

    result = runner.invoke(app, ["serve", "--qr"])
    assert result.exit_code != 2, (
        f"serve --qr failed unexpectedly (exit 2): {result.output!r}"
    )
    assert len(captured) == 1, (
        f"Expected uvicorn.run called once; captured: {captured!r}"
    )
    assert captured[0].get("host") == "0.0.0.0", (
        f"serve --qr must bind 0.0.0.0; uvicorn.run kwargs: {captured[0]!r}"
    )
    assert "ssl_certfile" in captured[0], (
        f"serve --qr must pass ssl_certfile to uvicorn; got: {captured[0]!r}"
    )
    assert "ssl_keyfile" in captured[0], (
        f"serve --qr must pass ssl_keyfile to uvicorn; got: {captured[0]!r}"
    )


def test_qr_reuses_persisted_cert(monkeypatch, tmp_path):
    """RED — D-12: serve --qr must REUSE a persisted cert, not regenerate on second run.

    The cert file bytes/fingerprint must be identical after a second `serve --qr` at
    the same LAN IP. Fails today because cert persistence does not exist yet.
    Plan 02 implements _get_or_create_cert with data/certs/ persistence.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))

    import leopard44_kb.cli as cli_mod

    # Patch IP detection so both runs use the same LAN IP
    monkeypatch.setattr(cli_mod, "_detect_lan_ip", lambda: "192.168.1.42")

    # Let the REAL _get_or_create_cert run, but redirect data/certs to tmp_path
    # by monkeypatching the certs directory resolution inside cli_mod.
    # The certs home is data/certs/ which is resolved relative to repo_root().
    # We redirect via L44_CERTS_DIR env so the cert lands in tmp_path.
    certs_dir = tmp_path / "certs"
    certs_dir.mkdir()
    monkeypatch.setenv("L44_CERTS_DIR", str(certs_dir))

    # Patch segno and uvicorn so the command exits without blocking
    import segno, uvicorn

    monkeypatch.setattr(
        segno,
        "make",
        lambda url: type("Q", (), {"terminal": lambda self, **kw: None})(),
    )
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

    # First invocation — should create the cert
    result1 = runner.invoke(app, ["serve", "--qr"])
    assert result1.exit_code != 2, f"First serve --qr failed: {result1.output!r}"

    # Capture cert file bytes after first run
    cert_path = certs_dir / "l44-lan.crt"
    assert cert_path.exists(), (
        f"Expected cert file at {cert_path} after first serve --qr run; "
        f"output: {result1.output!r}"
    )
    bytes_run1 = cert_path.read_bytes()

    # Second invocation — cert must be REUSED (bytes identical)
    result2 = runner.invoke(app, ["serve", "--qr"])
    assert result2.exit_code != 2, f"Second serve --qr failed: {result2.output!r}"

    bytes_run2 = cert_path.read_bytes()
    assert bytes_run1 == bytes_run2, (
        "D-12: cert must be persisted and REUSED on second serve --qr at the same IP; "
        "bytes changed, indicating the cert was regenerated (one-time trust broken)"
    )


def test_qr_key_perms_0600(monkeypatch, tmp_path):
    """RED — review residual #6: persisted key file must be owner-only (0600 perms).

    After serve --qr, the private key at data/certs/l44-lan.key must have
    mode 0600. Fails today because the key is not written yet.
    Plan 02 implements _get_or_create_cert with explicit 0600 chmod.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))

    import leopard44_kb.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_detect_lan_ip", lambda: "192.168.1.42")

    certs_dir = tmp_path / "certs"
    certs_dir.mkdir()
    monkeypatch.setenv("L44_CERTS_DIR", str(certs_dir))

    import segno, uvicorn

    monkeypatch.setattr(
        segno,
        "make",
        lambda url: type("Q", (), {"terminal": lambda self, **kw: None})(),
    )
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

    result = runner.invoke(app, ["serve", "--qr"])
    assert result.exit_code != 2, f"serve --qr failed: {result.output!r}"

    key_path = certs_dir / "l44-lan.key"
    assert key_path.exists(), (
        f"Expected private key at {key_path} after serve --qr; "
        f"output: {result.output!r}"
    )
    mode = oct(key_path.stat().st_mode & 0o777)
    assert mode == "0o600", (
        f"Private key must have owner-only permissions (0600); got {mode}"
    )


def test_qr_host_ip_override(monkeypatch, tmp_path):
    """RED — review concern #5: --host-ip override makes QR/cert use the override IP.

    `serve --qr --host-ip 192.0.2.9` must use 192.0.2.9 even if _detect_lan_ip()
    returns something else. Fails today because --host-ip does not exist yet.
    Plan 02 adds the --host-ip option to serve_cmd.
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    captured = _patch_uvicorn_noop(monkeypatch)

    import leopard44_kb.cli as cli_mod

    # _detect_lan_ip returns a DIFFERENT IP to confirm the override takes precedence
    monkeypatch.setattr(cli_mod, "_detect_lan_ip", lambda: "172.18.0.1")

    cert_file = tmp_path / "c.crt"
    key_file = tmp_path / "k.key"
    cert_file.write_text("CERT")
    key_file.write_text("KEY")

    # Capture the IP passed to cert generation to assert the override is used
    used_ips: list[str] = []

    def _capture_cert(ip: str):
        used_ips.append(ip)
        return cert_file, key_file

    monkeypatch.setattr(cli_mod, "_get_or_create_cert", _capture_cert)

    import segno

    monkeypatch.setattr(
        segno,
        "make",
        lambda url: type("Q", (), {"terminal": lambda self, **kw: None})(),
    )

    result = runner.invoke(app, ["serve", "--qr", "--host-ip", "192.0.2.9"])
    assert result.exit_code != 2, (
        f"serve --qr --host-ip failed unexpectedly: {result.output!r}"
    )
    assert "192.0.2.9" in used_ips, (
        f"--host-ip 192.0.2.9 must be passed to cert generation; got IPs: {used_ips}"
    )


def test_prints_url(monkeypatch, tmp_path):
    """serve command prints the bound URL containing 'http://127.0.0.1:' and a port."""
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    _patch_uvicorn_noop(monkeypatch)

    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 2, (
        f"serve still a stub (exit 2): {result.output!r}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "http://127.0.0.1:" in combined, (
        f"Expected URL 'http://127.0.0.1:<port>' in output; got: {combined!r}"
    )


def test_port_autoincrements_when_busy(monkeypatch, tmp_path):
    """When default port (8000) is busy, serve uses the next available port.

    Binds a socket on port 8000 before invoking serve, then asserts the
    captured uvicorn.run port is > 8000 (RESEARCH Pattern 2: port pre-probe).
    """
    monkeypatch.setenv("L44_DB", str(tmp_path / "s.db"))
    captured = _patch_uvicorn_noop(monkeypatch)

    # Occupy port 8000 for the duration of the test
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        blocker.bind(("127.0.0.1", 8000))
        blocker.listen(1)

        result = runner.invoke(app, ["serve"])
        assert result.exit_code != 2, (
            f"serve still a stub (exit 2): {result.output!r}"
        )
        assert len(captured) == 1, (
            f"Expected uvicorn.run called once; captured: {captured!r}"
        )
        used_port = captured[0].get("port")
        assert used_port is not None and used_port > 8000, (
            f"Expected port > 8000 when 8000 is busy; got port={used_port}"
        )
    finally:
        blocker.close()


def test_cert_is_ios_compatible(monkeypatch, tmp_path):
    """The self-signed cert must satisfy Apple's TLS-server-cert policy.

    iOS Safari REJECTS the connection (errors after "Visit this Website") if the
    validity span exceeds 825 days or the serverAuth EKU is missing. A real-iPhone
    field UAT (2026-06-16) surfaced a 10-year/no-EKU cert that errored; this guards
    the regression. No dev-box client (curl/openssl/headless Chromium with
    ignore_https_errors) enforces this, so the assertion encodes the contract.
    """
    from cryptography.x509 import load_pem_x509_certificate, ExtendedKeyUsage
    from cryptography.x509.oid import ExtendedKeyUsageOID

    import leopard44_kb.cli as cli_mod

    certs_dir = tmp_path / "certs"
    certs_dir.mkdir()
    monkeypatch.setenv("L44_CERTS_DIR", str(certs_dir))

    cert_path, _ = cli_mod._get_or_create_cert("192.168.1.42")
    cert = load_pem_x509_certificate(cert_path.read_bytes())

    span_days = (cert.not_valid_after - cert.not_valid_before).days
    assert span_days <= 825, (
        f"iOS rejects certs with validity > 825 days; got {span_days}"
    )
    eku = cert.extensions.get_extension_for_class(ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku, (
        "iOS requires the serverAuth EKU on TLS server certs"
    )


def test_cert_noncompliant_is_regenerated(monkeypatch, tmp_path):
    """An existing non-compliant cert (long validity / no EKU) must auto-heal.

    The reuse path returns the persisted cert only when it is still iOS-compatible,
    so an owner who already generated a pre-fix 10-year/no-EKU cert gets a compliant
    one on the next `serve --qr` without manual deletion.
    """
    import datetime as _dt
    import ipaddress
    from cryptography import x509
    from cryptography.x509 import load_pem_x509_certificate, ExtendedKeyUsage
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    import leopard44_kb.cli as cli_mod

    certs_dir = tmp_path / "certs"
    certs_dir.mkdir()
    monkeypatch.setenv("L44_CERTS_DIR", str(certs_dir))

    # Forge a pre-fix cert: 10-year validity, no EKU, but the right SAN.
    target = "192.168.1.42"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "l44-local")])
    now = _dt.datetime.now(_dt.timezone.utc)
    old = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(now + _dt.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.IPv4Address(target))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    (certs_dir / "l44-lan.crt").write_bytes(old.public_bytes(serialization.Encoding.PEM))
    (certs_dir / "l44-lan.key").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )

    cert_path, _ = cli_mod._get_or_create_cert(target)
    cert = load_pem_x509_certificate(cert_path.read_bytes())

    span_days = (cert.not_valid_after - cert.not_valid_before).days
    assert span_days <= 825, "non-compliant 10-year cert should have been regenerated"
    eku = cert.extensions.get_extension_for_class(ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku, "regenerated cert must carry serverAuth EKU"
