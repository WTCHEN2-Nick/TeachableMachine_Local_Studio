from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from scripts import setup_voiceai_flash as flash


def test_fetch_pack_cleans_up_on_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    """A bad hash must leave no file at the destination name `probe_numicro_pack()`'s glob
    matches -- landing a corrupted/tampered pack there would make every later run trust it."""

    local_root = tmp_path / "numicro-pack"
    monkeypatch.setattr(flash, "LOCAL_NUMICRO_PACK_ROOT", local_root)
    monkeypatch.setattr(flash, "probe_numicro_pack", lambda: {"available": False, "path": None})
    recorded: list[tuple] = []
    monkeypatch.setattr(flash, "_record_runtime_config", lambda *args: recorded.append(args))

    def fake_download(url: str, destination: Path, *, label: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"not the real pack")

    monkeypatch.setattr(flash, "_download", fake_download)

    ok = flash._fetch_pack()

    destination = local_root / flash.PACK_FILENAME
    staging = destination.with_name(destination.name + ".download")
    assert ok is False
    assert not destination.exists()
    assert not staging.exists()
    assert recorded == []


def test_fetch_pack_lands_file_on_matching_checksum(tmp_path: Path, monkeypatch) -> None:
    """Verified before landing: once the staged file's hash matches, it takes the real name."""

    local_root = tmp_path / "numicro-pack"
    monkeypatch.setattr(flash, "LOCAL_NUMICRO_PACK_ROOT", local_root)
    monkeypatch.setattr(flash, "probe_numicro_pack", lambda: {"available": False, "path": None})
    recorded: list[tuple] = []
    monkeypatch.setattr(flash, "_record_runtime_config", lambda *args: recorded.append(args))

    content = b"fake pack bytes standing in for the real download"

    def fake_download(url: str, destination: Path, *, label: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    monkeypatch.setattr(flash, "_download", fake_download)
    monkeypatch.setattr(flash, "PACK_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(flash, "PACK_BYTES", len(content))

    ok = flash._fetch_pack()

    destination = local_root / flash.PACK_FILENAME
    staging = destination.with_name(destination.name + ".download")
    assert ok is True
    assert destination.is_file()
    assert destination.read_bytes() == content
    assert not staging.exists()
    assert recorded == [("numicro_pack", "path", str(destination.resolve()))]


def test_fetch_pack_rejects_wrong_size_before_hashing(tmp_path: Path, monkeypatch) -> None:
    """PACK_BYTES is checked before the SHA-256 pass -- a truncated download must not even
    reach the (more expensive) hash step, and must not land at the destination name."""

    local_root = tmp_path / "numicro-pack"
    monkeypatch.setattr(flash, "LOCAL_NUMICRO_PACK_ROOT", local_root)
    monkeypatch.setattr(flash, "probe_numicro_pack", lambda: {"available": False, "path": None})
    recorded: list[tuple] = []
    monkeypatch.setattr(flash, "_record_runtime_config", lambda *args: recorded.append(args))

    content = b"too short"

    def fake_download(url: str, destination: Path, *, label: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    monkeypatch.setattr(flash, "_download", fake_download)
    # Deliberately matches the wrong content: a real mismatch would fail on size long before
    # a SHA-256 comparison would ever be reached.
    monkeypatch.setattr(flash, "PACK_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(flash, "PACK_BYTES", len(content) + 1)

    ok = flash._fetch_pack()

    destination = local_root / flash.PACK_FILENAME
    staging = destination.with_name(destination.name + ".download")
    assert ok is False
    assert not destination.exists()
    assert not staging.exists()
    assert recorded == []


def test_install_pyocd_returns_false_on_pip_failure(monkeypatch) -> None:
    """A failed `pip install` is reported, not raised -- pyocd is optional, so a failure here
    must not stop `main()` from still attempting the pack fetch."""

    def fake_run(cmd, check=True):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(flash.subprocess, "run", fake_run)

    assert flash._install_pyocd() is False
