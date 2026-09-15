from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts import download_arm_toolchain as dl


def test_urls() -> None:
    assert dl.archive_url().endswith(
        "/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-mingw-w64-i686-arm-none-eabi.zip"
    )
    assert dl.checksum_url() == dl.archive_url() + ".sha256asc"


def test_parse_sha256asc() -> None:
    text = "abc123  arm-gnu-toolchain-14.2.rel1-mingw-w64-i686-arm-none-eabi.zip\n"
    assert dl.parse_sha256asc(text, dl.ARCHIVE_NAME) == "abc123"
    with pytest.raises(ValueError):
        dl.parse_sha256asc("zzz  other.zip\n", dl.ARCHIVE_NAME)


def _build_fake_archive(zip_path: Path) -> None:
    """A tiny zip that mimics the real archive's top-level-folder + bin/ layout."""

    root = f"arm-gnu-toolchain-{dl.TOOLCHAIN_VERSION}-mingw-w64-i686-arm-none-eabi"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{root}/bin/arm-none-eabi-gcc.exe", b"fake-gcc-binary")
        zf.writestr(f"{root}/lib/README.txt", b"placeholder\n")


def test_extract_and_install_moves_top_level_contents(tmp_path: Path) -> None:
    archive_path = tmp_path / "downloads" / dl.ARCHIVE_NAME
    _build_fake_archive(archive_path)
    dest = tmp_path / "runtime" / "arm-gnu-toolchain"

    dl._extract_and_install(archive_path, dest)

    gcc = dest / "bin" / "arm-none-eabi-gcc.exe"
    assert gcc.is_file()
    assert gcc.read_bytes() == b"fake-gcc-binary"
    assert (dest / "lib" / "README.txt").is_file()


def test_extract_and_install_rejects_unexpected_layout(tmp_path: Path) -> None:
    archive_path = tmp_path / dl.ARCHIVE_NAME
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("one/marker.txt", b"a\n")
        zf.writestr("two/marker.txt", b"b\n")
    dest = tmp_path / "dest"

    with pytest.raises(RuntimeError):
        dl._extract_and_install(archive_path, dest)

    assert not dest.exists() or not (dest / "bin" / "arm-none-eabi-gcc.exe").exists()


def test_main_cleans_up_on_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "runtime" / "arm-gnu-toolchain"

    def fake_download(url: str, destination: Path, *, label: str) -> None:
        if destination.name == dl.ARCHIVE_NAME:
            _build_fake_archive(destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"deadbeef  {dl.ARCHIVE_NAME}\n", encoding="utf-8")

    monkeypatch.setattr(dl, "_download", fake_download)
    monkeypatch.setattr(dl, "probe_toolchain", lambda: {"available": False})

    exit_code = dl.main(["--dest", str(dest)])

    assert exit_code != 0
    assert not dest.exists()
    assert not (tmp_path / "runtime" / dl.ARCHIVE_NAME).exists()
    assert not (tmp_path / "runtime" / (dl.ARCHIVE_NAME + ".sha256asc")).exists()


def test_main_succeeds_with_matching_checksum(tmp_path: Path, monkeypatch) -> None:
    import hashlib

    dest = tmp_path / "runtime" / "arm-gnu-toolchain"

    def fake_download(url: str, destination: Path, *, label: str) -> None:
        if destination.name == dl.ARCHIVE_NAME:
            _build_fake_archive(destination)
        else:
            archive = destination.parent / dl.ARCHIVE_NAME
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"{digest}  {dl.ARCHIVE_NAME}\n", encoding="utf-8")

    probed = {}

    def fake_probe() -> dict:
        probed["called"] = True
        return {"available": True, "bin_dir": str(dest / "bin"), "version": "fake 14.2.1", "source": "local",
                "checked": []}

    monkeypatch.setattr(dl, "_download", fake_download)
    monkeypatch.setattr(dl, "probe_toolchain", fake_probe)

    exit_code = dl.main(["--dest", str(dest), "--keep-zip"])

    assert exit_code == 0
    assert (dest / "bin" / "arm-none-eabi-gcc.exe").is_file()
    assert probed.get("called") is True
    # --keep-zip must leave the downloaded archive and checksum file in place.
    assert (tmp_path / "runtime" / dl.ARCHIVE_NAME).exists()
    assert (tmp_path / "runtime" / (dl.ARCHIVE_NAME + ".sha256asc")).exists()


def test_main_without_keep_zip_removes_downloads(tmp_path: Path, monkeypatch) -> None:
    import hashlib

    dest = tmp_path / "runtime" / "arm-gnu-toolchain"

    def fake_download(url: str, destination: Path, *, label: str) -> None:
        if destination.name == dl.ARCHIVE_NAME:
            _build_fake_archive(destination)
        else:
            archive = destination.parent / dl.ARCHIVE_NAME
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"{digest}  {dl.ARCHIVE_NAME}\n", encoding="utf-8")

    monkeypatch.setattr(dl, "_download", fake_download)
    monkeypatch.setattr(dl, "probe_toolchain", lambda: {"available": False, "bin_dir": None, "version": None,
                                                          "source": None, "checked": []})

    exit_code = dl.main(["--dest", str(dest)])

    assert exit_code == 0
    assert (dest / "bin" / "arm-none-eabi-gcc.exe").is_file()
    assert not (tmp_path / "runtime" / dl.ARCHIVE_NAME).exists()
    assert not (tmp_path / "runtime" / (dl.ARCHIVE_NAME + ".sha256asc")).exists()
