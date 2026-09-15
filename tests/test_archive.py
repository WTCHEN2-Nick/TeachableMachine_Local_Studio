from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tm_local.archive import ArchiveError, safe_extract_zip


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")
    with pytest.raises(ArchiveError):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_rejects_windows_drive(tmp_path: Path) -> None:
    archive = tmp_path / "bad_drive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("C:/escape.txt", "bad")
    with pytest.raises(ArchiveError):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_normal_archive(tmp_path: Path) -> None:
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("folder/value.txt", "ok")
    output = tmp_path / "out"
    safe_extract_zip(archive, output)
    assert (output / "folder" / "value.txt").read_text() == "ok"
