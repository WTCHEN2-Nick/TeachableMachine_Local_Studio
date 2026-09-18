from __future__ import annotations

import io
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts import check_publish as cp


def _home(user: str, sep: str = "\\", drive: str = "C:") -> bytes:
    """A home-directory path, assembled at run time so this file never contains one."""
    return f"{drive}{sep}Users{sep}{user}".encode()


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_flags_the_home_directory_of_a_real_user() -> None:
    findings = cp.find_leaks("docs/x.md", b"cd " + _home("jdoe") + b"\\Desktop\\proj")
    assert len(findings) == 1
    assert "docs/x.md" in findings[0]


def test_ignores_documentation_placeholders() -> None:
    text = "例如 C:\\Users\\你的名字\\Desktop 或 C:\\Users\\yourname\\Desktop".encode()
    assert cp.find_leaks("manual.md", text) == []


def test_ignores_shared_windows_profile_folders() -> None:
    assert cp.find_leaks("a.txt", b"C:\\Users\\Public\\Documents C:\\Users\\Default\\x") == []


def test_flags_forward_slash_json_escaped_and_lowercase_variants() -> None:
    for text in (_home("jdoe", "/"), _home("jdoe", "\\\\"), _home("jdoe").lower()):
        assert cp.find_leaks("f", text), text


def test_flags_another_persons_home_directory() -> None:
    assert cp.find_leaks("build.log", _home("alice", drive="D:") + b"\\proj\\main.c")


def test_does_not_flag_urls_or_program_paths() -> None:
    text = b"https://github.com/octocat/Hello-World and C:\\Program Files\\Git"
    assert cp.find_leaks("README.md", text) == []


def test_flags_a_per_user_keil_layout_file_by_name() -> None:
    findings = cp.find_leaks("fw/Keil/proj.uvguix.jdoe", b"clean content")
    assert len(findings) == 1


def test_scans_the_contents_and_names_inside_a_zip() -> None:
    data = _zip({
        "proj/main.c": b"int main(void) { return 0; }",
        "proj/Objects/main.d": _home("jdoe") + b"\\proj\\main.c",
        "proj/Keil/proj.uvguix.jdoe": b"layout",
    })
    findings = cp.find_leaks("fw.zip", data)
    assert len(findings) == 2
    assert all("fw.zip" in f for f in findings)
    assert any("Objects/main.d" in f for f in findings)
    assert any("uvguix" in f for f in findings)


def test_a_clean_zip_passes() -> None:
    assert cp.find_leaks("fw.zip", _zip({"proj/main.c": b"int x;"})) == []


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def _stage(repo: Path, name: str, data: bytes) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    subprocess.run(["git", "add", "--", name], cwd=repo, check=True)


def test_staged_scan_checks_what_will_be_committed_not_the_working_tree(repo: Path) -> None:
    _stage(repo, "notes.txt", _home("jdoe") + b"\\secret")
    (repo / "notes.txt").write_bytes(b"cleaned after staging, but the index still leaks")
    assert cp.main(["--staged"], repo=repo) == 1


def test_staged_scan_passes_when_nothing_leaks(repo: Path) -> None:
    _stage(repo, "notes.txt", b"C:\\Users\\yourname\\example")
    assert cp.main(["--staged"], repo=repo) == 0


def test_staged_scan_blocks_a_file_github_would_reject_for_size(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cp, "GITHUB_MAX_FILE_BYTES", 1000)
    _stage(repo, "installer.exe", b"\0" * 1001)
    assert cp.main(["--staged"], repo=repo) == 1


def test_staged_scan_allows_a_file_at_the_size_limit(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cp, "GITHUB_MAX_FILE_BYTES", 1000)
    _stage(repo, "installer.exe", b"\0" * 1000)
    assert cp.main(["--staged"], repo=repo) == 0


def test_staged_scan_handles_paths_with_spaces_parentheses_and_cjk(repo: Path) -> None:
    _stage(repo, "板子 (M5531)/fw.zip", _zip({"p/x.d": _home("jdoe", "/") + b"/p"}))
    assert cp.main(["--staged"], repo=repo) == 1
