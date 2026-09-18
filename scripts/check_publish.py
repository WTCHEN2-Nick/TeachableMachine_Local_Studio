"""Maintainer tool: refuse to commit anything that would publish a Windows home directory.

The public repository is a plain mirror of this folder, so whatever gets committed is
published. This check is wired in as the repository's pre-commit hook
(`scripts/git-hooks/pre-commit`, enabled with `git config core.hooksPath scripts/git-hooks`),
which means it also runs when committing from GitHub Desktop.

It reads what is *staged* -- the exact bytes that would go into the commit, not the working
tree -- and looks inside `.zip` files as well. Firmware projects zipped straight out of Keil
or VS Code carry build output (`.o`, `.d`, `.lst`, CMake/ninja state) full of absolute paths
such as `C:\\Users\\<name>\\Desktop\\...`, and Keil writes a per-user layout file named
`<project>.uvguix.<windows user name>`.

Usage:
  python scripts\\check_publish.py --staged    what the hook runs
  python scripts\\check_publish.py --all       every tracked file

Documentation placeholders (`C:\\Users\\yourname\\...`, `C:\\Users\\你的名字\\...`) and the
shared profile folders (`Public`, `Default`) are not reported.

It also refuses any single file over GitHub's 100 MiB limit. GitHub rejects such a push, and
because the rejected commit stays in local history, every push after it fails too -- far
easier to stop at commit time than to rewrite history afterwards.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

# X:\Users\<name> with \, /, or escaped \\ separators; the name stops at the next separator.
_HOME_DIR = re.compile(rb"(?<![A-Za-z])[A-Za-z]:[\\/]+Users[\\/]+([^\\/\s\"'<>|:*?\x00]+)", re.I)
_KEIL_LAYOUT = re.compile(r"\.uvguix\.([^/\\]+)$", re.I)
_NOT_A_PERSON = {
    "yourname", "username", "user", "你的名字", "使用者名稱",
    "public", "default", "all", "%username%",
}
_MAX_LINES = 30
# GitHub rejects any single file over 100 MiB on push; committing one strands every later push.
GITHUB_MAX_FILE_BYTES = 100 * 1024 * 1024


def _personal(name: bytes | str) -> bool:
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    return name.lower() not in _NOT_A_PERSON


def _describe(name: str, data: bytes) -> str | None:
    hits: list[str] = []
    layout = _KEIL_LAYOUT.search(name)
    if layout and _personal(layout.group(1)):
        hits.append("Keil per-user layout file")
    for label in (name.encode("utf-8"), data):
        matches = [m for m in _HOME_DIR.finditer(label) if _personal(m.group(1))]
        if matches:
            sample = matches[0].group(0).decode("utf-8", "replace")
            hits.append(sample if len(matches) == 1 else f"{sample} (x{len(matches)})")
    return f"{name}: " + "; ".join(hits) if hits else None


def find_leaks(name: str, data: bytes) -> list[str]:
    """One finding per offending file; a zip reports each offending entry instead."""
    if name.lower().endswith(".zip"):
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            pass
        else:
            findings = [f for f in [_describe(name, b"")] if f]
            with archive:
                for info in archive.infolist():
                    if not info.is_dir():
                        inner = find_leaks(info.filename, archive.read(info))
                        findings += [f"{name} -> {f}" for f in inner]
            return findings
    found = _describe(name, data)
    return [found] if found else []


def _git(repo: Path, *args: str) -> list[bytes]:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True).stdout
    return [p for p in out.split(b"\0") if p]


def _read_index(repo: Path, paths: list[bytes]) -> Iterator[tuple[str, bytes]]:
    """Stream staged blobs through one `git cat-file --batch` instead of a process per file."""
    with subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    ) as proc:
        assert proc.stdin is not None and proc.stdout is not None
        for path in paths:
            proc.stdin.write(b":" + path + b"\n")
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            if header[-1] == b"missing":
                continue
            data = proc.stdout.read(int(header[2]))
            proc.stdout.read(1)
            yield path.decode("utf-8", "replace"), data
        proc.stdin.close()


def main(argv: list[str] | None = None, *, repo: Path | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="擋下會公開電腦使用者路徑的 commit。")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--staged", action="store_true", help="只檢查這次要 commit 的檔案（預設）")
    scope.add_argument("--all", action="store_true", help="檢查所有已追蹤的檔案")
    args = parser.parse_args(argv)

    repo = repo or Path.cwd()
    if args.all:
        paths = _git(repo, "ls-files", "-z")
    else:
        paths = _git(repo, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    too_big: list[str] = []
    leaks: list[str] = []
    for name, data in _read_index(repo, paths):
        if len(data) > GITHUB_MAX_FILE_BYTES:
            too_big.append(f"{name}: {len(data) / 2**20:.1f} MB，超過 GitHub 單檔 100 MB 上限")
            continue
        leaks += find_leaks(name, data)
    if not too_big and not leaks:
        return 0

    findings = too_big + leaks
    print("[check_publish] 這次 commit 不能發佈到 GitHub，已擋下：")
    for line in findings[:_MAX_LINES]:
        print(f"  {line}")
    if len(findings) > _MAX_LINES:
        print(f"  ...另外還有 {len(findings) - _MAX_LINES} 處")
    if too_big:
        print("超過 100 MB 的檔案 push 一定會被 GitHub 拒絕：請把它加進 .gitignore"
              "（大型安裝檔改放 GitHub Releases）。")
    if leaks:
        print("含有電腦使用者路徑的檔案：請移除或改掉內容"
              "（GitHub Desktop：在左側取消勾選該檔案）。")
    print("確定是誤判才用：git commit --no-verify")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
