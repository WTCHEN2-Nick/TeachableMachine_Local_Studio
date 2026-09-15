"""Student tool: download and install the Arm GNU Toolchain into the Studio's runtime folder.

The Studio builds M55M1 firmware with `arm-none-eabi-gcc` but does not bundle it (GPL-3.0
with the GCC runtime library exception; students install it themselves). This script fetches
the pinned Arm-hosted release, verifies it against Arm's own `.sha256asc` checksum file, and
extracts it so `tm_local/mcu/toolchain.py`'s local-install probe (`LOCAL_TOOLCHAIN_ROOT`) finds
it without any PATH or environment-variable setup.

Usage:
  .venv\\Scripts\\python.exe scripts\\download_arm_toolchain.py
  .venv\\Scripts\\python.exe scripts\\download_arm_toolchain.py --dest D:\\tools\\arm-gnu-toolchain
  .venv\\Scripts\\python.exe scripts\\download_arm_toolchain.py --keep-zip

Only runs the network when a student invokes it directly; 01_INSTALL.bat does not call this.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tm_local.archive import safe_extract_zip
from tm_local.mcu.paths import LOCAL_TOOLCHAIN_ROOT
from tm_local.mcu.toolchain import probe_toolchain

TOOLCHAIN_VERSION = "14.2.rel1"
ARCHIVE_NAME = f"arm-gnu-toolchain-{TOOLCHAIN_VERSION}-mingw-w64-i686-arm-none-eabi.zip"
BASE_URL = f"https://developer.arm.com/-/media/Files/downloads/gnu/{TOOLCHAIN_VERSION}/binrel/"
_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_TIMEOUT = 300
_USER_AGENT = "TeachableMachine-Local-Studio/2.1.0"


def archive_url() -> str:
    return BASE_URL + ARCHIVE_NAME


def checksum_url() -> str:
    return archive_url() + ".sha256asc"


def parse_sha256asc(text: str, archive_name: str) -> str:
    """Parse Arm's `.sha256asc` format: one `<sha256>  <filename>` line (others are ignored)."""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip() == archive_name:
            return parts[0].strip()
    raise ValueError(f"checksum for {archive_name!r} not found in .sha256asc contents")


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, label: str) -> None:
    """Stream `url` to `destination`, printing progress; writes via a `.part` temp file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".part")
    temp.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:
            content_length = response.headers.get("Content-Length")
            total_bytes = int(content_length) if content_length else None
            downloaded = 0
            with temp.open("wb") as output:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        percent = min(100, downloaded * 100 // total_bytes)
                        print(f"\r[下載] {label}：{percent}%", end="", flush=True)
                    else:
                        print(f"\r[下載] {label}：{downloaded // 1024} KB", end="", flush=True)
        print()
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    temp.replace(destination)


def _extract_and_install(archive_path: Path, dest: Path) -> None:
    """Safely extract `archive_path` and move its single top-level folder's contents into `dest`."""

    with tempfile.TemporaryDirectory(dir=str(archive_path.parent), prefix=".arm-toolchain-extract-") as raw_tmp:
        extract_dir = Path(raw_tmp)
        safe_extract_zip(archive_path, extract_dir)
        top_level = [entry for entry in extract_dir.iterdir() if entry.is_dir()]
        if len(top_level) != 1:
            raise RuntimeError(
                f"解壓後預期只有一個頂層資料夾，實際找到 {len(top_level)} 個；"
                "壓縮檔內容可能與預期的 Arm GNU Toolchain 版面不符"
            )
        source_root = top_level[0]
        dest.mkdir(parents=True, exist_ok=True)
        for child in source_root.iterdir():
            target = dest / child.name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            shutil.move(str(child), str(target))

    gcc = dest / "bin" / "arm-none-eabi-gcc.exe"
    if not gcc.is_file():
        raise RuntimeError(f"解壓完成，但找不到 {gcc}；工具鏈可能不完整")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(
        description="下載並安裝 Arm GNU Toolchain 到 Teachable Machine Local Studio 的 runtime 資料夾。",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=LOCAL_TOOLCHAIN_ROOT,
        help=f"安裝目的地（預設：{LOCAL_TOOLCHAIN_ROOT}）",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="安裝完成後保留下載的壓縮檔與校驗檔（預設會刪除）",
    )
    args = parser.parse_args(argv)

    dest = args.dest.resolve()
    work_dir = dest.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_path = work_dir / ARCHIVE_NAME
    checksum_path = work_dir / (ARCHIVE_NAME + ".sha256asc")
    dest_existed_before = dest.exists()

    try:
        print(f"[資訊] 準備下載 Arm GNU Toolchain {TOOLCHAIN_VERSION}…")
        _download(archive_url(), archive_path, label="工具鏈壓縮檔")
        _download(checksum_url(), checksum_path, label="SHA-256 校驗檔")

        print("[資訊] 驗證 SHA-256…")
        expected = parse_sha256asc(checksum_path.read_text(encoding="utf-8"), ARCHIVE_NAME)
        actual = _sha256_of(archive_path)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"SHA-256 驗證失敗：預期 {expected}，實際 {actual}；下載可能已毀損或遭竄改，請重新執行本腳本。"
            )
        print("[OK] SHA-256 驗證通過。")

        print(f"[資訊] 解壓並安裝到 {dest} …")
        _extract_and_install(archive_path, dest)
        print(f"[OK] Arm GNU Toolchain 已安裝完成：{dest}")

        if not args.keep_zip:
            archive_path.unlink(missing_ok=True)
            checksum_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - top-level cleanup must catch any failure mode
        print(f"[錯誤] 安裝失敗：{exc}", file=sys.stderr)
        archive_path.unlink(missing_ok=True)
        checksum_path.unlink(missing_ok=True)
        if not dest_existed_before and dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        return 1

    info = probe_toolchain()
    if info.get("available"):
        print(f"[OK] 偵測到 Arm GNU Toolchain：{info.get('version')}（路徑：{info.get('bin_dir')}）")
    else:
        print("[警告] 安裝完成，但目前仍偵測不到工具鏈；請確認 --dest 路徑是否正確，或重新啟動 Studio 服務後再試一次。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
