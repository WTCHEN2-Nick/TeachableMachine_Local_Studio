"""Student tool: one-time setup for flashing the NuMaker-VoiceAI-M55M1 board.

Only students who own this specific board need to run this. Studio's base install does not
bundle `pyocd` or the Nuvoton device pack it needs, because most students have no VoiceAI
board and should not carry either dependency. This script installs `pyocd` into the project's
own `.venv` and downloads the Nuvoton `NuMicroM55_DFP` CMSIS-Pack, verifying its SHA-256
before the file lands, so `tm_local/mcu/toolchain.py`'s probes
(`probe_pyocd()` / `probe_numicro_pack()`) find both without any PATH or environment-variable
setup.

Usage:
  .venv\\Scripts\\python.exe scripts\\setup_voiceai_flash.py

Only runs the network (and installs a package into `.venv`) when a student with this board
invokes it directly; `01_INSTALL.bat` does not call this.

Pre-ship gate note: this environment has no outbound network access to GitHub, so `PACK_URL`
below has never actually been fetched end to end from here -- every development and test run
so far found the pack already present locally (via `TM_NUMICRO_PACK`, `runtime_config.json`,
or a pre-existing Keil/Arm pack cache) and took the "already installed, skip download" branch
in `_fetch_pack()`. `PACK_URL`, `PACK_SHA256` and `PACK_BYTES` were transcribed from the copy
already installed on the development machine and cross-checked against Arm's pack index
descriptor for this exact pack version, not confirmed by watching a real download complete.
The SHA-256 check itself still runs in full and is not weakened by this.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tm_local.mcu.paths import LOCAL_NUMICRO_PACK_ROOT
from tm_local.mcu.toolchain import probe_numicro_pack, probe_pyocd
from tm_local.runtime_config import load_runtime_config, save_runtime_config

PACK_FILENAME = "Nuvoton.NuMicroM55_DFP.3.1.4-rc.3.pack"
PACK_URL = (
    "https://github.com/OpenNuvoton/cmsis-packs/raw/master/Nuvoton_DFP/" + PACK_FILENAME
)
PACK_SHA256 = "1e230426921ae164cef1730fdc7c3c4037dd5e6578e1ba35e57ac8cfccbf6ac6"
PACK_BYTES = 5142663
_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_TIMEOUT = 300
_USER_AGENT = "TeachableMachine-Local-Studio/2.1.0"


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


def _record_runtime_config(section: str, key: str, value: str) -> None:
    """Merge `runtime_config['mcu'][section][key] = value` without disturbing other keys.

    `probe_pyocd()` and `probe_numicro_pack()` read `mcu.pyocd.exe` and `mcu.numicro_pack.path`
    respectively (confirmed by reading `tm_local/mcu/toolchain.py`, not guessed) -- these are
    the exact leaf keys they look for, alongside whatever `mcu.toolchain` / `mcu.nulink` /
    `mcu.vela` / `mcu.toolkit` `01_INSTALL.bat` already recorded there.
    """

    config = load_runtime_config()
    mcu = config.setdefault("mcu", {})
    if not isinstance(mcu, dict):
        mcu = {}
        config["mcu"] = mcu
    sub = mcu.setdefault(section, {})
    if not isinstance(sub, dict):
        sub = {}
        mcu[section] = sub
    sub[key] = value
    save_runtime_config(config)


def _install_pyocd() -> bool:
    """`pip install pyocd` into this project's own `.venv`. Returns True on success."""

    print("[資訊] 準備安裝 pyocd 到專案的 .venv…")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyocd"], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[錯誤] pyocd 安裝失敗：{exc}", file=sys.stderr)
        print(
            "[提示] 常見原因是網路逾時或暫時連不上 PyPI；請稍後重新執行本腳本再試一次。",
            file=sys.stderr,
        )
        return False

    info: dict[str, Any] = probe_pyocd()
    if not info.get("available"):
        print(
            "[警告] pip 回報安裝完成，但目前仍偵測不到 pyocd；"
            "請確認是否使用專案內的 .venv\\Scripts\\python.exe 執行本腳本。",
            file=sys.stderr,
        )
        return False

    print(f"[OK] pyocd 已安裝：{info['path']}")
    _record_runtime_config("pyocd", "exe", info["path"])
    return True


def _fetch_pack() -> bool:
    """Fetch the Nuvoton NuMicroM55 device pack, verifying SHA-256 before it lands.

    Skips the download outright if `probe_numicro_pack()` already finds a copy anywhere
    Studio looks (env var, runtime_config, this script's own `runtime/numicro-pack/`, or
    Arm's/Keil's pack caches) -- students who already installed Keil or Arm's pack manager for
    another board will often already have this.
    """

    existing = probe_numicro_pack()
    if existing.get("available"):
        print(f"[OK] 本機已有 NuMicroM55 device pack，略過下載：{existing['path']}")
        _record_runtime_config("numicro_pack", "path", existing["path"])
        return True

    destination = LOCAL_NUMICRO_PACK_ROOT / PACK_FILENAME
    staging = destination.with_name(destination.name + ".download")
    try:
        print("[資訊] 準備下載 Nuvoton NuMicroM55 device pack…")
        _download(PACK_URL, staging, label="device pack")

        actual_bytes = staging.stat().st_size
        if actual_bytes != PACK_BYTES:
            raise ValueError(
                f"檔案大小不符：預期 {PACK_BYTES} bytes，實際 {actual_bytes} bytes；"
                "下載可能不完整，請重新執行本腳本。"
            )

        print("[資訊] 驗證 SHA-256…")
        actual = _sha256_of(staging)
        if actual.lower() != PACK_SHA256.lower():
            raise ValueError(
                f"SHA-256 驗證失敗：預期 {PACK_SHA256}，實際 {actual}；"
                "下載可能已毀損或遭竄改，請重新執行本腳本。"
            )
        print("[OK] SHA-256 驗證通過。")
    except Exception as exc:  # noqa: BLE001 - report every failure mode, not just network errors
        staging.unlink(missing_ok=True)
        print(f"[錯誤] device pack 下載失敗：{exc}", file=sys.stderr)
        print(
            "[提示] 請自行取得 Nuvoton.NuMicroM55_DFP.3.1.4-rc.3.pack，"
            "並以環境變數 TM_NUMICRO_PACK 指向該檔案的完整路徑後重新啟動 Studio；"
            "手動燒錄流程不受影響。",
            file=sys.stderr,
        )
        return False

    # Verified before landing: only now does the file take the name probe_numicro_pack() looks for.
    staging.replace(destination)
    print(f"[OK] device pack 已安裝完成：{destination}")
    _record_runtime_config("numicro_pack", "path", str(destination.resolve()))
    return True


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(
        description="安裝 pyocd 並下載 NuMaker-VoiceAI-M55M1 燒錄所需的 Nuvoton device pack。",
    )
    parser.parse_args(argv)

    pyocd_ok = _install_pyocd()
    pack_ok = _fetch_pack()

    if pyocd_ok and pack_ok:
        print("[OK] VoiceAI 燒錄環境設定完成，現在可以在 Studio 中部署到這塊板子。")
        return 0
    print("[警告] 設定未完全成功，請參考上方訊息排除問題；手動燒錄流程仍可使用。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
