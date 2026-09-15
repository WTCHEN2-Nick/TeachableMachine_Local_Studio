"""Flashing a built firmware image onto a board: USB MSC drag-and-drop, Nu-Link and pyocd.

Nothing here builds anything -- `deploy_service`/`gcc_build` do that. This module only moves
`firmware.bin` onto the board: by copying it onto a bootloader's USB mass-storage drive
(NuGestureAI-M55M1), by driving the Nu-Link Command Tool (NuMaker-M55M1), or by driving pyocd
against the Nuvoton DFP pack (NuMaker-VoiceAI-M55M1, which has no on-board Nu-Link and whose
silicon the bundled Nu-Link Command Tool does not recognise at all). Kept stdlib-only and
cheap to import: `ctypes` is imported lazily inside the Windows-only helpers below, not at
module scope, so importing this module (even on a machine with no board plugged in) never
touches `ctypes.windll`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .gcc_build import run_tool

# Case-sensitive: a benign log line like "0 failures" must not turn a successful flash into a
# 400, so no bare lowercase "fail" here -- only the Nu-Link Command Tool's actual failure text.
_FAILURE_MARKERS = ("Not Supported", ">>> ERROR", "Fail")
_LOG_TAIL_CHARS = 4000


def _is_windows() -> bool:
    """A seam so tests can force the non-Windows path without mutating `sys.platform`."""
    return sys.platform == "win32"


def _volume_label(root: str) -> str | None:
    """The FAT volume label of `root` (e.g. `"C:\\\\"`), or `None` if it cannot be read."""
    if not _is_windows():
        return None
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(261)
        fs_buffer = ctypes.create_unicode_buffer(261)
        serial = ctypes.c_uint32()
        max_len = ctypes.c_uint32()
        flags = ctypes.c_uint32()
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(root),
            buffer,
            261,
            ctypes.byref(serial),
            ctypes.byref(max_len),
            ctypes.byref(flags),
            fs_buffer,
            261,
        )
        return buffer.value if ok else None
    except Exception:  # noqa: BLE001 - probing a drive must never raise
        return None


def find_msc_drives(label: str = "M55M1") -> list[str]:
    """Removable drive roots (e.g. `["D:\\\\"]`) whose volume label equals `label`.

    Returns `[]` on non-Windows platforms and on any ctypes failure -- this feeds the flash
    endpoint's "please plug in / enter bootloader" guidance, so it must never raise.
    """
    if not _is_windows():
        return []
    try:
        import ctypes

        mask = ctypes.windll.kernel32.GetLogicalDrives()
        found: list[str] = []
        for index in range(26):
            if not mask & (1 << index):
                continue
            root = f"{chr(ord('A') + index)}:\\"
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(root))
            if drive_type != 2:  # DRIVE_REMOVABLE
                continue
            if (_volume_label(root) or "").upper() == label.upper():
                found.append(root)
        return found
    except Exception:  # noqa: BLE001 - probing drives must never raise
        return []


def copy_to_msc(bin_path: Path, drive: str) -> dict:
    """Copy `bin_path` to `<drive>\\firmware.bin` after re-checking the M55M1 volume label.

    Never raises for I/O trouble: any failure (label mismatch, copy error) comes back as
    `{"ok": False, "message": ..., "target": ...}` so the endpoint can surface it as a 400.
    """
    root = drive if drive.endswith(("\\", "/")) else drive + "\\"
    label = _volume_label(root)
    if (label or "").upper() != "M55M1":
        label_desc = f"卷標 {label!r}" if label is not None else "無法讀取卷標"
        return {
            "ok": False,
            "message": f"{drive} 不是 M55M1 隨身碟（{label_desc}）；請先進入 bootloader 模式",
            "target": "",
        }
    target = Path(root) / "firmware.bin"
    try:
        shutil.copyfile(bin_path, target)
    except OSError as exc:
        return {"ok": False, "message": f"複製到 {drive} 失敗：{exc}", "target": str(target)}
    return {
        "ok": True,
        "message": f"已複製 firmware.bin 到 {drive}，請按一下板上的 Reset",
        "target": str(target),
    }


def _nulink_env(exe: Path) -> dict[str, str]:
    """`gcc_build.sanitized_env()`-equivalent contents, rooted at the Nu-Link exe's directory.

    Same shape the GCC build subprocess gets (System32 + SystemRoot on PATH behind the tool's
    own directory, SystemRoot, TEMP/TMP, USERPROFILE, PYTHONUTF8) so the Nu-Link Command Tool
    sees a normal enough Windows environment to resolve its own DLLs and temp files.
    """
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    env = {
        "PATH": os.pathsep.join(
            [str(exe.parent), os.path.join(system_root, "System32"), system_root]
        ),
        "SystemRoot": system_root,
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "USERPROFILE": os.environ.get("USERPROFILE", ""),
        "PYTHONUTF8": "1",
    }
    return {key: value for key, value in env.items() if value}


def nulink_program(bin_path: Path, nulink_exe: Path, timeout_seconds: int = 180) -> dict:
    """Program `bin_path` into APROM via the Nu-Link Command Tool and reset the board.

    Runs `-C` (connect), `-W APROM <bin> 1` (write), `-S` (reset) in order, stopping at the
    first step that fails: a non-zero return code, one of `_FAILURE_MARKERS` in stdout, or a
    per-step timeout. Never raises -- every outcome comes back as
    `{"ok": bool, "message": str, "log": str}`, with `log` the concatenated stdout of every
    step attempted (last ~4000 chars).
    """
    exe = Path(nulink_exe)
    env = _nulink_env(exe)
    steps = [
        ("連接開發板", [str(exe), "-C"]),
        ("寫入 APROM", [str(exe), "-W", "APROM", str(Path(bin_path).resolve()), "1"]),
        ("重置開發板", [str(exe), "-S"]),
    ]
    log: list[str] = []

    def rendered_log() -> str:
        return "\n".join(log)[-_LOG_TAIL_CHARS:]

    for title, cmd in steps:
        command_line = subprocess.list2cmdline(cmd)
        try:
            proc = run_tool(cmd, exe.parent, env, timeout_seconds)
        except subprocess.TimeoutExpired:
            log.append(f"$ {command_line}\n[TIMEOUT after {timeout_seconds}s]")
            return {"ok": False, "message": f"Nu-Link 逾時（{title}）", "log": rendered_log()}
        log.append(f"$ {command_line}\n{proc.stdout}")
        if proc.returncode != 0 or any(marker in proc.stdout for marker in _FAILURE_MARKERS):
            return {
                "ok": False,
                "message": f"Nu-Link {title} 失敗：{proc.stdout.strip()[-400:]}",
                "log": rendered_log(),
            }
    return {"ok": True, "message": "燒錄完成，板子已重置", "log": rendered_log()}


def pyocd_command(bin_path: Path, pyocd_exe: Path, pack_path: Path) -> list[str]:
    """The exact command NuML App Builder verified on a VoiceAI board.

    `-t M55M1R2LJAE` is the device name the DFP pack declares. The board's MCU marking is
    M55M1R2LJC7E -- Nuvoton's full ordering part number, with the package and temperature
    suffix -- and that string does not appear in the pack at all, so it cannot be used here.

    There is deliberately no reset step: `pyocd reset -m hw` leaves nRESET asserted on a
    Nu-Link2, which looks like a dead board.
    """
    return [
        str(pyocd_exe), "load",
        "--pack", str(pack_path),
        "-t", "M55M1R2LJAE",
        "--base-address", "0x00100000",
        str(bin_path),
    ]


def pyocd_program(
    bin_path: Path, pyocd_exe: Path, pack_path: Path, timeout_seconds: int = 300
) -> dict:
    """Program `bin_path` into APROM via `pyocd load` against the Nuvoton DFP pack.

    Unlike `nulink_program()`'s connect/write/reset three steps, `pyocd load` connects, writes
    and verifies in a single command -- and, per `pyocd_command()`, there is no reset step at
    all. Never raises: every outcome comes back as `{"ok": bool, "message": str, "log": str}`,
    with `log` the tool's stdout (last ~4000 chars), same shape as `nulink_program()`.
    """
    exe = Path(pyocd_exe)
    env = _nulink_env(exe)  # same minimal-environment shape, just rooted at pyocd's directory
    cmd = pyocd_command(Path(bin_path).resolve(), exe, Path(pack_path))
    command_line = subprocess.list2cmdline(cmd)
    try:
        proc = run_tool(cmd, exe.parent, env, timeout_seconds)
    except subprocess.TimeoutExpired:
        log = f"$ {command_line}\n[TIMEOUT after {timeout_seconds}s]"
        return {
            "ok": False,
            "message": f"pyocd 燒錄逾時（{timeout_seconds} 秒）",
            "log": log[-_LOG_TAIL_CHARS:],
        }
    log = f"$ {command_line}\n{proc.stdout}"
    if proc.returncode != 0:
        return {
            "ok": False,
            "message": f"pyocd 燒錄失敗：{proc.stdout.strip()[-400:]}",
            "log": log[-_LOG_TAIL_CHARS:],
        }
    return {"ok": True, "message": "pyocd 燒錄完成，請按一下板上的 Reset", "log": log[-_LOG_TAIL_CHARS:]}
