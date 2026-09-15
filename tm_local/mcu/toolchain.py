from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import PROJECT_ROOT
from .gcc_build import Toolchain
from .paths import (
    BSP_ROOT,
    LOCAL_NUMICRO_PACK_ROOT,
    LOCAL_TOOLCHAIN_ROOT,
    MANIFEST_JSON,
    VELA_EXE,
    toolkit_available,
)

LOGGER = logging.getLogger("tm_local.mcu.toolchain")
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PROGRAM_FILES_GLOB = "Arm GNU Toolchain arm-none-eabi/*/bin"
# Nuvoton.NuMicroM55_DFP.<version>.pack, where <version> is either a plain semver
# ("3.1.4") or a release candidate ("3.1.4-rc.3"). Arm's and Keil's pack caches can each
# hold several versions side by side, so the newest must be picked, not the first found.
_PACK_GLOB = "Nuvoton.NuMicroM55_DFP.*.pack"
_PACK_FILENAME_RE = re.compile(r"^Nuvoton\.NuMicroM55_DFP\.(?P<version>.+)\.pack$")
_PACK_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$")


def _gcc_in(bin_dir: Path) -> Path | None:
    for name in ("arm-none-eabi-gcc.exe", "arm-none-eabi-gcc.cmd", "arm-none-eabi-gcc"):
        candidate = bin_dir / name
        if candidate.is_file():
            return candidate
    return None


def candidate_bin_dirs() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    env = os.environ.get("TM_ARM_GCC_BIN")
    if env:
        out.append(("env", Path(env).expanduser().resolve()))
    try:
        from ..runtime_config import load_runtime_config

        configured = (load_runtime_config().get("mcu") or {}).get("toolchain") or {}
        if configured.get("bin_dir"):
            out.append(("runtime_config", Path(configured["bin_dir"]).resolve()))
    except Exception:
        LOGGER.debug("runtime_config toolchain probe failed", exc_info=True)
    on_path = shutil.which("arm-none-eabi-gcc")
    if on_path:
        out.append(("path", Path(on_path).resolve().parent))
    if LOCAL_TOOLCHAIN_ROOT.is_dir():
        if (LOCAL_TOOLCHAIN_ROOT / "bin").is_dir():
            out.append(("local", (LOCAL_TOOLCHAIN_ROOT / "bin").resolve()))
        for nested in sorted(LOCAL_TOOLCHAIN_ROOT.glob("*/bin")):
            out.append(("local", nested.resolve()))
    for root_var in ("ProgramFiles(x86)", "ProgramFiles"):
        root = os.environ.get(root_var)
        if root:
            for found in sorted(Path(root).glob(_PROGRAM_FILES_GLOB), reverse=True):
                out.append(("program_files", found.resolve()))
    seen: set[Path] = set()
    unique: list[tuple[str, Path]] = []
    for label, path in out:
        if path not in seen:
            seen.add(path)
            unique.append((label, path))
    return unique


def _version_of(gcc: Path) -> str | None:
    try:
        proc = subprocess.run(
            [str(gcc), "--version"], cwd=str(gcc.parent),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", timeout=20,
            creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout.strip().splitlines()[0]


def probe_toolchain() -> dict:
    checked: list[str] = []
    for label, bin_dir in candidate_bin_dirs():
        checked.append(str(bin_dir))
        gcc = _gcc_in(bin_dir)
        if gcc is None:
            continue
        version = _version_of(gcc)
        if version:
            return {"available": True, "bin_dir": str(bin_dir), "version": version, "source": label,
                    "checked": checked}
    return {"available": False, "bin_dir": None, "version": None, "source": None, "checked": checked}


def find_toolchain() -> Toolchain | None:
    info = probe_toolchain()
    if not info["available"]:
        return None
    return Toolchain(Path(info["bin_dir"]), info["version"])


def probe_nulink() -> dict:
    env = os.environ.get("TM_NULINK_EXE")
    if env:
        # The env var is authoritative, mirroring the toolchain's "env wins" rule: if the
        # student points at a specific NuLink.exe, a stray Program Files install must not
        # silently take over.
        candidate = Path(env).expanduser()
        if candidate.is_file():
            return {"available": True, "path": str(candidate.resolve())}
        return {"available": False, "path": None}
    candidates: list[Path] = []
    for root_var in ("ProgramFiles(x86)", "ProgramFiles"):
        root = os.environ.get(root_var)
        if root:
            candidates.append(Path(root) / "Nuvoton Tools" / "NuLink Command Tool" / "M55M1_M5531" / "NuLink.exe")
            candidates.append(Path(root) / "Nuvoton Tools" / "NuLink_Command_Tool" / "M55M1_M5531" / "NuLink.exe")
    on_path = shutil.which("NuLink.exe")
    if on_path and "M55M1" in on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return {"available": True, "path": str(candidate.resolve())}
    return {"available": False, "path": None}


def probe_pyocd() -> dict:
    """TM_PYOCD_EXE -> runtime_config -> the project's own .venv -> PATH.

    Studio never bundles pyocd (most students have no VoiceAI board and should not carry the
    dependency); this only detects what the student or `scripts/setup_voiceai_flash.py`
    installed. Same "env wins outright" structure as `probe_nulink()`.
    """
    env = os.environ.get("TM_PYOCD_EXE")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_file():
            return {"available": True, "path": str(candidate.resolve())}
        return {"available": False, "path": None}
    candidates: list[Path] = []
    try:
        from ..runtime_config import load_runtime_config

        configured = (load_runtime_config().get("mcu") or {}).get("pyocd") or {}
        if configured.get("exe"):
            candidates.append(Path(configured["exe"]))
    except Exception:
        LOGGER.debug("runtime_config pyocd probe failed", exc_info=True)
    candidates.append(PROJECT_ROOT / ".venv" / "Scripts" / "pyocd.exe")
    on_path = shutil.which("pyocd")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return {"available": True, "path": str(candidate.resolve())}
    return {"available": False, "path": None}


def _pack_sort_key(path: Path) -> tuple[int, int, int, int, int] | None:
    """Numeric sort key for a pack filename, or `None` if it doesn't parse.

    `(major, minor, patch, is_release, rc_or_zero)`: a plain string sort gets two things
    backwards -- `"10" < "9"`, and (worse) `"3.1.4-rc.3" > "3.1.4"` -- so a release candidate
    must lose to that same version's real release. `is_release` (0 for `-rc.N`, 1 otherwise)
    makes that comparison land the right way: `(3, 1, 4, 0, 3) < (3, 1, 4, 1, 0)`.
    """
    name_match = _PACK_FILENAME_RE.match(path.name)
    if not name_match:
        return None
    version_match = _PACK_SEMVER_RE.match(name_match.group("version"))
    if not version_match:
        return None
    major, minor, patch, rc = version_match.groups()
    if rc is not None:
        return (int(major), int(minor), int(patch), 0, int(rc))
    return (int(major), int(minor), int(patch), 1, 0)


def _newest_pack(roots: list[Path]) -> Path | None:
    """The highest-versioned `Nuvoton.NuMicroM55_DFP.*.pack` across every directory in `roots`."""
    best: tuple[tuple[int, int, int, int, int], Path] | None = None
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.glob(_PACK_GLOB):
            if not candidate.is_file():
                continue
            key = _pack_sort_key(candidate)
            if key is None:
                continue
            if best is None or key > best[0]:
                best = (key, candidate)
    return best[1] if best else None


def probe_numicro_pack(roots: list[Path] | None = None) -> dict:
    """TM_NUMICRO_PACK -> runtime_config -> runtime/numicro-pack/ -> Arm's pack cache ->
    Keil's pack cache.

    `runtime/numicro-pack/` is where `scripts/setup_voiceai_flash.py` downloads the pack; it
    must be searched unconditionally (like `LOCAL_TOOLCHAIN_ROOT` is for `probe_toolchain()`),
    so detection works even if `runtime_config` was never written or has since been reset.

    Arm's pack manager keeps every version it has downloaded in
    `%LOCALAPPDATA%\\Arm\\Packs\\.Download`; Keil keeps its own in
    `C:\\Keil_v5\\ARM\\PACK\\.Download`. All three locations may hold several versions of
    Nuvoton.NuMicroM55_DFP side by side, so the newest wins (see `_pack_sort_key()`).

    `roots`, when given, replaces the whole search with a scan of just those directories --
    the seam `tests/test_mcu_toolchain.py` uses to exercise version selection without
    touching the real pack caches or `TM_NUMICRO_PACK`/`runtime_config`.
    """
    if roots is not None:
        found = _newest_pack(roots)
        if found is not None:
            return {"available": True, "path": str(found.resolve())}
        return {"available": False, "path": None}

    env = os.environ.get("TM_NUMICRO_PACK")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_file():
            return {"available": True, "path": str(candidate.resolve())}
        return {"available": False, "path": None}

    configured_path = None
    try:
        from ..runtime_config import load_runtime_config

        configured = (load_runtime_config().get("mcu") or {}).get("numicro_pack") or {}
        configured_path = configured.get("path")
    except Exception:
        LOGGER.debug("runtime_config numicro_pack probe failed", exc_info=True)
    if configured_path:
        candidate = Path(configured_path).expanduser()
        if candidate.is_file():
            return {"available": True, "path": str(candidate.resolve())}

    search_roots: list[Path] = [LOCAL_NUMICRO_PACK_ROOT]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        search_roots.append(Path(local_app_data) / "Arm" / "Packs" / ".Download")
    search_roots.append(Path(r"C:\Keil_v5\ARM\PACK\.Download"))
    found = _newest_pack(search_roots)
    if found is not None:
        return {"available": True, "path": str(found.resolve())}
    return {"available": False, "path": None}


def probe_all() -> dict:
    tc = probe_toolchain()
    nulink = probe_nulink()
    pyocd = probe_pyocd()
    numicro_pack = probe_numicro_pack()
    vela = {"available": VELA_EXE.is_file(), "path": str(VELA_EXE)}
    bsp_tag = None
    if MANIFEST_JSON.is_file():
        try:
            bsp_tag = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))["sources"]["m55m1bsp_tag"]
        except (OSError, ValueError, KeyError, TypeError):
            bsp_tag = None
    toolkit = {"available": toolkit_available() and BSP_ROOT.is_dir(), "bsp_tag": bsp_tag}
    return {
        "available": bool(tc["available"] and vela["available"] and toolkit["available"]),
        "toolchain": tc,
        "nulink": nulink,
        "pyocd": pyocd,
        "numicro_pack": numicro_pack,
        "vela": vela,
        "toolkit": toolkit,
        "probed_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
