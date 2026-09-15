"""Compile the MCU frontend for the host and load it through ctypes.

The C in ``frontend/`` is plain portable C so that its numeric behaviour can be
pinned down on a PC in seconds instead of one Keil build plus a flash cycle per
iteration.  The same translation units are what the firmware compiles.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
# Never inside mcu_toolkit/: that tree is hashed file-for-file by manifest.json and must carry
# no .bat and no build output.  workspace/tmp/ is git-ignored.
DEFAULT_BUILD_DIR = Path("workspace") / "tmp" / "mcu_host"

_VCVARS = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    r"\VC\Auxiliary\Build\vcvars64.bat"
)

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_cached: ctypes.CDLL | None = None
_cached_stamp: tuple[str, tuple[tuple[str, str], ...]] | None = None


def find_vcvars() -> Path | None:
    """Locate vcvars64.bat, or None when this machine has no MSVC C compiler.

    Order: the TM_VCVARS64 override, then whatever vswhere.exe reports for the newest install
    that actually carries the x64 C toolset, then the App Builder's own fixed path.  The caller
    is expected to skip -- not fail -- on None: a student machine need not have a C compiler.
    """
    override = os.environ.get("TM_VCVARS64")
    if override and Path(override).is_file():
        return Path(override)
    for root_var in ("ProgramFiles(x86)", "ProgramFiles"):
        root = os.environ.get(root_var)
        if not root:
            continue
        vswhere = Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if not vswhere.is_file():
            continue
        try:
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=_CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in result.stdout.splitlines():
            install = line.strip()
            if not install:
                continue
            candidate = Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if candidate.is_file():
                return candidate
    return _VCVARS if _VCVARS.is_file() else None


def _stamp() -> tuple[tuple[str, str], ...]:
    """Rebuild on content, not mtime: a checkout or a re-vendor rewrites every timestamp."""
    return tuple(
        (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(FRONTEND.glob("*.[ch]"))
    )


def _stamp_file(build_dir: Path) -> Path:
    return build_dir / "numl_frontend.stamp.json"


def _read_stamp(build_dir: Path) -> tuple[tuple[str, str], ...] | None:
    try:
        raw = json.loads(_stamp_file(build_dir).read_text(encoding="utf-8"))
        return tuple((str(name), str(digest)) for name, digest in raw)
    except (OSError, ValueError, TypeError):
        return None


def _msvc_environment(vcvars: Path) -> dict[str, str]:
    """The environment vcvars64.bat exports, so cl.exe can be launched directly afterwards.

    Upstream generated a batch script instead, to sidestep the quoting rules that apply when a
    path containing spaces goes through "cmd.exe /c".  The Studio cannot: tests/test_distribution
    asserts the whole tree holds exactly two student batch files (01_INSTALL, 02_START), and a
    generated third one fails that wherever it lands -- workspace/ included.  Only the vcvars
    path crosses cmd here, as its own argv entry, and the compiler itself never does.
    """
    result = subprocess.run(
        ["cmd.exe", "/c", "call", str(vcvars), "&&", "set"],
        capture_output=True,
        text=True,
        timeout=300,
        creationflags=_CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{vcvars} failed:\n" + (result.stdout or "") + (result.stderr or "")
        )
    env = dict(os.environ)
    for line in result.stdout.splitlines():
        name, sep, value = line.partition("=")
        if sep and name and " " not in name:
            env[name] = value
    return env


def _compile(build_dir: Path) -> Path:
    sources = sorted(FRONTEND.glob("*.c"))
    if not sources:
        raise FileNotFoundError(f"No C sources in {FRONTEND}")
    vcvars = find_vcvars()
    if vcvars is None:
        raise FileNotFoundError(
            "MSVC environment script (vcvars64.bat) not found; set TM_VCVARS64 or install "
            "the Visual Studio Build Tools C++ workload"
        )

    build_dir.mkdir(parents=True, exist_ok=True)
    dll = build_dir / "numl_frontend.dll"
    env = _msvc_environment(vcvars)
    # CreateProcess resolves a bare name against the *parent* PATH, which has no cl.exe, so the
    # compiler is resolved against the PATH vcvars just produced and then launched by full path.
    compiler = shutil.which("cl.exe", path=env.get("PATH", ""))
    if compiler is None:
        raise RuntimeError(f"{vcvars} ran but left no cl.exe on PATH")

    # /fp:precise stops the compiler reassociating float arithmetic, which
    # would silently change results relative to the target build.  Outputs are bare names
    # because the build runs with build_dir as its working directory.
    result = subprocess.run(
        [
            compiler,
            "/nologo",
            "/LD",
            "/O2",
            "/fp:precise",
            "/W3",
            "/WX",
            f"/Fe:{dll.name}",
            *[str(path) for path in sources],
            "/link",
            "/INCREMENTAL:NO",
        ],
        capture_output=True,
        text=True,
        cwd=str(build_dir),
        env=env,
        timeout=600,
        creationflags=_CREATE_NO_WINDOW,
    )
    if result.returncode != 0 or not dll.is_file():
        raise RuntimeError(
            "Host build of the frontend failed:\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )
    _stamp_file(build_dir).write_text(
        json.dumps([list(entry) for entry in _stamp()], indent=2), encoding="utf-8"
    )
    return dll


def frontend(build_dir: Path | None = None) -> ctypes.CDLL:
    """Build (if needed) and return the loaded frontend library.

    ``build_dir`` defaults to the git-ignored ``workspace/tmp/mcu_host`` below the current
    working directory.  The DLL is reused across processes while the sha256 of the C sources
    recorded beside it still matches, and rebuilt the moment any of them changes.
    """

    global _cached, _cached_stamp
    target = (Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR).resolve()
    stamp = _stamp()
    key = (str(target), stamp)
    if _cached is not None and _cached_stamp == key:
        return _cached
    dll = target / "numl_frontend.dll"
    if not (dll.is_file() and _read_stamp(target) == stamp):
        dll = _compile(target)
    _cached = ctypes.CDLL(str(dll))
    _cached_stamp = key
    return _cached


def float_buffer(count: int):
    return (ctypes.c_float * count)()


def as_array(buffer, shape=None):
    import numpy as np

    values = np.ctypeslib.as_array(buffer).astype(np.float32, copy=True)
    return values.reshape(shape) if shape else values
