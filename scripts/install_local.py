from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
VENV_ROOT = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_ROOT / "Scripts" / "python.exe"
VERSION = "2.1.0"


class Tee:
    def __init__(self, primary, *files):  # noqa: ANN001
        self.primary = primary
        self.files = files
        self.lock = threading.RLock()

    @property
    def encoding(self):  # noqa: ANN201
        return getattr(self.primary, "encoding", "utf-8")

    def write(self, value: str) -> int:
        with self.lock:
            try:
                self.primary.write(value)
            except Exception:
                pass
            for handle in self.files:
                try:
                    handle.write(value)
                except Exception:
                    pass
        return len(value)

    def flush(self) -> None:
        with self.lock:
            try:
                self.primary.flush()
            except Exception:
                pass
            for handle in self.files:
                try:
                    handle.flush()
                except Exception:
                    pass

    def isatty(self) -> bool:
        try:
            return bool(self.primary.isatty())
        except Exception:
            return False


class InstallFailure(RuntimeError):
    pass


def setup_log() -> tuple[Path, Path, Any, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = LOG_ROOT / f"install_{stamp}.log"
    latest_path = LOG_ROOT / "LATEST_INSTALL.log"
    session_handle = session_path.open("a", encoding="utf-8", buffering=1)
    latest_handle = latest_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, session_handle, latest_handle)
    sys.stderr = Tee(sys.__stderr__, session_handle, latest_handle)
    return session_path, latest_path, session_handle, latest_handle


def command_text(command: Iterable[str | os.PathLike[str]]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def decode_output(payload: bytes) -> str:
    if not payload:
        return ""
    if b"\x00" in payload[:256]:
        try:
            return payload.decode("utf-16le", errors="replace").replace("\x00", "")
        except Exception:
            pass
    for encoding in ("utf-8", "cp950", "mbcs"):
        try:
            return payload.decode(encoding)
        except Exception:
            continue
    return payload.decode("utf-8", errors="replace")


def run(
    command: list[str | os.PathLike[str]],
    *,
    check: bool = True,
    capture: bool = False,
    quiet: bool = False,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a child command and mirror its output into the installer log.

    ``quiet`` suppresses echoing the captured stdout/stderr (the command line itself is
    still logged) — used by the MCU probe so its raw JSON never lands ahead of the
    friendly ``[MCU ]`` summary lines.
    """

    normalized = [str(item) for item in command]
    rendered = command_text(normalized)
    print(f"\n> {rendered}", flush=True)
    if capture:
        completed = subprocess.run(
            normalized,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if not quiet:
            if completed.stdout:
                print(decode_output(completed.stdout).rstrip(), flush=True)
            if completed.stderr:
                print(decode_output(completed.stderr).rstrip(), file=sys.stderr, flush=True)
    else:
        process = subprocess.Popen(
            normalized,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None
        while True:
            raw = process.stdout.readline()
            if not raw:
                break
            print(decode_output(raw), end="", flush=True)
        completed = subprocess.CompletedProcess(
            normalized,
            process.wait(),
            stdout=b"",
            stderr=b"",
        )
    if check and completed.returncode != 0:
        raise InstallFailure(f"Command failed with exit code {completed.returncode}: {rendered}")
    return completed


def validate_base_python() -> None:
    if platform.system() != "Windows":
        raise InstallFailure("This distribution supports native Windows only.")
    if platform.python_implementation() != "CPython":
        raise InstallFailure("Only normal CPython is supported.")
    if sys.version_info[:2] != (3, 13):
        raise InstallFailure(f"Python 3.13 is required; current version is {platform.python_version()}.")
    if sys.maxsize <= 2**32:
        raise InstallFailure("64-bit Python 3.13 is required.")
    if getattr(sys, "_is_gil_enabled", None) and not sys._is_gil_enabled():  # type: ignore[attr-defined]
        raise InstallFailure("The experimental free-threaded Python 3.13t build is not supported.")


def ensure_venv() -> None:
    if VENV_ROOT.exists() and not VENV_PYTHON.is_file():
        print("[RECREATE] Incomplete .venv detected; removing it.")
        shutil.rmtree(VENV_ROOT, ignore_errors=False)
    if VENV_PYTHON.is_file():
        completed = run(
            [VENV_PYTHON, "-c", "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,13) else 1)"],
            check=False,
        )
        if completed.returncode == 0:
            print(f"[REUSE] Existing project environment: {VENV_ROOT}")
            return
        print("[RECREATE] Existing .venv is not Python 3.13; removing it.")
        shutil.rmtree(VENV_ROOT, ignore_errors=False)
    print(f"[CREATE] Creating project-local environment: {VENV_ROOT}")
    run([sys.executable, "-m", "venv", VENV_ROOT])


def probe_mcu_toolchain(env: dict[str, str]) -> dict:
    """Optional firmware toolchain probe. Missing tools only disable the MCU tab."""
    code = (
        "import json; from tm_local.mcu.toolchain import probe_all; "
        "print(json.dumps(probe_all()))"
    )
    try:
        completed = run(
            [str(VENV_PYTHON), "-c", code], check=False, capture=True, quiet=True, env=env
        )
        info = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001 - keep the installer alive
        info = {"available": False, "error": str(exc)}
    if not isinstance(info, dict):
        # A syntactically valid but non-object payload (``null``, a bare string, a list, ...)
        # must not escape as-is: every caller below assumes a dict, and this step must never
        # fail the install.
        info = {"available": False, "error": "unexpected probe output"}
    tc = info.get("toolchain") or {}
    if tc.get("available"):
        print(f"[MCU ] Arm GNU Toolchain : {tc.get('version')} ({tc.get('bin_dir')})")
    else:
        print("[MCU ] Arm GNU Toolchain not found. Firmware build is disabled until it is installed.")
        print("[MCU ] Download: https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads")
        print(f"[MCU ] Either run the installer, or unzip the .zip build into: {PROJECT_ROOT / 'runtime' / 'arm-gnu-toolchain'}")
        print(
            "[MCU ] Or let Local Studio download and verify it for you: "
            ".venv\\Scripts\\python.exe scripts\\download_arm_toolchain.py"
        )
    nulink = info.get("nulink") or {}
    print(f"[MCU ] Nu-Link Command Tool: {'found ' + nulink['path'] if nulink.get('available') else 'not found (optional, NuMaker-M55M1 only)'}")
    return info


def install_windows_environment() -> dict:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "TF_USE_LEGACY_KERAS": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    print("\n[1/5] Updating pip / setuptools / wheel")
    run(
        [
            VENV_PYTHON,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--timeout",
            "240",
            "--retries",
            "5",
            "pip",
            "setuptools",
            "wheel",
        ],
        env=env,
    )
    print("\n[2/5] Installing Windows packages (TensorFlow is a large download)")
    run(
        [
            VENV_PYTHON,
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            "--timeout",
            "300",
            "--retries",
            "5",
            "-r",
            PROJECT_ROOT / "requirements.txt",
        ],
        env=env,
    )
    print("\n[3/5] Checking dependency consistency")
    run([VENV_PYTHON, "-m", "pip", "check"], env=env)
    print("\n[4/5] Running TensorFlow / Keras / strict INT8 verification")
    run([VENV_PYTHON, PROJECT_ROOT / "scripts" / "verify_install.py"], env=env)
    # Optional asset prefetch. Network failure here must not invalidate the environment.
    run([VENV_PYTHON, PROJECT_ROOT / "scripts" / "prefetch_assets.py"], check=False, env=env)
    print("\n[5/5] MCU toolchain probe (optional; never fails the install)")
    mcu_info = probe_mcu_toolchain(env)
    return mcu_info


def powershell_json(script: str) -> Any:
    completed = run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=False,
        capture=True,
    )
    if completed.returncode != 0:
        return None
    text = decode_output(completed.stdout).strip().lstrip("\ufeff")
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def windows_memory_gib() -> float | None:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024**3), 2)
    except Exception:
        pass
    return None


def detect_video_controllers() -> list[dict[str, Any]]:
    payload = powershell_json(
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress"
    )
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def build_runtime_config(
    video_controllers: list[dict[str, Any]], mcu: dict[str, Any] | None = None
) -> dict[str, Any]:
    logical = max(1, int(os.cpu_count() or 1))
    memory = windows_memory_gib()
    # Leave one core for Windows when possible and cap classroom machines at 8 threads.
    thread_limit = max(1, min(8, logical - 1 if logical > 2 else logical))
    return {
        "schema_version": 2,
        "selected_backend": "windows_tensorflow_cpu",
        "backend_label": "Windows CPU",
        "reason": (
            "Windows-only mode is enabled. TensorFlow 2.21 with Python 3.13 runs "
            "natively on Windows CPU; no Linux environment is used."
        ),
        "windows_only": True,
        "cpu": {
            "logical_cores": logical,
            "thread_limit": thread_limit,
            "memory_gib": memory,
        },
        "video_controllers": video_controllers,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mcu": mcu or {"available": False},
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    original_stdout, original_stderr = sys.stdout, sys.stderr
    session_path, latest_path, session_handle, latest_handle = setup_log()
    try:
        os.chdir(PROJECT_ROOT)
        validate_base_python()
        print("=" * 78)
        print(f" Teachable Machine Local Studio v{VERSION}")
        print(" Windows only · Python 3.13 · project-local .venv · CPU runtime")
        print(" No Linux, CUDA, or secondary runtime setup will be requested.")
        print("=" * 78)
        print(f"Base Python : {sys.executable}")
        print(f"Install log : {latest_path}")

        ensure_venv()
        mcu_info = install_windows_environment()

        controllers = detect_video_controllers()
        runtime_config = build_runtime_config(controllers, mcu=mcu_info)
        write_json(RUNTIME_ROOT / "runtime_config.json", runtime_config)
        environment_report = {
            "version": VERSION,
            "installed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "project_root": str(PROJECT_ROOT),
            "base_python": sys.executable,
            "venv_python": str(VENV_PYTHON),
            "platform": platform.platform(),
            "video_controllers": controllers,
            "runtime": runtime_config,
            "install_log": str(session_path),
        }
        write_json(RUNTIME_ROOT / "environment_report.json", environment_report)
        (PROJECT_ROOT / ".venv_ready").write_text(
            f"Version={VERSION}\nPython=3.13\nBackend=windows_tensorflow_cpu\n",
            encoding="utf-8",
        )

        print("\n" + "=" * 78)
        print("[DONE] Installation completed")
        print(f"Environment  : {VENV_ROOT}")
        print("Runtime      : Windows CPU")
        print("Linux/WSL    : disabled")
        print(f"MCU toolchain: {'available' if mcu_info.get('available') else 'missing'}")
        print(f"Log          : {latest_path}")
        print("Next        : double-click 02_START.bat")
        print("Fresh reinstall: close Local Studio, delete .venv, then run 01_INSTALL.bat again.")
        print("Do not delete the workspace folder.")
        print("=" * 78)
        return 0
    except Exception as exc:
        print("\n" + "=" * 78, file=sys.stderr)
        print(f"[ERROR] Installation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Diagnostic log: {latest_path}", file=sys.stderr)
        print("You may delete .venv and run 01_INSTALL.bat again. Do not delete workspace.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        return 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = original_stdout, original_stderr
        session_handle.close()
        latest_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
