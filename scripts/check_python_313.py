from __future__ import annotations

import argparse
import platform
import struct
import sys
import sysconfig
from pathlib import Path


def validate() -> list[str]:
    errors: list[str] = []
    if platform.python_implementation() != "CPython":
        errors.append("必須使用標準 CPython。")
    if sys.version_info[:2] != (3, 13):
        errors.append(
            "必須使用 Python 3.13.x；目前是 "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}。"
        )
    if struct.calcsize("P") * 8 != 64:
        errors.append("必須使用 64-bit Python。")
    if bool(sysconfig.get_config_var("Py_GIL_DISABLED")):
        errors.append("不支援 experimental free-threaded Python 3.13t，請安裝一般版 Python 3.13。")
    try:
        import ensurepip  # noqa: F401
        import venv  # noqa: F401
    except Exception as exc:
        errors.append(f"Python 缺少 venv/ensurepip：{exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-executable", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    errors = validate()
    if args.print_executable:
        if errors:
            return 1
        print(Path(sys.executable).resolve())
        return 0
    if not args.quiet:
        print("Executable :", Path(sys.executable).resolve())
        print("Python     :", sys.version.replace("\n", " "))
        print("Runtime    :", platform.python_implementation())
        print("Bitness    :", struct.calcsize("P") * 8)
        print("Free-threaded:", bool(sysconfig.get_config_var("Py_GIL_DISABLED")))
    if errors:
        if not args.quiet:
            print("\n[FAILED]")
            for error in errors:
                print(" -", error)
        return 1
    if not args.quiet:
        print("[OK] Python 3.13 64-bit 可建立專案 .venv。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
