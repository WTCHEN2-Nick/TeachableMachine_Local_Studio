from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import paths as mcu_paths  # module reference: tests monkeypatch MCU_TEMP_ROOT
from .codegen import scrub_paths
from .errors import DeployError
from .records import BuildManifest, SourceFile

TEMPLATE_CPU_FLAGS = ("-mcpu=cortex-m55", "-mthumb")
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class Toolchain:
    bin_dir: Path
    version: str

    def tool(self, name: str) -> Path:
        exe = self.bin_dir / f"arm-none-eabi-{name}.exe"
        return exe if exe.exists() else self.bin_dir / f"arm-none-eabi-{name}"


@dataclass
class BuildResult:
    ok: bool
    message: str
    log_path: Path
    elf: Path | None = None
    bin: Path | None = None
    hex: Path | None = None
    map: Path | None = None
    text: int = 0
    data: int = 0
    bss: int = 0
    flash_used: int = 0
    sram01_used: int = 0
    noncacheable_used: int = 0
    seconds: float = 0.0
    failed_command: list[str] | None = None
    sections: dict[str, tuple[int, int]] = field(default_factory=dict)


def sanitized_env(tc: Toolchain) -> dict[str, str]:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    env = {
        "PATH": os.pathsep.join([str(tc.bin_dir), os.path.join(system_root, "System32"), system_root]),
        "SystemRoot": system_root,
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "USERPROFILE": os.environ.get("USERPROFILE", ""),
        "PYTHONUTF8": "1",
    }
    return {k: v for k, v in env.items() if v}


def run_tool(cmd: list[str], cwd: Path, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
        creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        check=False,
    )


def _common(manifest: BuildManifest) -> list[str]:
    return [*manifest.common_flags, *TEMPLATE_CPU_FLAGS]


def _includes_and_defines(manifest: BuildManifest) -> list[str]:
    out = ["-I."]
    out += [f"-I{p}" for p in manifest.includes]
    out += [f"-D{d}" for d in manifest.defines]
    return out


def _prefix_map_flags(manifest: BuildManifest) -> list[str]:
    """`-ffile-prefix-map=<old>=<new>` for every (old, new) pair in manifest.file_prefix_map.

    Applies to `__FILE__`/`__BASE_FILE__` and debug-info paths so the firmware image does not
    depend on the absolute install path of this Studio checkout or its mcu_toolkit/. Compile
    steps only -- never passed to the linker-script preprocess.
    """
    return [f"-ffile-prefix-map={old}={new}" for old, new in manifest.file_prefix_map]


def compile_command(tc: Toolchain, manifest: BuildManifest, src: SourceFile, obj_path: Path) -> list[str]:
    if src.kind == "cxx":
        tool, lang_flags = tc.tool("g++"), manifest.cxx_flags
    elif src.kind == "c":
        tool, lang_flags = tc.tool("gcc"), manifest.c_flags
    else:
        tool, lang_flags = tc.tool("gcc"), manifest.asm_flags
    tail = ["-c"] if src.kind == "asm" else ["-c", "-MMD", "-MP"]
    # <VPATH dir>/<basename> (native-separator directory, forward slash, basename) is the exact
    # shape `make` handed the compiler for the spike build, so __FILE__ matches byte-for-byte
    # once -ffile-prefix-map rewrites the directory portion (see _prefix_map_flags above).
    source_operand = f"{src.path.parent}/{src.path.name}"
    return [str(tool), *_common(manifest), *lang_flags, *_prefix_map_flags(manifest),
            *_includes_and_defines(manifest), *tail, source_operand, "-o", obj_path.as_posix()]


def preprocess_ld_command(tc: Toolchain, manifest: BuildManifest, out_ld: Path) -> list[str]:
    return [str(tc.tool("cpp")), "-E", "-x", "c", "-P", *_includes_and_defines(manifest),
            str(manifest.linker_script), "-o", out_ld.as_posix()]


def link_command(tc: Toolchain, manifest: BuildManifest, objects: list[Path], elf: Path, map_path: Path,
                 generated_ld: Path) -> list[str]:
    cmd = [str(tc.tool("gcc"))]
    cmd += [f"-L{d}" for d in manifest.lib_dirs]
    cmd += ["-o", elf.as_posix()]
    cmd += [o.as_posix() for o in objects]
    cmd += [f"-l{lib}" for lib in manifest.libs]
    cmd += ["-Wl,--start-group", *[f"-l{lib}" for lib in manifest.standard_libraries], "-Wl,--end-group"]
    cmd += [*manifest.ld_flags, *manifest.extra_ld_flags, *_common(manifest)]
    cmd += [f"-Wl,-Map={map_path.as_posix()},--cref", f"-T{generated_ld.as_posix()}"]
    return cmd


def parse_size_output(text: str) -> tuple[int, int, int]:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0].isdigit() and parts[-1] == "(TOTALS)":
            return int(parts[0]), int(parts[1]), int(parts[2])
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            return int(parts[0]), int(parts[1]), int(parts[2])
    raise DeployError("無法解析 arm-none-eabi-size 輸出")


_SECTION_LINE = re.compile(r"^(\.[\w.]+)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)")
_ADDR_LINE = re.compile(r"^\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)")


def parse_map_sections(map_path: Path) -> dict[str, tuple[int, int]]:
    sections: dict[str, tuple[int, int]] = {}
    pending: str | None = None
    for raw in Path(map_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith(".") and not raw.startswith(".."):
            m = _SECTION_LINE.match(raw)
            if m:
                sections[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16))
                pending = None
            else:
                pending = raw.split()[0] if raw.split() else None
            continue
        if pending is not None:
            m = _ADDR_LINE.match(raw)
            if m:
                sections[pending] = (int(m.group(1), 16), int(m.group(2), 16))
            pending = None
    return sections


def _sum_sections(sections: dict[str, tuple[int, int]], prefix: str) -> int:
    return sum(size for name, (_addr, size) in sections.items() if name.startswith(prefix))


def build(
    manifest: BuildManifest,
    toolchain: Toolchain,
    *,
    progress: ProgressFn | None = None,
    jobs: int | None = None,
    timeout_seconds: int = 1200,
    log_path: Path | None = None,
) -> BuildResult:
    started = time.monotonic()
    cwd = manifest.app_dir / "GCC"
    build_dir = cwd / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or (build_dir / "build.log")
    env = sanitized_env(toolchain)
    deadline = started + timeout_seconds
    log_lines: list[str] = [f"toolchain: {toolchain.bin_dir} ({toolchain.version})", f"cwd: {cwd}"]
    # build.log is handed to the student and packed into the firmware ZIP, so every absolute
    # path in it (the commands, the `cwd:` line) is rewritten the way -ffile-prefix-map rewrites
    # it for the compiler -- otherwise a shared ZIP carries the Windows username.
    scrub_pairs = [
        *manifest.file_prefix_map,
        (str(mcu_paths.MCU_TEMP_ROOT), "TMP"),
        (str(toolchain.bin_dir), "TOOLCHAIN"),
    ]

    def emit(value: float, message: str) -> None:
        if progress:
            progress(value, message)

    def finish(ok: bool, message: str, **extra) -> BuildResult:
        log_path.write_text(scrub_paths("\n".join(log_lines) + "\n", scrub_pairs), encoding="utf-8")
        return BuildResult(ok=ok, message=message, log_path=log_path,
                           seconds=time.monotonic() - started, **extra)

    def timed_out(cmd: list[str]) -> BuildResult:
        message = f"建置逾時（{timeout_seconds} 秒），已終止（見 build.log）"
        log_lines.append(f"$ {subprocess.list2cmdline(cmd)}")
        log_lines.append(f"[TIMEOUT] {message}")
        return finish(False, message, failed_command=cmd)

    def failed(exc: BaseException) -> BuildResult:
        """Any non-timeout failure of a build step (a vanished binary, an unparsable size
        output, ...) must still produce a written build.log, because deploy_service copies it
        out for the student and quotes its tail in the DeployError.

        `result.message` (unlike `log_lines`, which `finish()` scrubs before writing the file)
        travels straight into that DeployError, so `str(exc)` -- which can itself embed an
        absolute path, e.g. a FileNotFoundError naming the missing file -- must be scrubbed
        here too, not just in the log file."""
        message = scrub_paths(
            f"建置失敗：{type(exc).__name__}: {exc}（見 build.log）", scrub_pairs
        )
        log_lines.append(f"[ERROR] {message}")
        return finish(False, message)

    def guarded_run(cmd: list[str]) -> subprocess.CompletedProcess:
        """run_tool() honoring the *overall* deadline, not a fresh per-call budget.

        If the deadline has already passed before this step even starts, fail it as a timeout
        immediately instead of handing subprocess.run() a manufactured positive timeout.
        """
        time_left = deadline - time.monotonic()
        if time_left <= 0:
            raise subprocess.TimeoutExpired(cmd, timeout_seconds)
        return run_tool(cmd, cwd, env, max(1, int(time_left)))

    manifest.check_object_collisions()
    ordered = [*manifest.cxx_sources(), *manifest.c_sources(), *manifest.asm_sources()]
    objects: dict[SourceFile, Path] = {s: Path("build") / manifest.object_name(s) for s in ordered}
    total = len(ordered)
    done = 0
    workers = jobs or max(1, min(8, (os.cpu_count() or 2) - 1))

    def compile_one(src: SourceFile):
        cmd = compile_command(toolchain, manifest, src, objects[src])
        return src, cmd, guarded_run(cmd)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(compile_one, src) for src in ordered]
        for future in as_completed(futures):
            try:
                src, cmd, proc = future.result()
            except subprocess.TimeoutExpired as exc:
                pool.shutdown(wait=True, cancel_futures=True)
                failed_cmd = list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else None
                return timed_out(failed_cmd if failed_cmd is not None else [])
            except Exception as exc:  # noqa: BLE001 - a raised step becomes a failed BuildResult
                pool.shutdown(wait=True, cancel_futures=True)
                return failed(exc)
            log_lines.append(f"$ {subprocess.list2cmdline(cmd)}")
            if proc.stdout:
                log_lines.append(proc.stdout.rstrip())
            if proc.returncode != 0:
                pool.shutdown(wait=True, cancel_futures=True)
                return finish(False, f"編譯失敗：{src.path.name}（見 build.log）", failed_command=cmd)
            done += 1
            emit(done / (total + 3), f"編譯 {src.path.name}（{done}/{total}）")

    try:
        generated_ld = Path("build") / f"{manifest.project_name}.generated.ld"
        cmd = preprocess_ld_command(toolchain, manifest, generated_ld)
        try:
            proc = guarded_run(cmd)
        except subprocess.TimeoutExpired:
            return timed_out(cmd)
        log_lines.append(f"$ {subprocess.list2cmdline(cmd)}")
        log_lines.append(proc.stdout.rstrip())
        if proc.returncode != 0:
            return finish(False, "linker script 前處理失敗（見 build.log）", failed_command=cmd)

        elf = Path("build") / f"{manifest.project_name}.elf"
        map_rel = Path("build") / f"{manifest.project_name}.map"
        cmd = link_command(toolchain, manifest, [objects[s] for s in ordered], elf, map_rel, generated_ld)
        emit((total + 1) / (total + 3), "連結中…")
        try:
            proc = guarded_run(cmd)
        except subprocess.TimeoutExpired:
            return timed_out(cmd)
        log_lines.append(f"$ {subprocess.list2cmdline(cmd)}")
        log_lines.append(proc.stdout.rstrip())
        if proc.returncode != 0:
            hint = ""
            if "overflowed" in proc.stdout:
                hint = "；記憶體區段溢出，請降低模型大小（image_size／alpha／encoder_depth）"
            return finish(False, f"連結失敗{hint}（見 build.log）", failed_command=cmd)

        size_cmd = [str(toolchain.tool("size")), "--totals", elf.as_posix()]
        try:
            proc = guarded_run(size_cmd)
        except subprocess.TimeoutExpired:
            return timed_out(size_cmd)
        log_lines.append(f"$ {subprocess.list2cmdline(size_cmd)}\n{proc.stdout.rstrip()}")
        text, data, bss = parse_size_output(proc.stdout)
        hex_rel = Path("build") / f"{manifest.project_name}.hex"
        bin_rel = Path("build") / f"{manifest.project_name}.bin"
        for fmt, out in (("ihex", hex_rel), ("binary", bin_rel)):
            cmd = [str(toolchain.tool("objcopy")), "-O", fmt, elf.as_posix(), out.as_posix()]
            try:
                proc = guarded_run(cmd)
            except subprocess.TimeoutExpired:
                return timed_out(cmd)
            log_lines.append(f"$ {subprocess.list2cmdline(cmd)}\n{proc.stdout.rstrip()}")
            if proc.returncode != 0:
                return finish(False, f"objcopy {fmt} 失敗（見 build.log）", failed_command=cmd)
        emit(1.0, "建置完成")
        sections = parse_map_sections(cwd / map_rel)
        bin_path = cwd / bin_rel
        return finish(
            True, "建置完成",
            elf=cwd / elf, bin=bin_path, hex=cwd / hex_rel, map=cwd / map_rel,
            text=text, data=data, bss=bss,
            flash_used=bin_path.stat().st_size,
            sram01_used=_sum_sections(sections, ".sram01"),
            noncacheable_used=_sum_sections(sections, ".NonCacheable"),
            sections=sections,
        )
    except subprocess.TimeoutExpired as exc:
        cmd = list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else []
        return timed_out(cmd)
    except Exception as exc:  # noqa: BLE001 - a raised step becomes a failed BuildResult
        return failed(exc)
