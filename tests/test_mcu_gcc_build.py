from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tm_local.mcu import gcc_build, records

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mcu" / "spike_x_imgclass"
UPSTREAM_ENV_VAR = "TM_NUML_UPSTREAM_ROOT"


def _toolchain() -> gcc_build.Toolchain:
    exe = shutil.which("arm-none-eabi-gcc")
    if not exe:
        pytest.skip("arm-none-eabi-gcc not installed")
    return gcc_build.Toolchain(Path(exe).parent, "path")


def _mini_manifest(tmp_path: Path) -> records.BuildManifest:
    app = tmp_path / "app"
    (app / "GCC").mkdir(parents=True)
    (app / "main.c").write_text(
        'int g_counter __attribute__((section(".sram01"))) = 0;\n'
        "int add(int a, int b) { return a + b; }\n"
        "void Reset_Handler(void) { g_counter = add(1, 2); for (;;) {} }\n"
        '__attribute__((section(".vectors"))) const void *vectors[] = '
        "{ (void *)0x20020000, (void *)Reset_Handler };\n",
        encoding="utf-8",
    )
    (app / "extra.cpp").write_text('extern "C" int mul(int a, int b) { return a * b; }\n', encoding="utf-8")
    (app / "GCC" / "gcc.ld").write_text(
        "MEMORY { FLASH (rx) : ORIGIN = 0x00100000, LENGTH = 0x200000\n"
        "         SRAM01 (rw) : ORIGIN = 0x81F00000, LENGTH = 0x100000 }\n"
        "ENTRY(Reset_Handler)\n"
        "SECTIONS {\n"
        "  .text : { KEEP(*(.vectors)) *(.text*) *(.rodata*) } > FLASH\n"
        "  .sram01 (NOLOAD) : { *(.sram01) } > SRAM01\n"
        "  .bss (NOLOAD) : { *(.bss*) } > SRAM01\n"
        "}\n",
        encoding="utf-8",
    )
    srcs = [
        records.SourceFile((app / "main.c").resolve(), "c", "..\\..\\app\\main.c"),
        records.SourceFile((app / "extra.cpp").resolve(), "cxx", "..\\..\\app\\extra.cpp"),
    ]
    return records.BuildManifest(
        project_name="Mini", app_dir=app.resolve(), sources=srcs, includes=[app.resolve()],
        defines=["MINI=1"], lib_dirs=[], libs=[],
        common_flags=["-mfloat-abi=hard", "-O1", "-ffunction-sections", "-fdata-sections", "-mthumb"],
        c_flags=["-std=c11"], cxx_flags=["-fno-rtti", "-fno-exceptions", "-std=c++14"], asm_flags=[],
        ld_flags=["-specs=nosys.specs", "-Wl,--gc-sections", "-nostartfiles"],
        standard_libraries=["gcc", "c"],
        linker_script=(app / "GCC" / "gcc.ld").resolve(),
    )


def test_command_shapes(tmp_path: Path) -> None:
    tc = gcc_build.Toolchain(Path("C:/tc/bin"), "x")
    m = _mini_manifest(tmp_path)
    src = m.c_sources()[0]
    cmd = gcc_build.compile_command(tc, m, src, Path("build/main.o"))
    assert Path(cmd[0]).name.startswith("arm-none-eabi-gcc")
    assert cmd[1:6] == ["-mfloat-abi=hard", "-O1", "-ffunction-sections", "-fdata-sections", "-mthumb"]
    assert cmd[6:8] == ["-mcpu=cortex-m55", "-mthumb"]
    assert "-std=c11" in cmd and "-I." in cmd and "-DMINI=1" in cmd
    assert cmd[-6:-3] == ["-c", "-MMD", "-MP"] and cmd[-2] == "-o"
    ldcmd = gcc_build.preprocess_ld_command(tc, m, Path("build/Mini.generated.ld"))
    assert ldcmd[1:5] == ["-E", "-x", "c", "-P"]
    link = gcc_build.link_command(
        tc, m, [Path("build/extra.o"), Path("build/main.o")], Path("build/Mini.elf"),
        Path("build/Mini.map"), Path("build/Mini.generated.ld"),
    )
    assert link[-1] == "-Tbuild/Mini.generated.ld" and link[-2] == "-Wl,-Map=build/Mini.map,--cref"
    assert "-Wl,--start-group" in link and link[link.index("-Wl,--start-group") + 1] == "-lgcc"


def test_compile_command_prefix_map_and_relative_source(tmp_path: Path) -> None:
    tc = gcc_build.Toolchain(Path("C:/tc/bin"), "x")
    m = _mini_manifest(tmp_path)
    m.file_prefix_map = [(str(m.app_dir), "APP"), (str(m.app_dir / "GCC"), "APP_GCC")]
    src = m.c_sources()[0]
    cmd = gcc_build.compile_command(tc, m, src, Path("build/main.o"))
    assert f"-ffile-prefix-map={m.app_dir}=APP" in cmd
    assert f"-ffile-prefix-map={m.app_dir / 'GCC'}=APP_GCC" in cmd
    assert f"{src.path.parent}/{src.path.name}" in cmd
    # never on the ld preprocess command
    ldcmd = gcc_build.preprocess_ld_command(tc, m, Path("build/Mini.generated.ld"))
    assert not any(flag.startswith("-ffile-prefix-map=") for flag in ldcmd)


def test_parse_size_and_map(tmp_path: Path) -> None:
    text = (
        "   text\t   data\t    bss\t    dec\t    hex\tfilename\n"
        " 500680\t   6752\t 485280\t 992712\t  f25c8\tbuild/NN_ImgClassInference.elf\n"
        " 500680\t   6752\t 485280\t 992712\t  f25c8\t(TOTALS)\n"
    )
    assert gcc_build.parse_size_output(text) == (500680, 6752, 485280)
    map_text = (
        "Linker script and memory map\n\n"
        ".text           0x00100000    0x2936c\n"
        " *(.vectors)\n"
        ".sram01_hyperram\n"
        "                0x81f00400    0x26c00 load address 0x81f00000\n"
        ".NonCacheable.ZeroInit\n"
        "                0x81f00000      0x400\n"
        ".bss            0x20001638     0x1234\n"
    )
    (tmp_path / "x.map").write_text(map_text, encoding="utf-8")
    sections = gcc_build.parse_map_sections(tmp_path / "x.map")
    assert sections[".text"] == (0x00100000, 0x2936C)
    assert sections[".sram01_hyperram"] == (0x81F00400, 0x26C00)
    assert sections[".NonCacheable.ZeroInit"] == (0x81F00000, 0x400)


def test_build_mini_program(tmp_path: Path) -> None:
    tc = _toolchain()
    m = _mini_manifest(tmp_path)
    result = gcc_build.build(m, tc, timeout_seconds=300)
    assert result.ok, result.message
    assert result.bin is not None and result.bin.stat().st_size > 8
    assert result.elf is not None and result.map is not None and result.log_path.is_file()
    assert result.text > 0
    assert (m.app_dir / "GCC" / "build" / "Mini.generated.ld").is_file()


def test_build_reports_compile_error(tmp_path: Path) -> None:
    tc = _toolchain()
    m = _mini_manifest(tmp_path)
    # NOTE: arm-none-eabi-gcc 14.2 treats a bare `return ;` in a non-void function as a
    # semantic error ("'return' with no value, in function returning non-void"), not a
    # parse error. `return +;` is an actual syntax error and reliably yields "expected
    # expression" on this toolchain -- verified locally against the installed 14.2.1.
    (m.app_dir / "main.c").write_text("int broken(void) { return +; }\n", encoding="utf-8")
    result = gcc_build.build(m, tc, timeout_seconds=300)
    assert result.ok is False
    assert "main.c" in result.message and result.failed_command is not None
    assert "expected expression" in result.log_path.read_text(encoding="utf-8", errors="replace")


def test_build_handles_timeout_without_crashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build() runs inside the FastAPI JobManager worker thread in production, and Preview uses
    a TFLite interpreter in that same process -- an uncaught subprocess.TimeoutExpired escaping
    a hung gcc/ld must not happen. Every run_tool() call site inside build() must catch it and
    report a failed BuildResult instead."""
    tc = gcc_build.Toolchain(Path("C:/fake/bin"), "fake")
    m = _mini_manifest(tmp_path)

    def _always_times_out(cmd, cwd, env, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(gcc_build, "run_tool", _always_times_out)
    result = gcc_build.build(m, tc, timeout_seconds=5)
    assert result.ok is False
    assert "建置逾時" in result.message
    assert result.failed_command is not None
    assert result.log_path.is_file()
    assert "建置逾時" in result.log_path.read_text(encoding="utf-8", errors="replace")


def test_build_reports_a_raised_step_as_a_failed_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool that vanishes mid-build (OSError) must land as finish(False, ...) with a written
    build.log -- deploy_service copies that log out and quotes its tail to the student."""
    tc = gcc_build.Toolchain(Path("C:/fake/bin"), "fake")
    m = _mini_manifest(tmp_path)

    def _explodes(cmd, cwd, env, timeout):
        raise OSError("The system cannot find the file specified")  # a vanished binary

    monkeypatch.setattr(gcc_build, "run_tool", _explodes)
    result = gcc_build.build(m, tc, timeout_seconds=300)
    assert result.ok is False
    assert "OSError" in result.message and "建置失敗" in result.message
    assert result.log_path.is_file()
    assert "OSError" in result.log_path.read_text(encoding="utf-8", errors="replace")


def test_build_reports_an_unparsable_size_output_as_a_failed_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same for a post-compile step raising DeployError (parse_size_output on empty stdout)."""
    tc = gcc_build.Toolchain(Path("C:/fake/bin"), "fake")
    m = _mini_manifest(tmp_path)

    def _succeeds(cmd, cwd, env, timeout):
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(gcc_build, "run_tool", _succeeds)
    result = gcc_build.build(m, tc, timeout_seconds=300)
    assert result.ok is False
    assert "DeployError" in result.message
    assert "arm-none-eabi-size" in result.log_path.read_text(encoding="utf-8", errors="replace")


def test_build_log_never_carries_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build.log ships to the student inside the firmware ZIP, so the app/BSP/temp prefixes are
    rewritten exactly the way -ffile-prefix-map rewrites them for the compiler."""
    tc = gcc_build.Toolchain(tmp_path / "toolchain" / "bin", "fake")
    m = _mini_manifest(tmp_path)
    m.file_prefix_map = [(str(m.app_dir), "APP")]
    monkeypatch.setattr(gcc_build.mcu_paths, "MCU_TEMP_ROOT", tmp_path / "temp")

    def _fails(cmd, cwd, env, timeout):
        return subprocess.CompletedProcess(cmd, 1, f"{cmd[-3]}:1:20: error: expected expression")

    monkeypatch.setattr(gcc_build, "run_tool", _fails)
    result = gcc_build.build(m, tc, timeout_seconds=300)
    log = result.log_path.read_text(encoding="utf-8", errors="replace")
    assert result.ok is False
    assert str(m.app_dir) not in log
    assert str(tc.bin_dir) not in log
    assert "APP" in log and "TOOLCHAIN" in log
    assert "expected expression" in log  # the diagnostic itself survives untouched


def test_build_result_message_never_carries_the_app_dir_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: `result.message` goes straight into a `DeployError` the student reads --
    `finish()`'s scrub only covers the written build.log, so a raised exception whose own text
    embeds the app dir (e.g. a vanished-file OSError naming the path) must be scrubbed on the
    `failed()` path too, not just in the log file."""
    tc = gcc_build.Toolchain(Path("C:/fake/bin"), "fake")
    m = _mini_manifest(tmp_path)
    m.file_prefix_map = [(str(m.app_dir), "APP")]

    def _explodes(cmd, cwd, env, timeout):
        raise OSError(f"cannot find {m.app_dir / 'missing.o'}: No such file or directory")

    monkeypatch.setattr(gcc_build, "run_tool", _explodes)
    result = gcc_build.build(m, tc, timeout_seconds=300)
    assert result.ok is False
    assert str(m.app_dir) not in result.message
    assert "APP" in result.message
    assert "No such file or directory" in result.message  # the diagnostic itself survives


def test_build_times_out_when_deadline_already_elapsed(tmp_path: Path) -> None:
    """A remaining budget of <= 0 before a step even starts must fail as a timeout, not silently
    hand subprocess.run() a manufactured positive timeout (or crash trying to)."""
    tc = gcc_build.Toolchain(Path("C:/fake/bin"), "fake")
    m = _mini_manifest(tmp_path)
    result = gcc_build.build(m, tc, timeout_seconds=0)
    assert result.ok is False
    assert "建置逾時" in result.message


def test_spike_x_imgclass_bytes_match(tmp_path: Path) -> None:
    """Ground truth, per spec Sec5.4: build the frozen spike sources against the UNPATCHED
    upstream NuML template (the shipped/patched template drops hyperram_code.c, which the frozen
    BoardInit.cpp still calls -- see task-4-report.md Sec4a) and reproduce the spike .bin
    byte-for-byte via -ffile-prefix-map overrides matching the spike's own relative-path build
    cwd (<template>/projectfiles/make_gcc_arm; see make_n_B_x_imgclass.txt), so the result does
    not depend on where this Studio checkout or the upstream NuML_Toolkit are installed
    (task-4-report.md Sec4b)."""
    from tm_local.mcu import paths

    tc = _toolchain()
    upstream_raw = os.environ.get(UPSTREAM_ENV_VAR)
    if not upstream_raw:
        pytest.skip(f"set {UPSTREAM_ENV_VAR} to a NuML_TFLM_Tool checkout to run this test")
    upstream_root = Path(upstream_raw)
    if not upstream_root.is_dir():
        pytest.skip(
            f"unpatched upstream NuML tree not found at {upstream_root}; "
            f"set {UPSTREAM_ENV_VAR} to a NuML_TFLM_Tool checkout"
        )
    template_rel = paths.template_dir("NuMaker-M55M1", "imgclass").relative_to(paths.NUML_ROOT)
    template = upstream_root / template_rel
    upstream_bsp_root = (upstream_root / "templates" / "M55M1BSP").resolve()
    app = tmp_path / "NN_ImgClassInference"
    shutil.copytree(FIXTURE / "NN_ImgClassInference", app)
    app = app.resolve()
    m = records.load_records(template, app, upstream_bsp_root)
    # The spike's make ran from <template>/projectfiles/make_gcc_arm, so its relative record
    # paths are exactly 4 levels up to the BSP root and 2 levels up to the app dir.
    m.file_prefix_map = [
        (str(upstream_bsp_root), r"..\..\..\.."),
        (str(app), r"..\..\NN_ImgClassInference"),
    ]

    # progen's ArmMLApi.yaml record names ..\Model.cpp, but the BSP has tracked this file as
    # Model.cc since V3.01.001 (clean upstream git history -- no Model.cpp anywhere in the
    # tree), and yet the spike's own make log (make_n_B_x_imgclass.txt) shows it was compiled
    # as "...\application\api\common\source/Model.cpp" -- so the spike's build tree carried a
    # Model.cpp copy of this file that upstream never tracked. Production code does not need
    # to reproduce that: records.resolve_record_path()'s .cpp -> .cc fallback finds and
    # compiles the real Model.cc under its real name (see test_cpp_record_maps_to_cc_on_disk),
    # which is what every non-parity build path uses. Only this historical byte-for-byte
    # parity test needs the untracked Model.cpp copy, so it aliases the real file under that
    # name here instead of changing how records.py resolves .cpp/.cc in general.
    common_source_dir = r"application\api\common\source"
    model_matches = [
        s for s in m.sources
        if s.path.name == "Model.cc" and str(s.path.parent).endswith(common_source_dir)
    ]
    assert len(model_matches) == 1, (
        f"expected exactly one Model.cc under {common_source_dir}, found {model_matches}"
    )
    model_cc = model_matches[0]
    alias_dir = tmp_path / "spike_alias"
    alias_dir.mkdir()
    alias_cpp = alias_dir / "Model.cpp"
    shutil.copyfile(model_cc.path, alias_cpp)
    m.sources = [s for s in m.sources if s is not model_cc]
    m.sources.append(records.SourceFile(alias_cpp.resolve(), "cxx", model_cc.progen_relpath))
    m.file_prefix_map.append((
        str(alias_dir.resolve()),
        r"..\..\..\..\ThirdParty\ml-embedded-evaluation-kit\source\application\api\common\source",
    ))

    m.check_object_collisions()
    result = gcc_build.build(m, tc, timeout_seconds=900, jobs=8)
    assert result.ok, result.message
    expected_sha = (FIXTURE / "reference_bin.sha256").read_text().strip()
    expected_size = int((FIXTURE / "reference_bin.size").read_text().strip())
    assert result.bin.stat().st_size == expected_size
    assert hashlib.sha256(result.bin.read_bytes()).hexdigest() == expected_sha
