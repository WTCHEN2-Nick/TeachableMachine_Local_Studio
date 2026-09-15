"""Real two-board firmware build for an image project.

Everything here needs the vendored `mcu_toolkit/` plus a real Arm GNU Toolchain; each case
skips (never fails) when either is missing. The builds are genuine -- ~40-90 s per board --
and are the acceptance evidence for the deploy pipeline.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from tm_local.mcu import boards, contract, gcc_build, paths, project_builder, toolchain, vela
from tm_local.mcu.errors import DeployError, long_path_reason
from tm_local.mcu.project_builder import ARENA_DEFINE
from tm_local.project_store import ProjectStore

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mcu"
BOARD_NAMES = ["NuMaker-M55M1", "NuGestureAI-M55M1"]


def _skip_unless_ready() -> None:
    if not paths.toolkit_available() or not paths.VELA_EXE.is_file():
        pytest.skip("mcu_toolkit not vendored")
    # A Studio installed too deep for Windows MAX_PATH makes arm-none-eabi-gcc fail to
    # resolve the deepest BSP headers, with an English "No such file or directory" that has
    # nothing to do with what these cases assert. Skip with the reason a student would get.
    reason = long_path_reason(paths.MCU_TOOLKIT_ROOT)
    if reason:
        pytest.skip(reason)
    if toolchain.find_toolchain() is None:
        pytest.skip("arm-none-eabi-gcc missing")
    if not (paths.APPS_ROOT / "image" / "NuGestureAI-M55M1" / "main.cpp.in").is_file():
        pytest.skip("image templates not vendored")


def _contract(board_name: str, tmp_path: Path) -> contract.DeployContract:
    int8 = tmp_path / "models" / "image_classifier_int8.tflite"
    int8.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURES / "vww4_128_128_INT8.tflite", int8)
    return contract.DeployContract(
        kind="image",
        application="imgclass",
        board=boards.board_for(board_name),
        project_id="p",
        project_name="VWW Test",
        labels=["not_person", "person"],
        models_dir=int8.parent,
        int8_path=int8,
        model_sha256=hashlib.sha256(int8.read_bytes()).hexdigest(),
        input=contract.TensorInfo("in", (1, 128, 128, 3), "int8", 1.0, -128),
        output=contract.TensorInfo("out", (1, 2), "int8", 0.00390625, -128),
        image_size=128,
        audio_frontend=None,
        yamnet_frontend=None,
        encoder_depth=None,
        detection_threshold=0.5,
        class_thresholds=[0.5, 0.5],
        hop_seconds=0.5,
        clip_seconds=1.0,
        peak_hold_seconds=1.5,
    )


def _compile_commands(assembled: project_builder.AssembledProject) -> list[list[str]]:
    tc = gcc_build.Toolchain(Path("C:/tc/bin"), "probe")
    manifest = assembled.manifest
    return [
        gcc_build.compile_command(tc, manifest, src, Path("build") / manifest.object_name(src))
        for src in manifest.sources
    ]


def test_app_module_rejects_unknown_application() -> None:
    with pytest.raises(DeployError):
        project_builder.app_module("does_not_exist")


@pytest.mark.parametrize("board", BOARD_NAMES)
def test_assemble_and_build_image(board: str, tmp_path: Path) -> None:
    _skip_unless_ready()
    c = _contract(board, tmp_path)
    contract.validate(c)
    v = vela.compile_model(c.int8_path, tmp_path / "vela")
    assembled = project_builder.assemble(c, v, tmp_path / "work")
    assert assembled.project_name == "NN_ImgClassInference"
    assert assembled.app_dir == tmp_path / "work" / "NN_ImgClassInference"
    assert (assembled.app_dir / "Model" / "NN_Model_INT8.tflite.cpp").is_file()
    assert (assembled.app_dir / "Model" / "include" / "MobileNetModel.hpp").is_file()
    labels_cpp = (assembled.app_dir / "Model" / "Labels.cpp").read_text(encoding="utf-8")
    assert labels_cpp.count('"person"') == 1 and '"not_person"' in labels_cpp
    assert (assembled.app_dir / "GCC" / "gcc.ld").read_text(encoding="utf-8").count(
        "SRAM_NONCACHEABLE"
    ) >= 1
    assert not (assembled.app_dir / "KEIL").exists()
    main = (assembled.app_dir / "main.cpp").read_text(encoding="utf-8")
    assert f"#define ACTIVATION_BUF_SZ ({v.arena_bytes})" in main

    names = {s.path.name for s in assembled.manifest.sources}
    commands = _compile_commands(assembled)
    # The vendored template hard-codes a 1 MiB arena as a -D. It must be REPLACED (not merely
    # removed -- BufAttributes.hpp #warning's when it is undefined) by the real Vela arena, and
    # exactly once, so every translation unit agrees with main.cpp's own #undef/#define.
    for cmd in commands:
        arena_flags = [f for f in cmd if f.startswith(f"-D{ARENA_DEFINE}")]
        assert arena_flags == [f"-D{ARENA_DEFINE}={v.arena_bytes}"], arena_flags
    assert not any("0x100000" in flag for cmd in commands for flag in cmd)
    if board == "NuGestureAI-M55M1":
        assert {"numl_cdc.c", "numl_cdc_retarget.c", "numl_overlay.c"} <= names
        # numl_cdc_retarget.c textually #includes the BSP retarget.c, so the BSP copy must
        # not be compiled separately or stdout_putchar()/_write() are defined twice at link.
        assert "retarget.c" not in names
        assert not any(cmd[-3].endswith("/retarget.c") for cmd in commands)
    else:
        assert "retarget.c" in names and "numl_cdc.c" not in names

    tc = toolchain.find_toolchain()
    result = gcc_build.build(assembled.manifest, tc, timeout_seconds=900, jobs=8)
    assert result.ok, result.message + "\n" + result.log_path.read_text(
        encoding="utf-8", errors="replace"
    )[-3000:]
    assert 300_000 < result.bin.stat().st_size < 2_097_152
    assert 0 < result.sram01_used <= 1_048_576
    assert result.noncacheable_used >= 0x400  # sdh.o DMA buffers live in the non-cacheable region

    # Firmware bytes must not depend on where the project was assembled: records.py's default
    # -ffile-prefix-map rewrites the BSP/app prefixes, so a rebuild from a different work dir
    # has to produce a byte-identical .bin.
    again = project_builder.assemble(c, v, tmp_path / "work-elsewhere-with-a-longer-name")
    second = gcc_build.build(again.manifest, tc, timeout_seconds=900, jobs=8)
    assert second.ok, second.message
    assert hashlib.sha256(second.bin.read_bytes()).hexdigest() == hashlib.sha256(
        result.bin.read_bytes()
    ).hexdigest()


def _trained_image_project(store: ProjectStore, models_src: Path) -> str:
    """Seed a trained+exported image project whose INT8 artifact is the vww4 fixture."""
    project = store.create_project("image", "VWW")  # kind first; seeds two default classes
    raw = store.get_raw_project(project["id"])
    for class_item, name in zip(raw["classes"], ("not_person", "person")):
        store.update_class(project["id"], class_item["id"], {"name": name})
    models = store.project_dir(project["id"]) / "models"
    models.mkdir(parents=True, exist_ok=True)
    int8 = models / "image_classifier_int8.tflite"
    shutil.copy2(models_src, int8)
    (models / "image_classifier.keras").write_bytes(b"fake")
    entry = {
        "path": "image_classifier_int8.tflite",
        "sha256": hashlib.sha256(int8.read_bytes()).hexdigest(),
        "inputs": [
            {"name": "in", "shape": [1, 128, 128, 3], "dtype": "int8", "scale": 1.0,
             "zero_point": -128}
        ],
        "outputs": [
            {"name": "out", "shape": [1, 2], "dtype": "int8", "scale": 0.00390625,
             "zero_point": -128}
        ],
        "strict_full_integer": True,
        "integer_input_output": True,
    }
    store.set_training_result(
        project["id"],
        report={
            "project_kind": "image",
            "settings": {"image_size": 128},
            "conversion": {"state": "generated", "models": {"int8": entry}},
        },
        artifacts={"keras": "image_classifier.keras", "int8": "image_classifier_int8.tflite"},
    )
    return project["id"]


def test_run_deploy_end_to_end(store: ProjectStore, tmp_path: Path, monkeypatch) -> None:
    """Full run_deploy() on a seeded image project; export is stubbed out to skip TensorFlow."""
    _skip_unless_ready()
    from tm_local.mcu import deploy_service as svc

    project_id = _trained_image_project(store, FIXTURES / "vww4_128_128_INT8.tflite")
    models = store.project_dir(project_id) / "models"
    export_calls: list[tuple] = []
    monkeypatch.setattr(
        svc,
        "ensure_export_artifacts",
        lambda *a, **k: export_calls.append((a, k)) or {"generated": {}},
    )
    monkeypatch.setattr(paths, "MCU_TEMP_ROOT", tmp_path / "tmp")

    seen: list[tuple[float, str]] = []
    outcome = svc.run_deploy(
        store, project_id, "NuGestureAI-M55M1", lambda v, m: seen.append((v, m))
    )
    messages = [m for _v, m in seen]

    assert export_calls and export_calls[0][0][1:3] == (project_id, ["int8"])
    assert outcome.board == "NuGestureAI-M55M1"
    assert outcome.bin_path.is_file() and outcome.zip_path.is_file()
    assert outcome.report["image"]["flash_used"] == outcome.bin_path.stat().st_size
    assert outcome.report["schema_version"] == 1
    assert outcome.report["kind"] == "image" and outcome.report["application"] == "imgclass"
    assert outcome.report["labels"] == ["not_person", "person"]
    assert outcome.report["flash_method"] == "msc"
    # One source of truth for the flashing steps: the report carries the text the UI renders.
    assert outcome.report["flash_instructions"] == svc.flash_instructions(
        boards.board_for("NuGestureAI-M55M1"), "image"
    )
    assert outcome.report["vela"]["arena_bytes"] > 0
    assert outcome.report["bin"]["sha256"] == hashlib.sha256(
        outcome.bin_path.read_bytes()
    ).hexdigest()
    assert outcome.report["image"]["flash_limit"] == 2_097_152
    assert outcome.report["image"]["sram01_limit"] == 1_048_576
    assert outcome.report["model"]["file"] == "image_classifier_int8.tflite"
    assert outcome.report["toolchain"]["version"]

    target = svc.deploy_dir(models, "NuGestureAI-M55M1")
    assert target == models / "mcu" / "NuGestureAI-M55M1"
    for name in ("deploy_report.json", "firmware.bin", "firmware.elf", "firmware.map",
                 "firmware.hex", "build.log", "labels.txt", "README_FLASH_zh-TW.txt",
                 "source.zip"):
        assert (target / name).is_file(), name
    assert (target / "labels.txt").read_text(encoding="utf-8") == "0 not_person\n1 person\n"
    assert "隨身碟" in (target / "README_FLASH_zh-TW.txt").read_text(encoding="utf-8")

    assert outcome.zip_path.name == "VWW_NuGestureAI-M55M1_firmware.zip"
    with zipfile.ZipFile(outcome.zip_path) as archive:
        assert sorted(archive.namelist()) == sorted([
            "firmware.bin", "firmware.elf", "firmware.map", "build.log", "deploy_report.json",
            "labels.txt", "README_FLASH_zh-TW.txt", "source.zip",
        ])
    with zipfile.ZipFile(target / "source.zip") as archive:
        members = archive.namelist()
    assert "main.cpp" in members and "Model/NN_Model_INT8.tflite.cpp" in members
    assert not any(name.startswith("GCC/build/") for name in members)

    assert (tmp_path / "tmp").is_dir() and not any((tmp_path / "tmp").iterdir())
    assert not target.with_name(f"{target.name}.tmp").exists()  # staging dir swapped in, not left
    assert any("Vela" in m for m in messages) and any("編譯" in m for m in messages)
    assert [v for v, _m in seen] == sorted(v for v, _m in seen)
    assert seen[0][0] == 0.0 and seen[-1][0] == 1.0

    # build.log travels inside source-less hands (the ZIP is shared with a teacher): no absolute
    # path from this machine, and therefore no Windows username, may appear in it.
    build_log = (target / "build.log").read_text(encoding="utf-8", errors="replace")
    assert str(tmp_path) not in build_log
    assert str(paths.BSP_ROOT) not in build_log and "BSP" in build_log

    reports = svc.read_deploy_reports(models)
    assert reports["NuGestureAI-M55M1"]["bin"]["bytes"] == outcome.bin_path.stat().st_size

    # A failed REBUILD must not destroy the firmware the student already has: only build.log
    # is refreshed, everything else stays until a new build has fully succeeded.
    previous_bin = outcome.bin_path.read_bytes()

    def _times_out(cmd, cwd, env, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(gcc_build, "run_tool", _times_out)
    with pytest.raises(DeployError, match="建置逾時"):
        svc.run_deploy(store, project_id, "NuGestureAI-M55M1", lambda v, m: None)
    assert outcome.bin_path.read_bytes() == previous_bin
    assert outcome.zip_path.is_file()
    assert svc.read_deploy_reports(models)["NuGestureAI-M55M1"]["bin"]["sha256"] == (
        outcome.report["bin"]["sha256"]
    )
    assert "建置逾時" in (target / "build.log").read_text(encoding="utf-8", errors="replace")
    assert not target.with_name(f"{target.name}.tmp").exists()


def test_run_deploy_reports_missing_toolchain(store: ProjectStore, tmp_path: Path,
                                              monkeypatch) -> None:
    if not paths.toolkit_available():
        pytest.skip("mcu_toolkit not vendored")
    from tm_local.mcu import deploy_service as svc

    project_id = _trained_image_project(store, FIXTURES / "vww4_128_128_INT8.tflite")
    monkeypatch.setattr(svc, "ensure_export_artifacts", lambda *a, **k: {"generated": {}})
    monkeypatch.setattr(svc, "find_toolchain", lambda: None)
    monkeypatch.setattr(paths, "MCU_TEMP_ROOT", tmp_path / "tmp")
    with pytest.raises(DeployError, match="Arm GNU Toolchain"):
        svc.run_deploy(store, project_id, "NuGestureAI-M55M1", lambda v, m: None)
    assert not (tmp_path / "tmp").exists()  # never created: we failed before the work dir


def test_run_deploy_surfaces_build_failure(store: ProjectStore, tmp_path: Path,
                                           monkeypatch) -> None:
    """A failed gcc build must raise DeployError and still leave build.log for the student."""
    _skip_unless_ready()
    from tm_local.mcu import deploy_service as svc

    project_id = _trained_image_project(store, FIXTURES / "vww4_128_128_INT8.tflite")
    models = store.project_dir(project_id) / "models"
    monkeypatch.setattr(svc, "ensure_export_artifacts", lambda *a, **k: {"generated": {}})
    monkeypatch.setattr(paths, "MCU_TEMP_ROOT", tmp_path / "tmp")

    def _fail(cmd, cwd, env, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(gcc_build, "run_tool", _fail)
    with pytest.raises(DeployError, match="建置逾時"):
        svc.run_deploy(store, project_id, "NuMaker-M55M1", lambda v, m: None)
    assert (svc.deploy_dir(models, "NuMaker-M55M1") / "build.log").is_file()
    assert not any((tmp_path / "tmp").iterdir())


def _board_or_skip(name: str) -> boards.Board:
    if not paths.BOARDS_JSON.is_file():
        pytest.skip("mcu_toolkit not vendored")
    return boards.board_for(name)


def _failed_result(tmp_path: Path, message: str, log: str) -> gcc_build.BuildResult:
    log_path = tmp_path / "build.log"
    log_path.write_text(log, encoding="utf-8")
    return gcc_build.BuildResult(ok=False, message=message, log_path=log_path)


# Real arm-none-eabi-ld 14.2 wordings, both shapes, for each region the linker script declares.
_SRAM01_LOG = (
    "arm-none-eabi/bin/ld.exe: NN_ImgClassInference.elf section `.sram01_hyperram' "
    "will not fit in region `SRAM01_HYPERRAM'\n"
    "arm-none-eabi/bin/ld.exe: region `SRAM01_HYPERRAM' overflowed by 262144 bytes\n"
)
_FLASH_LOG = "arm-none-eabi/bin/ld.exe: region `FLASH' overflowed by 51200 bytes\n"
_DTCM_LOG = (
    "arm-none-eabi/bin/ld.exe: NN_ImgClassInference.elf section `.bss' "
    "will not fit in region `DTCM'\n"
)


def test_link_overflow_message_names_sram01(tmp_path: Path) -> None:
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuMaker-M55M1")
    result = _failed_result(tmp_path, "連結失敗；記憶體區段溢出（見 build.log）", _SRAM01_LOG)
    with pytest.raises(DeployError) as excinfo:
        svc._raise_build_failure(result, board, 900_000, "image")
    text = str(excinfo.value)
    assert "SRAM01" in text and "1,048,576" in text and "900,000" in text
    assert "262,144" in text  # the linker's own overflow figure
    assert "flash" not in text.lower()


def test_link_overflow_message_names_flash(tmp_path: Path) -> None:
    """A model too big for FLASH never produces a .bin, so _check_limits can never see it --
    the student must be told about flash here, not about SRAM01."""
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuGestureAI-M55M1")
    result = _failed_result(tmp_path, "連結失敗；記憶體區段溢出（見 build.log）", _FLASH_LOG)
    with pytest.raises(DeployError) as excinfo:
        svc._raise_build_failure(result, board, 900_000, "image")
    text = str(excinfo.value)
    assert "flash" in text and "2,097,152" in text and "51,200" in text
    assert "SRAM01" not in text and "tensor arena" not in text


def test_link_overflow_unknown_region_falls_back_to_the_build_message(tmp_path: Path) -> None:
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuMaker-M55M1")
    result = _failed_result(tmp_path, "連結失敗；記憶體區段溢出（見 build.log）", _DTCM_LOG)
    with pytest.raises(DeployError) as excinfo:
        svc._raise_build_failure(result, board, 900_000, "image")
    text = str(excinfo.value)
    assert "記憶體區段 DTCM" in text
    assert "SRAM01" not in text and "2,097,152" not in text


@pytest.mark.parametrize(
    ("message", "log"),
    [
        ("編譯失敗：main.cpp（見 build.log）", "main.cpp:12:5: error: expected expression\n"),
        ("建置逾時（1200 秒），已終止（見 build.log）", "[TIMEOUT] 建置逾時（1200 秒）\n"),
    ],
)
def test_non_overflow_failures_pass_the_message_through(
    message: str, log: str, tmp_path: Path
) -> None:
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuMaker-M55M1")
    with pytest.raises(DeployError) as excinfo:
        svc._raise_build_failure(_failed_result(tmp_path, message, log), board, 900_000, "image")
    text = str(excinfo.value)
    assert text.startswith(message) and log.strip() in text
    assert "SRAM01" not in text and "tensor arena" not in text


def _ok_result(*, flash: int, sram01: int) -> gcc_build.BuildResult:
    return gcc_build.BuildResult(
        ok=True, message="建置完成", log_path=Path("build.log"),
        flash_used=flash, sram01_used=sram01,
    )


def test_check_limits_accepts_a_firmware_inside_both_budgets() -> None:
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuMaker-M55M1")
    svc._check_limits(_ok_result(flash=2_097_152, sram01=1_048_576), board, "image")  # at both


def test_check_limits_rejects_oversized_flash() -> None:
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuMaker-M55M1")
    with pytest.raises(DeployError, match="2,097,153"):
        svc._check_limits(_ok_result(flash=2_097_153, sram01=1_000), board, "image")


def test_check_limits_rejects_oversized_sram01() -> None:
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuGestureAI-M55M1")
    with pytest.raises(DeployError, match="SRAM01"):
        svc._check_limits(_ok_result(flash=1_000, sram01=1_048_577), board, "image")


def _vela_result(flash_bytes: int) -> vela.VelaResult:
    """A synthetic Vela verdict: only `flash_bytes` matters to `_check_flash_budget()`."""
    return vela.VelaResult(
        vela_tflite=Path("model_vela.tflite"),
        summary_csv=Path("summary.csv"),
        sram_bytes=151_360,
        flash_bytes=flash_bytes,
        arena_bytes=182_272,
        optimise="Performance",
        npu_ops=1,
        cpu_ops=0,
        log_tail="",
    )


def test_flash_budget_accepts_a_model_that_fits_with_the_code_baseline() -> None:
    """A depth-12 known_sound model (1,563,552 B after Vela) has to pass on the X board."""
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuMaker-M55M1")
    svc._check_flash_budget(_vela_result(1_563_552), board, "known_sound")
    # Exactly at the limit is still a fit, not a refusal.
    svc._check_flash_budget(
        _vela_result(board.internal_flash_bytes - svc.FIRMWARE_CODE_BASELINE_BYTES),
        board,
        "image",
    )


def test_flash_budget_rejects_a_deep_known_sound_model_before_gcc() -> None:
    """Depth 13 (2,066,832 B after Vela) cannot fit 2 MiB, and the advice must name the knob."""
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuMaker-M55M1")
    with pytest.raises(DeployError) as excinfo:
        svc._check_flash_budget(_vela_result(2_066_832), board, "known_sound")
    text = str(excinfo.value)
    assert "2,066,832" in text and "2,097,152" in text
    assert "encoder_depth" in text and str(board.known_sound_max_depth) in text
    assert "image_size" not in text


@pytest.mark.parametrize(
    ("kind", "expected", "forbidden"),
    [
        ("known_sound", "encoder_depth", "image_size"),
        ("image", "image_size", "encoder_depth"),
        ("audio", "韌體程式碼本身", "encoder_depth"),
    ],
)
def test_shrink_advice_names_this_kinds_knob(
    kind: str, expected: str, forbidden: str, tmp_path: Path
) -> None:
    from tm_local.mcu import deploy_service as svc

    board = _board_or_skip("NuMaker-M55M1")
    advice = svc._shrink_advice(kind, board)
    assert expected in advice and forbidden not in advice
    # The same sentence has to reach the two link-failure branches and the two post-link
    # checks, otherwise only one of the four paths is kind-aware.
    for log in (_FLASH_LOG, _SRAM01_LOG):
        result = _failed_result(tmp_path, "連結失敗；記憶體區段溢出（見 build.log）", log)
        with pytest.raises(DeployError) as excinfo:
            svc._raise_build_failure(result, board, 900_000, kind)
        assert advice in str(excinfo.value)
    with pytest.raises(DeployError) as excinfo:
        svc._check_limits(_ok_result(flash=2_097_153, sram01=1_000), board, kind)
    assert advice in str(excinfo.value)
    with pytest.raises(DeployError) as excinfo:
        svc._check_limits(_ok_result(flash=1_000, sram01=1_048_577), board, kind)
    assert advice in str(excinfo.value)


def test_flash_instructions_cover_both_methods() -> None:
    if not paths.toolkit_available():
        pytest.skip("mcu_toolkit not vendored")
    from tm_local.mcu import deploy_service as svc

    msc = svc.flash_instructions(boards.board_for("NuGestureAI-M55M1"))
    nulink = svc.flash_instructions(boards.board_for("NuMaker-M55M1"))
    assert "隨身碟" in msc and "firmware.bin" in msc
    assert "Nu-Link" in nulink and "APROM" in nulink
    # The kind defaults to "image" so app.py's flash-failure fallback keeps working.
    assert msc == svc.flash_instructions(boards.board_for("NuGestureAI-M55M1"), "image")


def test_flash_instructions_follow_the_chosen_method_not_the_boards_default() -> None:
    """NuGestureAI-M55M1's default is msc, but the user proved Nu-Link works on it too --
    passing method="nulink" must describe the Nu-Link steps, not silently fall back to msc."""
    if not paths.toolkit_available():
        pytest.skip("mcu_toolkit not vendored")
    from tm_local.mcu import deploy_service as svc

    board = boards.board_for("NuGestureAI-M55M1")
    assert board.flash_method == "msc"  # sanity: msc is still the board's default
    default_text = svc.flash_instructions(board, "image")
    chosen_text = svc.flash_instructions(board, "image", "nulink")
    assert default_text != chosen_text
    assert "隨身碟" in default_text and "Nu-Link" not in default_text.split("結果觀看")[0]
    assert "Nu-Link" in chosen_text and "APROM" in chosen_text
    # Explicitly passing the board's own default reproduces the two-argument call exactly.
    assert svc.flash_instructions(board, "image", "msc") == default_text


@pytest.mark.parametrize("board_name", BOARD_NAMES)
def test_flash_instructions_describe_the_right_output_per_kind(board_name: str) -> None:
    """Only the image app has a picture; the audio apps print to a COM port and nothing else."""
    if not paths.toolkit_available():
        pytest.skip("mcu_toolkit not vendored")
    from tm_local.mcu import deploy_service as svc

    board = boards.board_for(board_name)
    texts = {kind: svc.flash_instructions(board, kind) for kind in ("image", "known_sound", "audio")}
    # Same board, same flashing steps -- only the "what you will see" paragraph differs. The
    # board LABEL mentions the LCD on both boards, so every claim below is about that paragraph
    # alone, never about the whole message.
    steps = texts["image"].split("結果觀看")[0]
    parts = {}
    for kind, text in texts.items():
        assert text.startswith(steps), kind
        parts[kind] = text[len(steps):]
    image, known_sound, kws = parts["image"], parts["known_sound"], parts["audio"]
    assert ("LCD" in image) == board.has_lcd
    assert ("相機 app" in image) != board.has_lcd
    for text in (known_sound, kws):
        # Never "open the camera app" / "watch the LCD": those windows stay empty here. (The
        # GestureAI wording does say there is no camera, which is the opposite instruction.)
        assert "LCD" not in text and "相機 app" not in text and "攝影機影像" not in text
        assert "COM port" in text and "115200 8N1" in text and "PuTTY" in text
    assert "INFO threshold[" in known_sound and "live rms" in known_sound
    assert "KWS DETECTED" in kws and "KWS hop=" in kws
    assert "KWS" not in known_sound and "threshold" not in kws
    # The GestureAI audio builds enumerate a plain CDC port, with no camera function.
    if not board.has_lcd:
        assert "PID 0x1105" in known_sound and "PID 0x1105" in kws


def test_read_deploy_reports_tolerates_missing_and_broken(tmp_path: Path) -> None:
    from tm_local.mcu import reports

    assert reports.read_deploy_reports(tmp_path / "nope") == {}
    good = tmp_path / "mcu" / "BoardA"
    bad = tmp_path / "mcu" / "BoardB"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (good / reports.REPORT_NAME).write_text('{"board": "BoardA"}', encoding="utf-8")
    (bad / reports.REPORT_NAME).write_text("{not json", encoding="utf-8")
    assert reports.read_deploy_reports(tmp_path) == {"BoardA": {"board": "BoardA"}}


def test_read_deploy_reports_skips_tmp_swap_directories(tmp_path: Path) -> None:
    """R5: a hard kill during the rebuild swap can leave `models/mcu/<board>.tmp` behind
    (see `deploy_service.deploy_dir()`'s sibling `.tmp` staging dir) -- it is not a board."""
    from tm_local.mcu import reports

    good = tmp_path / "mcu" / "BoardA"
    stale_tmp = tmp_path / "mcu" / "BoardA.tmp"
    good.mkdir(parents=True)
    stale_tmp.mkdir(parents=True)
    (good / reports.REPORT_NAME).write_text('{"board": "BoardA"}', encoding="utf-8")
    (stale_tmp / reports.REPORT_NAME).write_text('{"board": "BoardA"}', encoding="utf-8")
    assert reports.read_deploy_reports(tmp_path) == {"BoardA": {"board": "BoardA"}}
