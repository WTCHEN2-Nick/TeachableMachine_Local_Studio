"""Real two-board firmware build for a `known_sound` project.

The token helpers and the record overlay are plain unit tests; the two build cases are
genuine `arm-none-eabi-gcc` builds of the vendored generic template (~60-120 s per board,
twice per board for the determinism check) and are the acceptance evidence for the
known_sound firmware app. Every case skips -- never fails -- when `mcu_toolkit/`, Vela or
the Arm GNU Toolchain is missing.

This file is also the first real exercise of `codegen.generate_model_sources()` on the
GenericCodegen path (`Model/NNModel.{hpp,cpp}`); `test_mcu_image_build.py` only ever drove
the imgclass codegen.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tm_local.mcu import boards, contract, gcc_build, paths, project_builder, toolchain, vela
from tm_local.mcu.apps import known_sound as ks_app
from tm_local.mcu.errors import DeployError, long_path_reason
from tm_local.mcu.project_builder import ARENA_DEFINE
from tm_local.project_store import ProjectStore
from tm_local.yamnet_model import frontend_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "mcu"
BOARD_NAMES = ["NuMaker-M55M1", "NuGestureAI-M55M1"]
LABELS = ["gun", "dog", "Background"]

# Set TM_MCU_KEEP_BIN=1 to drop the built firmware into the git-ignored
# workspace/deliverables/ for manual flashing; the default run leaves the repo untouched.
KEEP_BIN = os.environ.get("TM_MCU_KEEP_BIN") == "1"

pytestmark = pytest.mark.skipif(
    not paths.BOARDS_JSON.is_file(), reason="mcu_toolkit not vendored"
)


def _entry() -> dict[str, Any]:
    return json.loads((FIXTURES / "known_sound3_int8.report.json").read_text(encoding="utf-8"))


def _tensor(spec: dict[str, Any]) -> contract.TensorInfo:
    return contract.TensorInfo(
        spec["name"],
        tuple(spec["shape"]),
        spec["dtype"],
        float(spec["scale"]),
        int(spec["zero_point"]),
    )


def _contract(board_name: str, tmp_path: Path) -> contract.DeployContract:
    entry = _entry()
    int8 = tmp_path / "models" / "known_sound_yamnet_classifier_d6_int8.tflite"
    int8.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURES / "known_sound3_int8.tflite", int8)
    return contract.DeployContract(
        kind="known_sound",
        application="known_sound",
        board=boards.board_for(board_name),
        project_id="p",
        project_name="Known Sound Test",
        labels=list(LABELS),
        models_dir=int8.parent,
        int8_path=int8,
        model_sha256=hashlib.sha256(int8.read_bytes()).hexdigest(),
        input=_tensor(entry["inputs"][0]),
        output=_tensor(entry["outputs"][0]),
        image_size=None,
        audio_frontend=None,
        yamnet_frontend=frontend_contract(),
        encoder_depth=6,
        detection_threshold=0.5,
        class_thresholds=[0.6, 0.5, 0.9],
        hop_seconds=0.5,
        clip_seconds=1.0,
        peak_hold_seconds=1.5,
    )


# --------------------------------------------------------------------------- tokens


def test_tokens_for(tmp_path: Path) -> None:
    c = _contract("NuMaker-M55M1", tmp_path)
    tokens = ks_app.tokens_for(c, 183296)
    assert tokens["ARENA"] == "183296" and tokens["CLASS_COUNT"] == "3"
    assert tokens["CLIP_SAMPLES"] == "16000" and tokens["HOP_SAMPLES"] == "8000"
    assert tokens["HOLD_HOPS"] == "3"
    assert tokens["LABELS"] == '    "gun",\n    "dog",\n    "Background",'
    assert tokens["THRESHOLDS"] == "    0.600000f,\n    0.500000f,\n    0.900000f,"


def test_tokens_for_covers_exactly_the_templates_placeholders(tmp_path: Path) -> None:
    """Neither more nor fewer tokens than `main.cpp.in` declares.

    `render_tokens()` refuses a leftover `@@TOKEN@@`, so a missing key is a deploy-time
    failure; a surplus key is silently ignored, which is how a renamed placeholder would
    otherwise keep an obsolete value alive here.
    """
    template = paths.APPS_ROOT / "known_sound" / "main.cpp.in"
    if not template.is_file():
        pytest.skip("known_sound pack not vendored")
    placeholders = set(re.findall(r"@@([A-Z0-9_]+)@@", template.read_text(encoding="utf-8")))
    assert set(ks_app.tokens_for(_contract("NuMaker-M55M1", tmp_path), 4096)) == placeholders


def test_tokens_for_rejects_a_threshold_per_class_mismatch(tmp_path: Path) -> None:
    c = _contract("NuMaker-M55M1", tmp_path)
    c.class_thresholds = [0.5, 0.5]
    with pytest.raises(DeployError, match="門檻"):
        ks_app.tokens_for(c, 4096)


def test_hold_hops_is_at_least_one_hop(tmp_path: Path) -> None:
    """A peak hold shorter than one hop still has to hold for a hop, not zero."""
    c = _contract("NuMaker-M55M1", tmp_path)
    c.peak_hold_seconds = 0.1
    c.hop_seconds = 0.5
    assert ks_app.tokens_for(c, 4096)["HOLD_HOPS"] == "1"


def test_hop_samples_round_to_whole_samples(tmp_path: Path) -> None:
    c = _contract("NuMaker-M55M1", tmp_path)
    c.hop_seconds = 0.25
    tokens = ks_app.tokens_for(c, 4096)
    assert tokens["CLIP_SAMPLES"] == "16000" and tokens["HOP_SAMPLES"] == "4000"


def test_clip_geometry_is_pinned_to_the_c_frontend(tmp_path: Path) -> None:
    """Anything but a 1.000 s window would read past the front-end's fixed PCM buffer.

    `frontend/numl_yamnet_frontend.h` hard-codes `NUML_YAMNET_CLIP_SAMPLES 16000` and sizes
    its buffer from it, while `main.cpp` fills that buffer with `LIVE_WINDOW_SAMPLES`. The
    two must agree, and only `tokens_for()` can notice before the board does.
    """
    header = paths.APPS_ROOT / "known_sound" / "frontend" / "numl_yamnet_frontend.h"
    if header.is_file():
        assert (
            f"#define NUML_YAMNET_CLIP_SAMPLES     {ks_app.FRONTEND_CLIP_SAMPLES}"
            in header.read_text(encoding="utf-8")
        )
    c = _contract("NuMaker-M55M1", tmp_path)
    c.clip_seconds = 0.96  # YAMNet's own patch length -- still not this firmware's window
    with pytest.raises(DeployError, match="1 秒"):
        ks_app.tokens_for(c, 4096)


def test_a_hop_longer_than_the_window_is_refused(tmp_path: Path) -> None:
    c = _contract("NuMaker-M55M1", tmp_path)
    c.hop_seconds = 1.5
    with pytest.raises(DeployError, match="跳距"):
        ks_app.tokens_for(c, 4096)
    c.hop_seconds = 1.0  # exactly the window is fine: back-to-back, non-overlapping clips
    assert ks_app.tokens_for(c, 4096)["HOP_SAMPLES"] == "16000"


# --------------------------------------------------------------------------- overlay


def test_board_define_map_matches_boards_json() -> None:
    assert ks_app.BOARD_DEFINE == {
        "NuMaker-M55M1": "BOARD_NUMAKER_X_M55M1D",
        "NuGestureAI-M55M1": "BOARD_NUGESTUREAI_M55M1",
        "NuMaker-VoiceAI-M55M1": "BOARD_NUMAKER_VOICEAI_M55M1",
    }
    assert set(ks_app.BOARD_DEFINE) == set(boards.load_boards())


@pytest.mark.parametrize("board", BOARD_NAMES)
def test_overlay_keys_stay_inside_the_project_builder_contract(board: str, tmp_path: Path) -> None:
    overlay = ks_app.overlay(_contract(board, tmp_path))
    assert set(overlay) <= {
        "add_sources",
        "remove_source_basenames",
        "add_includes",
        "add_defines",
        "remove_defines",
    }
    assert overlay["add_defines"] == [ks_app.BOARD_DEFINE[board]]
    names = [Path(p).name for p in overlay["add_sources"]]
    assert {"numl_yamnet_frontend.c", "numl_known_sound.c", "numl_dmic.c"} <= set(names)
    assert {"dmic.c", "lppdma.c", "pmc.c"} <= set(names)
    # The app-relative entries stay relative here; project_builder joins them to app_dir.
    assert not Path(overlay["add_sources"][0]).is_absolute()
    assert Path("Frontend") in [Path(p) for p in overlay["add_includes"]]
    if board == "NuGestureAI-M55M1":
        assert overlay["remove_source_basenames"] == ["retarget.c"]
        assert {"numl_cdc.c", "numl_usbd.c", "numl_cdc_retarget.c", "hsusbd.c"} <= set(names)
    else:
        assert "remove_source_basenames" not in overlay
        assert not {"numl_cdc.c", "numl_usbd.c", "hsusbd.c"} & set(names)


def test_overlay_rejects_an_unknown_board(tmp_path: Path) -> None:
    c = _contract("NuMaker-M55M1", tmp_path)
    c.board = boards.Board(
        name="NuMaker-Imaginary",
        label="x",
        numl_template_board="NuMaker-M55M1",
        has_lcd=True,
        has_camera=True,
        display_output="lcd",
        audio_dmic=boards.AudioDmic("SET_DMIC0_CLK_PB4", "SET_DMIC0_DAT_PB5", 0),
        known_sound_max_depth=14,
        internal_flash_bytes=1,
        sram01_bytes=1,
        flash_methods=("nulink",),
    )
    with pytest.raises(DeployError, match="NuMaker-Imaginary"):
        ks_app.overlay(c)


# ------------------------------------------------------- contract.collect() plumbing


def _seed_project(
    store: ProjectStore,
    *,
    class_thresholds: list[float] | None = None,
    detection_threshold: float = 0.5,
    encoder_depth: int = 11,
) -> str:
    """A trained + Strict-INT8-exported known_sound project whose artifact is the fixture."""
    project = store.create_project("known_sound", "KS")
    pid = project["id"]
    raw = store.get_raw_project(pid)
    class_ids = [item["id"] for item in raw["classes"]]
    for class_id, name in zip(class_ids, LABELS, strict=True):
        store.update_class(pid, class_id, {"name": name})
    models = store.project_dir(pid) / "models"
    models.mkdir(parents=True, exist_ok=True)
    int8_name = "known_sound_yamnet_classifier_d11_int8.tflite"
    shutil.copy2(FIXTURES / "known_sound3_int8.tflite", models / int8_name)
    shutil.copy2(FIXTURES / "yamnet_frontend.json", models / "yamnet_frontend.json")
    entry = _entry()
    conversion = {"state": "generated", "models": {"int8": entry}}
    (models / "conversion_report.json").write_text(
        json.dumps({"models": {"int8": entry}}), encoding="utf-8"
    )
    settings: dict[str, Any] = {
        "detection_threshold": detection_threshold,
        "hop_seconds": 0.5,
        "clip_seconds": 1.0,
        "preview_peak_hold_seconds": 1.5,
    }
    if class_thresholds is not None:
        settings["class_thresholds"] = dict(zip(class_ids, class_thresholds, strict=True))
    store.set_training_result(
        pid,
        report={
            "project_kind": "known_sound",
            "settings": settings,
            "encoder": {"depth": encoder_depth},
            "conversion": conversion,
        },
        artifacts={"keras": "known_sound.keras", "int8": int8_name},
    )
    return pid


def test_collect_resolves_per_class_thresholds_into_the_tokens(store: ProjectStore) -> None:
    pid = _seed_project(store, class_thresholds=[0.6, 0.5, 0.9])
    c = contract.collect(store, pid, boards.board_for("NuGestureAI-M55M1"))
    contract.validate(c)
    assert c.labels == LABELS
    assert c.class_thresholds == [0.6, 0.5, 0.9]
    assert ks_app.tokens_for(c, 4096)["THRESHOLDS"] == (
        "    0.600000f,\n    0.500000f,\n    0.900000f,"
    )


def test_collect_falls_back_to_detection_threshold_for_every_class(store: ProjectStore) -> None:
    pid = _seed_project(store, detection_threshold=0.42)
    c = contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))
    contract.validate(c)
    assert c.class_thresholds == [0.42, 0.42, 0.42]
    assert ks_app.tokens_for(c, 4096)["THRESHOLDS"] == (
        "    0.420000f,\n    0.420000f,\n    0.420000f,"
    )


def test_encoder_depth_over_the_board_cap_is_refused(store: ProjectStore) -> None:
    pid = _seed_project(store, encoder_depth=12)
    c = contract.collect(store, pid, boards.board_for("NuGestureAI-M55M1"))
    with pytest.raises(DeployError, match="encoder_depth"):
        contract.validate(c)


# ----------------------------------------------------------------- real board builds


def _skip_unless_ready() -> None:
    if not paths.toolkit_available() or not paths.VELA_EXE.is_file():
        pytest.skip("mcu_toolkit or vela not vendored")
    # A Studio installed too deep for Windows MAX_PATH makes arm-none-eabi-gcc fail to
    # resolve the deepest BSP headers, with an English "No such file or directory" that has
    # nothing to do with what these cases assert. Skip with the reason a student would get.
    reason = long_path_reason(paths.MCU_TOOLKIT_ROOT)
    if reason:
        pytest.skip(reason)
    if not (paths.APPS_ROOT / "known_sound" / "main.cpp.in").is_file():
        pytest.skip("known_sound pack not vendored")
    if toolchain.find_toolchain() is None:
        pytest.skip("arm-none-eabi-gcc missing")


def _compile_commands(assembled: project_builder.AssembledProject) -> list[list[str]]:
    tc = gcc_build.Toolchain(Path("C:/tc/bin"), "probe")
    manifest = assembled.manifest
    return [
        gcc_build.compile_command(tc, manifest, src, Path("build") / manifest.object_name(src))
        for src in manifest.sources
    ]


@pytest.mark.parametrize("board", BOARD_NAMES)
def test_assemble_and_build_known_sound(board: str, tmp_path: Path) -> None:
    _skip_unless_ready()
    c = _contract(board, tmp_path)
    contract.validate(c)
    v = vela.compile_model(c.int8_path, tmp_path / "vela")
    assembled = project_builder.assemble(c, v, tmp_path / "work")
    assert assembled.project_name == "NN_ModelInference"
    assert assembled.app_dir == tmp_path / "work" / "NN_ModelInference"

    main = (assembled.app_dir / "main.cpp").read_text(encoding="utf-8")
    assert "s_afThreshold[KNOWN_SOUND_CLASSES]" in main and "0.900000f" in main
    assert "@@" not in main
    assert f"#define ACTIVATION_BUF_SZ ({v.arena_bytes})" in main
    assert main.count("#define LIVE_HOLD_HOPS           3") == 1
    # The GenericCodegen path (untested before this file) has to produce both model sources.
    assert (assembled.app_dir / "Model" / "include" / "NNModel.hpp").is_file()
    assert (assembled.app_dir / "Model" / "NNModel.cpp").is_file()
    assert (assembled.app_dir / "Model" / "NN_Model_INT8.tflite.cpp").is_file()
    for name in (
        "numl_yamnet_frontend.c",
        "numl_yamnet_frontend.h",
        "numl_yamnet_tables.h",
        "numl_known_sound.c",
        "numl_known_sound.h",
        "numl_dmic.c",
        "numl_dmic.h",
        "BoardConfig.h",
    ):
        assert (assembled.app_dir / "Frontend" / name).is_file(), name

    names = {s.path.name for s in assembled.manifest.sources}
    assert {
        "dmic.c",
        "lppdma.c",
        "pmc.c",
        "numl_yamnet_frontend.c",
        "numl_known_sound.c",
        "numl_dmic.c",
    } <= names
    defines = assembled.manifest.defines
    board_defines = [d for d in defines if d.startswith("BOARD_")]
    commands = _compile_commands(assembled)
    # The staged Frontend/ dir must be on every translation unit's include path (BoardConfig.h
    # is included by numl_dmic.c as well as main.cpp), and exactly once.
    frontend_inc = f"-I{assembled.app_dir / 'Frontend'}"
    for cmd in commands:
        assert cmd.count(frontend_inc) == 1, cmd
        # The vendored template hard-codes a 1 MiB arena; it must be REPLACED by the real
        # Vela arena exactly once, so every TU agrees with main.cpp's own #undef/#define.
        arena_flags = [f for f in cmd if f.startswith(f"-D{ARENA_DEFINE}")]
        assert arena_flags == [f"-D{ARENA_DEFINE}={v.arena_bytes}"], arena_flags
        assert len([f for f in cmd if f.startswith("-DBOARD_")]) == 1, cmd
    assert not any("0x100000" in flag for cmd in commands for flag in cmd)

    if board == "NuGestureAI-M55M1":
        assert board_defines == ["BOARD_NUGESTUREAI_M55M1"]
        assert "BOARD_NUMAKER_X_M55M1D" not in defines
        assert {"numl_cdc.c", "numl_usbd.c", "numl_cdc_retarget.c", "hsusbd.c"} <= names
        # numl_cdc_retarget.c textually #includes the BSP retarget.c, so the BSP copy must
        # not be compiled separately or _write()/stdout_putchar() are defined twice at link.
        assert "retarget.c" not in names
        assert not any(cmd[-3].replace("\\", "/").endswith("/retarget.c") for cmd in commands)
        for name in ("numl_cdc.h", "numl_usbd.h"):
            assert (assembled.app_dir / "Device" / "include" / name).is_file(), name
    else:
        assert board_defines == ["BOARD_NUMAKER_X_M55M1D"]
        assert "retarget.c" in names and "numl_usbd.c" not in names
        assert not (assembled.app_dir / "Device" / "CDC").exists()

    tc = toolchain.find_toolchain()
    result = gcc_build.build(assembled.manifest, tc, timeout_seconds=900, jobs=8)
    assert result.ok, result.message + "\n" + result.log_path.read_text(
        encoding="utf-8", errors="replace"
    )[-4000:]
    assert 100_000 < result.bin.stat().st_size < 2_097_152
    assert 0 < result.sram01_used <= 1_048_576
    # numl_dmic.c's ping-pong DMA blocks and descriptors live in the non-cacheable region.
    assert result.noncacheable_used > 0
    if KEEP_BIN:
        out = PROJECT_ROOT / "workspace" / "deliverables"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.bin, out / f"known_sound3_{board}.bin")

    # Firmware bytes must not depend on where the project was assembled: records.py's default
    # -ffile-prefix-map rewrites the BSP/app prefixes, so a rebuild from a different work dir
    # has to produce a byte-identical .bin.
    again = project_builder.assemble(c, v, tmp_path / "work-elsewhere-with-a-longer-name")
    second = gcc_build.build(again.manifest, tc, timeout_seconds=900, jobs=8)
    assert second.ok, second.message
    assert hashlib.sha256(second.bin.read_bytes()).hexdigest() == hashlib.sha256(
        result.bin.read_bytes()
    ).hexdigest()


def test_run_deploy_end_to_end(store: ProjectStore, tmp_path: Path, monkeypatch: Any) -> None:
    """Full `run_deploy()` on a seeded known_sound project; export is stubbed to skip TensorFlow.

    The build cases above drive `assemble()` + `gcc_build.build()` directly. This one goes in
    through the same entry point the HTTP job uses, so it also covers `contract.collect()`,
    the packaging and the report the UI reads -- in particular that `kind`/`application` say
    `known_sound`, not the `audio` the generic template is built from.
    """
    _skip_unless_ready()
    from tm_local.mcu import deploy_service as svc

    pid = _seed_project(store, class_thresholds=[0.6, 0.5, 0.9])
    models = store.project_dir(pid) / "models"
    export_calls: list[tuple] = []
    monkeypatch.setattr(
        svc,
        "ensure_export_artifacts",
        lambda *a, **k: export_calls.append((a, k)) or {"generated": {}},
    )
    monkeypatch.setattr(paths, "MCU_TEMP_ROOT", tmp_path / "tmp")

    seen: list[tuple[float, str]] = []
    outcome = svc.run_deploy(store, pid, "NuGestureAI-M55M1", lambda v, m: seen.append((v, m)))

    assert export_calls and export_calls[0][0][1:3] == (pid, ["int8"])
    assert outcome.report["kind"] == "known_sound"
    assert outcome.report["application"] == "known_sound"
    assert outcome.report["labels"] == LABELS
    assert outcome.report["flash_instructions"] == svc.flash_instructions(
        boards.board_for("NuGestureAI-M55M1"), "known_sound"
    )
    # No camera window, no LCD: the audio firmware only ever talks over the COM port. (The
    # board LABEL names both, so this is about the "what you will see" paragraph alone.)
    result_text = outcome.report["flash_instructions"].split("結果觀看")[1]
    assert "COM port" in result_text and "INFO threshold[" in result_text
    assert "相機 app" not in result_text and "LCD" not in result_text
    assert outcome.report["model"]["file"] == "known_sound_yamnet_classifier_d11_int8.tflite"
    assert outcome.report["bin"]["sha256"] == hashlib.sha256(
        outcome.bin_path.read_bytes()
    ).hexdigest()
    assert outcome.report["vela"]["arena_bytes"] > 0

    target = svc.deploy_dir(models, "NuGestureAI-M55M1")
    assert outcome.zip_path == target / "KS_NuGestureAI-M55M1_firmware.zip"
    with zipfile.ZipFile(outcome.zip_path) as archive:
        assert sorted(archive.namelist()) == sorted([
            "firmware.bin", "firmware.elf", "firmware.map", "build.log", "deploy_report.json",
            "labels.txt", "README_FLASH_zh-TW.txt", "source.zip",
        ])
    assert (target / "labels.txt").read_text(encoding="utf-8") == "0 gun\n1 dog\n2 Background\n"
    with zipfile.ZipFile(target / "source.zip") as archive:
        members = set(archive.namelist())
    # The staged frontend and the CDC transport have to travel with the source, or the student
    # cannot rebuild what was flashed.
    assert {"main.cpp", "Frontend/numl_yamnet_frontend.c", "Frontend/BoardConfig.h"} <= members
    assert "Device/CDC/numl_usbd.c" in members
    assert not any(name.startswith("GCC/build/") for name in members)
    # The per-class thresholds really reached the firmware source, not just the token table.
    main_cpp = zipfile.ZipFile(target / "source.zip").read("main.cpp").decode("utf-8")
    assert "0.600000f" in main_cpp and "0.900000f" in main_cpp

    assert [v for v, _m in seen] == sorted(v for v, _m in seen)
    assert seen[0][0] == 0.0 and seen[-1][0] == 1.0
    assert (tmp_path / "tmp").is_dir() and not any((tmp_path / "tmp").iterdir())
