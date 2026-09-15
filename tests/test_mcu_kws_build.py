"""Real two-board firmware build for an `audio` (KWS) project.

The template, token and overlay checks are plain unit tests; the two build cases are
genuine `arm-none-eabi-gcc` builds of the vendored generic template (~60-120 s per board,
twice per board for the determinism check) and are the acceptance evidence for the KWS
firmware app. Every case skips -- never fails -- when `mcu_toolkit/`, the vendored KWS
template, Vela or the Arm GNU Toolchain is missing.

The template assertions are the only place that can prove `main.cpp.in` actually honours
the `HAS_BACKGROUND` token: `kws_codegen` emits the flag, but only the template decides
whether a `["yes", "no"]` project silently makes "yes" untriggerable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from scripts import vendor_kws_template
from tm_local.mcu import (
    boards,
    contract,
    gcc_build,
    kws_codegen,
    paths,
    project_builder,
    toolchain,
    vela,
)
from tm_local.mcu.apps import kws as kws_app
from tm_local.mcu.errors import DeployError, long_path_reason
from tm_local.mcu.project_builder import ARENA_DEFINE
from tm_local.project_store import ProjectStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "mcu"
TEMPLATE = paths.APPS_ROOT / "audio_kws" / "main.cpp.in"
BOARD_NAMES = ["NuMaker-M55M1", "NuGestureAI-M55M1"]
LABELS = ["Background Noise", "yes", "no"]
INT8_NAME = "audio_classifier_spectrogram_int8.tflite"

# Set TM_MCU_KEEP_BIN=1 to drop the built firmware into the git-ignored
# workspace/deliverables/ for manual flashing; the default run leaves the repo untouched.
KEEP_BIN = os.environ.get("TM_MCU_KEEP_BIN") == "1"

pytestmark = pytest.mark.skipif(
    not paths.BOARDS_JSON.is_file(), reason="mcu_toolkit not vendored"
)


def _entry() -> dict[str, Any]:
    return json.loads((FIXTURES / "kws3_int8.report.json").read_text(encoding="utf-8"))


def _frontend() -> dict[str, Any]:
    return json.loads((FIXTURES / "audio_frontend_default.json").read_text(encoding="utf-8"))


def _tensor(spec: dict[str, Any]) -> contract.TensorInfo:
    return contract.TensorInfo(
        spec["name"],
        tuple(spec["shape"]),
        spec["dtype"],
        float(spec["scale"]),
        int(spec["zero_point"]),
    )


def _contract(
    board_name: str, tmp_path: Path, labels: list[str] | None = None
) -> contract.DeployContract:
    entry = _entry()
    int8 = tmp_path / "models" / INT8_NAME
    int8.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURES / "kws3_int8.tflite", int8)
    names = list(LABELS if labels is None else labels)
    return contract.DeployContract(
        kind="audio",
        application="kws",
        board=boards.board_for(board_name),
        project_id="p",
        project_name="KWS Test",
        labels=names,
        models_dir=int8.parent,
        int8_path=int8,
        model_sha256=hashlib.sha256(int8.read_bytes()).hexdigest(),
        input=_tensor(entry["inputs"][0]),
        output=_tensor(entry["outputs"][0]),
        image_size=None,
        audio_frontend=_frontend(),
        yamnet_frontend=None,
        encoder_depth=None,
        detection_threshold=0.6,
        class_thresholds=[0.6] * len(names),
        hop_seconds=0.5,
        clip_seconds=1.0,
        peak_hold_seconds=0.0,
    )


def _skip_without_template() -> None:
    if not TEMPLATE.is_file():
        pytest.skip("kws template not vendored")


# --------------------------------------------------------------------------- template


def test_template_markers() -> None:
    _skip_without_template()
    text = TEMPLATE.read_text(encoding="utf-8")
    for token in (
        "@@ARENA@@",
        "@@HANN@@",
        "@@MEL_WEIGHTS@@",
        "@@MEL_NNZ@@",
        "@@DMIC_CLK_MACRO@@",
        "@@DMIC_DAT_MACRO@@",
        "@@DMIC_CHANNEL_MASK@@",
        "@@LABELS@@",
        "@@TRIGGER_THRESHOLD@@",
        "@@TRIGGER_MARGIN@@",
        "@@REARM_SCORE@@",
        "@@RMS_GATE_DBFS@@",
        "@@SMOOTH_WINDOWS@@",
        "@@HAS_BACKGROUND@@",
        "@@INPUT_SCALE@@",
    ):
        assert token in text, token
    for forbidden in (
        "999999",
        "__L0__",
        "s_melLeft",
        "s_melCenter",
        "s_melRight",
        "s_melInverseSum",
        "arm_cos_f32",
        "SET_DMIC0_CLK_PB4",
        "DMIC_CTL_CHEN0_Msk",
    ):
        assert forbidden not in text, forbidden
    assert "NUML_USE_USB_CDC" in text
    assert "s_hannTable" in text
    assert "s_melWeights[NUML_MEL_NNZ]" in text
    assert "BOARD_LOG_PUMP();" in text
    # Bounded wait for a terminal to open the CDC port before the banner prints.
    assert "BOARD_LOG_WAIT();" in text and "NUML_BOARD_LOG_WAIT_MS" in text
    assert "NuML_CDC_IsOpen()" in text
    assert "NVT_NONCACHEABLE" in text
    assert "TEACHABLE_MACHINE_LOCAL_STUDIO_AUDIO_KWS_RUNTIME_V1" in text
    # The sparse tables carry row-normalised weights already, so multiplying by an inverse
    # sum a second time would scale every mel bin twice.
    assert "s_melStart" in text and "s_melOffset" in text


def test_template_tokens_are_exactly_the_kws_codegen_table(tmp_path: Path) -> None:
    """Neither more nor fewer placeholders than `kws_codegen.kws_tokens()` supplies.

    `render_tokens()` refuses a leftover `@@TOKEN@@`, so a missing key is a deploy-time
    failure; a surplus key is silently ignored, which is how a renamed placeholder would
    otherwise keep an obsolete value alive.
    """
    _skip_without_template()
    placeholders = set(re.findall(r"@@([A-Z0-9_]+)@@", TEMPLATE.read_text(encoding="utf-8")))
    tokens = kws_codegen.kws_tokens(_contract("NuMaker-M55M1", tmp_path), 102400)
    assert placeholders == set(tokens)
    assert len(placeholders) == 35


def test_template_gates_every_background_index_use() -> None:
    """`NUML_BACKGROUND_INDEX` may only be read inside `#if NUML_HAS_BACKGROUND`.

    A `["yes", "no"]` project has no background class, so the index falls back to 0. Read
    ungated, that would make "yes" permanently untriggerable and turn it into its own
    trigger-margin baseline -- firmware that looks fine and never fires.
    """
    _skip_without_template()
    text = TEMPLATE.read_text(encoding="utf-8")
    # The comment stripper is shared with the vendoring script (the template's own comments
    # have to name the macro to explain the gate); the span scan below is this file's own, so
    # a bug in the script's gate check cannot hide an ungated read from both sides at once.
    lines = vendor_kws_template.code_lines(text)
    gated = _has_background_spans(lines)
    for number, line in enumerate(lines):
        if "NUML_BACKGROUND_INDEX" not in line:
            continue
        if line.startswith("#define NUML_BACKGROUND_INDEX"):
            continue
        assert number in gated, f"ungated NUML_BACKGROUND_INDEX at line {number + 1}: {line}"
    assert gated, "template has no #if NUML_HAS_BACKGROUND block at all"
    # ...and the shipped file still satisfies the vendoring-time rule, so a hand edit is caught.
    vendor_kws_template.assert_background_gated(text)


def test_no_background_arm_rearms_without_the_score_gate() -> None:
    """The `#else` arm must release the trigger on the trigger conditions, not on a score.

    `s_triggerArmed` is cleared on every detection and can only come back through this test.
    With no background class the scores are one softmax, so the winner is always >= 0.5 --
    above any usable `NUML_REARM_SCORE` (0.40 by default) -- and the only remaining path back
    would be the RMS gate: the firmware would fire once per quiet period rather than once per
    spoken word. The background arm keeps upstream's score rule, which works there because
    the background class itself takes the lead when nothing is being said.
    """
    _skip_without_template()
    text = TEMPLATE.read_text(encoding="utf-8")
    lines = vendor_kws_template.code_lines(text)
    gated = _has_background_spans(lines)
    released = [n for n, line in enumerate(lines) if "const bool released" in line]
    assert len(released) == 2, "each arm must define its own `released`"
    with_background, without_background = released
    assert with_background in gated and without_background not in gated
    assert "average[best] < NUML_REARM_SCORE" in lines[with_background]
    assert "!scorePass || !marginPass" in lines[without_background]
    # ...and the re-arm test must actually ask it. `background` is a compile-time false in the
    # `#else` arm, so the condition there collapses to `quiet || released`.
    rearm = [n for n, line in enumerate(lines) if "{ s_triggerArmed = true; }" in line]
    assert len(rearm) == 1
    guard = lines[rearm[0] - 2]  # `if (...)`, then its `{`, then the counter
    assert guard.strip() == "if (quiet || background || released)", guard
    # The score constant survives as the background arm's rule, not as dead config: exactly
    # its #define plus the one use, with the comments stripped.
    assert sum(line.count("NUML_REARM_SCORE") for line in lines) == 2


def _has_background_spans(lines: list[str]) -> set[int]:
    """Line numbers inside a `#if NUML_HAS_BACKGROUND` ... `#else` block."""
    inside: set[int] = set()
    depth = 0
    opened_at = -1
    for number, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#if"):
            depth += 1
            if opened_at < 0 and stripped.startswith("#if NUML_HAS_BACKGROUND"):
                opened_at = depth
            continue
        if stripped.startswith("#endif"):
            if opened_at == depth:
                opened_at = -1
            depth -= 1
            continue
        if stripped.startswith("#else") and opened_at == depth:
            opened_at = -1
            continue
        if opened_at > 0:
            inside.add(number)
    return inside


# ----------------------------------------------------------------------------- render


def test_render_has_no_tokens_left(tmp_path: Path) -> None:
    _skip_without_template()
    c = _contract("NuMaker-M55M1", tmp_path)
    text = kws_codegen.render_kws_main(c, 102400, TEMPLATE)
    assert "@@" not in text
    assert "SET_DMIC0_CLK_PB4();" in text and "SET_DMIC0_DAT_PB5();" in text
    assert text.count("DMIC_CTL_CHEN0_Msk") == 2
    assert "#define NUML_INFERENCE_HOP     (4000U)" in text
    assert "#define ACTIVATION_BUF_SZ (102400)" in text
    assert "#define NUML_HAS_BACKGROUND    (1)" in text
    assert '"Background Noise", "yes", "no"' in text
    assert "#define NUML_MEL_NNZ           (454U)" in text


def test_render_for_the_gesture_board_uses_its_own_dmic_wiring(tmp_path: Path) -> None:
    _skip_without_template()
    text = kws_codegen.render_kws_main(_contract("NuGestureAI-M55M1", tmp_path), 102400, TEMPLATE)
    assert "SET_DMIC1_CLK_PB2();" in text and "SET_DMIC1_DAT_PB3();" in text
    assert text.count("DMIC_CTL_CHEN2_Msk") == 2


def test_render_without_a_background_class_disables_the_gate(tmp_path: Path) -> None:
    _skip_without_template()
    c = _contract("NuMaker-M55M1", tmp_path, labels=["yes", "no"])
    text = kws_codegen.render_kws_main(c, 102400, TEMPLATE)
    assert "#define NUML_HAS_BACKGROUND    (0)" in text
    assert "#define NUML_LABEL_COUNT       (2U)" in text


def test_missing_template_raises_a_student_facing_error(monkeypatch: Any) -> None:
    monkeypatch.setattr(kws_app, "APPS_ROOT", Path("C:/nonexistent-apps-root"))
    with pytest.raises(DeployError, match="韌體樣板"):
        kws_app.template_path()


def test_importing_the_app_module_does_not_pull_in_tensorflow() -> None:
    """`deploy_service` imports this on every deploy; TensorFlow costs ~10 s and ~1 GB."""
    code = "import tm_local.mcu.apps.kws, sys; print('tensorflow' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout


# ---------------------------------------------------------------------------- overlay


@pytest.mark.parametrize("board", BOARD_NAMES)
def test_overlay_keys_stay_inside_the_project_builder_contract(board: str, tmp_path: Path) -> None:
    overlay = kws_app.overlay(_contract(board, tmp_path))
    assert set(overlay) <= {
        "add_sources",
        "remove_source_basenames",
        "add_includes",
        "add_defines",
        "remove_defines",
    }
    names = {Path(p).name for p in overlay["add_sources"]}
    assert {"dmic.c", "lppdma.c", "pmc.c"} <= names
    for name in ("dmic.c", "lppdma.c", "pmc.c"):
        assert (paths.BSP_ROOT / "Library" / "StdDriver" / "src" / name).is_file(), name
    if board == "NuGestureAI-M55M1":
        assert overlay["add_defines"] == ["NUML_USE_USB_CDC"]
        assert overlay["remove_source_basenames"] == ["retarget.c"]
        assert {"numl_cdc.c", "numl_usbd.c", "numl_cdc_retarget.c", "hsusbd.c"} <= names
        assert paths.BSP_ROOT / "Library" / "StdDriver" / "src" in [
            Path(p) for p in overlay["add_includes"]
        ]
    else:
        assert overlay["add_defines"] == []
        assert "remove_source_basenames" not in overlay
        assert not {"numl_cdc.c", "numl_usbd.c", "hsusbd.c"} & names


@pytest.mark.parametrize("board", BOARD_NAMES)
def test_prepare_writes_main_and_stages_cdc_only_when_lcd_less(
    board: str, tmp_path: Path
) -> None:
    _skip_without_template()
    c = _contract(board, tmp_path)
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    kws_app.prepare(c, _FakeVela(102400), app_dir)
    main = (app_dir / "main.cpp").read_text(encoding="utf-8")
    assert "@@" not in main
    staged = app_dir / "Device"
    if board == "NuGestureAI-M55M1":
        for name in kws_app.CDC_HEADERS:
            assert (staged / "include" / name).is_file(), name
        for name in kws_app.CDC_SOURCES:
            assert (staged / "CDC" / name).is_file(), name
    else:
        assert not staged.exists()


class _FakeVela:
    def __init__(self, arena_bytes: int) -> None:
        self.arena_bytes = arena_bytes


# ----------------------------------------------------- contract.collect() plumbing


def _seed_project(store: ProjectStore, labels: list[str] | None = None) -> str:
    """A trained + Strict-INT8-exported `audio` project whose artifact is the fixture."""
    names = list(LABELS if labels is None else labels)
    project = store.create_project("audio", "KWS")
    pid = project["id"]
    class_ids = [item["id"] for item in store.get_raw_project(pid)["classes"]]
    while len(class_ids) < len(names):
        class_ids.append(store.add_class(pid, f"Class {len(class_ids) + 1}")["classes"][-1]["id"])
    for class_id, name in zip(class_ids, names, strict=True):
        store.update_class(pid, class_id, {"name": name})
    models = store.project_dir(pid) / "models"
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURES / "kws3_int8.tflite", models / INT8_NAME)
    shutil.copy2(FIXTURES / "audio_frontend_default.json", models / "audio_frontend.json")
    entry = _entry()
    (models / "conversion_report.json").write_text(
        json.dumps({"models": {"int8": entry}}), encoding="utf-8"
    )
    store.set_training_result(
        pid,
        report={
            "project_kind": "audio",
            "settings": {"detection_threshold": 0.6, "clip_seconds": 1.0, "hop_seconds": 0.5},
            "conversion": {"state": "generated", "models": {"int8": entry}},
        },
        artifacts={"keras": "audio.keras", "int8": INT8_NAME},
    )
    return pid


def test_collect_produces_a_renderable_kws_contract(store: ProjectStore) -> None:
    _skip_without_template()
    pid = _seed_project(store)
    c = contract.collect(store, pid, boards.board_for("NuGestureAI-M55M1"))
    contract.validate(c)
    assert c.application == "kws" and c.labels == LABELS
    text = kws_codegen.render_kws_main(c, 102400, TEMPLATE)
    assert "@@" not in text and "#define NUML_HAS_BACKGROUND    (1)" in text


def test_collect_honours_a_retuned_kws_runtime(store: ProjectStore) -> None:
    _skip_without_template()
    pid = _seed_project(store)
    raw = store.get_raw_project(pid)
    report = dict(raw["training"]["report"])
    report["settings"] = {
        **report["settings"],
        "kws_runtime": {
            "smooth_windows": 6,
            "required_hits": 3,
            "trigger_margin": 0.35,
            "rearm_score": 0.25,
            "rms_gate_dbfs": -48.0,
        },
    }
    store.set_training_result(pid, report=report, artifacts=raw["training"]["artifacts"])
    c = contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))
    text = kws_codegen.render_kws_main(c, 102400, TEMPLATE)
    # All five hard-coded upstream constants really became tokens; an inert one would leave
    # the App Builder's own value in the firmware and the student's setting would do nothing.
    assert "#define NUML_SMOOTH_WINDOWS    (6U)" in text
    assert "#define NUML_REQUIRED_HITS     (3U)" in text
    assert "#define NUML_TRIGGER_MARGIN    (0.35f)" in text
    assert "#define NUML_REARM_SCORE       (0.25f)" in text
    assert "#define NUML_RMS_GATE_DBFS     (-48.0f)" in text


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
    _skip_without_template()
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
def test_assemble_and_build_kws(board: str, tmp_path: Path) -> None:
    _skip_unless_ready()
    c = _contract(board, tmp_path)
    contract.validate(c)
    v = vela.compile_model(c.int8_path, tmp_path / "vela")
    assembled = project_builder.assemble(c, v, tmp_path / "work")
    assert assembled.project_name == "NN_ModelInference"

    main = (assembled.app_dir / "main.cpp").read_text(encoding="utf-8")
    assert "@@" not in main
    assert "s_melWeights[NUML_MEL_NNZ]" in main
    assert "s_hannTable[NUML_FRAME_LENGTH]" in main
    assert "NUML_HAS_BACKGROUND" in main
    assert "s_melInverseSum" not in main and "arm_cos_f32" not in main
    assert f"#define ACTIVATION_BUF_SZ ({v.arena_bytes})" in main

    names = {s.path.name for s in assembled.manifest.sources}
    assert {"dmic.c", "lppdma.c", "pmc.c"} <= names
    commands = _compile_commands(assembled)
    for cmd in commands:
        # The vendored template hard-codes a 1 MiB arena; it must be REPLACED by the real
        # Vela arena exactly once, so every TU agrees with main.cpp's own #undef/#define.
        arena_flags = [f for f in cmd if f.startswith(f"-D{ARENA_DEFINE}")]
        assert arena_flags == [f"-D{ARENA_DEFINE}={v.arena_bytes}"], arena_flags
    assert not any("0x100000" in flag for cmd in commands for flag in cmd)

    if board == "NuGestureAI-M55M1":
        assert "NUML_USE_USB_CDC" in assembled.manifest.defines
        assert {"numl_cdc.c", "numl_usbd.c", "numl_cdc_retarget.c", "hsusbd.c"} <= names
        # numl_cdc_retarget.c textually #includes the BSP retarget.c, so the BSP copy must
        # not be compiled separately or _write()/stdout_putchar() are defined twice at link.
        assert "retarget.c" not in names
        assert not any(cmd[-3].replace("\\", "/").endswith("/retarget.c") for cmd in commands)
        assert "SET_DMIC1_CLK_PB2();" in main and "DMIC_CTL_CHEN2_Msk" in main
    else:
        assert "NUML_USE_USB_CDC" not in assembled.manifest.defines
        assert "retarget.c" in names
        assert "SET_DMIC0_CLK_PB4();" in main and "DMIC_CTL_CHEN0_Msk" in main

    tc = toolchain.find_toolchain()
    result = gcc_build.build(assembled.manifest, tc, timeout_seconds=900, jobs=8)
    assert result.ok, result.message + "\n" + result.log_path.read_text(
        encoding="utf-8", errors="replace"
    )[-4000:]
    assert 100_000 < result.bin.stat().st_size < 2_097_152
    assert 0 < result.sram01_used <= 1_048_576
    # The LPPDMA ping-pong blocks and descriptors live in the non-cacheable region.
    assert result.noncacheable_used > 0
    if KEEP_BIN:
        out = PROJECT_ROOT / "workspace" / "deliverables"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.bin, out / f"kws3_{board}.bin")

    # Firmware bytes must not depend on where the project was assembled: records.py's
    # -ffile-prefix-map rewrites the BSP/app prefixes, so a rebuild from a different work
    # dir has to produce a byte-identical .bin.
    again = project_builder.assemble(c, v, tmp_path / "work-elsewhere-with-a-longer-name")
    second = gcc_build.build(again.manifest, tc, timeout_seconds=900, jobs=8)
    assert second.ok, second.message
    assert hashlib.sha256(second.bin.read_bytes()).hexdigest() == hashlib.sha256(
        result.bin.read_bytes()
    ).hexdigest()


def test_no_background_render_compiles(tmp_path: Path) -> None:
    """The `NUML_HAS_BACKGROUND (0)` branch has to compile, and no real build ever renders it.

    Both firmware builds above use the three-label fixture, whose "Background Noise" class sets
    the flag to 1, so the `#else` arm -- the one a `["yes", "no"]` project gets -- is never fed
    to a compiler by the rest of this file. A syntax-only pass is enough to catch the thing that
    actually goes wrong there: an identifier that only exists inside the `#if` arm.

    The include list and defines are lifted from the *real* compile command for `main.cpp`, so
    this cannot drift from what the build does; only the tail changes (`-fsyntax-only` instead
    of `-c -o`, plus -Wall -Wextra).
    """
    _skip_unless_ready()
    tc = toolchain.find_toolchain()
    c = _contract("NuMaker-M55M1", tmp_path)
    contract.validate(c)
    v = vela.compile_model(c.int8_path, tmp_path / "vela")
    assembled = project_builder.assemble(c, v, tmp_path / "work")

    main_src = next(s for s in assembled.manifest.sources if s.path.name == "main.cpp")
    real = gcc_build.compile_command(
        tc, assembled.manifest, main_src, Path("build") / "main.o"
    )
    # Everything except the compile-and-emit tail: the compiler, the CPU/ABI flags, the C++
    # flags, the -ffile-prefix-maps, `-I.`, every -I and every -D.
    flags = [f for f in real[1:-6] if f not in ("-c", "-MMD", "-MP")]

    no_background = _contract("NuMaker-M55M1", tmp_path, labels=["yes", "no"])
    rendered = kws_codegen.render_kws_main(no_background, v.arena_bytes, TEMPLATE)
    assert "#define NUML_HAS_BACKGROUND    (0)" in rendered
    target = assembled.app_dir / "main_no_background.cpp"
    target.write_text(rendered, encoding="utf-8", newline="\n")

    operand = f"{target.parent}/{target.name}"
    cmd = [real[0], *flags, "-std=c++17", "-fsyntax-only", "-Wall", "-Wextra", operand]
    gcc_cwd = assembled.app_dir / "GCC"
    out = subprocess.run(
        cmd, cwd=str(gcc_cwd), capture_output=True, text=True, timeout=600, check=False
    )
    assert out.returncode == 0, (out.stdout or "") + (out.stderr or "")

    # Negative control: the compiler really did see the `#else` arm, not just skip it. Renaming
    # a local that only exists there must break the very same command -- once for the margin
    # baseline, once for the re-arm release, the two things that arm alone defines.
    for original, typo in (
        ("float runnerUp = 0.0f;", "float runnerUpTypo = 0.0f;"),
        (
            "const bool released = !scorePass || !marginPass;",
            "const bool releasedTypo = !scorePass || !marginPass;",
        ),
    ):
        broken = rendered.replace(original, typo)
        assert broken != rendered, original
        target.write_text(broken, encoding="utf-8", newline="\n")
        again = subprocess.run(
            cmd, cwd=str(gcc_cwd), capture_output=True, text=True, timeout=600, check=False
        )
        assert again.returncode != 0, original


def test_run_deploy_end_to_end(store: ProjectStore, tmp_path: Path, monkeypatch: Any) -> None:
    """Full `run_deploy()` on a seeded `audio` project; export is stubbed to skip TensorFlow.

    The build cases above drive `assemble()` + `gcc_build.build()` directly. This one goes in
    through the same entry point the HTTP job uses, so it also covers `contract.collect()`,
    the front-end sidecar the KWS tokens are computed from, the packaging, and the report the
    UI reads -- `kind` stays `audio` while `application` is `kws`.
    """
    _skip_unless_ready()
    from tm_local.mcu import deploy_service as svc

    pid = _seed_project(store)
    models = store.project_dir(pid) / "models"
    export_calls: list[tuple] = []
    monkeypatch.setattr(
        svc,
        "ensure_export_artifacts",
        lambda *a, **k: export_calls.append((a, k)) or {"generated": {}},
    )
    monkeypatch.setattr(paths, "MCU_TEMP_ROOT", tmp_path / "tmp")

    seen: list[tuple[float, str]] = []
    outcome = svc.run_deploy(store, pid, "NuMaker-M55M1", lambda v, m: seen.append((v, m)))

    assert export_calls and export_calls[0][0][1:3] == (pid, ["int8"])
    assert outcome.report["kind"] == "audio" and outcome.report["application"] == "kws"
    assert outcome.report["labels"] == LABELS
    assert outcome.report["flash_instructions"] == svc.flash_instructions(
        boards.board_for("NuMaker-M55M1"), "audio"
    )
    # The X board has an LCD, but the KWS firmware never draws on it. (The board LABEL names
    # the LCD, so the claim is about the "what you will see" paragraph, not the whole text.)
    flash_text = outcome.report["flash_instructions"]
    assert "KWS DETECTED" in flash_text
    assert "LCD" not in flash_text.split("結果觀看")[1]
    assert outcome.report["model"]["file"] == INT8_NAME
    assert outcome.report["bin"]["sha256"] == hashlib.sha256(
        outcome.bin_path.read_bytes()
    ).hexdigest()
    assert outcome.report["vela"]["arena_bytes"] > 0

    target = svc.deploy_dir(models, "NuMaker-M55M1")
    assert outcome.zip_path == target / "KWS_NuMaker-M55M1_firmware.zip"
    with zipfile.ZipFile(outcome.zip_path) as archive:
        assert sorted(archive.namelist()) == sorted([
            "firmware.bin", "firmware.elf", "firmware.map", "build.log", "deploy_report.json",
            "labels.txt", "README_FLASH_zh-TW.txt", "source.zip",
        ])
    assert (target / "labels.txt").read_text(encoding="utf-8") == (
        "0 Background Noise\n1 yes\n2 no\n"
    )
    with zipfile.ZipFile(target / "source.zip") as archive:
        members = set(archive.namelist())
        main_cpp = archive.read("main.cpp").decode("utf-8")
    assert "main.cpp" in members and not any(n.startswith("GCC/build/") for n in members)
    # This board has an LCD/UART, so the CDC transport must NOT have been staged.
    assert not any(n.startswith("Device/CDC/") for n in members)
    # The front-end tables really came from audio_frontend.json, not from the App Builder.
    assert "s_hannTable" in main_cpp and "#define NUML_HAS_BACKGROUND    (1)" in main_cpp
    assert "SET_DMIC0_CLK_PB4();" in main_cpp

    assert [v for v, _m in seen] == sorted(v for v, _m in seen)
    assert seen[0][0] == 0.0 and seen[-1][0] == 1.0
    assert (tmp_path / "tmp").is_dir() and not any((tmp_path / "tmp").iterdir())
