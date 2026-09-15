"""Known Sound firmware app: render `main.cpp` and stage the YAMNet frontend + DMIC sources.

The runtime itself is vendored, not generated: `mcu_toolkit/apps/known_sound/` holds the
`main.cpp.in` template (produced once by `scripts/vendor_known_sound.py`), the five frontend
files whose C the host tests compile, and the three device files. This module only fills the
template's seven `@@TOKEN@@`s, copies those eight files into `<app_dir>/Frontend/`, stages the
USB-CDC log transport on an LCD-less board, and reports the project-record overlay the build
needs -- so it stays cheap to import (stdlib plus `..codegen` / `..errors` / `..paths`; no
jinja2, no yaml, no TensorFlow).

Every score this firmware prints is an independent sigmoid, one per class, judged against its
own threshold: no argmax, no softmax, and they do not sum to one. That is why the thresholds
travel as a per-class array rather than a single number.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..codegen import cpp_escape, render_tokens
from ..errors import DeployError
from ..paths import APPS_ROOT, BSP_ROOT
from ._cdc import CDC_VARIANTS, stage_cdc

# BoardConfig.h selects the DMIC pin pair, the DMIC channel and the printf transport from
# exactly one of these. `boards.json` owns the pins themselves; this map only says which
# compile-time branch of BoardConfig.h a board name selects.
BOARD_DEFINE = {
    "NuMaker-M55M1": "BOARD_NUMAKER_X_M55M1D",
    "NuGestureAI-M55M1": "BOARD_NUGESTUREAI_M55M1",
    "NuMaker-VoiceAI-M55M1": "BOARD_NUMAKER_VOICEAI_M55M1",
}

FRONTEND_FILES = (
    "numl_yamnet_frontend.c",
    "numl_yamnet_frontend.h",
    "numl_yamnet_tables.h",
    "numl_known_sound.c",
    "numl_known_sound.h",
)
DEVICE_FILES = ("numl_dmic.c", "numl_dmic.h", "BoardConfig.h")
# The three of the eight staged files that are actually compiled; the rest are headers.
COMPILED = ("numl_yamnet_frontend.c", "numl_known_sound.c", "numl_dmic.c")
# BSP drivers the generic template does not already compile: DMIC itself, the low-power PDMA
# that feeds it, and the power-management controller its clock domain needs.
DRIVER_SOURCES = ("dmic.c", "lppdma.c", "pmc.c")
# A plain USB virtual COM port; `_cdc` owns the file list, and the source names are repeated
# in overlay() below because they also have to be compiled.
CDC_VARIANT = "cdc_only"
CDC_HEADERS, CDC_SOURCES = CDC_VARIANTS[CDC_VARIANT]
STAGE_DIR = "Frontend"

# The frontend contract is locked at 16 kHz (see `yamnet_model.frontend_contract()`), so the
# clip/hop seconds convert to samples with a fixed rate rather than one read from the model.
SAMPLE_RATE = 16000
# ...and the vendored C front-end fixes the analysis window too: `NUML_YAMNET_CLIP_SAMPLES`
# in `frontend/numl_yamnet_frontend.h` is a literal 16000, and its PCM buffer is sized from
# it. `tokens_for()` refuses any other geometry rather than emitting a template that would
# read past that buffer. Pinned against the header by tests/test_mcu_known_sound_build.py.
FRONTEND_CLIP_SAMPLES = 16000


def _pack_file(*parts: str) -> Path:
    path = APPS_ROOT.joinpath("known_sound", *parts)
    if not path.is_file():
        raise DeployError(
            f"找不到 known_sound 韌體檔 {path}；mcu_toolkit 不完整，請重新執行 01_INSTALL.bat"
        )
    return path


def tokens_for(contract: Any, arena_bytes: int) -> dict[str, str]:
    """The seven `@@TOKEN@@` values `main.cpp.in` needs, as a pure function of the contract.

    `HOLD_HOPS` is the peak-hold window expressed in inference hops and is floored at 1: a
    project whose `preview_peak_hold_seconds` is shorter than one hop still has to hold the
    peak for the hop it was measured in, not for zero hops (which would make the display
    flicker off between two halves of the same 200 ms event).
    """
    if len(contract.class_thresholds) != len(contract.labels):
        raise DeployError("每類門檻數量與類別數不符")
    if not contract.labels:
        raise DeployError("專案沒有任何類別，無法產生 known_sound 韌體")
    hop_seconds = float(contract.hop_seconds)
    if not hop_seconds > 0.0:
        raise DeployError(f"hop_seconds 必須大於 0，目前是 {contract.hop_seconds}")
    clip_samples = round(float(contract.clip_seconds) * SAMPLE_RATE)
    hop_samples = round(hop_seconds * SAMPLE_RATE)
    if clip_samples <= 0 or hop_samples <= 0:
        raise DeployError(
            f"clip_seconds／hop_seconds 換算成取樣數必須大於 0，目前是 "
            f"{contract.clip_seconds}／{contract.hop_seconds}"
        )
    # The vendored C front-end hard-codes `#define NUML_YAMNET_CLIP_SAMPLES 16000` and sizes
    # its PCM buffer from it, while `main.cpp` fills that buffer with LIVE_WINDOW_SAMPLES
    # (= CLIP_SAMPLES) samples. A project whose clip is anything but exactly 1.000 s would
    # therefore either read uninitialised tail samples or overrun the buffer -- silently, on
    # the board. Refuse it here instead, where the student can still be told why.
    if clip_samples != FRONTEND_CLIP_SAMPLES:
        raise DeployError(
            f"known_sound 韌體的分析窗固定是 1 秒（{FRONTEND_CLIP_SAMPLES} 取樣），"
            f"這個專案是 {contract.clip_seconds} 秒（{clip_samples} 取樣），無法部署到開發板"
        )
    if hop_samples > clip_samples:
        raise DeployError(
            f"hop_seconds（{contract.hop_seconds} 秒）不可以大於分析窗的 1 秒："
            "跳距比窗還長會讓韌體漏掉中間的聲音"
        )
    hold_hops = max(1, round(float(contract.peak_hold_seconds) / hop_seconds))
    return {
        "ARENA": str(int(arena_bytes)),
        "CLASS_COUNT": str(len(contract.labels)),
        "CLIP_SAMPLES": str(clip_samples),
        "HOP_SAMPLES": str(hop_samples),
        "HOLD_HOPS": str(hold_hops),
        "LABELS": "\n".join(f'    "{cpp_escape(name)}",' for name in contract.labels),
        "THRESHOLDS": "\n".join(f"    {float(t):.6f}f," for t in contract.class_thresholds),
    }


def prepare(contract: Any, vela: Any, app_dir: Path) -> None:
    """Stage `<app_dir>/Frontend/` (8 files), write `main.cpp`, and add USB CDC when LCD-less.

    Runs last in `project_builder.assemble()`, so the `main.cpp` written here deliberately
    replaces the numeric-placeholder one NuML's GenericCodegen just wrote.
    """
    app_dir = Path(app_dir)
    stage = app_dir / STAGE_DIR
    stage.mkdir(parents=True, exist_ok=True)
    for name in FRONTEND_FILES:
        shutil.copy2(_pack_file("frontend", name), stage / name)
    for name in DEVICE_FILES:
        shutil.copy2(_pack_file("device", name), stage / name)

    template = _pack_file("main.cpp.in")
    text = render_tokens(
        template.read_text(encoding="utf-8"), tokens_for(contract, vela.arena_bytes)
    )
    (app_dir / "main.cpp").write_text(text, encoding="utf-8", newline="\n")

    if contract.board.has_lcd:
        return
    # No LCD: BoardConfig.h routes printf through a USB CDC virtual COM port, so the CDC/USBD
    # runtime ships with the project. `Device/include` is already on the template's include
    # path (Device.yaml), which is what makes BoardConfig.h's `#include "numl_cdc.h"` resolve.
    stage_cdc(app_dir, CDC_VARIANT)


def overlay(contract: Any) -> dict:
    """Project-record changes this app needs; relative paths are joined to the app dir."""
    board = contract.board
    define = BOARD_DEFINE.get(board.name)
    if define is None:
        raise DeployError(
            f"known_sound 韌體不支援板子 {board.name}；可用：{', '.join(BOARD_DEFINE)}"
        )
    driver = BSP_ROOT / "Library" / "StdDriver" / "src"
    stage = Path(STAGE_DIR)
    result: dict = {
        "add_sources": [stage / name for name in COMPILED]
        + [driver / name for name in DRIVER_SOURCES],
        "add_includes": [stage],
        "add_defines": [define],
    }
    if not board.has_lcd:
        cdc_dir = Path("Device") / "CDC"
        result["add_sources"] += [cdc_dir / name for name in CDC_SOURCES]
        result["add_sources"].append(driver / "hsusbd.c")
        # numl_cdc_retarget.c #includes the BSP retarget.c itself, so the BSP copy must not
        # also be compiled -- two definitions of _write()/stdout_putchar() would not link.
        result["remove_source_basenames"] = ["retarget.c"]
        # ...and that textual #include has to be able to find it.
        result["add_includes"] = [*result["add_includes"], driver]
    return result
