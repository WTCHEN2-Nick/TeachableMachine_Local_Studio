"""KWS (`audio` kind) firmware app: render `main.cpp` and stage its board-specific sources.

The runtime is vendored, not generated at deploy time: `mcu_toolkit/apps/audio_kws/main.cpp.in`
is produced once by `scripts/vendor_kws_template.py` from the App Builder's live Audio-KWS
runtime. This module only fills its 35 `@@TOKEN@@`s (via `kws_codegen.render_kws_main()`),
stages the USB-CDC log transport on an LCD-less board, and reports the project-record overlay
the build needs -- so it stays cheap to import (stdlib plus `..kws_codegen` / `..errors` /
`..paths`; no jinja2, no yaml, no TensorFlow).

The firmware's log-mel front-end is driven by tables exported from `tm_local/audio_frontend.py`,
so training, Preview, the INT8 calibration and the board all run one recipe; the DMIC pin pair
and channel mask come from `boards.json` through `boards.AudioDmic`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import DeployError
from ..kws_codegen import render_kws_main
from ..paths import APPS_ROOT, BSP_ROOT
from ._cdc import CDC_VARIANTS, stage_cdc

# A plain USB virtual COM port; `_cdc` owns the file list, and the source names are repeated
# in overlay() below because they also have to be compiled.
CDC_VARIANT = "cdc_only"
CDC_HEADERS, CDC_SOURCES = CDC_VARIANTS[CDC_VARIANT]
# The compile-time switch `main.cpp.in` tests to route printf() over the USB CDC port; on a
# board with an LCD/UART the same macros expand to no-ops.
CDC_DEFINE = "NUML_USE_USB_CDC"
# BSP drivers the generic template does not already compile: the digital microphone itself,
# the low-power PDMA that ping-pongs its FIFO into SRAM, and the power-management controller
# whose APLL1 clock domain the DMIC runs from.
DRIVER_SOURCES = ("dmic.c", "lppdma.c", "pmc.c")


def _driver_dir() -> Path:
    return BSP_ROOT / "Library" / "StdDriver" / "src"


def template_path() -> Path:
    path = APPS_ROOT / "audio_kws" / "main.cpp.in"
    if not path.is_file():
        raise DeployError(
            f"找不到 KWS 韌體樣板 {path}；mcu_toolkit 不完整，請重新執行 01_INSTALL.bat"
        )
    return path


def prepare(contract: Any, vela: Any, app_dir: Path) -> None:
    """Write `<app_dir>/main.cpp` and, on an LCD-less board, stage the USB CDC sources.

    Runs last in `project_builder.assemble()`, so the `main.cpp` written here deliberately
    replaces the synthetic-input one NuML's GenericCodegen just wrote.
    """
    app_dir = Path(app_dir)
    text = render_kws_main(contract, vela.arena_bytes, template_path())
    (app_dir / "main.cpp").write_text(text, encoding="utf-8", newline="\n")
    if contract.board.has_lcd:
        return
    # No LCD: printf() leaves over a USB CDC virtual COM port, so the CDC/USBD runtime ships
    # with the project. `Device/include` is already on the template's include path
    # (Device.yaml), which is what makes main.cpp's `#include "numl_cdc.h"` resolve.
    stage_cdc(app_dir, CDC_VARIANT)


def overlay(contract: Any) -> dict:
    """Project-record changes this app needs; relative paths are joined to the app dir."""
    driver = _driver_dir()
    result: dict = {
        "add_sources": [driver / name for name in DRIVER_SOURCES],
        "add_includes": [],
        "add_defines": [],
    }
    if contract.board.has_lcd:
        return result
    cdc_dir = Path("Device") / "CDC"
    result["add_sources"] += [cdc_dir / name for name in CDC_SOURCES]
    result["add_sources"].append(driver / "hsusbd.c")
    # numl_cdc_retarget.c #includes the BSP retarget.c itself, so the BSP copy must not also
    # be compiled -- two definitions of _write()/stdout_putchar() would not link.
    result["remove_source_basenames"] = ["retarget.c"]
    # ...and that textual #include has to be able to find it.
    result["add_includes"].append(driver)
    result["add_defines"].append(CDC_DEFINE)
    return result
