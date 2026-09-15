"""Stage one of the vendored USB-CDC runtimes into an assembled project.

All three firmware apps do the same thing on an LCD-less board: printf() has nowhere to go, so
the CDC runtime from `mcu_toolkit/apps/common/cdc/<variant>/` has to ship with the project --
headers into `Device/include` (already on the template's include path via `Device.yaml`, which
is what lets `#include "numl_cdc.h"` resolve) and sources into `Device/CDC`.

Two variants, and they are not interchangeable:

* `cdc_only` -- a plain virtual COM port (interfaces 0/1, PID 0x1105). Used by the audio apps
  on an LCD-less board: nothing about the board lacks a camera (the GestureAI has one), the
  audio firmware simply never opens it, so there is no UVC function to fold the CDC into. It
  brings its own USB device stack (`numl_usbd.c`).
* `uvc_composite` -- the CDC function folded into the image app's existing UVC device, plus the
  frame overlay that draws the predicted label. Its USB stack is the UVC one already in the
  template, so there is no `numl_usbd.c` here.

Each app keeps its own copy of the file names it stages (they are part of its record overlay,
which lists the sources to compile), so this module only performs the copy.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..errors import DeployError
from ..paths import APPS_ROOT

#: Variant name -> (headers into Device/include, sources into Device/CDC).
CDC_VARIANTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "cdc_only": (
        ("numl_cdc.h", "numl_usbd.h"),
        ("numl_cdc.c", "numl_usbd.c", "numl_cdc_retarget.c"),
    ),
    "uvc_composite": (
        ("numl_cdc.h", "numl_overlay.h"),
        ("numl_cdc.c", "numl_cdc_retarget.c", "numl_overlay.c"),
    ),
}


def stage_cdc(app_dir: Path, variant: str) -> None:
    """Copy `variant`'s five files into `<app_dir>/Device/{include,CDC}`.

    Raises `DeployError` (student-facing) when the vendored source file is missing, which is
    what an incomplete `mcu_toolkit/` looks like from here.
    """
    try:
        headers, sources = CDC_VARIANTS[variant]
    except KeyError:
        raise DeployError(f"未知的 USB CDC 版本 {variant}") from None
    app_dir = Path(app_dir)
    src = APPS_ROOT / "common" / "cdc" / variant
    include_dir = app_dir / "Device" / "include"
    cdc_dir = app_dir / "Device" / "CDC"
    include_dir.mkdir(parents=True, exist_ok=True)
    cdc_dir.mkdir(parents=True, exist_ok=True)
    for name, dest in [(n, include_dir) for n in headers] + [(n, cdc_dir) for n in sources]:
        source = src / name
        if not source.is_file():
            raise DeployError(f"找不到 USB CDC 來源檔 {source}")
        shutil.copy2(source, dest / name)
