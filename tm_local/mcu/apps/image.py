"""Image-classification firmware app: render `main.cpp` and stage its board-specific sources.

The heavy lifting (rendering NuML's jinja2 `main.cpp` and applying the App Builder's
fixed-memory / USB-CDC patches) happens once at vendoring time in
`scripts/vendor_image_templates.py`; this module only fills the `@@ARENA@@` token and copies
files, so it stays cheap to import (no jinja2, no yaml, no TensorFlow).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..codegen import render_tokens
from ..errors import DeployError
from ..paths import APPS_ROOT, BSP_ROOT
from ._cdc import CDC_VARIANTS, stage_cdc

# The CDC function folded into this app's existing UVC device, plus the frame overlay that
# draws the predicted label. `_cdc` owns the file list; the source names are repeated in
# overlay() below because they also have to be compiled.
_CDC_VARIANT = "uvc_composite"
_CDC_HEADERS, _CDC_SOURCES = CDC_VARIANTS[_CDC_VARIANT]


def template_path(board_name: str) -> Path:
    path = APPS_ROOT / "image" / board_name / "main.cpp.in"
    if not path.is_file():
        raise DeployError(f"找不到 image 韌體樣板 {path}")
    return path


def prepare(contract: Any, vela: Any, app_dir: Path) -> None:
    """Write `<app_dir>/main.cpp` and, on an LCD-less board, stage the USB CDC sources."""
    text = render_tokens(
        template_path(contract.board.name).read_text(encoding="utf-8"),
        {"ARENA": str(int(vela.arena_bytes))},
    )
    app_dir = Path(app_dir)
    (app_dir / "main.cpp").write_text(text, encoding="utf-8", newline="\n")
    if contract.board.has_lcd:
        return
    # No LCD: the label is drawn into the UVC frame and printf() goes out over the
    # composite CDC serial port, so the CDC/overlay runtime has to ship with the project.
    stage_cdc(app_dir, _CDC_VARIANT)


def overlay(contract: Any) -> dict:
    """Project-record changes this app needs; `add_sources` are relative to the app dir."""
    if contract.board.has_lcd:
        return {}
    cdc_dir = Path("Device") / "CDC"
    return {
        "add_sources": [cdc_dir / name for name in _CDC_SOURCES],
        # numl_cdc_retarget.c #includes the BSP retarget.c itself, so the BSP copy must
        # not also be compiled -- two definitions of _write()/fputc() would not link.
        "remove_source_basenames": ["retarget.c"],
        "add_includes": [BSP_ROOT / "Library" / "StdDriver" / "src"],
    }
