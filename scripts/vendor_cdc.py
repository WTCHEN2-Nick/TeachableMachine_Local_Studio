"""Render the App Builder's USB CDC / overlay C sources once into mcu_toolkit/apps/common/cdc/.

The App Builder (read-only source; ``tools/usb_cdc_runtime.py`` and
``tools/usb_cdc_device.py``) post-processes a *generated* Keil project by
writing C sources into ``<project>/Device/{include,CDC}``. This script drives
those two writers against a throwaway project directory and copies the result
into a static location the Studio ships, so the runtime never needs the App
Builder itself.

Both writers share one ``numl_cdc_retarget.c`` template that assumes a fixed
project nesting depth (``<project>/NN_.../Device/CDC/``) to reach the BSP with
a relative include. Vendored flat under ``apps/common/cdc/<variant>/`` that
depth is wrong, so ``_patch_retarget`` re-bases the BSP includes and adds the
GCC "tee" seam described in the Task 9 brief.
"""
from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

STUDIO_MARK = "/* Modified for Teachable Machine Local Studio: GCC printf tee seam. */\n"

# Anchor 1: the include of the real BSP retarget.c. The App Builder writes it
# relative to its own project nesting (5 levels for the uvc_composite runtime,
# possibly fewer for the cdc-only device writer); vendored flat it must become
# a bare quote-include so it resolves via -I<BSP>/Library/StdDriver/src.
_RETARGET_C_INCLUDE = re.compile(r'#include\s+"(?:\.\./)+Library/StdDriver/src/retarget\.c"')

# Anchor 2: the ARMCC/GCC "_sys_* layer" includes nested inside
# `#if defined (__ARMCC_VERSION)` / `#elif defined (__GNUC__)`. Same problem as
# anchor 1. GCC's quote-include search falls back through the -I list in
# order when the current-file-relative lookup misses (see gcc docs on
# "Quote includes"), and this build always passes
# -I<BSP>/Library/StdDriver/src (needed for anchor 1), so re-basing both to 2
# levels up from *that* directory reaches the BSP unambiguously no matter
# where this file is vendored -- it mirrors how the BSP's own retarget.c
# includes retarget_GCC.c (also 2 levels up from its own directory).
_SYS_LAYER_INCLUDE = re.compile(
    r'#include\s+"(?:\.\./)+Library/(Device/Nuvoton/M55M1/Source/(?:ARM|GCC)/retarget_(?:ARMCC|GCC)\.c)"'
)

# Anchor 3: the __GNUC__ branch guard. The App Builder writes
# "defined (__GNUC__)" (with a space); normalise it so later code (and tests)
# can anchor on the literal "defined(__GNUC__)".
_GNUC_GUARD = re.compile(r"#elif\s+defined\s*\(__GNUC__\)")

_RETARGET_GCC_INCLUDE = re.compile(
    r'([ \t]*)#include\s+"\.\./\.\./Device/Nuvoton/M55M1/Source/GCC/retarget_GCC\.c"'
)


def _patch_retarget(path: Path) -> None:
    """Make one App-Builder-rendered numl_cdc_retarget.c buildable with GCC.

    Any anchor found zero or more-than-expected times raises SystemExit: a
    silent partial patch would be worse than a hard failure at vendor time.
    """
    text = path.read_text(encoding="utf-8")

    text, n = _RETARGET_C_INCLUDE.subn('#include "retarget.c"', text)
    if n != 1:
        raise SystemExit(f"{path}: retarget.c include anchor found {n} times (expected 1)")

    text, n = _SYS_LAYER_INCLUDE.subn(r'#include "../../\1"', text)
    if n != 2:
        raise SystemExit(f"{path}: expected 2 ARMCC/GCC _sys_ layer includes, found {n}")

    text, n = _GNUC_GUARD.subn("#elif defined(__GNUC__)", text)
    if n != 1:
        raise SystemExit(f"{path}: __GNUC__ branch guard not found")

    gnu = text.index("defined(__GNUC__)")
    match = _RETARGET_GCC_INCLUDE.search(text, gnu)
    if not match:
        raise SystemExit(f"{path}: retarget_GCC.c include not found after __GNUC__ branch")
    indent = match.group(1)
    seam = (
        f"{indent}#undef stdout_putchar\n"
        f"{indent}#undef stdin_getchar\n"
        f"{indent}#undef stderr_putchar\n"
        f"{indent}int stdout_putchar(int ch);   "
        "/* tee defined below; retarget_GCC.c's _write() must bind to it */\n"
    )
    text = text[: match.start()] + seam + text[match.start() :]

    path.write_text(STUDIO_MARK + text, encoding="utf-8", newline="\n")


def _import_app_builder_tools(app_builder_root: Path):
    # usb_cdc_device.py does `if __package__: from .usb_cdc_runtime import ...`,
    # so it must be imported as part of the `tools` package (its
    # `tools/__init__.py` is an empty marker file) -- import it bare and the
    # relative import fails. Insert the App Builder root (not tools/ itself)
    # so `tools` resolves as a package.
    sys.path.insert(0, str(app_builder_root))
    from tools import usb_cdc_device, usb_cdc_runtime  # App Builder modules (read-only use)

    return usb_cdc_runtime, usb_cdc_device


def vendor_cdc(app_builder_root: Path, dest_root: Path) -> None:
    usb_cdc_runtime, usb_cdc_device = _import_app_builder_tools(app_builder_root)

    out = dest_root / "apps" / "common" / "cdc"
    if out.exists():
        shutil.rmtree(out)

    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "proj"
        usb_cdc_runtime._write_cdc_sources(project, True)
        composite = out / "uvc_composite"
        composite.mkdir(parents=True)
        for name in ("numl_cdc.h", "numl_overlay.h"):
            shutil.copy2(project / "Device" / "include" / name, composite / name)
        for name in ("numl_cdc.c", "numl_cdc_retarget.c", "numl_overlay.c"):
            shutil.copy2(project / "Device" / "CDC" / name, composite / name)
        _patch_retarget(composite / "numl_cdc_retarget.c")

    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "proj"
        usb_cdc_device._write_sources(project)
        only = out / "cdc_only"
        only.mkdir(parents=True)
        for name in ("numl_cdc.h", "numl_usbd.h"):
            shutil.copy2(project / "Device" / "include" / name, only / name)
        for name in ("numl_cdc.c", "numl_usbd.c", "numl_cdc_retarget.c"):
            shutil.copy2(project / "Device" / "CDC" / name, only / name)
        _patch_retarget(only / "numl_cdc_retarget.c")

    (out / "README.md").write_text(
        "USB CDC serial + UVC overlay sources rendered from NuML App Builder v0.1.8 (Apache-2.0).\n"
        "uvc_composite/: CDC on interfaces 2/3 beside the UVC camera (image kind on NuGestureAI).\n"
        "cdc_only/: CDC-only HSUSB device on interfaces 0/1 (audio kinds on NuGestureAI).\n"
        "numl_cdc_retarget.c is patched so GCC's _write() reaches the CDC tee (see the file header).\n",
        encoding="utf-8",
    )
