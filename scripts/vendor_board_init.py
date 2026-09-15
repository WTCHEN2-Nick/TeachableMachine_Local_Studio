"""Vendor NuMaker-VoiceAI-M55M1's BoardInit.cpp from the App Builder (GCC edition).

Run by the maintainer through ``vendor_mcu_toolkit.py board-init``; students never run it.

The NuML template set has no VoiceAI template of its own, so the board's project is generated
from the NuGestureAI-M55M1 template -- whose debug console lives on UART5. This board's console
is UART4 on the J3 header, and the BSP's ``SetDebugUartMFP()``
(``Library/Device/Nuvoton/M55M1/Source/system_M55M1.c``) only has branches for UART index 0, 5
and 6, so the borrowed template's ``SetDebugUartMFP()`` / ``SetDebugUartCLK()`` /
``InitDebugUart()`` do not fit this board at all. ``known_sound/device/BoardInit_VoiceAI.cpp`` in
the App Builder distribution is the file Nuvoton ships to cover that gap -- a whole translation
unit, not a compiler define -- and it already carries its own Apache-2.0 SPDX header and needs no
edits, so this script copies it byte for byte rather than patching it the way ``vendor_cdc.py``
and ``vendor_known_sound.py`` patch theirs.

Two sibling files live next to it upstream (``BoardInit_GestureAI.cpp``,
``BoardInit_X_M55M1D.cpp``); only the VoiceAI one is vendored here. The other two boards use
their own NuML template's ``BoardInit.cpp`` and must not be disturbed.

Task 3 registers ``mcu_toolkit/boards/NuMaker-VoiceAI-M55M1/BoardInit_VoiceAI.cpp`` as this
board's ``extra_sources`` entry in ``boards.json``; ``tm_local/mcu/boards.py``'s
``board_source_dir()`` resolves exactly that path, so the destination here is not negotiable.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

SOURCE_REL = Path("known_sound") / "device" / "BoardInit_VoiceAI.cpp"
DEST_REL = Path("boards") / "NuMaker-VoiceAI-M55M1" / "BoardInit_VoiceAI.cpp"


def _fail(message: str) -> None:
    raise SystemExit(f"vendor_board_init: {message}")


def vendor_board_init(app_builder_root: Path, dest_root: Path) -> None:
    src = app_builder_root / SOURCE_REL
    if not src.is_file():
        _fail(f"App Builder VoiceAI BoardInit not found at {src}")

    out = dest_root / DEST_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    # Byte for byte: the file needs no changes (see module docstring), so unlike
    # vendor_cdc.py / vendor_known_sound.py there is no Studio header to add here.
    shutil.copy2(src, out)

    data = out.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    print(f"[OK] board-init: wrote {out} ({len(data)} bytes, sha256 {digest})")
