r"""Startup sweep of the MCU deploy leftovers in `TEMP_ROOT`.

Two kinds of file outlive a deploy:

* `TEMP_ROOT\tm_local_firmware_*.zip` -- the throwaway copy `app._register_deploy_download()`
  hands `/api/exports/{token}`. That endpoint unlinks whatever it serves, so only the copies
  the student never downloaded survive (~4 MB each, one per completed build and per repeated
  download click).
* `TEMP_ROOT\mcu\<hex>\` -- a deploy work dir (~50 MB) that `run_deploy()`'s `finally` could
  not remove because the process was hard-killed mid-build.

Both are safe to delete *at startup* and only at startup: the download tokens live in
`create_app()`'s in-memory dict, so a fresh process can no longer serve any of these files, and
no build is running yet. Deliberately narrow -- `tm_local_model_*.zip` (model export) and
`tm_local_project_*.zip` (project archive) are left alone, and so is anything else under
`TEMP_ROOT`.

Import stays stdlib-only (no TensorFlow, no build stack) so `create_app()` can call it.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

FIRMWARE_TEMP_GLOB = "tm_local_firmware_*.zip"
WORK_DIR_NAME = "mcu"
LOGGER = logging.getLogger("tm_local.mcu.housekeeping")


def sweep_temp_root(temp_root: Path | None = None) -> list[Path]:
    """Delete orphaned firmware ZIP copies and deploy work dirs; return what went.

    Never raises: a missing `TEMP_ROOT`, a file another process still holds open, or a
    permission error must not stop the Studio from starting.
    """
    if temp_root is None:
        from ..config import TEMP_ROOT

        temp_root = TEMP_ROOT
    root = Path(temp_root)
    removed: list[Path] = []
    try:
        if not root.is_dir():
            return removed
        for leftover in sorted(root.glob(FIRMWARE_TEMP_GLOB)):
            try:
                leftover.unlink()
            except OSError as exc:
                LOGGER.warning("無法刪除暫存韌體檔 %s：%s", leftover, exc)
                continue
            removed.append(leftover)
        work_root = root / WORK_DIR_NAME
        if work_root.is_dir():
            for work in sorted(work_root.iterdir()):
                if not work.is_dir():
                    continue
                shutil.rmtree(work, ignore_errors=True)
                if work.exists():
                    LOGGER.warning("無法刪除 MCU 部署暫存資料夾：%s", work)
                    continue
                removed.append(work)
    except OSError as exc:  # pragma: no cover - defensive: listing TEMP_ROOT itself failed
        LOGGER.warning("清理暫存資料夾失敗：%s", exc)
        return removed
    if removed:
        LOGGER.info("已清理 %d 個 MCU 部署暫存檔／資料夾：%s", len(removed),
                    ", ".join(p.name for p in removed))
    return removed
