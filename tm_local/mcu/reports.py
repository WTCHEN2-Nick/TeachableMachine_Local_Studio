"""Where a board's deploy artifacts live, and how to read the reports back.

Deliberately its own module rather than part of `deploy_service`: `project_store._public()`
surfaces the per-board deploy reports in a project's public dict, and `project_store` must not
import `deploy_service` (which pulls in the whole build pipeline). This module imports nothing
but `utils`, so it stays free for anyone to import. `deploy_service` re-exports both functions
under its own name for callers that already hold it.
"""
from __future__ import annotations

from pathlib import Path

from ..utils import path_within, read_json
from .errors import DeployError

REPORT_NAME = "deploy_report.json"


def deploy_dir(models_dir: Path, board_name: str) -> Path:
    """`models/mcu/<board>/` -- the per-board folder holding the firmware and its report."""
    root = Path(models_dir) / "mcu"
    try:
        path_within(root, board_name)  # traversal guard only; keep the caller's path form
    except ValueError as exc:
        raise DeployError(f"板子名稱不合法：{board_name!r}") from exc
    return root / board_name


def read_deploy_reports(models_dir: Path) -> dict[str, dict]:
    """Every board's `deploy_report.json` under `models/mcu/`, keyed by board name.

    Never raises: a project whose `models/` was never deployed, or whose report file is
    truncated/hand-edited, simply contributes nothing -- this feeds a project's public dict and
    must not be able to break listing a project.

    Skips directory names ending in `.tmp`: a rebuild assembles into a sibling `<board>.tmp`
    before swapping it in over `<board>` (see `deploy_service.deploy_dir()`), and a hard kill
    during that swap window can leave the `.tmp` staging directory behind -- it is not a real
    board name and must not show up as one. Also skips anything containing `.old-`: the old
    `<board>` directory is renamed aside to `<board>.old-<pid>` for the microseconds it takes to
    swap the new build in (Windows cannot delete-then-rename a directory still held open by
    antivirus atomically), and a hard kill in that window can likewise leave it behind with its
    own (stale) deploy_report.json still readable.
    """
    root = Path(models_dir) / "mcu"
    out: dict[str, dict] = {}
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if child.name.endswith(".tmp") or ".old-" in child.name:
            continue
        report = child / REPORT_NAME
        if not child.is_dir() or not report.is_file():
            continue
        try:
            payload = read_json(report)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            out[child.name] = payload
    return out
