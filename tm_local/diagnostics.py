from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LOG_ROOT, PROJECT_ROOT
from .runtime_config import ENVIRONMENT_REPORT_PATH, RUNTIME_CONFIG_PATH, load_runtime_config


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "not installed"


def _tail_text(path: Path, maximum_bytes: int = 512_000) -> str:
    if not path.is_file():
        return "(log file does not exist)"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum_bytes), os.SEEK_SET)
            payload = handle.read()
        text = payload.decode("utf-8", errors="replace")
        if size > maximum_bytes:
            text = "[older log content omitted]\n" + text
        return text
    except Exception as exc:
        return f"(cannot read log: {type(exc).__name__}: {exc})"


def _json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_diagnostic_text(store, jobs=None) -> str:  # noqa: ANN001
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    disk = shutil.disk_usage(PROJECT_ROOT)
    runtime = load_runtime_config()
    environment_report = _json_file(ENVIRONMENT_REPORT_PATH)
    projects = []
    try:
        for item in store.list_projects():
            projects.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "kind": item.get("kind"),
                    "sample_count": item.get("sample_count"),
                    "training_state": (item.get("training") or {}).get("state"),
                    "updated_at": item.get("updated_at"),
                }
            )
    except Exception as exc:
        projects = [{"error": f"{type(exc).__name__}: {exc}"}]

    recent_jobs = []
    if jobs is not None and hasattr(jobs, "recent"):
        try:
            recent_jobs = jobs.recent(limit=20)
        except Exception as exc:
            recent_jobs = [{"error": f"{type(exc).__name__}: {exc}"}]

    sections = [
        "=" * 78,
        "Teachable Machine Local Studio diagnostic log",
        "=" * 78,
        f"Generated           : {now}",
        f"Project root        : {PROJECT_ROOT}",
        f"Python executable   : {sys.executable}",
        f"Python version      : {platform.python_version()}",
        f"Operating system    : {platform.platform()}",
        f"Logical CPU cores   : {os.cpu_count()}",
        f"Disk free           : {disk.free / (1024 ** 3):.2f} GiB",
        f"TensorFlow          : {_package_version('tensorflow')}",
        f"TF-Keras            : {_package_version('tf-keras')}",
        f"NumPy               : {_package_version('numpy')}",
        "",
        "[Runtime selection]",
        json.dumps(runtime, ensure_ascii=False, indent=2),
        "",
        "[Install environment report]",
        json.dumps(environment_report, ensure_ascii=False, indent=2)
        if environment_report is not None
        else "(not available)",
        "",
        "[Projects — metadata only, no images/audio included]",
        json.dumps(projects, ensure_ascii=False, indent=2),
        "",
        "[Recent jobs]",
        json.dumps(recent_jobs, ensure_ascii=False, indent=2),
        "",
        "[LATEST.log — current/most recent Local Studio session]",
        _tail_text(LOG_ROOT / "LATEST.log"),
        "",
        "[LATEST_INSTALL.log — most recent 01_INSTALL.bat]",
        _tail_text(LOG_ROOT / "LATEST_INSTALL.log"),
        "",
    ]
    return "\n".join(sections)
