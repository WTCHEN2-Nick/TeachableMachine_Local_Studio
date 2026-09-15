from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from .config import PROJECT_ROOT
from .utils import atomic_write_json, utc_now_iso

RUNTIME_ROOT = PROJECT_ROOT / "runtime"
RUNTIME_CONFIG_PATH = RUNTIME_ROOT / "runtime_config.json"
ENVIRONMENT_REPORT_PATH = RUNTIME_ROOT / "environment_report.json"


def _default_config() -> dict[str, Any]:
    logical = max(1, int(os.cpu_count() or 1))
    threads = max(1, min(8, logical - 1 if logical > 2 else logical))
    return {
        "schema_version": 2,
        "selected_backend": "windows_tensorflow_cpu",
        "backend_label": "Windows CPU",
        "reason": (
            "Windows-only mode is enabled. TensorFlow runs on the native Windows CPU; "
            "no Linux environment is used."
        ),
        "windows_only": True,
        "cpu": {
            "logical_cores": logical,
            "thread_limit": threads,
        },
        "video_controllers": [],
        "updated_at": utc_now_iso(),
        "mcu": {"available": False},
    }


def normalize_runtime_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    result = _default_config()
    if isinstance(payload, dict):
        if isinstance(payload.get("cpu"), dict):
            result["cpu"].update(payload["cpu"])
        if isinstance(payload.get("video_controllers"), list):
            result["video_controllers"] = payload["video_controllers"]
        if payload.get("updated_at"):
            result["updated_at"] = payload["updated_at"]
        mcu = payload.get("mcu")
        result["mcu"] = deepcopy(mcu) if isinstance(mcu, dict) else {"available": False}
    # Old v2.0.5/v2.0.6 configs may still say WSL2 GPU. This Windows-only
    # release intentionally overrides them instead of attempting Linux setup.
    result["schema_version"] = 2
    result["selected_backend"] = "windows_tensorflow_cpu"
    result["backend_label"] = "Windows CPU"
    result["windows_only"] = True
    result["reason"] = (
        "Windows-only mode is enabled. TensorFlow runs on the native Windows CPU; "
        "no Linux environment is used."
    )
    return result


def load_runtime_config() -> dict[str, Any]:
    try:
        payload = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    return normalize_runtime_config(payload)


def save_runtime_config(payload: dict[str, Any]) -> dict[str, Any]:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    normalized = normalize_runtime_config(payload)
    normalized["updated_at"] = utc_now_iso()
    atomic_write_json(RUNTIME_CONFIG_PATH, normalized)
    return normalized


def public_runtime_info(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    config = normalize_runtime_config(payload) if payload is not None else load_runtime_config()
    return {
        "selected_backend": "windows_tensorflow_cpu",
        "backend_label": "Windows CPU",
        "reason": config.get("reason"),
        "windows_only": True,
        # Kept for UI/API backward compatibility with v2.0.5/v2.0.6.
        "gpu_names": [],
        "nvidia_detected": False,
        "gpu_verified": False,
        "mcu": deepcopy(config.get("mcu") or {"available": False}),
    }


def apply_runtime_environment(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    config = normalize_runtime_config(payload) if payload is not None else load_runtime_config()
    cpu = config.get("cpu", {}) if isinstance(config.get("cpu"), dict) else {}
    threads = max(1, min(32, int(cpu.get("thread_limit") or (os.cpu_count() or 1))))
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(max(1, min(4, threads // 2 or 1))))
    return config


def runtime_config_for_report() -> dict[str, Any]:
    return deepcopy(load_runtime_config())
