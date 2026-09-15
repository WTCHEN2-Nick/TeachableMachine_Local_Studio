from __future__ import annotations

import logging
from typing import Any

from .audio_pipeline import train_audio_project
from .config import normalize_kind
from .image_pipeline import train_image_project
from .project_store import ProjectStore
from .runtime_config import public_runtime_info
from .training_common import TrainingError
from .utils import atomic_write_json

LOGGER = logging.getLogger("tm_local.training")


def _persist_training_report(store: ProjectStore, project_id: str, result: dict[str, Any]) -> None:
    report = result.get("report")
    if not isinstance(report, dict):
        return
    models_dir = store.project_dir(project_id) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(models_dir / "training_report.json", report)


def train_project(
    store: ProjectStore,
    project_id: str,
    kind: str,
    options: dict[str, Any] | None,
    progress,
) -> dict[str, Any]:
    settings = dict(options or {})
    normalized_kind = normalize_kind(kind)
    LOGGER.info(
        "Starting native Windows CPU training: project=%s kind=%s",
        project_id,
        normalized_kind,
    )
    if normalized_kind == "image":
        result = train_image_project(store, project_id, settings, progress)
    elif normalized_kind == "audio":
        result = train_audio_project(store, project_id, settings, progress)
    elif normalized_kind == "abnormal_sound":
        # Keep the import lazy: the YAMNet pipeline imports TensorFlow only inside the
        # functions that need it, and ordinary image/audio startup must remain lightweight.
        from .yamnet_anomaly_pipeline import train_yamnet_abnormal_sound_project

        result = train_yamnet_abnormal_sound_project(
            store, project_id, settings, progress
        )
    elif normalized_kind == "known_sound":
        from .known_sound_pipeline import train_known_sound_project

        result = train_known_sound_project(store, project_id, settings, progress)
    else:
        raise TrainingError(f"Unsupported project kind: {normalized_kind!r}.")

    runtime = public_runtime_info()
    report = result.setdefault("report", {})
    report["training_runtime"] = {
        "backend": "windows_tensorflow_cpu",
        "label": "Windows CPU",
        "configured_backend": runtime.get("selected_backend"),
        "windows_only": True,
    }
    _persist_training_report(store, project_id, result)
    return result
