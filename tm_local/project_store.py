from __future__ import annotations

import io
import json
import logging
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .archive import ArchiveError, safe_extract_zip, zip_directory
from .audio_frontend import (
    config_from_mapping,
    encode_wav_bytes,
    log_mel_spectrogram,
    read_wav_bytes,
    spectrogram_thumbnail,
    split_clips,
)
from .config import (
    ANOMALY_EVAL_ROLE,
    CLASS_COLORS,
    MAX_CLASS_NAME_LENGTH,
    MAX_CLASSES,
    MAX_PROJECT_NAME_LENGTH,
    NORMAL_TRAIN_ROLE,
    PROJECT_KIND_LABELS,
    PROJECTS_ROOT,
    TEMP_ROOT,
    allowed_setting_keys,
    defaults_for_kind,
    ensure_workspace,
    is_audio_like,
    is_known_kind,
    normalize_kind,
    validate_abnormal_sound_settings,
    validate_audio_settings,
    validate_image_settings,
    validate_known_sound_settings,
)
from .mcu.reports import read_deploy_reports
from .utils import atomic_write_json, clean_text, path_within, read_json, utc_now_iso


class ProjectError(ValueError):
    pass


class ProjectNotFound(ProjectError):
    pass


def _validate_kind_settings(
    kind: str, settings: dict[str, Any], class_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validate settings with the contract owned by the selected project kind.

    Returns the NORMALIZED settings dict callers must persist. ``validate_image_settings``
    and ``validate_audio_settings`` merge onto their kind's defaults and return a fresh
    dict; ``validate_abnormal_sound_settings`` and ``validate_known_sound_settings``
    mutate their argument in place and return that same object. Either way, the return
    value is authoritative -- it is where string booleans ("false"), numeric strings, and
    other request-boundary input get coerced into real JSON types, and a caller that kept
    using its original ``settings`` reference instead would persist the un-normalized
    input (a stored ``"false"`` string reads back truthy through ``bool(...)``).

    ``class_ids``, when given, cross-checks the class-scoped keys (``class_thresholds``
    and ``background_class_id``) against the project's actual classes -- omit it (the
    default) where no class list is available yet, such as ``create_project()``.
    """

    normalized = normalize_kind(kind)
    try:
        if normalized == "image":
            settings = validate_image_settings(settings)
        elif normalized == "audio":
            settings = validate_audio_settings(settings)
        elif normalized == "abnormal_sound":
            settings = validate_abnormal_sound_settings(settings)
        elif normalized == "known_sound":
            settings = validate_known_sound_settings(settings)
        else:
            raise ProjectError(f"Unknown project kind: {normalized!r}.")
    except ValueError as exc:
        if isinstance(exc, ProjectError):
            raise
        raise ProjectError(str(exc)) from exc
    if class_ids is not None:
        known = set(class_ids)
        for key in settings.get("class_thresholds") or {}:
            if key not in known:
                raise ProjectError(f"class_thresholds 含有不存在的類別 id {key!r}")
        background = settings.get("background_class_id") or ""
        if background and background not in known:
            raise ProjectError(f"background_class_id {background!r} 不是這個專案的類別")
    return settings


def _json_device_metadata(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a bounded JSON-safe copy of optional browser/audio-device metadata."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProjectError("device_metadata must be a JSON object.")
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ProjectError("device_metadata must contain JSON-compatible values.") from exc
    if len(encoded.encode("utf-8")) > 16 * 1024:
        raise ProjectError("device_metadata is too large (maximum 16 KiB).")
    if not isinstance(decoded, dict):  # pragma: no cover - json round-trip is defensive
        raise ProjectError("device_metadata must be a JSON object.")
    return decoded


LOGGER = logging.getLogger(__name__)

# A zero-byte file whose NAME is the project's, dropped beside project.json so File Explorer
# shows which folder is which. The folder itself stays the project id and always will:
# `_project_dir()` accepts only hex digits and hyphens -- that is the path-traversal guard --
# and forty call sites build paths from that id. Renaming folders to project names would mean
# deleting the guard and rewriting all forty, for readability. A marker file costs nothing.
_MARKER_PREFIX = "【"
_MARKER_SUFFIX = "】.txt"
# Windows rejects these outright, plus control characters; trailing dots and spaces are silently
# dropped by the filesystem, which would make the name we wrote differ from the name we look for.
_MARKER_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _marker_filename(name: str) -> str:
    """The marker file name for `name`, safe on Windows and still readable in Chinese.

    `safe_identifier()` is deliberately not used here: it maps every non-ASCII character to an
    underscore, which would turn a Chinese project name into a row of underscores and defeat the
    only purpose this file has.
    """
    cleaned = _MARKER_FORBIDDEN.sub("_", name).strip().rstrip(". ").strip()
    return f"{_MARKER_PREFIX}{cleaned[:60] or '未命名專案'}{_MARKER_SUFFIX}"


def _refresh_name_marker(project_dir: Path, name: str) -> None:
    """Leave exactly one marker, carrying the current name.

    Best effort by design: a marker that cannot be written -- antivirus holding the directory, a
    name this filesystem refuses -- must never fail the save that carries the student's actual
    work. The marker is a convenience; project.json is the data.
    """
    wanted = _marker_filename(name)
    try:
        for stale in project_dir.glob(f"{_MARKER_PREFIX}*{_MARKER_SUFFIX}"):
            if stale.name != wanted:
                stale.unlink()
        marker = project_dir / wanted
        if not marker.exists():
            marker.touch()
    except OSError:
        LOGGER.debug("無法更新專案名稱標記檔：%s", project_dir, exc_info=True)


class ProjectStore:
    def __init__(
        self,
        projects_root: Path = PROJECTS_ROOT,
        temp_root: Path = TEMP_ROOT,
        *,
        recover_interrupted: bool = True,
    ):
        self.projects_root = Path(projects_root).resolve()
        self.temp_root = Path(temp_root).resolve()
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.recovered_interrupted_jobs = (
            self.recover_interrupted_training() if recover_interrupted else 0
        )

    def recover_interrupted_training(self) -> int:
        """Recover a native Keras model saved before an interrupted conversion.

        v2.0.3 converted four TensorFlow Lite variants inside the Train Model
        job. Closing Local Studio while it was stuck at 71% left the project
        marked as ``training`` even though the complete ``.keras`` model had
        already been written. On the next start we preserve that model and
        defer quantization to Export Model.
        """

        recovered = 0
        with self._lock:
            for metadata_path in self.projects_root.glob("*/project.json"):
                try:
                    payload = read_json(metadata_path)
                except Exception:
                    continue
                training = payload.get("training", {})
                if not isinstance(training, dict) or training.get("state") != "training":
                    continue
                kind = normalize_kind(payload.get("kind"))
                keras_candidates = {
                    "image": ("image_classifier.keras",),
                    "audio": ("audio_classifier_spectrogram.keras",),
                    "abnormal_sound": (
                        "abnormal_sound_yamnet.keras",
                        "yamnet_embedding.keras",
                    ),
                }.get(kind, ())
                models_dir = metadata_path.parent / "models"
                keras_path = next(
                    (
                        models_dir / name
                        for name in keras_candidates
                        if (models_dir / name).is_file()
                    ),
                    None,
                )
                missing_abnormal_sidecars: list[str] = []
                if keras_path is not None and kind == "abnormal_sound":
                    required_sidecars = ("yamnet_scorer.json", "yamnet_frontend.json")
                    missing_abnormal_sidecars = [
                        name for name in required_sidecars if not (models_dir / name).is_file()
                    ]
                if keras_path is None or missing_abnormal_sidecars:
                    if missing_abnormal_sidecars:
                        detail = (
                            "The previous abnormal-sound training was interrupted before its "
                            "scorer/frontend metadata was saved (missing: "
                            + ", ".join(missing_abnormal_sidecars)
                            + ")."
                        )
                    else:
                        detail = (
                            "The previous training process was interrupted before the model "
                            "was saved."
                        )
                    payload["training"] = {
                        "state": "failed",
                        "trained_at": None,
                        "report": {
                            "error": (
                                detail + " All samples are safe; press Train Model again."
                            )
                        },
                        "artifacts": {},
                    }
                    payload["updated_at"] = utc_now_iso()
                    atomic_write_json(metadata_path, payload)
                    continue
                keras_name = keras_path.name

                models_dir.mkdir(parents=True, exist_ok=True)
                report_path = models_dir / "training_report.json"
                try:
                    report = read_json(report_path) if report_path.is_file() else {}
                except Exception:
                    report = {}
                class_names = [
                    str(item.get("name", "Class"))
                    for item in payload.get("classes", [])
                    if isinstance(item, dict)
                ]
                class_counts = {
                    str(item.get("name", "Class")): sum(
                        1
                        for sample in payload.get("samples", {}).values()
                        if sample.get("class_id") == item.get("id")
                    )
                    for item in payload.get("classes", [])
                    if isinstance(item, dict)
                }
                report = dict(report or {})
                report.setdefault("project_kind", kind)
                report.setdefault("trained_at", utc_now_iso())
                if kind == "abnormal_sound":
                    report.setdefault(
                        "roles",
                        {
                            "normal_train": sum(
                                count
                                for item_name, count in class_counts.items()
                                if item_name == class_names[0]
                            )
                            if class_names
                            else 0,
                            "anomaly_eval_groups": class_names[1:],
                        },
                    )
                else:
                    report.setdefault("class_names", class_names)
                    report.setdefault("class_counts", class_counts)
                report.setdefault("settings", deepcopy(payload.get("settings", {})))
                report.setdefault("dataset", {})
                report.setdefault("evaluation", {})
                report.setdefault("history", {})
                report["recovered_after_interrupted_conversion"] = True
                report["conversion"] = {
                    "state": "pending",
                    "models": {},
                    "note": (
                        "The Keras model was recovered. TensorFlow Lite conversion now runs only "
                        "after Export Model is requested."
                    ),
                }
                atomic_write_json(report_path, report)
                if kind != "abnormal_sound":
                    (models_dir / "labels.txt").write_text(
                        "".join(f"{index} {name}\n" for index, name in enumerate(class_names)),
                        encoding="utf-8",
                    )
                if kind == "audio":
                    frontend_path = models_dir / "audio_frontend.json"
                    if not frontend_path.is_file():
                        frontend = config_from_mapping(payload.get("settings", {}))
                        atomic_write_json(frontend_path, frontend.to_dict())
                artifacts = {
                    "keras": keras_name,
                    "training_report": report_path.name,
                }
                for key, filename in (
                    ("audio_frontend", "audio_frontend.json"),
                    ("audio_frontend_reference", "audio_frontend_reference.py"),
                    ("anomaly_config", "anomaly_config.json"),
                    ("yamnet_scorer", "yamnet_scorer.json"),
                    ("yamnet_frontend", "yamnet_frontend.json"),
                    ("run_model", "run_model.py"),
                ):
                    if (models_dir / filename).is_file():
                        artifacts[key] = filename
                payload["training"] = {
                    "state": "trained",
                    "trained_at": str(report.get("trained_at") or utc_now_iso()),
                    "report": report,
                    "artifacts": artifacts,
                }
                payload["updated_at"] = utc_now_iso()
                atomic_write_json(metadata_path, payload)
                recovered += 1
        return recovered

    def _project_dir(self, project_id: str) -> Path:
        if not project_id or any(ch not in "0123456789abcdef-" for ch in project_id.lower()):
            raise ProjectNotFound("Project does not exist.")
        path = (self.projects_root / project_id).resolve()
        try:
            path.relative_to(self.projects_root)
        except ValueError as exc:
            raise ProjectNotFound("Project does not exist.") from exc
        return path

    def project_dir(self, project_id: str) -> Path:
        path = self._project_dir(project_id)
        if not (path / "project.json").is_file():
            raise ProjectNotFound("Project does not exist.")
        return path

    def _load(self, project_id: str) -> dict[str, Any]:
        path = self.project_dir(project_id) / "project.json"
        try:
            payload = read_json(path)
        except Exception as exc:
            raise ProjectError(f"Cannot read project metadata: {project_id}") from exc
        if payload.get("id") != project_id:
            raise ProjectError("Project metadata ID does not match its directory.")
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = utc_now_iso()
        project_dir = self._project_dir(payload["id"])
        atomic_write_json(project_dir / "project.json", payload)
        # Every mutation funnels through here -- creation, rename, settings, samples -- so one
        # call keeps the marker in step with the name for the life of the project.
        _refresh_name_marker(project_dir, str(payload.get("name") or ""))

    @staticmethod
    def _new_class(
        name: str,
        index: int,
        *,
        role: str | None = None,
        locked: bool = False,
    ) -> dict[str, Any]:
        item = {
            "id": uuid.uuid4().hex,
            "name": clean_text(name, maximum=MAX_CLASS_NAME_LENGTH, fallback=f"Class {index + 1}"),
            "color": CLASS_COLORS[index % len(CLASS_COLORS)],
            "created_at": utc_now_iso(),
        }
        if role is not None:
            item["role"] = str(role)
            item["locked"] = bool(locked)
        return item

    @staticmethod
    def _ensure_editable(payload: dict[str, Any]) -> None:
        if payload.get("training", {}).get("state") == "training":
            raise ProjectError("The project is training. Wait for training to finish before editing samples or settings.")

    def _clear_models(self, payload: dict[str, Any]) -> None:
        models_dir = self._project_dir(payload["id"]) / "models"
        if models_dir.exists():
            shutil.rmtree(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

    def create_project(self, kind: str, name: str | None = None) -> dict[str, Any]:
        kind = normalize_kind(kind)
        if not is_known_kind(kind):
            known = ", ".join(PROJECT_KIND_LABELS)
            raise ProjectError(f"Project kind must be one of: {known}.")
        project_id = str(uuid.uuid4())
        now = utc_now_iso()
        default_name = PROJECT_KIND_LABELS[kind]
        if kind == "image":
            classes = [
                self._new_class(label, index)
                for index, label in enumerate(("Class 1", "Class 2"))
            ]
        elif kind == "audio":
            classes = [
                self._new_class(label, index)
                for index, label in enumerate(("Background Noise", "Class 2"))
            ]
        elif kind == "known_sound":
            # Every class is equal, renameable and deletable -- there is no locked
            # container. Names stay generic so the tool is not hard-wired to one use case;
            # "Background" ships pre-made because a multi-label model with no negative
            # examples fires on everything, and users reliably forget to create it.
            classes = [
                self._new_class(label, index)
                for index, label in enumerate(("Class 1", "Class 2", "Background"))
            ]
        elif kind == "abnormal_sound":
            classes = [
                self._new_class("Normal", 0, role=NORMAL_TRAIN_ROLE, locked=True),
                self._new_class("Anomaly Examples", 1, role=ANOMALY_EVAL_ROLE),
            ]
        else:  # pragma: no cover - is_known_kind() above already rejected this
            raise ProjectError(f"Unknown project kind: {kind!r}.")
        project_settings = _validate_kind_settings(kind, defaults_for_kind(kind))
        payload = {
            "schema_version": 2,
            "id": project_id,
            "name": clean_text(
                name or default_name,
                maximum=MAX_PROJECT_NAME_LENGTH,
                fallback=default_name,
            ),
            "kind": kind,
            "created_at": now,
            "updated_at": now,
            "classes": classes,
            "samples": {},
            "settings": project_settings,
            "training": {
                "state": "untrained",
                "trained_at": None,
                "report": None,
                "artifacts": {},
            },
        }
        with self._lock:
            project_dir = self._project_dir(project_id)
            project_dir.mkdir(parents=True, exist_ok=False)
            (project_dir / "samples").mkdir()
            (project_dir / "thumbs").mkdir()
            (project_dir / "models").mkdir()
            self._save(payload)
        return self.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        for metadata in self.projects_root.glob("*/project.json"):
            try:
                payload = read_json(metadata)
                public = self._public(payload, include_samples=False)
                projects.append(public)
            except Exception:
                continue
        projects.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return projects

    def _public(self, payload: dict[str, Any], *, include_samples: bool = True) -> dict[str, Any]:
        result = deepcopy(payload)
        samples = result.pop("samples", {})
        by_class: dict[str, list[dict[str, Any]]] = {item["id"]: [] for item in result["classes"]}
        for sample in samples.values():
            class_id = sample.get("class_id")
            if class_id not in by_class:
                continue
            public_sample = deepcopy(sample)
            public_sample["url"] = f"/workspace/projects/{payload['id']}/{sample['relative_path']}"
            public_sample["thumbnail_url"] = (
                f"/workspace/projects/{payload['id']}/{sample['thumbnail_path']}"
            )
            by_class[class_id].append(public_sample)
        for class_item in result["classes"]:
            class_samples = sorted(
                by_class.get(class_item["id"], []),
                key=lambda sample: sample.get("created_at", ""),
            )
            class_item["sample_count"] = len(class_samples)
            class_item["recording_session_count"] = len(
                {
                    str(
                        sample.get("recording_session_id")
                        or sample.get("source_name")
                        or sample.get("id")
                    )
                    for sample in class_samples
                }
            )
            class_item["duration_seconds"] = float(
                sum(float(sample.get("duration_seconds", 0.0) or 0.0) for sample in class_samples)
            )
            if include_samples:
                # Keep the UI responsive while still preserving all samples on disk.
                class_item["samples"] = class_samples[-240:]
        result["total_samples"] = len(samples)
        result["minimum_samples_per_class"] = int(result["settings"].get("minimum_samples_per_class", 5))
        # Firmware built per board, keyed by board name. Read from disk rather than stored in
        # project.json: it is not a setting, so writing it must never touch the payload (and
        # never invalidate training). _invalidate_training() wipes models/, which removes the
        # stale firmware and its report together, so this goes back to {} on its own.
        result["deploy"] = read_deploy_reports(self._project_dir(payload["id"]) / "models")
        if normalize_kind(result.get("kind")) == "abnormal_sound":
            normal_class = next(
                (
                    item
                    for item in result["classes"]
                    if item.get("role") == NORMAL_TRAIN_ROLE
                ),
                None,
            )
            sessions = int((normal_class or {}).get("recording_session_count", 0))
            seconds = float((normal_class or {}).get("duration_seconds", 0.0))
            clips = int((normal_class or {}).get("sample_count", 0))
            minimum_sessions = int(result["settings"].get("minimum_train_sessions", 3))
            minimum_seconds = float(result["settings"].get("minimum_train_seconds", 60))
            calibrated_sessions = int(
                result["settings"].get("minimum_calibration_sessions", 6)
            )
            calibrated_seconds = float(
                result["settings"].get("minimum_calibration_seconds", 120)
            )
            result["abnormal_readiness"] = {
                "normal_clips": clips,
                "normal_sessions": sessions,
                "normal_seconds": seconds,
                "minimum_train_sessions": minimum_sessions,
                "minimum_train_seconds": minimum_seconds,
                "can_train": sessions >= minimum_sessions and seconds >= minimum_seconds,
                "minimum_calibration_sessions": calibrated_sessions,
                "minimum_calibration_seconds": calibrated_seconds,
                "calibration_candidate": (
                    sessions >= calibrated_sessions and seconds >= calibrated_seconds
                ),
                "note": "One long recording is one session even when it creates many clips.",
            }
        return result

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            return self._public(self._load(project_id), include_samples=True)

    def get_raw_project(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._load(project_id))

    def update_project(self, project_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            payload = self._load(project_id)
            setting_keys = allowed_setting_keys(payload.get("kind"))
            if "name" in changes:
                payload["name"] = clean_text(
                    changes["name"], maximum=MAX_PROJECT_NAME_LENGTH, fallback=payload["name"]
                )
            settings_changed = False
            if "settings" in changes and isinstance(changes["settings"], dict):
                self._ensure_editable(payload)
                for key, value in changes["settings"].items():
                    if key in setting_keys and payload["settings"].get(key) != value:
                        payload["settings"][key] = value
                        settings_changed = True
            if settings_changed:
                class_ids = [item["id"] for item in payload["classes"]]
                payload["settings"] = _validate_kind_settings(
                    payload["kind"], payload["settings"], class_ids=class_ids
                )
                self._invalidate_training(payload)
            self._save(payload)
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> None:
        with self._lock:
            project_dir = self.project_dir(project_id)
            shutil.rmtree(project_dir)

    def add_class(self, project_id: str, name: str | None = None) -> dict[str, Any]:
        with self._lock:
            payload = self._load(project_id)
            self._ensure_editable(payload)
            if len(payload["classes"]) >= MAX_CLASSES:
                raise ProjectError(f"A project can contain at most {MAX_CLASSES} classes.")
            index = len(payload["classes"])
            if normalize_kind(payload.get("kind")) == "abnormal_sound":
                payload["classes"].append(
                    self._new_class(
                        name or f"Anomaly Group {index}",
                        index,
                        role=ANOMALY_EVAL_ROLE,
                    )
                )
            else:
                payload["classes"].append(self._new_class(name or f"Class {index + 1}", index))
            self._invalidate_training(payload)
            self._save(payload)
        return self.get_project(project_id)

    def update_class(self, project_id: str, class_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            payload = self._load(project_id)
            self._ensure_editable(payload)
            class_item = next((item for item in payload["classes"] if item["id"] == class_id), None)
            if not class_item:
                raise ProjectError("Class does not exist.")
            if "name" in changes:
                if bool(class_item.get("locked")):
                    raise ProjectError(
                        "The Normal baseline container is locked and cannot be renamed."
                    )
                class_item["name"] = clean_text(
                    changes["name"], maximum=MAX_CLASS_NAME_LENGTH, fallback=class_item["name"]
                )
            self._save(payload)
        return self.get_project(project_id)

    def delete_class(self, project_id: str, class_id: str) -> dict[str, Any]:
        with self._lock:
            payload = self._load(project_id)
            self._ensure_editable(payload)
            class_item = next((item for item in payload["classes"] if item["id"] == class_id), None)
            if class_item is None:
                raise ProjectError("Class does not exist.")
            if normalize_kind(payload.get("kind")) == "abnormal_sound":
                if class_item.get("role") == NORMAL_TRAIN_ROLE or bool(class_item.get("locked")):
                    raise ProjectError(
                        "The Normal baseline container is locked and cannot be deleted."
                    )
            elif len(payload["classes"]) <= 2:
                raise ProjectError("At least two classes are required.")
            sample_ids = [sid for sid, sample in payload["samples"].items() if sample["class_id"] == class_id]
            for sample_id in sample_ids:
                self._delete_sample_files(payload, sample_id)
                payload["samples"].pop(sample_id, None)
            payload["classes"] = [item for item in payload["classes"] if item["id"] != class_id]
            # Prune the deleted class out of any class-scoped setting so a stale id never
            # lingers in class_thresholds/background_class_id. Training is already being
            # invalidated below, so this needs no extra _invalidate_training() of its own.
            (payload["settings"].get("class_thresholds") or {}).pop(class_id, None)
            if payload["settings"].get("background_class_id") == class_id:
                payload["settings"]["background_class_id"] = ""
            self._invalidate_training(payload)
            self._save(payload)
        return self.get_project(project_id)

    @staticmethod
    def _class_exists(payload: dict[str, Any], class_id: str) -> bool:
        return any(item["id"] == class_id for item in payload["classes"])

    def _invalidate_training(self, payload: dict[str, Any]) -> None:
        payload["training"] = {
            "state": "untrained",
            "trained_at": None,
            "report": None,
            "artifacts": {},
        }
        self._clear_models(payload)

    def add_image_bytes(
        self,
        project_id: str,
        class_id: str,
        payload_bytes: bytes,
        *,
        source_name: str = "capture.jpg",
    ) -> dict[str, Any]:
        with self._lock:
            payload = self._load(project_id)
            self._ensure_editable(payload)
            if payload["kind"] != "image":
                raise ProjectError("This endpoint is only available for image projects.")
            if not self._class_exists(payload, class_id):
                raise ProjectError("Class does not exist.")
            try:
                image = Image.open(io.BytesIO(payload_bytes))
                image = ImageOps.exif_transpose(image).convert("RGB")
            except Exception as exc:
                raise ProjectError(f"Cannot decode image: {source_name}") from exc
            if image.width < 8 or image.height < 8:
                raise ProjectError("Image is too small.")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            sample_id = uuid.uuid4().hex
            project_dir = self._project_dir(project_id)
            relative = Path("samples") / class_id / f"{sample_id}.jpg"
            thumb_relative = Path("thumbs") / class_id / f"{sample_id}.jpg"
            output = path_within(project_dir, relative)
            thumb_output = path_within(project_dir, thumb_relative)
            output.parent.mkdir(parents=True, exist_ok=True)
            thumb_output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output, "JPEG", quality=93, optimize=True)
            thumb = ImageOps.fit(image, (112, 112), method=Image.Resampling.LANCZOS)
            thumb.save(thumb_output, "JPEG", quality=84, optimize=True)
            payload["samples"][sample_id] = {
                "id": sample_id,
                "class_id": class_id,
                "kind": "image",
                "relative_path": relative.as_posix(),
                "thumbnail_path": thumb_relative.as_posix(),
                "source_name": clean_text(source_name, maximum=200, fallback="image.jpg"),
                "created_at": utc_now_iso(),
                "width": image.width,
                "height": image.height,
            }
            self._invalidate_training(payload)
            self._save(payload)
            return deepcopy(payload["samples"][sample_id])

    def add_audio_bytes(
        self,
        project_id: str,
        class_id: str,
        payload_bytes: bytes,
        *,
        source_name: str = "recording.wav",
        overlap: float = 0.0,
        maximum_clips: int = 120,
        recording_session_id: str | None = None,
        device_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._load(project_id)
            self._ensure_editable(payload)
            if not is_audio_like(payload.get("kind")):
                raise ProjectError("This endpoint is only available for audio-like projects.")
            if not self._class_exists(payload, class_id):
                raise ProjectError("Class does not exist.")
            payload["settings"] = _validate_kind_settings(payload["kind"], payload["settings"])
            frontend = config_from_mapping(payload["settings"])
            _, signal = read_wav_bytes(payload_bytes, frontend.sample_rate)
            source_samples = int(signal.size)
            clips = split_clips(signal, frontend.clip_samples, overlap=overlap)
            if len(clips) > maximum_clips:
                clips = clips[:maximum_clips]
            if source_samples <= frontend.clip_samples:
                starts = [0]
            else:
                step = max(1, int(round(frontend.clip_samples * (1.0 - overlap))))
                starts = list(range(0, source_samples - frontend.clip_samples + 1, step))
                last_end = starts[-1] + frontend.clip_samples if starts else 0
                if last_end < source_samples:
                    starts.append(max(0, source_samples - frontend.clip_samples))
            starts = starts[: len(clips)]
            if len(starts) != len(clips):  # pragma: no cover - mirrors split_clips exactly
                raise ProjectError("Internal audio split metadata mismatch.")

            generated_session_id = uuid.uuid4().hex
            session_id = clean_text(
                recording_session_id or generated_session_id,
                maximum=200,
                fallback=generated_session_id,
            )
            capture_metadata = _json_device_metadata(device_metadata)
            project_dir = self._project_dir(project_id)
            added: list[dict[str, Any]] = []
            for clip_index, clip in enumerate(clips):
                sample_id = uuid.uuid4().hex
                relative = Path("samples") / class_id / f"{sample_id}.wav"
                thumb_relative = Path("thumbs") / class_id / f"{sample_id}.png"
                output = path_within(project_dir, relative)
                thumb_output = path_within(project_dir, thumb_relative)
                output.parent.mkdir(parents=True, exist_ok=True)
                thumb_output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(encode_wav_bytes(clip, frontend.sample_rate))
                feature = log_mel_spectrogram(clip, frontend)
                spectrogram_thumbnail(feature, thumb_output)
                sample = {
                    "id": sample_id,
                    "class_id": class_id,
                    "kind": "audio",
                    "relative_path": relative.as_posix(),
                    "thumbnail_path": thumb_relative.as_posix(),
                    "source_name": clean_text(source_name, maximum=200, fallback="recording.wav"),
                    "source_clip_index": clip_index,
                    "source_start_sample": int(starts[clip_index]),
                    "source_end_sample": int(
                        min(source_samples, starts[clip_index] + frontend.clip_samples)
                    ),
                    "source_samples": source_samples,
                    "recording_session_id": session_id,
                    "created_at": utc_now_iso(),
                    "duration_seconds": frontend.clip_seconds,
                    "sample_rate": frontend.sample_rate,
                }
                if capture_metadata is not None:
                    sample["device_metadata"] = deepcopy(capture_metadata)
                payload["samples"][sample_id] = sample
                added.append(deepcopy(sample))
            self._invalidate_training(payload)
            self._save(payload)
            return added

    def _delete_sample_files(self, payload: dict[str, Any], sample_id: str) -> None:
        sample = payload["samples"].get(sample_id)
        if not sample:
            return
        project_dir = self._project_dir(payload["id"])
        for key in ("relative_path", "thumbnail_path"):
            try:
                path_within(project_dir, sample[key]).unlink(missing_ok=True)
            except (KeyError, ValueError):
                pass

    def delete_sample(self, project_id: str, sample_id: str) -> dict[str, Any]:
        with self._lock:
            payload = self._load(project_id)
            self._ensure_editable(payload)
            if sample_id not in payload["samples"]:
                raise ProjectError("Sample does not exist.")
            self._delete_sample_files(payload, sample_id)
            payload["samples"].pop(sample_id, None)
            self._invalidate_training(payload)
            self._save(payload)
        return self.get_project(project_id)

    def clear_class_samples(self, project_id: str, class_id: str) -> dict[str, Any]:
        with self._lock:
            payload = self._load(project_id)
            self._ensure_editable(payload)
            if not self._class_exists(payload, class_id):
                raise ProjectError("Class does not exist.")
            sample_ids = [sid for sid, sample in payload["samples"].items() if sample["class_id"] == class_id]
            for sample_id in sample_ids:
                self._delete_sample_files(payload, sample_id)
                payload["samples"].pop(sample_id, None)
            self._invalidate_training(payload)
            self._save(payload)
        return self.get_project(project_id)

    def set_training_started(self, project_id: str) -> None:
        with self._lock:
            payload = self._load(project_id)
            payload["training"] = {
                "state": "training",
                "trained_at": None,
                "report": None,
                "artifacts": {},
            }
            self._clear_models(payload)
            self._save(payload)

    def set_training_result(
        self,
        project_id: str,
        *,
        report: dict[str, Any],
        artifacts: dict[str, str],
    ) -> dict[str, Any]:
        with self._lock:
            payload = self._load(project_id)
            payload["training"] = {
                "state": "trained",
                "trained_at": utc_now_iso(),
                "report": report,
                "artifacts": artifacts,
            }
            self._save(payload)
        return self.get_project(project_id)

    def merge_training_export(
        self,
        project_id: str,
        *,
        artifacts: dict[str, str],
        conversion_report: dict[str, Any],
        calibration_samples: int,
    ) -> dict[str, Any]:
        """Merge lazily generated TFLite files into an existing trained project."""

        with self._lock:
            payload = self._load(project_id)
            training = payload.get("training", {})
            if not isinstance(training, dict) or training.get("state") != "trained":
                raise ProjectError("Train the model before exporting it.")
            merged_artifacts = dict(training.get("artifacts", {}))
            merged_artifacts.update({str(key): str(value) for key, value in artifacts.items()})
            report = deepcopy(training.get("report") or {})
            conversion = deepcopy(conversion_report or {})
            models = conversion.get("models") if isinstance(conversion, dict) else None
            if isinstance(models, dict) and models:
                conversion["state"] = "generated"
            else:
                conversion.setdefault("state", "pending")
            report["conversion"] = conversion
            dataset = report.setdefault("dataset", {})
            dataset["calibration_samples"] = int(max(0, calibration_samples))
            training["artifacts"] = merged_artifacts
            training["report"] = report
            payload["training"] = training
            models_dir = self._project_dir(project_id) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(models_dir / "training_report.json", report)
            self._save(payload)
        return self.get_project(project_id)

    def set_training_failed(self, project_id: str, message: str) -> None:
        with self._lock:
            payload = self._load(project_id)
            payload["training"] = {
                "state": "failed",
                "trained_at": None,
                "report": {"error": str(message)},
                "artifacts": {},
            }
            self._save(payload)

    def sample_paths_by_class(self, project_id: str) -> list[tuple[dict[str, Any], list[Path]]]:
        payload = self.get_raw_project(project_id)
        project_dir = self.project_dir(project_id)
        result: list[tuple[dict[str, Any], list[Path]]] = []
        for class_item in payload["classes"]:
            paths = [
                path_within(project_dir, sample["relative_path"])
                for sample in payload["samples"].values()
                if sample["class_id"] == class_item["id"]
            ]
            paths = [path for path in paths if path.is_file()]
            result.append((deepcopy(class_item), sorted(paths)))
        return result

    def audio_clip_refs(self, project_id: str) -> list[tuple[int, Path, str]]:
        """``(class_index, clip_path, session_id)`` for every audio sample still on disk.

        Ordered exactly like ``sample_paths_by_class()`` (class order, then sample id,
        which is also path order because every clip is ``samples/<class>/<sample_id>.wav``)
        so a caller can use the class index as the training label without re-deriving it.

        ``session_id`` is the ``recording_session_id`` written by ``add_audio_bytes()``.
        Clips that predate that field, or that arrived some other way, fall back to a
        per-sample ``sample-<id>`` so a session-disjoint split degrades to a clip-level one
        for those clips instead of lumping them all into one giant pseudo-session.
        """

        payload = self.get_raw_project(project_id)
        project_dir = self.project_dir(project_id)
        refs: list[tuple[int, Path, str]] = []
        for class_index, class_item in enumerate(payload["classes"]):
            for sample_id, sample in sorted(payload["samples"].items()):
                if sample.get("class_id") != class_item["id"]:
                    continue
                path = path_within(project_dir, sample["relative_path"])
                if path.is_file():
                    session = str(sample.get("recording_session_id") or f"sample-{sample_id}")
                    refs.append((class_index, path, session))
        return refs

    def export_project_archive(self, project_id: str, output: Path) -> Path:
        project_dir = self.project_dir(project_id)
        return zip_directory(project_dir, output, include_root=False)

    def import_project_archive(self, archive: Path) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="tm_local_import_", dir=self.temp_root) as temp:
            temp_dir = Path(temp)
            extracted = temp_dir / "extracted"
            safe_extract_zip(archive, extracted)
            candidates = list(extracted.rglob("project.json"))
            if not candidates:
                raise ArchiveError("Archive does not contain project.json.")
            candidates.sort(key=lambda path: len(path.parts))
            source_root = candidates[0].parent
            original = read_json(candidates[0])
            if not isinstance(original, dict) or not is_known_kind(original.get("kind")):
                raise ArchiveError("Imported project kind is invalid.")
            kind = normalize_kind(original["kind"])
            original_classes = original.get("classes")
            minimum_classes = 1 if kind == "abnormal_sound" else 2
            if (
                not isinstance(original_classes, list)
                or not minimum_classes <= len(original_classes) <= MAX_CLASSES
            ):
                raise ArchiveError(
                    f"Imported {kind} project must contain between "
                    f"{minimum_classes} and {MAX_CLASSES} class containers."
                )

            class_id_map: dict[str, str] = {}
            classes: list[dict[str, Any]] = []
            for index, class_item in enumerate(original_classes):
                if not isinstance(class_item, dict):
                    raise ArchiveError("Imported class metadata is invalid.")
                old_id = str(class_item.get("id", ""))
                if not old_id or old_id in class_id_map:
                    raise ArchiveError("Imported class IDs are invalid or duplicated.")
                if kind == "abnormal_sound":
                    expected_role = NORMAL_TRAIN_ROLE if index == 0 else ANOMALY_EVAL_ROLE
                    role = str(class_item.get("role") or expected_role).strip().lower()
                    if role != expected_role:
                        raise ArchiveError(
                            "Imported abnormal-sound roles are invalid: the first container "
                            f"must be {NORMAL_TRAIN_ROLE!r} and every later container must be "
                            f"{ANOMALY_EVAL_ROLE!r}."
                        )
                    new_class = self._new_class(
                        str(
                            class_item.get(
                                "name",
                                "Normal" if index == 0 else f"Anomaly Group {index}",
                            )
                        ),
                        index,
                        role=role,
                        locked=role == NORMAL_TRAIN_ROLE,
                    )
                else:
                    new_class = self._new_class(
                        str(class_item.get("name", f"Class {index + 1}")), index
                    )
                class_id_map[old_id] = new_class["id"]
                classes.append(new_class)

            samples: dict[str, dict[str, Any]] = {}
            original_samples = original.get("samples", {})
            if not isinstance(original_samples, dict):
                raise ArchiveError("Imported samples metadata is invalid.")
            for sample_value in original_samples.values():
                if not isinstance(sample_value, dict):
                    raise ArchiveError("Imported sample metadata is invalid.")
                old_class_id = str(sample_value.get("class_id", ""))
                if old_class_id not in class_id_map:
                    raise ArchiveError("Imported sample refers to a missing class.")
                relative_path = str(sample_value.get("relative_path", ""))
                thumbnail_path = str(sample_value.get("thumbnail_path", ""))
                try:
                    sample_file = path_within(source_root, relative_path)
                    thumbnail_file = path_within(source_root, thumbnail_path)
                except ValueError as exc:
                    raise ArchiveError("Imported sample contains an unsafe path.") from exc
                if not sample_file.is_file() or not thumbnail_file.is_file():
                    raise ArchiveError("Imported sample files are incomplete.")
                sample_id = uuid.uuid4().hex
                sample = deepcopy(sample_value)
                sample.update(
                    {
                        "id": sample_id,
                        "class_id": class_id_map[old_class_id],
                        "kind": "audio" if is_audio_like(kind) else "image",
                        "relative_path": Path(relative_path).as_posix(),
                        "thumbnail_path": Path(thumbnail_path).as_posix(),
                        "source_name": clean_text(
                            str(sample_value.get("source_name", "sample")),
                            maximum=200,
                            fallback="sample",
                        ),
                    }
                )
                samples[sample_id] = sample

            defaults = defaults_for_kind(kind)
            source_settings = original.get("settings", {})
            if isinstance(source_settings, dict):
                for key in defaults:
                    if key in source_settings:
                        defaults[key] = source_settings[key]
                # class_thresholds/background_class_id carry the ARCHIVE's class ids, but
                # every class was just remapped through class_id_map above. Remap here too
                # (dropping any id the archive doesn't actually have) -- otherwise the next
                # settings PATCH (every Train click revalidates settings against the
                # project's current classes) rejects a stale id as "not one of this
                # project's classes" and the imported project can never be retrained.
                if "background_class_id" in defaults:
                    old_background = str(defaults.get("background_class_id") or "")
                    defaults["background_class_id"] = class_id_map.get(old_background, "")
                if "class_thresholds" in defaults:
                    old_thresholds = defaults.get("class_thresholds")
                    defaults["class_thresholds"] = (
                        {
                            class_id_map[old_id]: value
                            for old_id, value in old_thresholds.items()
                            if str(old_id) in class_id_map
                        }
                        if isinstance(old_thresholds, dict)
                        else {}
                    )
            try:
                defaults = _validate_kind_settings(
                    kind, defaults, class_ids=[item["id"] for item in classes]
                )
            except (ProjectError, ValueError) as exc:
                raise ArchiveError(f"Imported {kind} settings are invalid: {exc}") from exc

            training = {"state": "untrained", "trained_at": None, "report": None, "artifacts": {}}
            original_training = original.get("training", {})
            if isinstance(original_training, dict) and original_training.get("state") == "trained":
                source_artifacts = original_training.get("artifacts", {})
                safe_artifacts: dict[str, str] = {}
                if isinstance(source_artifacts, dict):
                    for key, filename_value in source_artifacts.items():
                        filename = str(filename_value)
                        if not filename or Path(filename).name != filename:
                            continue
                        candidate = source_root / "models" / filename
                        if candidate.is_file():
                            safe_artifacts[str(key)] = filename
                # Import remaps every sample ID. Abnormal-sound thresholds bind to the
                # (clip_id, content_sha256) dataset fingerprint, so carrying its trained
                # scorer across that remap would make the calibration identity false. Keep
                # the samples/roles, but require retraining after import. Classifiers do not
                # have this ID-bound threshold contract and retain their existing behaviour.
                if kind != "abnormal_sound" and safe_artifacts.get("keras"):
                    training = {
                        "state": "trained",
                        "trained_at": original_training.get("trained_at"),
                        "report": deepcopy(original_training.get("report")),
                        "artifacts": safe_artifacts,
                    }

            new_id = str(uuid.uuid4())
            now = utc_now_iso()
            payload = {
                "schema_version": 2,
                "id": new_id,
                "name": clean_text(
                    str(original.get("name", "Imported Project")) + " (Imported)",
                    maximum=MAX_PROJECT_NAME_LENGTH,
                    fallback="Imported Project",
                ),
                "kind": kind,
                "created_at": now,
                "updated_at": now,
                "classes": classes,
                "samples": samples,
                "settings": defaults,
                "training": training,
            }
            destination = self._project_dir(new_id)
            with self._lock:
                shutil.copytree(source_root, destination)
                if kind == "abnormal_sound":
                    # The imported sample IDs were remapped, so no threshold/scorer binding
                    # from the source archive remains valid in this project.
                    imported_models = destination / "models"
                    if imported_models.exists():
                        shutil.rmtree(imported_models)
                    imported_models.mkdir(parents=True, exist_ok=True)
                self._save(payload)
            return self.get_project(new_id)



def default_store() -> ProjectStore:
    ensure_workspace()
    return ProjectStore()
