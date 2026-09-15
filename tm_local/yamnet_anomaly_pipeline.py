"""YAMNet feature-transfer backend for ``kind="abnormal_sound"``.

The pretrained encoder is frozen.  Training means fitting a session-balanced one-class
normal reference, calibrating its threshold on held-out normal sessions, and calibrating an
independent RMS branch.  User-provided anomaly groups are evaluation-only and cannot affect
the fitted scorer or either threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import anomaly_schema
from .abnormal_sound_pipeline import (
    AnomalyEvalGroup,
    NormalClipSet,
    ThresholdCalibration,
    calibrate_thresholds,
    clip_levels,
    collect_project_data,
    composite_ratio,
    session_split,
)
from .audio_frontend import config_from_mapping, fix_length, read_wav_bytes
from .config import (
    ABNORMAL_SOUND_DEFAULTS,
    DEFAULT_SENSITIVITY,
    sensitivity_preset,
    validate_abnormal_sound_settings,
)
from .project_store import ProjectStore
from .tflite_export import predict_tflite, representative_samples_sha256
from .training_common import ProgressFn, TrainingError, save_native_keras_model
from .utils import atomic_write_json, read_json, sha256_file, utc_now_iso
from .yamnet_model import (
    YAMNET_ASSET_VERSION,
    YAMNET_EMBEDDING_DIM,
    YAMNET_FRONTEND_SCHEMA_VERSION,
    YAMNET_PATCH_FRAMES,
    YAMNET_MEL_BANDS,
    YAMNET_WEIGHTS_SHA256,
    YamnetAssetError,
    build_yamnet_models,
    embedding_vectors,
    frontend_contract,
    normalized_embeddings,
    waveform_to_log_mel_patches,
)

SCORER_SCHEMA_VERSION = "yamnet-normal-reference-v1"
KERAS_ARTIFACT = "abnormal_sound_yamnet.keras"
SCORER_ARTIFACT = "yamnet_scorer.json"
FRONTEND_ARTIFACT = "yamnet_frontend.json"


@dataclass(frozen=True)
class FittedReference:
    center: np.ndarray
    scale: np.ndarray
    shrinkage: float
    scale_floor: float
    train_clips: int
    train_sessions: int

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        values = normalized_embeddings(embeddings)
        z_score = (values - self.center[None, :]) / self.scale[None, :]
        return np.mean(np.square(z_score), axis=1, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "session_balanced_robust_diagonal",
            "embedding_normalization": "l2",
            "distance": "mean_squared_standardized_residual",
            "shrinkage": self.shrinkage,
            "scale_floor": self.scale_floor,
            "train_clips": self.train_clips,
            "train_sessions": self.train_sessions,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }


def _setting(settings: Mapping[str, Any], key: str) -> Any:
    return settings.get(key, ABNORMAL_SOUND_DEFAULTS.get(key))


def _normal_set(value: Any, where: str) -> NormalClipSet:
    if not isinstance(value, NormalClipSet):
        raise TrainingError(f"{where} accepts the normal_train container only.")
    return value


def _quantile_higher(values: Sequence[float], quantile: float) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        raise TrainingError("Cannot calculate a quantile from an empty collection.")
    return float(np.quantile(array, float(quantile), method="higher"))


def fit_normal_reference(
    embeddings: np.ndarray,
    session_ids: Sequence[str],
    *,
    shrinkage: float = 0.50,
    scale_floor: float = 0.0001,
) -> FittedReference:
    """Fit a robust diagonal reference without estimating a 1024 x 1024 covariance.

    Every recording session contributes one median center and one within-session robust
    scale, so a single long recording cannot dominate several independent sessions.
    Per-dimension scales combine within-session and between-session variability, then shrink
    toward the global median scale to stabilize the many dimensions with little data.
    """

    values = normalized_embeddings(embeddings)
    sessions = tuple(str(item) for item in session_ids)
    if values.shape[0] != len(sessions):
        raise TrainingError("Embedding count and session ID count do not match.")
    unique = tuple(dict.fromkeys(sessions))
    if not unique:
        raise TrainingError("The normal reference has no recording sessions.")
    shrink = float(shrinkage)
    floor = float(scale_floor)
    if not 0.0 <= shrink <= 1.0 or not floor > 0.0:
        raise TrainingError("Invalid YAMNet scorer shrinkage or scale floor.")

    grouped = [values[np.asarray([item == session for item in sessions])] for session in unique]
    session_centers = np.stack([np.median(group, axis=0) for group in grouped])
    center = np.median(session_centers, axis=0)
    within = np.stack(
        [
            1.4826 * np.median(np.abs(group - session_center[None, :]), axis=0)
            for group, session_center in zip(grouped, session_centers)
        ]
    )
    within_scale = np.median(within, axis=0)
    between_scale = 1.4826 * np.median(np.abs(session_centers - center[None, :]), axis=0)
    raw_scale = np.sqrt(np.square(within_scale) + np.square(between_scale))
    positive = raw_scale[np.isfinite(raw_scale) & (raw_scale > floor)]
    global_scale = float(np.median(positive)) if positive.size else floor
    scale = np.sqrt((1.0 - shrink) * np.square(raw_scale) + shrink * global_scale**2)
    scale = np.maximum(np.where(np.isfinite(scale), scale, global_scale), floor)
    return FittedReference(
        center=np.asarray(center, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        shrinkage=shrink,
        scale_floor=floor,
        train_clips=int(values.shape[0]),
        train_sessions=len(unique),
    )


def _reference_from_dict(payload: Mapping[str, Any]) -> FittedReference:
    center = np.asarray(payload.get("center"), dtype=np.float64)
    scale = np.asarray(payload.get("scale"), dtype=np.float64)
    if center.shape != (YAMNET_EMBEDDING_DIM,) or scale.shape != (YAMNET_EMBEDDING_DIM,):
        raise TrainingError("YAMNet scorer center/scale has the wrong shape.")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise TrainingError("YAMNet scorer center/scale contains a non-finite value.")
    if np.any(scale <= 0.0):
        raise TrainingError("YAMNet scorer scale must be strictly positive.")
    return FittedReference(
        center=center,
        scale=scale,
        shrinkage=float(payload.get("shrinkage", 0.50)),
        scale_floor=float(payload.get("scale_floor", 0.0001)),
        train_clips=int(payload.get("train_clips", 0)),
        train_sessions=int(payload.get("train_sessions", 0)),
    )


def _quality_report(normal: NormalClipSet, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Mark recording problems; never auto-delete or auto-exclude a clip."""

    frontend = config_from_mapping(dict(settings))
    rows: list[dict[str, Any]] = []
    counts = {"ok": 0, "warning": 0, "high_review": 0}
    for sample in clip_levels(normal.clips, frontend):
        reasons: list[str] = []
        status = "ok"
        if sample.pad_fraction > 0.0:
            status = "high_review"
            reasons.append(f"{sample.pad_fraction * 100:.1f}% of the clip is zero padding")
        if sample.clipping_fraction >= 0.01:
            status = "high_review"
            reasons.append(f"{sample.clipping_fraction * 100:.2f}% of samples are clipped")
        elif sample.clipping_fraction >= 0.001 and status == "ok":
            status = "warning"
            reasons.append(f"{sample.clipping_fraction * 100:.3f}% of samples touch full scale")
        if sample.level_dbfs <= -75.0:
            status = "high_review"
            reasons.append(f"RMS {sample.level_dbfs:.1f} dBFS is near silence")
        elif sample.level_dbfs <= -60.0 and status == "ok":
            status = "warning"
            reasons.append(f"RMS {sample.level_dbfs:.1f} dBFS is very quiet")
        counts[status] += 1
        rows.append(
            {
                "clip_id": sample.clip_id,
                "session_id": sample.session_id,
                "status": status,
                "reasons": reasons,
                "level_dbfs": sample.level_dbfs,
                "peak_dbfs": sample.peak_dbfs,
                "clipping_fraction": sample.clipping_fraction,
                "pad_fraction": sample.pad_fraction,
            }
        )
    return {
        "policy": "mark only; the user decides whether to exclude a flagged normal clip",
        "counts": counts,
        "clips": rows,
        "unresolved_high_review": counts["high_review"],
    }


def _embeddings_for_clips(model: Any, clips: Sequence[Any]) -> np.ndarray:
    return embedding_vectors(model, (item.signal for item in clips))


def _embeddings_for_tflite(path: Path, clips: Sequence[Any]) -> np.ndarray:
    all_patches: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    for clip in clips:
        patches = waveform_to_log_mel_patches(clip.signal)
        start = len(all_patches)
        all_patches.extend(patches)
        spans.append((start, len(all_patches)))
    if not all_patches:
        return np.empty((0, YAMNET_EMBEDDING_DIM), dtype=np.float32)
    values = predict_tflite(Path(path), np.stack(all_patches).astype(np.float32))
    return np.stack([np.mean(values[start:end], axis=0) for start, end in spans]).astype(
        np.float32
    )


def _runtime_entry(
    *,
    reference: FittedReference,
    calibration: ThresholdCalibration,
    artifact_name: str,
    artifact_sha256: str,
    representative_sha256: str,
    confidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "artifact": artifact_name,
        "artifact_sha256": artifact_sha256,
        "representative_sha256": representative_sha256,
        "reference": reference.to_dict(),
        "threshold": calibration.to_dict(),
        "confidence": dict(confidence),
    }
    payload["binding_sha256"] = _runtime_binding_digest(payload)
    return payload


def _runtime_binding_digest(payload: Mapping[str, Any]) -> str:
    bound = {str(key): value for key, value in payload.items() if key != "binding_sha256"}
    canonical = json.dumps(bound, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _representative_digest(patches: Sequence[np.ndarray]) -> str:
    return representative_samples_sha256(patches)


def validate_frontend_artifact(path: Path) -> dict[str, Any]:
    """Require the exported frontend metadata to match the pinned implementation."""

    resolved = Path(path)
    if not resolved.is_file():
        raise TrainingError(
            "The pinned YAMNet frontend metadata is missing. Retrain the project before "
            "Preview or Export."
        )
    try:
        payload = read_json(resolved)
    except Exception as exc:
        raise TrainingError(
            "The YAMNet frontend metadata cannot be read. Retrain the project."
        ) from exc
    expected = frontend_contract()
    if payload != expected:
        raise TrainingError(
            "The YAMNet frontend metadata does not match this build's pinned contract. "
            "Retrain the project before Preview or Export."
        )
    return payload


def yamnet_representative_samples(
    store: ProjectStore,
    project_id: str,
    *,
    limit: int = 160,
) -> list[np.ndarray]:
    """YAMNet patches from normal_train only; anomaly_eval is structurally excluded."""

    project = store.get_raw_project(project_id)
    data = collect_project_data(store, project_id, project.get("settings") or {})
    normal = _normal_set(data.normal, "yamnet_representative_samples()")
    split = session_split(normal.session_ids)
    train_set = normal.subset(split.train)
    patches: list[np.ndarray] = []
    for clip in train_set.clips:
        patches.extend(waveform_to_log_mel_patches(clip.signal))
    if not patches:
        return []
    step = max(1, int(math.ceil(len(patches) / max(1, int(limit)))))
    return [np.asarray(item, dtype=np.float32) for item in patches[::step][:limit]]


def _evaluate_groups(
    groups: Sequence[AnomalyEvalGroup],
    *,
    encoder: Any,
    reference: FittedReference,
    calibration: ThresholdCalibration,
    settings: Mapping[str, Any],
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    frontend = config_from_mapping(dict(settings))
    results: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if progress:
            progress(
                0.78 + 0.08 * (index + 1) / max(1, len(groups)),
                f"Evaluating anomaly group {index + 1}/{len(groups)}…",
            )
        if not len(group):
            results.append(
                {"name": group.name, "clips": 0, "detected": 0, "detection_rate": None}
            )
            continue
        embeddings = _embeddings_for_clips(encoder, group.clips)
        shape = reference.score(embeddings)
        levels = clip_levels(group.clips, frontend)
        ratios = np.asarray(
            [
                composite_ratio(
                    float(shape[row]),
                    float(level.level_dbfs),
                    calibration.t_shape,
                    calibration.c_level_db,
                    calibration.t_level_db,
                )
                for row, level in enumerate(levels)
            ],
            dtype=np.float64,
        )
        detected = int(np.count_nonzero(ratios > 1.0))
        results.append(
            {
                "name": group.name,
                "role": "anomaly_eval",
                "clips": len(group),
                "detected": detected,
                "detection_rate": detected / float(len(group)),
                "median_ratio": float(np.median(ratios)),
                "maximum_ratio": float(np.max(ratios)),
            }
        )
    return {
        "note": "Evaluation only; these clips never fit the reference or either threshold.",
        "groups": results,
    }


def _readiness_message(reasons: Sequence[str]) -> str:
    return "The normal baseline is not ready: " + "; ".join(str(item) for item in reasons)


def _runtime_confidence_payload(
    readiness: Any,
    calibration: ThresholdCalibration,
    alpha: float,
) -> dict[str, Any]:
    """Apply the actual runtime audit result, not a zero-exceedance assumption."""

    reasons = [str(item) for item in readiness.reasons]
    independent_audit = not bool(calibration.exceedances_in_sample)
    if not independent_audit:
        reasons.append(
            "no independent audit clips were available; the exceedance count is in-sample"
        )
    actual_bound = float(calibration.fpr_upper_bound_95)
    if actual_bound > 2.0 * float(alpha) + 1e-12:
        reasons.append(
            f"this runtime's measured 95% false-positive upper bound is "
            f"{actual_bound:.4f}, above 2 x alpha = {2.0 * float(alpha):.4f}"
        )
    calibrated = (
        str(readiness.confidence) == "calibrated"
        and independent_audit
        and actual_bound <= 2.0 * float(alpha) + 1e-12
    )
    return {
        "level": "calibrated" if calibrated else "low",
        "fpr_upper_bound_95": actual_bound,
        "audit_clips": int(calibration.audit_clips),
        "observed_exceedances": int(calibration.observed_exceedances),
        "exceedances_in_sample": bool(calibration.exceedances_in_sample),
        "reasons": list(dict.fromkeys(reasons)),
    }


def train_yamnet_abnormal_sound_project(
    store: ProjectStore,
    project_id: str,
    options: Mapping[str, Any] | None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Fit the frozen-embedding YAMNet anomaly backend and save a native Keras encoder."""

    project = store.get_raw_project(project_id)
    if project.get("kind") != "abnormal_sound":
        raise TrainingError("YAMNet anomaly training requires an abnormal_sound project.")
    settings = {**ABNORMAL_SOUND_DEFAULTS, **(project.get("settings") or {})}
    provided_options = dict(options or {})
    unknown_options = sorted(set(provided_options).difference(ABNORMAL_SOUND_DEFAULTS))
    if unknown_options:
        raise TrainingError(
            "Unknown Abnormal Sound training option(s): " + ", ".join(unknown_options)
        )
    mismatched_options = sorted(
        key for key, value in provided_options.items() if settings.get(key) != value
    )
    if mismatched_options:
        raise TrainingError(
            "Abnormal Sound training options must first be saved in project settings so "
            "Train, Preview, and Export share one contract. Mismatched option(s): "
            + ", ".join(mismatched_options)
        )
    try:
        validate_abnormal_sound_settings(settings)
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc
    backend = str(_setting(settings, "detector_backend")).strip().lower()
    if backend != "yamnet_embedding":
        raise TrainingError(f"Unsupported abnormal-sound backend: {backend!r}.")

    if progress:
        progress(0.02, "Reading normal recording sessions…")
    data = collect_project_data(store, project_id, settings)
    normal = _normal_set(data.normal, "train_yamnet_abnormal_sound_project()")
    if not len(normal):
        raise TrainingError("Record normal audio before training the anomaly detector.")
    split = session_split(normal.session_ids)
    train_set = normal.subset(split.train)
    calibration_set = normal.subset(split.calibration)
    audit_set = normal.subset(split.audit)
    quality = _quality_report(normal, settings)
    preset_name = str(_setting(settings, "sensitivity") or DEFAULT_SENSITIVITY)
    preset = sensitivity_preset(preset_name)
    alpha = float(preset["alpha"])
    training_readiness = anomaly_schema.assess_readiness(
        train_sessions=len(split.train),
        train_seconds=data.normal_seconds,
        held_out_sessions=len(split.held_out),
        held_out_clips=len(calibration_set) + len(audit_set),
        alpha=alpha,
        unresolved_high_review=int(quality["unresolved_high_review"]),
        train_clips=len(train_set),
    )
    if not training_readiness.can_train:
        raise TrainingError(_readiness_message(training_readiness.reasons))

    if progress:
        progress(0.08, "Loading the pinned frozen YAMNet encoder…")
    try:
        _, encoder = build_yamnet_models()
    except YamnetAssetError as exc:
        raise TrainingError(str(exc)) from exc

    if progress:
        progress(0.15, f"Extracting embeddings from {len(train_set)} normal clips…")
    train_embeddings = _embeddings_for_clips(encoder, train_set.clips)
    reference = fit_normal_reference(
        train_embeddings,
        [item.session_id for item in train_set.clips],
        shrinkage=float(_setting(settings, "yamnet_scale_shrinkage")),
        scale_floor=float(_setting(settings, "yamnet_scale_floor")),
    )
    if progress:
        progress(0.52, "Calibrating on held-out normal sessions…")
    calibration_embeddings = _embeddings_for_clips(encoder, calibration_set.clips)
    calibration_scores = reference.score(calibration_embeddings)
    audit_scores: list[float] | None = None
    audit_levels = None
    if len(audit_set):
        audit_scores = reference.score(_embeddings_for_clips(encoder, audit_set.clips)).tolist()
        audit_levels = clip_levels(audit_set.clips, config_from_mapping(settings))
    thresholds = calibrate_thresholds(
        runtime="keras",
        train_levels=clip_levels(train_set.clips, config_from_mapping(settings)),
        calibration_shape_scores=calibration_scores.tolist(),
        calibration_levels=clip_levels(calibration_set.clips, config_from_mapping(settings)),
        alpha=alpha,
        level_floor_db=float(_setting(settings, "level_tolerance_floor_db")),
        audit_shape_scores=audit_scores,
        audit_levels=audit_levels,
    )

    if progress:
        progress(0.70, "Evaluating held-out anomaly groups…")
    evaluation = _evaluate_groups(
        data.groups,
        encoder=encoder,
        reference=reference,
        calibration=thresholds,
        settings=settings,
        progress=progress,
    )
    project_dir = store.project_dir(project_id)
    models_dir = project_dir / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)
    if progress:
        progress(0.88, "Saving the frozen YAMNet embedding model…")
    keras_path = save_native_keras_model(encoder, models_dir / KERAS_ARTIFACT)
    representatives = yamnet_representative_samples(store, project_id)
    representative_sha = _representative_digest(representatives)
    dataset_fp = anomaly_schema.dataset_fingerprint(normal.fingerprint_entries())
    split_fp = anomaly_schema.split_fingerprint(split.train, split.held_out)
    confidence_readiness = anomaly_schema.assess_readiness(
        train_sessions=len(split.train),
        train_seconds=data.normal_seconds,
        held_out_sessions=len(split.held_out),
        held_out_clips=thresholds.audit_clips,
        alpha=alpha,
        unresolved_high_review=int(quality["unresolved_high_review"]),
        train_clips=len(train_set),
    )
    runtime_confidence = _runtime_confidence_payload(
        confidence_readiness,
        thresholds,
        alpha,
    )
    runtime_entry = _runtime_entry(
        reference=reference,
        calibration=thresholds,
        artifact_name=keras_path.name,
        artifact_sha256=sha256_file(keras_path),
        representative_sha256=representative_sha,
        confidence=runtime_confidence,
    )
    scorer = {
        "schema_version": SCORER_SCHEMA_VERSION,
        "detector_backend": "yamnet_embedding",
        "detector_semantics": "open_set_normal_deviation_not_classification",
        "yamnet": {
            "asset_version": YAMNET_ASSET_VERSION,
            "weights_sha256": YAMNET_WEIGHTS_SHA256,
            "encoder_frozen": True,
            "embedding_dim": YAMNET_EMBEDDING_DIM,
            "official_head_used": False,
        },
        "frontend_schema_version": YAMNET_FRONTEND_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fp,
        "split_fingerprint": split_fp,
        "sensitivity": {
            "preset": preset_name,
            "alpha": alpha,
            "votes_required": int(preset["votes_required"]),
            "window_count": int(preset["window_count"]),
            "ratio_threshold": 1.0,
        },
        "confidence": runtime_confidence,
        "runtimes": {"keras": runtime_entry},
    }
    atomic_write_json(models_dir / SCORER_ARTIFACT, scorer)
    atomic_write_json(models_dir / FRONTEND_ARTIFACT, frontend_contract())
    atomic_write_json(models_dir / "contamination_report.json", quality)

    report = {
        "project_kind": "abnormal_sound",
        "detector_kind": "open_set_anomaly",
        "detector_backend": "yamnet_embedding",
        "trained_at": utc_now_iso(),
        "statement": (
            "Frozen YAMNet embeddings are adapted to the user's normal recordings with a "
            "one-class reference. The detector does not classify or name an anomaly."
        ),
        "settings": {
            "detector_backend": "yamnet_embedding",
            "sample_rate": 16000,
            "clip_seconds": float(_setting(settings, "clip_seconds")),
            "hop_seconds": float(_setting(settings, "hop_seconds")),
            "sensitivity": preset_name,
            "alpha": alpha,
            "votes_required": int(preset["votes_required"]),
            "window_count": int(preset["window_count"]),
            "yamnet_asset_version": YAMNET_ASSET_VERSION,
            "yamnet_weights_sha256": YAMNET_WEIGHTS_SHA256,
            "encoder_frozen": True,
            "encoder_parameter_count": int(encoder.count_params()),
            "embedding_dim": YAMNET_EMBEDDING_DIM,
            "scorer": "session_balanced_robust_diagonal",
        },
        "roles": {
            "normal_train_clips": len(normal),
            "anomaly_eval_groups": [group.name for group in data.groups],
            "excluded_by_user": list(data.excluded_clip_ids),
        },
        "dataset": {
            "normal_clips": len(normal),
            "normal_seconds": data.normal_seconds,
            "sessions": len(normal.session_ids),
            "train_sessions": list(split.train),
            "calibration_sessions": list(split.calibration),
            "audit_sessions": list(split.audit),
            "train_clips": len(train_set),
            "calibration_clips": len(calibration_set),
            "audit_clips": len(audit_set),
            "calibration_samples": 0,
            "dataset_fingerprint": dataset_fp,
            "split_fingerprint": split_fp,
        },
        "contamination": quality,
        "thresholds": {"keras": thresholds.to_dict()},
        "readiness": {
            "can_train": training_readiness.can_train,
            "confidence": runtime_confidence["level"],
            "fpr_upper_bound_95": runtime_confidence["fpr_upper_bound_95"],
            "observed_exceedances": runtime_confidence["observed_exceedances"],
            "audit_clips": runtime_confidence["audit_clips"],
            "reasons": list(runtime_confidence["reasons"]),
        },
        "evaluation": evaluation,
        "history": {
            "encoder_training_epochs": 0,
            "note": "YAMNet is frozen; Train fits only the normal reference and thresholds.",
        },
        "conversion": {
            "state": "pending",
            "note": "TFLite conversion and per-runtime recalibration run only on Export.",
        },
    }
    atomic_write_json(models_dir / "training_report.json", report)
    if progress:
        progress(1.0, "YAMNet normal reference trained. Preview is ready.")
    return {
        "report": report,
        "artifacts": {
            "keras": keras_path.name,
            "training_report": "training_report.json",
            "yamnet_scorer": SCORER_ARTIFACT,
            "yamnet_frontend": FRONTEND_ARTIFACT,
            "contamination_report": "contamination_report.json",
        },
    }


def load_scorer(
    path: Path,
    runtime_name: str,
    *,
    artifact_path: Path | None = None,
) -> tuple[dict[str, Any], FittedReference, dict[str, Any]]:
    payload = read_json(Path(path))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCORER_SCHEMA_VERSION:
        raise TrainingError("The YAMNet scorer schema is missing or unsupported. Retrain the project.")
    if payload.get("detector_backend") != "yamnet_embedding":
        raise TrainingError("The stored detector backend is not YAMNet embedding.")
    if payload.get("frontend_schema_version") != YAMNET_FRONTEND_SCHEMA_VERSION:
        raise TrainingError(
            "The YAMNet scorer uses a different frontend schema. Retrain the project."
        )
    runtimes = payload.get("runtimes")
    entry = runtimes.get(runtime_name) if isinstance(runtimes, dict) else None
    if not isinstance(entry, dict):
        raise TrainingError(
            f"The {runtime_name} YAMNet scorer is not calibrated. Export that runtime first."
        )
    reference_raw = entry.get("reference")
    threshold = entry.get("threshold")
    if not isinstance(reference_raw, dict) or not isinstance(threshold, dict):
        raise TrainingError("The stored YAMNet runtime scorer is incomplete.")
    expected_binding = str(entry.get("binding_sha256", ""))
    actual_binding = _runtime_binding_digest(entry)
    if not expected_binding or expected_binding != actual_binding:
        raise TrainingError(
            "The YAMNet scorer binding is invalid or was modified. Retrain or re-export "
            "this runtime before Preview."
        )
    if artifact_path is not None:
        resolved_artifact = Path(artifact_path)
        if str(entry.get("artifact", "")) != resolved_artifact.name:
            raise TrainingError(
                f"The {runtime_name} scorer is bound to {entry.get('artifact')!r}, not "
                f"{resolved_artifact.name!r}. Retrain or re-export this runtime."
            )
        expected_artifact_sha = str(entry.get("artifact_sha256", ""))
        if not resolved_artifact.is_file() or sha256_file(resolved_artifact) != expected_artifact_sha:
            raise TrainingError(
                f"The {runtime_name} model does not match its calibrated YAMNet scorer. "
                "Retrain or re-export this runtime."
            )
    reference = _reference_from_dict(reference_raw)
    if not float(threshold.get("t_shape", 0.0)) > 0.0:
        raise TrainingError("The stored YAMNet shape threshold is invalid.")
    return payload, reference, threshold


def score_waveform(
    *,
    project_dir: Path,
    artifact_name: str,
    runtime_name: str,
    waveform: np.ndarray,
    settings: Mapping[str, Any],
    predict: Callable[[Path, str, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Score one Preview window with an already-selected Keras/TFLite runtime."""

    scorer_path = Path(project_dir) / "models" / SCORER_ARTIFACT
    artifact_path = Path(project_dir) / "models" / artifact_name
    validate_frontend_artifact(Path(project_dir) / "models" / FRONTEND_ARTIFACT)
    scorer, reference, threshold = load_scorer(
        scorer_path,
        runtime_name,
        artifact_path=artifact_path,
    )
    raw = np.asarray(waveform, dtype=np.float32).reshape(-1)
    minimum = int(round(0.975 * 16000))
    if raw.size < minimum:
        runtime_entry = (scorer.get("runtimes") or {}).get(runtime_name, {})
        confidence_payload = runtime_entry.get("confidence", scorer.get("confidence", {}))
        return {
            "runtime": runtime_name,
            "detector_kind": "open_set_anomaly",
            "verdict": "uncertain",
            "anomaly_ratio": None,
            "confidence": (
                confidence_payload.get("level", "unknown")
                if isinstance(confidence_payload, dict)
                else str(confidence_payload)
            ),
            "readiness": confidence_payload,
            "message": "Need at least 0.975 seconds of audio for one YAMNet window.",
            "components": {},
            "smoothing": scorer.get("sensitivity", {}),
        }
    frontend = config_from_mapping(dict(settings))
    fixed = fix_length(raw, frontend.clip_samples)
    patches = waveform_to_log_mel_patches(fixed)
    embedding = np.mean(predict(Path(project_dir), artifact_name, patches), axis=0, keepdims=True)
    shape_score = float(reference.score(embedding)[0])
    t_shape = float(threshold["t_shape"])
    c_level = float(threshold["c_level_db"])
    t_level = float(threshold["t_level_db"])
    rms = float(20.0 * math.log10(max(float(np.sqrt(np.mean(fixed.astype(np.float64) ** 2))), 1e-5)))
    rms = float(np.clip(rms, -100.0, 0.0))
    shape_ratio = shape_score / max(t_shape, 1e-12)
    level_ratio = abs(rms - c_level) / max(t_level, 1e-12)
    ratio = max(shape_ratio, level_ratio)
    runtime_entry = (scorer.get("runtimes") or {}).get(runtime_name, {})
    confidence_payload = runtime_entry.get("confidence", scorer.get("confidence", {}))
    confidence_level = (
        str(confidence_payload.get("level", "unknown"))
        if isinstance(confidence_payload, dict)
        else str(confidence_payload)
    )
    window_verdict = "abnormal" if ratio > 1.0 else "normal"
    calibrated = confidence_level == "calibrated"
    confidence_reasons = (
        [str(item) for item in confidence_payload.get("reasons", [])]
        if isinstance(confidence_payload, dict)
        else []
    )
    return {
        "runtime": runtime_name,
        "detector_kind": "open_set_anomaly",
        "detector_backend": "yamnet_embedding",
        "verdict": window_verdict if calibrated else "uncertain",
        "window_verdict": window_verdict,
        "threshold_exceeded": bool(ratio > 1.0),
        "anomaly_ratio": float(ratio),
        "threshold": 1.0,
        "components": {
            "embedding": {
                "score": shape_score,
                "threshold": t_shape,
                "ratio": float(shape_ratio),
            },
            "level": {
                "dbfs": rms,
                "center_dbfs": c_level,
                "tolerance_db": t_level,
                "ratio": float(level_ratio),
            },
        },
        "confidence": confidence_level,
        "readiness": confidence_payload,
        "smoothing": scorer.get("sensitivity", {}),
        "message": (
            "Score only: collect more independent Normal sessions before a calibrated "
            "Normal/Abnormal verdict."
            + (f" Reason: {confidence_reasons[0]}" if confidence_reasons else "")
            if not calibrated
            else "Calibrated temporal voting is applied by the live Preview client."
        ),
        "statement": "Deviation from the collected normal baseline; not a sound class.",
    }


def score_wav_bytes(
    *,
    project_dir: Path,
    artifact_name: str,
    runtime_name: str,
    payload: bytes,
    settings: Mapping[str, Any],
    predict: Callable[[Path, str, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    _, waveform = read_wav_bytes(payload, 16000)
    return score_waveform(
        project_dir=project_dir,
        artifact_name=artifact_name,
        runtime_name=runtime_name,
        waveform=waveform,
        settings=settings,
        predict=predict,
    )


def recalibrate_exported_runtime(
    store: ProjectStore,
    project_id: str,
    runtime_name: str,
    artifact_path: Path,
) -> dict[str, Any]:
    """Refit reference + threshold in the selected runtime's embedding geometry."""

    project = store.get_raw_project(project_id)
    settings = {**ABNORMAL_SOUND_DEFAULTS, **(project.get("settings") or {})}
    models_dir = store.project_dir(project_id) / "models"
    validate_frontend_artifact(models_dir / FRONTEND_ARTIFACT)
    scorer, _, _ = load_scorer(
        models_dir / SCORER_ARTIFACT,
        "keras",
        artifact_path=models_dir / KERAS_ARTIFACT,
    )
    representatives = yamnet_representative_samples(store, project_id)
    representative_sha = _representative_digest(representatives)
    keras_entry = (scorer.get("runtimes") or {}).get("keras", {})
    if keras_entry.get("representative_sha256") != representative_sha:
        raise TrainingError(
            "The Normal training data no longer matches the trained YAMNet scorer. "
            "Retrain before exporting another runtime."
        )
    data = collect_project_data(store, project_id, settings)
    normal = _normal_set(data.normal, "recalibrate_exported_runtime()")
    split = session_split(normal.session_ids)
    train_set = normal.subset(split.train)
    calibration_set = normal.subset(split.calibration)
    audit_set = normal.subset(split.audit)
    train_embeddings = _embeddings_for_tflite(Path(artifact_path), train_set.clips)
    reference = fit_normal_reference(
        train_embeddings,
        [item.session_id for item in train_set.clips],
        shrinkage=float(_setting(settings, "yamnet_scale_shrinkage")),
        scale_floor=float(_setting(settings, "yamnet_scale_floor")),
    )
    calibration_scores = reference.score(
        _embeddings_for_tflite(Path(artifact_path), calibration_set.clips)
    )
    audit_scores = None
    audit_levels = None
    frontend = config_from_mapping(settings)
    if len(audit_set):
        audit_scores = reference.score(
            _embeddings_for_tflite(Path(artifact_path), audit_set.clips)
        ).tolist()
        audit_levels = clip_levels(audit_set.clips, frontend)
    preset = sensitivity_preset(_setting(settings, "sensitivity"))
    thresholds = calibrate_thresholds(
        runtime=runtime_name,
        train_levels=clip_levels(train_set.clips, frontend),
        calibration_shape_scores=calibration_scores.tolist(),
        calibration_levels=clip_levels(calibration_set.clips, frontend),
        alpha=float(preset["alpha"]),
        level_floor_db=float(_setting(settings, "level_tolerance_floor_db")),
        audit_shape_scores=audit_scores,
        audit_levels=audit_levels,
    )
    scorer_path = models_dir / SCORER_ARTIFACT
    quality = _quality_report(normal, settings)
    readiness = anomaly_schema.assess_readiness(
        train_sessions=len(split.train),
        train_seconds=data.normal_seconds,
        held_out_sessions=len(split.held_out),
        held_out_clips=thresholds.audit_clips,
        alpha=float(preset["alpha"]),
        unresolved_high_review=int(quality["unresolved_high_review"]),
        train_clips=len(train_set),
    )
    runtime_confidence = _runtime_confidence_payload(
        readiness,
        thresholds,
        float(preset["alpha"]),
    )
    runtimes = scorer.setdefault("runtimes", {})
    runtimes[runtime_name] = _runtime_entry(
        reference=reference,
        calibration=thresholds,
        artifact_name=Path(artifact_path).name,
        artifact_sha256=sha256_file(Path(artifact_path)),
        representative_sha256=representative_sha,
        confidence=runtime_confidence,
    )
    atomic_write_json(scorer_path, scorer)
    return {"threshold": thresholds.to_dict(), "confidence": runtime_confidence}


def write_yamnet_frontend_reference(output_dir: Path) -> Path:
    """Write the exact TensorFlow reference frontend shipped with exported models."""

    destination = Path(output_dir) / "yamnet_frontend_reference.py"
    source = '''"""Reference YAMNet log-mel frontend exported by Local Studio."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000


def read_wav(path):
    rate, raw = wavfile.read(path)
    dtype = raw.dtype
    if np.issubdtype(dtype, np.floating):
        scaled = raw.astype(np.float64)
    elif dtype == np.uint8:
        scaled = (raw.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        scaled = raw.astype(np.float64) / float(max(abs(info.min), info.max))
    else:
        raise ValueError(f"Unsupported WAV dtype: {dtype}")
    if scaled.ndim == 2:
        scaled = np.mean(scaled, axis=1)
    elif scaled.ndim != 1:
        raise ValueError(f"Expected mono/stereo WAV, got shape {scaled.shape}")
    raw = np.clip(
        np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0
    ).astype(np.float32)
    if int(rate) != SAMPLE_RATE:
        divisor = int(np.gcd(int(rate), SAMPLE_RATE))
        raw = resample_poly(raw, SAMPLE_RATE // divisor, int(rate) // divisor)
    return np.clip(
        np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0
    ).astype(np.float32)


def fix_length(waveform, length=16000):
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size > length:
        start = max(0, (values.size - length) // 2)
        return values[start:start + length]
    if values.size < length:
        output = np.zeros(length, dtype=np.float32)
        start = (length - values.size) // 2
        output[start:start + values.size] = values
        return output
    return values


def patches(waveform):
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    minimum = 15600
    if values.size < minimum:
        values = np.pad(values, (0, minimum - values.size))
    magnitude = tf.abs(tf.signal.stft(
        tf.convert_to_tensor(values), frame_length=400, frame_step=160, fft_length=512))
    mel = tf.signal.linear_to_mel_weight_matrix(
        64, 257, 16000, 125.0, 7500.0, dtype=tf.float32)
    log_mel = tf.math.log(tf.matmul(magnitude, mel) + 0.001)
    framed = tf.signal.frame(log_mel, 96, 48, axis=0)
    return np.asarray(framed.numpy(), dtype=np.float32)
'''
    destination.write_text(source, encoding="utf-8", newline="\n")
    return destination


def write_yamnet_runner(
    output_dir: Path,
    model_filename: str,
    runtime_name: str,
) -> Path:
    """Write a PC reference runner that applies scorer and RMS metadata honestly."""

    destination = Path(output_dir) / "run_model.py"
    expected_frontend_json = json.dumps(
        frontend_contract(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    source = f'''"""Run an exported YAMNet abnormal-sound detector on one WAV file."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from yamnet_frontend_reference import fix_length, patches, read_wav

MODEL_FILENAME = {model_filename!r}
RUNTIME_NAME = {runtime_name!r}
EXPECTED_FRONTEND = json.loads({expected_frontend_json!r})


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding_digest(entry):
    bound = {{str(key): value for key, value in entry.items() if key != "binding_sha256"}}
    canonical = json.dumps(bound, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dequantize(value, detail):
    if np.issubdtype(detail["dtype"], np.floating):
        return value.astype(np.float32)
    scale, zero = detail["quantization"]
    return (value.astype(np.float32) - float(zero)) * float(scale)


def quantize(value, detail):
    if np.issubdtype(detail["dtype"], np.floating):
        return value.astype(detail["dtype"])
    scale, zero = detail["quantization"]
    info = np.iinfo(detail["dtype"])
    return np.clip(np.round(value / scale + zero), info.min, info.max).astype(detail["dtype"])


def audio_windows(waveform):
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size < 15600:
        raise SystemExit("Need at least 0.975 seconds of audio for one YAMNet window.")
    if values.size <= 16000:
        return [(0, fix_length(values))]
    starts = list(range(0, values.size - 16000 + 1, 8000))
    return [(start, values[start:start + 16000]) for start in starts]


def score_window(interpreter, input_detail, output_detail, waveform, reference, threshold):
    fixed = fix_length(waveform)
    outputs = []
    for patch in patches(fixed):
        interpreter.set_tensor(input_detail["index"], quantize(patch[None, ...], input_detail))
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        outputs.append(dequantize(output, output_detail)[0])
    embedding = np.mean(np.stack(outputs), axis=0).astype(np.float64)
    embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
    center = np.asarray(reference["center"], dtype=np.float64)
    scale = np.asarray(reference["scale"], dtype=np.float64)
    shape_score = float(np.mean(np.square((embedding - center) / scale)))
    shape_ratio = shape_score / float(threshold["t_shape"])
    rms = float(np.sqrt(np.mean(fixed.astype(np.float64) ** 2)))
    level_dbfs = max(-100.0, min(0.0, 20.0 * math.log10(max(rms, 1e-5))))
    level_ratio = abs(level_dbfs - float(threshold["c_level_db"])) / float(
        threshold["t_level_db"]
    )
    ratio = max(shape_ratio, level_ratio)
    return {{
        "window_verdict": "abnormal" if ratio > 1.0 else "normal",
        "threshold_exceeded": bool(ratio > 1.0),
        "anomaly_ratio": float(ratio),
        "embedding_ratio": float(shape_ratio),
        "level_ratio": float(level_ratio),
        "level_dbfs": float(level_dbfs),
    }}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_model.py sound.wav")
    root = Path(__file__).resolve().parent
    frontend_path = root / "yamnet_frontend.json"
    try:
        actual_frontend = json.loads(frontend_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Cannot read yamnet_frontend.json: {{exc}}") from exc
    if actual_frontend != EXPECTED_FRONTEND:
        raise SystemExit("yamnet_frontend.json does not match this runner's pinned contract.")
    scorer = json.loads((root / "yamnet_scorer.json").read_text(encoding="utf-8"))
    runtime = RUNTIME_NAME
    entry = scorer["runtimes"][runtime]
    model_path = root / MODEL_FILENAME
    if entry.get("artifact") != MODEL_FILENAME:
        raise SystemExit("The scorer is bound to a different model filename.")
    if entry.get("artifact_sha256") != sha256(model_path):
        raise SystemExit("The model SHA-256 does not match the calibrated scorer.")
    if entry.get("binding_sha256") != binding_digest(entry):
        raise SystemExit("The scorer binding is invalid or was modified.")
    reference = entry["reference"]
    threshold = entry["threshold"]
    waveform = read_wav(sys.argv[1])
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    policy = scorer.get("sensitivity") or {{}}
    window_count = max(1, int(policy.get("window_count", 5)))
    votes_required = max(1, min(window_count, int(policy.get("votes_required", 3))))
    history = []
    flags = []
    for start, window in audio_windows(waveform):
        item = score_window(
            interpreter, input_detail, output_detail, window, reference, threshold
        )
        flags.append(bool(item["threshold_exceeded"]))
        recent = flags[-window_count:]
        votes = sum(recent)
        if len(recent) < window_count:
            temporal = "uncertain"
        else:
            temporal = "abnormal" if votes >= votes_required else "normal"
        item.update({{
            "start_seconds": float(start / 16000.0),
            "end_seconds": float((start + 16000) / 16000.0),
            "temporal_verdict": temporal,
            "votes": int(votes),
            "votes_required": votes_required,
            "window_count": window_count,
        }})
        history.append(item)
    confidence_payload = entry.get("confidence") or scorer.get("confidence") or {{}}
    confidence = str(confidence_payload.get("level", "unknown"))
    temporal_states = [item["temporal_verdict"] for item in history]
    if confidence != "calibrated" or not any(state != "uncertain" for state in temporal_states):
        verdict = "uncertain"
    elif "abnormal" in temporal_states:
        verdict = "abnormal"
    else:
        verdict = "normal"
    print(json.dumps({{
        "verdict": verdict,
        "max_anomaly_ratio": max(item["anomaly_ratio"] for item in history),
        "confidence": confidence,
        "confidence_details": confidence_payload,
        "smoothing": {{
            "votes_required": votes_required,
            "window_count": window_count,
            "hop_seconds": 0.5,
            "warmup_windows": window_count - 1,
        }},
        "windows": history,
        "message": (
            "score only; collect more independent Normal sessions for a calibrated verdict"
            if confidence != "calibrated"
            else "verdict uses one-second windows, 0.5-second hop, and temporal voting"
        ),
        "statement": "normal-baseline deviation, not a sound class",
    }}, indent=2))


if __name__ == "__main__":
    main()
'''
    destination.write_text(source, encoding="utf-8", newline="\n")
    return destination
