from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .audio_frontend import config_from_mapping
from .audio_pipeline import (
    _bounded_float,
    _feature_from_path,
    kws_decision_policy,
    split_audio_clips,
    write_audio_frontend_reference,
    write_audio_runner,
)
from .config import PROJECT_ROOT
from .image_pipeline import _load_image_array, write_image_runner
from .known_sound_pipeline import (
    representative_patches as known_sound_representative_patches,
    thresholds_for_labels as known_sound_thresholds,
    write_known_sound_runner,
)
from .project_store import ProjectStore
from .tflite_export import (
    ExportError,
    ProgressFn,
    VALID_TFLITE_FORMATS,
    build_model_download,
    representative_samples_sha256,
)
from .utils import atomic_write_json, read_json
from .yamnet_anomaly_pipeline import (
    FRONTEND_ARTIFACT as YAMNET_FRONTEND_ARTIFACT,
    SCORER_ARTIFACT as YAMNET_SCORER_ARTIFACT,
    load_scorer as load_yamnet_scorer,
    recalibrate_exported_runtime,
    validate_frontend_artifact as validate_yamnet_frontend_artifact,
    write_yamnet_frontend_reference,
    write_yamnet_runner,
    yamnet_representative_samples,
)

EXPORT_TIMEOUT_SECONDS = max(300, int(os.environ.get("TM_LOCAL_EXPORT_TIMEOUT", "1200")))
YAMNET_INTEGER_PARITY_LIMITS = {
    "mean_cosine_similarity": 0.95,
    "minimum_cosine_similarity": 0.80,
    "mean_relative_l2_error": 0.35,
}

# Below this top-1 agreement between the Float and INT8 classifier, a student is more
# likely looking at a training problem (too few samples, over-aggressive fine-tuning)
# than an MCU problem. image/audio are the only kinds with a plain top-1 comparison:
# known_sound is multi-label (per-class scores, not a single winner) and abnormal_sound
# has no classifier comparison at all, so both are exempt rather than silently exempt-by-
# missing-key.
INT8_AGREEMENT_MINIMUM = 0.95


def int8_agreement_warnings(
    kind: str, report: dict[str, Any], *, minimum: float = INT8_AGREEMENT_MINIMUM
) -> list[str]:
    """Traditional-Chinese warning(s) when INT8/Float top-1 agreement is too low.

    ``report`` is the conversion report shape (``{"models": {"int8": {...}}}``), not the
    outer training report. Pure function: no I/O, so it is trivially testable. Called from
    ``ensure_export_artifacts``, which persists the result into
    ``training.report.conversion.warnings``; ``build_model_download`` later reads that
    already-persisted list rather than calling this function again.
    """
    if kind not in {"image", "audio"}:
        return []
    entry = (report.get("models") or {}).get("int8") or {}
    agreement = (entry.get("comparison") or {}).get("top1_agreement")
    if agreement is None:
        return []
    try:
        agreement = float(agreement)
    except (TypeError, ValueError):
        # A malformed comparison is "no evidence", not a warning.
        return []
    if agreement >= minimum:
        return []
    message = (
        f"INT8 模型與 Float 模型的 top-1 一致率只有 {agreement * 100:.1f}%"
        f"（建議 ≥ {minimum * 100:.0f}%）："
        "量化後準確率可能下降，可增加樣本、降低增強強度或關閉微調後重新訓練與匯出"
    )
    return [message]


def _emit(progress: ProgressFn | None, value: float, message: str) -> None:
    if progress:
        progress(float(max(0.0, min(1.0, value))), str(message))


def _round_robin_paths(
    class_paths: list[tuple[dict[str, Any], list[Path]]], limit: int
) -> list[Path]:
    queues = [list(paths) for _, paths in class_paths]
    selected: list[Path] = []
    cursor = 0
    while len(selected) < limit and any(queues):
        queue = queues[cursor % len(queues)]
        if queue:
            selected.append(queue.pop(0))
        cursor += 1
    return selected


def _audio_train_class_paths(
    store: ProjectStore,
    project: dict[str, Any],
    settings: dict[str, Any],
    class_paths: list[tuple[dict[str, Any], list[Path]]],
) -> list[tuple[dict[str, Any], list[Path]]]:
    """``class_paths`` restricted to the clips the model actually trained on.

    Calibrating on every clip lets held-out audio set the INT8 quantisation ranges, which
    is exactly the leak the known_sound branch already avoids: the validation accuracy in
    the report would then describe a model whose numeric ranges were fitted with the
    validation set in hand.

    The split is RE-DERIVED rather than read back from the report because the report
    records session ids, not clip paths, and a fallback class (one with a single recording
    session) has no held-out session to name. Re-deriving is exact rather than approximate:
    ``ProjectStore._invalidate_training()`` deletes the whole ``models/`` tree whenever a
    sample is added or removed, so the clip list here is the one training saw, and
    ``split_audio_clips()`` is seeded.

    A class whose clips all landed in validation keeps its original list. Dropping it would
    take that class's dynamic range out of the calibration set entirely, which skews the
    quantisation ranges further than the leak this function exists to prevent.
    """

    refs = store.audio_clip_refs(project["id"])
    if not refs:
        return class_paths
    split, _info = split_audio_clips(
        refs,
        _bounded_float(settings.get("validation_split"), 0.0, 0.45, 0.2),
        bool(settings.get("session_disjoint_validation", True)),
        class_count=len(class_paths),
    )
    train = set(split.train_paths)
    restricted: list[tuple[dict[str, Any], list[Path]]] = []
    for class_item, paths in class_paths:
        kept = [path for path in paths if path in train]
        restricted.append((class_item, kept or list(paths)))
    return restricted


def _representative_samples(
    store: ProjectStore,
    project: dict[str, Any],
    progress: ProgressFn | None,
) -> list[np.ndarray]:
    report = project.get("training", {}).get("report") or {}
    settings = {**project.get("settings", {}), **(report.get("settings") or {})}
    if project["kind"] == "abnormal_sound":
        _emit(progress, 0.02, "Preparing normal-only YAMNet calibration patches…")
        samples = yamnet_representative_samples(store, project["id"])
        _emit(
            progress,
            0.14,
            f"Prepared {len(samples)} normal-only YAMNet calibration patches.",
        )
        return samples
    if project["kind"] == "known_sound":
        # Train sessions only: letting held-out audio set the quantisation ranges would
        # stop the validation set being held out.
        _emit(progress, 0.02, "Preparing YAMNet calibration patches from training audio…")
        samples = known_sound_representative_patches(store, project["id"])
        _emit(progress, 0.14, f"Prepared {len(samples)} calibration patches.")
        return samples
    class_paths = store.sample_paths_by_class(project["id"])
    if project["kind"] == "audio":
        class_paths = _audio_train_class_paths(store, project, settings, class_paths)
    paths = _round_robin_paths(class_paths, 96 if project["kind"] == "image" else 120)
    samples: list[np.ndarray] = []
    if project["kind"] == "image":
        image_size = int(settings.get("image_size", 224))
        for index, path in enumerate(paths):
            samples.append(_load_image_array(path, image_size))
            if index % 8 == 0 or index + 1 == len(paths):
                _emit(
                    progress,
                    0.02 + 0.12 * (index + 1) / max(1, len(paths)),
                    f"Preparing calibration image {index + 1}/{len(paths)}…",
                )
    else:
        frontend = config_from_mapping(settings)
        for index, path in enumerate(paths):
            samples.append(_feature_from_path(path, frontend))
            if index % 8 == 0 or index + 1 == len(paths):
                _emit(
                    progress,
                    0.02 + 0.12 * (index + 1) / max(1, len(paths)),
                    f"Preparing calibration audio {index + 1}/{len(paths)}…",
                )
    return samples


def _tail(path: Path, limit: int = 5000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()
    except OSError:
        return ""


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def _run_conversion_worker(
    *,
    keras_path: Path,
    representative: list[np.ndarray],
    models_dir: Path,
    prefix: str,
    selected: list[str],
    progress: ProgressFn | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not representative and any(name in {"int8", "uint8"} for name in selected):
        raise ExportError("No representative samples are available for integer quantization.")

    with tempfile.TemporaryDirectory(prefix="tm_local_convert_", dir=models_dir.parent) as temp_name:
        temp = Path(temp_name)
        rep_path = temp / "representative.npz"
        stacked = (
            np.stack([np.asarray(item, dtype=np.float32) for item in representative], axis=0)
            if representative
            else np.empty((0, 1), dtype=np.float32)
        )
        np.savez_compressed(rep_path, samples=stacked)
        progress_path = temp / "progress.json"
        result_path = temp / "result.json"
        spec_path = temp / "spec.json"
        log_path = temp / "conversion_worker.log"
        atomic_write_json(
            spec_path,
            {
                "keras_path": str(keras_path),
                "representative_path": str(rep_path),
                "output_dir": str(models_dir),
                "prefix": prefix,
                "selected": selected,
                "progress_path": str(progress_path),
                "result_path": str(result_path),
            },
        )
        env = os.environ.copy()
        env.setdefault("TF_USE_LEGACY_KERAS", "1")
        flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        started = time.monotonic()
        last_value = 0.0
        last_message = "Starting isolated TensorFlow Lite conversion…"
        with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
            process = subprocess.Popen(
                [sys.executable, "-m", "tm_local.conversion_worker", str(spec_path)],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=flags,
            )
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed > EXPORT_TIMEOUT_SECONDS:
                        _stop_process(process)
                        try:
                            log_handle.flush()
                            shutil.copy2(log_path, models_dir / "conversion_worker.log")
                        except OSError:
                            pass
                        raise ExportError(
                            f"TensorFlow Lite conversion exceeded {EXPORT_TIMEOUT_SECONDS // 60} "
                            "minutes and was stopped. The trained Keras model and samples are safe. "
                            "See models\\conversion_worker.log for details."
                        )
                    if progress_path.is_file():
                        try:
                            current = read_json(progress_path)
                            last_value = float(current.get("progress", last_value))
                            last_message = str(current.get("message", last_message))
                        except Exception:
                            pass
                    minutes, seconds = divmod(int(elapsed), 60)
                    _emit(
                        progress,
                        0.16 + 0.77 * max(0.0, min(1.0, last_value)),
                        f"{last_message} · elapsed {minutes}:{seconds:02d}",
                    )
                    time.sleep(0.75)
            except BaseException:
                _stop_process(process)
                raise

        if process.returncode != 0:
            if progress_path.is_file():
                try:
                    last_message = str(read_json(progress_path).get("message", last_message))
                except Exception:
                    pass
            # Keep the worker log beside the trained model so the user can
            # inspect it after the temporary job directory is removed.
            try:
                shutil.copy2(log_path, models_dir / "conversion_worker.log")
            except OSError:
                pass
            detail = f"Conversion worker failed: {last_message}"
            log_tail = _tail(log_path)
            if log_tail:
                detail += "\n\nWorker log tail:\n" + log_tail
            raise ExportError(detail)
        if not result_path.is_file():
            raise ExportError("Conversion worker ended without a result file.")
        result = read_json(result_path)
        artifacts = result.get("artifacts")
        report = result.get("report")
        if not isinstance(artifacts, dict) or not isinstance(report, dict):
            raise ExportError("Conversion worker returned an invalid result.")
        shutil.copy2(log_path, models_dir / "conversion_worker.log")
        return {str(k): str(v) for k, v in artifacts.items()}, report


def _filename(project: dict[str, Any], format_name: str) -> str:
    return f"{_model_prefix(project)}_{format_name}.tflite"


def _model_prefix(project: dict[str, Any]) -> str:
    kind = project.get("kind")
    if kind == "image":
        return "image_classifier"
    if kind == "audio":
        return "audio_classifier_spectrogram"
    if kind == "abnormal_sound":
        return "abnormal_sound_yamnet_embedding"
    if kind == "known_sound":
        # The depth is in the filename because it changes the model's size by up to 20x,
        # and a user comparing two exports needs to tell them apart on disk.
        report = (project.get("training") or {}).get("report") or {}
        depth = int((report.get("encoder") or {}).get("depth", 14))
        suffix = "" if depth >= 14 else f"_d{depth}"
        return f"known_sound_yamnet_classifier{suffix}"
    raise ExportError(f"Unsupported project kind: {kind!r}.")


# Per-class score parity limits for the multi-label classifier. Unlike abnormal_sound,
# which compares embedding GEOMETRY, a classifier is judged on whether quantization
# changes the ANSWER. detection_agreement is therefore the primary gate; the two score
# errors are secondary guards against a model that agrees by luck.
#
# These numbers are MEASURED, not guessed. Tone-vs-noise fixture, 44 representative
# patches, TF 2.21 Windows CPU, strict INT8:
#
#   epochs=120 (the default)  mean 0.0138  max 0.0528  detection agreement 1.000
#   epochs=20  (undertrained) mean 0.0349  max 0.1534  detection agreement 0.977
#
# The error concentrates almost entirely in mid-range scores (mean error 0.055 for scores
# in 0.2-0.8 versus 0.016 for saturated ones), because a sigmoid is steepest there. A
# well-trained head saturates and quantises cleanly; an undertrained one does not. The
# limits below pass both cases while still catching a genuinely broken conversion, so a
# student who under-trains gets a usable export rather than a confusing hard failure.
KNOWN_SOUND_PARITY_LIMITS = {
    "detection_agreement": 0.95,
    "mean_absolute_error": 0.08,
    "maximum_absolute_error": 0.25,
}


def _validate_known_sound_parity(
    conversion_report: dict[str, Any],
    requested: Iterable[str],
    representative: list[np.ndarray],
) -> None:
    """Reject integer classifier artifacts whose per-class scores drift too far."""

    model_reports = conversion_report.get("models")
    if not isinstance(model_reports, dict):
        raise ExportError("The TensorFlow Lite conversion report is missing model audits.")
    failures: list[str] = []
    expected_representative_sha = representative_samples_sha256(representative)
    for runtime_name in requested:
        if runtime_name not in {"int8", "uint8"}:
            continue
        runtime_report = model_reports.get(runtime_name)
        if not isinstance(runtime_report, dict) or runtime_report.get(
            "representative_sha256"
        ) != expected_representative_sha:
            failures.append(
                f"{runtime_name}: representative calibration provenance does not match"
            )
            continue
        comparison = runtime_report.get("comparison")
        if not isinstance(comparison, dict) or int(comparison.get("sample_count", 0)) < 1:
            failures.append(f"{runtime_name}: missing Keras/TFLite score comparison")
            continue
        try:
            mean_error = float(comparison["mean_absolute_error"])
            max_error = float(comparison["maximum_absolute_error"])
            agreement = float(comparison["detection_agreement"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{runtime_name}: invalid score comparison metrics")
            continue
        if not np.all(np.isfinite([mean_error, max_error, agreement])):
            failures.append(f"{runtime_name}: non-finite score comparison metrics")
            continue
        if agreement < KNOWN_SOUND_PARITY_LIMITS["detection_agreement"]:
            failures.append(
                f"{runtime_name}: quantization changed {(1.0 - agreement) * 100:.1f}% of "
                f"the detect/not-detect decisions (agreement={agreement:.4f})"
            )
        if (
            mean_error > KNOWN_SOUND_PARITY_LIMITS["mean_absolute_error"]
            or max_error > KNOWN_SOUND_PARITY_LIMITS["maximum_absolute_error"]
        ):
            failures.append(
                f"{runtime_name}: mean absolute score error={mean_error:.4f}, "
                f"maximum={max_error:.4f}"
            )
    if failures:
        raise ExportError(
            "Known Sound integer score parity check failed: " + "; ".join(failures)
        )


def _validate_yamnet_integer_parity(
    conversion_report: dict[str, Any],
    requested: Iterable[str],
    representative: list[np.ndarray],
) -> None:
    """Reject integer YAMNet artifacts whose embeddings drift too far.

    Full-integer tensor/operator checks establish deployability.  This second
    gate establishes that quantization did not silently destroy the embedding
    geometry used by the anomaly scorer.
    """

    model_reports = conversion_report.get("models")
    if not isinstance(model_reports, dict):
        raise ExportError("The TensorFlow Lite conversion report is missing model audits.")
    failures: list[str] = []
    expected_representative_sha = representative_samples_sha256(representative)
    for runtime_name in requested:
        if runtime_name not in {"int8", "uint8"}:
            continue
        runtime_report = model_reports.get(runtime_name)
        if not isinstance(runtime_report, dict) or runtime_report.get(
            "representative_sha256"
        ) != expected_representative_sha:
            failures.append(
                f"{runtime_name}: representative calibration provenance does not match"
            )
            continue
        comparison = runtime_report.get("comparison") if isinstance(runtime_report, dict) else None
        if not isinstance(comparison, dict) or int(comparison.get("sample_count", 0)) < 1:
            failures.append(f"{runtime_name}: missing Keras/TFLite embedding comparison")
            continue
        try:
            mean_cosine = float(comparison["mean_cosine_similarity"])
            minimum_cosine = float(comparison["minimum_cosine_similarity"])
            relative_l2 = float(comparison["mean_relative_l2_error"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{runtime_name}: invalid embedding comparison metrics")
            continue
        metrics = np.asarray([mean_cosine, minimum_cosine, relative_l2], dtype=np.float64)
        if not np.all(np.isfinite(metrics)):
            failures.append(f"{runtime_name}: non-finite embedding comparison metrics")
            continue
        if (
            mean_cosine < YAMNET_INTEGER_PARITY_LIMITS["mean_cosine_similarity"]
            or minimum_cosine < YAMNET_INTEGER_PARITY_LIMITS["minimum_cosine_similarity"]
            or relative_l2 > YAMNET_INTEGER_PARITY_LIMITS["mean_relative_l2_error"]
        ):
            failures.append(
                f"{runtime_name}: mean cosine={mean_cosine:.4f}, "
                f"minimum cosine={minimum_cosine:.4f}, relative L2={relative_l2:.4f}"
            )
    if failures:
        raise ExportError(
            "YAMNet integer embedding parity check failed: " + "; ".join(failures)
        )


def ensure_export_artifacts(
    store: ProjectStore,
    project_id: str,
    selected_formats: Iterable[str],
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    selected = list(dict.fromkeys(str(item).strip().lower() for item in selected_formats))
    invalid = [item for item in selected if item not in {*VALID_TFLITE_FORMATS, "keras"}]
    if invalid:
        raise ExportError("Unsupported export format(s): " + ", ".join(invalid))
    if not selected:
        raise ExportError("Select at least one model format.")

    project = store.get_raw_project(project_id)
    training = project.get("training", {})
    if training.get("state") != "trained":
        raise ExportError("Train the model before exporting it.")
    models_dir = store.project_dir(project_id) / "models"
    artifacts = dict(training.get("artifacts", {}))
    keras_name = artifacts.get("keras")
    keras_path = models_dir / str(keras_name or "")
    if not keras_name or not keras_path.is_file():
        raise ExportError("The trained Keras model is missing. Retrain the project.")
    if project["kind"] == "abnormal_sound":
        try:
            if artifacts.get("yamnet_frontend") != YAMNET_FRONTEND_ARTIFACT:
                raise ExportError(
                    "The trained project does not declare its pinned YAMNet frontend. "
                    "Retrain the project."
                )
            validate_yamnet_frontend_artifact(models_dir / YAMNET_FRONTEND_ARTIFACT)
            load_yamnet_scorer(
                models_dir / YAMNET_SCORER_ARTIFACT,
                "keras",
                artifact_path=keras_path,
            )
        except Exception as exc:
            raise ExportError(str(exc)) from exc

    tflite_requested = [name for name in selected if name in VALID_TFLITE_FORMATS]
    generated: dict[str, str] = {}
    report = training.get("report") or {}
    conversion_report = report.get("conversion") or {}
    calibration_count = int((report.get("dataset") or {}).get("calibration_samples", 0))

    if tflite_requested:
        # Every requested TFLite artifact goes through the isolated worker, even when a
        # same-named file already exists. convert_selected() freshly inspects its tensor
        # dtypes/operators and only reuses it when it exactly matches the requested format;
        # a stale or fake "int8" file is therefore never trusted by filename/size alone.
        needs_calibration = any(name in {"int8", "uint8"} for name in tflite_requested)
        if needs_calibration:
            _emit(progress, 0.01, "Preparing representative calibration data…")
            representative = _representative_samples(store, project, progress)
            calibration_count = len(representative)
        else:
            representative = []
            _emit(progress, 0.14, "No calibration data is needed for the selected format…")
        prefix = _model_prefix(project)
        generated, conversion_report = _run_conversion_worker(
            keras_path=keras_path,
            representative=representative,
            models_dir=models_dir,
            prefix=prefix,
            selected=tflite_requested,
            progress=progress,
        )
        if project["kind"] == "abnormal_sound":
            _validate_yamnet_integer_parity(
                conversion_report,
                tflite_requested,
                representative,
            )
        elif project["kind"] == "known_sound":
            _validate_known_sound_parity(
                conversion_report,
                tflite_requested,
                representative,
            )
        agreement_warnings = int8_agreement_warnings(project["kind"], conversion_report)
        if agreement_warnings and "int8" in conversion_report.get("models", {}):
            conversion_report["models"]["int8"]["warnings"] = agreement_warnings
        conversion_report["warnings"] = agreement_warnings
        artifacts.update(generated)
    else:
        _emit(progress, 0.90, "Requested model files already exist; reusing them.")

    if project["kind"] == "abnormal_sound":
        runtime_calibration = conversion_report.setdefault(
            "anomaly_runtime_calibration", {}
        )
        for runtime_name in tflite_requested:
            artifact_name = artifacts.get(runtime_name)
            artifact_path = models_dir / str(artifact_name or "")
            if not artifact_name or not artifact_path.is_file():
                continue
            _emit(
                progress,
                0.91,
                f"Recalibrating the {runtime_name} normal reference and threshold…",
            )
            runtime_calibration[runtime_name] = recalibrate_exported_runtime(
                store,
                project_id,
                runtime_name,
                artifact_path,
            )
        artifacts["yamnet_scorer"] = YAMNET_SCORER_ARTIFACT
        artifacts["yamnet_frontend"] = YAMNET_FRONTEND_ARTIFACT

    preferred = next((name for name in ("int8", "uint8", "float32", "dynamic") if name in selected), None)
    if preferred:
        filename = _filename(project, preferred)
        if project["kind"] == "image":
            settings = {**project.get("settings", {}), **(report.get("settings") or {})}
            write_image_runner(models_dir, int(settings.get("image_size", 224)), filename)
        elif project["kind"] == "audio":
            frontend_path = models_dir / "audio_frontend.json"
            if not frontend_path.is_file():
                frontend = config_from_mapping(project.get("settings", {}))
                frontend_path.write_text(
                    json.dumps(frontend.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            write_audio_frontend_reference(models_dir)
            # The training report wins over the live project settings: the runner must
            # decide with the threshold the model was evaluated against, which is also the
            # field mcu/contract.py compiles into the firmware.
            policy = kws_decision_policy(report.get("settings") or {})
            write_audio_runner(models_dir, filename, policy["detection_threshold"])
        elif project["kind"] == "abnormal_sound":
            write_yamnet_frontend_reference(models_dir)
            write_yamnet_runner(models_dir, filename, preferred)
            artifacts["yamnet_frontend_reference"] = "yamnet_frontend_reference.py"
        elif project["kind"] == "known_sound":
            # labels.txt is written by build_model_download() in the project's standard
            # "<index> <name>" form, as for the other classifier kinds. abnormal_sound is
            # the only kind that deliberately ships none.
            class_names = [item["name"] for item in project["classes"]]
            write_yamnet_frontend_reference(models_dir)
            # One threshold per class, in labels.txt order; a report written before
            # per-class thresholds existed falls back to the project-wide threshold.
            write_known_sound_runner(
                models_dir,
                filename,
                class_names,
                known_sound_thresholds(report.get("settings") or {}, len(class_names)),
            )
            artifacts["yamnet_frontend_reference"] = "yamnet_frontend_reference.py"
        artifacts["run_model"] = "run_model.py"
    if (models_dir / "conversion_worker.log").is_file():
        artifacts["conversion_log"] = "conversion_worker.log"
    artifacts["keras"] = keras_path.name
    if conversion_report:
        atomic_write_json(models_dir / "conversion_report.json", conversion_report)

    saved = store.merge_training_export(
        project_id,
        artifacts=artifacts,
        conversion_report=conversion_report,
        calibration_samples=calibration_count,
    )
    _emit(progress, 0.96, "Selected model formats are ready; packaging ZIP…")
    return {"project": saved, "generated": generated}


def create_model_export(
    *,
    store: ProjectStore,
    project_id: str,
    selected: Iterable[str],
    include_c_header: bool,
    output_zip: Path,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    requested = list(dict.fromkeys(str(item).strip().lower() for item in selected if str(item).strip()))
    result = ensure_export_artifacts(store, project_id, requested, progress)
    project = store.get_raw_project(project_id)
    _emit(progress, 0.97, "Packaging labels, reports and selected model files…")
    build_model_download(
        project=project,
        project_dir=store.project_dir(project_id),
        selected=requested,
        include_c_header=bool(include_c_header),
        output_zip=Path(output_zip),
    )
    _emit(progress, 1.0, "Model conversion and ZIP packaging complete.")
    report = project.get("training", {}).get("report") or {}
    calibration_samples = int((report.get("dataset") or {}).get("calibration_samples", 0))
    return {
        "project": store.get_project(project_id),
        "formats": requested,
        "calibration_samples": calibration_samples,
        "output_zip": str(Path(output_zip)),
        "generated": result.get("generated", {}),
    }
