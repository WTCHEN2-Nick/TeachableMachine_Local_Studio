from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .archive import ArchiveError
from .audio_frontend import read_wav_bytes
from .audio_pipeline import feature_from_wav_bytes
from .config import (
    MAX_FILES_PER_REQUEST,
    MAX_UPLOAD_BYTES,
    TEMP_ROOT,
    TOOL_NAME,
    WEB_ROOT,
    allowed_setting_keys,
    ensure_workspace,
    is_audio_like,
    is_multi_label,
)
from .export_service import create_model_export
from .image_pipeline import load_image_for_prediction
from .diagnostics import build_diagnostic_text
from .jobs import JobManager, ProgressCallback
from .mcu import paths as mcu_paths
from .mcu.errors import TOOLCHAIN_MISSING_MESSAGE, DeployError, long_path_reason
from .model_runtime import ModelRuntime
from .project_store import ProjectError, ProjectNotFound, ProjectStore
from .runtime_config import public_runtime_info
from .training_dispatch import train_project
from .tflite_export import ExportError
from .training_common import TrainingError
from .utils import read_json, safe_identifier, sha256_file
from .yamnet_anomaly_pipeline import score_wav_bytes


LOGGER = logging.getLogger("tm_local.app")

# One message for every "this project is already working" 409, whichever of train / export /
# deploy is running and whichever guard fires (the pre-check or the JobManager's own race check).
PROJECT_BUSY_MESSAGE = "此專案已有訓練、匯出或部署工作進行中。"


def training_settings_changed(project: dict[str, Any], options: dict[str, Any]) -> bool:
    """Whether ``options`` would actually change this project's stored settings.

    Exactly the comparison ProjectStore.update_project() makes internally -- the key has to
    be one this kind accepts, and the value has to differ -- so pressing Train twice without
    touching the Advanced panel cannot wipe a trained model. An unknown kind (which
    allowed_setting_keys() refuses) has no settings worth writing, so it answers False.
    """
    if not options:
        return False
    try:
        keys = allowed_setting_keys(project.get("kind"))
    except ValueError:
        return False
    stored = project.get("settings") or {}
    return any(key in keys and stored.get(key) != value for key, value in options.items())


def run_mcu_deploy(
    store: ProjectStore, project_id: str, board: str, progress: ProgressCallback
) -> Any:
    """Build firmware for `board`; the JobManager worker calls this.

    The import is deliberately inside the function: the MCU build stack (yaml, jinja2, the
    NuML codegen) must not be pulled in by `import tm_local.app` / `create_app()`, so a
    Studio with no mcu_toolkit still starts. Tests replace this name on the module.
    """
    from .mcu.deploy_service import run_deploy

    return run_deploy(store, project_id, board, progress)


def mcu_unavailable_reason(info: dict[str, Any]) -> str | None:
    """Student-facing reason a deploy cannot start, or None when everything is present."""
    # Path length first, and regardless of the probe: everything the probe found is still
    # there, gcc simply cannot open the deepest BSP headers once the install root pushes them
    # past Windows' MAX_PATH, and its English "No such file or directory" tells the student
    # nothing. run_deploy() repeats this check for deploys that did not come through here.
    too_long = long_path_reason(mcu_paths.MCU_TOOLKIT_ROOT)
    if too_long:
        return too_long
    if info.get("available"):
        return None
    if not (info.get("toolkit") or {}).get("available"):
        return "找不到開發板工具組 mcu_toolkit（韌體樣板）。請重新執行 01_INSTALL.bat。"
    if not (info.get("vela") or {}).get("available"):
        return r"找不到 Vela 編譯器（mcu_toolkit\vela）。請重新執行 01_INSTALL.bat。"
    if not (info.get("toolchain") or {}).get("available"):
        # Same wording run_deploy() raises later, so the pre-flight and the build agree.
        return TOOLCHAIN_MISSING_MESSAGE
    return "開發板工具鏈尚未就緒。請重新執行 01_INSTALL.bat 後重新啟動 Studio。"


def _kind_is_deployable(kind: str) -> bool:
    """True when `kind` has a firmware application id (config registry) *and* that
    application's `apps/<name>.py` module actually imports cleanly in this build.

    The second half is deliberately wrapped in a broad `except`, not just `DeployError`: a
    future `apps/kws.py` could fail with anything (a bad import inside a real dependency,
    say), and that must report only this one kind as unsupported instead of 500ing
    `/api/mcu/status` -- the endpoint that exists precisely to explain what is and is not
    available -- for every kind at once.
    """
    from .config import MCU_DEPLOYABLE_KINDS, mcu_application
    from .mcu.project_builder import application_supported

    if kind not in MCU_DEPLOYABLE_KINDS:
        return False
    try:
        return application_supported(mcu_application(kind))
    except Exception:
        LOGGER.debug("application_supported() failed for kind=%r", kind, exc_info=True)
        return False


def _currently_deployable_kind_names() -> list[str]:
    """Kind names this build can currently deploy, in `PROJECT_KINDS` registry order.

    Reuses `_kind_is_deployable()` -- the same per-kind, exception-guarded check
    `mcu_status()` uses for its own `deployable_kinds` -- instead of a second, separately
    written derivation. That second derivation (calling
    `project_builder.supported_applications()` directly) is exactly the Round 2 defect this
    replaced: it had no exception guard at all, so a real app module failing with anything
    other than `DeployError` (a broken transitive import, say) raised straight through the
    "here's what you can deploy" message a refused deploy request builds -- an unhandled 500
    on `POST .../deploy-mcu`, the same class of failure R3 already guards on `/api/mcu/status`.
    Sharing `_kind_is_deployable()` means both endpoints are guarded by construction, not by
    two call sites remembering to guard themselves the same way.
    """
    from .config import PROJECT_KINDS

    return [kind for kind in PROJECT_KINDS if _kind_is_deployable(kind)]


class CreateProjectRequest(BaseModel):
    kind: Literal["image", "audio", "abnormal_sound", "known_sound"]
    name: str | None = None


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    settings: dict[str, Any] | None = None


class CreateClassRequest(BaseModel):
    name: str | None = None


class UpdateClassRequest(BaseModel):
    name: str


class TrainRequest(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["int8"])
    include_c_header: bool = True


class DeployRequest(BaseModel):
    board: str


class FlashRequest(BaseModel):
    board: str
    method: Literal["msc", "nulink", "pyocd"]
    drive: str | None = None


# A single drive letter in the form `X:` or `X:\`, uppercase, no path components -- validated
# before request.drive ever touches the filesystem (a request body is untrusted input).
_DRIVE_LETTER_RE = re.compile(r"^[A-Z]:\\?$")


async def _read_upload(upload: UploadFile, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"{upload.filename or 'upload'} exceeds {limit} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    *,
    store: ProjectStore | None = None,
    jobs: JobManager | None = None,
    runtime: ModelRuntime | None = None,
) -> FastAPI:
    ensure_workspace()
    # Every firmware ZIP copy and deploy work dir left under TEMP_ROOT is an orphan by now:
    # download tokens live in this process's `export_files` dict and no build is running yet.
    from .mcu.housekeeping import sweep_temp_root

    sweep_temp_root()
    store = store or ProjectStore()
    recovered_interrupted_jobs = int(getattr(store, "recovered_interrupted_jobs", 0))
    jobs = jobs or JobManager(max_workers=1)
    runtime = runtime or ModelRuntime()
    export_files: dict[str, tuple[Path, str]] = {}
    export_files_lock = threading.RLock()

    app = FastAPI(
        title=TOOL_NAME,
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.store = store
    app.state.jobs = jobs

    @app.middleware("http")
    async def disable_frontend_cache(request: Request, call_next):
        response = await call_next(request)
        if request.url.path in {"/", "/index.html", "/app.js", "/style.css"}:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.state.runtime = runtime
    app.state.export_files = export_files
    app.state.recovered_interrupted_jobs = recovered_interrupted_jobs

    @app.exception_handler(ProjectNotFound)
    async def project_not_found_handler(_, exc: ProjectNotFound):  # noqa: ANN001
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    async def validation_handler(_, exc):  # noqa: ANN001
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    for exception_type in (ProjectError, TrainingError, ExportError, ArchiveError, DeployError):
        app.add_exception_handler(exception_type, validation_handler)

    def ensure_project_idle(project_id: str) -> None:
        active = jobs.active_for_project(project_id)
        if active and active.get("state") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail=PROJECT_BUSY_MESSAGE)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "name": TOOL_NAME,
            "version": __version__,
            "workspace": str(store.projects_root.parent),
            "pid": os.getpid(),
            "recovered_interrupted_jobs": recovered_interrupted_jobs,
            "runtime": public_runtime_info(),
        }

    @app.get("/api/diagnostics/download")
    def download_diagnostics() -> PlainTextResponse:
        text = build_diagnostic_text(store, jobs)
        filename = "TM_Local_Studio_Diagnostic.txt"
        return PlainTextResponse(
            text,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        return {"projects": store.list_projects()}

    @app.post("/api/projects")
    def create_project(request: CreateProjectRequest) -> dict[str, Any]:
        return store.create_project(request.kind, request.name)

    @app.post("/api/projects/import")
    async def import_project(file: UploadFile = File(...)) -> dict[str, Any]:
        payload = await _read_upload(file, limit=512 * 1024 * 1024)
        with tempfile.NamedTemporaryFile(suffix=".zip", dir=TEMP_ROOT, delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        try:
            return store.import_project_archive(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        project = store.get_project(project_id)
        project["active_job"] = jobs.active_for_project(project_id)
        return project

    @app.patch("/api/projects/{project_id}")
    def update_project(project_id: str, request: UpdateProjectRequest) -> dict[str, Any]:
        ensure_project_idle(project_id)
        changes = request.model_dump(exclude_none=True)
        return store.update_project(project_id, changes)

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str) -> dict[str, Any]:
        ensure_project_idle(project_id)
        project_dir = store.project_dir(project_id)
        runtime.clear_project(project_dir)
        store.delete_project(project_id)
        return {"ok": True}

    @app.post("/api/projects/{project_id}/classes")
    def add_class(project_id: str, request: CreateClassRequest) -> dict[str, Any]:
        ensure_project_idle(project_id)
        return store.add_class(project_id, request.name)

    @app.patch("/api/projects/{project_id}/classes/{class_id}")
    def update_class(project_id: str, class_id: str, request: UpdateClassRequest) -> dict[str, Any]:
        ensure_project_idle(project_id)
        return store.update_class(project_id, class_id, request.model_dump())

    @app.delete("/api/projects/{project_id}/classes/{class_id}")
    def delete_class(project_id: str, class_id: str) -> dict[str, Any]:
        ensure_project_idle(project_id)
        return store.delete_class(project_id, class_id)

    @app.post("/api/projects/{project_id}/classes/{class_id}/images")
    async def add_images(
        project_id: str,
        class_id: str,
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        ensure_project_idle(project_id)
        if len(files) > MAX_FILES_PER_REQUEST:
            raise HTTPException(status_code=400, detail=f"At most {MAX_FILES_PER_REQUEST} files per request.")
        added = 0
        errors: list[str] = []
        for upload in files:
            try:
                payload = await _read_upload(upload)
                store.add_image_bytes(
                    project_id,
                    class_id,
                    payload,
                    source_name=upload.filename or "image.jpg",
                )
                added += 1
            except Exception as exc:
                errors.append(f"{upload.filename or 'image'}: {exc}")
        return {"added": added, "errors": errors, "project": store.get_project(project_id)}

    @app.post("/api/projects/{project_id}/classes/{class_id}/audio")
    async def add_audio(
        project_id: str,
        class_id: str,
        files: list[UploadFile] = File(...),
        overlap: float = Form(0.0),
        recording_session_id: str | None = Form(None),
        device_label: str | None = Form(None),
        device_id: str | None = Form(None),
        device_settings: str | None = Form(None),
        device_metadata: str | None = Form(None),
    ) -> dict[str, Any]:
        ensure_project_idle(project_id)
        if len(files) > MAX_FILES_PER_REQUEST:
            raise HTTPException(status_code=400, detail=f"At most {MAX_FILES_PER_REQUEST} files per request.")
        added = 0
        errors: list[str] = []
        metadata: dict[str, Any] = {}
        if device_metadata:
            try:
                parsed_metadata = json.loads(device_metadata)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="device_metadata must be valid JSON.",
                ) from exc
            if not isinstance(parsed_metadata, dict):
                raise HTTPException(
                    status_code=400,
                    detail="device_metadata must be a JSON object.",
                )
            metadata.update(parsed_metadata)
        if device_label:
            metadata["label"] = str(device_label)
        if device_id:
            metadata["device_id"] = str(device_id)
        if device_settings:
            try:
                parsed_settings = json.loads(device_settings)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="device_settings must be valid JSON.",
                ) from exc
            if not isinstance(parsed_settings, dict):
                raise HTTPException(
                    status_code=400,
                    detail="device_settings must be a JSON object.",
                )
            metadata["track_settings"] = parsed_settings
        for upload in files:
            try:
                payload = await _read_upload(upload)
                samples = store.add_audio_bytes(
                    project_id,
                    class_id,
                    payload,
                    source_name=upload.filename or "audio.wav",
                    overlap=max(0.0, min(0.9, float(overlap))),
                    recording_session_id=recording_session_id,
                    device_metadata=metadata or None,
                )
                added += len(samples)
            except Exception as exc:
                errors.append(f"{upload.filename or 'audio'}: {exc}")
        return {"added": added, "errors": errors, "project": store.get_project(project_id)}

    @app.delete("/api/projects/{project_id}/samples/{sample_id}")
    def delete_sample(project_id: str, sample_id: str) -> dict[str, Any]:
        ensure_project_idle(project_id)
        return store.delete_sample(project_id, sample_id)

    @app.delete("/api/projects/{project_id}/classes/{class_id}/samples")
    def clear_class(project_id: str, class_id: str) -> dict[str, Any]:
        ensure_project_idle(project_id)
        return store.clear_class_samples(project_id, class_id)

    @app.post("/api/projects/{project_id}/train")
    def train(project_id: str, request: TrainRequest) -> dict[str, Any]:
        # Busy check first, and specifically before update_project(): a Train click on a
        # project that is already training / exporting / deploying has to answer the one
        # 409 PROJECT_BUSY_MESSAGE. Without this, a click that happens to carry a changed
        # Advanced option reaches update_project() -> _ensure_editable() -> ProjectError
        # -> HTTP 400 instead, so the same student action reports two different errors
        # depending on whether they touched a knob. jobs.submit() below still repeats the
        # check, because only it can close the race with a job starting in between.
        ensure_project_idle(project_id)
        project = store.get_raw_project(project_id)
        if training_settings_changed(project, request.options):
            # Persist the options BEFORE training starts, for EVERY kind: Preview, Export
            # and the firmware build all read the project settings, and none of them may
            # later see a different threshold / image size / deployment target than the
            # model was trained and evaluated with. It has to happen here rather than
            # inside the pipeline: once set_training_started() runs, _ensure_editable()
            # blocks settings writes. Only when something actually differs -- update_project
            # invalidates the trained model, and an unchanged click must not.
            try:
                store.update_project(project_id, {"settings": dict(request.options)})
            except ProjectError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            project = store.get_raw_project(project_id)

        def task(progress):  # noqa: ANN001
            store.set_training_started(project_id)
            try:
                result = train_project(
                    store,
                    project_id,
                    project["kind"],
                    # The VALIDATED settings, never request.options: both pipelines merge
                    # with {**project["settings"], **options}, so the raw client body would
                    # win over the values that were just normalised and stored. _require_bool
                    # accepts the string "false" and stores a real False -- but bool("false")
                    # is True, so the model would train with the option ON while the project
                    # file, the Export metadata and the firmware contract all said OFF. Every
                    # key the pipelines read is in this kind's *_DEFAULTS, so nothing the
                    # client could send is lost by preferring the stored copy.
                    dict(project["settings"]),
                    progress,
                )
                runtime.clear_project(store.project_dir(project_id))
                saved = store.set_training_result(
                    project_id,
                    report=result["report"],
                    artifacts=result["artifacts"],
                )
                return {"project": saved}
            except Exception as exc:
                store.set_training_failed(project_id, str(exc))
                raise

        try:
            return jobs.submit("train", project_id, task)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=PROJECT_BUSY_MESSAGE) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc

    def _artifact_for_runtime(project: dict[str, Any], runtime_name: str) -> str:
        artifacts = project.get("training", {}).get("artifacts", {})
        mapping = {
            "keras": "keras",
            "float": "keras",
            "float32": "float32",
            "int8": "int8",
            "uint8": "uint8",
            "dynamic": "dynamic",
        }
        key = mapping.get(runtime_name, "keras")
        filename = artifacts.get(key)
        if not filename:
            raise HTTPException(status_code=409, detail=f"Runtime model is unavailable: {runtime_name}")
        return filename

    def _normalized_runtime_name(runtime_name: str) -> str:
        name = str(runtime_name).strip().lower()
        return "keras" if name in {"keras", "float"} else name

    def _prediction_response(project: dict[str, Any], scores: np.ndarray, runtime_name: str) -> dict[str, Any]:
        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        labels = [item["name"] for item in project["classes"]]
        if len(values) != len(labels):
            raise HTTPException(status_code=500, detail="Model output count does not match project classes.")
        predictions = [
            {"class_name": label, "score": float(max(0.0, min(1.0, score)))}
            for label, score in zip(labels, values)
        ]
        predictions.sort(key=lambda item: item["score"], reverse=True)
        return {
            "runtime": runtime_name,
            # image/audio are softmax kinds, so their scores do sum to 1. known_sound
            # builds its own response because its sigmoid scores do not.
            "multi_label": is_multi_label(project.get("kind")),
            "predictions": predictions,
            "top_class": predictions[0]["class_name"] if predictions else None,
        }

    @app.post("/api/projects/{project_id}/predict/image")
    async def predict_image(
        project_id: str,
        file: UploadFile = File(...),
        runtime_name: str = Form("keras"),
    ) -> dict[str, Any]:
        project = store.get_raw_project(project_id)
        if project["kind"] != "image":
            raise HTTPException(status_code=400, detail="Project is not an image project.")
        if project.get("training", {}).get("state") != "trained":
            raise HTTPException(status_code=409, detail="Train the model first.")
        payload = await _read_upload(file)
        report = project["training"].get("report") or {}
        size = int(report.get("settings", {}).get("image_size", project["settings"].get("image_size", 224)))
        array = load_image_for_prediction(payload, size)
        artifact = _artifact_for_runtime(project, runtime_name)
        scores = runtime.predict(store.project_dir(project_id), artifact, array[np.newaxis, ...])[0]
        return _prediction_response(project, scores, runtime_name)

    @app.post("/api/projects/{project_id}/predict/audio")
    async def predict_audio(
        project_id: str,
        file: UploadFile = File(...),
        runtime_name: str = Form("keras"),
    ) -> dict[str, Any]:
        project = store.get_raw_project(project_id)
        if project["kind"] != "audio":
            raise HTTPException(status_code=400, detail="Project is not an audio project.")
        if project.get("training", {}).get("state") != "trained":
            raise HTTPException(status_code=409, detail="Train the model first.")
        payload = await _read_upload(file)
        report = project["training"].get("report") or {}
        settings = {**project["settings"], **report.get("settings", {})}
        feature = feature_from_wav_bytes(payload, settings)
        artifact = _artifact_for_runtime(project, runtime_name)
        scores = runtime.predict(store.project_dir(project_id), artifact, feature[np.newaxis, ...])[0]
        return _prediction_response(project, scores, runtime_name)

    @app.post("/api/projects/{project_id}/predict/abnormal-sound")
    async def predict_abnormal_sound(
        project_id: str,
        file: UploadFile = File(...),
        runtime_name: str = Form("keras"),
    ) -> dict[str, Any]:
        project = store.get_raw_project(project_id)
        if project.get("kind") != "abnormal_sound":
            raise HTTPException(
                status_code=400,
                detail="Project is not an Abnormal Sound project.",
            )
        if project.get("training", {}).get("state") != "trained":
            raise HTTPException(status_code=409, detail="Train the detector first.")
        payload = await _read_upload(file)
        normalized_runtime = _normalized_runtime_name(runtime_name)
        artifact = _artifact_for_runtime(project, normalized_runtime)
        report = project["training"].get("report") or {}
        settings = {**project.get("settings", {}), **(report.get("settings") or {})}
        return score_wav_bytes(
            project_dir=store.project_dir(project_id),
            artifact_name=artifact,
            runtime_name=normalized_runtime,
            payload=payload,
            settings=settings,
            predict=runtime.predict,
        )

    @app.post("/api/projects/{project_id}/predict/known-sound")
    async def predict_known_sound(
        project_id: str,
        file: UploadFile = File(...),
        runtime_name: str = Form("keras"),
    ) -> dict[str, Any]:
        project = store.get_raw_project(project_id)
        if project.get("kind") != "known_sound":
            raise HTTPException(
                status_code=400, detail="Project is not a Known Sound project."
            )
        if project.get("training", {}).get("state") != "trained":
            raise HTTPException(status_code=409, detail="Train the model first.")
        payload = await _read_upload(file)
        from .known_sound_pipeline import score_wav_bytes as score_known_sound

        return score_known_sound(
            store, project_id, payload, _normalized_runtime_name(runtime_name)
        )

    @app.post("/api/projects/{project_id}/audio-sanity")
    async def audio_sanity_check(
        project_id: str,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Name what the official YAMNet tagger hears in a just-recorded clip.

        Informational only: it never touches training, thresholds or stored metadata. Its
        job is to catch "the microphone did not record what you thought it did" while the
        user is still standing at the mic. A missing class map disables only this feature.
        """
        project = store.get_raw_project(project_id)
        if not is_audio_like(project.get("kind")):
            raise HTTPException(
                status_code=400, detail="This check is only available for audio projects."
            )
        payload = await _read_upload(file)
        from .known_sound_pipeline import describe_waveform
        from .yamnet_model import YAMNET_ASSET_VERSION, YAMNET_SAMPLE_RATE, YamnetClassMapError

        try:
            _, signal = read_wav_bytes(payload, YAMNET_SAMPLE_RATE)
            top = describe_waveform(signal, top_k=3)
        except YamnetClassMapError as exc:
            return {"available": False, "reason": str(exc), "top": []}
        return {"available": True, "top": top, "asset_version": YAMNET_ASSET_VERSION}

    @app.post("/api/projects/{project_id}/export-model")
    def start_export_model(project_id: str, request: ExportRequest) -> dict[str, Any]:
        ensure_project_idle(project_id)
        project = store.get_raw_project(project_id)
        selected = [item.strip().lower() for item in request.formats if item.strip()]
        if not selected:
            raise HTTPException(status_code=400, detail="Select at least one model format.")
        filename = safe_identifier(project["name"]) + "_TFLite.zip"

        def task(progress):  # noqa: ANN001
            fd, temp_name = tempfile.mkstemp(
                prefix="tm_local_model_", suffix=".zip", dir=TEMP_ROOT
            )
            os.close(fd)
            output = Path(temp_name)
            try:
                result = create_model_export(
                    store=store,
                    project_id=project_id,
                    selected=selected,
                    include_c_header=request.include_c_header,
                    output_zip=output,
                    progress=progress,
                )
                runtime.clear_project(store.project_dir(project_id))
                token = uuid.uuid4().hex
                with export_files_lock:
                    export_files[token] = (output, filename)
                return {
                    "download_url": f"/api/exports/{token}",
                    "filename": filename,
                    "formats": result["formats"],
                    "calibration_samples": result["calibration_samples"],
                    "project": result["project"],
                }
            except Exception:
                output.unlink(missing_ok=True)
                raise

        try:
            return jobs.submit("export", project_id, task)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=PROJECT_BUSY_MESSAGE) from exc

    @app.get("/api/mcu/status")
    def mcu_status() -> dict[str, Any]:
        """Probe the toolchain live on every call.

        Deliberately not the install-time runtime_config snapshot: the student may install the
        Arm toolchain (or unplug it) long after 01_INSTALL.bat ran, and the deploy button's
        enabled/disabled state has to follow the machine, not the log.
        """
        from .mcu import toolchain
        from .mcu.boards import load_boards

        info: dict[str, Any] = dict(toolchain.probe_all())
        try:
            boards = [
                {
                    "name": board.name,
                    "label": board.label,
                    "has_lcd": board.has_lcd,
                    "flash_method": board.flash_method,
                    "flash_methods": list(board.flash_methods),
                    "known_sound_max_depth": board.known_sound_max_depth,
                }
                for board in load_boards().values()
            ]
        except (DeployError, OSError, ValueError, KeyError, TypeError):
            # No mcu_toolkit yet, or a truncated / hand-edited boards.json: report zero boards
            # rather than failing the whole status call. This endpoint exists to explain why
            # deploying is unavailable, so it has to survive the very situations it reports.
            boards = []
        info["boards"] = boards
        # Two filters, in this order: the kind registry says the kind *has* a firmware
        # application, and project_builder says that application's app module actually exists
        # in this build. A kind whose apps/<name>.py is not written yet must not offer the
        # student a button that spends minutes on an export + Vela before refusing; it becomes
        # deployable again by itself once the module lands. `_currently_deployable_kind_names()`
        # checks one kind at a time (not via a single supported_applications() call) so a future
        # app module that raises something other than DeployError only takes that one kind
        # down, not the whole probe -- and `_validate_deploy_request()`'s refusal message below
        # shares this exact same derivation, so the two can never disagree.
        info["deployable_kinds"] = _currently_deployable_kind_names()
        return info

    def _validate_deploy_request(
        project_id: str, board: str, *, require_trained: bool = False
    ) -> None:
        """Reject a deploy/download/flash the board or kind registry cannot serve.

        Raises ProjectNotFound (404), HTTPException 400 for a desktop-only kind or an
        untrained project, or DeployError (400, mapped above) for an unknown board.

        `require_trained` is for starting a build only. run_deploy() refreshes the Strict INT8
        export on its way through, and export_service raises the English "Train the model before
        exporting it." deep inside the job; checking here turns that into a Chinese 400 before
        the student watches a spinner. The download and flash endpoints leave it off: they serve
        firmware that already exists, and its existence is the freshness check (invalidating
        training deletes models/, firmware included).
        """
        from .config import is_mcu_deployable, mcu_application
        from .mcu.boards import board_for
        from .mcu.project_builder import app_module

        project = store.get_raw_project(project_id)
        kind = project.get("kind")
        if not is_mcu_deployable(kind):
            supported = " / ".join(_currently_deployable_kind_names())
            raise HTTPException(
                status_code=400,
                detail=f"{kind} 專案不支援部署到開發板（只有 {supported} 可以）。",
            )
        # is_mcu_deployable() only says the kind has an application id; whether this build
        # can assemble that application is what app_module() knows. Without this check a deploy
        # of a kind whose firmware app is not written yet runs a full Strict INT8 export and a
        # Vela compile before raising the very same message from project_builder.assemble().
        app_module(mcu_application(kind))  # DeployError -> 400 「此版本尚未支援 … 韌體應用」
        if require_trained and (project.get("training") or {}).get("state") != "trained":
            raise HTTPException(status_code=400, detail="請先完成訓練，再部署到開發板。")
        board_for(board)  # unknown board -> DeployError -> 400, naming the supported boards

    def _register_deploy_download(zip_path: Path, filename: str) -> str:
        """Hand /api/exports/{token} a throwaway copy of the firmware ZIP.

        That endpoint unlinks whatever it serves, so the build output under
        models/mcu/<board>/ must never be registered directly -- it has to survive every
        download so the student can fetch it again without rebuilding.
        """
        fd, temp_name = tempfile.mkstemp(prefix="tm_local_firmware_", suffix=".zip", dir=TEMP_ROOT)
        os.close(fd)
        output = Path(temp_name)
        try:
            shutil.copyfile(zip_path, output)
        except OSError:
            output.unlink(missing_ok=True)
            raise
        token = uuid.uuid4().hex
        with export_files_lock:
            export_files[token] = (output, filename)
        return f"/api/exports/{token}"

    @app.post("/api/projects/{project_id}/deploy-mcu")
    def start_deploy_mcu(project_id: str, request: DeployRequest) -> dict[str, Any]:
        from .mcu import toolchain

        board = request.board.strip()
        _validate_deploy_request(project_id, board, require_trained=True)
        ensure_project_idle(project_id)
        reason = mcu_unavailable_reason(toolchain.probe_all())
        if reason:
            raise HTTPException(status_code=400, detail=reason)

        def task(progress):
            outcome = run_mcu_deploy(store, project_id, board, progress)
            # The deploy refreshes the Strict INT8 export on the way through, so drop the
            # cached Preview models exactly as the export job does.
            runtime.clear_project(store.project_dir(project_id))
            filename = outcome.zip_path.name
            return {
                "download_url": _register_deploy_download(outcome.zip_path, filename),
                "filename": filename,
                "board": outcome.board,
                "report": outcome.report,
                "project": store.get_project(project_id),
            }

        try:
            return jobs.submit("deploy", project_id, task)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=PROJECT_BUSY_MESSAGE) from exc

    @app.post("/api/projects/{project_id}/deploy-mcu/download")
    def download_last_deploy(project_id: str, request: DeployRequest) -> dict[str, Any]:
        from .mcu.reports import deploy_dir

        board = request.board.strip()
        _validate_deploy_request(project_id, board)
        target = deploy_dir(store.project_dir(project_id) / "models", board)
        builds = sorted(target.glob("*_firmware.zip")) if target.is_dir() else []
        if not builds:
            raise HTTPException(
                status_code=400,
                detail=f"{board} 還沒有建置成功的韌體，請先執行「部署到開發板」。",
            )
        latest = max(builds, key=lambda item: item.stat().st_mtime)
        return {
            "download_url": _register_deploy_download(latest, latest.name),
            "filename": latest.name,
        }

    @app.post("/api/projects/{project_id}/deploy-mcu/flash")
    def flash_mcu(project_id: str, request: FlashRequest) -> dict[str, Any]:
        """Flash the last successful build for `board` -- synchronous, no job.

        Order of checks: project/kind/board validity -> the build actually exists on disk ->
        `firmware.bin` still matches the sha256 recorded at build time (guards against a
        hand-edited or partially-copied file) -> the requested method matches how this board is
        actually flashed -> the method-specific preflight (an MSC drive, or the Nu-Link tool).
        """
        from .mcu import flash
        from .mcu.boards import board_for
        from .mcu.deploy_service import flash_instructions
        from .mcu.reports import deploy_dir
        from .mcu.toolchain import probe_nulink, probe_numicro_pack, probe_pyocd

        board_name = request.board.strip()
        _validate_deploy_request(project_id, board_name)
        ensure_project_idle(project_id)
        board = board_for(board_name)
        target = deploy_dir(store.project_dir(project_id) / "models", board.name)
        report_path = target / "deploy_report.json"
        bin_path = target / "firmware.bin"
        if not report_path.is_file() or not bin_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"{board.label} 還沒有建置成功的韌體，請先執行「部署到開發板」。",
            )
        try:
            report = read_json(report_path)
        except (OSError, ValueError):
            report = None
        # Covers both a corrupt/hand-edited deploy_report.json (not JSON, or not a JSON object --
        # read_json's ValueError, or the isinstance check below) and a firmware.bin that no
        # longer matches what was recorded at build time; the student needs the same fix either
        # way, so one message rather than two saying almost the same thing.
        if not isinstance(report, dict):
            raise HTTPException(status_code=400, detail="部署紀錄損毀或不符，請重新執行部署。")
        # Both preflight failures below hand the student the flashing steps, and those steps end
        # with what they should then SEE -- which is the camera window / LCD label only for an
        # image build. This report is the one that named this firmware's kind, so use it: an
        # audio build must point at the COM port, not at a window that will stay black.
        flash_kind = str(report.get("kind") or "image")
        expected_sha = (report.get("bin") or {}).get("sha256")
        if sha256_file(bin_path) != expected_sha:
            raise HTTPException(status_code=400, detail="部署紀錄損毀或不符，請重新執行部署。")
        if request.method not in board.flash_methods:
            allowed = "、".join(board.flash_methods)
            raise HTTPException(
                status_code=400,
                detail=f"{board.label} 支援的燒錄方式是 {allowed}，不是 {request.method}。",
            )

        if request.method == "msc":
            if request.drive is not None:
                drive = request.drive.strip()
                if not _DRIVE_LETTER_RE.fullmatch(drive):
                    raise HTTPException(
                        status_code=400,
                        detail=f"磁碟機代號格式錯誤：{request.drive!r}；需為 X: 或 X:\\ 這種格式。",
                    )
                drives = [drive]
            else:
                drives = flash.find_msc_drives()
            if not drives:
                raise HTTPException(
                    status_code=400,
                    detail=flash_instructions(board, flash_kind, request.method),
                )
            result = flash.copy_to_msc(bin_path, drives[0])
        elif request.method == "nulink":
            nulink = probe_nulink()
            if not nulink["available"]:
                raise HTTPException(
                    status_code=400,
                    detail=flash_instructions(board, flash_kind, request.method),
                )
            result = flash.nulink_program(bin_path, Path(nulink["path"]))
        elif request.method == "pyocd":
            pyocd = probe_pyocd()
            pack = probe_numicro_pack()
            if not pyocd["available"] or not pack["available"]:
                raise HTTPException(
                    status_code=400,
                    detail=flash_instructions(board, flash_kind, request.method),
                )
            result = flash.pyocd_program(bin_path, Path(pyocd["path"]), Path(pack["path"]))
        else:
            raise HTTPException(
                status_code=400, detail=f"未知的燒錄方式：{request.method}"
            )

        if not result["ok"]:
            raise HTTPException(status_code=400, detail=result["message"])
        return result

    @app.get("/api/exports/{token}")
    def download_export(token: str, background_tasks: BackgroundTasks):
        with export_files_lock:
            item = export_files.pop(token, None)
        if not item:
            raise HTTPException(status_code=404, detail="Export file not found or already downloaded.")
        output, filename = item
        if not output.is_file():
            raise HTTPException(status_code=404, detail="Export file is no longer available.")
        background_tasks.add_task(output.unlink, missing_ok=True)
        return FileResponse(output, media_type="application/zip", filename=filename)

    @app.get("/api/projects/{project_id}/export-project")
    def export_project(project_id: str, background_tasks: BackgroundTasks):
        ensure_project_idle(project_id)
        project = store.get_raw_project(project_id)
        filename = safe_identifier(project["name"]) + "_Project.zip"
        fd, temp_name = tempfile.mkstemp(prefix="tm_local_project_", suffix=".zip", dir=TEMP_ROOT)
        os.close(fd)
        output = Path(temp_name)
        store.export_project_archive(project_id, output)
        background_tasks.add_task(output.unlink, missing_ok=True)
        return FileResponse(output, media_type="application/zip", filename=filename)

    # Project samples are intentionally available only on the local server.
    app.mount("/workspace", StaticFiles(directory=str(store.projects_root.parent)), name="workspace")
    app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="web")
    return app
