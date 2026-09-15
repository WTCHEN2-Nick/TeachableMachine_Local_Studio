from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tm_local.app import PROJECT_BUSY_MESSAGE, create_app
from tm_local.jobs import JobManager
from tm_local.model_runtime import ModelRuntime
from tm_local.project_store import ProjectStore


def make_client(store: ProjectStore) -> TestClient:
    return TestClient(create_app(store=store, jobs=JobManager(max_workers=1), runtime=ModelRuntime()))


class RecordingJobs(JobManager):
    """JobManager that records submissions instead of running them.

    POST /train persists the Advanced-panel options synchronously and only then queues the
    job; swallowing the job keeps these tests off TensorFlow and, more importantly, stops
    the worker's set_training_started() from racing the assertions about training state.
    """

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.submitted: list[tuple[str, str | None]] = []

    def submit(self, job_type: str, project_id: str | None, task: Any) -> dict[str, Any]:
        self.submitted.append((job_type, project_id))
        return {"id": "test-job", "state": "queued", "progress": 0.0, "message": "queued"}


class InlineJobs(RecordingJobs):
    """JobManager that runs the submitted task immediately, in the request thread.

    RecordingJobs swallows the task, which is what the persistence tests want. This one has
    to actually reach train_project() to observe the arguments the endpoint handed it.
    """

    def submit(self, job_type: str, project_id: str | None, task: Any) -> dict[str, Any]:
        super().submit(job_type, project_id, task)
        task(lambda *args, **kwargs: None)
        return {"id": "test-job", "state": "completed", "progress": 1.0, "message": "done"}


def test_frontend_assets_are_not_cached(store: ProjectStore) -> None:
    with make_client(store) as client:
        for path in ("/", "/index.html", "/app.js", "/style.css"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"


def test_health_and_project_crud(store: ProjectStore) -> None:
    with make_client(store) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        created = client.post("/api/projects", json={"kind": "image", "name": "API Image"})
        assert created.status_code == 200
        project = created.json()
        fetched = client.get(f"/api/projects/{project['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["name"] == "API Image"
        renamed = client.patch(f"/api/projects/{project['id']}", json={"name": "Renamed"})
        assert renamed.json()["name"] == "Renamed"
        deleted = client.delete(f"/api/projects/{project['id']}")
        assert deleted.json() == {"ok": True}


def test_image_and_audio_upload_api(store: ProjectStore, jpeg_bytes: bytes, wav_bytes: bytes) -> None:
    with make_client(store) as client:
        image = client.post("/api/projects", json={"kind": "image"}).json()
        image_class = image["classes"][0]["id"]
        response = client.post(
            f"/api/projects/{image['id']}/classes/{image_class}/images",
            files=[("files", ("sample.jpg", jpeg_bytes, "image/jpeg"))],
        )
        assert response.status_code == 200
        assert response.json()["added"] == 1

        audio = client.post("/api/projects", json={"kind": "audio"}).json()
        audio_class = audio["classes"][0]["id"]
        response = client.post(
            f"/api/projects/{audio['id']}/classes/{audio_class}/audio",
            files=[("files", ("tone.wav", wav_bytes, "audio/wav"))],
            data={"overlap": "0"},
        )
        assert response.status_code == 200
        assert response.json()["added"] == 2


def test_async_export_job_and_one_time_download(store: ProjectStore, monkeypatch) -> None:  # noqa: ANN001
    import time
    import zipfile
    from pathlib import Path

    from tm_local import app as app_module

    project = store.create_project("image", "Export API")
    models = store.project_dir(project["id"]) / "models"
    keras_path = models / "image_classifier.keras"
    keras_path.write_bytes(b"fake keras")
    store.set_training_result(
        project["id"],
        report={
            "settings": {"image_size": 224},
            "dataset": {"calibration_samples": 0},
            "evaluation": {},
            "conversion": {"state": "pending", "models": {}},
        },
        artifacts={"keras": keras_path.name},
    )

    def fake_export(*, store, project_id, selected, include_c_header, output_zip, progress):  # noqa: ANN001
        progress(0.2, "Converting…")
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr("model.tflite", b"TFL3")
        progress(1.0, "Ready")
        return {
            "project": store.get_project(project_id),
            "formats": list(selected),
            "calibration_samples": 2,
        }

    monkeypatch.setattr(app_module, "create_model_export", fake_export)
    with make_client(store) as client:
        started = client.post(
            f"/api/projects/{project['id']}/export-model",
            json={"formats": ["int8"], "include_c_header": True},
        )
        assert started.status_code == 200
        job_id = started.json()["id"]
        job = None
        for _ in range(100):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["state"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job and job["state"] == "completed", job
        url = job["result"]["download_url"]
        downloaded = client.get(url)
        assert downloaded.status_code == 200
        assert downloaded.content.startswith(b"PK")
        assert client.get(url).status_code == 404


# (kind, options POST /train sends, the settings keys that must come back changed).
# One entry per project kind, because /train persisting for EVERY kind is the whole point of
# this endpoint change: before it, only known_sound wrote its options back. abnormal_sound is
# the case worth spelling out -- `sensitivity` is in ABNORMAL_SOUND_DEFAULTS, so /train now
# invalidates a trained abnormal_sound model where it previously left it alone.
TRAIN_OPTION_CASES = [
    (
        "image",
        {
            "epochs": 12,
            "augmentation_level": "strong",
            "dropout": 0.35,
            "fine_tune_blocks": 2,
            "image_size": 160,
            "deployment_target": "NuGestureAI-M55M1",
        },
    ),
    (
        "audio",
        {
            "epochs": 12,
            "augmentation_level": "strong",
            "spec_augment": True,
            "session_disjoint_validation": False,
            "background_weight": 2.0,
            "dropout": 0.35,
            "detection_threshold": 0.7,
            "deployment_target": "NuMaker-M55M1",
        },
    ),
    ("abnormal_sound", {"sensitivity": "low_false_alarm"}),
    (
        "known_sound",
        {
            "epochs": 77,
            "detection_threshold": 0.7,
            "head_dropout": 0.25,
            "waveform_augment_level": "light",
            "background_weight": 2.0,
            "encoder_depth": 10,
            "deployment_target": "NuMaker-M55M1",
        },
    ),
]


@pytest.mark.parametrize("kind,options", TRAIN_OPTION_CASES, ids=[c[0] for c in TRAIN_OPTION_CASES])
def test_train_persists_advanced_options_for_every_kind(
    store: ProjectStore, kind: str, options: dict[str, Any]
) -> None:
    """POST /train must store the Advanced-panel options before the job starts.

    Only known_sound used to do this, so an image student's augmentation_level /
    fine_tune_blocks / deployment_target were read by the pipeline for that one run and then
    forgotten -- Export and the firmware build saw the old values.
    """
    jobs = RecordingJobs()
    with TestClient(create_app(store=store, jobs=jobs, runtime=ModelRuntime())) as client:
        project = client.post("/api/projects", json={"kind": kind, "name": "Tunables"}).json()
        project_id = project["id"]
        # Every case has to actually change something, or the endpoint is allowed to skip
        # the write and the test would pass without exercising anything.
        assert any(project["settings"].get(key) != value for key, value in options.items())
        started = client.post(f"/api/projects/{project_id}/train", json={"options": options})
        assert started.status_code == 200, started.text
        assert jobs.submitted == [("train", project_id)]
        stored = client.get(f"/api/projects/{project_id}").json()["settings"]
        for key, value in options.items():
            assert stored[key] == value, key

        # Pressing Train again without touching the panel must not invalidate the model.
        models = store.project_dir(project_id) / "models"
        models.mkdir(parents=True, exist_ok=True)
        keras_path = models / "model.keras"
        keras_path.write_bytes(b"fake keras")
        store.set_training_result(
            project_id,
            report={"settings": dict(options), "evaluation": {}},
            artifacts={"keras": keras_path.name},
        )
        assert store.get_project(project_id)["training"]["state"] == "trained"
        again = client.post(f"/api/projects/{project_id}/train", json={"options": options})
        assert again.status_code == 200, again.text
        after = client.get(f"/api/projects/{project_id}").json()
        assert after["training"]["state"] == "trained"
        assert keras_path.is_file()


def test_train_uses_the_validated_settings_not_the_raw_request_body(
    store: ProjectStore, monkeypatch
) -> None:
    """The pipeline must receive the settings /train just validated, not the client body.

    `_require_bool` accepts the STRING "false" and stores a real `False`, so the project
    file, Export metadata and firmware contract all recorded the option as OFF -- while
    image_pipeline/audio_pipeline merge `{**project["settings"], **options}`, so the raw
    body won and the model trained with `bool("false")`, i.e. ON. No error at any layer.
    Not reachable from web/app.js (it sends real JSON types); reachable from the HTTP API.
    """
    from tm_local import app as app_module

    seen: dict[str, Any] = {}

    def fake_train_project(store_arg, project_id, kind, options, progress):
        seen["kind"] = kind
        seen["options"] = dict(options or {})
        models = store_arg.project_dir(project_id) / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "model.keras").write_bytes(b"fake keras")
        return {
            "report": {"settings": dict(options or {}), "evaluation": {}},
            "artifacts": {"keras": "model.keras"},
        }

    monkeypatch.setattr(app_module, "train_project", fake_train_project)
    jobs = InlineJobs()
    with TestClient(create_app(store=store, jobs=jobs, runtime=ModelRuntime())) as client:
        project = client.post("/api/projects", json={"kind": "audio"}).json()
        project_id = project["id"]
        started = client.post(
            f"/api/projects/{project_id}/train",
            json={"options": {"early_stopping": "false", "spec_augment": "false"}},
        )
        assert started.status_code == 200, started.text
        assert jobs.submitted == [("train", project_id)]
        stored = client.get(f"/api/projects/{project_id}").json()["settings"]
        assert stored["early_stopping"] is False
        assert stored["spec_augment"] is False
        # The whole point: real booleans, not the strings that were posted.
        assert seen["options"]["early_stopping"] is False
        assert seen["options"]["spec_augment"] is False
        # ...and the complete persisted dict, so the pipelines' `{**settings, **options}`
        # merge cannot reintroduce a value the validator never saw.
        assert seen["options"] == stored


def test_train_on_a_busy_project_answers_the_one_busy_message(store: ProjectStore) -> None:
    """A Train click on a project that is already working must always be the same 409.

    /train persists the Advanced options before it queues the job, and update_project() ->
    _ensure_editable() raises a ProjectError (HTTP 400) on a project that is training. So
    without an idle check at the top of train(), the very same click reported 409 when the
    student had not touched the panel and 400 when they had.
    """

    class BusyJobs(RecordingJobs):
        def active_for_project(self, project_id: str) -> dict[str, Any] | None:
            return {"id": "busy", "state": "running", "project_id": project_id}

    jobs = BusyJobs()
    with TestClient(create_app(store=store, jobs=jobs, runtime=ModelRuntime())) as client:
        project = client.post("/api/projects", json={"kind": "image"}).json()
        # Both halves of the real busy state: a job the manager owns, and the training flag
        # _ensure_editable() looks at. The 400 only appeared once both were true.
        store.set_training_started(project["id"])
        for options in ({}, {"epochs": 12, "augmentation_level": "strong"}):
            response = client.post(
                f"/api/projects/{project['id']}/train", json={"options": options}
            )
            assert response.status_code == 409, (options, response.text)
            assert response.json()["detail"] == PROJECT_BUSY_MESSAGE
        assert jobs.submitted == []
        # Nothing was written: a refused click must not invalidate anything either.
        assert store.get_project(project["id"])["settings"]["augmentation_level"] == "medium"


def test_train_rejects_an_out_of_range_option_in_traditional_chinese(store: ProjectStore) -> None:
    jobs = RecordingJobs()
    with TestClient(create_app(store=store, jobs=jobs, runtime=ModelRuntime())) as client:
        project = client.post("/api/projects", json={"kind": "image"}).json()
        response = client.post(
            f"/api/projects/{project['id']}/train", json={"options": {"dropout": 0.9}}
        )
        assert response.status_code == 400
        assert "dropout" in response.json()["detail"]
        assert jobs.submitted == []


def test_patch_settings_rejects_invalid_tunables_with_a_chinese_message(
    store: ProjectStore,
) -> None:
    """The panel PATCHes settings before it starts training, so this is the message the
    student actually sees in the toast."""
    with make_client(store) as client:
        image = client.post("/api/projects", json={"kind": "image"}).json()
        out_of_range = client.patch(
            f"/api/projects/{image['id']}", json={"settings": {"dropout": 0.9}}
        )
        assert out_of_range.status_code == 400
        detail = out_of_range.json()["detail"]
        assert "dropout" in detail and "必須介於 0.0 與 0.6 之間" in detail

        bad_level = client.patch(
            f"/api/projects/{image['id']}", json={"settings": {"augmentation_level": "extreme"}}
        )
        assert bad_level.status_code == 400
        assert "augmentation_level" in bad_level.json()["detail"]

        # A board target narrows image_size to the five sizes the camera firmware supports.
        board = client.patch(
            f"/api/projects/{image['id']}",
            json={"settings": {"deployment_target": "NuMaker-M55M1", "image_size": 320}},
        )
        assert board.status_code == 400
        assert "部署到開發板時 image_size 只能是" in board.json()["detail"]
        # The refused PATCH must leave the project untouched.
        assert client.get(f"/api/projects/{image['id']}").json()["settings"]["image_size"] == 224

        known = client.post("/api/projects", json={"kind": "known_sound"}).json()
        deep = client.patch(
            f"/api/projects/{known['id']}",
            json={"settings": {"deployment_target": "NuMaker-M55M1", "encoder_depth": 14}},
        )
        assert deep.status_code == 400
        assert "encoder_depth 上限是 12" in deep.json()["detail"]

        class_id = known["classes"][0]["id"]
        bad_threshold = client.patch(
            f"/api/projects/{known['id']}", json={"settings": {"class_thresholds": {class_id: 1.5}}}
        )
        assert bad_threshold.status_code == 400
        assert "class_thresholds" in bad_threshold.json()["detail"]
