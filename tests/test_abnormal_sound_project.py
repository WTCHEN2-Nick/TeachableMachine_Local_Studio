from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tm_local.app import create_app
from tm_local.export_service import ensure_export_artifacts
from tm_local.jobs import JobManager
from tm_local.model_runtime import ModelRuntime
from tm_local.project_store import ProjectError, ProjectStore
from tm_local.tflite_export import ExportError
from tm_local.training_common import TrainingError
from tm_local.yamnet_anomaly_pipeline import train_yamnet_abnormal_sound_project


def test_abnormal_project_roles_sessions_and_locked_normal(
    store: ProjectStore,
    wav_bytes: bytes,
) -> None:
    project = store.create_project("abnormal_sound", "Machine")
    assert project["kind"] == "abnormal_sound"
    assert [item["role"] for item in project["classes"]] == [
        "normal_train",
        "anomaly_eval",
    ]
    normal = project["classes"][0]
    assert normal["locked"] is True
    samples = store.add_audio_bytes(
        project["id"],
        normal["id"],
        wav_bytes,
        source_name="normal_session.wav",
        recording_session_id="session-a",
        device_metadata={"label": "GestureAI DMIC Mic 0416", "sampleRate": 16000},
    )
    assert len(samples) == 2
    assert {sample["recording_session_id"] for sample in samples} == {"session-a"}
    assert samples[0]["source_start_sample"] == 0
    assert samples[1]["source_start_sample"] == 16000
    assert samples[0]["device_metadata"]["label"] == "GestureAI DMIC Mic 0416"
    current = store.get_project(project["id"])
    assert current["abnormal_readiness"]["normal_sessions"] == 1
    assert current["abnormal_readiness"]["normal_seconds"] == 2.0
    assert current["abnormal_readiness"]["minimum_calibration_sessions"] == 6
    assert current["abnormal_readiness"]["calibration_candidate"] is False
    with pytest.raises(ProjectError, match="locked"):
        store.update_class(project["id"], normal["id"], {"name": "Not normal"})
    with pytest.raises(ProjectError, match="locked"):
        store.delete_class(project["id"], normal["id"])
    added = store.add_class(project["id"], "Bearing Fault Evaluation")
    assert added["classes"][-1]["role"] == "anomaly_eval"


def test_abnormal_archive_import_invalidates_bound_scorer(
    store: ProjectStore,
    wav_bytes: bytes,
    tmp_path: Path,
) -> None:
    project = store.create_project("abnormal_sound", "Archive")
    normal_id = project["classes"][0]["id"]
    store.add_audio_bytes(project["id"], normal_id, wav_bytes)
    models = store.project_dir(project["id"]) / "models"
    (models / "abnormal_sound_yamnet.keras").write_bytes(b"fake keras")
    (models / "yamnet_scorer.json").write_text("{}", encoding="utf-8")
    (models / "yamnet_frontend.json").write_text("{}", encoding="utf-8")
    store.set_training_result(
        project["id"],
        report={"detector_backend": "yamnet_embedding"},
        artifacts={
            "keras": "abnormal_sound_yamnet.keras",
            "yamnet_scorer": "yamnet_scorer.json",
            "yamnet_frontend": "yamnet_frontend.json",
        },
    )
    archive = tmp_path / "abnormal.zip"
    store.export_project_archive(project["id"], archive)
    imported = store.import_project_archive(archive)
    assert imported["kind"] == "abnormal_sound"
    assert imported["training"]["state"] == "untrained"
    assert imported["total_samples"] == 2
    assert list((store.project_dir(imported["id"]) / "models").iterdir()) == []


def test_abnormal_api_accepts_mic_metadata_and_has_dedicated_preview_route(
    store: ProjectStore,
    wav_bytes: bytes,
    monkeypatch,
) -> None:  # noqa: ANN001
    from tm_local import app as app_module

    captured: dict[str, object] = {}

    def fake_score_wav_bytes(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "runtime": kwargs["runtime_name"],
            "detector_kind": "open_set_anomaly",
            "verdict": "normal",
            "anomaly_ratio": 0.4,
            "components": {"embedding": {"ratio": 0.4}, "level": {"ratio": 0.2}},
        }

    monkeypatch.setattr(app_module, "score_wav_bytes", fake_score_wav_bytes)
    with TestClient(
        create_app(store=store, jobs=JobManager(max_workers=1), runtime=ModelRuntime())
    ) as client:
        created = client.post(
            "/api/projects", json={"kind": "abnormal_sound", "name": "DMIC"}
        )
        assert created.status_code == 200
        project = created.json()
        normal_id = project["classes"][0]["id"]
        upload = client.post(
            f"/api/projects/{project['id']}/classes/{normal_id}/audio",
            files=[("files", ("normal.wav", wav_bytes, "audio/wav"))],
            data={
                "recording_session_id": "browser-session",
                "device_label": "GestureAI DMIC Mic 0416",
                "device_id": "local-device",
                "device_settings": json.dumps({"sampleRate": 48000, "channelCount": 1}),
            },
        )
        assert upload.status_code == 200
        raw = store.get_raw_project(project["id"])
        first_sample = next(iter(raw["samples"].values()))
        assert first_sample["recording_session_id"] == "browser-session"
        assert first_sample["device_metadata"]["track_settings"]["sampleRate"] == 48000

        models = store.project_dir(project["id"]) / "models"
        (models / "abnormal_sound_yamnet.keras").write_bytes(b"fake")
        (models / "yamnet_scorer.json").write_text("{}", encoding="utf-8")
        store.set_training_result(
            project["id"],
            report={"settings": {"detector_backend": "yamnet_embedding"}},
            artifacts={
                "keras": "abnormal_sound_yamnet.keras",
                "yamnet_scorer": "yamnet_scorer.json",
            },
        )
        response = client.post(
            f"/api/projects/{project['id']}/predict/abnormal-sound",
            files={"file": ("preview.wav", wav_bytes, "audio/wav")},
            data={"runtime_name": "keras"},
        )
        assert response.status_code == 200
        assert response.json()["verdict"] == "normal"
        assert response.json()["detector_kind"] == "open_set_anomaly"
        assert captured["artifact_name"] == "abnormal_sound_yamnet.keras"


def test_abnormal_project_archive_contains_project_json(
    store: ProjectStore,
    tmp_path: Path,
) -> None:
    project = store.create_project("abnormal_sound")
    output = tmp_path / "project.zip"
    store.export_project_archive(project["id"], output)
    with zipfile.ZipFile(io.BytesIO(output.read_bytes())) as archive:
        assert "project.json" in archive.namelist()


def test_abnormal_project_rejects_partially_wired_dense_ae_backend(
    store: ProjectStore,
) -> None:
    project = store.create_project("abnormal_sound")
    with pytest.raises(ProjectError, match="yamnet_embedding only"):
        store.update_project(
            project["id"],
            {"settings": {"detector_backend": "dense_ae"}},
        )


def test_yamnet_window_geometry_is_locked_and_train_options_must_be_persisted(
    store: ProjectStore,
) -> None:
    project = store.create_project("abnormal_sound")
    with pytest.raises(ProjectError, match="clip_seconds=1.0"):
        store.update_project(project["id"], {"settings": {"clip_seconds": 2.0}})
    with pytest.raises(ProjectError, match="hop_seconds=0.5"):
        store.update_project(project["id"], {"settings": {"hop_seconds": 0.25}})
    with pytest.raises(TrainingError, match="must first be saved"):
        train_yamnet_abnormal_sound_project(
            store,
            project["id"],
            {"sensitivity": "sensitive"},
        )
    store.update_project(project["id"], {"settings": {"sensitivity": "sensitive"}})
    with pytest.raises(TrainingError, match="Record normal audio"):
        train_yamnet_abnormal_sound_project(
            store,
            project["id"],
            {"sensitivity": "sensitive"},
        )


def test_abnormal_export_rejects_missing_frontend_metadata(store: ProjectStore) -> None:
    project = store.create_project("abnormal_sound")
    models = store.project_dir(project["id"]) / "models"
    keras_path = models / "abnormal_sound_yamnet.keras"
    keras_path.write_bytes(b"fake")
    (models / "yamnet_scorer.json").write_text("{}", encoding="utf-8")
    store.set_training_result(
        project["id"],
        report={"detector_backend": "yamnet_embedding"},
        artifacts={
            "keras": keras_path.name,
            "yamnet_scorer": "yamnet_scorer.json",
            "yamnet_frontend": "yamnet_frontend.json",
        },
    )
    with pytest.raises(ExportError, match="frontend metadata is missing"):
        ensure_export_artifacts(store, project["id"], ["keras"])
