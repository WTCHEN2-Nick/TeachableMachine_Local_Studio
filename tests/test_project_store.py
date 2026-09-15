from __future__ import annotations

import zipfile
from pathlib import Path

from tm_local.project_store import ProjectStore


def test_image_project_sample_lifecycle(store: ProjectStore, jpeg_bytes: bytes) -> None:
    project = store.create_project("image", "Images")
    class_id = project["classes"][0]["id"]
    sample = store.add_image_bytes(project["id"], class_id, jpeg_bytes, source_name="camera.jpg")
    current = store.get_project(project["id"])
    assert current["kind"] == "image"
    assert current["total_samples"] == 1
    assert current["classes"][0]["sample_count"] == 1
    assert (store.project_dir(project["id"]) / sample["relative_path"]).is_file()
    store.delete_sample(project["id"], sample["id"])
    assert store.get_project(project["id"])["total_samples"] == 0


def test_audio_project_splits_wav(store: ProjectStore, wav_bytes: bytes) -> None:
    project = store.create_project("audio", "Audio")
    class_id = project["classes"][0]["id"]
    samples = store.add_audio_bytes(project["id"], class_id, wav_bytes, source_name="tone.wav")
    assert len(samples) == 2
    current = store.get_project(project["id"])
    assert current["total_samples"] == 2
    assert all(sample["sample_rate"] == 16000 for sample in samples)
    assert all((store.project_dir(project["id"]) / sample["thumbnail_path"]).is_file() for sample in samples)


def test_project_export_and_import(store: ProjectStore, jpeg_bytes: bytes, tmp_path: Path) -> None:
    project = store.create_project("image", "Export Me")
    store.add_image_bytes(project["id"], project["classes"][0]["id"], jpeg_bytes)
    archive = tmp_path / "project.zip"
    store.export_project_archive(project["id"], archive)
    with zipfile.ZipFile(archive) as zf:
        assert "project.json" in zf.namelist()
    imported = store.import_project_archive(archive)
    assert imported["id"] != project["id"]
    assert imported["name"].endswith("(Imported)")
    assert imported["total_samples"] == 1


def test_recover_interrupted_training_when_keras_model_exists(tmp_path: Path) -> None:
    projects = tmp_path / "workspace" / "projects"
    temp = tmp_path / "workspace" / "tmp"
    first = ProjectStore(projects, temp)
    project = first.create_project("image", "Recover")
    first.set_training_started(project["id"])
    models = first.project_dir(project["id"]) / "models"
    (models / "image_classifier.keras").write_bytes(b"fake-complete-keras-model")

    second = ProjectStore(projects, temp)
    recovered = second.get_project(project["id"])
    assert second.recovered_interrupted_jobs == 1
    assert recovered["training"]["state"] == "trained"
    assert recovered["training"]["artifacts"]["keras"] == "image_classifier.keras"
    assert recovered["training"]["report"]["conversion"]["state"] == "pending"
    assert (models / "training_report.json").is_file()


def test_recover_interrupted_training_without_model_marks_failed(tmp_path: Path) -> None:
    projects = tmp_path / "workspace" / "projects"
    temp = tmp_path / "workspace" / "tmp"
    first = ProjectStore(projects, temp)
    project = first.create_project("audio", "Interrupted")
    first.set_training_started(project["id"])

    second = ProjectStore(projects, temp)
    recovered = second.get_project(project["id"])
    assert second.recovered_interrupted_jobs == 0
    assert recovered["training"]["state"] == "failed"
    assert "samples are safe" in recovered["training"]["report"]["error"]
