from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tm_local.mcu import boards, contract
from tm_local.mcu.errors import DeployError
from tm_local.project_store import ProjectStore

FAKE_INT8_BYTES = b"fake int8"
FAKE_INT8_SHA256 = hashlib.sha256(FAKE_INT8_BYTES).hexdigest()


def _int8_report(shape_in, shape_out, sha256=FAKE_INT8_SHA256, name_in="serving_default_input:0"):
    return {
        "models": {
            "int8": {
                "path": "m_int8.tflite",
                "sha256": sha256,
                "inputs": [
                    {
                        "name": name_in,
                        "shape": list(shape_in),
                        "dtype": "int8",
                        "scale": 1.0,
                        "zero_point": -128,
                    }
                ],
                "outputs": [
                    {
                        "name": "out",
                        "shape": list(shape_out),
                        "dtype": "int8",
                        "scale": 0.00390625,
                        "zero_point": -128,
                    }
                ],
                "strict_full_integer": True,
                "integer_input_output": True,
            }
        }
    }


def _trained_image_project(store: ProjectStore, *, image_size=224, classes=("cat", "dog")) -> str:
    # ProjectStore.create_project(kind, name) takes kind FIRST -- and an "image" project
    # is seeded with two default classes ("Class 1", "Class 2"), so we rename those in
    # place rather than add_class()-ing new ones on top (which would leave 4 classes, not
    # the 2 the test expects). Renaming via update_class() also does not invalidate
    # training, so it is safe to do before writing the fake model files below.
    project = store.create_project("image", "Img")
    raw = store.get_raw_project(project["id"])
    existing = raw["classes"]
    for class_item, name in zip(existing, classes):
        store.update_class(project["id"], class_item["id"], {"name": name})
    for name in classes[len(existing) :]:
        store.add_class(project["id"], name)

    models = store.project_dir(project["id"]) / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "image_classifier.keras").write_bytes(b"fake keras")
    (models / "image_classifier_int8.tflite").write_bytes(FAKE_INT8_BYTES)
    report = {
        "project_kind": "image",
        "settings": {"image_size": image_size, "backbone_used": "mobilenet_v2_alpha_0.35"},
        "conversion": {
            "state": "generated",
            **_int8_report((1, image_size, image_size, 3), (1, len(classes))),
        },
    }
    store.set_training_result(
        project["id"],
        report=report,
        artifacts={
            "keras": "image_classifier.keras",
            "training_report": "training_report.json",
            "int8": "image_classifier_int8.tflite",
            "conversion_report": "conversion_report.json",
        },
    )
    (models / "conversion_report.json").write_text(
        json.dumps(_int8_report((1, image_size, image_size, 3), (1, len(classes)))),
        encoding="utf-8",
    )
    return project["id"]


def test_collect_image_contract(store: ProjectStore) -> None:
    pid = _trained_image_project(store)
    c = contract.collect(store, pid, boards.board_for("NuGestureAI-M55M1"))
    assert c.kind == "image" and c.application == "imgclass"
    assert c.labels == ["cat", "dog"]
    assert c.input.shape == (1, 224, 224, 3) and c.input.dtype == "int8" and c.input.zero_point == -128
    assert c.output.shape == (1, 2)
    assert c.image_size == 224
    assert c.int8_path.name == "image_classifier_int8.tflite"
    contract.validate(c)


def test_requires_int8_export(store: ProjectStore) -> None:
    project = store.create_project("image", "Img")
    store.add_class(project["id"], "a")
    with pytest.raises(DeployError, match="INT8"):
        contract.collect(store, project["id"], boards.board_for("NuMaker-M55M1"))


def test_abnormal_sound_not_deployable(store: ProjectStore) -> None:
    project = store.create_project("abnormal_sound", "Ab")
    with pytest.raises(DeployError, match="不支援"):
        contract.collect(store, project["id"], boards.board_for("NuMaker-M55M1"))


def test_image_size_lock(store: ProjectStore) -> None:
    pid = _trained_image_project(store, image_size=200)
    c = contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))
    with pytest.raises(DeployError, match="image_size"):
        contract.validate(c)


def test_validate_rejects_label_mismatch(tmp_path: Path) -> None:
    board = boards.board_for("NuMaker-M55M1")
    c = contract.DeployContract(
        kind="image", application="imgclass", board=board, project_id="p", project_name="n",
        labels=["a", "b", "c"], models_dir=tmp_path, int8_path=tmp_path / "m.tflite", model_sha256="x",
        input=contract.TensorInfo("in", (1, 224, 224, 3), "int8", 1.0, -128),
        output=contract.TensorInfo("out", (1, 2), "int8", 0.0039, -128),
        image_size=224, audio_frontend=None, yamnet_frontend=None, encoder_depth=None,
        detection_threshold=0.5, class_thresholds=[0.5, 0.5], hop_seconds=0.5, clip_seconds=1.0,
        peak_hold_seconds=1.5, kws_runtime=dict(contract.DEFAULT_KWS_RUNTIME),
    )
    with pytest.raises(DeployError, match="類別數"):
        contract.validate(c)


def test_validate_known_sound_depth_cap(tmp_path: Path) -> None:
    from tm_local.yamnet_model import frontend_contract

    board = boards.board_for("NuGestureAI-M55M1")
    base = {
        "kind": "known_sound", "application": "known_sound", "board": board, "project_id": "p",
        "project_name": "n", "labels": ["a", "b"], "models_dir": tmp_path,
        "int8_path": tmp_path / "m.tflite", "model_sha256": "x",
        "input": contract.TensorInfo("in", (1, 96, 64), "int8", 0.04, 46),
        "output": contract.TensorInfo("out", (1, 2), "int8", 0.0039, -128),
        "image_size": None, "audio_frontend": None, "yamnet_frontend": frontend_contract(),
        "encoder_depth": 12, "detection_threshold": 0.5, "class_thresholds": [0.5, 0.5],
        "hop_seconds": 0.5, "clip_seconds": 1.0, "peak_hold_seconds": 1.5,
        "kws_runtime": dict(contract.DEFAULT_KWS_RUNTIME),
    }
    with pytest.raises(DeployError, match="encoder_depth"):
        contract.validate(contract.DeployContract(**base))
    base["encoder_depth"] = 11
    contract.validate(contract.DeployContract(**base))


def test_validate_audio_frontend_lock(tmp_path: Path) -> None:
    board = boards.board_for("NuMaker-M55M1")
    frontend = {"sample_rate": 44100, "clip_seconds": 1.0, "window_ms": 25.0, "hop_ms": 10.0, "fft_size": 2048,
                "mel_bins": 40, "fmin": 20.0, "fmax": 22050.0, "db_floor": -80.0, "frame_count": 273,
                "feature_shape": [273, 40, 1]}
    c = contract.DeployContract(
        kind="audio", application="kws", board=board, project_id="p", project_name="n", labels=["bg", "yes"],
        models_dir=tmp_path, int8_path=tmp_path / "m.tflite", model_sha256="x",
        input=contract.TensorInfo("in", (1, 273, 40, 1), "int8", 0.0039, -128),
        output=contract.TensorInfo("out", (1, 2), "int8", 0.0039, -128),
        image_size=None, audio_frontend=frontend, yamnet_frontend=None, encoder_depth=None,
        detection_threshold=0.5, class_thresholds=[0.5, 0.5], hop_seconds=0.25, clip_seconds=1.0,
        peak_hold_seconds=0.0, kws_runtime=dict(contract.DEFAULT_KWS_RUNTIME),
    )
    with pytest.raises(DeployError, match="sample_rate"):
        contract.validate(c)


def test_validate_audio_frontend_missing_raises_deploy_error(tmp_path: Path) -> None:
    # K2: a bare DeployContract with audio_frontend=None used to crash validate() with a
    # raw KeyError on frontend["frame_count"] -- every MCU_AUDIO_FRONTEND key check passed
    # vacuously via frontend.get(key, expected). Must raise DeployError, not crash.
    board = boards.board_for("NuMaker-M55M1")
    c = contract.DeployContract(
        kind="audio", application="kws", board=board, project_id="p", project_name="n",
        labels=["bg", "yes"], models_dir=tmp_path, int8_path=tmp_path / "m.tflite", model_sha256="x",
        input=contract.TensorInfo("in", (1, 98, 40, 1), "int8", 0.0039, -128),
        output=contract.TensorInfo("out", (1, 2), "int8", 0.0039, -128),
        image_size=None, audio_frontend=None, yamnet_frontend=None, encoder_depth=None,
        detection_threshold=0.5, class_thresholds=[0.5, 0.5], hop_seconds=0.5, clip_seconds=1.0,
        peak_hold_seconds=0.0, kws_runtime=dict(contract.DEFAULT_KWS_RUNTIME),
    )
    with pytest.raises(DeployError):
        contract.validate(c)


def test_int8_artifact_path_traversal_rejected(store: ProjectStore) -> None:
    # K1: artifacts["int8"] is a leaf name taken straight from project.json. A hand-edited
    # value that escapes models_dir must be rejected via path_within(), never joined raw.
    pid = _trained_image_project(store)
    raw = store.get_raw_project(pid)
    report = raw["training"]["report"]
    store.set_training_result(
        pid,
        report=report,
        artifacts={
            "keras": "image_classifier.keras",
            "training_report": "training_report.json",
            "int8": "../../../../Windows/win.ini",
            "conversion_report": "conversion_report.json",
        },
    )
    with pytest.raises(DeployError):
        contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))


def test_truncated_conversion_report_json_raises_deploy_error(store: ProjectStore) -> None:
    # K2: a corrupted models/conversion_report.json used to raise a raw
    # json.JSONDecodeError out of collect() once the report's own "conversion.models" is
    # empty and the code falls back to reading the sidecar file from disk.
    pid = _trained_image_project(store)
    raw = store.get_raw_project(pid)
    report = raw["training"]["report"]
    artifacts = raw["training"]["artifacts"]
    report["conversion"] = {"state": "generated"}  # drop "models" -> forces the fallback
    store.set_training_result(pid, report=report, artifacts=artifacts)
    conv_path = store.project_dir(pid) / "models" / "conversion_report.json"
    conv_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DeployError):
        contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))


def test_conversion_report_fallback_used_when_report_missing_models(store: ProjectStore) -> None:
    # Minor: when the training report's own "conversion.models" is empty (e.g. the report
    # was written by a stage that only sets "state"), collect() must still succeed by
    # reading the sidecar models/conversion_report.json written at export time.
    pid = _trained_image_project(store)
    raw = store.get_raw_project(pid)
    report = raw["training"]["report"]
    artifacts = raw["training"]["artifacts"]
    report["conversion"] = {"state": "generated"}  # drop "models"; conversion_report.json still valid
    store.set_training_result(pid, report=report, artifacts=artifacts)
    c = contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))
    assert c.model_sha256 == FAKE_INT8_SHA256
    assert c.input.shape == (1, 224, 224, 3)


def test_non_strict_int8_rejected(store: ProjectStore) -> None:
    pid = _trained_image_project(store)
    raw = store.get_raw_project(pid)
    report = raw["training"]["report"]
    artifacts = raw["training"]["artifacts"]
    report["conversion"]["models"]["int8"]["strict_full_integer"] = False
    store.set_training_result(pid, report=report, artifacts=artifacts)
    with pytest.raises(DeployError, match="INT8"):
        contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))


def test_collect_resolves_class_thresholds_by_class_id(store: ProjectStore) -> None:
    # K3: class_thresholds is a {class_id: 0..1} map in training report settings, resolved
    # into one float per label (project class order), falling back to detection_threshold
    # for classes with no entry.
    pid = _trained_image_project(store)
    raw = store.get_raw_project(pid)
    class_ids = [item["id"] for item in raw["classes"]]
    report = raw["training"]["report"]
    artifacts = raw["training"]["artifacts"]
    report["settings"]["detection_threshold"] = 0.5
    report["settings"]["class_thresholds"] = {class_ids[0]: 0.75}
    store.set_training_result(pid, report=report, artifacts=artifacts)
    c = contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))
    assert c.class_thresholds == [0.75, 0.5]


def _trained_audio_project(
    store: ProjectStore,
    *,
    settings: dict | None = None,
    classes=("Background Noise", "yes"),
) -> str:
    """Seed a trained+exported `audio` project whose training report carries `settings`.

    An audio project is created with two classes already ("Background Noise", "Class 2"),
    so rename those in place before adding any extra ones -- add_class() on top would
    leave more classes than the caller asked for, and it returns the PROJECT dict, not the
    new class, so the id has to be read back from get_raw_project().
    """

    from tm_local.audio_frontend import AudioFrontendConfig

    project = store.create_project("audio", "Aud")
    pid = project["id"]
    existing = store.get_raw_project(pid)["classes"]
    for class_item, name in zip(existing, classes):
        store.update_class(pid, class_item["id"], {"name": name})
    for name in classes[len(existing) :]:
        store.add_class(pid, name)

    frontend = AudioFrontendConfig().to_dict()
    models = store.project_dir(pid) / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "audio_frontend.json").write_text(json.dumps(frontend), encoding="utf-8")
    (models / "audio_classifier_spectrogram_int8.tflite").write_bytes(FAKE_INT8_BYTES)
    shape_in = (1, frontend["frame_count"], frontend["mel_bins"], 1)
    conversion = _int8_report(shape_in, (1, len(classes)))
    report = {
        "project_kind": "audio",
        "settings": {**frontend, **(settings or {})},
        "conversion": {"state": "generated", **conversion},
    }
    store.set_training_result(
        pid,
        report=report,
        artifacts={
            "keras": "audio_classifier_spectrogram.keras",
            "training_report": "training_report.json",
            "int8": "audio_classifier_spectrogram_int8.tflite",
            "conversion_report": "conversion_report.json",
        },
    )
    (models / "conversion_report.json").write_text(json.dumps(conversion), encoding="utf-8")
    return pid


def test_collect_audio_reads_detection_threshold_and_background_index(
    store: ProjectStore,
) -> None:
    pid = _trained_audio_project(
        store,
        settings={"detection_threshold": 0.65, "background_class_index": 1},
        classes=("red", "green"),
    )
    c = contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))
    assert c.detection_threshold == 0.65
    assert c.background_index == 1
    # hop_seconds / preview_peak_hold_seconds are not audio settings; the defaults hold.
    assert c.hop_seconds == 0.5 and c.peak_hold_seconds == 1.5
    contract.validate(c)


def test_collect_background_index_defaults_to_none(store: ProjectStore) -> None:
    pid = _trained_audio_project(store)
    c = contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))
    assert c.background_index is None
    assert c.detection_threshold == 0.5


@pytest.mark.parametrize("value", [-1, 9, "1", None, 1.5, True])
def test_collect_background_index_rejects_out_of_range_or_non_int(
    store: ProjectStore, value: object
) -> None:
    # A hand-edited or stale report must fall back to the firmware's label matcher rather
    # than baking a nonsense class index into the generated C.
    pid = _trained_audio_project(store, settings={"background_class_index": value})
    c = contract.collect(store, pid, boards.board_for("NuMaker-M55M1"))
    assert c.background_index is None
