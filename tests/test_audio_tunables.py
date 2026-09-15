"""Audio-kind tunables: augmentation presets, SpecAugment, session-disjoint validation,
background weighting and dropout.

Everything except the four TensorFlow cases at the bottom is pure logic over synthetic
paths, so the bulk of this file runs without importing TensorFlow at all.
"""

from __future__ import annotations

import ast
import inspect
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tm_local import audio_pipeline
from tm_local.project_store import ProjectStore


def test_presets_medium_equals_legacy() -> None:
    assert audio_pipeline.AUDIO_AUGMENTATION_PRESETS["medium"] == {
        "shift_div": 12,
        "gain": 0.10,
        "noise": 0.012,
    }
    assert audio_pipeline.AUDIO_AUGMENTATION_PRESETS["off"] is None
    assert audio_pipeline.AUDIO_AUGMENTATION_PRESETS["light"] == {
        "shift_div": 24,
        "gain": 0.05,
        "noise": 0.006,
    }
    assert audio_pipeline.AUDIO_AUGMENTATION_PRESETS["strong"] == {
        "shift_div": 6,
        "gain": 0.20,
        "noise": 0.020,
    }


def test_spec_augment_masks_bounds() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        t, f = audio_pipeline.spec_augment_masks(98, 40, rng)
        assert 0 <= t.start <= t.stop <= 98 and t.stop - t.start <= 9
        assert 0 <= f.start <= f.stop <= 40 and f.stop - f.start <= 4


def test_spec_augment_masks_are_deterministic_per_seed() -> None:
    first = [audio_pipeline.spec_augment_masks(98, 40, np.random.default_rng(7)) for _ in range(3)]
    second = [audio_pipeline.spec_augment_masks(98, 40, np.random.default_rng(7)) for _ in range(3)]
    assert first == second


def test_background_class_index() -> None:
    classes = [{"id": "a", "name": "yes"}, {"id": "b", "name": "Background Noise"}]
    assert audio_pipeline.background_class_index(classes, "") == 1
    assert audio_pipeline.background_class_index(classes, "a") == 0
    assert audio_pipeline.background_class_index([{"id": "x", "name": "cat"}], "") is None
    # A stale id (its class was deleted) falls back to the name hints rather than deciding
    # the project has no background class at all.
    assert audio_pipeline.background_class_index(classes, "gone") == 1
    assert audio_pipeline.background_class_index([{"id": "x", "name": "cat"}], "gone") is None


def test_background_class_index_shares_the_firmware_matcher() -> None:
    """The board and the trainer must agree on which class means "nothing was said"."""

    from tm_local.mcu import kws_codegen

    classes = [{"id": "a", "name": "開燈"}, {"id": "b", "name": "環境音"}]
    labels = [item["name"] for item in classes]
    assert audio_pipeline.background_class_index(classes, "") == kws_codegen.background_index(labels)
    assert kws_codegen.has_background(labels) is True
    # A hint the firmware knows ("環境音") but a hand-rolled English-only list would miss.
    assert audio_pipeline.background_class_index(classes, "") == 1


def _refs(spec: dict[int, dict[str, int]]) -> list[tuple[int, Path, str]]:
    refs = []
    for class_index, sessions in spec.items():
        for session, count in sessions.items():
            for i in range(count):
                refs.append((class_index, Path(f"/c{class_index}/{session}_{i}.wav"), session))
    return refs


def test_split_audio_clips_session_disjoint_and_fallback() -> None:
    refs = _refs({0: {"s1": 5, "s2": 5, "s3": 5}, 1: {"only": 6}})
    split, info = audio_pipeline.split_audio_clips(refs, 0.25, True)
    train_sessions = {
        p.name.split("_")[0] for p, label in zip(split.train_paths, split.train_labels) if label == 0
    }
    val_sessions = {
        p.name.split("_")[0]
        for p, label in zip(split.validation_paths, split.validation_labels)
        if label == 0
    }
    assert train_sessions.isdisjoint(val_sessions) and val_sessions
    assert info["fallback_classes"] == [1] and "split_warning" in info
    assert any(label == 1 for label in split.validation_labels)
    assert info["split_rule"].startswith("session-disjoint")
    clip_split, clip_info = audio_pipeline.split_audio_clips(refs, 0.25, False)
    assert clip_info["split_rule"].startswith("clip-level")
    assert len(clip_split.validation_paths) > 0
    assert clip_info["fallback_classes"] == []


def test_split_audio_clips_keeps_every_clip_and_is_deterministic() -> None:
    refs = _refs({0: {"s1": 4, "s2": 4}, 1: {"t1": 4, "t2": 4}})
    split, info = audio_pipeline.split_audio_clips(refs, 0.25, True)
    assert len(split.train_paths) + len(split.validation_paths) == len(refs)
    assert len(split.train_paths) == len(split.train_labels)
    assert len(split.validation_paths) == len(split.validation_labels)
    assert set(split.train_paths) | set(split.validation_paths) == {p for _, p, _ in refs}
    assert sorted(info["train_sessions"]) and sorted(info["validation_sessions"])
    again, _ = audio_pipeline.split_audio_clips(refs, 0.25, True)
    assert again.train_paths == split.train_paths
    assert again.validation_labels == split.validation_labels


def test_split_audio_clips_without_validation_is_clip_level() -> None:
    refs = _refs({0: {"s1": 3, "s2": 3}, 1: {"t1": 3}})
    split, info = audio_pipeline.split_audio_clips(refs, 0.0, True)
    assert split.validation_paths == []
    assert info["split_rule"].startswith("clip-level")


def test_store_audio_clip_refs(store: ProjectStore, wav_bytes: bytes) -> None:
    project = store.create_project("audio", "Aud")
    class_id = store.get_raw_project(project["id"])["classes"][0]["id"]
    store.add_audio_bytes(
        project["id"],
        class_id,
        wav_bytes,
        source_name="take.wav",
        recording_session_id="sess-1",
    )
    refs = store.audio_clip_refs(project["id"])
    assert refs and all(r[2] == "sess-1" and r[0] == 0 and r[1].is_file() for r in refs)
    # Order matches sample_paths_by_class() so class indices and labels stay aligned.
    by_class = store.sample_paths_by_class(project["id"])
    assert [path for _, path, _ in refs] == list(by_class[0][1])


def test_store_audio_clip_refs_falls_back_to_the_sample_id(
    store: ProjectStore, wav_bytes: bytes
) -> None:
    """Clips imported before recording_session_id existed must still split per clip."""

    project = store.create_project("audio", "Aud")
    raw = store.get_raw_project(project["id"])
    class_id = raw["classes"][1]["id"]
    store.add_audio_bytes(project["id"], class_id, wav_bytes, source_name="take.wav")
    payload = store.get_raw_project(project["id"])
    sample_id = next(iter(payload["samples"]))
    payload["samples"][sample_id].pop("recording_session_id", None)
    store._save(payload)  # simulating a project.json written by an older build
    refs = store.audio_clip_refs(project["id"])
    orphan = [ref for ref in refs if ref[1].stem == sample_id]
    assert orphan and orphan[0][2] == f"sample-{sample_id}"
    assert all(ref[0] == 1 for ref in refs)


# ----------------------------------------------------------------------------------
# INT8 calibration must not look at the held-out audio
# ----------------------------------------------------------------------------------


def test_audio_calibration_restricts_itself_to_the_training_clips(
    store: ProjectStore, wav_bytes: bytes
) -> None:
    """The known_sound branch already calibrated on train sessions only; audio did not.

    Letting held-out clips set the INT8 quantisation ranges means the validation accuracy
    the report quotes describes a model the validation set helped shape.
    """

    from tm_local import export_service

    project = store.create_project("audio", "Aud")
    project_id = project["id"]
    for class_item in store.get_raw_project(project_id)["classes"]:
        for index in range(3):
            store.add_audio_bytes(
                project_id,
                class_item["id"],
                wav_bytes,
                source_name=f"take{index}.wav",
                recording_session_id=f"{class_item['id']}-s{index}",
            )
    payload = store.get_raw_project(project_id)
    settings = payload["settings"]
    class_paths = store.sample_paths_by_class(project_id)
    split, _info = audio_pipeline.split_audio_clips(
        store.audio_clip_refs(project_id),
        float(settings["validation_split"]),
        True,
        class_count=len(class_paths),
    )
    assert split.validation_paths, "the fixture must hold something out to prove anything"

    restricted = export_service._audio_train_class_paths(store, payload, settings, class_paths)
    kept = [path for _, paths in restricted for path in paths]
    assert set(kept).isdisjoint(split.validation_paths)
    assert set(kept) == set(split.train_paths)
    # Same classes in the same order: _round_robin_paths() still gets one queue per class,
    # so calibration stays balanced across classes rather than favouring the first one.
    assert [item["id"] for item, _ in restricted] == [item["id"] for item, _ in class_paths]


def test_audio_calibration_keeps_a_class_with_no_training_clips(
    store: ProjectStore, wav_bytes: bytes
) -> None:
    """A class that contributed nothing to training keeps its own clips.

    Handing back an empty list would take that class's dynamic range out of the
    calibration set altogether, which skews the quantisation ranges further than the leak
    this restriction exists to prevent.
    """

    from tm_local import export_service

    project = store.create_project("audio", "Aud")
    project_id = project["id"]
    class_item = store.get_raw_project(project_id)["classes"][0]
    for index in range(3):
        store.add_audio_bytes(
            project_id,
            class_item["id"],
            wav_bytes,
            source_name=f"take{index}.wav",
            recording_session_id=f"{class_item['id']}-s{index}",
        )
    payload = store.get_raw_project(project_id)
    ghost = [Path("/nowhere/ghost.wav")]
    class_paths = store.sample_paths_by_class(project_id) + [({"id": "ghost"}, ghost)]
    restricted = export_service._audio_train_class_paths(
        store, payload, payload["settings"], class_paths
    )
    assert restricted[-1][1] == ghost


def test_audio_calibration_tolerates_a_project_with_no_clips(store: ProjectStore) -> None:
    """An untrained-looking project (no samples at all) must not blow up here.

    tests/test_export_warnings.py drives the real ensure_export_artifacts() over exactly
    that shape.
    """

    from tm_local import export_service

    project = store.create_project("audio", "Empty")
    payload = store.get_raw_project(project["id"])
    class_paths = store.sample_paths_by_class(project["id"])
    assert (
        export_service._audio_train_class_paths(
            store, payload, payload["settings"], class_paths
        )
        == class_paths
    )


# ----------------------------------------------------------------------------------
# The KWS trigger policy: one resolver, two export consumers
# ----------------------------------------------------------------------------------


def test_kws_decision_policy_defaults_clamp_and_drop_unknown_keys() -> None:
    from tm_local.config import AUDIO_DEFAULTS, KWS_RUNTIME_DEFAULTS

    assert audio_pipeline.kws_decision_policy({}) == {
        "detection_threshold": float(AUDIO_DEFAULTS["detection_threshold"]),
        "kws_runtime": dict(KWS_RUNTIME_DEFAULTS),
    }
    # A report written by an older build has no settings mapping at all.
    assert audio_pipeline.kws_decision_policy(None)["detection_threshold"] == float(
        AUDIO_DEFAULTS["detection_threshold"]
    )
    policy = audio_pipeline.kws_decision_policy(
        {
            "detection_threshold": 0.8,
            "kws_runtime": {"required_hits": 3, "not_a_real_knob": 1},
        }
    )
    assert policy["detection_threshold"] == 0.8
    assert policy["kws_runtime"]["required_hits"] == 3
    assert "not_a_real_knob" not in policy["kws_runtime"]
    # Same clamp as mcu/contract.py, so the ZIP can never claim a threshold the board
    # would refuse to compile.
    assert audio_pipeline.kws_decision_policy({"detection_threshold": 9.0})[
        "detection_threshold"
    ] == 0.99
    assert audio_pipeline.kws_decision_policy({"detection_threshold": "nope"})[
        "detection_threshold"
    ] == float(AUDIO_DEFAULTS["detection_threshold"])


def test_both_audio_export_consumers_share_the_resolver() -> None:
    """run_model.py and metadata.json must not drift apart with two copies of the rule."""

    from tm_local import export_service
    from tm_local.tflite_export import build_model_download

    assert export_service.kws_decision_policy is audio_pipeline.kws_decision_policy
    assert "kws_decision_policy" in inspect.getsource(build_model_download)


def test_runner_bakes_the_detection_threshold(tmp_path: Path) -> None:
    audio_pipeline.write_audio_runner(
        tmp_path, "audio_classifier_spectrogram_int8.tflite", 0.65
    )
    source = (tmp_path / "run_model.py").read_text(encoding="utf-8")
    ast.parse(source)  # the runner must at least be valid Python
    # Pure ASCII: a student's editor re-saving this as cp950 must not break the import.
    source.encode("ascii")
    assert "DETECTION_THRESHOLD = 0.65" in source
    assert "best_score >= DETECTION_THRESHOLD" in source
    assert "nothing above threshold" in source
    # Out-of-range values are clamped, not written through.
    audio_pipeline.write_audio_runner(tmp_path, "m.tflite", 3.0)
    assert "DETECTION_THRESHOLD = 0.99" in (tmp_path / "run_model.py").read_text(
        encoding="utf-8"
    )


def test_export_metadata_carries_the_kws_trigger_policy(tmp_path: Path) -> None:
    """Spec 6.4: the trigger params travel in metadata.json, not only in the report."""

    from tm_local.config import KWS_RUNTIME_DEFAULTS
    from tm_local.tflite_export import build_model_download

    project_dir = tmp_path / "project"
    models = project_dir / "models"
    models.mkdir(parents=True)
    (models / "audio_classifier_spectrogram_int8.tflite").write_bytes(b"TFL3" + bytes(range(8)))
    project = {
        "id": "p1",
        "name": "Words",
        "kind": "audio",
        "classes": [{"id": "a", "name": "Background Noise"}, {"id": "b", "name": "On"}],
        "training": {
            "state": "trained",
            "artifacts": {"int8": "audio_classifier_spectrogram_int8.tflite"},
            "report": {
                "settings": {
                    "detection_threshold": 0.72,
                    "kws_runtime": {"required_hits": 4},
                }
            },
        },
    }
    output = build_model_download(
        project=project,
        project_dir=project_dir,
        selected=["int8"],
        include_c_header=False,
        output_zip=tmp_path / "model.zip",
    )
    with zipfile.ZipFile(output) as archive:
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
    assert metadata["detection_threshold"] == 0.72
    assert metadata["kws_runtime"]["required_hits"] == 4
    assert metadata["kws_runtime"]["smooth_windows"] == KWS_RUNTIME_DEFAULTS["smooth_windows"]
    assert metadata["labels"] == ["Background Noise", "On"]


# ======================================================================================
# TensorFlow-backed cases (model + tf.data). Four of them, all tiny, all marked slow so
# `pytest -m "not slow"` keeps its promise of never importing TensorFlow.
# ======================================================================================


@pytest.mark.slow
def test_build_audio_model_honours_dropout() -> None:
    tf = pytest.importorskip("tensorflow")

    tf.keras.backend.clear_session()
    model = audio_pipeline._build_audio_model((98, 40, 1), 3, dropout=0.45)
    layer = model.get_layer("dropout")
    assert pytest.approx(float(layer.rate)) == 0.45
    assert model.output_shape[-1] == 3
    default_model = audio_pipeline._build_audio_model((98, 40, 1), 2)
    assert pytest.approx(float(default_model.get_layer("dropout").rate)) == 0.20


@pytest.mark.slow
def test_make_dataset_augmentation_off_and_spec_augment_bands() -> None:
    tf = pytest.importorskip("tensorflow")

    tf.keras.utils.set_random_seed(1337)
    features = np.ones((8, 20, 10, 1), dtype=np.float32)
    labels = np.zeros((8,), dtype=np.int32)

    untouched = audio_pipeline._make_dataset(
        features, labels, 4, True, augmentation="off", spec_augment=False
    )
    for batch, _ in untouched:
        assert np.array_equal(batch.numpy(), np.ones_like(batch.numpy()))

    masked = audio_pipeline._make_dataset(
        features, labels, 1, True, augmentation="off", spec_augment=True
    )
    saw_mask = False
    for batch, _ in masked:
        sample = batch.numpy()[0, :, :, 0]
        zero_rows = np.flatnonzero(np.all(sample == 0.0, axis=1))
        zero_cols = np.flatnonzero(np.all(sample == 0.0, axis=0))
        for zeros, limit in ((zero_rows, 20 // 10), (zero_cols, 10 // 10)):
            assert zeros.size <= limit
            if zeros.size:
                saw_mask = True
                assert np.array_equal(zeros, np.arange(zeros[0], zeros[0] + zeros.size))
        # Anything outside the two bands is left exactly as it was.
        kept = np.ones_like(sample)
        kept[zero_rows, :] = 0.0
        kept[:, zero_cols] = 0.0
        assert np.array_equal(sample, kept)
    assert saw_mask, "spec_augment never masked anything in 8 draws"


@pytest.mark.slow
def test_background_weight_skip_is_visible_and_recorded(
    store: ProjectStore, wav_bytes: bytes
) -> None:
    """A project with no background class must SAY the weighting was skipped.

    Silently ignoring it costs a full retrain to discover, and the report used to echo the
    requested weight as though it had been applied.
    """

    pytest.importorskip("tensorflow")

    project = store.create_project("audio", "Aud")
    raw = store.get_raw_project(project["id"])
    # Rename every class away from the firmware's background hint list, so
    # background_class_index() genuinely finds nothing.
    for index, class_item in enumerate(raw["classes"]):
        store.update_class(project["id"], class_item["id"], {"name": f"word{index}"})
    for class_item in store.get_raw_project(project["id"])["classes"]:
        for index in range(3):
            store.add_audio_bytes(
                project["id"],
                class_item["id"],
                wav_bytes,
                source_name=f"take{index}.wav",
                recording_session_id=f"{class_item['id']}-s{index}",
            )
    messages: list[str] = []
    result = audio_pipeline.train_audio_project(
        store,
        project["id"],
        {"epochs": 1, "minimum_samples_per_class": 2, "background_weight": 3.0},
        lambda _value, message: messages.append(message),
    )
    settings = result["report"]["settings"]
    assert settings["background_class_index"] is None
    assert settings["background_weight_requested"] == 3.0
    assert settings["background_weight_applied"] == 1.0
    skipped = [message for message in messages if "略過" in message]
    assert len(skipped) == 1, messages
    assert "background_weight" in skipped[0] and "背景類別" in skipped[0]


@pytest.mark.slow
def test_train_audio_project_records_tunables(store: ProjectStore, wav_bytes: bytes) -> None:
    pytest.importorskip("tensorflow")

    from tm_local.config import KWS_RUNTIME_DEFAULTS

    project = store.create_project("audio", "Aud")
    raw = store.get_raw_project(project["id"])
    background_id = raw["classes"][0]["id"]  # "Background Noise"
    for class_item in raw["classes"]:
        for index in range(3):
            store.add_audio_bytes(
                project["id"],
                class_item["id"],
                wav_bytes,
                source_name=f"take{index}.wav",
                recording_session_id=f"{class_item['id']}-s{index}",
            )
    result = audio_pipeline.train_audio_project(
        store,
        project["id"],
        {
            "epochs": 1,
            "minimum_samples_per_class": 2,
            "augmentation_level": "light",
            "spec_augment": True,
            "background_weight": 2.0,
            "dropout": 0.35,
            "detection_threshold": 0.65,
        },
    )
    settings = result["report"]["settings"]
    assert settings["augmentation_level"] == "light"
    assert settings["spec_augment"] is True
    assert settings["session_disjoint_validation"] is True
    assert settings["background_weight_requested"] == 2.0
    assert settings["background_weight_applied"] == 2.0
    assert settings["background_class_index"] == 0
    assert settings["dropout"] == 0.35
    assert settings["detection_threshold"] == 0.65
    assert settings["deployment_target"] == "pc"
    assert settings["early_stopping"] is True
    assert settings["kws_runtime"] == dict(KWS_RUNTIME_DEFAULTS)
    dataset = result["report"]["dataset"]
    assert dataset["split_rule"].startswith("session-disjoint")
    assert dataset["fallback_classes"] == []
    assert dataset["validation_samples"] > 0
    assert set(dataset["train_sessions"]).isdisjoint(dataset["validation_sessions"])
    assert background_id == raw["classes"][0]["id"]
