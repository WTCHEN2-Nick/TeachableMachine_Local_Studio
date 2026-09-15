"""Tunable parameters for the known_sound kind.

Per-class thresholds, background weighting, head dropout and waveform augmentation.
Everything that can be checked without TensorFlow is checked without it; the two
TensorFlow-backed tests are marked ``slow`` because they build the real YAMNet encoder.
"""

from __future__ import annotations

import ast
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tm_local import known_sound_pipeline as ks
from tm_local.audio_frontend import encode_wav_bytes
from tm_local.project_store import ProjectStore

# ----------------------------------------------------------------------------------
# Per-class thresholds
# ----------------------------------------------------------------------------------


def test_resolve_class_thresholds() -> None:
    classes = [{"id": "a", "name": "gun"}, {"id": "b", "name": "dog"}]
    assert ks.resolve_class_thresholds(
        {"detection_threshold": 0.5, "class_thresholds": {"b": 0.8}}, classes
    ) == [0.5, 0.8]
    assert ks.resolve_class_thresholds({"detection_threshold": 0.4}, classes) == [0.4, 0.4]


def test_resolve_class_thresholds_tolerates_stored_strings() -> None:
    """Settings that round-tripped through a form may carry strings, not floats."""

    classes = [{"id": "a", "name": "gun"}, {"id": "b", "name": "dog"}]
    resolved = ks.resolve_class_thresholds(
        {"detection_threshold": "0.5", "class_thresholds": {"b": "0.75"}}, classes
    )
    assert resolved == [0.5, 0.75]
    # A stale id left behind by a deleted class must not shift the other classes.
    assert ks.resolve_class_thresholds(
        {"detection_threshold": 0.5, "class_thresholds": {"gone": 0.9}}, classes
    ) == [0.5, 0.5]


# ----------------------------------------------------------------------------------
# Waveform augmentation
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("level, copies", [("off", 0), ("light", 1), ("medium", 2), ("strong", 3)])
def test_waveform_augmentations(level: str, copies: int) -> None:
    rng = np.random.default_rng(1)
    waves = (rng.uniform(-0.3, 0.3, (4, 16000))).astype(np.float32)
    labels = np.array([0, 1, 0, 1])
    aug, aug_labels = ks.build_waveform_augmentations(waves, labels, level)
    assert aug.shape == (4 * copies, 16000) and aug_labels.shape == (4 * copies,)
    if copies:
        assert aug.dtype == np.float32 and float(np.max(np.abs(aug))) <= 1.0
        assert not np.allclose(aug[0], waves[0])
        assert list(aug_labels[:4]) == [0, 1, 0, 1]
    again, _ = ks.build_waveform_augmentations(waves, labels, level)
    assert np.array_equal(aug, again)  # deterministic


def test_waveform_augmentations_on_empty_input() -> None:
    empty = np.zeros((0, 16000), dtype=np.float32)
    aug, aug_labels = ks.build_waveform_augmentations(
        empty, np.zeros((0,), dtype=np.int64), "strong"
    )
    assert aug.shape == (0, 16000) and aug_labels.shape == (0,)
    assert aug.dtype == np.float32 and aug_labels.dtype == np.int64


def test_waveform_augmentations_unknown_level_is_off() -> None:
    waves = np.zeros((2, 16000), dtype=np.float32)
    aug, _ = ks.build_waveform_augmentations(waves, np.array([0, 1]), "extreme")
    assert aug.shape == (0, 16000)


# ----------------------------------------------------------------------------------
# Background weighting
# ----------------------------------------------------------------------------------


def test_background_sample_weights() -> None:
    targets = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    assert ks.background_sample_weights(targets, None, 3.0) is None
    assert ks.background_sample_weights(targets, 1, 1.0) is None
    weights = ks.background_sample_weights(targets, 1, 0.25)
    assert weights is not None
    assert list(weights) == [1.0, 0.25, 0.25]


def test_background_sample_weights_leave_mixup_rows_alone() -> None:
    """A mixup row is "target AND background"; weighting it would scale the target too."""

    real = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    weights = ks.background_sample_weights(real, 1, 0.25, mixup_rows=3)
    assert weights is not None
    assert list(weights) == [1.0, 0.25, 1.0, 1.0, 1.0]
    assert weights.dtype == np.float32
    # Still nothing to do when the knob is untouched, however many mixup rows there are.
    assert ks.background_sample_weights(real, 1, 1.0, mixup_rows=3) is None


# ----------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------


class _FakeHead:
    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores

    def predict(self, embeddings, verbose=0):
        return self._scores


def test_evaluate_uses_per_class_thresholds() -> None:
    scores = np.array([[0.6, 0.6], [0.3, 0.9]])
    targets = np.array([[1, 0], [0, 1]], dtype=np.float32)
    result = ks._evaluate(_FakeHead(scores), np.zeros((2, 8)), targets, ["gun", "dog"], [0.5, 0.7])
    gun, dog = result["per_class"]["gun"], result["per_class"]["dog"]
    assert gun["true_positive"] == 1 and gun["false_positive"] == 0
    assert dog["false_positive"] == 0 and dog["true_positive"] == 1
    assert result["thresholds"] == [0.5, 0.7] and result["threshold"] == 0.5


# ----------------------------------------------------------------------------------
# One resolver for every consumer of a stored report
# ----------------------------------------------------------------------------------


def test_thresholds_for_labels_falls_back_for_an_old_report() -> None:
    assert ks.thresholds_for_labels({"detection_threshold": 0.7}, 3) == [0.7, 0.7, 0.7]
    assert ks.thresholds_for_labels({}, 2) == [0.5, 0.5]
    assert ks.thresholds_for_labels(
        {"detection_threshold": 0.5, "class_thresholds_by_label": [0.4, 0.9]}, 2
    ) == [0.4, 0.9]


@pytest.mark.parametrize(
    "stored",
    [
        [0.5],
        [0.5, 0.5, 0.5],
        "0.5",
        {"a": 0.5},
        [0.5, "high"],
        [0.5, None],
        [0.5, float("nan")],
        # Out of range: 1.5 is a threshold no score can reach and -0.2 is one every score
        # exceeds, so both silently disable the class they are supposed to tune.
        [0.5, 1.5],
        [0.5, -0.2],
        [0.5, 0.0],
        [0.5, 1.0],
    ],
)
def test_thresholds_for_labels_rejects_a_malformed_list(stored: object) -> None:
    """A silent fallback would ship a model that decides differently from Preview."""

    from tm_local.tflite_export import ExportError

    with pytest.raises(ExportError) as excinfo:
        ks.thresholds_for_labels(
            {"detection_threshold": 0.5, "class_thresholds_by_label": stored}, 2
        )
    message = str(excinfo.value)
    assert "class_thresholds" in message
    # Not "再匯出": thresholds_for_labels() also runs on the Preview path, where there is
    # no export to retry.
    assert "請重新訓練後再試一次" in message
    assert "再匯出" not in message


def test_both_export_consumers_share_the_resolver() -> None:
    """The runner and metadata.json must not drift apart with two copies of the rule."""

    from tm_local import export_service

    assert export_service.known_sound_thresholds is ks.thresholds_for_labels


def test_validator_normalises_every_key_it_validates() -> None:
    """The eleven pre-plan keys were coerced into a local and thrown away.

    ``PATCH {"settings": {"detection_threshold": "0.7"}}`` therefore persisted the string
    ``"0.7"``, while the sibling ``validate_audio_settings`` stored a float for the same
    key. Nothing in ``web/app.js`` sends a string (it wraps every field in ``Number(...)``)
    so no live defect was found, but the validator's own docstring promised normalisation.
    """

    from tm_local.config import KNOWN_SOUND_DEFAULTS, validate_known_sound_settings

    submitted = {
        **KNOWN_SOUND_DEFAULTS,
        "encoder_backend": " YAMNet_Embedding ",
        "sample_rate": "16000",
        "clip_seconds": "1",
        "hop_seconds": "0.5",
        "encoder_depth": "12",
        "detection_threshold": "0.7",
        "mixup_ratio": "0.25",
        "preview_peak_hold_seconds": "1.5",
        "minimum_sessions_per_class": "3",
        "minimum_clips_per_class": "4",
        "epochs": "40",
        "batch_size": "16",
        "learning_rate": "0.001",
    }
    result = validate_known_sound_settings(submitted)
    assert result["encoder_backend"] == "yamnet_embedding"
    assert result["sample_rate"] == 16000 and isinstance(result["sample_rate"], int)
    assert result["clip_seconds"] == 1.0 and isinstance(result["clip_seconds"], float)
    assert result["hop_seconds"] == 0.5 and isinstance(result["hop_seconds"], float)
    assert result["encoder_depth"] == 12 and isinstance(result["encoder_depth"], int)
    assert result["detection_threshold"] == 0.7
    assert result["mixup_ratio"] == 0.25
    assert result["preview_peak_hold_seconds"] == 1.5
    assert result["minimum_sessions_per_class"] == 3
    assert result["minimum_clips_per_class"] == 4
    assert result["epochs"] == 40 and result["batch_size"] == 16
    assert result["learning_rate"] == 0.001
    for key in (
        "sample_rate",
        "clip_seconds",
        "hop_seconds",
        "detection_threshold",
        "mixup_ratio",
        "preview_peak_hold_seconds",
        "minimum_sessions_per_class",
        "minimum_clips_per_class",
        "epochs",
        "batch_size",
        "learning_rate",
    ):
        assert not isinstance(result[key], str), key


# ----------------------------------------------------------------------------------
# Exported runner and ZIP metadata
# ----------------------------------------------------------------------------------


def test_runner_bakes_one_threshold_per_class(tmp_path: Path) -> None:
    path = ks.write_known_sound_runner(
        tmp_path, "known_sound_yamnet_classifier_int8.tflite", ["gun", "dog"], [0.4, 0.85]
    )
    source = path.read_text(encoding="utf-8")
    ast.parse(source)  # the runner must at least be valid Python
    assert "THRESHOLDS = [0.4, 0.85]" in source
    assert "THRESHOLD =" not in source
    assert "THRESHOLDS[index]" in source
    assert "do not add up to 100%" in source


def _write_known_sound_support(models: Path) -> None:
    """The support files build_model_download() insists on for a known_sound export."""

    models.mkdir(parents=True, exist_ok=True)
    (models / "known_sound_yamnet_classifier_int8.tflite").write_bytes(b"TFL3" + bytes(range(8)))
    (models / "conversion_report.json").write_text("{}", encoding="utf-8")
    (models / "training_report.json").write_text("{}", encoding="utf-8")
    (models / "yamnet_frontend.json").write_text("{}", encoding="utf-8")
    (models / "yamnet_frontend_reference.py").write_text("", encoding="utf-8")
    (models / "run_model.py").write_text("", encoding="utf-8")


def test_export_metadata_carries_known_sound_thresholds(tmp_path: Path) -> None:
    from tm_local.tflite_export import build_model_download

    project_dir = tmp_path / "project"
    models = project_dir / "models"
    models.mkdir(parents=True)
    _write_known_sound_support(models)
    project = {
        "id": "p1",
        "name": "Sounds",
        "kind": "known_sound",
        "classes": [{"id": "a", "name": "Gun"}, {"id": "b", "name": "Dog"}],
        "training": {
            "state": "trained",
            "artifacts": {"int8": "known_sound_yamnet_classifier_int8.tflite"},
            "report": {
                "settings": {
                    "detection_threshold": 0.6,
                    "class_thresholds": {"b": 0.85},
                    "class_thresholds_by_label": [0.6, 0.85],
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
    assert metadata["detection_threshold"] == 0.6
    assert metadata["class_thresholds"] == [0.6, 0.85]
    assert metadata["labels"] == ["Gun", "Dog"]


def test_export_metadata_falls_back_to_the_single_threshold(tmp_path: Path) -> None:
    """An old project trained before per-class thresholds still exports valid metadata."""

    from tm_local.tflite_export import build_model_download

    project_dir = tmp_path / "project"
    models = project_dir / "models"
    models.mkdir(parents=True)
    _write_known_sound_support(models)
    project = {
        "id": "p1",
        "name": "Sounds",
        "kind": "known_sound",
        "classes": [{"id": "a", "name": "Gun"}, {"id": "b", "name": "Dog"}],
        "training": {
            "state": "trained",
            "artifacts": {"int8": "known_sound_yamnet_classifier_int8.tflite"},
            "report": {"settings": {"detection_threshold": 0.7}},
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
    assert metadata["detection_threshold"] == 0.7
    assert metadata["class_thresholds"] == [0.7, 0.7]


def test_export_refuses_a_report_whose_threshold_list_is_broken(tmp_path: Path) -> None:
    from tm_local.tflite_export import ExportError, build_model_download

    project_dir = tmp_path / "project"
    models = project_dir / "models"
    models.mkdir(parents=True)
    _write_known_sound_support(models)
    project = {
        "id": "p1",
        "name": "Sounds",
        "kind": "known_sound",
        "classes": [{"id": "a", "name": "Gun"}, {"id": "b", "name": "Dog"}],
        "training": {
            "state": "trained",
            "artifacts": {"int8": "known_sound_yamnet_classifier_int8.tflite"},
            "report": {
                "settings": {
                    "detection_threshold": 0.7,
                    "class_thresholds_by_label": [0.7],  # one class went missing
                }
            },
        },
    }
    with pytest.raises(ExportError, match="class_thresholds"):
        build_model_download(
            project=project,
            project_dir=project_dir,
            selected=["int8"],
            include_c_header=False,
            output_zip=tmp_path / "model.zip",
        )


# ----------------------------------------------------------------------------------
# TensorFlow-backed
# ----------------------------------------------------------------------------------


@pytest.mark.slow
def test_head_dropout_keeps_trainable_count() -> None:
    pytest.importorskip("tensorflow")
    model = ks.build_known_sound_model(3, encoder_depth=6, head_dropout=0.3)
    names = [layer.name for layer in model.layers]
    assert "head_dropout" in names and names.index("head_dropout") < names.index("class_scores")
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    from tm_local.yamnet_model import encoder_output_dim

    assert trainable == encoder_output_dim(6) * 3 + 3
    # The freeze loop sets trainable=False on every layer but the head, including this
    # Dropout. Only BatchNormalization treats that as "run in inference mode", so the
    # dropout must still actually drop during training -- otherwise the knob is a no-op.
    patch = np.random.RandomState(0).uniform(-5.0, 0.0, (1, 96, 64)).astype(np.float32)
    draws = {float(model(patch, training=True).numpy().sum()) for _ in range(8)}
    assert len(draws) > 1


@pytest.mark.slow
def test_ensure_export_artifacts_rejects_a_malformed_class_thresholds_list(
    store: ProjectStore,
) -> None:
    """Task 7 Part A group 2: test_export_refuses_a_report_whose_threshold_list_is_broken
    above only proves build_model_download() (the hand-assembled ZIP-download path) raises
    on a malformed class_thresholds_by_label. The real Export Model job goes through
    export_service.ensure_export_artifacts() -> write_known_sound_runner() ->
    thresholds_for_labels(), a different call site with its own project/report plumbing --
    this exercises that one directly. ``float32`` is requested (not int8/uint8) so the
    conversion needs no representative calibration samples and
    _validate_known_sound_parity() is a no-op for it (it only inspects int8/uint8
    models), reaching the threshold-resolution raise via a real subprocess conversion
    without either.
    """

    pytest.importorskip("tensorflow")
    from tm_local.export_service import ensure_export_artifacts
    from tm_local.tflite_export import ExportError
    from tm_local.training_common import save_native_keras_model

    project = store.create_project("known_sound", "MalformedThresholds")
    project_id = project["id"]
    class_count = len(project["classes"])

    model = ks.build_known_sound_model(class_count, encoder_depth=2)
    models_dir = store.project_dir(project_id) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    keras_path = models_dir / "known_sound_yamnet_classifier_d2.keras"
    save_native_keras_model(model, keras_path)

    store.set_training_result(
        project_id,
        report={
            "encoder": {"depth": 2},
            "settings": {
                "detection_threshold": 0.5,
                # One entry short of the project's default class count -- the exact
                # malformed shape thresholds_for_labels() must reject rather than
                # silently padding, trimming or falling back to detection_threshold.
                "class_thresholds_by_label": [0.5] * (class_count - 1),
            },
        },
        artifacts={"keras": keras_path.name},
    )

    with pytest.raises(ExportError, match="class_thresholds"):
        ensure_export_artifacts(store, project_id, ["float32"], lambda *_: None)


def _tone_wav(frequency: float, seconds: float, amplitude: float = 0.3) -> bytes:
    sample_rate = 16000
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    signal = amplitude * np.sin(2.0 * math.pi * frequency * t)
    return encode_wav_bytes(signal.astype(np.float32), sample_rate)


def _noise_wav(seconds: float, seed: int = 0) -> bytes:
    rng = np.random.RandomState(seed)
    signal = (rng.randn(int(16000 * seconds)) * 0.15).astype(np.float32)
    return encode_wav_bytes(np.clip(signal, -1.0, 1.0), 16000)


@pytest.mark.slow
def test_known_sound_report_settings_carry_class_thresholds_dict(
    store: ProjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCU contract reads report.settings.class_thresholds as a {class_id: float} dict."""

    pytest.importorskip("tensorflow")

    # Record how many clips each encoder pass saw: train first, validation second.
    embedded_rows: list[int] = []
    original_embed = ks._embed_waveforms

    def spy(encoder, waveforms, **kwargs):
        embedded_rows.append(int(waveforms.shape[0]))
        return original_embed(encoder, waveforms, **kwargs)

    monkeypatch.setattr(ks, "_embed_waveforms", spy)

    project = store.create_project("known_sound", "Tunables")
    project_id = project["id"]
    tone_class, noise_class = project["classes"][0]["id"], project["classes"][1]["id"]
    background_class = project["classes"][2]["id"]
    for session in ("tone-a", "tone-b"):
        store.add_audio_bytes(
            project_id, tone_class, _tone_wav(440.0, 4.0), recording_session_id=session
        )
    for index, session in enumerate(("noise-a", "noise-b")):
        store.add_audio_bytes(
            project_id, noise_class, _noise_wav(4.0, seed=index), recording_session_id=session
        )
    for index, session in enumerate(("bg-a", "bg-b")):
        store.add_audio_bytes(
            project_id,
            background_class,
            _noise_wav(4.0, seed=10 + index),
            recording_session_id=session,
        )

    thresholds_by_id = {noise_class: 0.8}
    store.update_project(
        project_id,
        {
            "settings": {
                "class_thresholds": thresholds_by_id,
                "background_class_id": background_class,
                "background_weight": 0.5,
                "head_dropout": 0.25,
                "waveform_augment_level": "light",
                "deployment_target": "pc",
            }
        },
    )
    result = ks.train_known_sound_project(
        store,
        project_id,
        {"epochs": 3, "minimum_clips_per_class": 4, "mixup_ratio": 0.0},
        lambda *_: None,
    )
    settings = result["report"]["settings"]
    # The DICT, copied verbatim from project settings -- tm_local/mcu/contract.py reads it.
    assert settings["class_thresholds"] == thresholds_by_id
    assert settings["class_thresholds_by_label"] == [0.5, 0.8, 0.5]
    assert settings["background_weight_requested"] == 0.5
    assert settings["background_weight_applied"] == 0.5
    assert settings["background_class_index"] == 2
    assert settings["head_dropout"] == 0.25
    assert settings["waveform_augment_level"] == "light"
    assert settings["early_stopping"] is True
    assert settings["deployment_target"] == "pc"
    assert result["report"]["dataset"]["augmented_clips"] > 0
    # Unchanged contract other tasks depend on.
    assert result["report"]["encoder"]["depth"] == 14
    assert result["report"]["evaluation"]["thresholds"] == [0.5, 0.8, 0.5]

    store.set_training_result(
        project_id, report=result["report"], artifacts=result["artifacts"]
    )
    scored = ks.score_wav_bytes(store, project_id, _tone_wav(440.0, 1.0), "keras")
    assert scored["thresholds"] == [0.5, 0.8, 0.5]
    by_name = {item["class_name"]: item for item in scored["predictions"]}
    assert by_name["Class 2"]["detected"] == (by_name["Class 2"]["score"] >= 0.8)

    # Augmentation touched the TRAIN side only: "light" doubles the training clips
    # (1 copy per clip) while the validation pass still saw exactly the held-out clips.
    dataset = result["report"]["dataset"]
    train_rows, validation_rows = embedded_rows
    assert dataset["augmented_clips"] == dataset["train_clips"]
    assert train_rows == dataset["train_clips"] * 2
    assert validation_rows == dataset["validation_clips"]
    assert dataset["train_clips"] + dataset["validation_clips"] == dataset["total_clips"]

    # Preview refuses a report whose stored list no longer matches the class list rather
    # than quietly deciding with different thresholds than the exported model will.
    from tm_local.tflite_export import ExportError

    broken = json.loads(json.dumps(result["report"]))
    broken["settings"]["class_thresholds_by_label"] = [0.5, 0.8]
    store.set_training_result(project_id, report=broken, artifacts=result["artifacts"])
    with pytest.raises(ExportError, match="class_thresholds"):
        ks.score_wav_bytes(store, project_id, _tone_wav(440.0, 1.0), "keras")


@pytest.mark.slow
def test_known_sound_defaults_train_exactly_as_before(store: ProjectStore) -> None:
    """A project with none of the new keys must behave like the pre-tunables build."""

    pytest.importorskip("tensorflow")

    project = store.create_project("known_sound", "Legacy")
    project_id = project["id"]
    tone_class, noise_class = project["classes"][0]["id"], project["classes"][1]["id"]
    store.delete_class(project_id, project["classes"][2]["id"])
    for session in ("tone-a", "tone-b"):
        store.add_audio_bytes(
            project_id, tone_class, _tone_wav(440.0, 4.0), recording_session_id=session
        )
    for index, session in enumerate(("noise-a", "noise-b")):
        store.add_audio_bytes(
            project_id, noise_class, _noise_wav(4.0, seed=index), recording_session_id=session
        )
    result = ks.train_known_sound_project(
        store,
        project_id,
        {"epochs": 3, "minimum_clips_per_class": 4, "mixup_ratio": 0.0},
        lambda *_: None,
    )
    settings = result["report"]["settings"]
    assert settings["class_thresholds"] == {}
    assert settings["class_thresholds_by_label"] == [0.5, 0.5]
    assert settings["head_dropout"] == 0.0
    assert settings["waveform_augment_level"] == "off"
    assert settings["background_weight_requested"] == 1.0
    assert settings["background_weight_applied"] == 1.0
    assert settings["background_class_index"] is None
    assert result["report"]["dataset"]["augmented_clips"] == 0


@pytest.mark.slow
def test_background_weight_reaches_fit_and_spares_mixup_rows(
    store: ProjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weights must arrive at head.fit, and mixup rows must stay at 1.0."""

    tf = pytest.importorskip("tensorflow")

    project = store.create_project("known_sound", "Weighted")
    project_id = project["id"]
    tone_class, noise_class = project["classes"][0]["id"], project["classes"][1]["id"]
    background_class = project["classes"][2]["id"]
    for session in ("tone-a", "tone-b"):
        store.add_audio_bytes(
            project_id, tone_class, _tone_wav(440.0, 4.0), recording_session_id=session
        )
    for index, session in enumerate(("noise-a", "noise-b")):
        store.add_audio_bytes(
            project_id, noise_class, _noise_wav(4.0, seed=index), recording_session_id=session
        )
    for index, session in enumerate(("bg-a", "bg-b")):
        store.add_audio_bytes(
            project_id,
            background_class,
            _noise_wav(4.0, seed=20 + index),
            recording_session_id=session,
        )
    store.update_project(
        project_id,
        {"settings": {"background_class_id": background_class, "background_weight": 0.5}},
    )

    captured: dict[str, object] = {}
    original_fit = tf.keras.Model.fit

    def spy_fit(self, *args, **kwargs):
        captured["sample_weight"] = kwargs.get("sample_weight")
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(tf.keras.Model, "fit", spy_fit)
    result = ks.train_known_sound_project(
        store,
        project_id,
        {"epochs": 1, "minimum_clips_per_class": 4, "mixup_ratio": 0.5},
        lambda *_: None,
    )

    dataset = result["report"]["dataset"]
    assert dataset["mixup_clips"] > 0
    weights = np.asarray(captured["sample_weight"], dtype=np.float32)
    real_rows = dataset["train_clips"] + dataset["augmented_clips"]
    assert weights.shape == (real_rows + dataset["mixup_clips"],)
    # Real background clips carry the weight; every other real clip stays at 1.0.
    assert set(np.unique(weights[:real_rows]).tolist()) == {0.5, 1.0}
    # Mixed rows are multi-hot, so weighting them would scale the target class too.
    assert np.all(weights[real_rows:] == 1.0)


@pytest.mark.slow
def test_head_dropout_trains_and_exports_a_strict_integer_classifier(
    store: ProjectStore,
) -> None:
    """Task 7 Part A group 3: head_dropout (build_known_sound_model) inserts a real
    Dropout layer between the frozen embedding and the trainable head. Dropout is a
    no-op at inference, but the strict-INT8 contract (CLAUDE.md decision 3, enforced by
    tflite_export.inspect_tflite() through ensure_export_artifacts) is a hard export gate
    to verify directly rather than assume holds with a new layer in the graph. This
    mirrors test_known_sound.py's _build_trained_known_sound_project /
    test_export_produces_strict_integer_classifier_and_working_runner recipe -- the
    proven path through a real int8 export that clears the quantisation-parity gate --
    with head_dropout added and a smaller encoder_depth for speed.
    """

    pytest.importorskip("tensorflow")
    from tm_local.export_service import ensure_export_artifacts

    project = store.create_project("known_sound", "DropoutExport")
    project_id = project["id"]
    tone_class, noise_class = project["classes"][0]["id"], project["classes"][1]["id"]
    store.delete_class(project_id, project["classes"][2]["id"])
    for session in ("tone-a", "tone-b"):
        store.add_audio_bytes(
            project_id, tone_class, _tone_wav(440.0, 11.0), recording_session_id=session
        )
    for index, session in enumerate(("noise-a", "noise-b")):
        store.add_audio_bytes(
            project_id, noise_class, _noise_wav(11.0, seed=index), recording_session_id=session
        )

    result = ks.train_known_sound_project(
        store,
        project_id,
        {"epochs": 120, "encoder_depth": 6, "head_dropout": 0.3},
        lambda *_: None,
    )
    assert result["report"]["settings"]["head_dropout"] == 0.3
    store.set_training_result(project_id, report=result["report"], artifacts=result["artifacts"])

    ensure_export_artifacts(store, project_id, ["int8"], lambda *_: None)
    project = store.get_raw_project(project_id)
    audit = project["training"]["report"]["conversion"]["models"]["int8"]

    # Strict full-integer contract: integer I/O, no float tensors, no Flex/custom ops --
    # unaffected by a Dropout layer sitting upstream of the trained head.
    assert audit["strict_full_integer"] is True
    assert audit["integer_input_output"] is True
    assert audit["float_tensor_count"] == 0
    assert audit["flex_operators"] == []
    assert audit["custom_operators"] == []
    # The decision must survive quantisation -- this is the gate that matters.
    assert audit["comparison"]["detection_agreement"] >= 0.95
