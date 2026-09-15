"""Tests for the known_sound kind: frozen YAMNet encoder + multi-label sigmoid head.

The pure-logic tests (session split, data gates, mixup) carry most of the correctness
weight and run without TensorFlow. The TensorFlow-backed tests are marked slow-ish but
still run in the normal suite because the head is tiny and the encoder runs once.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tm_local import config
from tm_local.audio_frontend import encode_wav_bytes
from tm_local.known_sound_pipeline import (
    ClipRef,
    build_mixup_examples,
    check_data_gates,
    split_sessions,
)
from tm_local.project_store import ProjectStore
from tm_local.training_common import TrainingError


# ----------------------------------------------------------------------------------
# kind registry
# ----------------------------------------------------------------------------------


def test_known_sound_is_registered_as_a_kind() -> None:
    assert "known_sound" in config.PROJECT_KINDS
    assert config.is_known_kind("known_sound")
    assert config.is_audio_like("known_sound")
    assert config.is_classifier("known_sound")
    assert config.is_multi_label("known_sound")
    assert config.PROJECT_KIND_LABELS["known_sound"] == "Known Sound Project"
    defaults = config.defaults_for_kind("known_sound")
    assert defaults["encoder_backend"] == "yamnet_embedding"
    assert defaults is not config.KNOWN_SOUND_DEFAULTS  # fresh mutable copy


def test_abnormal_sound_is_audio_like_but_not_a_classifier() -> None:
    """The two questions are distinct; conflating them is what the registry prevents."""
    assert config.is_audio_like("abnormal_sound")
    assert not config.is_classifier("abnormal_sound")
    assert not config.is_multi_label("audio")


def test_unknown_kind_still_raises() -> None:
    with pytest.raises(ValueError):
        config.defaults_for_kind("no_such_kind")


def test_known_sound_settings_validation() -> None:
    base = config.defaults_for_kind("known_sound")
    assert config.validate_known_sound_settings(dict(base)) is not None
    for key, bad in (
        ("sample_rate", 44100),
        ("clip_seconds", 2.0),
        ("hop_seconds", 1.0),
        ("detection_threshold", 0.0),
        ("detection_threshold", 1.0),
        ("mixup_ratio", -0.1),
        ("minimum_sessions_per_class", 1),
        ("epochs", 0),
        ("learning_rate", 0.0),
    ):
        with pytest.raises(ValueError):
            config.validate_known_sound_settings({**base, key: bad})


# ----------------------------------------------------------------------------------
# session-disjoint split
# ----------------------------------------------------------------------------------


def _clips(spec: dict[int, dict[str, int]]) -> list[ClipRef]:
    """spec = {class_index: {session_id: clip_count}}."""
    out: list[ClipRef] = []
    for class_index, sessions in spec.items():
        for session_id, count in sessions.items():
            for n in range(count):
                out.append(
                    ClipRef(
                        path=Path(f"{session_id}_{n}.wav"),
                        class_index=class_index,
                        session_id=session_id,
                    )
                )
    return out


def test_split_never_lets_a_session_cross_the_boundary() -> None:
    clips = _clips({0: {"a": 10, "b": 10, "c": 10}, 1: {"d": 10, "e": 10}})
    split = split_sessions(clips, val_fraction=0.25)
    assert set(split.train_sessions).isdisjoint(split.validation_sessions)
    train_ids = {clip.session_id for clip in split.train_clips}
    val_ids = {clip.session_id for clip in split.validation_clips}
    assert train_ids.isdisjoint(val_ids)


def test_split_is_deterministic() -> None:
    clips = _clips({0: {"a": 5, "b": 5, "c": 5}, 1: {"d": 5, "e": 5, "f": 5}})
    first = split_sessions(clips, val_fraction=0.25)
    second = split_sessions(list(reversed(clips)), val_fraction=0.25)
    assert first.train_sessions == second.train_sessions
    assert first.validation_sessions == second.validation_sessions


def test_every_class_keeps_at_least_one_session_on_each_side() -> None:
    clips = _clips({0: {"a": 4, "b": 4}, 1: {"c": 4, "d": 4}, 2: {"e": 4, "f": 4}})
    split = split_sessions(clips, val_fraction=0.25)
    for class_index in (0, 1, 2):
        assert any(c.class_index == class_index for c in split.train_clips)
        assert any(c.class_index == class_index for c in split.validation_clips)


# ----------------------------------------------------------------------------------
# data gates
# ----------------------------------------------------------------------------------


def test_gate_rejects_a_single_session_per_class_and_names_the_class() -> None:
    clips = _clips({0: {"a": 40}, 1: {"b": 40}})
    with pytest.raises(TrainingError) as excinfo:
        check_data_gates(
            clips, ["Gunshot", "Background"], minimum_sessions=2, minimum_clips=20
        )
    message = str(excinfo.value)
    assert "Gunshot" in message and "Background" in message
    assert "session" in message.lower()


def test_gate_rejects_too_few_clips_and_names_the_shortfall() -> None:
    clips = _clips({0: {"a": 3, "b": 3}, 1: {"c": 30, "d": 30}})
    with pytest.raises(TrainingError) as excinfo:
        check_data_gates(clips, ["Sparse", "Plenty"], minimum_sessions=2, minimum_clips=20)
    message = str(excinfo.value)
    assert "Sparse" in message
    assert "Plenty" not in message  # only the failing class is reported


def test_gate_requires_at_least_two_classes() -> None:
    clips = _clips({0: {"a": 30, "b": 30}})
    with pytest.raises(TrainingError):
        check_data_gates(clips, ["Only"], minimum_sessions=2, minimum_clips=20)


def test_gate_passes_when_satisfied() -> None:
    clips = _clips({0: {"a": 20, "b": 20}, 1: {"c": 20, "d": 20}})
    check_data_gates(clips, ["A", "B"], minimum_sessions=2, minimum_clips=20)


# ----------------------------------------------------------------------------------
# mixup
# ----------------------------------------------------------------------------------


def test_mixup_targets_are_multi_hot_across_two_different_classes() -> None:
    waveforms = np.zeros((6, 16000), dtype=np.float32)
    waveforms[:3] = 0.2
    waveforms[3:] = 0.4
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    mixed_waves, mixed_targets = build_mixup_examples(
        waveforms, labels, class_count=2, count=8, seed=7
    )
    assert mixed_waves.shape == (8, 16000)
    assert mixed_targets.shape == (8, 2)
    # every mixup example activates exactly two distinct classes
    assert np.all(mixed_targets.sum(axis=1) == 2)
    assert set(np.unique(mixed_targets)) <= {0.0, 1.0}


def test_mixup_is_deterministic_for_a_seed() -> None:
    waveforms = np.random.RandomState(0).randn(8, 16000).astype(np.float32) * 0.1
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1], dtype=np.int64)
    a_waves, a_targets = build_mixup_examples(waveforms, labels, class_count=3, count=5, seed=3)
    b_waves, b_targets = build_mixup_examples(waveforms, labels, class_count=3, count=5, seed=3)
    assert np.array_equal(a_waves, b_waves)
    assert np.array_equal(a_targets, b_targets)


def test_mixup_does_not_clip() -> None:
    waveforms = np.ones((4, 16000), dtype=np.float32) * 0.9
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    mixed, _ = build_mixup_examples(waveforms, labels, class_count=2, count=4, seed=1)
    assert np.all(np.abs(mixed) <= 1.0)


def test_mixup_returns_empty_when_only_one_class_present() -> None:
    waveforms = np.zeros((4, 16000), dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)
    mixed, targets = build_mixup_examples(waveforms, labels, class_count=1, count=4, seed=1)
    assert mixed.shape[0] == 0
    assert targets.shape[0] == 0


def test_mixup_count_zero_disables_it() -> None:
    waveforms = np.zeros((4, 16000), dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    mixed, targets = build_mixup_examples(waveforms, labels, class_count=2, count=0, seed=1)
    assert mixed.shape[0] == 0 and targets.shape[0] == 0


# ----------------------------------------------------------------------------------
# project store
# ----------------------------------------------------------------------------------


def test_create_known_sound_project_uses_generic_class_names(store: ProjectStore) -> None:
    project = store.create_project("known_sound", "Known Sound")
    assert project["kind"] == "known_sound"
    names = [item["name"] for item in project["classes"]]
    assert names == ["Class 1", "Class 2", "Background"]
    # every class is editable and equal -- no locked container like abnormal_sound
    assert all(not item.get("locked", False) for item in project["classes"])
    assert all("role" not in item or item["role"] is None for item in project["classes"])


def test_known_sound_settings_survive_update(store: ProjectStore) -> None:
    project = store.create_project("known_sound", "Known Sound")
    updated = store.update_project(project["id"], {"settings": {"detection_threshold": 0.7}})
    assert updated["settings"]["detection_threshold"] == pytest.approx(0.7)


# ----------------------------------------------------------------------------------
# class map asset (recording sanity check)
# ----------------------------------------------------------------------------------


def test_class_map_loads_and_pins_the_expected_indices() -> None:
    from tm_local.yamnet_model import load_class_map

    names = load_class_map()
    assert len(names) == 521
    assert names[0] == "Speech"
    assert names[69] == "Dog"
    assert names[70] == "Bark"
    assert names[421] == "Gunshot, gunfire"
    assert names[435] == "Glass"
    assert names[437] == "Shatter"


def test_class_map_missing_raises_its_own_error(tmp_path: Path) -> None:
    from tm_local.yamnet_model import YamnetClassMapError, class_map_available, load_class_map

    missing = tmp_path / "absent.csv"
    with pytest.raises(YamnetClassMapError):
        load_class_map(missing)
    assert class_map_available(missing) is False


def test_class_map_tamper_is_detected(tmp_path: Path) -> None:
    from tm_local.yamnet_model import YamnetClassMapError, load_class_map

    tampered = tmp_path / "tampered.csv"
    tampered.write_text("index,mid,display_name\n0,/m/09x0r,Speech\n", encoding="utf-8")
    with pytest.raises(YamnetClassMapError):
        load_class_map(tampered)


# ----------------------------------------------------------------------------------
# HTTP API
# ----------------------------------------------------------------------------------


def _client(store: ProjectStore):
    from fastapi.testclient import TestClient

    from tm_local.app import create_app
    from tm_local.jobs import JobManager
    from tm_local.model_runtime import ModelRuntime

    return TestClient(
        create_app(store=store, jobs=JobManager(max_workers=1), runtime=ModelRuntime())
    )


def test_api_creates_known_sound_project(store: ProjectStore) -> None:
    with _client(store) as client:
        created = client.post("/api/projects", json={"kind": "known_sound", "name": "API KS"})
        assert created.status_code == 200, created.text
        project = created.json()
        assert project["kind"] == "known_sound"
        assert [item["name"] for item in project["classes"]] == ["Class 1", "Class 2", "Background"]


def test_api_rejects_unknown_kind(store: ProjectStore) -> None:
    with _client(store) as client:
        assert client.post("/api/projects", json={"kind": "nope"}).status_code == 422


def test_api_predict_requires_training_and_right_kind(
    store: ProjectStore, wav_bytes: bytes
) -> None:
    with _client(store) as client:
        known = client.post("/api/projects", json={"kind": "known_sound"}).json()
        files = {"file": ("a.wav", wav_bytes, "audio/wav")}
        # trained check comes before anything expensive
        response = client.post(f"/api/projects/{known['id']}/predict/known-sound", files=files)
        assert response.status_code == 409

        audio = client.post("/api/projects", json={"kind": "audio"}).json()
        wrong = client.post(
            f"/api/projects/{audio['id']}/predict/known-sound",
            files={"file": ("a.wav", wav_bytes, "audio/wav")},
        )
        assert wrong.status_code == 400


@pytest.mark.slow
def test_api_audio_sanity_check_names_the_sound(store: ProjectStore) -> None:
    """The check that would have caught 'you recorded speech, not gunshots'."""
    with _client(store) as client:
        project = client.post("/api/projects", json={"kind": "known_sound"}).json()
        silence = encode_wav_bytes(np.zeros(16000, dtype=np.float32), 16000)
        response = client.post(
            f"/api/projects/{project['id']}/audio-sanity",
            files={"file": ("silence.wav", silence, "audio/wav")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["available"] is True
        assert len(body["top"]) == 3
        # Digital silence is the one input whose AudioSet answer is not in doubt.
        assert body["top"][0]["name"] == "Silence"
        assert body["top"][0]["score"] > 0.5


def test_api_audio_sanity_check_rejects_image_projects(
    store: ProjectStore, wav_bytes: bytes
) -> None:
    with _client(store) as client:
        image = client.post("/api/projects", json={"kind": "image"}).json()
        response = client.post(
            f"/api/projects/{image['id']}/audio-sanity",
            files={"file": ("a.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 400


# ----------------------------------------------------------------------------------
# model + end to end (TensorFlow)
# ----------------------------------------------------------------------------------


def test_encoder_depth_maps_to_the_official_block_widths() -> None:
    from tm_local.yamnet_model import (
        YAMNET_BLOCK_COUNT,
        encoder_layer_name,
        encoder_output_dim,
    )

    assert YAMNET_BLOCK_COUNT == 14
    # Widths of YAMNet's MobileNetV1 blocks; the back-loading is why truncating pays off.
    assert encoder_output_dim(6) == 256
    assert encoder_output_dim(10) == 512
    assert encoder_output_dim(13) == 1024
    assert encoder_output_dim(14) == 1024
    assert encoder_layer_name(10) == "layer10_pointwise_relu"
    for bad in (0, 1, 15, 99):
        with pytest.raises(ValueError):
            encoder_output_dim(bad)
        with pytest.raises(ValueError):
            encoder_layer_name(bad)


def test_encoder_depth_defaults_to_the_full_encoder() -> None:
    """Existing projects must keep behaving exactly as before this setting existed."""
    assert config.KNOWN_SOUND_DEFAULTS["encoder_depth"] == 14
    assert config.defaults_for_kind("known_sound")["encoder_depth"] == 14


def test_encoder_depth_is_validated() -> None:
    base = config.defaults_for_kind("known_sound")
    for good in (2, 6, 10, 14):
        config.validate_known_sound_settings({**base, "encoder_depth": good})
    for bad in (1, 0, -3, 15):
        with pytest.raises(ValueError):
            config.validate_known_sound_settings({**base, "encoder_depth": bad})


@pytest.mark.slow
@pytest.mark.parametrize("depth,expected_dim", [(6, 256), (10, 512), (14, 1024)])
def test_truncated_encoder_builds_with_the_right_width(depth: int, expected_dim: int) -> None:
    from tm_local.known_sound_pipeline import build_known_sound_model

    model = build_known_sound_model(class_count=3, encoder_depth=depth)
    assert model.output_shape[-1] == 3
    assert model.get_layer("embedding").output_shape[-1] == expected_dim
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    assert trainable == expected_dim * 3 + 3
    # A shallower encoder must be strictly smaller -- that is the entire point.
    total = int(sum(np.prod(w.shape) for w in model.weights))
    if depth < 14:
        full = build_known_sound_model(class_count=3, encoder_depth=14)
        assert total < int(sum(np.prod(w.shape) for w in full.weights))
    scores = model.predict(np.zeros((2, 96, 64), dtype=np.float32), verbose=0)
    assert scores.shape == (2, 3)


def test_head_is_multi_label_and_encoder_is_frozen() -> None:
    from tm_local.known_sound_pipeline import build_known_sound_model

    model = build_known_sound_model(class_count=4)
    assert model.output_shape[-1] == 4
    # only the head trains: 1024 * 4 + 4
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    assert trainable == 1024 * 4 + 4
    scores = model.predict(np.zeros((2, 96, 64), dtype=np.float32), verbose=0)
    assert scores.shape == (2, 4)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    # independent sigmoids: nothing forces a sum of 1
    assert not np.allclose(scores.sum(axis=1), 1.0)


def _tone_wav(frequency: float, seconds: float = 11.0, amplitude: float = 0.3) -> bytes:
    sample_rate = 16000
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    signal = amplitude * np.sin(2.0 * math.pi * frequency * t)
    return encode_wav_bytes(signal.astype(np.float32), sample_rate)


def _noise_wav(seconds: float = 11.0, seed: int = 0) -> bytes:
    rng = np.random.RandomState(seed)
    signal = (rng.randn(int(16000 * seconds)) * 0.15).astype(np.float32)
    return encode_wav_bytes(np.clip(signal, -1.0, 1.0), 16000)


@pytest.mark.slow
def test_train_preview_end_to_end(store: ProjectStore) -> None:
    """Two acoustically distinct classes, two sessions each, must train and separate."""
    from tm_local.known_sound_pipeline import (
        score_wav_bytes,
        train_known_sound_project,
    )

    project = store.create_project("known_sound", "E2E")
    project_id = project["id"]
    tone_class, noise_class = project["classes"][0]["id"], project["classes"][1]["id"]
    # drop the third (Background) class so the fixture stays small
    store.delete_class(project_id, project["classes"][2]["id"])

    for session in ("tone-a", "tone-b"):
        store.add_audio_bytes(
            project_id, tone_class, _tone_wav(440.0), recording_session_id=session
        )
    for index, session in enumerate(("noise-a", "noise-b")):
        store.add_audio_bytes(
            project_id, noise_class, _noise_wav(seed=index), recording_session_id=session
        )

    result = train_known_sound_project(store, project_id, {"epochs": 30}, lambda *_: None)
    # JobManager persists the result in production; do the same so Preview can load it.
    store.set_training_result(
        project_id, report=result["report"], artifacts=result["artifacts"]
    )
    report = result["report"]
    assert report["project_kind"] == "known_sound"
    assert report["detector_kind"] == "multi_label_classifier"
    assert report["encoder"]["frozen"] is True
    # session-disjoint validation actually happened
    assert set(report["dataset"]["train_sessions"]).isdisjoint(
        report["dataset"]["validation_sessions"]
    )

    scores = score_wav_bytes(store, project_id, _tone_wav(440.0, seconds=1.0), "keras")
    assert len(scores["predictions"]) == 2
    assert scores["multi_label"] is True
    by_name = {item["class_name"]: item["score"] for item in scores["predictions"]}
    assert by_name["Class 1"] > by_name["Class 2"]


def _build_trained_known_sound_project(store: ProjectStore, encoder_depth: int = 14) -> str:
    from tm_local.known_sound_pipeline import train_known_sound_project

    project = store.create_project("known_sound", "Export E2E")
    project_id = project["id"]
    tone_class, noise_class = project["classes"][0]["id"], project["classes"][1]["id"]
    store.delete_class(project_id, project["classes"][2]["id"])
    for session in ("tone-a", "tone-b"):
        store.add_audio_bytes(
            project_id, tone_class, _tone_wav(440.0), recording_session_id=session
        )
    for index, session in enumerate(("noise-a", "noise-b")):
        store.add_audio_bytes(
            project_id, noise_class, _noise_wav(seed=index), recording_session_id=session
        )
    # Use the real default epoch count: the head trains on cached embeddings so this is
    # still seconds, and an undertrained head leaves scores in the mid-range where a
    # sigmoid is steepest and quantisation error is worst -- not representative.
    result = train_known_sound_project(
        store, project_id, {"epochs": 120, "encoder_depth": encoder_depth}, lambda *_: None
    )
    store.set_training_result(
        project_id, report=result["report"], artifacts=result["artifacts"]
    )
    return project_id


@pytest.mark.slow
def test_truncated_depth_trains_and_exports_a_smaller_named_artifact(
    store: ProjectStore,
) -> None:
    """Depth 10 must produce a materially smaller model and a distinguishable filename."""
    from tm_local.export_service import ensure_export_artifacts

    project_id = _build_trained_known_sound_project(store, encoder_depth=10)
    project = store.get_raw_project(project_id)
    report = project["training"]["report"]
    assert report["encoder"]["depth"] == 10
    assert report["encoder"]["truncated"] is True
    assert report["encoder"]["cut_layer"] == "layer10_pointwise_relu"
    assert report["encoder"]["embedding_dim"] == 512
    assert report["head"]["trainable_parameters"] == 512 * 2 + 2

    ensure_export_artifacts(store, project_id, ["int8"], lambda *_: None)
    project = store.get_raw_project(project_id)
    name = project["training"]["artifacts"]["int8"]
    assert "_d10" in name, name
    size = (store.project_dir(project_id) / "models" / name).stat().st_size
    # Depth 10 is measured at ~1.23 MB of source .tflite versus ~3.51 MB at full depth.
    assert size < 2_000_000, size


@pytest.mark.slow
def test_export_produces_strict_integer_classifier_and_working_runner(
    store: ProjectStore, tmp_path: Path
) -> None:
    """The whole point of the kind is a self-contained, strictly-quantised classifier."""
    import subprocess
    import sys
    import zipfile

    from tm_local.export_service import create_model_export, ensure_export_artifacts

    project_id = _build_trained_known_sound_project(store)
    ensure_export_artifacts(store, project_id, ["int8"], lambda *_: None)

    project = store.get_raw_project(project_id)
    conversion = project["training"]["report"]["conversion"]
    audit = conversion["models"]["int8"]

    # Strict full-integer contract: integer I/O, no float tensors, no Flex/custom ops.
    assert audit["strict_full_integer"] is True
    assert audit["integer_input_output"] is True
    assert audit["float_tensor_count"] == 0
    assert audit["flex_operators"] == []
    assert audit["custom_operators"] == []
    assert audit["inputs"][0]["dtype"] == "int8"
    assert audit["outputs"][0]["dtype"] == "int8"
    # One score per class, from a 96x64 YAMNet patch.
    assert list(audit["inputs"][0]["shape"])[-2:] == [96, 64]
    assert list(audit["outputs"][0]["shape"])[-1] == 2
    # The decision must survive quantisation -- this is the gate that matters.
    assert audit["comparison"]["detection_agreement"] >= 0.95

    # Package the ZIP and actually run the exported runner on a real WAV.
    output_zip = tmp_path / "known_sound.zip"
    create_model_export(
        store=store,
        project_id=project_id,
        selected=["int8"],
        include_c_header=False,
        output_zip=output_zip,
        progress=lambda *_: None,
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(output_zip) as archive:
        archive.extractall(extracted)
    names = {item.name for item in extracted.rglob("*")}
    assert "run_model.py" in names
    assert "labels.txt" in names
    assert "yamnet_frontend.json" in names
    assert "yamnet_frontend_reference.py" in names

    root = next(path for path in extracted.rglob("run_model.py")).parent
    wav_path = tmp_path / "probe.wav"
    wav_path.write_bytes(_tone_wav(440.0, seconds=1.0))
    completed = subprocess.run(
        [sys.executable, "run_model.py", str(wav_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Class 1" in completed.stdout
    assert "do not add up to 100%" in completed.stdout
