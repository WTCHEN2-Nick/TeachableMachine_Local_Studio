from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tm_local import config, image_pipeline
from tm_local.project_store import ProjectStore


def test_presets_cover_levels_and_medium_equals_legacy() -> None:
    assert set(image_pipeline.IMAGE_AUGMENTATION_PRESETS) == {"off", "light", "medium", "strong"}
    medium = image_pipeline.IMAGE_AUGMENTATION_PRESETS["medium"]
    assert medium == {"flip": True, "brightness": 0.08, "contrast": (0.85, 1.15), "zoom": 0.0,
                      "rotation_deg": 0.0, "hue": 0.0, "saturation": None}
    assert image_pipeline.IMAGE_AUGMENTATION_PRESETS["off"] is None


def test_presets_cover_exactly_the_settings_enum() -> None:
    """The validator accepts config.AUGMENTATION_LEVELS; every one must have a preset."""

    assert set(image_pipeline.IMAGE_AUGMENTATION_PRESETS) == set(config.AUGMENTATION_LEVELS)
    assert config.IMAGE_DEFAULTS["augmentation_level"] in image_pipeline.IMAGE_AUGMENTATION_PRESETS


def test_fine_tune_layer_names_picks_last_blocks_without_bn() -> None:
    names = ["Conv1", "bn_Conv1", "block_1_expand", "block_15_expand", "block_15_project_BN",
             "block_16_expand",
             "block_16_depthwise", "block_16_project", "block_16_project_BN", "Conv_1",
             "Conv_1_bn", "out_relu"]
    chosen = image_pipeline.fine_tune_layer_names(names, 1)
    assert chosen == {"block_16_expand", "block_16_depthwise", "block_16_project", "Conv_1",
                      "out_relu"}
    chosen2 = image_pipeline.fine_tune_layer_names(names, 2)
    assert "block_15_expand" in chosen2 and "block_15_project_BN" not in chosen2
    assert image_pipeline.fine_tune_layer_names(names, 0) == set()


def test_fine_tune_layer_names_is_pure_and_clamps() -> None:
    names = ["block_15_expand", "block_16_expand", "block_16_expand_BN", "Conv_1", "out_relu"]
    original = list(names)
    assert image_pipeline.fine_tune_layer_names(names, -3) == set()
    assert image_pipeline.fine_tune_layer_names(names, 99) == {
        "block_15_expand",
        "block_16_expand",
        "Conv_1",
        "out_relu",
    }
    assert names == original
    # No block_* layers at all (small_cnn) still yields only the tail layers it can see.
    assert image_pipeline.fine_tune_layer_names(["conv2d", "dense"], 2) == set()


def test_fine_tune_layer_names_never_unfreezes_batch_norm() -> None:
    names = [
        "block_16_expand",
        "block_16_expand_BN",
        "block_16_depthwise_BN",
        "block_16_project_BN",
        "bn_Conv1",
        "Conv_1_bn",
    ]
    chosen = image_pipeline.fine_tune_layer_names(names, 4)
    assert chosen == {"block_16_expand"}
    assert not any("bn" in name.lower() for name in chosen)


@pytest.mark.parametrize(
    "bad",
    [
        {"image_size": 224.7},
        {"fine_tune_blocks": 2.9},
        {"epochs": 30.9},
        {"batch_size": 16.5},
    ],
)
def test_integer_settings_reject_a_non_integer(bad: dict) -> None:
    """``int(2.9) == 2`` silently trained a different model from the one requested.

    Not reachable from the panel (these four controls are ``<select>``s and integer
    ``<input type="number">``s) but reachable over HTTP, where truncating is a repair no
    other branch of these validators performs.
    """

    with pytest.raises(ValueError, match="必須是整數"):
        config.validate_image_settings({**config.IMAGE_DEFAULTS, **bad})


def test_integer_settings_still_accept_integral_strings_and_floats() -> None:
    """Rejecting 224.7 must not also reject the request boundary's "224"."""

    result = config.validate_image_settings(
        {**config.IMAGE_DEFAULTS, "image_size": "224", "fine_tune_blocks": 2.0, "epochs": "30"}
    )
    assert result["image_size"] == 224 and isinstance(result["image_size"], int)
    assert result["fine_tune_blocks"] == 2 and isinstance(result["fine_tune_blocks"], int)
    assert result["epochs"] == 30 and isinstance(result["epochs"], int)


@pytest.mark.slow
def test_build_model_dropout_and_base_handle() -> None:
    pytest.importorskip("tensorflow")
    model, backbone, base = image_pipeline._build_model(96, 3, "small_cnn", 0.35, None, dropout=0.45)
    dropouts = [layer for layer in model.layers if layer.__class__.__name__ == "Dropout"]
    assert dropouts and abs(dropouts[0].rate - 0.45) < 1e-9
    assert base is None and backbone == "small_cnn"


@pytest.mark.slow
def test_build_model_graph_has_no_augmentation_layers_and_stable_signature(tmp_path) -> None:
    """Augmentation lives in tf.data only; the exported graph must be unchanged."""

    tf = pytest.importorskip("tensorflow")
    from tm_local.tflite_export import export_inference_saved_model

    model, _backbone, _base = image_pipeline._build_model(96, 2, "small_cnn", 0.35, None,
                                                          dropout=0.5)
    assert not any(
        layer.__class__.__name__.startswith("Random") for layer in model.layers
    )
    assert next(layer.name for layer in model.layers) == "image"
    assert model.layers[-1].name == "scores"

    saved = export_inference_saved_model(model, tmp_path / "saved_model")
    loaded = tf.saved_model.load(str(saved))
    signature = loaded.signatures["serving_default"]
    (spec,) = list(signature.structured_input_signature[1].values())
    assert tuple(spec.shape.as_list()) == (None, 96, 96, 3)
    assert spec.dtype == tf.float32
    assert set(signature.structured_outputs) == {"scores"}


@pytest.mark.slow
def test_augmented_dataset_shapes(tmp_path) -> None:
    tf = pytest.importorskip("tensorflow")
    from PIL import Image

    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (40, 30), (index * 40, 10, 200)).save(path)
        paths.append(path)
    for level in ("off", "light", "medium", "strong"):
        ds = image_pipeline._make_dataset(paths, [0, 1, 0, 1], 64, 2, True, augmentation=level)
        images, labels = next(iter(ds))
        assert images.shape == (2, 64, 64, 3) and labels.shape == (2,)
        assert float(tf.reduce_max(images)) <= 255.0 and float(tf.reduce_min(images)) >= 0.0


def _gradient_png(path: Path) -> Path:
    """A non-uniform image: contrast/flip/zoom are no-ops on a flat colour."""

    array = np.zeros((48, 64, 3), dtype=np.uint8)
    array[..., 0] = np.linspace(0, 255, 64).astype(np.uint8)[None, :]
    array[..., 1] = np.linspace(255, 0, 48).astype(np.uint8)[:, None]
    array[..., 2] = 128
    Image.fromarray(array).save(path)
    return path


@pytest.mark.slow
@pytest.mark.parametrize("level", ["light", "medium", "strong"])
def test_augmentation_actually_changes_the_pixels(tmp_path, level: str) -> None:
    """Every enabled preset must perturb the batch; only "off" leaves it alone."""

    tf = pytest.importorskip("tensorflow")

    # One sample per dataset: the training pipeline shuffles, so a larger batch would
    # compare images in different orders.
    path = _gradient_png(tmp_path / "gradient.png")
    tf.keras.utils.set_random_seed(20260912)
    augmented = next(
        iter(image_pipeline._make_dataset([path], [0], 32, 1, True, augmentation=level))
    )[0]
    clean = next(iter(image_pipeline._make_dataset([path], [0], 32, 1, False)))[0]
    assert float(tf.reduce_max(tf.abs(augmented - clean))) > 1.0
    off = next(iter(image_pipeline._make_dataset([path], [0], 32, 1, True, augmentation="off")))[0]
    assert float(tf.reduce_max(tf.abs(off - clean))) == 0.0


@pytest.mark.slow
def test_augmentation_off_matches_the_validation_pipeline(tmp_path) -> None:
    """"off" must be a genuine no-op: identical pixels to the evaluation pipeline."""

    tf = pytest.importorskip("tensorflow")

    # One sample per dataset: the training pipeline shuffles, so anything larger would
    # compare batches in different orders.
    path = tmp_path / "only.png"
    Image.new("RGB", (40, 30), (90, 10, 200)).save(path)
    off = next(iter(image_pipeline._make_dataset([path], [0], 32, 1, True, augmentation="off")))
    evaluation = next(iter(image_pipeline._make_dataset([path], [0], 32, 1, False)))
    assert float(tf.reduce_max(tf.abs(off[0] - evaluation[0]))) == 0.0


@pytest.mark.slow
def test_unknown_augmentation_level_falls_back_to_medium(tmp_path) -> None:
    tf = pytest.importorskip("tensorflow")

    path = _gradient_png(tmp_path / "a.png")

    def batch(level: str):
        tf.keras.utils.set_random_seed(777)
        dataset = image_pipeline._make_dataset([path], [0], 32, 1, True, augmentation=level)
        return next(iter(dataset))[0]

    images = batch("nonsense")
    assert images.shape == (1, 32, 32, 3)
    # The fallback is "medium", not "off": the pixels must actually have been touched...
    clean = next(iter(image_pipeline._make_dataset([path], [0], 32, 1, False)))[0]
    assert float(tf.reduce_max(tf.abs(images - clean))) > 1.0
    # ... and it must be "medium" specifically, not some other enabled preset.
    assert float(tf.reduce_max(tf.abs(images - batch("medium")))) == 0.0
    assert float(tf.reduce_max(tf.abs(images - batch("strong")))) > 1.0


@pytest.mark.slow
def test_unknown_augmentation_level_is_normalised_in_the_report(store: ProjectStore) -> None:
    pytest.importorskip("tensorflow")

    project_id = _image_project(store, samples_per_class=3)
    result = image_pipeline.train_image_project(
        store,
        project_id,
        {"backbone": "small_cnn", "epochs": 1, "image_size": 96, "batch_size": 2,
         "minimum_samples_per_class": 2, "augmentation_level": "nonsense"},
        None,
    )
    assert result["report"]["settings"]["augmentation_level"] == "medium"


def _jpeg(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(output, "JPEG", quality=90)
    return output.getvalue()


def _image_project(store: ProjectStore, samples_per_class: int = 4) -> str:
    project = store.create_project("image", "tunables")
    project_id = project["id"]
    palette = [(200, 20, 20), (20, 20, 200)]
    for index, class_item in enumerate(project["classes"]):
        base = palette[index % len(palette)]
        for offset in range(samples_per_class):
            color = tuple(min(255, value + offset * 7) for value in base)
            store.add_image_bytes(project_id, class_item["id"], _jpeg(color))  # type: ignore[arg-type]
    return project_id


def _strip_settings(store: ProjectStore, project_id: str, keys: tuple[str, ...]) -> None:
    """Rewrite project.json as a pre-tunables project would have been stored."""

    path = store.project_dir(project_id) / "project.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        payload["settings"].pop(key, None)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[float, str]] = []

    def __call__(self, fraction: float, message: str) -> None:
        self.events.append((float(fraction), str(message)))

    def assert_monotonic(self) -> None:
        values = [fraction for fraction, _ in self.events]
        assert values == sorted(values), values
        assert values and 0.0 <= values[0] and values[-1] <= 1.0


@pytest.mark.slow
def test_legacy_project_without_new_keys_trains_with_todays_behaviour(
    store: ProjectStore,
) -> None:
    pytest.importorskip("tensorflow")

    project_id = _image_project(store)
    _strip_settings(
        store,
        project_id,
        ("augmentation_level", "dropout", "fine_tune_blocks", "deployment_target"),
    )
    recorder = _Recorder()
    result = image_pipeline.train_image_project(
        store,
        project_id,
        {"backbone": "small_cnn", "epochs": 2, "image_size": 96, "batch_size": 2,
         "minimum_samples_per_class": 2},
        recorder,
    )
    settings: dict[str, Any] = result["report"]["settings"]
    assert settings["augmentation_level"] == "medium"
    assert settings["dropout"] == pytest.approx(0.2)
    assert settings["fine_tune_blocks_requested"] == 0
    assert settings["fine_tune_blocks_used"] == 0
    assert settings["fine_tune_epochs_completed"] == 0
    assert settings["deployment_target"] == "pc"
    assert settings["early_stopping"] is True
    assert settings["epochs_completed"] == 2
    recorder.assert_monotonic()
    # fine_tune_blocks was never requested, so there is nothing to warn about.
    assert not any("略過" in message for _value, message in recorder.events)
    # No fine-tune phase: phase 1 owns the whole 0.10-0.67 band.
    training_events = [value for value, message in recorder.events if value < 0.69]
    assert max(training_events) == pytest.approx(0.67)


@pytest.mark.slow
def test_tunables_reach_the_report_and_small_cnn_cannot_fine_tune(store: ProjectStore) -> None:
    pytest.importorskip("tensorflow")

    project_id = _image_project(store)
    recorder = _Recorder()
    result = image_pipeline.train_image_project(
        store,
        project_id,
        {"backbone": "small_cnn", "epochs": 2, "image_size": 96, "batch_size": 2,
         "minimum_samples_per_class": 2, "augmentation_level": "strong", "dropout": 0.45,
         "fine_tune_blocks": 3, "deployment_target": "NuMaker-M55M1", "early_stopping": False},
        recorder,
    )
    settings = result["report"]["settings"]
    assert settings["augmentation_level"] == "strong"
    assert settings["dropout"] == pytest.approx(0.45)
    assert settings["fine_tune_blocks_requested"] == 3
    # small_cnn exposes no MobileNet base, so phase 2 must be skipped, not crash. The
    # report has to say 0 blocks were used, not echo the 3 that were asked for.
    assert settings["fine_tune_blocks_used"] == 0
    assert settings["fine_tune_epochs_completed"] == 0
    assert settings["deployment_target"] == "NuMaker-M55M1"
    assert settings["early_stopping"] is False
    recorder.assert_monotonic()
    assert max(value for value, _ in recorder.events if value < 0.69) == pytest.approx(0.67)
    # The student asked for fine-tuning and did not get it: say why, in Chinese.
    skipped = [message for _value, message in recorder.events if "略過" in message]
    assert len(skipped) == 1, recorder.events
    assert "fine_tune_blocks" in skipped[0] and "small_cnn" in skipped[0]


@pytest.mark.slow
def test_mobilenet_fine_tune_runs_a_second_phase(store: ProjectStore) -> None:
    pytest.importorskip("tensorflow")

    project_id = _image_project(store, samples_per_class=3)
    recorder = _Recorder()
    result = image_pipeline.train_image_project(
        store,
        project_id,
        {"backbone": "mobilenet_v2", "mobilenet_alpha": 0.35, "epochs": 2, "image_size": 224,
         "batch_size": 3, "minimum_samples_per_class": 2, "fine_tune_blocks": 1,
         "augmentation_level": "light"},
        recorder,
    )
    settings = result["report"]["settings"]
    if not str(settings["backbone_used"]).startswith("mobilenet_v2"):
        pytest.skip("ImageNet weights unavailable; the fine-tune phase cannot be exercised.")
    assert settings["fine_tune_blocks_requested"] == 1
    assert settings["fine_tune_blocks_used"] == 1
    assert settings["fine_tune_epochs_completed"] == max(1, 2 // 2)
    assert settings["epochs_completed"] == 2
    recorder.assert_monotonic()
    assert not any("略過" in message for _value, message in recorder.events)
    handover = next(
        index
        for index, (_value, message) in enumerate(recorder.events)
        if "微調最後" in message
    )
    assert recorder.events[handover][0] == pytest.approx(0.50)
    # Phase 1 stops at 0.50 so the fine-tune phase owns 0.50-0.67.
    assert max(value for value, _message in recorder.events[:handover]) == pytest.approx(0.50)
    phase_two = [value for value, _message in recorder.events[handover + 1:] if value < 0.69]
    assert max(phase_two) == pytest.approx(0.67)
    history = result["report"]["history"]
    assert len(history["loss"]) == 3  # 2 frozen epochs + 1 fine-tune epoch


def test_pipeline_keeps_the_native_keras_save_contract() -> None:
    source = Path(image_pipeline.__file__).read_text(encoding="utf-8")
    assert "save_native_keras_model(model, keras_path)" in source
    assert "include_optimizer=False" not in source
    assert "base(x, training=False)" in source
