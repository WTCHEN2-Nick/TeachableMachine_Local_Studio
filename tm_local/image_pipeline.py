from __future__ import annotations

import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .project_store import ProjectStore
from .training_common import (
    ProgressFn,
    TrainingError,
    class_weight_from_labels,
    history_to_json,
    make_keras_progress_callback,
    save_native_keras_model,
    stratified_path_split,
)
from .utils import utc_now_iso


def _bounded_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return fallback


def _bounded_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return fallback


def _load_image_array(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)


# Augmentation presets. "medium" is bit-for-bit the augmentation this pipeline has always
# applied, so a project stored before the tunable-parameters plan trains exactly as before.
# Augmentation only ever happens inside the tf.data input pipeline -- the saved model graph
# (and therefore the INT8 export signature) is identical for every level.
IMAGE_AUGMENTATION_PRESETS: dict[str, dict | None] = {
    "off": None,
    "light": {
        "flip": True,
        "brightness": 0.04,
        "contrast": (0.92, 1.08),
        "zoom": 0.0,
        "rotation_deg": 0.0,
        "hue": 0.0,
        "saturation": None,
    },
    "medium": {
        "flip": True,
        "brightness": 0.08,
        "contrast": (0.85, 1.15),
        "zoom": 0.0,
        "rotation_deg": 0.0,
        "hue": 0.0,
        "saturation": None,
    },
    "strong": {
        "flip": True,
        "brightness": 0.15,
        "contrast": (0.70, 1.30),
        "zoom": 0.15,
        "rotation_deg": 10.0,
        "hue": 0.03,
        "saturation": (0.8, 1.2),
    },
}

_BLOCK_NAME = re.compile(r"block_(\d+)_")


def fine_tune_layer_names(layer_names: Sequence[str], blocks: int) -> set[str]:
    """Return the MobileNetV2 layer names to unfreeze for ``blocks`` trailing blocks.

    Pure: it only reads ``layer_names``. BatchNormalization layers are never unfrozen --
    the head sees ``base(x, training=False)``, so their moving statistics stay frozen and
    letting their gamma/beta drift would silently break the INT8 calibration.
    """

    blocks = max(0, int(blocks))
    if blocks == 0:
        return set()
    block_ids = sorted(
        {
            int(match.group(1))
            for name in layer_names
            for match in [_BLOCK_NAME.match(name)]
            if match
        }
    )
    wanted = set(block_ids[-blocks:])
    chosen: set[str] = set()
    for name in layer_names:
        lower = name.lower()
        if lower.endswith("_bn") or "_bn_" in lower or lower.startswith("bn_"):
            continue
        match = _BLOCK_NAME.match(name)
        if (match and int(match.group(1)) in wanted) or name in {"Conv_1", "out_relu"}:
            chosen.add(name)
    return chosen


def _build_model(
    image_size: int,
    class_count: int,
    backbone: str,
    mobilenet_alpha: float,
    progress: ProgressFn | None,
    dropout: float = 0.2,
):
    """Build the classifier. Returns ``(model, actual_backbone, base_or_None)``.

    ``base`` is the MobileNetV2 feature extractor when one was actually loaded, and None
    for every small-CNN path -- fine-tuning is only possible in the former case.
    """

    import tensorflow as tf

    actual_backbone = backbone
    base = None
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), dtype=tf.float32, name="image")
    if backbone == "mobilenet_v2":
        try:
            base = tf.keras.applications.MobileNetV2(
                input_shape=(image_size, image_size, 3),
                include_top=False,
                weights="imagenet",
                alpha=mobilenet_alpha,
            )
            base.trainable = False
            x = tf.keras.layers.Rescaling(
                1.0 / 127.5, offset=-1.0, name="pixel_normalization"
            )(inputs)
            x = base(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
            x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
        except Exception as exc:
            base = None
            actual_backbone = "small_cnn_fallback"
            if progress:
                progress(
                    0.08,
                    "ImageNet MobileNetV2 weights are unavailable; using the built-in small CNN. "
                    f"({type(exc).__name__})",
                )
            x = tf.keras.layers.Rescaling(1.0 / 255.0, name="pixel_normalization")(inputs)
            x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(dropout)(x)
    else:
        actual_backbone = "small_cnn"
        x = tf.keras.layers.Rescaling(1.0 / 255.0, name="pixel_normalization")(inputs)
        x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="scores")(x)
    if actual_backbone == "mobilenet_v2":
        actual_backbone = f"mobilenet_v2_alpha_{mobilenet_alpha:g}"
    return tf.keras.Model(inputs, outputs, name="tm_local_image_classifier"), actual_backbone, base


def _make_dataset(
    paths: list[Path],
    labels: list[int],
    image_size: int,
    batch_size: int,
    training: bool,
    augmentation: str = "medium",
):
    import tensorflow as tf

    preset = (
        IMAGE_AUGMENTATION_PRESETS.get(augmentation, IMAGE_AUGMENTATION_PRESETS["medium"])
        if training
        else None
    )
    rotation = None
    if preset and preset["rotation_deg"] > 0:
        rotation = tf.keras.layers.RandomRotation(
            preset["rotation_deg"] / 360.0, fill_mode="reflect"
        )
    string_paths = [str(path) for path in paths]
    dataset = tf.data.Dataset.from_tensor_slices((string_paths, labels))
    if training:
        dataset = dataset.shuffle(max(32, len(paths)), reshuffle_each_iteration=True)

    def decode(path, label):  # noqa: ANN001
        data = tf.io.read_file(path)
        image = tf.io.decode_image(data, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        if preset and preset["zoom"] > 0:
            enlarged = round(image_size * (1.0 + preset["zoom"]))
            image = tf.image.resize(image, [enlarged, enlarged], antialias=True)
            image = tf.image.random_crop(image, [image_size, image_size, 3])
        else:
            image = tf.image.resize(image, [image_size, image_size], antialias=True)
        image = tf.cast(image, tf.float32)
        if preset:
            if preset["flip"]:
                image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, preset["brightness"] * 255.0)
            image = tf.image.random_contrast(image, preset["contrast"][0], preset["contrast"][1])
            if preset["hue"] > 0:
                image = tf.image.random_hue(image / 255.0, preset["hue"]) * 255.0
            if preset["saturation"]:
                image = (
                    tf.image.random_saturation(
                        image / 255.0, preset["saturation"][0], preset["saturation"][1]
                    )
                    * 255.0
                )
            if rotation is not None:
                image = rotation(image[None, ...], training=True)[0]
            image = tf.clip_by_value(image, 0.0, 255.0)
        return image, tf.cast(label, tf.int32)

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def write_image_runner(
    models_dir: Path,
    image_size: int,
    model_filename: str = "image_classifier_int8.tflite",
) -> None:
    script = f'''from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

MODEL = Path(__file__).with_name("{model_filename}")
LABELS = [line.split(" ", 1)[1].strip() for line in Path(__file__).with_name("labels.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
IMAGE_SIZE = {image_size}


def main(path: str) -> None:
    image = ImageOps.fit(Image.open(path).convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BILINEAR)
    value = np.asarray(image, dtype=np.float32)[None, ...]
    interpreter = tf.lite.Interpreter(model_path=str(MODEL))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    scale, zero = input_detail["quantization"]
    if np.issubdtype(input_detail["dtype"], np.integer):
        info = np.iinfo(input_detail["dtype"])
        value = np.clip(np.round(value / scale + zero), info.min, info.max).astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], value)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_detail["index"])[0]
    if np.issubdtype(output_detail["dtype"], np.integer):
        out_scale, out_zero = output_detail["quantization"]
        scores = (scores.astype(np.float32) - out_zero) * out_scale
    for label, score in sorted(zip(LABELS, scores), key=lambda item: item[1], reverse=True):
        print(f"{{label}}: {{score:.4f}}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_model.py image.jpg")
    main(sys.argv[1])
'''
    (models_dir / "run_model.py").write_text(script, encoding="utf-8")


def image_representative_samples(
    store: ProjectStore,
    project_id: str,
    image_size: int,
    *,
    limit: int = 100,
) -> list[np.ndarray]:
    """Load a balanced representative image set for post-training quantization."""

    class_paths = store.sample_paths_by_class(project_id)
    selected: list[Path] = []
    max_count = max((len(paths) for _, paths in class_paths), default=0)
    for offset in range(max_count):
        for _, paths in class_paths:
            if offset < len(paths):
                selected.append(paths[offset])
                if len(selected) >= max(1, int(limit)):
                    return [_load_image_array(path, image_size) for path in selected]
    return [_load_image_array(path, image_size) for path in selected]


def train_image_project(
    store: ProjectStore,
    project_id: str,
    options: dict[str, Any] | None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    options = dict(options or {})
    project = store.get_raw_project(project_id)
    if project["kind"] != "image":
        raise TrainingError("Project is not an image project.")
    settings = {**project["settings"], **options}
    image_size = _bounded_int(settings.get("image_size"), 96, 320, 224)
    epochs = _bounded_int(settings.get("epochs"), 1, 300, 30)
    batch_size = _bounded_int(settings.get("batch_size"), 1, 128, 16)
    learning_rate = _bounded_float(settings.get("learning_rate"), 1e-6, 0.1, 0.001)
    validation_split = _bounded_float(settings.get("validation_split"), 0.0, 0.45, 0.2)
    backbone = str(settings.get("backbone", "mobilenet_v2"))
    requested_alpha = _bounded_float(settings.get("mobilenet_alpha"), 0.35, 1.0, 0.35)
    mobilenet_alpha = min(
        (0.35, 0.50, 0.75, 1.0), key=lambda value: abs(value - requested_alpha)
    )
    minimum = _bounded_int(settings.get("minimum_samples_per_class"), 2, 100, 5)
    augmentation_level = str(settings.get("augmentation_level", "medium"))
    if augmentation_level not in IMAGE_AUGMENTATION_PRESETS:
        # _make_dataset falls back to "medium" anyway; normalize here too so the training
        # report never claims a level that was not actually applied.
        augmentation_level = "medium"
    dropout = _bounded_float(settings.get("dropout"), 0.0, 0.6, 0.2)
    fine_tune_blocks = _bounded_int(settings.get("fine_tune_blocks"), 0, 4, 0)
    early_stopping = bool(settings.get("early_stopping", True))

    class_paths = store.sample_paths_by_class(project_id)
    shortages = [
        f"{class_item['name']} ({len(paths)}/{minimum})"
        for class_item, paths in class_paths
        if len(paths) < minimum
    ]
    if shortages:
        raise TrainingError("Each class needs more image samples: " + ", ".join(shortages))
    if progress:
        progress(0.02, "Preparing image dataset…")
    split = stratified_path_split(class_paths, validation_split)
    if not split.train_paths:
        raise TrainingError("Training dataset is empty.")

    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(1337)
    model, actual_backbone, base = _build_model(
        image_size, len(class_paths), backbone, mobilenet_alpha, progress, dropout=dropout
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    train_ds = _make_dataset(
        split.train_paths,
        split.train_labels,
        image_size,
        batch_size,
        True,
        augmentation=augmentation_level,
    )
    validation_ds = None
    if split.validation_paths:
        validation_ds = _make_dataset(
            split.validation_paths,
            split.validation_labels,
            image_size,
            batch_size,
            False,
        )
    can_fine_tune = bool(fine_tune_blocks) and base is not None
    if fine_tune_blocks and not can_fine_tune and progress:
        # small_cnn, or MobileNetV2 whose ImageNet weights could not be loaded: there is no
        # pretrained backbone to unfreeze. Say so instead of silently ignoring the setting.
        progress(
            0.09,
            f"此 backbone（{actual_backbone}）不支援微調，已略過 fine_tune_blocks="
            f"{fine_tune_blocks} 設定。微調只適用於 MobileNetV2。",
        )
    phase_one_end = 0.50 if can_fine_tune else 0.67
    callbacks: list[Any] = [
        make_keras_progress_callback(epochs, progress, start=0.10, end=phase_one_end)
    ]
    monitor = "val_loss" if validation_ds is not None else "loss"
    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=max(3, min(10, epochs // 4)),
                restore_best_weights=True,
            )
        )
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        class_weight=class_weight_from_labels(split.train_labels, len(class_paths)),
    )
    # Captured before the fine-tune phase appends to history: "epochs_completed" stays
    # comparable with "epochs_requested", and the extra epochs are reported separately as
    # "fine_tune_epochs_completed". The history curve below covers both phases.
    epochs_completed = len(history.history.get("loss", []))
    fine_tune_epochs_completed = 0
    if can_fine_tune:
        if progress:
            progress(0.50, f"微調最後 {fine_tune_blocks} 個 MobileNet block…")
        chosen = fine_tune_layer_names([layer.name for layer in base.layers], fine_tune_blocks)
        base.trainable = True
        for layer in base.layers:
            layer.trainable = layer.name in chosen
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_epochs = max(1, epochs // 2)
        fine_callbacks: list[Any] = [
            make_keras_progress_callback(fine_epochs, progress, start=0.50, end=0.67)
        ]
        if early_stopping:
            fine_callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=max(3, min(10, fine_epochs // 4)),
                    restore_best_weights=True,
                )
            )
        fine_history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=fine_epochs,
            verbose=0,
            callbacks=fine_callbacks,
            class_weight=class_weight_from_labels(split.train_labels, len(class_paths)),
        )
        fine_tune_epochs_completed = len(fine_history.history.get("loss", []))
        for key, values in fine_history.history.items():
            history.history.setdefault(key, []).extend(values)

    evaluation_ds = validation_ds if validation_ds is not None else train_ds
    evaluation = model.evaluate(evaluation_ds, verbose=0, return_dict=True)
    if progress:
        progress(0.69, "Saving trained image model…")

    project_dir = store.project_dir(project_id)
    models_dir = project_dir / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)
    keras_path = models_dir / "image_classifier.keras"
    save_native_keras_model(model, keras_path)

    artifacts = {"keras": keras_path.name}
    labels = [item["name"] for item, _ in class_paths]
    (models_dir / "labels.txt").write_text(
        "".join(f"{index} {label}\n" for index, label in enumerate(labels)), encoding="utf-8"
    )
    training_report = {
        "project_kind": "image",
        "trained_at": utc_now_iso(),
        "class_names": labels,
        "class_counts": {item["name"]: len(paths) for item, paths in class_paths},
        "settings": {
            "image_size": image_size,
            "epochs_requested": epochs,
            "epochs_completed": epochs_completed,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_split": validation_split,
            "backbone_requested": backbone,
            "backbone_used": actual_backbone,
            "mobilenet_alpha": mobilenet_alpha if backbone == "mobilenet_v2" else None,
            "early_stopping": early_stopping,
            "augmentation_level": augmentation_level,
            "dropout": dropout,
            # Requested vs used, matching backbone_requested/backbone_used above: a
            # small_cnn run (or a MobileNetV2 whose ImageNet weights failed to load) has no
            # backbone to unfreeze, so it fine-tunes 0 blocks however many were asked for.
            # A single key would have made the report claim a second phase that never ran.
            "fine_tune_blocks_requested": fine_tune_blocks,
            "fine_tune_blocks_used": fine_tune_blocks if can_fine_tune else 0,
            "fine_tune_epochs_completed": fine_tune_epochs_completed,
            "deployment_target": str(settings.get("deployment_target", "pc")),
        },
        "dataset": {
            "train_samples": len(split.train_paths),
            "validation_samples": len(split.validation_paths),
            "calibration_samples": 0,
        },
        "evaluation": {key: float(value) for key, value in evaluation.items()},
        "history": history_to_json(history),
        "conversion": {
            "state": "pending",
            "note": "TensorFlow Lite files are generated only when Export Model is requested.",
        },
    }
    if progress:
        progress(0.96, "Writing image training report…")
    (models_dir / "training_report.json").write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifacts["training_report"] = "training_report.json"
    if progress:
        progress(1.0, "Image model trained. Preview is ready; use Export Model for INT8 quantization.")
    return {"report": training_report, "artifacts": artifacts}


def load_image_for_prediction(path_or_bytes: Path | bytes, image_size: int) -> np.ndarray:
    if isinstance(path_or_bytes, Path):
        return _load_image_array(path_or_bytes, image_size)
    with Image.open(__import__("io").BytesIO(path_or_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)
