from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ProgressFn = Callable[[float, str], None]


class TrainingError(RuntimeError):
    pass


def save_native_keras_model(model: Any, path: Path) -> Path:
    """Save a complete model using the native ``.keras`` format.

    Keras' native ``.keras`` writer does not accept the legacy
    ``include_optimizer`` keyword.  Passing that keyword raises a ValueError
    after training has already completed.  Call ``model.save(path)`` with no
    format-specific keyword, then load with ``compile=False`` when optimizer
    state is not needed.
    """

    destination = Path(path)
    if destination.suffix.lower() != ".keras":
        raise TrainingError("Native Keras model path must end in .keras.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))
    if not destination.is_file():
        raise TrainingError(f"Keras model was not created: {destination}")
    return destination


@dataclass
class DatasetSplit:
    train_paths: list[Path]
    train_labels: list[int]
    validation_paths: list[Path]
    validation_labels: list[int]


def stratified_path_split(
    class_paths: Sequence[tuple[Any, Sequence[Path]]],
    validation_split: float,
    *,
    seed: int = 1337,
) -> DatasetSplit:
    if not 0.0 <= validation_split < 0.5:
        raise TrainingError("validation_split must be in [0, 0.5).")
    rng = random.Random(seed)
    train_paths: list[Path] = []
    train_labels: list[int] = []
    validation_paths: list[Path] = []
    validation_labels: list[int] = []
    for label_index, (_, paths_value) in enumerate(class_paths):
        paths = list(paths_value)
        rng.shuffle(paths)
        if not paths:
            continue
        validation_count = 0
        if validation_split > 0 and len(paths) >= 3:
            validation_count = max(1, int(round(len(paths) * validation_split)))
            validation_count = min(validation_count, len(paths) - 1)
        validation = paths[:validation_count]
        train = paths[validation_count:]
        train_paths.extend(train)
        train_labels.extend([label_index] * len(train))
        validation_paths.extend(validation)
        validation_labels.extend([label_index] * len(validation))
    train_order = list(range(len(train_paths)))
    validation_order = list(range(len(validation_paths)))
    rng.shuffle(train_order)
    rng.shuffle(validation_order)
    return DatasetSplit(
        train_paths=[train_paths[index] for index in train_order],
        train_labels=[train_labels[index] for index in train_order],
        validation_paths=[validation_paths[index] for index in validation_order],
        validation_labels=[validation_labels[index] for index in validation_order],
    )


def class_weight_from_labels(labels: Sequence[int], class_count: int) -> dict[int, float]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=class_count)
    total = int(np.sum(counts))
    if total == 0:
        return {}
    result: dict[int, float] = {}
    for index, count in enumerate(counts):
        if count:
            result[index] = total / (class_count * int(count))
    return result


def make_keras_progress_callback(
    total_epochs: int,
    progress: ProgressFn | None,
    *,
    start: float,
    end: float,
):
    import tensorflow as tf

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):  # noqa: ANN001
            if progress:
                progress(start, "Training started…")

        def on_epoch_end(self, epoch, logs=None):  # noqa: ANN001
            logs = logs or {}
            fraction = (epoch + 1) / max(1, total_epochs)
            value = start + (end - start) * fraction
            accuracy = logs.get("accuracy")
            val_accuracy = logs.get("val_accuracy")
            loss = logs.get("loss")
            parts = [f"Epoch {epoch + 1}/{total_epochs}"]
            if accuracy is not None:
                parts.append(f"accuracy={accuracy:.3f}")
            if val_accuracy is not None:
                parts.append(f"val_accuracy={val_accuracy:.3f}")
            if loss is not None:
                parts.append(f"loss={loss:.3f}")
            if progress:
                progress(value, " · ".join(parts))

    return ProgressCallback()


def history_to_json(history: Any) -> dict[str, list[float]]:
    return {
        str(key): [float(value) for value in values]
        for key, values in getattr(history, "history", {}).items()
    }
