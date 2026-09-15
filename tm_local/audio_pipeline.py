from __future__ import annotations

import json
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .audio_frontend import (
    AudioFrontendConfig,
    config_from_mapping,
    log_mel_spectrogram,
    read_wav_file,
)
from .config import AUDIO_DEFAULTS, DEPLOYMENT_TARGETS, KWS_RUNTIME_DEFAULTS
from .project_store import ProjectStore
from .training_common import (
    DatasetSplit,
    ProgressFn,
    TrainingError,
    class_weight_from_labels,
    history_to_json,
    make_keras_progress_callback,
    save_native_keras_model,
    stratified_path_split,
)
from .utils import utc_now_iso

# Waveform/feature augmentation strength, exposed to students as one word. "medium" is
# byte-for-byte what training did before this setting existed (+-1/12 clip shift, gain in
# U(0.90, 1.10), N(0, 0.012) noise), so an existing project retrains identically.
AUDIO_AUGMENTATION_PRESETS: dict[str, dict[str, float] | None] = {
    "off": None,
    "light": {"shift_div": 24, "gain": 0.05, "noise": 0.006},
    "medium": {"shift_div": 12, "gain": 0.10, "noise": 0.012},
    "strong": {"shift_div": 6, "gain": 0.20, "noise": 0.020},
}

# SpecAugment band widths, as a fraction (1/10) of each axis. Deliberately small: a 1 s
# clip only holds ~98 frames, and masking a tenth of them already removes a whole phoneme.
SPEC_AUGMENT_MASK_DIVISOR = 10


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


def _feature_from_path(path: Path, frontend: AudioFrontendConfig) -> np.ndarray:
    _, signal = read_wav_file(path, frontend.sample_rate)
    return log_mel_spectrogram(signal, frontend)


def spec_augment_masks(
    frame_count: int, mel_bins: int, rng: np.random.Generator
) -> tuple[slice, slice]:
    """One SpecAugment draw: the time band and the frequency band to zero out.

    Pure and seedable so the rule can be tested without TensorFlow; ``_make_dataset()``
    reproduces exactly this rule with ``tf.random.uniform`` inside the tf.data graph
    (a numpy RNG cannot be called per element there without breaking graph mode).
    Either width may come out 0, which is the "no mask on this axis" case.
    """

    time_limit = max(1, frame_count // SPEC_AUGMENT_MASK_DIVISOR)
    mel_limit = max(1, mel_bins // SPEC_AUGMENT_MASK_DIVISOR)
    time_width = int(rng.integers(0, time_limit + 1))
    time_start = int(rng.integers(0, max(1, frame_count - time_width + 1)))
    mel_width = int(rng.integers(0, mel_limit + 1))
    mel_start = int(rng.integers(0, max(1, mel_bins - mel_width + 1)))
    return (
        slice(time_start, time_start + time_width),
        slice(mel_start, mel_start + mel_width),
    )


def background_class_index(
    classes: Sequence[dict[str, Any]], background_class_id: str
) -> int | None:
    """Index of the class that means "nothing was said", or None when there is none.

    An explicit ``background_class_id`` (picked in the UI) wins. When it names no existing
    class -- a stale id left behind by a deleted class -- the NAME fallback still runs
    rather than declaring the project background-less, because a project that still HAS a
    class called "Background Noise" plainly has one.

    The name fallback delegates to ``tm_local.mcu.kws_codegen``: that module already owns
    the English and Traditional-Chinese hint list the FIRMWARE matches against, and training
    and the board disagreeing about which class is background is exactly the kind of bug
    that is invisible on the PC and wrong on the device. The import is local because it is
    needed once per training run and audio_pipeline must stay cheap to import.
    """

    from .mcu.kws_codegen import background_index, has_background

    if background_class_id:
        for index, item in enumerate(classes):
            if item.get("id") == background_class_id:
                return index
    labels = [str(item.get("name", "")) for item in classes]
    return background_index(labels) if has_background(labels) else None


def split_audio_clips(
    refs: Sequence[tuple[int, Path, str]],
    validation_split: float,
    session_disjoint: bool,
    seed: int = 1337,
    *,
    class_count: int | None = None,
) -> tuple[DatasetSplit, dict[str, Any]]:
    """Split audio clips into train/validation, per class, without leaking recordings.

    With ``session_disjoint=True`` the split reuses ``known_sound_pipeline.split_sessions``
    so clips sliced out of one take never straddle the split -- adjacent slices of the same
    recording are near-duplicates, and letting them sit on both sides is what makes the
    reported accuracy systematically optimistic. A class with fewer than two recording
    sessions cannot be split that way, so that class alone falls back to the clip-level
    ``stratified_path_split`` and is named in ``info["fallback_classes"]`` plus a
    student-facing ``info["split_warning"]``; refusing to train would be worse teaching
    than training with a documented caveat.
    """

    from .known_sound_pipeline import ClipRef, split_sessions

    info: dict[str, Any] = {
        "split_rule": "clip-level stratified split",
        "train_sessions": [],
        "validation_sessions": [],
        "fallback_classes": [],
    }
    by_class: dict[int, list[tuple[Path, str]]] = {}
    for class_index, path, session in refs:
        by_class.setdefault(int(class_index), []).append((Path(path), str(session)))
    total_classes = max(class_count or 0, max(by_class, default=-1) + 1)

    if not session_disjoint or not 0.0 < validation_split < 1.0:
        class_paths = [
            (None, [path for path, _ in by_class.get(index, [])]) for index in range(total_classes)
        ]
        return stratified_path_split(class_paths, validation_split, seed=seed), info

    info["split_rule"] = "session-disjoint; clips from one recording never cross the split"
    train_paths: list[Path] = []
    train_labels: list[int] = []
    validation_paths: list[Path] = []
    validation_labels: list[int] = []
    for class_index in sorted(by_class):
        entries = by_class[class_index]
        if len({session for _, session in entries}) < 2:
            info["fallback_classes"].append(class_index)
            fallback = stratified_path_split(
                [(None, [path for path, _ in entries])], validation_split, seed=seed
            )
            train_paths.extend(fallback.train_paths)
            train_labels.extend([class_index] * len(fallback.train_paths))
            validation_paths.extend(fallback.validation_paths)
            validation_labels.extend([class_index] * len(fallback.validation_paths))
            continue
        clips = [
            ClipRef(path=path, class_index=class_index, session_id=session)
            for path, session in entries
        ]
        split = split_sessions(clips, val_fraction=validation_split)
        train_paths.extend(clip.path for clip in split.train_clips)
        train_labels.extend([class_index] * len(split.train_clips))
        validation_paths.extend(clip.path for clip in split.validation_clips)
        validation_labels.extend([class_index] * len(split.validation_clips))
        info["train_sessions"].extend(split.train_sessions)
        info["validation_sessions"].extend(split.validation_sessions)
    if info["fallback_classes"]:
        info["split_warning"] = (
            "這些類別只有一個錄音場次，驗證改用 clip-level 切分，該類別的 accuracy 可能偏高："
            + ", ".join(str(index) for index in info["fallback_classes"])
        )

    rng = random.Random(seed)
    train_order = list(range(len(train_paths)))
    validation_order = list(range(len(validation_paths)))
    rng.shuffle(train_order)
    rng.shuffle(validation_order)
    return (
        DatasetSplit(
            train_paths=[train_paths[index] for index in train_order],
            train_labels=[train_labels[index] for index in train_order],
            validation_paths=[validation_paths[index] for index in validation_order],
            validation_labels=[validation_labels[index] for index in validation_order],
        ),
        info,
    )


def _build_audio_model(
    feature_shape: tuple[int, int, int], class_count: int, dropout: float = 0.2
):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=feature_shape, dtype=tf.float32, name="spectrogram")
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(float(dropout), name="dropout")(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="scores")(x)
    return tf.keras.Model(inputs, outputs, name="tm_local_audio_classifier")


def _make_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool,
    *,
    augmentation: str = "medium",
    spec_augment: bool = False,
):
    """Batch the log-mel features, augmenting only the training half.

    Augmentation happens here, on the already-extracted features -- the front-end itself
    (`audio_frontend.log_mel_spectrogram`) and the exported graph are untouched, so the
    PC reference implementation and the MCU C port stay bit-compatible with training.
    """

    import tensorflow as tf

    preset = AUDIO_AUGMENTATION_PRESETS.get(
        str(augmentation).strip().lower(), AUDIO_AUGMENTATION_PRESETS["medium"]
    )
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if training:
        dataset = dataset.shuffle(max(32, len(features)), reshuffle_each_iteration=True)

        if preset is not None or spec_augment:

            def augment(feature, label):  # noqa: ANN001
                if preset is not None:
                    max_shift = tf.maximum(1, tf.shape(feature)[0] // int(preset["shift_div"]))
                    shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
                    feature = tf.roll(feature, shift=shift, axis=0)
                    gain_range = float(preset["gain"])
                    gain = tf.random.uniform([], 1.0 - gain_range, 1.0 + gain_range)
                    noise = tf.random.normal(tf.shape(feature), stddev=float(preset["noise"]))
                    feature = tf.clip_by_value(feature * gain + noise, 0.0, 1.0)
                if spec_augment:
                    # Same rule as spec_augment_masks(), drawn inside the graph.
                    frames = tf.shape(feature)[0]
                    bins = tf.shape(feature)[1]
                    t_w = tf.random.uniform(
                        [], 0, frames // SPEC_AUGMENT_MASK_DIVISOR + 1, dtype=tf.int32
                    )
                    t_0 = tf.random.uniform([], 0, tf.maximum(1, frames - t_w + 1), dtype=tf.int32)
                    f_w = tf.random.uniform(
                        [], 0, bins // SPEC_AUGMENT_MASK_DIVISOR + 1, dtype=tf.int32
                    )
                    f_0 = tf.random.uniform([], 0, tf.maximum(1, bins - f_w + 1), dtype=tf.int32)
                    t_idx = tf.range(frames)
                    f_idx = tf.range(bins)
                    t_mask = tf.logical_and(t_idx >= t_0, t_idx < t_0 + t_w)
                    f_mask = tf.logical_and(f_idx >= f_0, f_idx < f_0 + f_w)
                    keep = tf.logical_not(tf.logical_or(t_mask[:, None], f_mask[None, :]))
                    feature = feature * tf.cast(keep, feature.dtype)[..., None]
                return feature, label

            dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def write_audio_frontend_reference(models_dir: Path) -> None:
    source = r'''from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def scale_pcm_dtype(data: np.ndarray, dtype, out_dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return data.astype(out_dtype)
    if dtype == np.uint8:
        return (data.astype(out_dtype) - 128.0) / 128.0
    info = np.iinfo(dtype)
    return data.astype(out_dtype) / float(max(abs(info.min), info.max))


def normalize_pcm(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 2:
        # Scale first, mix second -- must stay bit-identical to
        # tm_local/audio_frontend.py:_normalize_pcm(). Averaging raw integer channels
        # first leaves the values on the +-32768 scale, which then satisfies
        # issubdtype(floating), skips the integer rescale and is clipped into a
        # full-scale square wave.
        value = scale_pcm_dtype(data, data.dtype, np.float64).mean(axis=1).astype(np.float32)
    else:
        value = scale_pcm_dtype(data, data.dtype, np.float32)
    return np.clip(np.nan_to_num(value), -1.0, 1.0).astype(np.float32)


def load_wav(path: str | Path, target_rate: int) -> np.ndarray:
    rate, data = wavfile.read(path)
    signal = normalize_pcm(np.asarray(data))
    if rate != target_rate:
        gcd = math.gcd(int(rate), int(target_rate))
        signal = resample_poly(signal, target_rate // gcd, rate // gcd).astype(np.float32)
    return signal


def fix_length(signal: np.ndarray, length: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size >= length:
        start = max(0, (signal.size - length) // 2)
        return signal[start:start + length]
    output = np.zeros(length, dtype=np.float32)
    start = (length - signal.size) // 2
    output[start:start + signal.size] = signal
    return output


def hz_to_mel(value):
    return 2595.0 * np.log10(1.0 + np.asarray(value, dtype=np.float64) / 700.0)


def mel_to_hz(value):
    return 700.0 * (10.0 ** (np.asarray(value, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(config: dict) -> np.ndarray:
    rate = int(config["sample_rate"])
    fft_size = int(config["fft_size"])
    mel_bins = int(config["mel_bins"])
    points = np.linspace(hz_to_mel(config["fmin"]), hz_to_mel(config["fmax"]), mel_bins + 2)
    bins = np.floor((fft_size + 1) * mel_to_hz(points) / rate).astype(int)
    bins = np.clip(bins, 0, fft_size // 2)
    filters = np.zeros((mel_bins, fft_size // 2 + 1), dtype=np.float32)
    for index in range(1, mel_bins + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]
        center = max(left + 1, center)
        right = max(center + 1, right)
        for position in range(left, min(center, filters.shape[1])):
            filters[index - 1, position] = (position - left) / max(1, center - left)
        for position in range(center, min(right, filters.shape[1])):
            filters[index - 1, position] = (right - position) / max(1, right - center)
    filters /= np.maximum(filters.sum(axis=1, keepdims=True), 1e-12)
    return filters


def extract_feature(path: str | Path, config_path: str | Path | None = None) -> np.ndarray:
    config_path = Path(config_path or Path(__file__).with_name("audio_frontend.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    signal = fix_length(load_wav(path, int(config["sample_rate"])), int(config["clip_samples"]))
    window = int(config["window_samples"])
    hop = int(config["hop_samples"])
    frames = int(config["frame_count"])
    needed = window + (frames - 1) * hop
    signal = np.pad(signal, (0, max(0, needed - signal.size)))
    shape = (frames, window)
    strides = (signal.strides[0] * hop, signal.strides[0])
    framed = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()
    framed *= np.hanning(window).astype(np.float32)[None, :]
    spectrum = np.fft.rfft(framed, n=int(config["fft_size"]), axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    mel = power @ mel_filterbank(config).T
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db -= np.max(db)
    floor = float(config["db_floor"])
    normalized = (np.clip(db, floor, 0.0) - floor) / -floor
    return normalized.astype(np.float32)[..., None]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python audio_frontend_reference.py sample.wav")
    value = extract_feature(sys.argv[1])
    print(value.shape, value.dtype, float(value.min()), float(value.max()))
'''
    (models_dir / "audio_frontend_reference.py").write_text(source, encoding="utf-8")


def kws_decision_policy(report_settings: Any) -> dict[str, Any]:
    """The trigger policy an exported audio model was tuned under.

    One resolver for every export consumer -- the generated ``run_model.py`` and the ZIP's
    ``metadata.json`` -- so the two can never drift into two answers to "when does this
    count as heard". ``report_settings`` is ``training_report["settings"]``, the same
    mapping ``mcu/contract.py:232-259`` reads, and the clamping and the "only known keys"
    merge deliberately reproduce ``contract.collect()`` / ``kws_codegen.py:367`` so the ZIP
    and the firmware describe one policy rather than two.
    """

    settings = report_settings if isinstance(report_settings, dict) else {}
    threshold = _bounded_float(
        settings.get("detection_threshold"),
        0.05,
        0.99,
        float(AUDIO_DEFAULTS["detection_threshold"]),
    )
    runtime = dict(KWS_RUNTIME_DEFAULTS)
    stored = settings.get("kws_runtime")
    if isinstance(stored, dict):
        runtime.update({key: value for key, value in stored.items() if key in runtime})
    return {"detection_threshold": threshold, "kws_runtime": runtime}


def write_audio_runner(
    models_dir: Path,
    model_filename: str = "audio_classifier_spectrogram_int8.tflite",
    detection_threshold: float = AUDIO_DEFAULTS["detection_threshold"],
) -> None:
    """Write the PC reference runner for an exported audio (KWS) classifier.

    ``detection_threshold`` is the student's 偵測門檻 setting, baked in so the exported
    runner decides "I heard it" with the same number the board firmware compiles in
    (``mcu.kws_codegen`` reads the identical value out of the training report). Without it
    the runner reports the bare top-1 label, and a softmax always HAS a top-1 -- silence
    and an unknown word both come back as somebody's keyword.
    """

    threshold = max(0.05, min(0.99, float(detection_threshold)))
    source = rf'''from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from audio_frontend_reference import extract_feature

MODEL = Path(__file__).with_name("{model_filename}")
LABELS = [line.split(" ", 1)[1].strip() for line in Path(__file__).with_name("labels.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
# The detection_threshold this model was exported with. The board firmware compiles in
# the same number, so the ZIP and the device agree on what counts as heard.
DETECTION_THRESHOLD = {threshold!r}


def main(path: str) -> None:
    value = extract_feature(path)[None, ...]
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
    ranked = sorted(zip(LABELS, scores), key=lambda item: item[1], reverse=True)
    for label, score in ranked:
        print(f"{{label}}: {{score:.4f}}")
    if not ranked:
        raise SystemExit("The model produced no scores; re-export the model.")
    best_label, best_score = ranked[0]
    print()
    if best_score >= DETECTION_THRESHOLD:
        print(f"Detected: {{best_label}} ({{best_score:.1%}} >= threshold {{DETECTION_THRESHOLD:.0%}})")
    else:
        print(f"Detected: (nothing above threshold {{DETECTION_THRESHOLD:.0%}}; best was {{best_label}} at {{best_score:.1%}})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_model.py sample.wav")
    main(sys.argv[1])
'''
    (models_dir / "run_model.py").write_text(source, encoding="utf-8")


# Backward-compatible private names used by the test suite and older patches.
_write_audio_frontend_reference = write_audio_frontend_reference
_write_audio_runner = write_audio_runner


def audio_representative_samples(
    store: ProjectStore,
    project_id: str,
    settings: dict[str, Any],
    *,
    limit: int = 120,
) -> list[np.ndarray]:
    """Extract a balanced log-mel representative set for TFLite calibration."""

    frontend = config_from_mapping(settings)
    class_paths = store.sample_paths_by_class(project_id)
    selected: list[Path] = []
    max_count = max((len(paths) for _, paths in class_paths), default=0)
    for offset in range(max_count):
        for _, paths in class_paths:
            if offset < len(paths):
                selected.append(paths[offset])
                if len(selected) >= max(1, int(limit)):
                    return [_feature_from_path(path, frontend) for path in selected]
    return [_feature_from_path(path, frontend) for path in selected]


def train_audio_project(
    store: ProjectStore,
    project_id: str,
    options: dict[str, Any] | None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    options = dict(options or {})
    project = store.get_raw_project(project_id)
    if project["kind"] != "audio":
        raise TrainingError("Project is not an audio project.")
    settings = {**project["settings"], **options}
    frontend = config_from_mapping(settings)
    epochs = _bounded_int(settings.get("epochs"), 1, 300, 40)
    batch_size = _bounded_int(settings.get("batch_size"), 1, 128, 16)
    learning_rate = _bounded_float(settings.get("learning_rate"), 1e-6, 0.1, 0.001)
    validation_split = _bounded_float(settings.get("validation_split"), 0.0, 0.45, 0.2)
    minimum = _bounded_int(settings.get("minimum_samples_per_class"), 2, 200, 8)
    early_stopping = bool(settings.get("early_stopping", True))
    augmentation = str(settings.get("augmentation_level", "medium")).strip().lower()
    if augmentation not in AUDIO_AUGMENTATION_PRESETS:
        augmentation = "medium"
    spec_augment = bool(settings.get("spec_augment", False))
    session_disjoint = bool(settings.get("session_disjoint_validation", True))
    background_weight = _bounded_float(settings.get("background_weight"), 0.25, 4.0, 1.0)
    background_class_id = str(settings.get("background_class_id", "") or "")
    dropout = _bounded_float(settings.get("dropout"), 0.0, 0.6, 0.2)
    detection_threshold = _bounded_float(settings.get("detection_threshold"), 0.05, 0.99, 0.5)
    deployment_target = str(settings.get("deployment_target", "pc"))
    if deployment_target not in DEPLOYMENT_TARGETS:
        deployment_target = "pc"

    class_paths = store.sample_paths_by_class(project_id)
    shortages = [
        f"{class_item['name']} ({len(paths)}/{minimum})"
        for class_item, paths in class_paths
        if len(paths) < minimum
    ]
    if shortages:
        raise TrainingError("Each class needs more audio samples: " + ", ".join(shortages))
    if progress:
        progress(0.02, "Extracting log-mel spectrograms…")
    split, split_info = split_audio_clips(
        store.audio_clip_refs(project_id),
        validation_split,
        session_disjoint,
        class_count=len(class_paths),
    )
    all_paths = split.train_paths + split.validation_paths
    feature_map: dict[Path, np.ndarray] = {}
    for index, path in enumerate(all_paths):
        feature_map[path] = _feature_from_path(path, frontend)
        if progress and (index % 8 == 0 or index + 1 == len(all_paths)):
            progress(0.03 + 0.12 * (index + 1) / max(1, len(all_paths)), f"Audio feature {index + 1}/{len(all_paths)}")
    train_features = np.stack([feature_map[path] for path in split.train_paths]).astype(np.float32)
    train_labels = np.asarray(split.train_labels, dtype=np.int32)
    validation_features = (
        np.stack([feature_map[path] for path in split.validation_paths]).astype(np.float32)
        if split.validation_paths
        else None
    )
    validation_labels = (
        np.asarray(split.validation_labels, dtype=np.int32) if split.validation_paths else None
    )

    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(1337)
    model = _build_audio_model(frontend.feature_shape, len(class_paths), dropout=dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    train_ds = _make_dataset(
        train_features,
        train_labels,
        batch_size,
        True,
        augmentation=augmentation,
        spec_augment=spec_augment,
    )
    validation_ds = None
    if validation_features is not None and validation_labels is not None:
        validation_ds = _make_dataset(validation_features, validation_labels, batch_size, False)
    callbacks: list[Any] = [
        make_keras_progress_callback(epochs, progress, start=0.16, end=0.67)
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
    # Balanced class weights, then the student's background emphasis on top: a background
    # class is usually the one class you can record endlessly, so its own weight is the knob
    # for "stop firing on silence" (>1) or "stop being deaf to quiet keywords" (<1).
    class_weight = class_weight_from_labels(split.train_labels, len(class_paths))
    background = background_class_index(project["classes"], background_class_id)
    background_weight_applied = 1.0
    if background is not None and background in class_weight:
        class_weight[background] *= background_weight
        background_weight_applied = background_weight
    elif abs(background_weight - 1.0) > 1e-9 and progress:
        # No class means "nothing was said", so there is nothing to emphasise. Say so here
        # rather than let the student sit through a full retrain and receive a model
        # weighted exactly as before. Same contract as the fine_tune_blocks skip in
        # image_pipeline.py: name the setting and name the fix.
        progress(
            0.15,
            f"這個專案沒有背景類別，已略過 background_weight={background_weight} 設定。"
            "請在進階設定的「背景類別」選一個類別，或把某個類別改名為「背景音」"
            "後重新訓練。",
        )
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    evaluation_ds = validation_ds if validation_ds is not None else train_ds
    evaluation = model.evaluate(evaluation_ds, verbose=0, return_dict=True)
    if progress:
        progress(0.69, "Saving trained audio model…")

    project_dir = store.project_dir(project_id)
    models_dir = project_dir / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)
    keras_path = models_dir / "audio_classifier_spectrogram.keras"
    save_native_keras_model(model, keras_path)
    artifacts = {"keras": keras_path.name}
    labels = [item["name"] for item, _ in class_paths]
    (models_dir / "labels.txt").write_text(
        "".join(f"{index} {label}\n" for index, label in enumerate(labels)), encoding="utf-8"
    )
    (models_dir / "audio_frontend.json").write_text(
        json.dumps(frontend.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_audio_frontend_reference(models_dir)
    training_report = {
        "project_kind": "audio",
        "trained_at": utc_now_iso(),
        "class_names": labels,
        "class_counts": {item["name"]: len(paths) for item, paths in class_paths},
        "settings": {
            **frontend.to_dict(),
            "epochs_requested": epochs,
            "epochs_completed": len(history.history.get("loss", [])),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_split": validation_split,
            "early_stopping": early_stopping,
            "augmentation_level": augmentation,
            "spec_augment": spec_augment,
            "session_disjoint_validation": session_disjoint,
            # Requested vs applied, the same split as epochs_requested/epochs_completed
            # above: the weighting is skipped entirely when the project has no background
            # class, and a report that only echoed the request would describe a run that
            # never happened. ``_applied`` is what class_weight was actually multiplied by.
            "background_weight_requested": background_weight,
            "background_weight_applied": background_weight_applied,
            # Resolved INDEX, not the class id: the firmware indexes score arrays, and the
            # id would be meaningless there. None means "this project has no background
            # class", which kws_codegen turns into HAS_BACKGROUND = 0.
            "background_class_index": background,
            "dropout": dropout,
            "detection_threshold": detection_threshold,
            "deployment_target": deployment_target,
            # Firmware-side smoothing/trigger policy. Carried in the report so a deploy
            # reads one file, and so re-deploying an old project cannot silently pick up a
            # different policy than the one it was trained and reported against.
            "kws_runtime": dict(KWS_RUNTIME_DEFAULTS),
        },
        "dataset": {
            "train_samples": len(split.train_paths),
            "validation_samples": len(split.validation_paths),
            "calibration_samples": 0,
            **split_info,
        },
        "evaluation": {key: float(value) for key, value in evaluation.items()},
        "history": history_to_json(history),
        "conversion": {
            "state": "pending",
            "note": "TensorFlow Lite files are generated only when Export Model is requested.",
        },
    }
    if progress:
        progress(0.96, "Writing audio training report…")
    (models_dir / "training_report.json").write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifacts.update(
        {
            "training_report": "training_report.json",
            "audio_frontend": "audio_frontend.json",
            "audio_frontend_reference": "audio_frontend_reference.py",
        }
    )
    if progress:
        progress(1.0, "Audio model trained. Preview is ready; use Export Model for INT8 quantization.")
    return {"report": training_report, "artifacts": artifacts}


def feature_from_wav_bytes(payload: bytes, settings: dict[str, Any]) -> np.ndarray:
    from .audio_frontend import read_wav_bytes

    frontend = config_from_mapping(settings)
    _, signal = read_wav_bytes(payload, frontend.sample_rate)
    return log_mel_spectrogram(signal, frontend)
