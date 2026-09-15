"""Pinned YAMNet feature extractor and frontend used by Abnormal Sound projects.

The architecture and signal constants follow TensorFlow Models' Apache-2.0 licensed
YAMNet implementation.  The product model deliberately accepts a 96 x 64 log-mel patch
instead of a raw waveform: this keeps FFT/complex operations outside the TensorFlow Lite
flatbuffer and gives strict-integer export a realistic path.

TensorFlow imports stay inside functions.  ``tm_local.__init__`` must set
``TF_USE_LEGACY_KERAS=1`` before any of those functions are called because the official
HDF5 weights use the Keras 2 model layout.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import PROJECT_ROOT

YAMNET_SAMPLE_RATE = 16_000
YAMNET_WINDOW_SECONDS = 0.025
YAMNET_HOP_SECONDS = 0.010
YAMNET_MEL_BANDS = 64
YAMNET_MEL_MIN_HZ = 125.0
YAMNET_MEL_MAX_HZ = 7_500.0
YAMNET_LOG_OFFSET = 0.001
YAMNET_PATCH_FRAMES = 96
YAMNET_PATCH_HOP_FRAMES = 48
YAMNET_EMBEDDING_DIM = 1_024
YAMNET_CLASS_COUNT = 521
YAMNET_FRONTEND_SCHEMA_VERSION = "yamnet-logmel-patch-v2"

YAMNET_WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
YAMNET_WEIGHTS_SHA256 = "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744"
YAMNET_ASSET_VERSION = f"audioset-yamnet-h5-{YAMNET_WEIGHTS_SHA256[:12]}"
YAMNET_ASSET_DIR = PROJECT_ROOT / "assets" / "yamnet"
YAMNET_WEIGHTS_PATH = YAMNET_ASSET_DIR / "yamnet.h5"

# The 521 AudioSet display names that go with the pinned weights, used ONLY by the
# recording sanity check (see known_sound_pipeline.describe_waveform). It is a separate,
# separately-pinned asset because the weight file does not carry the vocabulary.
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/"
    "yamnet/yamnet_class_map.csv"
)
YAMNET_CLASS_MAP_SHA256 = "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"
YAMNET_CLASS_MAP_PATH = YAMNET_ASSET_DIR / "yamnet_class_map.csv"


class YamnetAssetError(RuntimeError):
    """The pinned YAMNet weight asset is absent or does not match its digest."""


class YamnetClassMapError(RuntimeError):
    """The pinned AudioSet class map is absent or does not match its digest.

    Deliberately a separate exception: a missing class map disables only the optional
    recording sanity check, while a missing weight file stops training outright.
    """


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_yamnet_weights(path: Path | None = None) -> Path:
    """Return the verified local weight path; never download during a Train job."""

    resolved = Path(path or YAMNET_WEIGHTS_PATH).resolve()
    if not resolved.is_file():
        raise YamnetAssetError(
            "The pinned YAMNet weights are not installed. Run 01_INSTALL.bat once while "
            "connected to the internet, then retry. Existing recordings are safe."
        )
    actual = _sha256(resolved)
    if actual != YAMNET_WEIGHTS_SHA256:
        raise YamnetAssetError(
            "The local YAMNet weight file failed its SHA-256 check. Delete "
            f"{resolved} and run 01_INSTALL.bat again. Expected {YAMNET_WEIGHTS_SHA256}, "
            f"got {actual}."
        )
    return resolved


def load_class_map(path: Path | None = None) -> list[str]:
    """Return the 521 official AudioSet display names, in output order.

    Raises YamnetClassMapError when the asset is missing or altered. Callers that treat
    the sanity check as optional should catch that specific exception.
    """

    resolved = Path(path or YAMNET_CLASS_MAP_PATH).resolve()
    if not resolved.is_file():
        raise YamnetClassMapError(
            "The pinned YAMNet class map is not installed, so sound names cannot be "
            "shown. Run 01_INSTALL.bat once while connected to the internet. Training "
            "and export are unaffected."
        )
    actual = _sha256(resolved)
    if actual != YAMNET_CLASS_MAP_SHA256:
        raise YamnetClassMapError(
            "The local YAMNet class map failed its SHA-256 check. Delete "
            f"{resolved} and run 01_INSTALL.bat again."
        )
    import csv

    with resolved.open(newline="", encoding="utf-8") as handle:
        names = [str(row["display_name"]) for row in csv.DictReader(handle)]
    if len(names) != YAMNET_CLASS_COUNT:
        raise YamnetClassMapError(
            f"The YAMNet class map has {len(names)} rows, expected {YAMNET_CLASS_COUNT}."
        )
    return names


def class_map_available(path: Path | None = None) -> bool:
    """True when the class map asset is present and matches its digest."""

    try:
        load_class_map(path)
    except YamnetClassMapError:
        return False
    return True


_LAYER_DEFS: tuple[tuple[str, int, int], ...] = (
    ("conv", 2, 32),
    ("separable", 1, 64),
    ("separable", 2, 128),
    ("separable", 1, 128),
    ("separable", 2, 256),
    ("separable", 1, 256),
    ("separable", 2, 512),
    ("separable", 1, 512),
    ("separable", 1, 512),
    ("separable", 1, 512),
    ("separable", 1, 512),
    ("separable", 1, 512),
    ("separable", 2, 1024),
    ("separable", 1, 1024),
)


#: Number of blocks in the official topology. Block 1 is a plain conv; 2..14 are separable.
YAMNET_BLOCK_COUNT = len(_LAYER_DEFS)
#: Shallowest cut that still yields a usable embedding. Below this the receptive field is
#: too small for 0.96 s of audio to mean anything.
YAMNET_MIN_ENCODER_DEPTH = 2


def encoder_layer_name(depth: int) -> str:
    """Name of the ReLU whose output is the embedding when the encoder is cut at ``depth``.

    Truncating is worthwhile because YAMNet is MobileNetV1-shaped and its parameters are
    heavily back-loaded: block 14 alone is ~33% of the encoder and blocks 13-14 together
    are ~49%. Cutting therefore shrinks the exported model far more than proportionally.
    """

    value = int(depth)
    if not YAMNET_MIN_ENCODER_DEPTH <= value <= YAMNET_BLOCK_COUNT:
        raise ValueError(
            f"encoder_depth must be between {YAMNET_MIN_ENCODER_DEPTH} and "
            f"{YAMNET_BLOCK_COUNT}, got {value}."
        )
    return f"layer{value}_pointwise_relu"


def encoder_output_dim(depth: int) -> int:
    """Embedding width produced by cutting at ``depth`` (the block's pointwise filters)."""

    value = int(depth)
    if not YAMNET_MIN_ENCODER_DEPTH <= value <= YAMNET_BLOCK_COUNT:
        raise ValueError(
            f"encoder_depth must be between {YAMNET_MIN_ENCODER_DEPTH} and "
            f"{YAMNET_BLOCK_COUNT}, got {value}."
        )
    return int(_LAYER_DEFS[value - 1][2])


def _batch_norm(tf: Any, value: Any, name: str) -> Any:
    return tf.keras.layers.BatchNormalization(
        name=name,
        center=True,
        scale=False,
        epsilon=1e-4,
    )(value)


def _conv_block(tf: Any, value: Any, *, index: int, stride: int, filters: int) -> Any:
    prefix = f"layer{index}"
    value = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
        activation=None,
        name=f"{prefix}_conv",
    )(value)
    value = _batch_norm(tf, value, f"{prefix}_conv_bn")
    return tf.keras.layers.ReLU(name=f"{prefix}_relu")(value)


def _separable_block(
    tf: Any,
    value: Any,
    *,
    index: int,
    stride: int,
    filters: int,
) -> Any:
    prefix = f"layer{index}"
    value = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=stride,
        depth_multiplier=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=f"{prefix}_depthwise_conv",
    )(value)
    value = _batch_norm(tf, value, f"{prefix}_depthwise_conv_bn")
    value = tf.keras.layers.ReLU(name=f"{prefix}_depthwise_relu")(value)
    value = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=f"{prefix}_pointwise_conv",
    )(value)
    value = _batch_norm(tf, value, f"{prefix}_pointwise_conv_bn")
    return tf.keras.layers.ReLU(name=f"{prefix}_pointwise_relu")(value)


def build_yamnet_models(weights_path: Path | None = None) -> tuple[Any, Any]:
    """Build the exact official topology and return ``(full_tagger, encoder)``.

    The full 521-output model is constructed solely so Keras can load the official HDF5
    file by topology.  The returned encoder shares those loaded variables and has one
    standard tensor output, the 1024-D embedding.  No custom layer is serialized.
    """

    import tensorflow as tf

    weights = require_yamnet_weights(weights_path)
    inputs = tf.keras.layers.Input(
        shape=(YAMNET_PATCH_FRAMES, YAMNET_MEL_BANDS),
        dtype=tf.float32,
        name="log_mel_patch",
    )
    value = tf.keras.layers.Reshape(
        (YAMNET_PATCH_FRAMES, YAMNET_MEL_BANDS, 1),
        name="reshape_patch",
    )(inputs)
    for index, (kind, stride, filters) in enumerate(_LAYER_DEFS, start=1):
        if kind == "conv":
            value = _conv_block(tf, value, index=index, stride=stride, filters=filters)
        else:
            value = _separable_block(
                tf,
                value,
                index=index,
                stride=stride,
                filters=filters,
            )
    embedding = tf.keras.layers.GlobalAveragePooling2D(name="embedding")(value)
    logits = tf.keras.layers.Dense(
        units=YAMNET_CLASS_COUNT,
        use_bias=True,
        name="classifier_logits",
    )(embedding)
    predictions = tf.keras.layers.Activation("sigmoid", name="class_scores")(logits)
    full_model = tf.keras.Model(inputs, predictions, name="yamnet_official_patch_tagger")
    full_model.load_weights(str(weights))
    encoder = tf.keras.Model(inputs, embedding, name="yamnet_embedding_encoder")
    encoder.trainable = False
    if int(full_model.count_params()) != 3_751_369:
        raise YamnetAssetError(
            "The constructed YAMNet topology has an unexpected parameter count: "
            f"{full_model.count_params()} (expected 3,751,369)."
        )
    if int(encoder.count_params()) != 3_217_344:
        raise YamnetAssetError(
            "The YAMNet encoder has an unexpected parameter count: "
            f"{encoder.count_params()} (expected 3,217,344)."
        )
    return full_model, encoder


def frontend_contract() -> dict[str, Any]:
    return {
        "schema_version": YAMNET_FRONTEND_SCHEMA_VERSION,
        "sample_rate": YAMNET_SAMPLE_RATE,
        "stft_window_seconds": YAMNET_WINDOW_SECONDS,
        "stft_hop_seconds": YAMNET_HOP_SECONDS,
        "fft_length": 512,
        "periodic_hann": True,
        "spectrum": "magnitude",
        "mel_bands": YAMNET_MEL_BANDS,
        "mel_min_hz": YAMNET_MEL_MIN_HZ,
        "mel_max_hz": YAMNET_MEL_MAX_HZ,
        "log_offset": YAMNET_LOG_OFFSET,
        "patch_frames": YAMNET_PATCH_FRAMES,
        "patch_hop_frames": YAMNET_PATCH_HOP_FRAMES,
        "patch_seconds": 0.96,
        "patch_hop_seconds": 0.48,
        "padding_policy": "pad_only_when_shorter_than_first_complete_patch",
        "model_input_shape": [YAMNET_PATCH_FRAMES, YAMNET_MEL_BANDS],
        "inference_window_seconds": 1.0,
        "inference_hop_seconds": 0.5,
        "inference_tail_policy": "complete_half_second_grid_windows_only",
    }


def waveform_to_log_mel_patches(waveform: np.ndarray) -> np.ndarray:
    """Compute official YAMNet log-mel patches, without a mostly-silent trailing patch.

    A waveform shorter than the 975 ms needed for the first output is right-padded.  Longer
    waveforms emit complete 0.96 s patches at a 0.48 s hop; unlike the waveform demo model,
    this product path does not pad a partial *final* hop into an extra mostly-zero patch.
    Stored Local Studio clips are exactly one second, so they yield one patch.
    """

    import tensorflow as tf

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    minimum_samples = int(
        round((0.96 + YAMNET_WINDOW_SECONDS - YAMNET_HOP_SECONDS) * YAMNET_SAMPLE_RATE)
    )
    if values.size < minimum_samples:
        values = np.pad(values, (0, minimum_samples - values.size))
    tensor = tf.convert_to_tensor(values, dtype=tf.float32)
    window_samples = int(round(YAMNET_WINDOW_SECONDS * YAMNET_SAMPLE_RATE))
    hop_samples = int(round(YAMNET_HOP_SECONDS * YAMNET_SAMPLE_RATE))
    magnitude = tf.abs(
        tf.signal.stft(
            tensor,
            frame_length=window_samples,
            frame_step=hop_samples,
            fft_length=512,
        )
    )
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=YAMNET_MEL_BANDS,
        num_spectrogram_bins=257,
        sample_rate=YAMNET_SAMPLE_RATE,
        lower_edge_hertz=YAMNET_MEL_MIN_HZ,
        upper_edge_hertz=YAMNET_MEL_MAX_HZ,
        dtype=tf.float32,
    )
    log_mel = tf.math.log(tf.matmul(magnitude, mel_matrix) + YAMNET_LOG_OFFSET)
    patches = tf.signal.frame(
        log_mel,
        frame_length=YAMNET_PATCH_FRAMES,
        frame_step=YAMNET_PATCH_HOP_FRAMES,
        axis=0,
    )
    result = np.asarray(patches.numpy(), dtype=np.float32)
    if result.ndim != 3 or result.shape[1:] != (YAMNET_PATCH_FRAMES, YAMNET_MEL_BANDS):
        raise ValueError(f"Unexpected YAMNet patch shape: {result.shape!r}.")
    if result.shape[0] < 1:
        raise ValueError("The waveform did not produce a complete YAMNet patch.")
    if not np.all(np.isfinite(result)):
        raise ValueError("The YAMNet frontend produced non-finite values.")
    return result


def embedding_vectors(
    encoder: Any,
    waveforms: Iterable[np.ndarray],
    *,
    batch_size: int = 24,
) -> np.ndarray:
    """Return one mean-pooled 1024-D embedding for every waveform."""

    patches: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    for waveform in waveforms:
        current = waveform_to_log_mel_patches(np.asarray(waveform, dtype=np.float32))
        start = len(patches)
        patches.extend(current)
        spans.append((start, len(patches)))
    if not patches:
        return np.empty((0, YAMNET_EMBEDDING_DIM), dtype=np.float32)
    batch = np.stack(patches, axis=0).astype(np.float32)
    raw = np.asarray(encoder.predict(batch, batch_size=max(1, int(batch_size)), verbose=0))
    if raw.shape != (len(patches), YAMNET_EMBEDDING_DIM):
        raise ValueError(f"Unexpected YAMNet embedding shape: {raw.shape!r}.")
    return np.stack([np.mean(raw[start:end], axis=0) for start, end in spans]).astype(np.float32)


def normalized_embeddings(values: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings; absolute signal level is handled by the RMS branch."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != YAMNET_EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding matrix (*, {YAMNET_EMBEDDING_DIM}), got {array.shape!r}."
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.maximum(norms, 1e-12)).astype(np.float64)
