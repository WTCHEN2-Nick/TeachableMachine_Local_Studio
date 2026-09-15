"""Maintainer tool: build tiny strict-INT8 models with the Studio's audio contracts.

Regenerates the fixtures under ``tests/fixtures/mcu/`` used by the MCU firmware-build
tests: two small strict-INT8 ``.tflite`` models (random weights -- only shapes, dtypes
and strict-INT8-ness matter, never accuracy) plus their conversion reports, and the two
audio front-end contract sidecars the Studio itself writes next to a trained project's
model. Every artifact is produced by calling the same Studio code the real pipelines use
(``tm_local.tflite_export``, ``tm_local.audio_frontend``, ``tm_local.yamnet_model``) so
the fixtures stay byte-for-byte consistent with what ``tm_local/mcu/contract.py`` and
``export_service.py`` expect -- never hand-written JSON.

Usage::

    .venv\\Scripts\\python.exe scripts\\make_mcu_fixtures.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "mcu"


def _convert(model, rep_shape: tuple[int, ...], rep_low: float, rep_high: float, out: Path) -> dict:
    """Export ``model`` the Studio's way and quantize it with the real strict-INT8 path.

    Mirrors what ``export_service.py`` does for a trained project: write an explicit
    ``serving_default`` SavedModel signature (``export_inference_saved_model`` -- this
    is the reason ``TFLiteConverter.from_keras_model`` is never used anywhere in this
    repo), convert it with the shared ``convert_full_integer`` helper against a
    representative dataset in the model's natural input range, then run the converted
    bytes through ``inspect_tflite`` to get the exact report-entry shape
    ``tm_local/mcu/contract.py`` reads (``inputs[0]`` / ``outputs[0]``).
    """

    import numpy as np

    from tm_local.tflite_export import (
        convert_full_integer,
        export_inference_saved_model,
        inspect_tflite,
    )

    rng = np.random.default_rng(1337)
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = export_inference_saved_model(model, Path(tmp) / "saved")
        samples = [rng.uniform(rep_low, rep_high, size=rep_shape).astype(np.float32) for _ in range(32)]
        blob = convert_full_integer(saved_dir, samples, "int8")
    out.write_bytes(blob)
    entry = inspect_tflite(out)
    return entry


def build_kws() -> None:
    import tensorflow as tf

    from tm_local.audio_frontend import AudioFrontendConfig
    from tm_local.utils import atomic_write_json

    inputs = tf.keras.Input(shape=(98, 40, 1), name="spectrogram")
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="scores")(x)
    model = tf.keras.Model(inputs, outputs)
    entry = _convert(model, (98, 40, 1), 0.0, 1.0, FIXTURES / "kws3_int8.tflite")
    atomic_write_json(FIXTURES / "kws3_int8.report.json", entry)

    atomic_write_json(FIXTURES / "audio_frontend_default.json", AudioFrontendConfig().to_dict())


def build_known_sound() -> None:
    import tensorflow as tf

    from tm_local.utils import atomic_write_json
    from tm_local.yamnet_model import frontend_contract

    inputs = tf.keras.Input(shape=(96, 64), name="log_mel_patch")
    x = tf.keras.layers.Reshape((96, 64, 1))(inputs)
    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(3, activation="sigmoid", name="class_scores")(x)
    model = tf.keras.Model(inputs, outputs)
    # YAMNet log-mel patches are dB-like log values, not a [0, 1] normalized range.
    entry = _convert(model, (96, 64), -6.9, 3.0, FIXTURES / "known_sound3_int8.tflite")
    atomic_write_json(FIXTURES / "known_sound3_int8.report.json", entry)

    atomic_write_json(FIXTURES / "yamnet_frontend.json", frontend_contract())


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    build_kws()
    build_known_sound()
    print("fixtures written to", FIXTURES)


if __name__ == "__main__":
    main()
