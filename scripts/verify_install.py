from __future__ import annotations

import os
import platform
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> int:
    import numpy as np
    import tensorflow as tf

    print(f"[VERIFY] Python {platform.python_version()}")
    print(f"[VERIFY] TensorFlow {tf.__version__}")
    print(f"[VERIFY] TensorFlow devices: {tf.config.list_physical_devices()}")

    root = Path(tempfile.mkdtemp(prefix="tm_local_verify_"))
    try:
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(4,), dtype=tf.float32, name="features"),
                tf.keras.layers.Dense(6, activation="relu"),
                tf.keras.layers.Dense(2, activation="softmax", name="scores"),
            ],
            name="install_check",
        )
        sample = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        before = model(sample, training=False).numpy()
        keras_path = root / "model.keras"
        model.save(str(keras_path))
        loaded = tf.keras.models.load_model(str(keras_path), compile=False)
        after = loaded(sample, training=False).numpy()
        if not np.allclose(before, after, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Native .keras save/load output mismatch.")
        print("[PASS] Native .keras save/load")

        class InferenceModule(tf.Module):
            def __init__(self, keras_model):  # noqa: ANN001
                super().__init__()
                self.model = keras_model

            @tf.function(
                input_signature=[tf.TensorSpec([None, 4], tf.float32, name="features")]
            )
            def serve(self, features):  # noqa: ANN001
                return {"scores": self.model(features, training=False)}

        saved_model = root / "saved_model"
        module = InferenceModule(loaded)
        tf.saved_model.save(
            module,
            str(saved_model),
            signatures={"serving_default": module.serve.get_concrete_function()},
        )

        def representative_dataset():
            for index in range(12):
                value = np.asarray(
                    [[index / 12.0, 0.25, 0.5, 1.0 - index / 12.0]],
                    dtype=np.float32,
                )
                yield [value]

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite = converter.convert()
        tflite_path = root / "model_int8.tflite"
        tflite_path.write_bytes(tflite)
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_dtype = interpreter.get_input_details()[0]["dtype"]
        output_dtype = interpreter.get_output_details()[0]["dtype"]
        float_tensors = [
            detail["name"]
            for detail in interpreter.get_tensor_details()
            if detail["dtype"] in (np.float16, np.float32, np.float64)
        ]
        if input_dtype != np.int8 or output_dtype != np.int8 or float_tensors:
            raise RuntimeError(
                f"Strict INT8 verification failed: input={input_dtype}, output={output_dtype}, "
                f"float_tensors={len(float_tensors)}"
            )
        print("[PASS] Strict INT8 TensorFlow Lite conversion")
        print("[PASS] Local Studio installation verification completed")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
