from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from .tflite_export import predict_tflite


class ModelRuntime:
    def __init__(self):
        self._lock = threading.RLock()
        self._keras_cache: dict[str, tuple[float, Any]] = {}

    def _load_keras(self, path: Path):
        import tensorflow as tf

        key = str(path.resolve())
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._keras_cache.get(key)
            if cached and cached[0] == mtime:
                return cached[1]
            model = tf.keras.models.load_model(path, compile=False)
            self._keras_cache[key] = (mtime, model)
            return model

    def predict(self, project_dir: Path, artifact_name: str, batch: np.ndarray) -> np.ndarray:
        models_dir = project_dir / "models"
        path = models_dir / artifact_name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".tflite":
            return predict_tflite(path, np.asarray(batch, dtype=np.float32))
        model = self._load_keras(path)
        return np.asarray(model.predict(batch, verbose=0), dtype=np.float32)

    def clear_project(self, project_dir: Path) -> None:
        models_root = (project_dir / "models").resolve()
        with self._lock:
            for key in list(self._keras_cache):
                try:
                    Path(key).resolve().relative_to(models_root)
                except ValueError:
                    continue
                self._keras_cache.pop(key, None)
