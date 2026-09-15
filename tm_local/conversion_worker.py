from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .tflite_export import convert_selected
from .utils import atomic_write_json, sha256_file, utc_now_iso


class ProgressHeartbeat:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.started = time.monotonic()
        self.value = 0.0
        self.message = "Starting TensorFlow Lite conversion…"
        self.state = "running"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "progress": float(self.value),
                "message": self.message,
                "elapsed_seconds": float(time.monotonic() - self.started),
                "pid": os.getpid(),
                "updated_at": utc_now_iso(),
            }

    def _write(self) -> None:
        atomic_write_json(self.path, self._payload())

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self._write()
            except OSError:
                pass

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        self._thread.start()

    def update(self, value: float, message: str) -> None:
        with self._lock:
            self.value = max(self.value, min(1.0, max(0.0, float(value))))
            self.message = str(message)
        self._write()

    def finish(self, state: str, message: str) -> None:
        with self._lock:
            self.state = state
            self.value = 1.0 if state == "completed" else self.value
            self.message = str(message)
        self._write()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        try:
            self._write()
        except OSError:
            pass


def _read_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "keras_path",
        "representative_path",
        "output_dir",
        "prefix",
        "selected",
        "progress_path",
        "result_path",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError("Conversion spec is missing: " + ", ".join(sorted(missing)))
    return payload


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if len(argv) != 1:
        print("Usage: python -m tm_local.conversion_worker spec.json", file=sys.stderr)
        return 2
    spec = _read_spec(Path(argv[0]).resolve())
    heartbeat = ProgressHeartbeat(Path(spec["progress_path"]).resolve())
    heartbeat.start()
    try:
        heartbeat.update(0.005, "Loading the trained Keras model…")
        import tensorflow as tf

        model = tf.keras.models.load_model(str(Path(spec["keras_path"]).resolve()), compile=False)
        with np.load(Path(spec["representative_path"]).resolve(), allow_pickle=False) as archive:
            raw = np.asarray(archive["samples"], dtype=np.float32)
        representatives = [raw[index] for index in range(int(raw.shape[0]))]
        heartbeat.update(0.01, "Model loaded; exporting a stable inference graph…")
        artifacts, report = convert_selected(
            model,
            representatives,
            Path(spec["output_dir"]).resolve(),
            str(spec["prefix"]),
            list(spec["selected"]),
            progress=heartbeat.update,
            source_model_sha256=sha256_file(Path(spec["keras_path"]).resolve()),
        )
        atomic_write_json(
            Path(spec["result_path"]).resolve(),
            {"artifacts": artifacts, "report": report},
        )
        heartbeat.finish("completed", "TensorFlow Lite conversion completed.")
        return 0
    except Exception as exc:
        heartbeat.finish("failed", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        heartbeat.close()


if __name__ == "__main__":
    raise SystemExit(main())
