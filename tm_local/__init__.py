"""Local image/audio/anomalous-sound studio with strict TensorFlow Lite export."""

from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

__version__ = "2.1.0"
