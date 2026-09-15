from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tm_local.audio_frontend import encode_wav_bytes
from tm_local.project_store import ProjectStore


@pytest.fixture()
def store(tmp_path: Path) -> ProjectStore:
    return ProjectStore(tmp_path / "workspace" / "projects", tmp_path / "workspace" / "tmp")


@pytest.fixture()
def jpeg_bytes() -> bytes:
    image = Image.new("RGB", (96, 72), (32, 120, 220))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=90)
    return output.getvalue()


@pytest.fixture()
def wav_bytes() -> bytes:
    sample_rate = 16000
    t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    signal = 0.25 * np.sin(2.0 * math.pi * 440.0 * t)
    return encode_wav_bytes(signal.astype(np.float32), sample_rate)
