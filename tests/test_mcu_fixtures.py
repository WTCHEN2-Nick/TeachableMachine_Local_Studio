from __future__ import annotations

import hashlib
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mcu"


def _entry(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.report.json").read_text(encoding="utf-8"))


def test_kws_fixture_contract() -> None:
    entry = _entry("kws3_int8")
    assert entry["inputs"][0]["shape"] == [1, 98, 40, 1] and entry["inputs"][0]["dtype"] == "int8"
    assert entry["outputs"][0]["shape"] == [1, 3] and entry["outputs"][0]["dtype"] == "int8"
    assert entry["strict_full_integer"] is True
    assert entry["sha256"] == hashlib.sha256((FIXTURES / "kws3_int8.tflite").read_bytes()).hexdigest()
    frontend = json.loads((FIXTURES / "audio_frontend_default.json").read_text(encoding="utf-8"))
    assert frontend["frame_count"] == 98 and frontend["mel_bins"] == 40 and frontend["sample_rate"] == 16000


def test_known_sound_fixture_contract() -> None:
    entry = _entry("known_sound3_int8")
    assert entry["inputs"][0]["shape"] == [1, 96, 64]
    assert entry["outputs"][0]["shape"] == [1, 3]
    assert entry["strict_full_integer"] is True
    assert "LOGISTIC" in entry["operators"]
    frontend = json.loads((FIXTURES / "yamnet_frontend.json").read_text(encoding="utf-8"))
    assert frontend["mel_bands"] == 64 and frontend["patch_frames"] == 96
