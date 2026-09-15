from __future__ import annotations

import numpy as np

from tm_local.audio_frontend import (
    AudioFrontendConfig,
    encode_wav_bytes,
    fix_length,
    log_mel_spectrogram,
    read_wav_bytes,
    split_clips,
)


def test_wav_roundtrip_and_feature_shape() -> None:
    config = AudioFrontendConfig()
    t = np.arange(config.clip_samples, dtype=np.float32) / config.sample_rate
    signal = 0.3 * np.sin(2 * np.pi * 523.25 * t)
    payload = encode_wav_bytes(signal, config.sample_rate)
    sample_rate, decoded = read_wav_bytes(payload, config.sample_rate)
    feature = log_mel_spectrogram(decoded, config)
    assert sample_rate == config.sample_rate
    assert decoded.shape == (config.clip_samples,)
    assert feature.shape == config.feature_shape
    assert feature.dtype == np.float32
    assert np.isfinite(feature).all()
    assert float(feature.min()) >= 0.0
    assert float(feature.max()) <= 1.0


def test_split_clips_and_fix_length() -> None:
    signal = np.arange(25, dtype=np.float32)
    clips = split_clips(signal, 10, include_padded_tail=True)
    assert len(clips) == 3
    assert all(clip.shape == (10,) for clip in clips)
    assert np.array_equal(fix_length(np.arange(3, dtype=np.float32), 7)[2:5], np.arange(3))


def test_exported_audio_frontend_reference_matches_runtime(tmp_path) -> None:  # noqa: ANN001
    import importlib.util
    import json

    from tm_local.audio_pipeline import _write_audio_frontend_reference

    config = AudioFrontendConfig()
    models = tmp_path / "models"
    models.mkdir()
    (models / "audio_frontend.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8"
    )
    _write_audio_frontend_reference(models)
    t = np.arange(config.clip_samples, dtype=np.float32) / config.sample_rate
    signal = (0.23 * np.sin(2 * np.pi * 733.0 * t)).astype(np.float32)
    wav_path = models / "probe.wav"
    wav_path.write_bytes(encode_wav_bytes(signal, config.sample_rate))

    spec = importlib.util.spec_from_file_location(
        "exported_audio_frontend", models / "audio_frontend_reference.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exported = module.extract_feature(wav_path)
    _, decoded = read_wav_bytes(wav_path.read_bytes(), config.sample_rate)
    runtime = log_mel_spectrogram(decoded, config)
    assert exported.shape == runtime.shape
    assert np.allclose(exported, runtime, atol=2e-6)
