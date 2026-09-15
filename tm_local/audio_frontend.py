from __future__ import annotations

import io
import math
import wave
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioFrontendConfig:
    sample_rate: int = 16000
    clip_seconds: float = 1.0
    window_ms: float = 25.0
    hop_ms: float = 10.0
    fft_size: int = 512
    mel_bins: int = 40
    fmin: float = 20.0
    fmax: float = 8000.0
    db_floor: float = -80.0

    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_ms / 1000.0))

    @property
    def hop_samples(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def frame_count(self) -> int:
        if self.clip_samples < self.window_samples:
            return 1
        return 1 + (self.clip_samples - self.window_samples) // self.hop_samples

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return (self.frame_count, self.mel_bins, 1)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "clip_samples": self.clip_samples,
                "window_samples": self.window_samples,
                "hop_samples": self.hop_samples,
                "frame_count": self.frame_count,
                "feature_shape": list(self.feature_shape),
                "normalization": "10*log10(mel_power), clipped to [db_floor, 0], mapped to [0, 1]",
            }
        )
        return payload


def config_from_mapping(mapping: dict) -> AudioFrontendConfig:
    fields = {key: mapping[key] for key in AudioFrontendConfig.__dataclass_fields__ if key in mapping}
    config = AudioFrontendConfig(**fields)
    if config.fft_size < config.window_samples:
        raise ValueError("fft_size must be greater than or equal to the audio window size.")
    if config.fmax > config.sample_rate / 2 + 1e-6:
        config = AudioFrontendConfig(**{**asdict(config), "fmax": config.sample_rate / 2})
    return config


def _scale_pcm_dtype(data: np.ndarray, dtype: np.dtype, out_dtype: type) -> np.ndarray:
    """Map raw WAV samples onto the [-1, 1] scale implied by their source dtype.

    Signed integers divide by 32768 (``max(|info.min|, info.max)``), not 32767; the MCU
    mean-square bounds are derived from that same constant, so it must not drift.
    """

    if np.issubdtype(dtype, np.floating):
        return data.astype(out_dtype)
    if dtype == np.uint8:
        return (data.astype(out_dtype) - 128.0) / 128.0
    if np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        scale = float(max(abs(info.min), info.max))
        return data.astype(out_dtype) / scale
    raise ValueError(f"Unsupported WAV dtype: {dtype}")


def _normalize_pcm(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 2:
        # Scale first, mix second. Averaging raw integer channels leaves the values on the
        # +-32768 scale, which then satisfies `issubdtype(floating)`, skips the integer
        # rescale and is finally clipped into a full-scale square wave -- silently pinning
        # such an upload at ~0 dBFS and poisoning every level statistic derived from it.
        result = _scale_pcm_dtype(data, data.dtype, np.float64).mean(axis=1).astype(np.float32)
    else:
        result = _scale_pcm_dtype(data, data.dtype, np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def read_wav_bytes(payload: bytes, target_sample_rate: int | None = None) -> tuple[int, np.ndarray]:
    if not payload or len(payload) < 44:
        raise ValueError("Audio payload is empty or too small to be a WAV file.")
    try:
        sample_rate, data = wavfile.read(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError("Cannot decode WAV. Browser uploads are converted to PCM WAV before sending.") from exc
    signal = _normalize_pcm(np.asarray(data))
    if sample_rate <= 0:
        raise ValueError("Invalid WAV sample rate.")
    if target_sample_rate and sample_rate != target_sample_rate:
        gcd = math.gcd(int(sample_rate), int(target_sample_rate))
        signal = resample_poly(signal, target_sample_rate // gcd, sample_rate // gcd).astype(np.float32)
        sample_rate = target_sample_rate
    return int(sample_rate), signal


def read_wav_file(path: Path, target_sample_rate: int | None = None) -> tuple[int, np.ndarray]:
    return read_wav_bytes(path.read_bytes(), target_sample_rate)


def encode_wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return output.getvalue()


def fix_length(signal: np.ndarray, length: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == length:
        return signal
    if signal.size > length:
        start = max(0, (signal.size - length) // 2)
        return signal[start : start + length]
    output = np.zeros(length, dtype=np.float32)
    start = (length - signal.size) // 2
    output[start : start + signal.size] = signal
    return output


def split_clips(
    signal: np.ndarray,
    clip_samples: int,
    *,
    overlap: float = 0.0,
    include_padded_tail: bool = True,
) -> list[np.ndarray]:
    if clip_samples <= 0:
        raise ValueError("clip_samples must be positive.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size <= clip_samples:
        return [fix_length(signal, clip_samples)]
    step = max(1, int(round(clip_samples * (1.0 - overlap))))
    clips: list[np.ndarray] = []
    for start in range(0, signal.size - clip_samples + 1, step):
        clips.append(signal[start : start + clip_samples].copy())
    last_end = (0 if not clips else (len(clips) - 1) * step + clip_samples)
    if include_padded_tail and last_end < signal.size:
        clips.append(fix_length(signal[-clip_samples:], clip_samples))
    return clips


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def mel_filterbank(
    sample_rate: int,
    fft_size: int,
    mel_bins: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    fmax = min(float(fmax), sample_rate / 2.0)
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), mel_bins + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, fft_size // 2)
    filters = np.zeros((mel_bins, fft_size // 2 + 1), dtype=np.float32)
    for index in range(1, mel_bins + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]
        if center <= left:
            center = min(left + 1, fft_size // 2)
        if right <= center:
            right = min(center + 1, fft_size // 2)
        for position in range(left, center):
            filters[index - 1, position] = (position - left) / max(1, center - left)
        for position in range(center, right):
            filters[index - 1, position] = (right - position) / max(1, right - center)
    normalizer = filters.sum(axis=1, keepdims=True)
    filters = filters / np.maximum(normalizer, 1e-12)
    return filters.astype(np.float32)


def log_mel_spectrogram(signal: np.ndarray, config: AudioFrontendConfig) -> np.ndarray:
    signal = fix_length(signal, config.clip_samples)
    window_size = config.window_samples
    hop_size = config.hop_samples
    frame_count = config.frame_count
    padded = np.pad(signal, (0, max(0, window_size + (frame_count - 1) * hop_size - signal.size)))
    shape = (frame_count, window_size)
    strides = (padded.strides[0] * hop_size, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides).copy()
    frames *= np.hanning(window_size).astype(np.float32)[None, :]
    spectrum = np.fft.rfft(frames, n=config.fft_size, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    filters = mel_filterbank(
        config.sample_rate,
        config.fft_size,
        config.mel_bins,
        float(config.fmin),
        float(config.fmax),
    )
    mel_power = power @ filters.T
    db = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    db -= np.max(db) if db.size else 0.0
    db = np.clip(db, config.db_floor, 0.0)
    normalized = (db - config.db_floor) / max(1e-6, -config.db_floor)
    return normalized.astype(np.float32)[..., np.newaxis]


# The largest magnitude a full-scale int16 sample reaches after `_scale_pcm_dtype()`
# divides by 32768. Used only to count clipped samples as a recording-quality
# diagnostic; it never enters an anomaly score.
CLIPPING_LEVEL = 32767.0 / 32768.0


def rms_dbfs(signal: np.ndarray, config: AudioFrontendConfig) -> float:
    """Absolute loudness of one clip, in dBFS -- the level branch of the anomaly scorer.

    `log_mel_spectrogram()` subtracts its own maximum (``db -= np.max(db)``), so the shape
    branch is blind to absolute loudness by construction: a signal 10 dB louder with an
    identical spectral shape yields a bit-identical tensor. Loudness is therefore carried
    by this separate scalar instead of by changing the frontend.

    Measured on exactly the buffer `log_mel_spectrogram()` analyses -- after resampling and
    after `fix_length(signal, config.clip_samples)`, before the STFT -- so the server, the
    exported runner and the MCU all score the same samples.
    """

    values = fix_length(signal, config.clip_samples).astype(np.float64)
    rms = float(np.sqrt(np.mean(values ** 2))) if values.size else 0.0
    if not math.isfinite(rms):
        # A broken clip must read as silence, never as full scale: returning 0 dBFS here
        # would drag C_level / T_level to the top of the range with no error message.
        rms = 0.0
    return float(np.clip(20.0 * math.log10(max(rms, 1e-5)), -100.0, 0.0))


def level_diagnostics(
    signal: np.ndarray,
    config: AudioFrontendConfig,
    *,
    source_samples: int | None = None,
) -> dict:
    """Recording-quality diagnostics for one clip.

    Only ``rms_dbfs`` may be scored. ``peak_dbfs`` and ``clipping_fraction`` are review
    signals about the microphone and gain, never terms in the composite ratio.

    Keys:
      ``rms_dbfs``          identical to `rms_dbfs()`.
      ``peak_dbfs``         peak sample level in dBFS, floored at -100.
      ``clipping_fraction`` fraction of analysed samples at or above full scale.
      ``pad_fraction``      fraction of the analysed buffer that is zero padding added by
                            `fix_length()`. Centre-padding a short recording lowers its RMS
                            while leaving the peak-normalised log-mel untouched, i.e. it
                            fabricates a level anomaly the shape branch cannot contradict.
                            Callers must exclude clips with ``pad_fraction > 0`` from the
                            C_level / T_level calibration.

    Pass ``source_samples`` (the length of the recording the clip came from) when the clip
    has already been through `split_clips()` / `fix_length()`; padding is no longer visible
    in such a buffer and ``pad_fraction`` would otherwise read 0.0.
    """

    raw = np.asarray(signal, dtype=np.float32).reshape(-1)
    length = int(config.clip_samples)
    available = int(raw.size if source_samples is None else max(0, int(source_samples)))
    pad_fraction = 0.0 if available >= length else (length - available) / float(max(1, length))
    values = fix_length(raw, length).astype(np.float64)
    if values.size:
        peak = float(np.max(np.abs(values)))
        clipped = int(np.count_nonzero(np.abs(values) >= CLIPPING_LEVEL))
        clipping_fraction = clipped / float(values.size)
    else:
        peak = 0.0
        clipping_fraction = 0.0
    if not math.isfinite(peak):
        # Surface a non-finite clip as hot rather than hiding it as silence.
        peak = 1.0
    return {
        "rms_dbfs": rms_dbfs(raw, config),
        "peak_dbfs": float(np.clip(20.0 * math.log10(max(peak, 1e-5)), -100.0, 0.0)),
        "clipping_fraction": float(clipping_fraction),
        "pad_fraction": float(pad_fraction),
    }


def spectrogram_thumbnail(feature: np.ndarray, output: Path, *, width: int = 160, height: int = 92) -> None:
    values = np.asarray(feature, dtype=np.float32)
    if values.ndim == 3:
        values = values[..., 0]
    values = np.clip(values, 0.0, 1.0)
    # A compact blue-purple-orange palette similar to the familiar audio sample tiles.
    anchors = np.array(
        [
            [4, 16, 38],
            [10, 46, 91],
            [52, 45, 134],
            [117, 64, 145],
            [224, 97, 70],
            [255, 190, 80],
        ],
        dtype=np.float32,
    )
    scaled = values * (len(anchors) - 1)
    low = np.floor(scaled).astype(int)
    high = np.clip(low + 1, 0, len(anchors) - 1)
    fraction = (scaled - low)[..., None]
    rgb = anchors[low] * (1.0 - fraction) + anchors[high] * fraction
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # Time grows left-to-right and frequency bottom-to-top.
    image = Image.fromarray(np.flipud(np.transpose(rgb, (1, 0, 2))), mode="RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def batch_features(paths: Iterable[Path], config: AudioFrontendConfig) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for path in paths:
        _, signal = read_wav_file(path, config.sample_rate)
        result.append(log_mel_spectrogram(signal, config))
    return result
