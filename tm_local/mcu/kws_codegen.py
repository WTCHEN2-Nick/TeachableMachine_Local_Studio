"""KWS (`audio` kind) code generation: front-end tables, a numpy emulation, and the
`main.cpp.in` token table.

Everything the firmware needs to rebuild the Studio's log-mel front-end is *exported from
the Studio's own front-end* rather than recomputed with a second, "equivalent" recipe:
`kws_tables()` calls `audio_frontend.mel_filterbank()` and `np.hanning()` -- the exact
objects `audio_frontend.log_mel_spectrogram()` uses during training, Preview and INT8
calibration -- and serialises them into the C tables. A drifting mel filterbank or a
periodic-vs-symmetric Hann mismatch is invisible on the board (it produces plausible but
systematically wrong scores), so the contract is enforced by tests instead:
`dense_from_tables()` reconstructs the dense filterbank from the sparse tables for a
bit-exact `np.array_equal` comparison, and `emulate_frontend()` replays the algorithm's
post-FFT arithmetic step by step in float32 so the parity test can bound the difference
against `log_mel_spectrogram()`. The FFT itself is not modelled (numpy computes it in
float64, the board uses CMSIS-DSP `arm_rfft_fast_f32`), so the guarantee those tests give
is "same tables, same recipe", with on-device agreement expected within <= 1 int8 LSB and
confirmed only by Task 7's on-target comparison -- see `emulate_frontend()`.

Stays TensorFlow-free at import time (numpy + `..audio_frontend` only), so the parity test
and `deploy_service` can import it without paying for a TF import.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..audio_frontend import AudioFrontendConfig, config_from_mapping, mel_filterbank
from .codegen import cpp_escape, render_tokens
from .errors import DeployError

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the runtime import graph flat
    from .contract import DeployContract

# Substring hints for "this class means 'nothing was said'". Matched against the project's
# own class names lowercased, in project order; the first hit wins. `str.lower()` is a
# no-op on CJK, so one substring test covers both the case-insensitive ASCII hints and the
# Traditional-Chinese ones students actually type. `_silence_` (the Speech Commands corpus
# spelling) was dropped as redundant: substring matching means "silence" already covers it.
# "背景音" is likewise subsumed by "背景" and kept only as documentation of the vocabulary.
#
# A project with NO matching class is a real configuration, not an error -- see
# `has_background()` for why the firmware must be told which case it is in.
BACKGROUND_HINTS = (
    "background",
    "silence",
    "noise",
    "背景",
    "背景音",
    "環境音",
    "安靜",
    "無聲",
    "雜音",
)

# The float32 clamp applied before `10*log10()`, mirroring `log_mel_spectrogram()`'s own
# `np.maximum(mel_power, 1e-10)`. Exported as the LOG_EPSILON token so the C code cannot
# silently pick a different floor.
DEFAULT_LOG_EPSILON = 1e-10


@dataclass(frozen=True)
class KwsTables:
    """The front-end constants the firmware embeds, in the exact layout the C code indexes.

    `hann` is the symmetric `np.hanning(window_samples)` (NOT the periodic window some
    STFT libraries default to) cast to float32.

    The mel filterbank is stored sparsely, per mel bin `m`: the filter touches spectrum
    bins `[mel_start[m], mel_start[m] + mel_count[m])`, whose weights live at
    `mel_weights[mel_offset[m] : mel_offset[m] + mel_count[m]]`. For a 512-point RFFT the
    dense matrix would be `mel_bins x 257` floats (40 KiB at 40 mel bins); the sparse form
    is ~1/6 of that and is what makes the per-frame mel accumulation a short inner loop.
    """

    hann: np.ndarray
    mel_start: list[int]
    mel_count: list[int]
    mel_offset: list[int]
    mel_weights: list[float]


def kws_tables(config: AudioFrontendConfig) -> KwsTables:
    """Export the Studio's own Hann window and mel filterbank as the firmware's C tables."""

    filters = mel_filterbank(
        config.sample_rate,
        config.fft_size,
        config.mel_bins,
        float(config.fmin),
        float(config.fmax),
    )
    starts: list[int] = []
    counts: list[int] = []
    offsets: list[int] = []
    weights: list[float] = []
    for row in filters:
        nonzero = np.flatnonzero(row)
        if nonzero.size == 0:
            # An all-zero row happens at the top of the range when two adjacent mel points
            # land on the same (clipped) FFT bin. Emit a one-element zero span rather than
            # a zero-length one so the C inner loop never has to special-case count == 0.
            start, count = 0, 1
        else:
            start, count = int(nonzero[0]), int(nonzero[-1] - nonzero[0] + 1)
        starts.append(start)
        counts.append(count)
        offsets.append(len(weights))
        weights.extend(float(v) for v in row[start : start + count].astype(np.float32))
    hann = np.hanning(config.window_samples).astype(np.float32)
    return KwsTables(hann, starts, counts, offsets, weights)


def dense_from_tables(tables: KwsTables, mel_bins: int, spectrum_bins: int) -> np.ndarray:
    """Rebuild the dense `(mel_bins, spectrum_bins)` filterbank from the sparse tables.

    Test-only inverse of `kws_tables()`: it is what lets the parity test assert
    `np.array_equal(dense, mel_filterbank(...))`, i.e. that the sparse export is lossless.
    """

    dense = np.zeros((mel_bins, spectrum_bins), dtype=np.float32)
    for mel in range(mel_bins):
        start = tables.mel_start[mel]
        count = tables.mel_count[mel]
        offset = tables.mel_offset[mel]
        span = np.asarray(tables.mel_weights[offset : offset + count], dtype=np.float32)
        dense[mel, start : start + count] = span
    return dense


def emulate_frontend(
    pcm_int16: np.ndarray,
    config: AudioFrontendConfig,
    tables: KwsTables,
    scale: float,
    zero_point: int,
    log_epsilon: float = DEFAULT_LOG_EPSILON,
) -> np.ndarray:
    """numpy model of the C front-end in `mcu_toolkit/apps/audio_kws/main.cpp.in`.

    Follows the C data flow rather than the shortest numpy spelling: int16 PCM divided by
    32768, symmetric Hann, zero-padded to `fft_size`, real FFT, power as `re*re + im*im`
    accumulated in float32 (the C code has no float64 anywhere), sparse mel projection,
    `10*log10(max(p, log_epsilon))`, minus the clip's own maximum, clipped to
    `[db_floor, 0]`, mapped to `[0, 1]`, then quantised with round-half-even (`lrintf` under
    the default FE_TONEAREST rounding mode) and saturated to int8.

    **What the parity test actually proves.** Two things, and not a third:

    1. The exported tables reproduce `audio_frontend.mel_filterbank()` and
       `np.hanning(window_samples)` *exactly* -- `dense_from_tables()` round-trips to a
       bit-identical float32 array.
    2. The post-FFT arithmetic mirrors `audio_frontend.log_mel_spectrogram()` step by step,
       so training, Preview, INT8 calibration and this emulation share one recipe.

    The FFT itself is NOT modelled: this function calls `np.fft.rfft`, which computes in
    float64, while the firmware calls CMSIS-DSP `arm_rfft_fast_f32` in float32. So a 0 LSB
    result here is a statement about the tables and the surrounding arithmetic, not a
    bit-exactness claim about the device. On-device agreement is expected to be within
    <= 1 LSB (Task 7's tolerance), and only Task 7's on-target comparison can confirm it.

    Returns `(frame_count, mel_bins)` int8 -- the tensor the firmware hands to TFLM.
    """

    if config.clip_samples < config.window_samples:
        # frame_count would collapse to 1 and the single frame would be shorter than the
        # window, which numpy reports only as an opaque broadcast error.
        raise DeployError(
            f"clip_samples {config.clip_samples} 小於 window_samples "
            f"{config.window_samples}，無法切出完整音框；請調高 clip_seconds 或調低 window_ms"
        )
    pcm = np.asarray(pcm_int16, dtype=np.int16)
    if pcm.size != config.clip_samples:
        raise DeployError(
            f"PCM 長度 {pcm.size} 與 clip_samples {config.clip_samples} 不符，無法模擬前端"
        )
    signal = pcm.astype(np.float32) / np.float32(32768.0)

    frames = np.zeros((config.frame_count, config.fft_size), dtype=np.float32)
    for frame in range(config.frame_count):
        offset = frame * config.hop_samples
        window = signal[offset : offset + config.window_samples]
        frames[frame, : config.window_samples] = window * tables.hann

    spectrum = np.fft.rfft(frames, n=config.fft_size, axis=1)
    real = spectrum.real.astype(np.float32)
    imag = spectrum.imag.astype(np.float32)
    power = (real * real + imag * imag).astype(np.float32)

    dense = dense_from_tables(tables, config.mel_bins, config.fft_size // 2 + 1)
    mel_power = np.maximum(power @ dense.T, np.float32(log_epsilon))
    db = (10.0 * np.log10(mel_power)).astype(np.float32)
    relative = np.clip(db - np.max(db), np.float32(config.db_floor), np.float32(0.0))
    # Dividing by a bare `-db_floor`, without `log_mel_spectrogram()`'s `max(1e-6, ...)`
    # divide-by-zero guard, is deliberate: the App Builder C divides by the plain constant,
    # and `contract.validate()` pins db_floor to -80 for every deployable project, so the
    # guard can never engage. Reproducing it here would model arithmetic the board does not
    # perform. A future kind that allows db_floor == 0 must add the guard on BOTH sides.
    normalized = (relative - np.float32(config.db_floor)) / np.float32(-config.db_floor)

    quantized = np.rint(normalized / np.float32(scale)).astype(np.int32) + int(zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def _matched_background(labels: Sequence[str]) -> int | None:
    """Index of the first class whose name reads as background/silence, else None."""

    for index, label in enumerate(labels):
        lowered = str(label).lower()
        if any(hint in lowered for hint in BACKGROUND_HINTS):
            return index
    return None


def background_index(labels: Sequence[str]) -> int:
    """Index of the project's background/silence class, or 0 when it has none.

    The 0 fallback is only meaningful together with `has_background()`: the firmware uses
    this index BOTH as the class that must never trigger an action AND as the trigger-margin
    baseline (`average[best] >= average[BACKGROUND_INDEX] + TRIGGER_MARGIN`). For a project
    like `["yes", "no"]` with no background class, taking the fallback at face value would
    silently make "yes" untriggerable and turn it into its own margin reference. So callers
    must gate both uses on `HAS_BACKGROUND`; this function only supplies the index to use
    when there IS one.
    """

    matched = _matched_background(labels)
    return 0 if matched is None else matched


def has_background(labels: Sequence[str]) -> bool:
    """Whether any class name reads as background/silence -- see `background_index()`."""

    return _matched_background(labels) is not None


def float_literal(value: float) -> str:
    """Render a Python float as a C `float` literal (`0.5f`, `-80.0f`, `1e-10f`).

    `%.10g` keeps every float32 value exact (float32 needs 9 significant digits to
    round-trip) while staying readable. The `.0` is appended when `%g` produced a bare
    integer so the literal cannot be mistaken for an `int` in C's usual-arithmetic rules,
    and the `f` suffix keeps it single precision -- without it a `double` literal would
    silently promote whole expressions in the firmware's front-end to double.
    """

    text = f"{float(value):.10g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text + "f"


def c_array(
    values: Sequence[Any],
    formatter: Callable[[Any], str],
    width: int = 10,
    indent: str = "    ",
) -> str:
    """Render `values` as the body of a C array initialiser: `width` per line, indented."""

    rows: list[str] = []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        rows.append(indent + ", ".join(formatter(v) for v in chunk))
    return ",\n".join(rows)


def _c_string(value: str) -> str:
    return '"' + cpp_escape(value) + '"'


# `contract.collect()` copies kws_runtime values straight out of the project's training
# settings, so anything a hand-edited project.json holds -- a string, null, NaN, a negative
# hop -- arrives here unchecked. Left unvalidated it would be baked into the firmware as a
# C literal and surface as a board that never triggers, or as a compile error deep inside a
# generated file. These helpers turn each one into a student-facing DeployError naming the
# exact key instead.
def _runtime_number(runtime: dict[str, Any], key: str, default: float) -> float:
    value = runtime.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeployError(f"kws_runtime 的 {key} 必須是數字，目前是 {value!r}；請修正訓練設定")
    number = float(value)
    if not math.isfinite(number):
        raise DeployError(f"kws_runtime 的 {key} 必須是有限數字，目前是 {value!r}；請修正訓練設定")
    return number


def _runtime_count(runtime: dict[str, Any], key: str, default: int) -> int:
    number = _runtime_number(runtime, key, default)
    if number < 1 or number != int(number):
        raise DeployError(
            f"kws_runtime 的 {key} 必須是大於 0 的整數，目前是 {number}；請修正訓練設定"
        )
    return int(number)


def _runtime_ranged(
    runtime: dict[str, Any], key: str, default: float, low: float, high: float, described: str
) -> float:
    number = _runtime_number(runtime, key, default)
    if not low <= number <= high:
        raise DeployError(f"kws_runtime 的 {key} 必須{described}，目前是 {number}；請修正訓練設定")
    return number


def kws_tokens(contract: DeployContract, arena_bytes: int) -> dict[str, str]:
    """Build the `@@TOKEN@@` table `mcu_toolkit/apps/audio_kws/main.cpp.in` consumes.

    Every value is already rendered as C source text (literals carry their `f` suffix,
    arrays are full initialiser bodies), so the template only ever substitutes strings.
    """

    if contract.audio_frontend is None:
        raise DeployError("audio 專案缺少 audio_frontend.json，請重新 Export")
    try:
        config = config_from_mapping(contract.audio_frontend)
    except (KeyError, TypeError, ValueError) as exc:
        raise DeployError("audio_frontend.json 內容不合法，請重新 Export") from exc

    tables = kws_tables(config)
    runtime = contract.kws_runtime
    if not isinstance(runtime, dict):
        raise DeployError("kws_runtime 設定格式錯誤，請修正訓練設定")

    # A hop longer than the clip would leave gaps of audio no window ever sees; a hop of 0
    # would make the firmware re-score the same window forever. Both bounds are exclusive
    # at the bottom, so the range helper cannot express them.
    hop_seconds = _runtime_number(runtime, "inference_hop_seconds", 0.25)
    if not 0.0 < hop_seconds <= config.clip_seconds:
        raise DeployError(
            f"kws_runtime 的 inference_hop_seconds 必須大於 0 且不超過 clip_seconds"
            f"（{config.clip_seconds}），目前是 {hop_seconds}；請修正訓練設定"
        )
    smooth_windows = _runtime_count(runtime, "smooth_windows", 4)
    required_hits = _runtime_count(runtime, "required_hits", 2)
    trigger_margin = _runtime_ranged(runtime, "trigger_margin", 0.20, 0.0, 1.0, "介於 0 與 1 之間")
    rearm_score = _runtime_ranged(runtime, "rearm_score", 0.40, 0.0, 1.0, "介於 0 與 1 之間")
    rms_gate_dbfs = _runtime_ranged(
        runtime, "rms_gate_dbfs", -55.0, -120.0, 0.0, "是不大於 0 的 dBFS 值（-120 ~ 0）"
    )
    # Not merely "positive": this floor is part of the front-end contract, not a tunable. The
    # firmware's log-mel has to be the same function as `audio_frontend.log_mel_spectrogram()`,
    # which hard-codes `np.maximum(mel_power, 1e-10)`; a different floor here would silently
    # shift every quiet mel bin on the board relative to the spectrograms the model trained on.
    log_epsilon = _runtime_number(runtime, "log_epsilon", DEFAULT_LOG_EPSILON)
    if log_epsilon != DEFAULT_LOG_EPSILON:
        raise DeployError(
            f"kws_runtime 的 log_epsilon 必須與訓練前處理一致（{DEFAULT_LOG_EPSILON}），"
            f"目前是 {log_epsilon}；請修正訓練設定"
        )
    if required_hits > smooth_windows:
        raise DeployError(
            f"kws_runtime 的 required_hits（{required_hits}）不可大於 smooth_windows"
            f"（{smooth_windows}），否則永遠不會觸發；請修正訓練設定"
        )

    # One inference every `inference_hop` new samples; >= 1 so a hop that rounds to zero
    # cannot make the firmware's ring buffer advance by nothing and spin on one window.
    inference_hop = max(1, round(hop_seconds * config.sample_rate))
    # Below 0.05 every frame trips; above 0.99 an int8 softmax (max score 127 -> ~0.996)
    # can never reach the threshold and the board would look dead.
    threshold = min(0.99, max(0.05, float(contract.detection_threshold)))
    # An explicit background class chosen in the Studio wins over the name hints: it is what
    # training actually weighted (audio_pipeline.background_class_index()), and the two sides
    # disagreeing is invisible on the PC and wrong on the board. None means "the project did
    # not record one", which is every project trained before this field existed.
    explicit_background = contract.background_index
    if explicit_background is None:
        background = background_index(contract.labels)
        has_background_class = has_background(contract.labels)
    else:
        background = int(explicit_background)
        if not 0 <= background < len(contract.labels):
            raise DeployError(
                f"background_class_id 指到第 {background} 個類別，但專案只有 "
                f"{len(contract.labels)} 個類別；請重新選擇背景類別後重新訓練"
            )
        has_background_class = True

    dmic = contract.board.audio_dmic
    return {
        "ARENA": str(int(arena_bytes)),
        "SAMPLE_RATE": str(config.sample_rate),
        "CLIP_SAMPLES": str(config.clip_samples),
        "INFERENCE_HOP": str(inference_hop),
        "FRAME_LENGTH": str(config.window_samples),
        "FRAME_STEP": str(config.hop_samples),
        "FRAME_COUNT": str(config.frame_count),
        "FFT_LENGTH": str(config.fft_size),
        "MEL_BINS": str(config.mel_bins),
        "MEL_NNZ": str(len(tables.mel_weights)),
        "LABEL_COUNT": str(len(contract.labels)),
        "BACKGROUND_INDEX": str(background),
        # Whether BACKGROUND_INDEX names a real background class. The firmware uses that
        # index both to suppress triggering and as the trigger-margin baseline; on a project
        # like ["yes", "no"] the 0 fallback means neither use is valid, so the template must
        # gate both on `#if NUML_HAS_BACKGROUND`.
        "HAS_BACKGROUND": "1" if has_background_class else "0",
        "SMOOTH_WINDOWS": str(smooth_windows),
        "REQUIRED_HITS": str(required_hits),
        # After a trigger, mute for one full clip's worth of hops: the windows that still
        # overlap the keyword would otherwise re-fire on the same utterance.
        "COOLDOWN_HOPS": str(math.ceil(config.clip_samples / inference_hop)),
        "TRIGGER_THRESHOLD": float_literal(threshold),
        "TRIGGER_MARGIN": float_literal(trigger_margin),
        "REARM_SCORE": float_literal(rearm_score),
        "RMS_GATE_DBFS": float_literal(rms_gate_dbfs),
        "DB_FLOOR": float_literal(config.db_floor),
        "LOG_EPSILON": float_literal(log_epsilon),
        "INPUT_SCALE": float_literal(contract.input.scale),
        "INPUT_ZERO_POINT": str(int(contract.input.zero_point)),
        "OUTPUT_SCALE": float_literal(contract.output.scale),
        "OUTPUT_ZERO_POINT": str(int(contract.output.zero_point)),
        "LABELS": c_array(list(contract.labels), _c_string),
        "HANN": c_array(tables.hann.tolist(), float_literal),
        "MEL_START": c_array(tables.mel_start, str),
        "MEL_COUNT": c_array(tables.mel_count, str),
        "MEL_OFFSET": c_array(tables.mel_offset, str),
        "MEL_WEIGHTS": c_array(tables.mel_weights, float_literal),
        "DMIC_CLK_MACRO": dmic.clk_macro,
        "DMIC_DAT_MACRO": dmic.dat_macro,
        "DMIC_CHANNEL_MASK": dmic.channel_mask,
    }


def render_kws_main(contract: DeployContract, arena_bytes: int, template_path: Path) -> str:
    """Render `main.cpp.in`; a token the table does not supply raises `DeployError`."""

    text = Path(template_path).read_text(encoding="utf-8")
    return render_tokens(text, kws_tokens(contract, arena_bytes))
