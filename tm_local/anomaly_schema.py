"""Frozen contract for ``anomaly_config.json`` (Abnormal Sound open-set detector).

This module is the SINGLE SOURCE OF TRUTH for the detector metadata that is shared by
three independent implementations which must agree bit-for-bit:

1. the server (training / calibration / preview inside ``tm_local``),
2. the exported PC-side Python runner that ships inside the export ZIP,
3. the MCU documentation and its C implementation.

Nothing here trains, converts or scores real audio.  It only *defines and enforces* the
contract, so it deliberately depends on the standard library and ``typing`` alone --
importing this module must never pull in tensorflow or numpy.  Every consumer imports it,
including the ones that run without TensorFlow installed.

What the contract is, in one paragraph
--------------------------------------
Train on NORMAL audio only.  A dense denoising autoencoder
``200 -> 128 -> 64 -> 8 -> 64 -> 128 -> 200`` (ReLU hidden, **sigmoid** output, **no**
BatchNormalization, 69,200 parameters) reconstructs 5-frame contexts taken from the
UNCHANGED ``audio_frontend.log_mel_spectrogram()`` output ``(98, 40, 1)``.  That gives 94
contexts of 200 dims per one-second clip.  ``S_shape`` is the arithmetic mean of the
largest 10 per-context reconstruction MSEs.  Because the frontend subtracts its own
maximum (``db -= np.max(db)``) it is blind to absolute loudness -- a signal 10 dB louder
with identical spectral shape produces a bit-identical tensor (measured AUC 0.421, worse
than chance) -- so loudness is carried by a separate scalar ``L`` (RMS dBFS of the PCM,
computed after resample and ``fix_length(16000)`` and before the STFT).  The composite is
``R = max(S_shape / T_shape[runtime], abs(L - C_level) / T_level)`` and ``R > 1`` means the
window deviates from the collected normal baseline.  The detector never names the anomaly;
it is not a classifier.

Why sigmoid is load-bearing (do not "simplify" it to linear)
------------------------------------------------------------
A sigmoid output pins the TFLite int8 output quantization to a fixed scale of exactly
``1/256 = 0.00390625``.  Measured on real GestureAI DMIC recordings this dropped the
quantization noise floor from ``2.002e-04`` (linear output, BatchNorm architecture) to
``2.393e-06``, improving the score margin ``(P95 - P50) / noise_floor`` from 30.5x to
81.9x.  A linear output derives its scale from the representative data and is unbounded.

Public surface
--------------
``SCORER_SCHEMA_VERSION`` / ``FRONTEND_SCHEMA_VERSION``, the dataclass tree
(:class:`AnomalyConfig` and its parts), :func:`build_config`, :func:`to_dict`,
:func:`from_dict`, :func:`dumps`, :func:`loads`, :func:`validate`,
:func:`fpr_upper_bound_95`, :func:`calibration_confidence`, :func:`binding_tuple`,
:func:`binding_fingerprint`, :func:`verify_binding`, plus the small pure-Python scoring and
split helpers that exist so that no consumer has to re-derive a constant.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any, Final, Iterable, Mapping, Sequence

__all__ = [
    "SCORER_SCHEMA_VERSION",
    "FRONTEND_SCHEMA_VERSION",
    "FRONTEND_CONTRACT",
    "FEATURE_RELATIVE_LOG_MEL",
    "CONTEXT_FRAMES",
    "N_CONTEXTS",
    "VECTOR_DIM",
    "TOP_K",
    "AGGREGATION_MEAN_OF_LARGEST_K",
    "INTEGER_MSE_DEFINITION",
    "LEVEL_STATISTIC",
    "LEVEL_COMPUTED_AT",
    "LEVEL_FLOOR_AMPLITUDE",
    "LEVEL_CLIP_DB",
    "MIN_T_LEVEL_DB",
    "COMPOSITE_RULE",
    "COMPOSITE_EXCEED_WHEN",
    "HOP_SECONDS",
    "WINDOW_COUNT",
    "WARMUP_STATE",
    "STATE_NORMAL",
    "STATE_ABNORMAL",
    "STATE_UNCERTAIN",
    "SENSITIVITY_PRESETS",
    "RUNTIME_NAMES",
    "QUANTIZED_RUNTIMES",
    "CONFIDENCE_CALIBRATED",
    "CONFIDENCE_LOW",
    "ROLE_NORMAL_TRAIN",
    "ROLE_ANOMALY_EVAL",
    "ROLE_NAMES",
    "MAX_ANOMALY_EVAL_GROUPS",
    "QC_OK",
    "QC_WARNING",
    "QC_HIGH_REVIEW",
    "QC_STATUSES",
    "TRAIN_MIN_SESSIONS",
    "TRAIN_MIN_SECONDS",
    "CALIBRATED_MIN_SESSIONS",
    "CALIBRATED_MIN_SECONDS",
    "CALIBRATED_MIN_HELD_OUT_SESSIONS",
    "CALIBRATED_MIN_HELD_OUT_CLIPS",
    "SPREAD_OVER_NOISE_FLOOR_WARNING",
    "AnomalySchemaError",
    "BindingMismatchError",
    "ModelSpec",
    "ShapeSpec",
    "LevelSpec",
    "CompositeSpec",
    "EventSmoothingSpec",
    "SensitivitySpec",
    "CalibrationSpec",
    "RuntimeThreshold",
    "AnomalyConfig",
    "ReadinessReport",
    "build_config",
    "build_runtime_threshold",
    "to_dict",
    "from_dict",
    "dumps",
    "loads",
    "validate",
    "fpr_upper_bound_95",
    "binomial_upper_bound_95",
    "assess_readiness",
    "calibration_confidence",
    "binding_tuple",
    "binding_fingerprint",
    "verify_binding",
    "dataset_fingerprint",
    "split_fingerprint",
    "held_out_session_count",
    "select_held_out_sessions",
    "assert_frontend_matches",
    "frontend_contract_digest",
    "preset_for",
    "top_k_mean",
    "rms_dbfs",
    "shape_ratio",
    "level_ratio",
    "composite_ratio",
    "exceeds",
    "smooth_state",
    "mac_per_context",
    "parameter_count",
    "validate_roles",
]


# --------------------------------------------------------------------------------------
# Schema versions
# --------------------------------------------------------------------------------------

#: Version of the SCORER contract: model topology, context stacking, top-k aggregation,
#: level statistic, composite rule, temporal voting and the binding tuple.  Bump this
#: whenever a change would make an old ``T_shape`` mean something different.
SCORER_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Version of the FEATURE FRONTEND contract implemented by
#: ``tm_local/audio_frontend.py:log_mel_spectrogram()``.  That code is frozen and must stay
#: bit-identical across the server, the exported reference implementation and the MCU port.
#: Bump this only together with a synchronised change in all three places.
FRONTEND_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The exact frontend parameters the scorer was defined against.  Mirrored here (rather
#: than imported) so this module stays numpy-free; :func:`assert_frontend_matches` checks a
#: live ``AudioFrontendConfig.to_dict()`` against it.
FRONTEND_CONTRACT: Final[dict[str, Any]] = {
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "window_ms": 25.0,
    "hop_ms": 10.0,
    "fft_size": 512,
    "mel_bins": 40,
    "fmin": 20.0,
    "fmax": 8000.0,
    "db_floor": -80.0,
    "frame_count": 98,
    "feature_shape": [98, 40, 1],
    "normalization": "10*log10(mel_power), clipped to [db_floor, 0], mapped to [0, 1]",
}


# --------------------------------------------------------------------------------------
# Frozen scorer constants
# --------------------------------------------------------------------------------------

FEATURE_RELATIVE_LOG_MEL: Final[str] = "relative_log_mel"
CONTEXT_FRAMES: Final[int] = 5
N_CONTEXTS: Final[int] = 94  # 98 frames - 5 + 1
VECTOR_DIM: Final[int] = 200  # 5 frames x 40 mel bins
#: ABSOLUTE INTEGER.  Never store a fraction: round(0.10 * 94) == 9 != 10, and letting each
#: language recompute the rounding guarantees PC/MCU divergence.
TOP_K: Final[int] = 10
AGGREGATION_MEAN_OF_LARGEST_K: Final[str] = "mean_of_largest_k"
INTEGER_MSE_DEFINITION: Final[str] = "dequantized_quantized_input_vs_dequantized_output"

LEVEL_STATISTIC: Final[str] = "rms_dbfs"
LEVEL_COMPUTED_AT: Final[str] = "after_fix_length_before_stft"
LEVEL_FLOOR_AMPLITUDE: Final[float] = 1e-5
LEVEL_CLIP_DB: Final[tuple[float, float]] = (-100.0, 0.0)
#: ``T_level`` is floored so a pathologically quiet, pathologically uniform baseline cannot
#: produce a hair-trigger level branch.
MIN_T_LEVEL_DB: Final[float] = 3.0

COMPOSITE_RULE: Final[str] = "max(shape_ratio, level_ratio)"
COMPOSITE_EXCEED_WHEN: Final[str] = "R > 1"

HOP_SECONDS: Final[float] = 0.5
WINDOW_COUNT: Final[int] = 5
WARMUP_STATE: Final[str] = "UNCERTAIN"

STATE_NORMAL: Final[str] = "NORMAL"
STATE_ABNORMAL: Final[str] = "ABNORMAL"
STATE_UNCERTAIN: Final[str] = "UNCERTAIN"

#: preset name -> (alpha, votes_required out of WINDOW_COUNT)
SENSITIVITY_PRESETS: Final[dict[str, tuple[float, int]]] = {
    "sensitive": (0.10, 2),
    "balanced": (0.05, 3),
    "low_false_alarm": (0.02, 3),
}

#: ``T_shape`` is calibrated PER RUNTIME.  Measured threshold drift keras -> int8 was +70%
#: on the rejected BatchNorm/linear architecture; the sigmoid architecture barely drifts,
#: but the contract stays per-runtime because "this architecture happens to be stable" is
#: not a guarantee.
RUNTIME_NAMES: Final[tuple[str, ...]] = ("keras", "float32", "dynamic", "int8", "uint8")
#: Only these runtimes have integer I/O and therefore real quantization parameters.
QUANTIZED_RUNTIMES: Final[frozenset[str]] = frozenset({"int8", "uint8"})

CONFIDENCE_CALIBRATED: Final[str] = "calibrated"
CONFIDENCE_LOW: Final[str] = "low"

ROLE_NORMAL_TRAIN: Final[str] = "normal_train"
ROLE_ANOMALY_EVAL: Final[str] = "anomaly_eval"
ROLE_NAMES: Final[tuple[str, str]] = (ROLE_NORMAL_TRAIN, ROLE_ANOMALY_EVAL)
#: MAX_CLASSES is 20: exactly one locked normal_train container plus 0..19 named
#: anomaly_eval groups.  anomaly_eval data is used ONLY for per-group evaluation reporting.
MAX_ANOMALY_EVAL_GROUPS: Final[int] = 19

QC_OK: Final[str] = "ok"
QC_WARNING: Final[str] = "warning"
QC_HIGH_REVIEW: Final[str] = "high_review"
QC_STATUSES: Final[tuple[str, ...]] = (QC_OK, QC_WARNING, QC_HIGH_REVIEW)

# Data readiness minimums (measured: a contaminated baseline inflated the level IQR 15.9x
# and dropped detection from 92.3% to 7.7%, so "enough independent sessions" is a safety
# requirement, not a nicety).
TRAIN_MIN_SESSIONS: Final[int] = 3
TRAIN_MIN_SECONDS: Final[float] = 60.0
CALIBRATED_MIN_SESSIONS: Final[int] = 6
CALIBRATED_MIN_SECONDS: Final[float] = 120.0
CALIBRATED_MIN_HELD_OUT_SESSIONS: Final[int] = 2
CALIBRATED_MIN_HELD_OUT_CLIPS: Final[int] = 40

#: Diagnostic only.  ``(P95 - P50) / measured_noise_floor`` below this earns a strong
#: warning in the conversion report; it never on its own decides pass/fail.  Measured 81.9x
#: on the selected sigmoid architecture.
SPREAD_OVER_NOISE_FLOOR_WARNING: Final[float] = 20.0

_MODEL_LAYER_UNITS: Final[tuple[int, ...]] = (200, 128, 64, 8, 64, 128, 200)
_SHA256_LENGTH: Final[int] = 64
_FLOAT_TOLERANCE: Final[float] = 1e-12


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class AnomalySchemaError(ValueError):
    """Raised when an ``anomaly_config.json`` payload violates the frozen contract."""


class BindingMismatchError(AnomalySchemaError):
    """Raised when a stored threshold is not bound to the artifacts in front of us.

    The caller MUST re-verify (recalibrate) rather than fall back to another runtime's
    threshold.
    """


# --------------------------------------------------------------------------------------
# Small pure helpers used by the validators
# --------------------------------------------------------------------------------------


def _clip(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _fail(message: str) -> None:
    raise AnomalySchemaError(message)


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{where}: expected a JSON object, got {type(value).__name__}.")
    return value


def _require_keys(payload: Mapping[str, Any], expected: Sequence[str], where: str) -> None:
    present = set(payload)
    missing = [name for name in expected if name not in present]
    if missing:
        _fail(f"{where}: missing required field(s) {sorted(missing)}.")
    unknown = sorted(present - set(expected))
    if unknown:
        _fail(f"{where}: unknown field(s) {unknown}; allowed fields are {list(expected)}.")


def _require_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float):
            _fail(
                f"{where}: must be an absolute integer, got the float {value!r}. Never store a "
                "fraction or a float-typed count: round(0.10 * 94) = 9 != 10, and letting each "
                "language recompute the rounding guarantees PC/MCU divergence."
            )
        _fail(f"{where}: must be an integer, got {type(value).__name__} {value!r}.")
    if minimum is not None and value < minimum:
        _fail(f"{where}: must be >= {minimum}, got {value}.")
    return int(value)


def _require_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where}: must be a number, got {type(value).__name__} {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{where}: must be finite, got {number!r}.")
    return number


def _require_text(value: Any, where: str, *, maximum: int = 200) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{where}: must be a non-empty string, got {value!r}.")
    if len(value) > maximum:
        _fail(f"{where}: must be at most {maximum} characters, got {len(value)}.")
    return value


def _require_exact(value: Any, expected: Any, where: str, *, why: str = "") -> Any:
    if value != expected:
        suffix = f" {why}" if why else ""
        _fail(f"{where}: must be {expected!r}, got {value!r}.{suffix}")
    return value


def _require_sha256(value: Any, where: str) -> str:
    text = _require_text(value, where, maximum=_SHA256_LENGTH)
    if len(text) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in text):
        _fail(f"{where}: must be a 64-character lowercase sha256 hex digest, got {value!r}.")
    return text


def _require_fingerprint(value: Any, where: str) -> str:
    text = _require_text(value, where, maximum=200)
    if any(char.isspace() for char in text):
        _fail(f"{where}: must not contain whitespace, got {value!r}.")
    return text


def _require_version(value: Any, where: str) -> str:
    text = _require_text(value, where, maximum=32)
    parts = text.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        _fail(f"{where}: must be a MAJOR.MINOR.PATCH version string, got {value!r}.")
    return text


# --------------------------------------------------------------------------------------
# The dataclass tree
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Autoencoder topology.  ``bottleneck_dim`` is config + binding hash, never UI."""

    architecture: str = "dense_denoising_autoencoder"
    layer_units: tuple[int, ...] = _MODEL_LAYER_UNITS
    bottleneck_dim: int = 8
    hidden_activation: str = "relu"
    output_activation: str = "sigmoid"
    batch_normalization: bool = False
    denoising_sigma: float = 0.01
    parameter_count: int = 69200


@dataclass(frozen=True)
class ShapeSpec:
    """The reconstruction branch.  ``top_k`` and ``n_contexts`` are absolute integers."""

    feature: str = FEATURE_RELATIVE_LOG_MEL
    context_frames: int = CONTEXT_FRAMES
    n_contexts: int = N_CONTEXTS
    vector_dim: int = VECTOR_DIM
    top_k: int = TOP_K
    aggregation: str = AGGREGATION_MEAN_OF_LARGEST_K
    integer_mse_definition: str = INTEGER_MSE_DEFINITION


@dataclass(frozen=True)
class LevelSpec:
    """The loudness branch the shape branch structurally cannot see."""

    c_level_db: float
    t_level_db: float
    statistic: str = LEVEL_STATISTIC
    computed_at: str = LEVEL_COMPUTED_AT
    floor_amplitude: float = LEVEL_FLOOR_AMPLITUDE
    clip_db: tuple[float, float] = LEVEL_CLIP_DB


@dataclass(frozen=True)
class CompositeSpec:
    rule: str = COMPOSITE_RULE
    exceed_when: str = COMPOSITE_EXCEED_WHEN


@dataclass(frozen=True)
class EventSmoothingSpec:
    votes_required: int
    hop_seconds: float = HOP_SECONDS
    window_count: int = WINDOW_COUNT
    warmup_state: str = WARMUP_STATE


@dataclass(frozen=True)
class SensitivitySpec:
    preset: str
    alpha: float

    @classmethod
    def from_preset(cls, preset: str) -> "SensitivitySpec":
        alpha, _votes = preset_for(preset)
        return cls(preset=preset, alpha=alpha)


@dataclass(frozen=True)
class CalibrationSpec:
    """Honest calibration bookkeeping.  A quantile is never an accuracy guarantee."""

    confidence: str
    held_out_clips: int
    held_out_sessions: int
    observed_exceedances: int
    fpr_upper_bound_95: float


@dataclass(frozen=True)
class RuntimeThreshold:
    """Per-runtime shape threshold plus everything it is bound to."""

    t_shape: float
    artifact_sha256: str
    keras_sha256: str
    representative_sha256: str
    dataset_fingerprint: str
    split_fingerprint: str
    spread_over_noise_floor: float
    normal_p50: float
    normal_p95: float
    input_scale: float | None = None
    input_zero_point: int | None = None
    output_scale: float | None = None
    output_zero_point: int | None = None


@dataclass(frozen=True)
class AnomalyConfig:
    """The whole of ``anomaly_config.json``."""

    level: LevelSpec
    event_smoothing: EventSmoothingSpec
    sensitivity: SensitivitySpec
    calibration: CalibrationSpec
    runtimes: dict[str, RuntimeThreshold]
    model: ModelSpec = ModelSpec()
    shape: ShapeSpec = ShapeSpec()
    composite: CompositeSpec = CompositeSpec()
    scorer_schema_version: str = SCORER_SCHEMA_VERSION
    frontend_schema_version: str = FRONTEND_SCHEMA_VERSION


@dataclass(frozen=True)
class ReadinessReport:
    """Outcome of the data-readiness / calibration-confidence decision."""

    can_train: bool
    confidence: str
    fpr_upper_bound_95: float
    reasons: tuple[str, ...]

    @property
    def is_calibrated(self) -> bool:
        return self.confidence == CONFIDENCE_CALIBRATED


# --------------------------------------------------------------------------------------
# Presets, statistics and honest bounds
# --------------------------------------------------------------------------------------


def preset_for(preset: str) -> tuple[float, int]:
    """Return ``(alpha, votes_required)`` for a sensitivity preset name."""

    if not isinstance(preset, str) or preset not in SENSITIVITY_PRESETS:
        _fail(
            f"sensitivity.preset: unknown preset {preset!r}; "
            f"allowed presets are {sorted(SENSITIVITY_PRESETS)}."
        )
    return SENSITIVITY_PRESETS[preset]


def fpr_upper_bound_95(v_clips: int) -> float:
    """95% one-sided upper bound on the false-positive rate after ``V`` clean clips.

    With ``V`` independent held-out normal clips and ZERO observed exceedances, the largest
    true rate ``p`` still consistent with that observation at 95% confidence solves
    ``(1 - p) ** V = 0.05``, i.e. ``p = 1 - 0.05 ** (1 / V)``.

    ``fpr_upper_bound_95(40) == 0.0721...`` -- so "0 out of 40" is honestly "at most ~7.2%",
    never "0%".  The UI must show the observed exceedance count next to this number and must
    never present it, or the calibration quantile, as an accuracy guarantee.

    When exceedances were actually observed this closed form is optimistic; use
    :func:`binomial_upper_bound_95` for the exact Clopper-Pearson upper limit.
    """

    clips = _require_int(v_clips, "held_out_clips", minimum=1)
    return 1.0 - 0.05 ** (1.0 / clips)


def binomial_upper_bound_95(v_clips: int, exceedances: int = 0) -> float:
    """Exact 95% one-sided Clopper-Pearson upper limit for ``n`` of ``V`` exceedances.

    Equals :func:`fpr_upper_bound_95` when ``exceedances == 0``.  Pure standard library:
    bisection on the binomial CDF, no scipy.
    """

    clips = _require_int(v_clips, "held_out_clips", minimum=1)
    observed = _require_int(exceedances, "observed_exceedances", minimum=0)
    if observed > clips:
        _fail(f"observed_exceedances ({observed}) cannot exceed held_out_clips ({clips}).")
    if observed == clips:
        return 1.0

    def cdf(probability: float) -> float:
        total = 0.0
        for successes in range(observed + 1):
            total += (
                math.comb(clips, successes)
                * probability**successes
                * (1.0 - probability) ** (clips - successes)
            )
        return total

    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if cdf(middle) > 0.05:
            low = middle
        else:
            high = middle
    return high


def assess_readiness(
    *,
    train_sessions: int,
    train_seconds: float,
    held_out_sessions: int,
    held_out_clips: int,
    alpha: float,
    unresolved_high_review: int = 0,
    train_clips: int | None = None,
) -> ReadinessReport:
    """Decide whether we may train at all, and whether the result may be called calibrated.

    ``can_train`` needs >= 3 independent normal sessions AND >= 60 s of normal audio, and
    both sides of the session split must hold >= 1 session AND >= 1 clip.

    ``confidence == "calibrated"`` additionally needs >= 6 sessions, >= 120 s, >= 2 held-out
    sessions, >= 40 held-out clips, ZERO unresolved high-review contamination clips, and --
    the honesty gate -- ``fpr_upper_bound_95(held_out_clips) <= 2 * alpha``.  Anything less
    is ``"low"``: the project is score-only and the UI must say so.

    ``train_sessions`` counts the sessions used for training; ``held_out_sessions`` /
    ``held_out_clips`` count the independent calibration side.  Clips from one recording
    session NEVER cross that boundary.
    """

    reasons: list[str] = []
    sessions = _require_int(train_sessions, "train_sessions", minimum=0)
    seconds = _require_float(train_seconds, "train_seconds")
    out_sessions = _require_int(held_out_sessions, "held_out_sessions", minimum=0)
    out_clips = _require_int(held_out_clips, "held_out_clips", minimum=0)
    alpha_value = _require_float(alpha, "sensitivity.alpha")
    unresolved = _require_int(unresolved_high_review, "unresolved_high_review", minimum=0)
    if seconds < 0.0:
        _fail(f"train_seconds: must be >= 0, got {seconds}.")
    if not 0.0 < alpha_value < 1.0:
        _fail(f"sensitivity.alpha: must be strictly between 0 and 1, got {alpha_value}.")

    total_sessions = sessions + out_sessions
    can_train = True
    if total_sessions < TRAIN_MIN_SESSIONS:
        can_train = False
        reasons.append(
            f"needs >= {TRAIN_MIN_SESSIONS} independent normal recording sessions, "
            f"has {total_sessions}"
        )
    if seconds < TRAIN_MIN_SECONDS:
        can_train = False
        reasons.append(
            f"needs >= {TRAIN_MIN_SECONDS:.0f} s of normal audio, has {seconds:.1f} s"
        )
    if sessions < 1 or out_sessions < 1:
        can_train = False
        reasons.append(
            "the train/calibration split must leave >= 1 session on each side "
            f"(train={sessions}, held-out={out_sessions})"
        )
    if out_clips < 1:
        can_train = False
        reasons.append("the calibration side must hold >= 1 clip, has 0")
    if train_clips is not None and _require_int(train_clips, "train_clips", minimum=0) < 1:
        can_train = False
        reasons.append("the training side must hold >= 1 clip, has 0")

    bound = fpr_upper_bound_95(out_clips) if out_clips >= 1 else 1.0

    calibrated = can_train
    if total_sessions < CALIBRATED_MIN_SESSIONS:
        calibrated = False
        reasons.append(
            f"calibration needs >= {CALIBRATED_MIN_SESSIONS} sessions, has {total_sessions}"
        )
    if seconds < CALIBRATED_MIN_SECONDS:
        calibrated = False
        reasons.append(
            f"calibration needs >= {CALIBRATED_MIN_SECONDS:.0f} s of normal audio, "
            f"has {seconds:.1f} s"
        )
    if out_sessions < CALIBRATED_MIN_HELD_OUT_SESSIONS:
        calibrated = False
        reasons.append(
            f"calibration needs >= {CALIBRATED_MIN_HELD_OUT_SESSIONS} held-out sessions, "
            f"has {out_sessions}"
        )
    if out_clips < CALIBRATED_MIN_HELD_OUT_CLIPS:
        calibrated = False
        reasons.append(
            f"calibration needs >= {CALIBRATED_MIN_HELD_OUT_CLIPS} held-out clips, "
            f"has {out_clips}"
        )
    if unresolved > 0:
        calibrated = False
        reasons.append(
            f"{unresolved} contamination high-review clip(s) are still unresolved; the user "
            "must decide on each one before the detector may be called calibrated"
        )
    if bound > 2.0 * alpha_value + _FLOAT_TOLERANCE:
        calibrated = False
        reasons.append(
            f"the 95% one-sided false-positive upper bound is {bound:.4f} with "
            f"{out_clips} held-out clip(s), which exceeds 2 x alpha = "
            f"{2.0 * alpha_value:.4f}; collect more held-out normal clips"
        )

    return ReadinessReport(
        can_train=can_train,
        confidence=CONFIDENCE_CALIBRATED if calibrated else CONFIDENCE_LOW,
        fpr_upper_bound_95=bound,
        reasons=tuple(reasons),
    )


def calibration_confidence(
    *,
    train_sessions: int,
    train_seconds: float,
    held_out_sessions: int,
    held_out_clips: int,
    alpha: float,
    unresolved_high_review: int = 0,
    train_clips: int | None = None,
) -> str:
    """``"calibrated"`` or ``"low"`` -- see :func:`assess_readiness` for the criteria."""

    return assess_readiness(
        train_sessions=train_sessions,
        train_seconds=train_seconds,
        held_out_sessions=held_out_sessions,
        held_out_clips=held_out_clips,
        alpha=alpha,
        unresolved_high_review=unresolved_high_review,
        train_clips=train_clips,
    ).confidence


# --------------------------------------------------------------------------------------
# Session split and fingerprints
# --------------------------------------------------------------------------------------


def held_out_session_count(n_sessions: int) -> int:
    """``ceil(0.25 * n_sessions)`` -- the number of calibration (held-out) sessions."""

    count = _require_int(n_sessions, "n_sessions", minimum=0)
    return math.ceil(0.25 * count)


def select_held_out_sessions(session_ids: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split recording sessions deterministically into ``(train, held_out)``.

    Ordering is by ``sha256(session_id)`` so every implementation and every re-run picks the
    same sessions.  Clips from one session NEVER cross the boundary; both sides must end up
    with at least one session or we refuse and say why.
    """

    unique = sorted({_require_text(value, "session_id", maximum=200) for value in session_ids})
    if not unique:
        _fail("session split: no recording sessions were provided.")
    ordered = sorted(
        unique, key=lambda item: (hashlib.sha256(item.encode("utf-8")).hexdigest(), item)
    )
    count = held_out_session_count(len(ordered))
    held_out = tuple(ordered[:count])
    train = tuple(ordered[count:])
    if not held_out or not train:
        _fail(
            f"session split: {len(ordered)} recording session(s) cannot be split into "
            "training and calibration sides that both hold >= 1 session; record normal "
            f"audio in at least {TRAIN_MIN_SESSIONS} separate sessions."
        )
    return train, held_out


def dataset_fingerprint(entries: Iterable[tuple[str, str]]) -> str:
    """Fingerprint the INCLUDED ``normal_train`` clips as ``(clip_id, content_sha256)``.

    anomaly_eval clips never enter this fingerprint: they never enter training, threshold
    calibration, contamination QC statistics or the quantization representative set, so they
    must not be able to invalidate a threshold either.
    """

    digest = hashlib.sha256()
    rows = sorted(
        (
            _require_text(clip_id, "dataset_fingerprint.clip_id", maximum=200),
            _require_sha256(clip_sha, "dataset_fingerprint.clip_sha256"),
        )
        for clip_id, clip_sha in entries
    )
    if not rows:
        _fail("dataset_fingerprint: at least one included normal_train clip is required.")
    digest.update(f"anomaly-dataset/v1/{len(rows)}".encode("utf-8"))
    for clip_id, clip_sha in rows:
        digest.update(b"\x1f")
        digest.update(clip_id.encode("utf-8"))
        digest.update(b"\x1e")
        digest.update(clip_sha.encode("utf-8"))
    return digest.hexdigest()


def split_fingerprint(
    train_sessions: Iterable[str], held_out_sessions: Iterable[str]
) -> str:
    """Fingerprint the train/calibration session split itself.

    The same clips split a different way produce a different threshold, so the split is part
    of the threshold's identity.
    """

    train = sorted(
        {
            _require_text(item, "split_fingerprint.train_session", maximum=200)
            for item in train_sessions
        }
    )
    held = sorted(
        {
            _require_text(item, "split_fingerprint.held_out_session", maximum=200)
            for item in held_out_sessions
        }
    )
    if not train or not held:
        _fail("split_fingerprint: both the train and held-out session lists must be non-empty.")
    overlap = sorted(set(train) & set(held))
    if overlap:
        _fail(
            "split_fingerprint: recording session(s) "
            f"{overlap} appear on both sides of the split; clips from one session must "
            "never cross the train/calibration boundary."
        )
    digest = hashlib.sha256()
    digest.update(b"anomaly-split/v1")
    for label, sessions in (("train", train), ("held_out", held)):
        digest.update(b"\x1d")
        digest.update(label.encode("utf-8"))
        for session in sessions:
            digest.update(b"\x1f")
            digest.update(session.encode("utf-8"))
    return digest.hexdigest()


def frontend_contract_digest() -> str:
    """sha256 of the canonical :data:`FRONTEND_CONTRACT`, for embedding in reports."""

    canonical = json.dumps(FRONTEND_CONTRACT, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_frontend_matches(frontend: Mapping[str, Any]) -> None:
    """Check a live ``AudioFrontendConfig.to_dict()`` against :data:`FRONTEND_CONTRACT`.

    Call this before training or exporting.  The frontend is a three-way contract (server,
    exported reference implementation, MCU doc); if it drifts, every stored ``T_shape``
    silently means something else.
    """

    mapping = _require_mapping(frontend, "audio_frontend")
    problems = []
    for key, expected in FRONTEND_CONTRACT.items():
        if key not in mapping:
            problems.append(f"{key}: missing")
            continue
        actual = mapping[key]
        if isinstance(expected, list):
            actual = list(actual) if isinstance(actual, (list, tuple)) else actual
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            if abs(float(actual) - expected) > _FLOAT_TOLERANCE:
                problems.append(f"{key}: expected {expected!r}, got {actual!r}")
            continue
        if actual != expected:
            problems.append(f"{key}: expected {expected!r}, got {actual!r}")
    if problems:
        _fail(
            "audio_frontend does not match FRONTEND_SCHEMA_VERSION "
            f"{FRONTEND_SCHEMA_VERSION}: " + "; ".join(problems)
        )


# --------------------------------------------------------------------------------------
# Scoring primitives (pure Python, shared with the exported runner and the MCU doc)
# --------------------------------------------------------------------------------------


def top_k_mean(errors: Sequence[float], top_k: int = TOP_K) -> float:
    """``S_shape``: arithmetic mean of the ``top_k`` LARGEST per-context MSEs."""

    values = [_require_float(item, "context_mse") for item in errors]
    count = _require_int(top_k, "shape.top_k", minimum=1)
    if not values:
        _fail("top_k_mean: no per-context reconstruction errors were provided.")
    if count > len(values):
        _fail(f"top_k_mean: top_k={count} exceeds the {len(values)} available contexts.")
    values.sort(reverse=True)
    return math.fsum(values[:count]) / float(count)


def rms_dbfs(mean_square: float) -> float:
    """``L``: RMS dBFS from the mean of ``pcm ** 2``.

    Computed on the PCM AFTER resample and ``fix_length(16000)`` and BEFORE the STFT::

        rms = sqrt(mean(pcm ** 2))
        L   = clip(20 * log10(max(rms, 1e-5)), -100.0, 0.0)

    The MCU equivalent is ``uint64 sum(int16 ** 2) / N`` scaled to full scale, which is why
    RMS was chosen over a mel-domain peak: it has a clean integer definition and an explicit
    dBFS reference point.
    """

    value = _require_float(mean_square, "level.mean_square")
    if value < 0.0:
        _fail(f"level.mean_square: must be >= 0, got {value}.")
    amplitude = max(math.sqrt(value), LEVEL_FLOOR_AMPLITUDE)
    return _clip(20.0 * math.log10(amplitude), LEVEL_CLIP_DB[0], LEVEL_CLIP_DB[1])


def shape_ratio(s_shape: float, t_shape: float) -> float:
    """``S_shape / T_shape[runtime]``."""

    score = _require_float(s_shape, "s_shape")
    threshold = _require_float(t_shape, "t_shape")
    if threshold <= 0.0:
        _fail(f"t_shape: must be > 0, got {threshold}.")
    return score / threshold


def level_ratio(level_db: float, c_level_db: float, t_level_db: float) -> float:
    """``abs(L - C_level) / T_level`` -- deviation in EITHER direction is a deviation."""

    level = _require_float(level_db, "level_db")
    center = _require_float(c_level_db, "level.c_level_db")
    tolerance = _require_float(t_level_db, "level.t_level_db")
    if tolerance < MIN_T_LEVEL_DB:
        _fail(f"level.t_level_db: must be >= {MIN_T_LEVEL_DB}, got {tolerance}.")
    return abs(level - center) / tolerance


def composite_ratio(
    *, s_shape: float, t_shape: float, level_db: float, c_level_db: float, t_level_db: float
) -> float:
    """``R = max(shape_ratio, level_ratio)``."""

    return max(
        shape_ratio(s_shape, t_shape),
        level_ratio(level_db, c_level_db, t_level_db),
    )


def exceeds(ratio: float) -> bool:
    """``R > 1`` means this window exceeds the collected normal baseline."""

    return _require_float(ratio, "composite_ratio") > 1.0


def smooth_state(
    flags: Sequence[bool], votes_required: int, window_count: int = WINDOW_COUNT
) -> str:
    """Temporal vote over the most recent windows (0.5 s hop).

    Fewer than ``window_count`` windows since start is ``UNCERTAIN`` (warming up) -- never
    alarm early.  The UI must always show the RAW per-window exceedance timeline as well,
    not only the voted alarm.
    """

    count = _require_int(window_count, "event_smoothing.window_count", minimum=1)
    votes = _require_int(votes_required, "event_smoothing.votes_required", minimum=1)
    if votes > count:
        _fail(f"event_smoothing.votes_required ({votes}) cannot exceed window_count ({count}).")
    history = list(flags)
    if len(history) < count:
        return STATE_UNCERTAIN
    recent = history[-count:]
    hits = sum(1 for item in recent if bool(item))
    return STATE_ABNORMAL if hits >= votes else STATE_NORMAL


def mac_per_context(layer_units: Sequence[int] = _MODEL_LAYER_UNITS) -> int:
    """Multiply-accumulates for one 200-dim context (68,608 for the locked topology)."""

    units = [_require_int(item, "model.layer_units", minimum=1) for item in layer_units]
    return sum(units[index] * units[index + 1] for index in range(len(units) - 1))


def parameter_count(layer_units: Sequence[int] = _MODEL_LAYER_UNITS) -> int:
    """Trainable parameters including biases (69,200 for the locked topology)."""

    units = [_require_int(item, "model.layer_units", minimum=1) for item in layer_units]
    return sum(
        units[index] * units[index + 1] + units[index + 1] for index in range(len(units) - 1)
    )


def validate_roles(roles: Sequence[str]) -> None:
    """Exactly one locked ``normal_train`` container plus 0..19 ``anomaly_eval`` groups."""

    names = [_require_text(item, "role", maximum=64) for item in roles]
    unknown = sorted({name for name in names if name not in ROLE_NAMES})
    if unknown:
        _fail(f"roles: unknown role(s) {unknown}; allowed roles are {list(ROLE_NAMES)}.")
    normals = names.count(ROLE_NORMAL_TRAIN)
    evals = names.count(ROLE_ANOMALY_EVAL)
    if normals != 1:
        _fail(f"roles: exactly one '{ROLE_NORMAL_TRAIN}' container is required, found {normals}.")
    if evals > MAX_ANOMALY_EVAL_GROUPS:
        _fail(
            f"roles: at most {MAX_ANOMALY_EVAL_GROUPS} '{ROLE_ANOMALY_EVAL}' groups are "
            f"allowed, found {evals}."
        )


# --------------------------------------------------------------------------------------
# Threshold binding
# --------------------------------------------------------------------------------------

#: Field order of the binding tuple.  Keep stable: :func:`binding_fingerprint` hashes it.
BINDING_FIELDS: Final[tuple[str, ...]] = (
    "scorer_schema_version",
    "frontend_schema_version",
    "runtime",
    "bottleneck_dim",
    "keras_sha256",
    "artifact_sha256",
    "representative_sha256",
    "dataset_fingerprint",
    "split_fingerprint",
    "input_scale",
    "input_zero_point",
    "output_scale",
    "output_zero_point",
)


def binding_tuple(config: "AnomalyConfig | Mapping[str, Any]", runtime: str) -> tuple[Any, ...]:
    """Everything a stored ``T_shape`` is bound to, in :data:`BINDING_FIELDS` order.

    A threshold is valid ONLY if every element matches the artifacts in front of us.  On any
    mismatch the caller must RE-VERIFY (recalibrate that runtime) -- never silently fall back
    to another runtime's threshold, and never reuse the keras threshold for int8.

    Why ``keras_sha256`` alone is not enough
    ----------------------------------------
    Converting the SAME ``.keras`` twice with different representative subsets yields
    different input/output scales and zero points, and the reconstruction MSE is measured in
    those units -- so the two runs produce differently-scaled scores while ``keras_sha256``
    stays byte-identical.  ``artifact_sha256`` (the converted file) and the four quantization
    parameters are therefore part of the identity, not decoration.

    Why file size is NOT a valid identity check
    -------------------------------------------
    Measured: two strict-INT8 conversions of the same architecture differed by 512 bytes
    (90,952 B vs 90,440 B) purely because of the representative set.  Conversion is not
    byte-reproducible, so any gate keyed on file size is wrong.  ``representative_sha256``
    pins the representative subset itself.

    ``bottleneck_dim`` is included because it changes what "reconstructed well" means, and
    the locked spec requires it in the binding hash even though it is never shown in the UI.
    """

    resolved = config if isinstance(config, AnomalyConfig) else from_dict(config)
    name = _require_text(runtime, "runtime", maximum=32)
    if name not in RUNTIME_NAMES:
        _fail(f"runtimes: unknown runtime name {name!r}; allowed names are {list(RUNTIME_NAMES)}.")
    entry = resolved.runtimes.get(name)
    if entry is None:
        _fail(
            f"runtimes[{name!r}]: no calibrated threshold is stored for this runtime; "
            "calibrate it instead of reusing another runtime's threshold."
        )
    assert entry is not None  # for type checkers; _fail already raised
    return (
        resolved.scorer_schema_version,
        resolved.frontend_schema_version,
        name,
        resolved.model.bottleneck_dim,
        entry.keras_sha256,
        entry.artifact_sha256,
        entry.representative_sha256,
        entry.dataset_fingerprint,
        entry.split_fingerprint,
        entry.input_scale,
        entry.input_zero_point,
        entry.output_scale,
        entry.output_zero_point,
    )


def _binding_token(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        # float.hex() is exact and platform-stable, unlike repr rounding rules.
        return f"f:{value.hex()}"
    return f"s:{value}"


def binding_fingerprint(config: "AnomalyConfig | Mapping[str, Any]", runtime: str) -> str:
    """sha256 over the binding tuple -- a single value to store and compare."""

    tokens = [_binding_token(item) for item in binding_tuple(config, runtime)]
    payload = "\x1f".join(tokens).encode("utf-8")
    return hashlib.sha256(b"anomaly-binding/v1\x1f" + payload).hexdigest()


def verify_binding(
    config: "AnomalyConfig | Mapping[str, Any]",
    runtime: str,
    observed: "Mapping[str, Any] | Sequence[Any]",
) -> None:
    """Raise :class:`BindingMismatchError` unless every binding element matches.

    ``observed`` is either a mapping keyed by :data:`BINDING_FIELDS` (extra keys ignored,
    missing keys reported) or a sequence in that exact order.
    """

    expected = binding_tuple(config, runtime)
    if isinstance(observed, Mapping):
        missing = [name for name in BINDING_FIELDS if name not in observed]
        if missing:
            raise BindingMismatchError(
                f"threshold binding for runtime {runtime!r} cannot be verified: the observed "
                f"artifacts are missing {missing}. Re-verify (recalibrate) this runtime; do "
                "not fall back to another runtime's threshold."
            )
        actual = tuple(observed[name] for name in BINDING_FIELDS)
    else:
        actual = tuple(observed)
        if len(actual) != len(BINDING_FIELDS):
            raise BindingMismatchError(
                f"threshold binding for runtime {runtime!r} cannot be verified: expected "
                f"{len(BINDING_FIELDS)} elements in the order {list(BINDING_FIELDS)}, got "
                f"{len(actual)}."
            )

    differences = []
    for name, want, got in zip(BINDING_FIELDS, expected, actual):
        if want is None or got is None:
            matches = want is None and got is None
        elif isinstance(want, (int, float)) and not isinstance(want, bool):
            matches = (
                isinstance(got, (int, float))
                and not isinstance(got, bool)
                and float(got) == float(want)
            )
        else:
            matches = want == got
        if not matches:
            differences.append(f"{name}: stored {want!r} != observed {got!r}")
    if differences:
        raise BindingMismatchError(
            f"threshold binding mismatch for runtime {runtime!r}: "
            + "; ".join(differences)
            + ". The stored T_shape does not describe these artifacts; re-verify "
            "(recalibrate) this runtime. Never fall back to another runtime's threshold: "
            "measured keras -> int8 threshold drift was +70% on a rejected architecture."
        )


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------


def build_runtime_threshold(
    *,
    t_shape: float,
    artifact_sha256: str,
    keras_sha256: str,
    representative_sha256: str,
    dataset_fingerprint: str,
    split_fingerprint: str,
    spread_over_noise_floor: float,
    normal_p50: float,
    normal_p95: float,
    input_scale: float | None = None,
    input_zero_point: int | None = None,
    output_scale: float | None = None,
    output_zero_point: int | None = None,
) -> RuntimeThreshold:
    """Assemble one runtime entry.  Quantization params are required for int8/uint8."""

    return RuntimeThreshold(
        t_shape=float(t_shape),
        artifact_sha256=artifact_sha256,
        keras_sha256=keras_sha256,
        representative_sha256=representative_sha256,
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_fingerprint,
        spread_over_noise_floor=float(spread_over_noise_floor),
        normal_p50=float(normal_p50),
        normal_p95=float(normal_p95),
        input_scale=None if input_scale is None else float(input_scale),
        input_zero_point=None if input_zero_point is None else int(input_zero_point),
        output_scale=None if output_scale is None else float(output_scale),
        output_zero_point=None if output_zero_point is None else int(output_zero_point),
    )


def build_config(
    *,
    preset: str,
    c_level_db: float,
    t_level_db: float,
    runtimes: Mapping[str, RuntimeThreshold],
    train_sessions: int,
    train_seconds: float,
    held_out_sessions: int,
    held_out_clips: int,
    observed_exceedances: int,
    unresolved_high_review: int = 0,
    train_clips: int | None = None,
    bottleneck_dim: int = 8,
    model: ModelSpec | None = None,
    shape: ShapeSpec | None = None,
) -> AnomalyConfig:
    """Build a fully validated :class:`AnomalyConfig`.

    ``t_level_db`` is floored at :data:`MIN_T_LEVEL_DB` here rather than rejected, because the
    floor is part of the level-scorer definition; every other insufficiency raises.  The
    calibration confidence and the honest 95% bound are computed, never passed in.
    """

    alpha, votes = preset_for(preset)
    report = assess_readiness(
        train_sessions=train_sessions,
        train_seconds=train_seconds,
        held_out_sessions=held_out_sessions,
        held_out_clips=held_out_clips,
        alpha=alpha,
        unresolved_high_review=unresolved_high_review,
        train_clips=train_clips,
    )
    if not report.can_train:
        _fail("cannot build an anomaly config: " + "; ".join(report.reasons) + ".")

    resolved_model = model if model is not None else ModelSpec(bottleneck_dim=bottleneck_dim)
    if model is None and bottleneck_dim != 8:
        units = list(_MODEL_LAYER_UNITS)
        units[3] = _require_int(bottleneck_dim, "model.bottleneck_dim", minimum=1)
        resolved_model = ModelSpec(
            layer_units=tuple(units),
            bottleneck_dim=units[3],
            parameter_count=parameter_count(units),
        )

    config = AnomalyConfig(
        model=resolved_model,
        shape=shape if shape is not None else ShapeSpec(),
        level=LevelSpec(
            c_level_db=float(c_level_db),
            t_level_db=max(float(t_level_db), MIN_T_LEVEL_DB),
        ),
        composite=CompositeSpec(),
        event_smoothing=EventSmoothingSpec(votes_required=votes),
        sensitivity=SensitivitySpec(preset=preset, alpha=alpha),
        calibration=CalibrationSpec(
            confidence=report.confidence,
            held_out_clips=int(held_out_clips),
            held_out_sessions=int(held_out_sessions),
            observed_exceedances=int(observed_exceedances),
            fpr_upper_bound_95=fpr_upper_bound_95(held_out_clips),
        ),
        runtimes=dict(runtimes),
    )
    return validate(config)


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------


def to_dict(config: AnomalyConfig) -> dict[str, Any]:
    """Serialise to the exact ``anomaly_config.json`` shape (stable key order)."""

    if not isinstance(config, AnomalyConfig):
        _fail(f"to_dict: expected an AnomalyConfig, got {type(config).__name__}.")
    return {
        "scorer_schema_version": config.scorer_schema_version,
        "frontend_schema_version": config.frontend_schema_version,
        "model": {
            "architecture": config.model.architecture,
            "layer_units": list(config.model.layer_units),
            "bottleneck_dim": config.model.bottleneck_dim,
            "hidden_activation": config.model.hidden_activation,
            "output_activation": config.model.output_activation,
            "batch_normalization": config.model.batch_normalization,
            "denoising_sigma": config.model.denoising_sigma,
            "parameter_count": config.model.parameter_count,
        },
        "shape": {
            "feature": config.shape.feature,
            "context_frames": config.shape.context_frames,
            "n_contexts": config.shape.n_contexts,
            "vector_dim": config.shape.vector_dim,
            "top_k": config.shape.top_k,
            "aggregation": config.shape.aggregation,
            "integer_mse_definition": config.shape.integer_mse_definition,
        },
        "level": {
            "statistic": config.level.statistic,
            "computed_at": config.level.computed_at,
            "floor_amplitude": config.level.floor_amplitude,
            "clip_db": list(config.level.clip_db),
            "c_level_db": config.level.c_level_db,
            "t_level_db": config.level.t_level_db,
        },
        "composite": {
            "rule": config.composite.rule,
            "exceed_when": config.composite.exceed_when,
        },
        "event_smoothing": {
            "hop_seconds": config.event_smoothing.hop_seconds,
            "window_count": config.event_smoothing.window_count,
            "votes_required": config.event_smoothing.votes_required,
            "warmup_state": config.event_smoothing.warmup_state,
        },
        "sensitivity": {
            "preset": config.sensitivity.preset,
            "alpha": config.sensitivity.alpha,
        },
        "calibration": {
            "confidence": config.calibration.confidence,
            "held_out_clips": config.calibration.held_out_clips,
            "held_out_sessions": config.calibration.held_out_sessions,
            "observed_exceedances": config.calibration.observed_exceedances,
            "fpr_upper_bound_95": config.calibration.fpr_upper_bound_95,
        },
        "runtimes": {
            name: {
                "t_shape": entry.t_shape,
                "artifact_sha256": entry.artifact_sha256,
                "keras_sha256": entry.keras_sha256,
                "representative_sha256": entry.representative_sha256,
                "dataset_fingerprint": entry.dataset_fingerprint,
                "split_fingerprint": entry.split_fingerprint,
                "input_scale": entry.input_scale,
                "input_zero_point": entry.input_zero_point,
                "output_scale": entry.output_scale,
                "output_zero_point": entry.output_zero_point,
                "spread_over_noise_floor": entry.spread_over_noise_floor,
                "normal_p50": entry.normal_p50,
                "normal_p95": entry.normal_p95,
            }
            for name, entry in config.runtimes.items()
        },
    }


_TOP_LEVEL_KEYS: Final[tuple[str, ...]] = (
    "scorer_schema_version",
    "frontend_schema_version",
    "model",
    "shape",
    "level",
    "composite",
    "event_smoothing",
    "sensitivity",
    "calibration",
    "runtimes",
)
_MODEL_KEYS: Final[tuple[str, ...]] = tuple(item.name for item in fields(ModelSpec))
_SHAPE_KEYS: Final[tuple[str, ...]] = tuple(item.name for item in fields(ShapeSpec))
_LEVEL_KEYS: Final[tuple[str, ...]] = (
    "statistic",
    "computed_at",
    "floor_amplitude",
    "clip_db",
    "c_level_db",
    "t_level_db",
)
_COMPOSITE_KEYS: Final[tuple[str, ...]] = tuple(item.name for item in fields(CompositeSpec))
_EVENT_KEYS: Final[tuple[str, ...]] = (
    "hop_seconds",
    "window_count",
    "votes_required",
    "warmup_state",
)
_SENSITIVITY_KEYS: Final[tuple[str, ...]] = tuple(item.name for item in fields(SensitivitySpec))
_CALIBRATION_KEYS: Final[tuple[str, ...]] = tuple(item.name for item in fields(CalibrationSpec))
_RUNTIME_KEYS: Final[tuple[str, ...]] = (
    "t_shape",
    "artifact_sha256",
    "keras_sha256",
    "representative_sha256",
    "dataset_fingerprint",
    "split_fingerprint",
    "input_scale",
    "input_zero_point",
    "output_scale",
    "output_zero_point",
    "spread_over_noise_floor",
    "normal_p50",
    "normal_p95",
)


def from_dict(payload: Mapping[str, Any]) -> AnomalyConfig:
    """Parse a JSON-shaped mapping into :class:`AnomalyConfig` (structure only).

    Types and required keys are checked here; semantic checks live in :func:`validate`,
    which this function does NOT call.  Use :func:`loads` or :func:`validate` when you want
    both.
    """

    root = _require_mapping(payload, "anomaly_config")
    _require_keys(root, _TOP_LEVEL_KEYS, "anomaly_config")

    model_raw = _require_mapping(root["model"], "model")
    _require_keys(model_raw, _MODEL_KEYS, "model")
    units_raw = model_raw["layer_units"]
    if not isinstance(units_raw, (list, tuple)) or not units_raw:
        _fail(f"model.layer_units: must be a non-empty list of integers, got {units_raw!r}.")
    if not isinstance(model_raw["batch_normalization"], bool):
        _fail(
            "model.batch_normalization: must be a boolean, got "
            f"{type(model_raw['batch_normalization']).__name__}."
        )
    model = ModelSpec(
        architecture=_require_text(model_raw["architecture"], "model.architecture", maximum=64),
        layer_units=tuple(
            _require_int(item, "model.layer_units[]", minimum=1) for item in units_raw
        ),
        bottleneck_dim=_require_int(model_raw["bottleneck_dim"], "model.bottleneck_dim", minimum=1),
        hidden_activation=_require_text(
            model_raw["hidden_activation"], "model.hidden_activation", maximum=32
        ),
        output_activation=_require_text(
            model_raw["output_activation"], "model.output_activation", maximum=32
        ),
        batch_normalization=bool(model_raw["batch_normalization"]),
        denoising_sigma=_require_float(model_raw["denoising_sigma"], "model.denoising_sigma"),
        parameter_count=_require_int(
            model_raw["parameter_count"], "model.parameter_count", minimum=1
        ),
    )

    shape_raw = _require_mapping(root["shape"], "shape")
    _require_keys(shape_raw, _SHAPE_KEYS, "shape")
    shape = ShapeSpec(
        feature=_require_text(shape_raw["feature"], "shape.feature", maximum=64),
        context_frames=_require_int(shape_raw["context_frames"], "shape.context_frames", minimum=1),
        n_contexts=_require_int(shape_raw["n_contexts"], "shape.n_contexts", minimum=1),
        vector_dim=_require_int(shape_raw["vector_dim"], "shape.vector_dim", minimum=1),
        top_k=_require_int(shape_raw["top_k"], "shape.top_k", minimum=1),
        aggregation=_require_text(shape_raw["aggregation"], "shape.aggregation", maximum=64),
        integer_mse_definition=_require_text(
            shape_raw["integer_mse_definition"], "shape.integer_mse_definition", maximum=128
        ),
    )

    level_raw = _require_mapping(root["level"], "level")
    _require_keys(level_raw, _LEVEL_KEYS, "level")
    clip_raw = level_raw["clip_db"]
    if not isinstance(clip_raw, (list, tuple)) or len(clip_raw) != 2:
        _fail(f"level.clip_db: must be a two-element [low, high] list, got {clip_raw!r}.")
    level = LevelSpec(
        c_level_db=_require_float(level_raw["c_level_db"], "level.c_level_db"),
        t_level_db=_require_float(level_raw["t_level_db"], "level.t_level_db"),
        statistic=_require_text(level_raw["statistic"], "level.statistic", maximum=64),
        computed_at=_require_text(level_raw["computed_at"], "level.computed_at", maximum=64),
        floor_amplitude=_require_float(level_raw["floor_amplitude"], "level.floor_amplitude"),
        clip_db=(
            _require_float(clip_raw[0], "level.clip_db[0]"),
            _require_float(clip_raw[1], "level.clip_db[1]"),
        ),
    )

    composite_raw = _require_mapping(root["composite"], "composite")
    _require_keys(composite_raw, _COMPOSITE_KEYS, "composite")
    composite = CompositeSpec(
        rule=_require_text(composite_raw["rule"], "composite.rule", maximum=64),
        exceed_when=_require_text(
            composite_raw["exceed_when"], "composite.exceed_when", maximum=32
        ),
    )

    event_raw = _require_mapping(root["event_smoothing"], "event_smoothing")
    _require_keys(event_raw, _EVENT_KEYS, "event_smoothing")
    event = EventSmoothingSpec(
        votes_required=_require_int(
            event_raw["votes_required"], "event_smoothing.votes_required", minimum=1
        ),
        hop_seconds=_require_float(event_raw["hop_seconds"], "event_smoothing.hop_seconds"),
        window_count=_require_int(
            event_raw["window_count"], "event_smoothing.window_count", minimum=1
        ),
        warmup_state=_require_text(
            event_raw["warmup_state"], "event_smoothing.warmup_state", maximum=32
        ),
    )

    sensitivity_raw = _require_mapping(root["sensitivity"], "sensitivity")
    _require_keys(sensitivity_raw, _SENSITIVITY_KEYS, "sensitivity")
    sensitivity = SensitivitySpec(
        preset=_require_text(sensitivity_raw["preset"], "sensitivity.preset", maximum=32),
        alpha=_require_float(sensitivity_raw["alpha"], "sensitivity.alpha"),
    )

    calibration_raw = _require_mapping(root["calibration"], "calibration")
    _require_keys(calibration_raw, _CALIBRATION_KEYS, "calibration")
    calibration = CalibrationSpec(
        confidence=_require_text(
            calibration_raw["confidence"], "calibration.confidence", maximum=32
        ),
        held_out_clips=_require_int(
            calibration_raw["held_out_clips"], "calibration.held_out_clips", minimum=0
        ),
        held_out_sessions=_require_int(
            calibration_raw["held_out_sessions"], "calibration.held_out_sessions", minimum=0
        ),
        observed_exceedances=_require_int(
            calibration_raw["observed_exceedances"], "calibration.observed_exceedances", minimum=0
        ),
        fpr_upper_bound_95=_require_float(
            calibration_raw["fpr_upper_bound_95"], "calibration.fpr_upper_bound_95"
        ),
    )

    runtimes_raw = _require_mapping(root["runtimes"], "runtimes")
    runtimes: dict[str, RuntimeThreshold] = {}
    for name, entry_raw in runtimes_raw.items():
        where = f"runtimes[{name!r}]"
        if name not in RUNTIME_NAMES:
            _fail(
                f"runtimes: unknown runtime name {name!r}; allowed names are "
                f"{list(RUNTIME_NAMES)}."
            )
        entry = _require_mapping(entry_raw, where)
        _require_keys(entry, _RUNTIME_KEYS, where)
        runtimes[name] = RuntimeThreshold(
            t_shape=_require_float(entry["t_shape"], f"{where}.t_shape"),
            artifact_sha256=_require_sha256(entry["artifact_sha256"], f"{where}.artifact_sha256"),
            keras_sha256=_require_sha256(entry["keras_sha256"], f"{where}.keras_sha256"),
            representative_sha256=_require_sha256(
                entry["representative_sha256"], f"{where}.representative_sha256"
            ),
            dataset_fingerprint=_require_fingerprint(
                entry["dataset_fingerprint"], f"{where}.dataset_fingerprint"
            ),
            split_fingerprint=_require_fingerprint(
                entry["split_fingerprint"], f"{where}.split_fingerprint"
            ),
            spread_over_noise_floor=_require_float(
                entry["spread_over_noise_floor"], f"{where}.spread_over_noise_floor"
            ),
            normal_p50=_require_float(entry["normal_p50"], f"{where}.normal_p50"),
            normal_p95=_require_float(entry["normal_p95"], f"{where}.normal_p95"),
            input_scale=(
                None
                if entry["input_scale"] is None
                else _require_float(entry["input_scale"], f"{where}.input_scale")
            ),
            input_zero_point=(
                None
                if entry["input_zero_point"] is None
                else _require_int(entry["input_zero_point"], f"{where}.input_zero_point")
            ),
            output_scale=(
                None
                if entry["output_scale"] is None
                else _require_float(entry["output_scale"], f"{where}.output_scale")
            ),
            output_zero_point=(
                None
                if entry["output_zero_point"] is None
                else _require_int(entry["output_zero_point"], f"{where}.output_zero_point")
            ),
        )

    return AnomalyConfig(
        model=model,
        shape=shape,
        level=level,
        composite=composite,
        event_smoothing=event,
        sensitivity=sensitivity,
        calibration=calibration,
        runtimes=runtimes,
        scorer_schema_version=_require_version(
            root["scorer_schema_version"], "scorer_schema_version"
        ),
        frontend_schema_version=_require_version(
            root["frontend_schema_version"], "frontend_schema_version"
        ),
    )


def dumps(config: AnomalyConfig, *, indent: int | None = 2) -> str:
    """Validate then serialise to JSON text."""

    return json.dumps(to_dict(validate(config)), ensure_ascii=False, indent=indent)


def loads(text: str) -> AnomalyConfig:
    """Parse JSON text and fully validate it."""

    if not isinstance(text, str):
        _fail(f"loads: expected JSON text, got {type(text).__name__}.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise AnomalySchemaError(f"anomaly_config.json is not valid JSON: {error}") from error
    return validate(payload)


# --------------------------------------------------------------------------------------
# Strict validation
# --------------------------------------------------------------------------------------


def _validate_model(model: ModelSpec, shape: ShapeSpec) -> None:
    units = list(model.layer_units)
    if len(units) < 3:
        _fail(f"model.layer_units: needs at least 3 layers, got {units}.")
    if units[0] != shape.vector_dim or units[-1] != shape.vector_dim:
        _fail(
            f"model.layer_units: the autoencoder must start and end at shape.vector_dim "
            f"({shape.vector_dim}), got {units[0]} -> ... -> {units[-1]}."
        )
    if model.bottleneck_dim != min(units):
        _fail(
            f"model.bottleneck_dim ({model.bottleneck_dim}) must be the smallest layer in "
            f"model.layer_units {units} (smallest is {min(units)})."
        )
    if model.batch_normalization:
        _fail(
            "model.batch_normalization: must be false. BatchNormalization is forbidden in "
            "this architecture -- the rejected BatchNorm/linear variant measured a "
            "quantization noise floor of 2.002e-04 versus 2.393e-06 here."
        )
    if model.output_activation != "sigmoid":
        _fail(
            f"model.output_activation: must be 'sigmoid', got {model.output_activation!r}. "
            "Sigmoid pins the TFLite int8 output scale to exactly 1/256 = 0.00390625; a "
            "linear output derives its scale from the representative data and is unbounded, "
            "which measured a 30.5x score margin instead of 81.9x."
        )
    if model.hidden_activation != "relu":
        _fail(f"model.hidden_activation: must be 'relu', got {model.hidden_activation!r}.")
    if not model.denoising_sigma > 0.0:
        _fail(
            f"model.denoising_sigma: must be > 0, got {model.denoising_sigma}. Training input "
            "is clip(x + N(0, sigma), 0, 1) with the clean x as target."
        )
    expected_params = parameter_count(units)
    if model.parameter_count != expected_params:
        _fail(
            f"model.parameter_count: {model.parameter_count} does not match the "
            f"{expected_params} parameters implied by layer_units {units}."
        )


def _validate_shape(shape: ShapeSpec) -> None:
    _require_exact(shape.feature, FEATURE_RELATIVE_LOG_MEL, "shape.feature")
    _require_exact(
        shape.aggregation,
        AGGREGATION_MEAN_OF_LARGEST_K,
        "shape.aggregation",
        why="S_shape is the arithmetic mean of the largest top_k per-context MSEs.",
    )
    _require_exact(
        shape.integer_mse_definition,
        INTEGER_MSE_DEFINITION,
        "shape.integer_mse_definition",
        why="Integer MSE compares the DEQUANTIZED quantized input against the DEQUANTIZED output.",
    )
    if shape.context_frames != CONTEXT_FRAMES:
        _fail(f"shape.context_frames: must be {CONTEXT_FRAMES}, got {shape.context_frames}.")
    if shape.vector_dim != shape.context_frames * FRONTEND_CONTRACT["mel_bins"]:
        _fail(
            f"shape.vector_dim: must be context_frames x mel_bins = "
            f"{shape.context_frames} x {FRONTEND_CONTRACT['mel_bins']} = "
            f"{shape.context_frames * FRONTEND_CONTRACT['mel_bins']}, got {shape.vector_dim}."
        )
    expected_contexts = FRONTEND_CONTRACT["frame_count"] - shape.context_frames + 1
    if shape.n_contexts != expected_contexts:
        _fail(
            f"shape.n_contexts: must be frame_count - context_frames + 1 = "
            f"{FRONTEND_CONTRACT['frame_count']} - {shape.context_frames} + 1 = "
            f"{expected_contexts}, got {shape.n_contexts}."
        )
    if shape.top_k != TOP_K:
        _fail(
            f"shape.top_k: must be the absolute integer {TOP_K}, got {shape.top_k}. Never "
            "store a fraction: round(0.10 * 94) = 9 != 10, and letting each language "
            "recompute the rounding guarantees PC/MCU divergence."
        )
    if shape.top_k > shape.n_contexts:
        _fail(
            f"shape.top_k ({shape.top_k}) cannot exceed shape.n_contexts ({shape.n_contexts})."
        )


def _validate_level(level: LevelSpec) -> None:
    _require_exact(level.statistic, LEVEL_STATISTIC, "level.statistic")
    _require_exact(
        level.computed_at,
        LEVEL_COMPUTED_AT,
        "level.computed_at",
        why="L is computed on the PCM after resample and fix_length(16000), before the STFT.",
    )
    if abs(level.floor_amplitude - LEVEL_FLOOR_AMPLITUDE) > _FLOAT_TOLERANCE:
        _fail(
            f"level.floor_amplitude: must be {LEVEL_FLOOR_AMPLITUDE}, got {level.floor_amplitude}."
        )
    if tuple(level.clip_db) != LEVEL_CLIP_DB:
        _fail(f"level.clip_db: must be {list(LEVEL_CLIP_DB)}, got {list(level.clip_db)}.")
    low, high = LEVEL_CLIP_DB
    if not low <= level.c_level_db <= high:
        _fail(
            f"level.c_level_db: must lie inside clip_db [{low}, {high}], got {level.c_level_db}."
        )
    if level.t_level_db < MIN_T_LEVEL_DB:
        _fail(
            f"level.t_level_db: must be >= {MIN_T_LEVEL_DB} dB, got {level.t_level_db}. The "
            "floor exists so a pathologically uniform baseline cannot produce a hair-trigger "
            "level branch."
        )
    if level.t_level_db > high - low:
        _fail(
            f"level.t_level_db: {level.t_level_db} dB is wider than the whole clip_db range "
            f"({high - low} dB), which can never be exceeded."
        )


def _validate_composite(composite: CompositeSpec) -> None:
    _require_exact(composite.rule, COMPOSITE_RULE, "composite.rule")
    _require_exact(composite.exceed_when, COMPOSITE_EXCEED_WHEN, "composite.exceed_when")


def _validate_event(event: EventSmoothingSpec, preset: str) -> None:
    if abs(event.hop_seconds - HOP_SECONDS) > _FLOAT_TOLERANCE:
        _fail(f"event_smoothing.hop_seconds: must be {HOP_SECONDS}, got {event.hop_seconds}.")
    if event.window_count != WINDOW_COUNT:
        _fail(f"event_smoothing.window_count: must be {WINDOW_COUNT}, got {event.window_count}.")
    _require_exact(
        event.warmup_state,
        WARMUP_STATE,
        "event_smoothing.warmup_state",
        why="Fewer than 5 windows since start must never alarm.",
    )
    if event.votes_required > event.window_count:
        _fail(
            f"event_smoothing.votes_required ({event.votes_required}) cannot exceed "
            f"window_count ({event.window_count})."
        )
    expected_votes = SENSITIVITY_PRESETS[preset][1]
    if event.votes_required != expected_votes:
        _fail(
            f"event_smoothing.votes_required: preset {preset!r} requires "
            f"{expected_votes}-of-{WINDOW_COUNT}, got {event.votes_required}."
        )


def _validate_sensitivity(sensitivity: SensitivitySpec) -> None:
    alpha, _votes = preset_for(sensitivity.preset)
    if abs(sensitivity.alpha - alpha) > _FLOAT_TOLERANCE:
        _fail(
            f"sensitivity.alpha: preset {sensitivity.preset!r} is defined with alpha={alpha}, "
            f"got {sensitivity.alpha}."
        )


def _validate_calibration(calibration: CalibrationSpec, alpha: float) -> None:
    if calibration.confidence not in (CONFIDENCE_CALIBRATED, CONFIDENCE_LOW):
        _fail(
            f"calibration.confidence: must be {CONFIDENCE_CALIBRATED!r} or {CONFIDENCE_LOW!r}, "
            f"got {calibration.confidence!r}."
        )
    if calibration.observed_exceedances > calibration.held_out_clips:
        _fail(
            f"calibration.observed_exceedances ({calibration.observed_exceedances}) cannot "
            f"exceed held_out_clips ({calibration.held_out_clips})."
        )
    if calibration.held_out_clips >= 1:
        expected = fpr_upper_bound_95(calibration.held_out_clips)
        if abs(calibration.fpr_upper_bound_95 - expected) > 1e-9:
            _fail(
                "calibration.fpr_upper_bound_95: must be "
                f"1 - 0.05 ** (1 / {calibration.held_out_clips}) = {expected!r}, "
                f"got {calibration.fpr_upper_bound_95!r}."
            )
    elif calibration.confidence == CONFIDENCE_CALIBRATED:
        _fail(
            "calibration.confidence: cannot be 'calibrated' with zero held-out clips; there is "
            "nothing the false-positive bound could be computed from."
        )
    if not 0.0 <= calibration.fpr_upper_bound_95 <= 1.0:
        _fail(
            "calibration.fpr_upper_bound_95: must lie in [0, 1], got "
            f"{calibration.fpr_upper_bound_95}."
        )
    if calibration.confidence == CONFIDENCE_CALIBRATED:
        if calibration.held_out_sessions < CALIBRATED_MIN_HELD_OUT_SESSIONS:
            _fail(
                f"calibration.confidence: 'calibrated' needs >= "
                f"{CALIBRATED_MIN_HELD_OUT_SESSIONS} held-out sessions, got "
                f"{calibration.held_out_sessions}."
            )
        if calibration.held_out_clips < CALIBRATED_MIN_HELD_OUT_CLIPS:
            _fail(
                f"calibration.confidence: 'calibrated' needs >= "
                f"{CALIBRATED_MIN_HELD_OUT_CLIPS} held-out clips, got "
                f"{calibration.held_out_clips}."
            )
        if calibration.fpr_upper_bound_95 > 2.0 * alpha + _FLOAT_TOLERANCE:
            _fail(
                "calibration.confidence: 'calibrated' needs the 95% one-sided false-positive "
                f"upper bound ({calibration.fpr_upper_bound_95:.4f}) to be <= 2 x alpha "
                f"({2.0 * alpha:.4f}); collect more held-out normal clips or report 'low'."
            )


def _validate_runtime(name: str, entry: RuntimeThreshold) -> None:
    where = f"runtimes[{name!r}]"
    if name not in RUNTIME_NAMES:
        _fail(f"runtimes: unknown runtime name {name!r}; allowed names are {list(RUNTIME_NAMES)}.")
    if not entry.t_shape > 0.0:
        _fail(f"{where}.t_shape: must be > 0, got {entry.t_shape}.")
    for field_name in ("artifact_sha256", "keras_sha256", "representative_sha256"):
        _require_sha256(getattr(entry, field_name), f"{where}.{field_name}")
    _require_fingerprint(entry.dataset_fingerprint, f"{where}.dataset_fingerprint")
    _require_fingerprint(entry.split_fingerprint, f"{where}.split_fingerprint")
    if entry.normal_p50 < 0.0 or entry.normal_p95 < 0.0:
        _fail(f"{where}: normal_p50/normal_p95 are MSEs and must be >= 0.")
    if entry.normal_p95 < entry.normal_p50:
        _fail(
            f"{where}: normal_p95 ({entry.normal_p95}) must be >= normal_p50 "
            f"({entry.normal_p50})."
        )
    if entry.spread_over_noise_floor < 0.0:
        _fail(
            f"{where}.spread_over_noise_floor: must be >= 0, got {entry.spread_over_noise_floor}."
        )

    quant_fields = (
        entry.input_scale,
        entry.input_zero_point,
        entry.output_scale,
        entry.output_zero_point,
    )
    if name in QUANTIZED_RUNTIMES:
        if any(value is None for value in quant_fields):
            _fail(
                f"{where}: runtime {name!r} has integer I/O, so input_scale, input_zero_point, "
                "output_scale and output_zero_point are all required -- they are part of the "
                "threshold binding tuple because the same .keras converted with a different "
                "representative subset yields differently-scaled MSE."
            )
        if not entry.input_scale > 0.0 or not entry.output_scale > 0.0:
            _fail(f"{where}: input_scale and output_scale must be > 0.")
        low, high = (-128, 127) if name == "int8" else (0, 255)
        for label, value in (
            ("input_zero_point", entry.input_zero_point),
            ("output_zero_point", entry.output_zero_point),
        ):
            if value is None or not low <= value <= high:
                _fail(f"{where}.{label}: must lie in [{low}, {high}] for {name}, got {value}.")
    elif any(value is not None for value in quant_fields):
        _fail(
            f"{where}: runtime {name!r} has float I/O and must store null for input_scale, "
            "input_zero_point, output_scale and output_zero_point."
        )


def validate(config: "AnomalyConfig | Mapping[str, Any]") -> AnomalyConfig:
    """Strictly validate a config (dataclass or JSON mapping) and return the dataclass.

    Raises :class:`AnomalySchemaError` with a precise, user-showable message on the first
    problem found.
    """

    resolved = config if isinstance(config, AnomalyConfig) else from_dict(config)

    if resolved.scorer_schema_version != SCORER_SCHEMA_VERSION:
        _fail(
            f"scorer_schema_version: this build implements {SCORER_SCHEMA_VERSION!r}, the "
            f"config declares {resolved.scorer_schema_version!r}. Re-calibrate; a threshold "
            "from another scorer version means something different."
        )
    if resolved.frontend_schema_version != FRONTEND_SCHEMA_VERSION:
        _fail(
            f"frontend_schema_version: this build implements {FRONTEND_SCHEMA_VERSION!r}, the "
            f"config declares {resolved.frontend_schema_version!r}. The log-mel frontend is a "
            "three-way contract (server / exported runner / MCU doc)."
        )

    _validate_shape(resolved.shape)
    _validate_model(resolved.model, resolved.shape)
    _validate_level(resolved.level)
    _validate_composite(resolved.composite)
    _validate_sensitivity(resolved.sensitivity)
    _validate_event(resolved.event_smoothing, resolved.sensitivity.preset)
    _validate_calibration(resolved.calibration, resolved.sensitivity.alpha)

    if not resolved.runtimes:
        _fail(
            "runtimes: at least one runtime threshold is required; T_shape is calibrated PER "
            f"RUNTIME (allowed names: {list(RUNTIME_NAMES)})."
        )
    for name, entry in resolved.runtimes.items():
        if not isinstance(entry, RuntimeThreshold):
            _fail(f"runtimes[{name!r}]: expected a RuntimeThreshold, got {type(entry).__name__}.")
        _validate_runtime(name, entry)

    first = next(iter(resolved.runtimes.values()))
    for name, entry in resolved.runtimes.items():
        if entry.keras_sha256 != first.keras_sha256:
            _fail(
                f"runtimes[{name!r}].keras_sha256: every runtime in one config must be derived "
                "from the same .keras model."
            )
        if entry.dataset_fingerprint != first.dataset_fingerprint:
            _fail(
                f"runtimes[{name!r}].dataset_fingerprint: every runtime in one config must be "
                "calibrated on the same normal_train dataset."
            )
        if entry.split_fingerprint != first.split_fingerprint:
            _fail(
                f"runtimes[{name!r}].split_fingerprint: every runtime in one config must be "
                "calibrated on the same train/calibration session split."
            )

    return resolved
