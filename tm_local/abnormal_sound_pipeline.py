"""Training and scoring core for ``kind="abnormal_sound"`` (open-set anomalous SOUND).

This module is the executable half of the contract frozen in :mod:`tm_local.anomaly_schema`.
It trains on NORMAL audio only and reports how far a one-second window deviates from the
collected normal baseline.  It is **not** a classifier: there are no labels, no class
weights, no ``sparse_categorical_crossentropy`` and no accuracy metric, and nothing here may
ever name the anomaly.  The user's anomalies are open-ended (dog, cat, fan, unknown).

Shape branch
    ``audio_frontend.log_mel_spectrogram()`` is UNCHANGED and still yields ``(98, 40, 1)``
    per clip.  Five consecutive frames are stacked into 94 context vectors of 200 dims; a
    dense denoising autoencoder ``200 -> 128 -> 64 -> 8 -> 64 -> 128 -> 200`` (ReLU hidden,
    **sigmoid** output, **no** BatchNormalization, 69,200 parameters) reconstructs them, and
    ``S_shape`` is the arithmetic mean of the LARGEST ``top_k=10`` per-context MSEs.
    ``top_k`` and ``n_contexts`` are absolute integers read from the settings; they are never
    recomputed from a fraction, because ``round(0.10 * 94) = 9 != 10`` and letting each
    language redo that rounding guarantees PC/MCU divergence.

Level branch
    The frontend's ``db -= np.max(db)`` discards absolute loudness by construction (a signal
    10 dB louder with identical spectral shape produces a bit-identical tensor, measured AUC
    0.421), so loudness is carried by a separate scalar ``L = rms_dbfs`` computed on the PCM
    after resample and ``fix_length(16000)`` and before the STFT.  ``C_level`` is the median
    over INCLUDED ``normal_train`` clips, ``T_level`` comes from independent calibration
    clips and is floored at 3 dB.  Raw-mel peak dB, PCM peak dBFS and clipping fraction are
    diagnostics only and never enter a score.  Clips whose ``pad_fraction`` is non-zero are
    excluded from the ``C_level`` / ``T_level`` calibration: centre-padding a short recording
    lowers its RMS without touching the peak-normalised log-mel, i.e. it fabricates a level
    anomaly the shape branch cannot contradict.

Composite and temporal smoothing
    ``R = max(S_shape / T_shape[runtime], abs(L - C_level) / T_level)`` and ``R > 1`` means
    the window exceeds the baseline.  Windows advance on a 0.5 s hop and are voted
    ``votes_required``-of-5; fewer than five windows since start is ``UNCERTAIN``.  The MVP
    deliberately applies no post-hoc ``margin_factor``.

Roles
    Exactly one locked ``normal_train`` container plus 0..19 user-nameable ``anomaly_eval``
    groups.  ``anomaly_eval`` data is used ONLY for per-group evaluation reporting.  That is
    enforced *structurally* here, not by convention: normal and evaluation clips are
    different Python types, and every function that trains, calibrates, computes
    contamination statistics or builds the quantization representative set takes a
    :class:`NormalClipSet` and rejects anything else with a ``TrainingError``.

TensorFlow is imported inside the functions that need it, exactly as the existing pipelines
do, so this module stays importable (and unit-testable) without TensorFlow installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Sequence

import numpy as np

from . import anomaly_schema
from .anomaly_schema import (
    CONFIDENCE_CALIBRATED,
    CONTEXT_FRAMES,
    MIN_T_LEVEL_DB,
    QC_HIGH_REVIEW,
    QC_OK,
    QC_WARNING,
    N_CONTEXTS,
    ROLE_ANOMALY_EVAL,
    ROLE_NORMAL_TRAIN,
    TOP_K,
    WINDOW_COUNT,
)
from .audio_frontend import (
    AudioFrontendConfig,
    config_from_mapping,
    fix_length,
    level_diagnostics,
    log_mel_spectrogram,
    read_wav_bytes,
    read_wav_file,
    split_clips,
)
from .config import (
    ABNORMAL_SOUND_DEFAULTS,
    ABNORMAL_SOUND_ENCODER_UNITS,
    ABNORMAL_SOUND_INPUT_DIM,
    ANOMALY_THRESHOLD_RUNTIMES,
    DEFAULT_SENSITIVITY,
    MAX_ANOMALY_EVAL_GROUPS,
    THRESHOLD_QUANTILE_METHOD,
    is_audio_like,
    normalize_kind,
    sensitivity_preset,
    validate_abnormal_sound_settings,
)
from .project_store import ProjectStore
from .training_common import (
    ProgressFn,
    TrainingError,
    history_to_json,
    make_keras_progress_callback,
    save_native_keras_model,
)
from .utils import utc_now_iso

__all__ = [
    "AnomalyEvalClip",
    "AnomalyEvalGroup",
    "ContaminationReport",
    "LevelSample",
    "NormalClip",
    "NormalClipSet",
    "SessionSplit",
    "ThresholdCalibration",
    "abnormal_sound_representative_samples",
    "build_autoencoder",
    "calibrate_thresholds",
    "clip_features",
    "clip_levels",
    "clip_shape_score",
    "collect_project_data",
    "composite_ratio",
    "contamination_scan",
    "context_batch",
    "context_vectors",
    "evaluate_groups",
    "make_anomaly_eval_clips",
    "make_normal_clips",
    "score_clips",
    "session_split",
    "shape_scores",
    "train_abnormal_sound_project",
    "train_autoencoder",
    "vote_verdict",
]

# Seed for every stochastic step (weight init, shuffling, denoising noise, dropout-free but
# still order-dependent kernels).  Same value the image/audio pipelines use, so a reader who
# knows one pipeline is not surprised by another.
RANDOM_SEED = 1337

# Robust-z cut-offs for the contamination scan.  A clip is only ever MARKED; nothing here
# deletes, moves or excludes a recording -- the user decides.
QC_WARNING_Z = 3.5
QC_HIGH_REVIEW_Z = 6.0

# Floor on the MAD used by the robust-z scan.  A very stable room drives MAD towards zero,
# which would turn ordinary dither into a "high review" flag on every clip.
QC_LEVEL_MAD_FLOOR_DB = 0.5
QC_SHAPE_MAD_FLOOR_LOG10 = 0.02

# Layer-1 (no model) recording-quality limits.
QC_CLIPPING_WARNING = 1e-3
QC_CLIPPING_HIGH_REVIEW = 1e-2
QC_SILENCE_WARNING_DBFS = -60.0
QC_DROPOUT_HIGH_REVIEW_DBFS = -95.0


# --------------------------------------------------------------------------------------
# Small local helpers (mirrors of audio_pipeline's, kept private to this module)
# --------------------------------------------------------------------------------------


def _bounded_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return fallback


def _bounded_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return fallback


def _setting(settings: Mapping[str, Any], key: str) -> Any:
    return settings.get(key, ABNORMAL_SOUND_DEFAULTS[key])


def _signal_sha256(signal: np.ndarray) -> str:
    """Content hash of one clip, for :func:`anomaly_schema.dataset_fingerprint`."""

    values = np.ascontiguousarray(np.asarray(signal, dtype=np.float32).reshape(-1))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_higher(values: Sequence[float], quantile: float) -> float:
    """Empirical quantile with numpy ``method="higher"`` -- the locked estimator.

    ``method`` is pinned to :data:`config.THRESHOLD_QUANTILE_METHOD` so the PC and the MCU
    cannot disagree about interpolation.  With few calibration clips this deliberately
    lands on an actual observed value rather than inventing one between two samples.
    """

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise TrainingError("Cannot take a quantile of an empty calibration set.")
    return float(np.quantile(array, float(quantile), method=THRESHOLD_QUANTILE_METHOD))


def _robust_z(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return 0.6745 * np.abs(np.asarray(values, dtype=np.float64) - center) / max(scale, 1e-12)


# --------------------------------------------------------------------------------------
# Clip containers.  The role separation is a TYPE separation, on purpose.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalClip:
    """One INCLUDED ``normal_train`` clip: the only material that may train or calibrate."""

    clip_id: str
    session_id: str
    signal: np.ndarray
    source_samples: int | None = None
    role: ClassVar[str] = ROLE_NORMAL_TRAIN

    @property
    def content_sha256(self) -> str:
        return _signal_sha256(self.signal)


@dataclass(frozen=True)
class AnomalyEvalClip:
    """One ``anomaly_eval`` clip.  Evaluation reporting only -- a different type on purpose.

    It is not a :class:`NormalClip` and never will be, so it cannot be handed to the
    trainer, the threshold calibration, the contamination statistics or the quantization
    representative set even by accident.
    """

    clip_id: str
    session_id: str
    signal: np.ndarray
    group: str
    source_samples: int | None = None
    role: ClassVar[str] = ROLE_ANOMALY_EVAL


@dataclass(frozen=True)
class NormalClipSet:
    """An immutable, type-checked collection of ``normal_train`` clips."""

    clips: tuple[NormalClip, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.clips, tuple):
            object.__setattr__(self, "clips", tuple(self.clips))
        for item in self.clips:
            if not isinstance(item, NormalClip):
                raise TrainingError(
                    "NormalClipSet accepts normal_train clips only; got "
                    f"{type(item).__name__}. anomaly_eval data must never reach training, "
                    "threshold calibration, contamination statistics or the quantization "
                    "representative set."
                )
        seen: set[str] = set()
        for item in self.clips:
            if item.clip_id in seen:
                raise TrainingError(f"Duplicate normal_train clip id {item.clip_id!r}.")
            seen.add(item.clip_id)

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self):
        return iter(self.clips)

    @property
    def session_ids(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for item in self.clips:
            if item.session_id not in ordered:
                ordered.append(item.session_id)
        return tuple(ordered)

    def subset(self, session_ids: Iterable[str]) -> "NormalClipSet":
        wanted = set(session_ids)
        return NormalClipSet(tuple(item for item in self.clips if item.session_id in wanted))

    def fingerprint_entries(self) -> list[tuple[str, str]]:
        return [(item.clip_id, item.content_sha256) for item in self.clips]


@dataclass(frozen=True)
class AnomalyEvalGroup:
    """One user-named evaluation group.  Never training material."""

    name: str
    clips: tuple[AnomalyEvalClip, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.clips, tuple):
            object.__setattr__(self, "clips", tuple(self.clips))
        for item in self.clips:
            if not isinstance(item, AnomalyEvalClip):
                raise TrainingError(
                    f"AnomalyEvalGroup {self.name!r} accepts anomaly_eval clips only; got "
                    f"{type(item).__name__}."
                )

    def __len__(self) -> int:
        return len(self.clips)


def _require_normal_set(value: Any, where: str) -> NormalClipSet:
    """The single choke point that keeps anomaly_eval data out of the normal-only paths."""

    if isinstance(value, NormalClipSet):
        return value
    raise TrainingError(
        f"{where} accepts a NormalClipSet (role={ROLE_NORMAL_TRAIN}) only; got "
        f"{type(value).__name__}. anomaly_eval data is for per-group evaluation reporting "
        "only -- it never enters training, threshold calibration, contamination QC "
        "statistics or the quantization representative set."
    )


# --------------------------------------------------------------------------------------
# Clip construction
# --------------------------------------------------------------------------------------


def _clip_signals(signal: np.ndarray, frontend: AudioFrontendConfig, overlap: float):
    return split_clips(signal, frontend.clip_samples, overlap=overlap)


def make_normal_clips(
    signal: np.ndarray,
    frontend: AudioFrontendConfig,
    *,
    session_id: str,
    clip_prefix: str | None = None,
    overlap: float = 0.0,
) -> list[NormalClip]:
    """Cut one recording into fixed-length ``normal_train`` clips of a single session.

    Every clip of one recording shares the ``recording_session_id``, which is what keeps
    clips of one session from crossing the train/calibration boundary later.
    """

    prefix = clip_prefix or session_id
    source = np.asarray(signal, dtype=np.float32).reshape(-1)
    pieces = _clip_signals(source, frontend, overlap)
    return [
        NormalClip(
            clip_id=f"{prefix}#{index:05d}",
            session_id=str(session_id),
            signal=np.asarray(piece, dtype=np.float32),
            source_samples=int(source.size),
        )
        for index, piece in enumerate(pieces)
    ]


def make_anomaly_eval_clips(
    signal: np.ndarray,
    frontend: AudioFrontendConfig,
    *,
    group: str,
    session_id: str,
    clip_prefix: str | None = None,
    overlap: float = 0.0,
) -> list[AnomalyEvalClip]:
    """Cut one recording into ``anomaly_eval`` clips.  Evaluation reporting only."""

    prefix = clip_prefix or session_id
    source = np.asarray(signal, dtype=np.float32).reshape(-1)
    pieces = _clip_signals(source, frontend, overlap)
    return [
        AnomalyEvalClip(
            clip_id=f"{prefix}#{index:05d}",
            session_id=str(session_id),
            signal=np.asarray(piece, dtype=np.float32),
            group=str(group),
            source_samples=int(source.size),
        )
        for index, piece in enumerate(pieces)
    ]


# --------------------------------------------------------------------------------------
# Features: the frontend is UNCHANGED; we only stack its frames into contexts.
# --------------------------------------------------------------------------------------


def context_vectors(
    spec: np.ndarray,
    *,
    context_frames: int = CONTEXT_FRAMES,
    n_contexts: int | None = None,
) -> np.ndarray:
    """``(98, 40, 1)`` or ``(98, 40)`` log-mel -> ``(94, 200)`` float32 context vectors.

    Frame-major layout: context ``i`` is ``spec[i:i + context_frames].reshape(-1)``, so the
    200 dims run ``frame0 mel0..39, frame1 mel0..39, ...``.  The exported PC runner and the
    MCU C code must use exactly this order.

    ``n_contexts`` is the absolute integer from the settings.  When given it is checked
    against the geometry rather than recomputed, so a settings/frontend mismatch is a loud
    error instead of a silently rescaled score.
    """

    values = np.asarray(spec, dtype=np.float32)
    if values.ndim == 3:
        if values.shape[-1] != 1:
            raise TrainingError(f"Expected a single-channel log-mel, got shape {values.shape}.")
        values = values[..., 0]
    if values.ndim != 2:
        raise TrainingError(f"Expected a (frames, mel_bins) log-mel, got shape {values.shape}.")
    frames, bins = values.shape
    if context_frames < 1:
        raise TrainingError("context_frames must be at least 1.")
    if frames < context_frames:
        raise TrainingError(
            f"The log-mel has {frames} frames, fewer than context_frames={context_frames}."
        )
    produced = frames - context_frames + 1
    if n_contexts is not None and int(n_contexts) != produced:
        raise TrainingError(
            f"n_contexts={int(n_contexts)} disagrees with the frontend geometry ({produced}). "
            "n_contexts is stored as an absolute integer and must be changed together with "
            "clip_seconds/context_frames, followed by a full recalibration."
        )
    windows = np.lib.stride_tricks.sliding_window_view(values, context_frames, axis=0)
    stacked = np.ascontiguousarray(np.transpose(windows, (0, 2, 1)))
    return stacked.reshape(produced, context_frames * bins).astype(np.float32)


def clip_features(clips: Sequence[Any], frontend: AudioFrontendConfig) -> np.ndarray:
    """``(N, 98, 40, 1)`` log-mel tensor for any sequence of clips (normal or eval)."""

    if not clips:
        return np.zeros((0, *frontend.feature_shape), dtype=np.float32)
    return np.stack(
        [log_mel_spectrogram(item.signal, frontend) for item in clips]
    ).astype(np.float32)


def context_batch(
    features: np.ndarray,
    *,
    context_frames: int = CONTEXT_FRAMES,
    n_contexts: int | None = None,
) -> np.ndarray:
    """``(N, 98, 40, 1)`` -> ``(N, 94, 200)``."""

    values = np.asarray(features, dtype=np.float32)
    if values.shape[0] == 0:
        contexts = int(n_contexts or 0)
        return np.zeros((0, contexts, ABNORMAL_SOUND_INPUT_DIM), dtype=np.float32)
    return np.stack(
        [
            context_vectors(item, context_frames=context_frames, n_contexts=n_contexts)
            for item in values
        ]
    ).astype(np.float32)


@dataclass(frozen=True)
class LevelSample:
    """One clip's level branch input plus the diagnostics that gate its use.

    ``pad_fraction > 0`` means ``fix_length()`` centre-padded a short recording, which lowers
    RMS while leaving the peak-normalised log-mel untouched.  Such clips are excluded from
    the ``C_level`` / ``T_level`` calibration; they are never deleted.
    """

    clip_id: str
    session_id: str
    level_dbfs: float
    peak_dbfs: float
    clipping_fraction: float
    pad_fraction: float

    @property
    def usable_for_level_calibration(self) -> bool:
        return self.pad_fraction <= 0.0 and math.isfinite(self.level_dbfs)


def clip_levels(clips: Sequence[Any], frontend: AudioFrontendConfig) -> list[LevelSample]:
    """RMS dBFS plus the recording-quality diagnostics for each clip."""

    samples: list[LevelSample] = []
    for item in clips:
        metrics = level_diagnostics(
            item.signal, frontend, source_samples=getattr(item, "source_samples", None)
        )
        samples.append(
            LevelSample(
                clip_id=item.clip_id,
                session_id=item.session_id,
                level_dbfs=float(metrics["rms_dbfs"]),
                peak_dbfs=float(metrics["peak_dbfs"]),
                clipping_fraction=float(metrics["clipping_fraction"]),
                pad_fraction=float(metrics["pad_fraction"]),
            )
        )
    return samples


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


def build_autoencoder(
    vector_dim: int = ABNORMAL_SOUND_INPUT_DIM,
    bottleneck_dim: int = 8,
    *,
    encoder_units: Sequence[int] = ABNORMAL_SOUND_ENCODER_UNITS,
    name: str = "tm_local_abnormal_sound_autoencoder",
):
    """Dense denoising autoencoder ``200 -> 128 -> 64 -> 8 -> 64 -> 128 -> 200``.

    ReLU hidden layers, **no BatchNormalization**, **sigmoid** output.  The sigmoid is
    load-bearing rather than cosmetic: it pins the TFLite int8 output quantization to a fixed
    scale of exactly ``1/256 = 0.00390625``, which dropped the measured quantization noise
    floor from 2.002e-04 (linear output) to 2.393e-06 and lifted the score margin from 30.5x
    to 81.9x.  A linear output derives its scale from the representative data and is
    unbounded.  Do not switch to linear and do not add BatchNorm.
    """

    import tensorflow as tf

    units = [int(item) for item in encoder_units]
    if any(item < 1 for item in units):
        raise TrainingError("Encoder units must all be >= 1.")
    if int(bottleneck_dim) < 1:
        raise TrainingError("bottleneck_dim must be at least 1.")
    inputs = tf.keras.Input(shape=(int(vector_dim),), dtype=tf.float32, name="context")
    x = inputs
    for index, count in enumerate(units):
        x = tf.keras.layers.Dense(count, activation="relu", name=f"encoder_{index + 1}")(x)
    x = tf.keras.layers.Dense(int(bottleneck_dim), activation="relu", name="bottleneck")(x)
    for index, count in enumerate(reversed(units)):
        x = tf.keras.layers.Dense(count, activation="relu", name=f"decoder_{index + 1}")(x)
    outputs = tf.keras.layers.Dense(
        int(vector_dim), activation="sigmoid", name="reconstruction"
    )(x)
    return tf.keras.Model(inputs, outputs, name=name)


def _make_denoising_dataset(vectors: np.ndarray, batch_size: int, sigma: float, seed: int):
    """``clip(x + N(0, sigma), 0, 1)`` in, clean ``x`` out.

    ``num_parallel_calls`` is deliberately left unset: an order-deterministic single-threaded
    map plus a seeded shuffle keeps the whole pipeline reproducible, which matters because
    the trained weights decide every stored threshold.
    """

    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(vectors)
    dataset = dataset.shuffle(
        min(max(1024, batch_size * 8), max(1, len(vectors))),
        seed=seed,
        reshuffle_each_iteration=True,
    )

    def corrupt(clean):  # noqa: ANN001
        noise = tf.random.normal(tf.shape(clean), stddev=sigma, dtype=tf.float32)
        return tf.clip_by_value(clean + noise, 0.0, 1.0), clean

    # No num_parallel_calls: a single-threaded, order-deterministic map plus a seeded
    # shuffle keeps the whole pipeline reproducible, which matters because the trained
    # weights decide every stored threshold.
    if sigma > 0.0:
        dataset = dataset.map(corrupt)
    else:
        dataset = dataset.map(lambda clean: (clean, clean))
    return dataset.batch(batch_size).prefetch(1)


@dataclass(frozen=True)
class AutoencoderTraining:
    model: Any
    history: dict[str, list[float]]
    epochs_requested: int
    epochs_completed: int
    train_vectors: int
    validation_vectors: int
    parameter_count: int


def train_autoencoder(
    normal_train: NormalClipSet,
    settings: Mapping[str, Any],
    *,
    frontend: AudioFrontendConfig | None = None,
    validation: NormalClipSet | None = None,
    progress: ProgressFn | None = None,
    progress_range: tuple[float, float] = (0.16, 0.67),
    seed: int = RANDOM_SEED,
) -> AutoencoderTraining:
    """Fit the denoising autoencoder on ``normal_train`` clips only."""

    normal_train = _require_normal_set(normal_train, "train_autoencoder()")
    if validation is not None:
        validation = _require_normal_set(validation, "train_autoencoder(validation=...)")
    if len(normal_train) == 0:
        raise TrainingError("The training side of the session split holds no clips.")

    resolved = frontend or config_from_mapping(dict(settings))
    context_frames = int(_setting(settings, "context_frames"))
    n_contexts = int(_setting(settings, "n_contexts"))
    epochs = _bounded_int(_setting(settings, "epochs"), 1, 300, 60)
    batch_size = _bounded_int(_setting(settings, "batch_size"), 1, 1024, 64)
    learning_rate = _bounded_float(_setting(settings, "learning_rate"), 1e-6, 0.1, 0.001)
    sigma = _bounded_float(_setting(settings, "denoise_sigma"), 0.0, 1.0, 0.01)
    bottleneck = _bounded_int(_setting(settings, "bottleneck_dim"), 1, 256, 8)

    train_features = clip_features(normal_train.clips, resolved)
    train_vectors = context_batch(
        train_features, context_frames=context_frames, n_contexts=n_contexts
    ).reshape(-1, context_frames * resolved.mel_bins)
    validation_vectors = None
    if validation is not None and len(validation):
        validation_features = clip_features(validation.clips, resolved)
        validation_vectors = context_batch(
            validation_features, context_frames=context_frames, n_contexts=n_contexts
        ).reshape(-1, context_frames * resolved.mel_bins)

    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_autoencoder(train_vectors.shape[1], bottleneck)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    train_ds = _make_denoising_dataset(train_vectors, batch_size, sigma, seed)
    validation_ds = None
    if validation_vectors is not None and len(validation_vectors):
        validation_ds = _make_denoising_dataset(validation_vectors, batch_size, 0.0, seed + 1)
    callbacks: list[Any] = [
        make_keras_progress_callback(
            epochs, progress, start=progress_range[0], end=progress_range[1]
        )
    ]
    if bool(settings.get("early_stopping", False)):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_ds is not None else "loss",
                patience=max(3, min(10, epochs // 4)),
                restore_best_weights=True,
            )
        )
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
    )
    return AutoencoderTraining(
        model=model,
        history=history_to_json(history),
        epochs_requested=epochs,
        epochs_completed=len(history.history.get("loss", [])),
        train_vectors=int(train_vectors.shape[0]),
        validation_vectors=int(0 if validation_vectors is None else validation_vectors.shape[0]),
        parameter_count=int(model.count_params()),
    )


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def clip_shape_score(
    vectors: np.ndarray, recon: np.ndarray, top_k: int = TOP_K
) -> float:
    """``S_shape``: arithmetic mean of the LARGEST ``top_k`` per-context MSEs.

    ``top_k`` is the absolute integer from the settings.  It is never derived from a
    fraction of ``n_contexts`` here or anywhere else.
    """

    original = np.asarray(vectors, dtype=np.float64)
    reconstructed = np.asarray(recon, dtype=np.float64)
    if original.shape != reconstructed.shape:
        raise TrainingError(
            f"Context/reconstruction shape mismatch: {original.shape} vs {reconstructed.shape}."
        )
    if original.ndim != 2:
        raise TrainingError(f"Expected (n_contexts, vector_dim), got shape {original.shape}.")
    errors = np.mean((original - reconstructed) ** 2, axis=1)
    # anomaly_schema.top_k_mean is the shared pure-Python reduction (math.fsum), so the
    # server, the exported runner and the MCU reference agree bit-for-bit on the ordering
    # and the summation.
    return float(anomaly_schema.top_k_mean([float(value) for value in errors], int(top_k)))


def shape_scores(
    model: Any,
    features: np.ndarray,
    *,
    top_k: int = TOP_K,
    context_frames: int = CONTEXT_FRAMES,
    n_contexts: int | None = None,
    batch_size: int = 2048,
) -> np.ndarray:
    """``S_shape`` for every clip in a ``(N, 98, 40, 1)`` feature batch."""

    values = np.asarray(features, dtype=np.float32)
    if values.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    contexts = context_batch(values, context_frames=context_frames, n_contexts=n_contexts)
    count, per_clip, dim = contexts.shape
    flat = contexts.reshape(count * per_clip, dim)
    # Eager batched __call__ rather than Model.predict(): predict() builds a new traced
    # tf.function per model, and the contamination scan fits one throwaway model per
    # session, which floods the student-facing log with retracing warnings.
    chunks: list[np.ndarray] = []
    step = max(1, int(batch_size))
    for start in range(0, flat.shape[0], step):
        piece = flat[start : start + step]
        try:
            chunks.append(np.asarray(model(piece, training=False), dtype=np.float32))
        except TypeError:  # pragma: no cover - non-keras callables
            chunks.append(np.asarray(model(piece), dtype=np.float32))
    recon = np.concatenate(chunks, axis=0).reshape(count, per_clip, dim)
    return np.asarray(
        [clip_shape_score(contexts[index], recon[index], top_k) for index in range(count)],
        dtype=np.float64,
    )


def composite_ratio(
    shape_score: float,
    level_dbfs: float,
    t_shape: float,
    c_level: float,
    t_level: float,
) -> float:
    """``R = max(S_shape / T_shape[runtime], abs(L - C_level) / T_level)``.

    ``R > 1`` means this window exceeds the collected normal baseline.  The MVP applies no
    post-hoc ``margin_factor``: a multiplicative fudge on top of the quantile was rejected
    because it is not derived from held-out data and hides, rather than fixes, a calibration
    set that is too small.
    """

    return float(
        anomaly_schema.composite_ratio(
            s_shape=float(shape_score),
            t_shape=float(t_shape),
            level_db=float(level_dbfs),
            c_level_db=float(c_level),
            t_level_db=float(t_level),
        )
    )


def vote_verdict(
    window_ratios: Sequence[float],
    votes_required: int,
    window_count: int = WINDOW_COUNT,
) -> str:
    """``NORMAL`` / ``ABNORMAL`` / ``UNCERTAIN`` from the most recent window ratios.

    Fewer than ``window_count`` windows since start is ``UNCERTAIN`` (warming up); the
    detector never alarms early.  The UI must always show the RAW per-window exceedance
    timeline as well, not only this voted verdict.
    """

    flags = [anomaly_schema.exceeds(float(value)) for value in window_ratios]
    return anomaly_schema.smooth_state(flags, int(votes_required), int(window_count))


@dataclass(frozen=True)
class ClipScores:
    clip_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    shape: np.ndarray
    level_dbfs: np.ndarray
    pad_fraction: np.ndarray

    def ratios(self, calibration: "ThresholdCalibration") -> np.ndarray:
        return np.asarray(
            [
                composite_ratio(
                    float(self.shape[index]),
                    float(self.level_dbfs[index]),
                    calibration.t_shape,
                    calibration.c_level_db,
                    calibration.t_level_db,
                )
                for index in range(len(self.clip_ids))
            ],
            dtype=np.float64,
        )


def score_clips(
    model: Any,
    clips: Sequence[Any],
    settings: Mapping[str, Any],
    *,
    frontend: AudioFrontendConfig | None = None,
) -> ClipScores:
    """Shape score + level for any sequence of clips (normal or anomaly_eval)."""

    resolved = frontend or config_from_mapping(dict(settings))
    context_frames = int(_setting(settings, "context_frames"))
    n_contexts = int(_setting(settings, "n_contexts"))
    top_k = int(_setting(settings, "top_k"))
    features = clip_features(clips, resolved)
    scores = shape_scores(
        model,
        features,
        top_k=top_k,
        context_frames=context_frames,
        n_contexts=n_contexts,
    )
    levels = clip_levels(clips, resolved)
    return ClipScores(
        clip_ids=tuple(item.clip_id for item in clips),
        session_ids=tuple(item.session_id for item in clips),
        shape=scores,
        level_dbfs=np.asarray([item.level_dbfs for item in levels], dtype=np.float64),
        pad_fraction=np.asarray([item.pad_fraction for item in levels], dtype=np.float64),
    )


# --------------------------------------------------------------------------------------
# Session split
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionSplit:
    """Deterministic train / calibration / audit session partition.

    ``val_sessions = ceil(0.25 * n_sessions)`` sessions are held out, ordered by
    ``sha256(session_id)`` so every re-run and every implementation picks the same ones.
    Clips from one session NEVER cross the boundary.

    The held-out side is split once more into ``calibration`` (which produces ``T_shape`` /
    ``T_level``) and ``audit`` (on which the observed false-positive count is measured).
    Measuring exceedances on the very clips that defined the quantile would, by construction
    with ``method="higher"``, return 0 and turn the reported upper bound into a fiction.
    With a single held-out session there is no audit side; the caller is told so explicitly
    and the count is reported as in-sample.
    """

    train: tuple[str, ...]
    calibration: tuple[str, ...]
    audit: tuple[str, ...]

    @property
    def held_out(self) -> tuple[str, ...]:
        return tuple(self.calibration) + tuple(self.audit)

    @property
    def has_audit(self) -> bool:
        return bool(self.audit)


def session_split(sessions: Iterable[str]) -> SessionSplit:
    """Split recording sessions into ``(train, calibration, audit)``.

    Refuses, with a message naming the shortfall, when the split cannot leave at least one
    session on each side.
    """

    try:
        train, held_out = anomaly_schema.select_held_out_sessions(sessions)
    except anomaly_schema.AnomalySchemaError as exc:
        raise TrainingError(str(exc)) from exc
    # Deterministic sub-split of the held-out side, same sha256 ordering.
    ordered = sorted(
        held_out, key=lambda item: (hashlib.sha256(item.encode("utf-8")).hexdigest(), item)
    )
    if len(ordered) < 2:
        return SessionSplit(train=tuple(train), calibration=tuple(ordered), audit=())
    cut = math.ceil(len(ordered) / 2)
    return SessionSplit(
        train=tuple(train), calibration=tuple(ordered[:cut]), audit=tuple(ordered[cut:])
    )


# --------------------------------------------------------------------------------------
# Contamination QC -- MARK ONLY.  Never delete, exclude or move a recording.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContaminationReport:
    clips: tuple[dict[str, Any], ...]
    counts: dict[str, int]
    notes: tuple[str, ...]
    level_center_dbfs: float
    level_mad_db: float
    shape_scan_ran: bool

    @property
    def high_review(self) -> tuple[dict[str, Any], ...]:
        return tuple(item for item in self.clips if item["status"] == QC_HIGH_REVIEW)

    @property
    def unresolved_high_review(self) -> int:
        return len(self.high_review)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": (
                "mark only -- the system never auto-deletes, auto-excludes or auto-moves a "
                "recording; the user decides on every flagged clip"
            ),
            "counts": dict(self.counts),
            "level_center_dbfs": self.level_center_dbfs,
            "level_mad_db": self.level_mad_db,
            "shape_scan_ran": self.shape_scan_ran,
            "notes": list(self.notes),
            "clips": [dict(item) for item in self.clips],
        }


def _layer_one_flags(sample: LevelSample) -> list[tuple[str, str]]:
    """No-model recording-quality checks.  Returns ``(status, reason)`` pairs."""

    found: list[tuple[str, str]] = []
    if not math.isfinite(sample.level_dbfs) or not math.isfinite(sample.peak_dbfs):
        found.append((QC_HIGH_REVIEW, "the clip did not decode to finite samples"))
    if sample.clipping_fraction >= QC_CLIPPING_HIGH_REVIEW:
        found.append(
            (
                QC_HIGH_REVIEW,
                f"{sample.clipping_fraction * 100:.2f}% of samples are at full scale "
                "(input gain is too high; the clipped spectrum is not this room)",
            )
        )
    elif sample.clipping_fraction >= QC_CLIPPING_WARNING:
        found.append(
            (
                QC_WARNING,
                f"{sample.clipping_fraction * 100:.3f}% of samples touch full scale",
            )
        )
    if sample.pad_fraction > 0.0:
        found.append(
            (
                QC_WARNING,
                f"{sample.pad_fraction * 100:.1f}% of this clip is zero padding added to a "
                "short recording; it is excluded from the level calibration",
            )
        )
    if sample.level_dbfs <= QC_DROPOUT_HIGH_REVIEW_DBFS:
        found.append((QC_HIGH_REVIEW, "the clip is digital silence (a dropout or a dead mic)"))
    elif sample.level_dbfs <= QC_SILENCE_WARNING_DBFS:
        found.append(
            (QC_WARNING, f"the clip is very quiet ({sample.level_dbfs:.1f} dBFS)")
        )
    return found


def _session_balanced_center_and_mad(
    values: np.ndarray, session_ids: Sequence[str], floor: float
) -> tuple[float, float]:
    """Median / MAD that one long session cannot dominate.

    Per-session medians are taken first, then the median across sessions.  A high-breakdown
    estimator is the point: a contaminated baseline was measured to inflate the level IQR
    15.9x and drag the threshold up until detection fell from 92.3% to 7.7%, so the statistic
    used to FIND contamination must itself survive it.
    """

    order: list[str] = []
    for item in session_ids:
        if item not in order:
            order.append(item)
    per_session = []
    for session in order:
        mask = np.asarray([item == session for item in session_ids], dtype=bool)
        if mask.any():
            per_session.append(float(np.median(values[mask])))
    center = float(np.median(np.asarray(per_session, dtype=np.float64)))
    deviations = np.abs(values - center)
    per_session_mad = []
    for session in order:
        mask = np.asarray([item == session for item in session_ids], dtype=bool)
        if mask.any():
            per_session_mad.append(float(np.median(deviations[mask])))
    mad = float(np.median(np.asarray(per_session_mad, dtype=np.float64)))
    return center, max(mad, floor)


def _provisional_shape_scores(
    normal: NormalClipSet,
    settings: Mapping[str, Any],
    frontend: AudioFrontendConfig,
    *,
    epochs: int,
    progress: ProgressFn | None,
) -> np.ndarray | None:
    """Session-cross-fitted provisional AE scores, or ``None`` when it cannot be run.

    For every session a throwaway autoencoder is fitted on all OTHER sessions and used to
    score that session's clips, so a contaminated session cannot teach the model to consider
    its own contamination normal.
    """

    sessions = normal.session_ids
    if len(sessions) < 2:
        return None
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:  # pragma: no cover - TF is present in the shipped .venv
        return None

    context_frames = int(_setting(settings, "context_frames"))
    n_contexts = int(_setting(settings, "n_contexts"))
    top_k = int(_setting(settings, "top_k"))
    scan_settings = dict(settings)
    scan_settings["epochs"] = int(epochs)
    scan_settings.pop("early_stopping", None)

    scores = np.zeros(len(normal.clips), dtype=np.float64)
    index_of = {item.clip_id: position for position, item in enumerate(normal.clips)}
    for order, session in enumerate(sessions):
        held = normal.subset([session])
        rest = normal.subset([item for item in sessions if item != session])
        if len(rest) == 0 or len(held) == 0:
            continue
        if progress:
            progress(
                0.04 + 0.06 * (order + 1) / max(1, len(sessions)),
                f"Contamination shape scan {order + 1}/{len(sessions)}",
            )
        fitted = train_autoencoder(
            rest,
            scan_settings,
            frontend=frontend,
            progress=None,
            seed=RANDOM_SEED + 101 + order,
        )
        features = clip_features(held.clips, frontend)
        held_scores = shape_scores(
            fitted.model,
            features,
            top_k=top_k,
            context_frames=context_frames,
            n_contexts=n_contexts,
        )
        for position, item in enumerate(held.clips):
            scores[index_of[item.clip_id]] = float(held_scores[position])
    return scores


def contamination_scan(
    normal_train: NormalClipSet,
    settings: Mapping[str, Any],
    *,
    frontend: AudioFrontendConfig | None = None,
    use_shape_scan: bool = True,
    shape_scan_epochs: int = 20,
    progress: ProgressFn | None = None,
) -> ContaminationReport:
    """Two-layer contamination QC over the ``normal_train`` baseline.

    Layer 1 (no model): RMS / peak dBFS, clipping fraction, silence-dropout and finiteness.
    Layer 2: a session-balanced median/MAD level scan plus a session-cross-fitted provisional
    autoencoder shape scan.

    The result only ever MARKS a clip ``ok`` / ``warning`` / ``high_review`` with a reason.
    Nothing is deleted, excluded or moved -- automatically discarding a student's recording is
    a worse failure than keeping a suspicious one, and while unresolved ``high_review`` clips
    remain the detector must not be reported as calibrated.

    ``anomaly_eval`` clips are structurally unable to reach this function, so they can never
    shift the contamination statistics of the normal baseline.
    """

    normal_train = _require_normal_set(normal_train, "contamination_scan()")
    resolved = frontend or config_from_mapping(dict(settings))
    notes: list[str] = []
    records: list[dict[str, Any]] = []
    if len(normal_train) == 0:
        return ContaminationReport(
            clips=(),
            counts={QC_OK: 0, QC_WARNING: 0, QC_HIGH_REVIEW: 0},
            notes=("no normal_train clips to scan",),
            level_center_dbfs=0.0,
            level_mad_db=0.0,
            shape_scan_ran=False,
        )

    samples = clip_levels(normal_train.clips, resolved)
    levels = np.asarray([item.level_dbfs for item in samples], dtype=np.float64)
    session_ids = [item.session_id for item in samples]
    center, mad = _session_balanced_center_and_mad(levels, session_ids, QC_LEVEL_MAD_FLOOR_DB)
    level_z = _robust_z(levels, center, mad)

    shape_values = None
    if use_shape_scan:
        shape_values = _provisional_shape_scores(
            normal_train,
            settings,
            resolved,
            epochs=shape_scan_epochs,
            progress=progress,
        )
        if shape_values is None:
            notes.append(
                "the provisional shape scan needs at least two recording sessions and "
                "TensorFlow; only the level scan ran"
            )
    else:
        notes.append("the provisional shape scan was disabled by the caller")

    shape_z = np.zeros(len(samples), dtype=np.float64)
    shape_center = 0.0
    if shape_values is not None:
        logs = np.log10(np.maximum(shape_values, 1e-12))
        shape_center, shape_mad = _session_balanced_center_and_mad(
            logs, session_ids, QC_SHAPE_MAD_FLOOR_LOG10
        )
        shape_z = _robust_z(logs, shape_center, shape_mad)

    counts = {QC_OK: 0, QC_WARNING: 0, QC_HIGH_REVIEW: 0}
    for index, sample in enumerate(samples):
        flags = _layer_one_flags(sample)
        z_level = float(level_z[index])
        if z_level > QC_HIGH_REVIEW_Z:
            flags.append(
                (
                    QC_HIGH_REVIEW,
                    f"loudness {sample.level_dbfs:.1f} dBFS is {z_level:.1f} robust sigma from "
                    f"the session-balanced baseline ({center:.1f} dBFS)",
                )
            )
        elif z_level > QC_WARNING_Z:
            flags.append(
                (
                    QC_WARNING,
                    f"loudness {sample.level_dbfs:.1f} dBFS is {z_level:.1f} robust sigma from "
                    f"the session-balanced baseline ({center:.1f} dBFS)",
                )
            )
        if shape_values is not None:
            z_shape = float(shape_z[index])
            if z_shape > QC_HIGH_REVIEW_Z:
                flags.append(
                    (
                        QC_HIGH_REVIEW,
                        f"spectral content is {z_shape:.1f} robust sigma from the other "
                        "sessions (cross-fitted provisional autoencoder)",
                    )
                )
            elif z_shape > QC_WARNING_Z:
                flags.append(
                    (
                        QC_WARNING,
                        f"spectral content is {z_shape:.1f} robust sigma from the other "
                        "sessions (cross-fitted provisional autoencoder)",
                    )
                )
        status = QC_OK
        if any(item[0] == QC_HIGH_REVIEW for item in flags):
            status = QC_HIGH_REVIEW
        elif any(item[0] == QC_WARNING for item in flags):
            status = QC_WARNING
        counts[status] += 1
        records.append(
            {
                "clip_id": sample.clip_id,
                "session_id": sample.session_id,
                "status": status,
                "reasons": [item[1] for item in flags],
                "metrics": {
                    "rms_dbfs": sample.level_dbfs,
                    "peak_dbfs": sample.peak_dbfs,
                    "clipping_fraction": sample.clipping_fraction,
                    "pad_fraction": sample.pad_fraction,
                    "level_robust_z": z_level,
                    "shape_score": (
                        None if shape_values is None else float(shape_values[index])
                    ),
                    "shape_robust_z": (None if shape_values is None else float(shape_z[index])),
                },
            }
        )

    if counts[QC_HIGH_REVIEW]:
        notes.append(
            f"{counts[QC_HIGH_REVIEW]} clip(s) need a human decision before this detector may "
            "be called calibrated; nothing was removed"
        )
    return ContaminationReport(
        clips=tuple(records),
        counts=counts,
        notes=tuple(notes),
        level_center_dbfs=float(center),
        level_mad_db=float(mad),
        shape_scan_ran=shape_values is not None,
    )


# --------------------------------------------------------------------------------------
# Threshold calibration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdCalibration:
    runtime: str
    t_shape: float
    c_level_db: float
    t_level_db: float
    alpha: float
    quantile: float
    calibration_clips: int
    level_calibration_clips: int
    excluded_padded_clips: int
    audit_clips: int
    observed_exceedances: int
    exceedances_in_sample: bool
    fpr_upper_bound_95: float
    normal_p50: float
    normal_p95: float
    spread: float
    level_floor_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "t_shape": self.t_shape,
            "c_level_db": self.c_level_db,
            "t_level_db": self.t_level_db,
            "alpha": self.alpha,
            "quantile": self.quantile,
            "calibration_clips": self.calibration_clips,
            "level_calibration_clips": self.level_calibration_clips,
            "excluded_padded_clips": self.excluded_padded_clips,
            "audit_clips": self.audit_clips,
            "observed_exceedances": self.observed_exceedances,
            "exceedances_in_sample": self.exceedances_in_sample,
            "fpr_upper_bound_95": self.fpr_upper_bound_95,
            "normal_p50": self.normal_p50,
            "normal_p95": self.normal_p95,
            "spread": self.spread,
            "level_floor_applied": self.level_floor_applied,
            "honesty_note": (
                "the quantile is an empirical estimate on a finite calibration set, not an "
                "accuracy guarantee; read observed_exceedances together with "
                "fpr_upper_bound_95"
            ),
        }


def calibrate_thresholds(
    *,
    runtime: str,
    train_levels: Sequence[LevelSample],
    calibration_shape_scores: Sequence[float],
    calibration_levels: Sequence[LevelSample],
    alpha: float,
    level_floor_db: float = MIN_T_LEVEL_DB,
    audit_shape_scores: Sequence[float] | None = None,
    audit_levels: Sequence[LevelSample] | None = None,
) -> ThresholdCalibration:
    """Calibrate ``T_shape`` (per runtime), ``C_level`` and ``T_level`` from normal clips.

    * ``T_shape`` and ``T_level`` are per-branch empirical quantiles at ``1 - alpha / 2``
      with numpy ``method="higher"``.
    * ``C_level`` is the median over INCLUDED ``normal_train`` clips.
    * ``T_level`` is floored at ``level_floor_db`` (3 dB): without a floor a very stable room
      drives it towards zero and every ordinary loudness wobble alarms.
    * Clips with ``pad_fraction > 0`` are excluded from ``C_level`` / ``T_level`` and counted.
    * The observed exceedance count is measured on the ``audit`` clips when they are given.
      Counting exceedances on the very clips that defined the quantile returns 0 by
      construction, which would make the reported bound meaningless; when no audit clips are
      available the count is still reported but flagged ``exceedances_in_sample``.

    ``T_shape`` is calibrated PER RUNTIME (keras / float32 / dynamic / int8 / uint8): measured
    threshold drift keras -> int8 was +70% on the rejected architecture, so this is a hard
    contract even though the accepted sigmoid architecture barely drifts.
    """

    name = str(runtime)
    if name not in ANOMALY_THRESHOLD_RUNTIMES:
        raise TrainingError(
            f"Unknown runtime {name!r}; T_shape is calibrated per runtime "
            f"({', '.join(ANOMALY_THRESHOLD_RUNTIMES)})."
        )
    if not 0.0 < float(alpha) < 1.0:
        raise TrainingError(f"alpha must be strictly between 0 and 1, got {alpha}.")

    scores = np.asarray([float(value) for value in calibration_shape_scores], dtype=np.float64)
    if scores.size == 0:
        raise TrainingError("The calibration side holds no clips, so T_shape cannot be set.")

    train_usable = [item for item in train_levels if item.usable_for_level_calibration]
    if not train_usable:
        raise TrainingError(
            "Every normal_train clip is zero padded, so C_level cannot be estimated. "
            "Record normal audio at least as long as clip_seconds."
        )
    c_level = float(np.median(np.asarray([item.level_dbfs for item in train_usable])))

    level_usable = [item for item in calibration_levels if item.usable_for_level_calibration]
    excluded = len(calibration_levels) - len(level_usable)
    if not level_usable:
        raise TrainingError(
            "Every calibration clip is zero padded, so T_level cannot be estimated. "
            "Zero padding lowers RMS without changing the peak-normalised log-mel, which "
            "fabricates a level anomaly the shape branch cannot contradict."
        )

    quantile = 1.0 - float(alpha) / 2.0
    t_shape = _quantile_higher(scores.tolist(), quantile)
    if not t_shape > 0.0:
        raise TrainingError(
            "T_shape came out as zero: the autoencoder reconstructs the calibration clips "
            "exactly. Collect more varied normal audio or reduce bottleneck_dim."
        )
    deviations = [abs(item.level_dbfs - c_level) for item in level_usable]
    raw_t_level = _quantile_higher(deviations, quantile)
    t_level = max(raw_t_level, float(level_floor_db), MIN_T_LEVEL_DB)

    audit_scores = list(audit_shape_scores or [])
    in_sample = False
    if audit_scores and len(audit_scores) == len(list(audit_levels or [])):
        # Keep the score/level pairing intact after the padded-clip filter.
        pairs = [
            (float(score), item)
            for score, item in zip(audit_scores, list(audit_levels or []))
            if item.usable_for_level_calibration
        ]
    elif audit_scores:
        raise TrainingError("audit_shape_scores and audit_levels must have the same length.")
    else:
        pairs = []
    if not pairs:
        in_sample = True
        pairs = [
            (float(score), item)
            for score, item in zip(scores.tolist(), calibration_levels)
            if item.usable_for_level_calibration
        ]
    exceedances = sum(
        1
        for score, item in pairs
        if composite_ratio(score, item.level_dbfs, t_shape, c_level, t_level) > 1.0
    )
    evaluated = len(pairs)
    bound = (
        float(anomaly_schema.binomial_upper_bound_95(evaluated, exceedances))
        if evaluated >= 1
        else 1.0
    )
    return ThresholdCalibration(
        runtime=name,
        t_shape=float(t_shape),
        c_level_db=float(c_level),
        t_level_db=float(t_level),
        alpha=float(alpha),
        quantile=float(quantile),
        calibration_clips=int(scores.size),
        level_calibration_clips=len(level_usable),
        excluded_padded_clips=int(excluded),
        audit_clips=int(evaluated),
        observed_exceedances=int(exceedances),
        exceedances_in_sample=bool(in_sample),
        fpr_upper_bound_95=float(bound),
        normal_p50=float(np.median(scores)),
        normal_p95=float(_quantile_higher(scores.tolist(), 0.95)),
        spread=float(_quantile_higher(scores.tolist(), 0.95) - float(np.median(scores))),
        level_floor_applied=bool(t_level > raw_t_level),
    )


# --------------------------------------------------------------------------------------
# Per-group evaluation (anomaly_eval only)
# --------------------------------------------------------------------------------------


def evaluate_groups(
    groups: Sequence[AnomalyEvalGroup],
    *,
    model: Any,
    calibration: ThresholdCalibration,
    settings: Mapping[str, Any],
    frontend: AudioFrontendConfig | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Per-named-group ``detected / total`` for ``anomaly_eval`` data only.

    Nothing computed here can move a threshold: the calibration is an input, and the groups
    are :class:`AnomalyEvalGroup` instances, which the trainer and the calibrator refuse.
    The report never names the anomaly beyond echoing the group name the user chose -- the
    detector only states that a window deviates from the collected normal baseline.
    """

    for group in groups:
        if not isinstance(group, AnomalyEvalGroup):
            raise TrainingError(
                "evaluate_groups() accepts AnomalyEvalGroup instances only; got "
                f"{type(group).__name__}."
            )
    if len(groups) > MAX_ANOMALY_EVAL_GROUPS:
        raise TrainingError(
            f"At most {MAX_ANOMALY_EVAL_GROUPS} anomaly_eval groups are allowed, "
            f"got {len(groups)}."
        )
    resolved = frontend or config_from_mapping(dict(settings))
    results: list[dict[str, Any]] = []
    for order, group in enumerate(groups):
        if progress:
            progress(
                0.80 + 0.08 * (order + 1) / max(1, len(groups)),
                f"Evaluating group {order + 1}/{len(groups)}",
            )
        if not len(group):
            results.append(
                {
                    "name": group.name,
                    "clips": 0,
                    "detected": 0,
                    "detection_rate": None,
                    "note": "no clips in this group",
                }
            )
            continue
        scored = score_clips(model, group.clips, settings, frontend=resolved)
        ratios = scored.ratios(calibration)
        shape_component = scored.shape / calibration.t_shape
        level_component = (
            np.abs(scored.level_dbfs - calibration.c_level_db) / calibration.t_level_db
        )
        detected = int(np.count_nonzero(ratios > 1.0))
        results.append(
            {
                "name": group.name,
                "clips": int(len(ratios)),
                "detected": detected,
                "detection_rate": float(detected) / float(len(ratios)),
                "median_ratio": float(np.median(ratios)),
                "min_ratio": float(np.min(ratios)),
                "max_ratio": float(np.max(ratios)),
                "shape_driven": int(np.count_nonzero(shape_component >= level_component)),
                "level_driven": int(np.count_nonzero(level_component > shape_component)),
            }
        )
    return {
        "role": ROLE_ANOMALY_EVAL,
        "note": (
            "anomaly_eval data is used for reporting only: it never enters training, "
            "threshold calibration, contamination QC statistics or the quantization "
            "representative set. The detector does not classify or name the sound; it only "
            "reports deviation from the collected normal baseline."
        ),
        "groups": results,
    }


# --------------------------------------------------------------------------------------
# Reading a project from the store
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectData:
    normal: NormalClipSet
    groups: tuple[AnomalyEvalGroup, ...]
    excluded_clip_ids: tuple[str, ...]
    normal_seconds: float


def _sample_session_id(sample: Mapping[str, Any]) -> str:
    """Recording session of one stored clip.

    ``recording_session_id`` is the contract.  Older projects predate it; one upload or one
    recording produced one source file, so ``source_name`` is the honest fallback grouping.
    Falling back to the sample id would make every clip its own "session" and silently defeat
    the entire session-independence guarantee, so that is the last resort only.
    """

    for key in ("recording_session_id", "session_id"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source = sample.get("source_name")
    if isinstance(source, str) and source.strip():
        return f"source:{source.strip()}"
    return f"clip:{sample.get('id', 'unknown')}"


def collect_project_data(
    store: ProjectStore,
    project_id: str,
    settings: Mapping[str, Any] | None = None,
) -> ProjectData:
    """Load a project's clips, split by ROLE, from the ProjectStore.

    Classes carry ``role``; a project that predates the field is read as "the first container
    is the locked normal_train one, the rest are anomaly_eval groups", which is the layout the
    UI creates.  A sample the user has explicitly marked ``excluded`` is left out of the
    normal set and reported; that is a user decision, never an automatic one.
    """

    payload = store.get_raw_project(project_id)
    resolved_settings = dict(settings or payload.get("settings") or {})
    frontend = config_from_mapping(resolved_settings)
    project_dir = store.project_dir(project_id)

    roles: list[str] = []
    class_roles: dict[str, tuple[str, str]] = {}
    for index, class_item in enumerate(payload.get("classes", [])):
        role = str(class_item.get("role") or "").strip().lower()
        if role not in (ROLE_NORMAL_TRAIN, ROLE_ANOMALY_EVAL):
            role = ROLE_NORMAL_TRAIN if index == 0 else ROLE_ANOMALY_EVAL
        roles.append(role)
        class_roles[class_item["id"]] = (role, str(class_item.get("name") or f"Group {index}"))
    try:
        anomaly_schema.validate_roles(roles)
    except anomaly_schema.AnomalySchemaError as exc:
        raise TrainingError(str(exc)) from exc

    normal: list[NormalClip] = []
    grouped: dict[str, list[AnomalyEvalClip]] = {}
    group_order: list[str] = []
    excluded: list[str] = []
    total_samples = 0
    for sample in sorted(
        payload.get("samples", {}).values(), key=lambda item: str(item.get("id", ""))
    ):
        entry = class_roles.get(sample.get("class_id"))
        if entry is None:
            continue
        role, group_name = entry
        path = project_dir.joinpath(*str(sample["relative_path"]).split("/"))
        if not path.is_file():
            continue
        _, signal = read_wav_file(path, frontend.sample_rate)
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        source_samples = int(sample.get("source_samples") or signal.size)
        clip_id = str(sample.get("id"))
        session_id = _sample_session_id(sample)
        if role == ROLE_NORMAL_TRAIN:
            total_samples += 1
            if bool(sample.get("excluded")):
                excluded.append(clip_id)
                continue
            normal.append(
                NormalClip(
                    clip_id=clip_id,
                    session_id=session_id,
                    signal=fix_length(signal, frontend.clip_samples),
                    source_samples=source_samples,
                )
            )
        else:
            if group_name not in grouped:
                grouped[group_name] = []
                group_order.append(group_name)
            grouped[group_name].append(
                AnomalyEvalClip(
                    clip_id=clip_id,
                    session_id=session_id,
                    signal=fix_length(signal, frontend.clip_samples),
                    group=group_name,
                    source_samples=source_samples,
                )
            )
    normal_set = NormalClipSet(tuple(normal))
    groups = tuple(
        AnomalyEvalGroup(name=name, clips=tuple(grouped[name])) for name in group_order
    )
    return ProjectData(
        normal=normal_set,
        groups=groups,
        excluded_clip_ids=tuple(excluded),
        normal_seconds=float(len(normal_set)) * float(frontend.clip_seconds),
    )


def abnormal_sound_representative_samples(
    store: ProjectStore,
    project_id: str,
    settings: Mapping[str, Any],
    *,
    limit: int = 4096,
) -> list[np.ndarray]:
    """Context vectors for TFLite calibration, drawn from ``normal_train`` clips ONLY.

    The representative set defines the input/output quantization scales, which are part of
    the threshold binding tuple.  Letting an ``anomaly_eval`` clip in would quietly rescale
    every stored ``T_shape``, so the set is taken from the type-checked normal set and the
    session order is preserved for reproducibility.
    """

    data = collect_project_data(store, project_id, settings)
    normal = _require_normal_set(data.normal, "abnormal_sound_representative_samples()")
    if len(normal) == 0:
        return []
    frontend = config_from_mapping(dict(settings))
    context_frames = int(_setting(settings, "context_frames"))
    n_contexts = int(_setting(settings, "n_contexts"))
    features = clip_features(normal.clips, frontend)
    vectors = context_batch(
        features, context_frames=context_frames, n_contexts=n_contexts
    ).reshape(-1, context_frames * frontend.mel_bins)
    step = max(1, int(math.ceil(vectors.shape[0] / max(1, int(limit)))))
    selected = vectors[::step][: max(1, int(limit))]
    return [np.asarray(row, dtype=np.float32) for row in selected]


# --------------------------------------------------------------------------------------
# Training entry point (same contract as train_audio_project)
# --------------------------------------------------------------------------------------


def _readiness_message(reasons: Sequence[str]) -> str:
    return (
        "Not enough normal audio to train an abnormal-sound detector yet: "
        + "; ".join(reasons)
        + "."
    )


def train_abnormal_sound_project(
    store: ProjectStore,
    project_id: str,
    options: dict[str, Any] | None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Train, calibrate and evaluate one ``abnormal_sound`` project.

    Returns ``{"report": ..., "artifacts": ...}`` exactly like ``train_audio_project`` so the
    job manager and the project store need no special case.  Training produces only the
    ``.keras`` model plus metadata; TFLite files are still generated only on Export, and the
    per-runtime ``T_shape`` for each quantized runtime is calibrated there.
    """

    options = dict(options or {})
    project = store.get_raw_project(project_id)
    kind = normalize_kind(project.get("kind"))
    if kind != "abnormal_sound":
        raise TrainingError("Project is not an abnormal sound project.")
    if not is_audio_like(kind):  # pragma: no cover - defensive, abnormal_sound is audio-like
        raise TrainingError("Abnormal sound projects must be audio-like.")

    settings = {**ABNORMAL_SOUND_DEFAULTS, **project.get("settings", {}), **options}
    try:
        validate_abnormal_sound_settings(settings)
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc
    frontend = config_from_mapping(settings)
    try:
        anomaly_schema.assert_frontend_matches(frontend.to_dict())
    except anomaly_schema.AnomalySchemaError as exc:
        raise TrainingError(str(exc)) from exc

    preset_name = str(_setting(settings, "sensitivity") or DEFAULT_SENSITIVITY)
    preset = sensitivity_preset(preset_name)
    alpha = float(preset["alpha"])
    votes_required = int(preset["votes_required"])
    top_k = int(_setting(settings, "top_k"))
    n_contexts = int(_setting(settings, "n_contexts"))
    context_frames = int(_setting(settings, "context_frames"))
    level_floor = float(_setting(settings, "level_tolerance_floor_db"))
    bottleneck = int(_setting(settings, "bottleneck_dim"))
    if top_k != TOP_K or n_contexts != N_CONTEXTS or context_frames != CONTEXT_FRAMES:
        raise TrainingError(
            f"This build implements scorer schema {anomaly_schema.SCORER_SCHEMA_VERSION}, which "
            f"pins context_frames={CONTEXT_FRAMES}, n_contexts={N_CONTEXTS} and the ABSOLUTE "
            f"integer top_k={TOP_K}; the project stores {context_frames}/{n_contexts}/{top_k}. "
            "Changing them changes what every stored threshold means."
        )

    if progress:
        progress(0.02, "Reading the normal baseline…")
    data = collect_project_data(store, project_id, settings)
    normal = _require_normal_set(data.normal, "train_abnormal_sound_project()")
    if len(normal) == 0:
        raise TrainingError(
            "The normal_train container holds no usable clips. Record normal audio first."
        )

    if progress:
        progress(0.04, "Scanning the normal baseline for contamination…")
    qc = contamination_scan(normal, settings, frontend=frontend, progress=progress)

    sessions = normal.session_ids
    split = session_split(sessions)
    train_set = normal.subset(split.train)
    calibration_set = normal.subset(split.calibration)
    audit_set = normal.subset(split.audit)

    # Gate on data volume BEFORE spending minutes on training.
    gate = anomaly_schema.assess_readiness(
        train_sessions=len(split.train),
        train_seconds=data.normal_seconds,
        held_out_sessions=len(split.held_out),
        held_out_clips=len(calibration_set) + len(audit_set),
        alpha=alpha,
        unresolved_high_review=qc.unresolved_high_review,
        train_clips=len(train_set),
    )
    if not gate.can_train:
        raise TrainingError(_readiness_message(gate.reasons))

    if progress:
        progress(0.14, f"Training on {len(train_set)} normal clips…")
    fitted = train_autoencoder(
        train_set,
        settings,
        frontend=frontend,
        validation=calibration_set if len(calibration_set) else None,
        progress=progress,
        progress_range=(0.16, 0.67),
    )
    model = fitted.model

    if progress:
        progress(0.69, "Calibrating thresholds on held-out normal sessions…")
    calibration_features = clip_features(calibration_set.clips, frontend)
    calibration_scores = shape_scores(
        model,
        calibration_features,
        top_k=top_k,
        context_frames=context_frames,
        n_contexts=n_contexts,
    )
    audit_scores = None
    audit_level_samples = None
    if len(audit_set):
        audit_features = clip_features(audit_set.clips, frontend)
        audit_scores = shape_scores(
            model,
            audit_features,
            top_k=top_k,
            context_frames=context_frames,
            n_contexts=n_contexts,
        ).tolist()
        audit_level_samples = clip_levels(audit_set.clips, frontend)
    thresholds = calibrate_thresholds(
        runtime="keras",
        train_levels=clip_levels(train_set.clips, frontend),
        calibration_shape_scores=calibration_scores.tolist(),
        calibration_levels=clip_levels(calibration_set.clips, frontend),
        alpha=alpha,
        level_floor_db=level_floor,
        audit_shape_scores=audit_scores,
        audit_levels=audit_level_samples,
    )

    # The reported confidence and the 95% bound must rest on the clips the exceedance count
    # was actually measured on. Counting exceedances on the clips that DEFINED the quantile
    # returns 0 by construction with method="higher", so quoting a bound over the whole
    # held-out side would inflate the evidence.
    readiness = anomaly_schema.assess_readiness(
        train_sessions=len(split.train),
        train_seconds=data.normal_seconds,
        held_out_sessions=len(split.held_out),
        held_out_clips=thresholds.audit_clips,
        alpha=alpha,
        unresolved_high_review=qc.unresolved_high_review,
        train_clips=len(train_set),
    )

    if progress:
        progress(0.80, "Evaluating the anomaly groups…")
    evaluation = evaluate_groups(
        data.groups,
        model=model,
        calibration=thresholds,
        settings=settings,
        frontend=frontend,
        progress=progress,
    )

    if progress:
        progress(0.90, "Saving the trained detector…")
    project_dir = store.project_dir(project_id)
    models_dir = project_dir / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)
    keras_path = models_dir / "abnormal_sound_autoencoder.keras"
    save_native_keras_model(model, keras_path)
    keras_sha = _file_sha256(keras_path)

    representative = abnormal_sound_representative_samples(store, project_id, settings)
    representative_digest = hashlib.sha256()
    representative_digest.update(f"anomaly-representative/v1/{len(representative)}".encode())
    for row in representative:
        representative_digest.update(np.ascontiguousarray(row, dtype=np.float32).tobytes())
    representative_sha = representative_digest.hexdigest()

    dataset_fp = anomaly_schema.dataset_fingerprint(normal.fingerprint_entries())
    split_fp = anomaly_schema.split_fingerprint(split.train, split.held_out)
    runtime_threshold = anomaly_schema.build_runtime_threshold(
        t_shape=thresholds.t_shape,
        artifact_sha256=keras_sha,
        keras_sha256=keras_sha,
        representative_sha256=representative_sha,
        dataset_fingerprint=dataset_fp,
        split_fingerprint=split_fp,
        # The keras runtime has no quantization noise floor to divide by; the ratio is
        # measured against the int8 artifact at Export time and reported there.
        spread_over_noise_floor=0.0,
        normal_p50=thresholds.normal_p50,
        normal_p95=thresholds.normal_p95,
    )
    try:
        anomaly_config = anomaly_schema.build_config(
            preset=preset_name,
            c_level_db=thresholds.c_level_db,
            t_level_db=thresholds.t_level_db,
            runtimes={"keras": runtime_threshold},
            train_sessions=len(split.train),
            train_seconds=data.normal_seconds,
            held_out_sessions=len(split.held_out),
            held_out_clips=thresholds.audit_clips,
            observed_exceedances=thresholds.observed_exceedances,
            unresolved_high_review=qc.unresolved_high_review,
            train_clips=len(train_set),
            bottleneck_dim=bottleneck,
        )
    except anomaly_schema.AnomalySchemaError as exc:
        raise TrainingError(str(exc)) from exc
    (models_dir / "anomaly_config.json").write_text(
        anomaly_schema.dumps(anomaly_config) + "\n", encoding="utf-8"
    )
    (models_dir / "audio_frontend.json").write_text(
        json.dumps(frontend.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (models_dir / "contamination_report.json").write_text(
        json.dumps(qc.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    from .audio_pipeline import write_audio_frontend_reference

    write_audio_frontend_reference(models_dir)

    confidence = anomaly_config.calibration.confidence
    training_report = {
        "project_kind": "abnormal_sound",
        "detector_kind": "open_set_anomaly",
        "trained_at": utc_now_iso(),
        "statement": (
            "This detector reports whether a sound deviates from the collected normal "
            "baseline. It does not classify and never names the anomaly."
        ),
        "roles": {
            "normal_train": len(normal),
            "anomaly_eval_groups": [group.name for group in data.groups],
            "excluded_by_user": list(data.excluded_clip_ids),
        },
        "settings": {
            **frontend.to_dict(),
            "context_frames": context_frames,
            "n_contexts": n_contexts,
            "top_k": top_k,
            "bottleneck_dim": bottleneck,
            "denoise_sigma": float(_setting(settings, "denoise_sigma")),
            "sensitivity": preset_name,
            "alpha": alpha,
            "votes_required": votes_required,
            "hop_seconds": float(_setting(settings, "hop_seconds")),
            "epochs_requested": fitted.epochs_requested,
            "epochs_completed": fitted.epochs_completed,
            "batch_size": _bounded_int(_setting(settings, "batch_size"), 1, 1024, 64),
            "learning_rate": _bounded_float(_setting(settings, "learning_rate"), 1e-6, 0.1, 1e-3),
            "parameter_count": fitted.parameter_count,
        },
        "dataset": {
            "normal_clips": len(normal),
            "normal_seconds": data.normal_seconds,
            "sessions": len(sessions),
            "train_sessions": list(split.train),
            "calibration_sessions": list(split.calibration),
            "audit_sessions": list(split.audit),
            "train_clips": len(train_set),
            "calibration_clips": len(calibration_set),
            "audit_clips": len(audit_set),
            "train_context_vectors": fitted.train_vectors,
            "calibration_samples": 0,
            "dataset_fingerprint": dataset_fp,
            "split_fingerprint": split_fp,
        },
        "contamination": qc.to_dict(),
        "readiness": {
            "can_train": readiness.can_train,
            "confidence": confidence,
            "fpr_upper_bound_95": readiness.fpr_upper_bound_95,
            "reasons": list(readiness.reasons),
        },
        "thresholds": {"keras": thresholds.to_dict()},
        "evaluation": evaluation,
        "history": fitted.history,
        "conversion": {
            "state": "pending",
            "note": (
                "TensorFlow Lite files are generated only when Export Model is requested; "
                "T_shape is then recalibrated for each runtime."
            ),
        },
    }
    if confidence != CONFIDENCE_CALIBRATED:
        training_report["warning"] = (
            "score-only / low confidence: " + "; ".join(readiness.reasons)
        )
    if progress:
        progress(0.96, "Writing the abnormal-sound training report…")
    (models_dir / "training_report.json").write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifacts = {
        "keras": keras_path.name,
        "training_report": "training_report.json",
        "anomaly_config": "anomaly_config.json",
        "audio_frontend": "audio_frontend.json",
        "audio_frontend_reference": "audio_frontend_reference.py",
        "contamination_report": "contamination_report.json",
    }
    if progress:
        progress(
            1.0,
            "Abnormal sound detector trained. Preview is ready; use Export Model for INT8.",
        )
    return {"report": training_report, "artifacts": artifacts}


def feature_from_wav_bytes(payload: bytes, settings: dict[str, Any]) -> np.ndarray:
    """Log-mel for one uploaded clip, matching ``audio_pipeline.feature_from_wav_bytes``."""

    frontend = config_from_mapping(settings)
    _, signal = read_wav_bytes(payload, frontend.sample_rate)
    return log_mel_spectrogram(signal, frontend)
