"""Known Sound: frozen YAMNet encoder + trainable multi-label sigmoid head.

This kind NAMES sounds, unlike ``abnormal_sound`` which only reports deviation from a
normal baseline. Each class gets an INDEPENDENT sigmoid score so two target sounds can be
reported at once; the scores do not sum to 1 and are not calibrated probabilities.

Three properties are load-bearing and must not be "tidied" away:

1. The encoder is frozen and the whole dataset is embedded exactly ONCE. The head is
   ~4K parameters, so running the 3.2M-parameter encoder per epoch would waste ~99% of
   training time for an identical result.
2. Validation splits by ``recording_session_id``, never by clip. Clips sliced from one
   20 s take are near-duplicates; letting them straddle the split is what makes the
   existing audio kind's accuracy look better than it really is.
3. Mixup sums waveforms from two DIFFERENT classes before the encoder runs. Collection is
   single-label, so this is the only place the model can learn simultaneous events.

TensorFlow is imported inside functions so that startup, and the pure-logic tests here,
stay fast.

Design document: docs/KNOWN_SOUND_zh-TW.md
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .audio_frontend import config_from_mapping, read_wav_bytes, read_wav_file
from .config import (
    KNOWN_SOUND_DEFAULT_ENCODER_DEPTH,
    KNOWN_SOUND_DEFAULTS,
    KNOWN_SOUND_VAL_SESSION_FRACTION,
    defaults_for_kind,
)
from .tflite_export import ExportError
from .training_common import (
    ProgressFn,
    TrainingError,
    history_to_json,
    make_keras_progress_callback,
    save_native_keras_model,
)
from .yamnet_model import (
    YAMNET_ASSET_VERSION,
    YAMNET_BLOCK_COUNT,
    YAMNET_EMBEDDING_DIM,
    YAMNET_MEL_BANDS,
    YAMNET_PATCH_FRAMES,
    YAMNET_SAMPLE_RATE,
    YAMNET_WEIGHTS_SHA256,
    build_yamnet_models,
    encoder_layer_name,
    encoder_output_dim,
    frontend_contract,
    load_class_map,
    waveform_to_log_mel_patches,
)

KERAS_ARTIFACT = "known_sound_yamnet.keras"
# Shares a filename with the abnormal_sound frontend artifact on purpose: it is the same
# pinned YAMNet contract, and build_model_download() already knows how to ship it.
FRONTEND_ARTIFACT = "yamnet_frontend.json"


# ======================================================================================
# Data model
# ======================================================================================


@dataclass(frozen=True)
class ClipRef:
    """One 1-second training clip and the recording session it came from."""

    path: Path
    class_index: int
    session_id: str


@dataclass(frozen=True)
class SessionSplit:
    train_clips: tuple[ClipRef, ...]
    validation_clips: tuple[ClipRef, ...]
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]


def _session_order_key(session_id: str) -> str:
    """Deterministic, content-derived ordering so the split never depends on disk order."""

    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()


def split_sessions(
    clips: Sequence[ClipRef],
    *,
    val_fraction: float = KNOWN_SOUND_VAL_SESSION_FRACTION,
) -> SessionSplit:
    """Split clips so that no recording session appears on both sides.

    The split is computed PER CLASS, so every class keeps at least one training and one
    validation session whenever it has two or more sessions. ``ceil`` is used rather than
    ``round`` for the same reason as VAL_SESSION_FRACTION: banker's rounding makes
    round(2.5) == 2, and a split that differs between implementations is not reproducible.
    """

    if not 0.0 < val_fraction < 1.0:
        raise TrainingError("val_fraction must be strictly between 0 and 1.")

    by_class: dict[int, dict[str, list[ClipRef]]] = defaultdict(lambda: defaultdict(list))
    for clip in clips:
        by_class[clip.class_index][clip.session_id].append(clip)

    train: list[ClipRef] = []
    validation: list[ClipRef] = []
    train_sessions: list[str] = []
    validation_sessions: list[str] = []

    for class_index in sorted(by_class):
        sessions = by_class[class_index]
        ordered = sorted(sessions, key=_session_order_key)
        if len(ordered) < 2:
            # A single session cannot be split. check_data_gates() rejects this before
            # training; keeping it on the training side here keeps the helper total.
            chosen: list[str] = []
        else:
            count = math.ceil(val_fraction * len(ordered))
            count = max(1, min(count, len(ordered) - 1))
            chosen = ordered[:count]
        chosen_set = set(chosen)
        for session_id in ordered:
            bucket = validation if session_id in chosen_set else train
            bucket.extend(sessions[session_id])
            (validation_sessions if session_id in chosen_set else train_sessions).append(
                session_id
            )

    return SessionSplit(
        train_clips=tuple(train),
        validation_clips=tuple(validation),
        train_sessions=tuple(train_sessions),
        validation_sessions=tuple(validation_sessions),
    )


def check_data_gates(
    clips: Sequence[ClipRef],
    class_names: Sequence[str],
    *,
    minimum_sessions: int,
    minimum_clips: int,
) -> None:
    """Raise TrainingError naming exactly which class is short of what.

    A generic "not enough data" message is useless to a student. Below two independent
    sessions per class, session-disjoint validation is impossible and any reported
    accuracy would be meaningless, so this refuses rather than warning.
    """

    if len(class_names) < 2:
        raise TrainingError(
            "A Known Sound project needs at least 2 classes. Add another class, for "
            "example one target sound plus a Background class."
        )

    sessions_per_class: dict[int, set[str]] = defaultdict(set)
    clips_per_class: dict[int, int] = defaultdict(int)
    for clip in clips:
        sessions_per_class[clip.class_index].add(clip.session_id)
        clips_per_class[clip.class_index] += 1

    problems: list[str] = []
    for index, name in enumerate(class_names):
        have_sessions = len(sessions_per_class.get(index, ()))
        have_clips = clips_per_class.get(index, 0)
        missing: list[str] = []
        if have_sessions < minimum_sessions:
            missing.append(
                f"{minimum_sessions - have_sessions} more independent recording "
                f"session(s) (has {have_sessions}, needs {minimum_sessions})"
            )
        if have_clips < minimum_clips:
            missing.append(
                f"{minimum_clips - have_clips} more clip(s) "
                f"(has {have_clips}, needs {minimum_clips})"
            )
        if missing:
            problems.append(f"'{name}' needs " + " and ".join(missing))

    if problems:
        raise TrainingError(
            "Not enough data to train a trustworthy classifier yet. "
            + "; ".join(problems)
            + ". Record each sound on separate occasions rather than one long take: "
            "20 clips sliced from a single recording are near-duplicates, not 20 examples."
        )


# ======================================================================================
# Tunable parameters (per-class thresholds, background weighting, waveform augmentation)
# ======================================================================================


#: Waveform augmentation presets. Augmentation happens on the WAVEFORM, before the frozen
#: encoder, for the same reason mixup does: YAMNet is non-linear, so perturbing embeddings
#: is not the same as perturbing the audio the microphone would actually hear. ``copies``
#: multiplies the training clip count, and every extra clip costs one encoder forward pass,
#: so "strong" is 4x the embedding time -- still seconds for a classroom-sized dataset.
WAVEFORM_AUGMENT_LEVELS: dict[str, dict[str, float] | None] = {
    "off": None,
    "light": {"copies": 1, "gain_db": 3.0, "shift_ms": 50, "snr_db": 30.0},
    "medium": {"copies": 2, "gain_db": 6.0, "shift_ms": 100, "snr_db": 20.0},
    "strong": {"copies": 3, "gain_db": 9.0, "shift_ms": 200, "snr_db": 10.0},
}


def resolve_class_thresholds(
    settings: Mapping[str, Any], classes: Sequence[Mapping[str, Any]]
) -> list[float]:
    """One detection threshold per class, in project class order.

    ``class_thresholds`` is a ``{class_id: 0..1}`` map holding only the classes the student
    actually retuned; everything else falls back to the project-wide
    ``detection_threshold``. Values are coerced defensively because settings that made a
    round trip through a form can arrive as strings, and an entry left behind by a deleted
    class must not shift the remaining classes.
    """

    fallback = _as_float(
        settings.get("detection_threshold"), KNOWN_SOUND_DEFAULTS["detection_threshold"]
    )
    table = settings.get("class_thresholds")
    if not isinstance(table, dict):
        table = {}
    return [_as_float(table.get(str(item.get("id"))), fallback) for item in classes]


def thresholds_for_labels(settings: Mapping[str, Any], label_count: int) -> list[float]:
    """The per-class thresholds a trained report hands to Preview, the runner and the ZIP.

    Training writes ``class_thresholds_by_label`` (see ``resolve_class_thresholds``). A
    report from before per-class thresholds existed has no list at all, and every class
    then uses the project-wide ``detection_threshold`` -- that is the backward-compatible
    path. A list that IS present but malformed raises instead of falling back silently:
    shipping a model whose baked-in thresholds are not the ones the student tuned is the
    exact failure this data path exists to prevent.

    The messages say "retrain and try again" rather than "retrain and export again": this
    runs on the Preview path too (``score_wav_bytes`` below), where there is no export to
    retry and an export-only instruction would leave the student looking for a button that
    is not the problem.
    """

    count = int(label_count)
    fallback = _as_float(
        settings.get("detection_threshold"), KNOWN_SOUND_DEFAULTS["detection_threshold"]
    )
    stored = settings.get("class_thresholds_by_label")
    if stored is None:
        return [fallback] * count
    if not isinstance(stored, (list, tuple)):
        raise ExportError(
            "訓練報告的 class_thresholds 門檻清單格式錯誤（應為每個類別一個 0 到 1 之間的數字），"
            "請重新訓練後再試一次。"
        )
    if len(stored) != count:
        raise ExportError(
            f"訓練報告的 class_thresholds 門檻數量（{len(stored)}）與類別數量（{count}）不符，"
            "請重新訓練後再試一次。"
        )
    values: list[float] = []
    for position, value in enumerate(stored):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        if not math.isfinite(number):
            raise ExportError(
                f"訓練報告的 class_thresholds 第 {position + 1} 個門檻不是數字（{value!r}），"
                "請重新訓練後再試一次。"
            )
        # The same exclusive 0..1 bound validate_known_sound_settings() enforces at save
        # time, checked again here because a report can also be hand-edited or imported.
        # Without it the message above quoted a range this function never checked, and a
        # threshold of 1.5 (which no score can ever reach) or -0.2 (which everything
        # exceeds) would be compiled into the runner and the firmware as if tuned.
        if not 0.0 < number < 1.0:
            raise ExportError(
                f"訓練報告的 class_thresholds 第 {position + 1} 個門檻是 {number}，"
                "必須介於 0 與 1 之間（不含），請重新訓練後再試一次。"
            )
        values.append(number)
    return values


def _as_float(value: Any, fallback: float) -> float:
    if value is None:
        return float(fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def build_waveform_augmentations(
    waves: np.ndarray,
    labels: np.ndarray,
    level: str,
    *,
    sample_rate: int = YAMNET_SAMPLE_RATE,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray]:
    """Extra training clips: random gain, circular time shift and white noise at a fixed SNR.

    These three are the distortions a different distance, a different take and a different
    room actually produce; pitch or time stretching would change what the sound IS. The
    shift is circular so the clip keeps its full length and every copy stays a valid
    1-second example. Returns ``(waves float32, labels int64)``, empty when ``level`` is
    ``off`` (or unrecognised, so a stale setting degrades to today's behaviour).
    """

    preset = WAVEFORM_AUGMENT_LEVELS.get(str(level))
    waves = np.asarray(waves, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if not preset or waves.shape[0] == 0:
        return np.zeros((0,) + waves.shape[1:], dtype=np.float32), np.zeros((0,), dtype=np.int64)

    # RandomState, not default_rng: this is the same seeded-and-reproducible contract the
    # mixup builder uses, so two runs on one dataset produce byte-identical training data.
    rng = np.random.RandomState(seed)
    out: list[np.ndarray] = []
    out_labels: list[int] = []
    max_shift = int(float(preset["shift_ms"]) * sample_rate / 1000.0)
    for _copy in range(int(preset["copies"])):
        for wave, label in zip(waves, labels):
            gain = 10.0 ** (rng.uniform(-preset["gain_db"], preset["gain_db"]) / 20.0)
            shifted = np.roll(wave * gain, int(rng.randint(-max_shift, max_shift + 1)))
            power = float(np.mean(shifted**2)) + 1e-12
            noise_power = power / (10.0 ** (float(preset["snr_db"]) / 10.0))
            noise = rng.normal(0.0, math.sqrt(noise_power), size=shifted.shape)
            noisy = shifted + noise.astype(np.float32)
            peak = float(np.max(np.abs(noisy)))
            if peak > 1.0:
                noisy = noisy / peak
            out.append(noisy.astype(np.float32))
            out_labels.append(int(label))
    return np.stack(out), np.asarray(out_labels, dtype=np.int64)


def background_sample_weights(
    targets: np.ndarray,
    background_index: int | None,
    background_weight: float,
    *,
    mixup_rows: int = 0,
) -> np.ndarray | None:
    """Per-example loss weights that down- or up-weight the background class.

    ``targets`` must be the REAL (pre-mixup) rows only, and ``mixup_rows`` the number of
    mixed examples appended after them; those trailing rows always get weight 1.0. A mixup
    row is multi-hot -- "target sound AND background" -- so weighting its whole loss by the
    background weight would also scale the target class it is there to teach, which is the
    opposite of what the knob means.

    Returns None when there is nothing to do, so the ``head.fit`` call stays byte-identical
    to the pre-tunables one for every project that does not use this knob.
    """

    weight = float(background_weight)
    if background_index is None or math.isclose(weight, 1.0):
        return None
    index = int(background_index)
    targets = np.asarray(targets, dtype=np.float32)
    if not 0 <= index < targets.shape[1]:
        return None
    weights = np.where(targets[:, index] > 0.5, weight, 1.0).astype(np.float32)
    if int(mixup_rows) > 0:
        weights = np.concatenate([weights, np.ones(int(mixup_rows), dtype=np.float32)])
    return weights


# ======================================================================================
# Mixup
# ======================================================================================


def build_mixup_examples(
    waveforms: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    count: int,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum pairs of clips from DIFFERENT classes into genuine multi-label examples.

    Collection stores one class per sample, so without this the model never sees two
    target sounds at once and the multi-label head has nothing to learn co-occurrence
    from. Mixing happens on the WAVEFORM, before the encoder, because YAMNet is
    non-linear: averaging embeddings is not the same as embedding a mixture.

    Returns ``(mixed_waveforms, multi_hot_targets)``; both are empty when mixing is
    impossible (fewer than two classes present) or disabled (``count <= 0``).
    """

    waveforms = np.asarray(waveforms, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    empty = (
        np.empty((0, waveforms.shape[1] if waveforms.ndim == 2 else 0), dtype=np.float32),
        np.empty((0, class_count), dtype=np.float32),
    )
    if count <= 0 or waveforms.ndim != 2 or waveforms.shape[0] == 0:
        return empty

    by_class: dict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        by_class[int(label)].append(position)
    present = sorted(by_class)
    if len(present) < 2:
        return empty

    rng = np.random.RandomState(seed)
    mixed = np.empty((count, waveforms.shape[1]), dtype=np.float32)
    targets = np.zeros((count, class_count), dtype=np.float32)

    for n in range(count):
        first_class, second_class = rng.choice(present, size=2, replace=False)
        first = waveforms[rng.choice(by_class[int(first_class)])]
        second = waveforms[rng.choice(by_class[int(second_class)])]
        # Random relative gain so the model does not only ever see equal-loudness mixes.
        gain = float(rng.uniform(0.5, 1.0))
        combined = first + gain * second
        peak = float(np.max(np.abs(combined)))
        if peak > 1.0:
            combined = combined / peak
        mixed[n] = combined.astype(np.float32)
        targets[n, int(first_class)] = 1.0
        targets[n, int(second_class)] = 1.0

    return mixed, targets


# ======================================================================================
# Model
# ======================================================================================


def build_known_sound_model(
    class_count: int,
    *,
    encoder_depth: int = KNOWN_SOUND_DEFAULT_ENCODER_DEPTH,
    weights_path: Path | None = None,
    head_dropout: float = 0.0,
) -> Any:
    """Frozen YAMNet encoder (optionally truncated) followed by Dense(class_count, sigmoid).

    Returned as ONE model so that Export produces a self-contained classifier: the TFLite
    artifact maps a 96x64 log-mel patch straight to per-class scores, and consumers do not
    have to reimplement a matmul. FFT stays outside the graph, matching the project rule
    that the frontend lives on the host/MCU side.

    ``encoder_depth`` keeps only the first N of YAMNet's 14 blocks. The input geometry is
    unchanged, so the frontend contract, the collected audio and the representative set all
    stay valid; only the embedding width and the model size change.

    ``head_dropout`` > 0 inserts a Dropout between the embedding and the head. Dropout has
    no weights, so the trainable parameter count stays exactly ``embedding_dim * n + n``
    -- the property test_known_sound.py pins.
    """

    import tensorflow as tf

    if int(class_count) < 2:
        raise TrainingError("A Known Sound model needs at least 2 classes.")

    full_tagger, encoder = build_yamnet_models(weights_path)
    depth = int(encoder_depth)
    if depth >= YAMNET_BLOCK_COUNT:
        features = encoder.output
    else:
        # Pool the truncated block's feature map the same way the full encoder does, so a
        # shallower model differs only in depth, not in how the embedding is formed.
        try:
            cut = full_tagger.get_layer(encoder_layer_name(depth)).output
        except ValueError as exc:
            raise TrainingError(str(exc)) from exc
        features = tf.keras.layers.GlobalAveragePooling2D(name="embedding")(cut)

    head_input = features
    if float(head_dropout) > 0.0:
        head_input = tf.keras.layers.Dropout(float(head_dropout), name="head_dropout")(features)
    scores = tf.keras.layers.Dense(
        int(class_count),
        activation="sigmoid",
        name="class_scores",
    )(head_input)
    model = tf.keras.Model(encoder.input, scores, name="known_sound_classifier")
    # Belt and braces: freezing the sub-model above is what matters, but a stray
    # layer.trainable flip elsewhere would silently start fine-tuning 3.2M parameters.
    # head_dropout is swept up by this loop and that is harmless: Dropout has no weights,
    # and only BatchNormalization treats trainable=False as "run in inference mode".
    for layer in model.layers:
        if layer.name != "class_scores":
            layer.trainable = False
    return model


def _embed_waveforms(
    encoder: Any,
    waveforms: np.ndarray,
    *,
    embedding_dim: int = YAMNET_EMBEDDING_DIM,
    batch_size: int = 32,
) -> np.ndarray:
    """One mean-pooled embedding per waveform. Runs the encoder exactly once."""

    if waveforms.shape[0] == 0:
        return np.empty((0, int(embedding_dim)), dtype=np.float32)
    patches = np.stack(
        [waveform_to_log_mel_patches(item)[0] for item in waveforms], axis=0
    ).astype(np.float32)
    values = np.asarray(encoder.predict(patches, batch_size=batch_size, verbose=0))
    if values.shape != (waveforms.shape[0], int(embedding_dim)):
        raise TrainingError(f"Unexpected embedding shape: {values.shape!r}.")
    return values.astype(np.float32)


# ======================================================================================
# Collecting clips from the project
# ======================================================================================


def _collect_clips(store: Any, project_id: str) -> tuple[list[ClipRef], list[str]]:
    payload = store.get_raw_project(project_id)
    class_names = [item["name"] for item in payload["classes"]]
    index_by_class_id = {item["id"]: index for index, item in enumerate(payload["classes"])}
    project_dir = store.project_dir(project_id)

    clips: list[ClipRef] = []
    for sample_id, entry in payload.get("samples", {}).items():
        class_id = entry.get("class_id")
        if class_id not in index_by_class_id:
            continue
        session_id = str(entry.get("recording_session_id") or f"sample-{sample_id}")
        clips.append(
            ClipRef(
                path=project_dir / "samples" / class_id / f"{sample_id}.wav",
                class_index=index_by_class_id[class_id],
                session_id=session_id,
            )
        )
    clips.sort(key=lambda item: (item.class_index, item.session_id, item.path.name))
    return clips, class_names


def _load_waveforms(clips: Sequence[ClipRef], sample_rate: int, clip_samples: int) -> np.ndarray:
    if not clips:
        return np.empty((0, clip_samples), dtype=np.float32)
    rows = np.zeros((len(clips), clip_samples), dtype=np.float32)
    for position, clip in enumerate(clips):
        _, signal = read_wav_file(clip.path, sample_rate)
        values = np.asarray(signal, dtype=np.float32).reshape(-1)
        if values.size >= clip_samples:
            rows[position] = values[:clip_samples]
        else:
            rows[position, : values.size] = values
    return rows


def _multi_hot(labels: Sequence[int], class_count: int) -> np.ndarray:
    targets = np.zeros((len(labels), class_count), dtype=np.float32)
    for position, label in enumerate(labels):
        targets[position, int(label)] = 1.0
    return targets


# ======================================================================================
# Training
# ======================================================================================


def train_known_sound_project(
    store: Any,
    project_id: str,
    options: Mapping[str, Any] | None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Fit the multi-label head on frozen YAMNet embeddings of the user's recordings."""

    import tensorflow as tf

    def report_progress(value: float, message: str) -> None:
        if progress:
            progress(value, message)

    payload = store.get_raw_project(project_id)
    settings = {**defaults_for_kind("known_sound"), **payload.get("settings", {})}
    settings.update({k: v for k, v in dict(options or {}).items() if k in settings})

    frontend = config_from_mapping(settings)
    clip_samples = frontend.clip_samples

    report_progress(0.02, "Checking the recordings…")
    clips, class_names = _collect_clips(store, project_id)
    check_data_gates(
        clips,
        class_names,
        minimum_sessions=int(settings["minimum_sessions_per_class"]),
        minimum_clips=int(settings["minimum_clips_per_class"]),
    )

    split = split_sessions(clips, val_fraction=KNOWN_SOUND_VAL_SESSION_FRACTION)
    class_count = len(class_names)

    report_progress(0.08, "Loading audio…")
    train_waves = _load_waveforms(split.train_clips, YAMNET_SAMPLE_RATE, clip_samples)
    val_waves = _load_waveforms(split.validation_clips, YAMNET_SAMPLE_RATE, clip_samples)
    train_labels = np.array([c.class_index for c in split.train_clips], dtype=np.int64)
    val_labels = np.array([c.class_index for c in split.validation_clips], dtype=np.int64)

    # Waveform augmentation from TRAIN sessions only, and before mixup so a mixed example
    # can be built from an augmented clip too. Validation stays untouched real audio.
    augment_level = str(settings.get("waveform_augment_level", "off"))
    aug_waves, aug_labels = build_waveform_augmentations(
        train_waves, train_labels, augment_level, sample_rate=YAMNET_SAMPLE_RATE
    )
    if aug_waves.shape[0]:
        train_waves = np.concatenate([train_waves, aug_waves], axis=0)
        train_labels = np.concatenate([train_labels, aug_labels], axis=0)

    train_targets = _multi_hot(train_labels, class_count)
    val_targets = _multi_hot(val_labels, class_count)
    # Everything after this many rows is a synthetic mixture, which background weighting
    # must not touch (see background_sample_weights).
    real_row_count = int(train_targets.shape[0])

    # Mixup from TRAIN sessions only. Validation must stay untouched real audio.
    mixup_count = int(round(float(settings["mixup_ratio"]) * len(train_waves)))
    mixed_waves, mixed_targets = build_mixup_examples(
        train_waves, train_labels, class_count=class_count, count=mixup_count
    )
    if mixed_waves.shape[0]:
        train_waves = np.concatenate([train_waves, mixed_waves], axis=0)
        train_targets = np.concatenate([train_targets, mixed_targets], axis=0)

    encoder_depth = int(settings["encoder_depth"])
    embedding_dim = encoder_output_dim(encoder_depth)

    head_dropout = _as_float(settings.get("head_dropout"), 0.0)

    report_progress(0.15, "Building the pretrained sound encoder…")
    model = build_known_sound_model(
        class_count, encoder_depth=encoder_depth, head_dropout=head_dropout
    )
    encoder = tf.keras.Model(model.input, model.get_layer("embedding").output)

    report_progress(0.25, f"Listening to {len(train_waves)} training clips…")
    train_embeddings = _embed_waveforms(encoder, train_waves, embedding_dim=embedding_dim)
    report_progress(0.55, f"Listening to {len(val_waves)} validation clips…")
    val_embeddings = _embed_waveforms(encoder, val_waves, embedding_dim=embedding_dim)

    # Train the head alone on the cached embeddings. This is mathematically identical to
    # training the fused model with a frozen encoder, and roughly 100x faster.
    head_input = tf.keras.Input(shape=(embedding_dim,), name="embedding_in")
    head_features = head_input
    if head_dropout > 0.0:
        # The SAME Dropout instance the exported model carries, so the cached-embedding
        # shortcut trains exactly the graph that gets saved (Dropout holds no weights, so
        # reusing the layer object costs nothing).
        head_features = model.get_layer("head_dropout")(head_input)
    head_layer = model.get_layer("class_scores")
    head = tf.keras.Model(head_input, head_layer(head_features), name="known_sound_head")
    head.compile(
        optimizer=tf.keras.optimizers.Adam(float(settings["learning_rate"])),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    epochs = int(settings["epochs"])
    callbacks = [make_keras_progress_callback(epochs, progress, start=0.60, end=0.92)]
    if bool(settings.get("early_stopping", True)) and len(val_embeddings):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )
        )

    # Background weighting. The import is local: audio_pipeline owns the ONE name matcher
    # the firmware also uses, and importing it at module scope would pull ProjectStore into
    # every process that merely wants the pure-logic helpers here.
    from .audio_pipeline import background_class_index

    bg_index = background_class_index(
        payload["classes"], str(settings.get("background_class_id", "") or "")
    )
    background_weight = _as_float(settings.get("background_weight"), 1.0)
    sample_weight = background_sample_weights(
        train_targets[:real_row_count],
        bg_index,
        background_weight,
        mixup_rows=int(train_targets.shape[0] - real_row_count),
    )
    # background_sample_weights() returns None both for "weight is 1.0, nothing to do" and
    # for "this project has no background class". Only the second case is a surprise, and
    # it is the one that costs a full retrain to discover.
    background_weight_applied = background_weight if sample_weight is not None else 1.0
    if sample_weight is None and abs(background_weight - 1.0) > 1e-9:
        report_progress(
            0.58,
            f"這個專案沒有背景類別，已略過 background_weight={background_weight} 設定。"
            "請在進階設定的「背景類別」選一個類別，或把某個類別改名為「背景音」"
            "後重新訓練。",
        )

    history = head.fit(
        train_embeddings,
        train_targets,
        validation_data=(val_embeddings, val_targets) if len(val_embeddings) else None,
        epochs=epochs,
        batch_size=int(settings["batch_size"]),
        verbose=0,
        callbacks=callbacks,
        sample_weight=sample_weight,
    )

    report_progress(0.94, "Measuring accuracy on held-out recordings…")
    threshold = float(settings["detection_threshold"])
    thresholds = resolve_class_thresholds(settings, payload["classes"])
    evaluation = _evaluate(head, val_embeddings, val_targets, class_names, thresholds)

    report_progress(0.97, "Saving the model…")
    models_dir = store.project_dir(project_id) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_native_keras_model(model, models_dir / KERAS_ARTIFACT)
    # Pin the exact frontend alongside the weights. The exported runner refuses to run if
    # this file disagrees with the contract the model was built against, so a hand-edited
    # frontend fails loudly instead of silently producing wrong scores.
    (models_dir / FRONTEND_ARTIFACT).write_text(
        json.dumps(frontend_contract(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "project_kind": "known_sound",
        "detector_kind": "multi_label_classifier",
        "encoder_backend": "yamnet_embedding",
        "statement": (
            "A frozen YAMNet encoder turns each 1-second clip into a 1024-number "
            "fingerprint; only a small head is trained on the user's labelled sounds. "
            "Every class gets an independent score, so two sounds can be reported at "
            "once. Scores are confidence values between known classes, not calibrated "
            "probabilities."
        ),
        "settings": {
            "encoder_backend": "yamnet_embedding",
            "sample_rate": YAMNET_SAMPLE_RATE,
            "clip_seconds": float(settings["clip_seconds"]),
            "hop_seconds": float(settings["hop_seconds"]),
            "epochs": epochs,
            "batch_size": int(settings["batch_size"]),
            "learning_rate": float(settings["learning_rate"]),
            "detection_threshold": threshold,
            "preview_peak_hold_seconds": float(settings["preview_peak_hold_seconds"]),
            "mixup_ratio": float(settings["mixup_ratio"]),
            "encoder_depth": encoder_depth,
            # The DICT, copied verbatim from project settings: tm_local/mcu/contract.py
            # resolves it per class id, so the ids must survive into the report. The list
            # below is the same thing already resolved into project class order, for the
            # exported runner and the ZIP metadata.
            "class_thresholds": dict(settings.get("class_thresholds") or {}),
            "class_thresholds_by_label": [float(value) for value in thresholds],
            # Requested vs applied: background_sample_weights() declines to weight anything
            # when the project has no background class, and a report that only echoed the
            # request would describe a run that never happened.
            "background_weight_requested": background_weight,
            "background_weight_applied": background_weight_applied,
            "background_class_index": bg_index,
            "head_dropout": head_dropout,
            "waveform_augment_level": augment_level,
            "early_stopping": bool(settings.get("early_stopping", True)),
            "deployment_target": str(settings.get("deployment_target", "pc")),
        },
        "encoder": {
            "frozen": True,
            "asset_version": YAMNET_ASSET_VERSION,
            "weights_sha256": YAMNET_WEIGHTS_SHA256,
            "depth": encoder_depth,
            "full_depth": YAMNET_BLOCK_COUNT,
            "truncated": encoder_depth < YAMNET_BLOCK_COUNT,
            "cut_layer": encoder_layer_name(encoder_depth)
            if encoder_depth < YAMNET_BLOCK_COUNT
            else "embedding",
            # Only the kept blocks are exported, so the full 3,217,344 count applies at
            # depth 14 only. This is the encoder actually present in the artifact.
            "parameter_count": int(
                sum(int(np.prod(w.shape)) for w in model.weights)
                - (embedding_dim * class_count + class_count)
            ),
            "embedding_dim": embedding_dim,
        },
        "head": {
            "units": class_count,
            "activation": "sigmoid",
            "loss": "binary_crossentropy",
            "trainable_parameters": embedding_dim * class_count + class_count,
        },
        "classes": list(class_names),
        "dataset": {
            "total_clips": len(clips),
            "train_clips": int(len(split.train_clips)),
            "validation_clips": int(len(split.validation_clips)),
            "mixup_clips": int(mixed_waves.shape[0]),
            "augmented_clips": int(aug_waves.shape[0]),
            "train_sessions": sorted(set(split.train_sessions)),
            "validation_sessions": sorted(set(split.validation_sessions)),
            "split_rule": "session-disjoint; clips from one recording never cross the split",
            "clips_per_class": {
                name: int(sum(1 for c in clips if c.class_index == index))
                for index, name in enumerate(class_names)
            },
            "sessions_per_class": {
                name: sorted({c.session_id for c in clips if c.class_index == index})
                for index, name in enumerate(class_names)
            },
        },
        "evaluation": evaluation,
        "frontend": frontend_contract(),
        "history": history_to_json(history),
        "conversion": {
            "state": "pending",
            "note": "TFLite conversion runs only on Export.",
        },
    }

    report_progress(1.0, "Model trained.")
    return {
        "artifacts": {"keras": KERAS_ARTIFACT, "yamnet_frontend": FRONTEND_ARTIFACT},
        "report": report,
    }


def _evaluate(
    head: Any,
    embeddings: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Per-class precision/recall on the session-disjoint validation set.

    ``thresholds`` is one value per class, in class order: a rare alarm sound and a common
    background hum do not deserve the same decision point, and reporting precision at a
    threshold the student is not actually using would be a lie.
    """

    if not len(embeddings):
        return {
            "available": False,
            "note": (
                "No held-out sessions were available, so no honest accuracy can be "
                "reported. Record each class on at least two separate occasions."
            ),
            "per_class": {},
        }

    thresholds = [float(value) for value in thresholds]
    scores = np.asarray(head.predict(embeddings, verbose=0), dtype=np.float64)
    predicted = scores >= np.asarray(thresholds, dtype=np.float64)[None, :]
    actual = np.asarray(targets, dtype=np.float64) >= 0.5

    per_class: dict[str, Any] = {}
    for index, name in enumerate(class_names):
        true_positive = int(np.sum(predicted[:, index] & actual[:, index]))
        false_positive = int(np.sum(predicted[:, index] & ~actual[:, index]))
        false_negative = int(np.sum(~predicted[:, index] & actual[:, index]))
        support = int(np.sum(actual[:, index]))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        per_class[name] = {
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": float(precision) if (true_positive + false_positive) else None,
            "recall": float(recall) if support else None,
        }

    return {
        "available": True,
        # "threshold" (singular) stays for readers written before per-class thresholds.
        "threshold": float(thresholds[0]) if thresholds else 0.0,
        "thresholds": thresholds,
        "validation_clips": int(len(embeddings)),
        "per_class": per_class,
        "note": (
            "Measured on recording sessions the model never trained on. It reflects this "
            "microphone, this room and this way of producing the sounds only."
        ),
    }


# ======================================================================================
# Inference
# ======================================================================================


def predict_scores(model: Any, waveform: np.ndarray) -> np.ndarray:
    """Per-class independent scores for one waveform."""

    patch = waveform_to_log_mel_patches(np.asarray(waveform, dtype=np.float32))[:1]
    return np.asarray(model.predict(patch, verbose=0), dtype=np.float32)[0]


def score_wav_bytes(
    store: Any,
    project_id: str,
    payload_bytes: bytes,
    runtime_name: str = "keras",
) -> dict[str, Any]:
    """Score a WAV against the trained model, returning the multi-label response body."""

    import tensorflow as tf

    payload = store.get_raw_project(project_id)
    training = payload.get("training") or {}
    if training.get("state") != "trained":
        raise TrainingError("Train the model first.")
    report = training.get("report") or {}
    class_names = [item["name"] for item in payload["classes"]]
    report_settings = report.get("settings") or {}
    threshold = _as_float(
        report_settings.get("detection_threshold"), KNOWN_SOUND_DEFAULTS["detection_threshold"]
    )
    # Preview decides with exactly the thresholds the exported model will use.
    thresholds = thresholds_for_labels(report_settings, len(class_names))

    artifact = (training.get("artifacts") or {}).get("keras")
    if not artifact:
        raise TrainingError("The trained model file is unavailable. Train again.")
    model = tf.keras.models.load_model(
        store.project_dir(project_id) / "models" / artifact, compile=False
    )

    _, signal = read_wav_bytes(payload_bytes, YAMNET_SAMPLE_RATE)
    scores = predict_scores(model, signal)
    if scores.shape[0] != len(class_names):
        raise TrainingError("Model output count does not match project classes.")

    predictions = [
        {
            "class_name": name,
            "score": float(min(1.0, max(0.0, value))),
            "threshold": float(limit),
            "detected": bool(value >= limit),
        }
        for name, value, limit in zip(class_names, scores, thresholds)
    ]
    predictions.sort(key=lambda item: item["score"], reverse=True)
    detected = [item["class_name"] for item in predictions if item["detected"]]
    return {
        "runtime": runtime_name,
        "multi_label": True,
        "threshold": threshold,
        "thresholds": thresholds,
        "predictions": predictions,
        "detected": detected,
        "top_class": predictions[0]["class_name"] if predictions else None,
        "note": (
            "Scores are independent confidence values and do not add up to 100%."
        ),
    }


# ======================================================================================
# Recording sanity check
# ======================================================================================


#: Cached official tagger for the sanity check. Rebuilding it means re-reading a 15 MB
#: HDF5 file, and the check runs after every single recording, so it is cached for the
#: life of the process. Training never uses this cache; it builds its own fresh model.
_SANITY_TAGGER: list[Any] = []


def _sanity_tagger() -> Any:
    if not _SANITY_TAGGER:
        full_tagger, _encoder = build_yamnet_models()
        _SANITY_TAGGER.append(full_tagger)
    return _SANITY_TAGGER[0]


def describe_waveform(waveform: np.ndarray, *, top_k: int = 3) -> list[dict[str, Any]]:
    """Name what the official 521-class AudioSet tagger hears in this audio.

    Purely informational. It never touches training, thresholds, the representative set
    or any stored metadata. Its job is to catch "you recorded something other than what
    you meant to" while the user is still standing at the microphone -- the failure that
    a deviation-only detector cannot possibly surface.
    """

    # Load the class map FIRST: if it is missing we must raise before paying for the model.
    names = load_class_map()
    full_tagger = _sanity_tagger()
    patches = waveform_to_log_mel_patches(np.asarray(waveform, dtype=np.float32))
    scores = np.asarray(full_tagger.predict(patches, verbose=0), dtype=np.float64)
    mean_scores = scores.mean(axis=0)
    order = np.argsort(mean_scores)[::-1][: max(1, int(top_k))]
    return [
        {"index": int(index), "name": names[int(index)], "score": float(mean_scores[index])}
        for index in order
    ]


# ======================================================================================
# Export
# ======================================================================================


def export_inference_saved_model(model: Any, output_dir: Path) -> Path:
    """Write a SavedModel with an explicit serving signature.

    TFLiteConverter.from_keras_model is banned project-wide because it leaves the input
    signature implicit; conversion must go through a named serving_default.
    """

    import tensorflow as tf

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                [1, YAMNET_PATCH_FRAMES, YAMNET_MEL_BANDS],
                tf.float32,
                name="log_mel_patch",
            )
        ]
    )
    def serving(log_mel_patch):  # noqa: ANN001
        return {"class_scores": model(log_mel_patch, training=False)}

    tf.saved_model.save(model, str(output), signatures={"serving_default": serving})
    return output


def representative_patches(
    store: Any,
    project_id: str,
    *,
    maximum: int = 128,
) -> list[np.ndarray]:
    """Log-mel patches for post-training quantisation, from TRAIN sessions only.

    Validation audio must not influence the quantisation ranges, or the held-out set stops
    being held out.
    """

    payload = store.get_raw_project(project_id)
    settings = {**defaults_for_kind("known_sound"), **payload.get("settings", {})}
    clip_samples = config_from_mapping(settings).clip_samples

    clips, _names = _collect_clips(store, project_id)
    split = split_sessions(clips, val_fraction=KNOWN_SOUND_VAL_SESSION_FRACTION)
    selected = list(split.train_clips)[: max(1, int(maximum))]
    waves = _load_waveforms(selected, YAMNET_SAMPLE_RATE, clip_samples)
    # One (96, 64) patch per clip, with NO batch dimension: tflite_export's representative
    # generator adds that itself.
    return [
        np.asarray(waveform_to_log_mel_patches(item)[0], dtype=np.float32) for item in waves
    ]


def write_known_sound_runner(
    output_dir: Path,
    model_filename: str,
    class_names: Sequence[str],
    thresholds: Sequence[float],
) -> Path:
    """Write a PC reference runner that scores one WAV against the exported classifier.

    It reuses the exported yamnet_frontend_reference module so the runner and the trainer
    provably share one frontend implementation rather than two that merely look alike.

    ``thresholds`` is one value per class, in labels.txt order; the runner bakes the list
    in so the exported model decides exactly as Preview did.
    """

    destination = Path(output_dir) / "run_model.py"
    threshold_values = [float(value) for value in thresholds]
    expected_frontend_json = json.dumps(
        frontend_contract(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    source = f'''"""Run an exported Known Sound classifier on one WAV file.

Every class has an INDEPENDENT score. They do not add up to 100%, and they are
confidence values between the classes this model was taught -- not calibrated
probabilities. A sound the model was never shown will still be scored against
whatever it does know.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from yamnet_frontend_reference import fix_length, patches, read_wav

MODEL_FILENAME = {model_filename!r}
CLASS_NAMES = {list(class_names)!r}
THRESHOLDS = {threshold_values!r}
EXPECTED_FRONTEND = json.loads({expected_frontend_json!r})


def load_frontend_contract(folder):
    path = folder / "yamnet_frontend.json"
    if not path.is_file():
        raise SystemExit("Missing yamnet_frontend.json; re-export the model.")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != EXPECTED_FRONTEND:
        raise SystemExit(
            "yamnet_frontend.json does not match the frontend this model was built "
            "with. Re-export the model rather than editing the file."
        )
    return actual


def score(folder, wav_path):
    load_frontend_contract(folder)
    model_path = folder / MODEL_FILENAME
    if not model_path.is_file():
        raise SystemExit("Missing model file: " + str(model_path))

    waveform = fix_length(read_wav(wav_path))
    batch = patches(waveform)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    rows = []
    for patch in batch:
        value = patch[np.newaxis, ...].astype(np.float32)
        scale, zero_point = input_detail["quantization"]
        if input_detail["dtype"] in (np.int8, np.uint8) and scale:
            info = np.iinfo(input_detail["dtype"])
            value = np.clip(
                np.round(value / scale + zero_point), info.min, info.max
            ).astype(input_detail["dtype"])
        interpreter.set_tensor(input_detail["index"], value)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0]
        out_scale, out_zero = output_detail["quantization"]
        if output_detail["dtype"] in (np.int8, np.uint8) and out_scale:
            raw = (raw.astype(np.float32) - out_zero) * out_scale
        rows.append(np.asarray(raw, dtype=np.float32))

    # One row per 0.96 s patch. Report the strongest evidence for each class across the
    # file: a gunshot lasting 150 ms would be averaged away by a mean.
    return np.max(np.stack(rows, axis=0), axis=0)


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_model.py path/to/audio.wav")
        return 2
    folder = Path(__file__).resolve().parent
    scores = score(folder, Path(sys.argv[1]))
    if len(scores) != len(CLASS_NAMES):
        raise SystemExit("Model output count does not match labels.txt.")
    if len(THRESHOLDS) != len(CLASS_NAMES):
        raise SystemExit("Threshold count does not match labels.txt.")
    order = np.argsort(scores)[::-1]
    print("Scores (independent; they do not add up to 100%):")
    for index in order:
        mark = "  <= detected" if scores[index] >= THRESHOLDS[index] else ""
        print(
            "  {{:<28s}} {{:6.1%}}  (threshold {{:.0%}}){{}}".format(
                CLASS_NAMES[index], scores[index], THRESHOLDS[index], mark
            )
        )
    detected = [CLASS_NAMES[i] for i in order if scores[i] >= THRESHOLDS[i]]
    print()
    print("Detected:", ", ".join(detected) if detected else "(nothing above threshold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    destination.write_text(source, encoding="utf-8")
    return destination
