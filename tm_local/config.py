from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WEB_ROOT = PROJECT_ROOT / "web"
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
WORKSPACE_ROOT = Path(os.environ.get("TM_LOCAL_WORKSPACE", DEFAULT_WORKSPACE_ROOT)).resolve()
PROJECTS_ROOT = WORKSPACE_ROOT / "projects"
TEMP_ROOT = WORKSPACE_ROOT / "tmp"
LOG_ROOT = PROJECT_ROOT / "logs"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"

TOOL_NAME = "Teachable Machine Local Studio"
TOOL_VERSION = "2.1.0"

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_FILES_PER_REQUEST = 300
MAX_CLASSES = 20
MAX_CLASS_NAME_LENGTH = 80
MAX_PROJECT_NAME_LENGTH = 120
MAX_PROJECT_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024

CLASS_COLORS = [
    "#ff6d00",
    "#d83b73",
    "#6b4eff",
    "#1967d2",
    "#00a67e",
    "#a142f4",
    "#e37400",
    "#0b8043",
    "#c5221f",
    "#5f6368",
]

IMAGE_DEFAULTS = {
    "image_size": 224,
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 0.001,
    "validation_split": 0.20,
    "backbone": "mobilenet_v2",
    "mobilenet_alpha": 0.35,
    "early_stopping": True,
    "minimum_samples_per_class": 5,
    # Tunable-parameters plan (2026-09-12), defaults equal to current behaviour.
    "augmentation_level": "medium",
    "dropout": 0.2,
    "fine_tune_blocks": 0,
    "deployment_target": "pc",
}

AUDIO_DEFAULTS = {
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "window_ms": 25.0,
    "hop_ms": 10.0,
    "fft_size": 512,
    "mel_bins": 40,
    "fmin": 20.0,
    "fmax": 8000.0,
    "db_floor": -80.0,
    "epochs": 40,
    "batch_size": 16,
    "learning_rate": 0.001,
    "validation_split": 0.20,
    "early_stopping": True,
    "minimum_samples_per_class": 8,
    # Tunable-parameters plan (2026-09-12). session_disjoint_validation=True is the one
    # default that changes existing behaviour (spec Sec.8.1); every other new key defaults
    # to what training already did.
    "augmentation_level": "medium",
    "spec_augment": False,
    "session_disjoint_validation": True,
    "background_weight": 1.0,
    "background_class_id": "",
    "dropout": 0.2,
    "detection_threshold": 0.5,
    "deployment_target": "pc",
}

# --------------------------------------------------------------------------------------
# Abnormal Sound (open-set anomalous sound detection, kind="abnormal_sound")
#
# Trains on NORMAL audio only and scores how far a window deviates from the collected
# normal baseline. It is NOT a classifier and must never name the anomaly.
#
# Every number below was measured, not guessed -- do not "tidy" them without re-running
# the experiments that produced them. See docs/ABNORMAL_SOUND_YAMNET_zh-TW.md for the
# data contract, frontend, scorer and the limits these values encode.
# --------------------------------------------------------------------------------------

# Locked model geometry. Dense denoising autoencoder
#   200 -> 128 -> 64 -> 8 -> 64 -> 128 -> 200
# hidden layers Dense(units, activation="relu") with NO BatchNormalization, output layer
# Dense(200, activation="sigmoid"). 69,200 parameters.
# The sigmoid output is load-bearing, not cosmetic: it pins the TFLite int8 output
# quantisation to a fixed scale of exactly 1/256 = 0.00390625, which drops the measured
# quantisation noise floor from 2.002e-04 (linear output) to 2.393e-06 and lifts the score
# margin from 30.5x to 81.9x. Do not switch to a linear output and do not add BatchNorm.
ABNORMAL_SOUND_ENCODER_UNITS = (128, 64)  # decoder mirrors this, then sigmoid Dense(200)
ABNORMAL_SOUND_INPUT_DIM = 200  # context_frames * mel_bins

# T_shape is calibrated separately for every runtime. Measured threshold drift from keras
# to int8 was +70% on the rejected architecture, so this is a hard contract even though the
# accepted sigmoid architecture barely drifts. "keras" is the trained .keras model; the rest
# are the TFLite variants produced by export_service.
ANOMALY_THRESHOLD_RUNTIMES = ("keras", "float32", "dynamic", "int8", "uint8")

# Roles. Exactly one locked "normal_train" container plus 0..MAX_ANOMALY_EVAL_GROUPS
# user-nameable "anomaly_eval" groups. anomaly_eval data is used ONLY for per-group
# evaluation reporting: never for training, threshold calibration, contamination QC
# statistics, or the quantisation representative set.
NORMAL_TRAIN_ROLE = "normal_train"
ANOMALY_EVAL_ROLE = "anomaly_eval"
ABNORMAL_SOUND_ROLES = (NORMAL_TRAIN_ROLE, ANOMALY_EVAL_ROLE)
MAX_ANOMALY_EVAL_GROUPS = MAX_CLASSES - 1

# Train/calibration split. Clips from one recording_session_id NEVER cross the boundary.
# val_sessions = ceil(VAL_SESSION_FRACTION * n_sessions), sessions ordered deterministically
# by sha256(session_id). ceil, never round: round() is banker's rounding, so round(2.5)=2
# while ceil(2.5)=3, and a split that differs between implementations is not reproducible.
# Deliberately a module constant and not a project setting, so the split rule cannot drift
# per project and silently change a calibrated threshold.
VAL_SESSION_FRACTION = 0.25

# Per-branch empirical quantile, taken at 1 - alpha/2 with numpy method="higher".
THRESHOLD_QUANTILE_METHOD = "higher"

ABNORMAL_SOUND_DEFAULTS = {
    # Backend is explicit so an abnormal project can never fall through to the audio
    # classifier. This product workflow is intentionally YAMNet-only; the research Dense
    # AE implementation remains in the source tree as a compact MCU comparison baseline,
    # but is not a partially wired user-selectable backend.
    "detector_backend": "yamnet_embedding",
    # Audio frontend contract. Byte-identical to the first nine AUDIO_DEFAULTS entries:
    # audio_frontend.log_mel_spectrogram() is deliberately UNCHANGED and still yields
    # (98, 40, 1) per 1 s clip. These keys feed audio_frontend.config_from_mapping().
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "window_ms": 25.0,
    "hop_ms": 10.0,
    "fft_size": 512,
    "mel_bins": 40,
    "fmin": 20.0,
    "fmax": 8000.0,
    "db_floor": -80.0,
    # Shape scorer. 5 consecutive frames are stacked into 200-dim context vectors, giving
    # 98 - 5 + 1 = 94 contexts per clip. S_shape is the arithmetic mean of the LARGEST
    # top_k per-context reconstruction MSEs.
    # top_k and n_contexts are stored as ABSOLUTE INTEGERS and never as a fraction:
    # round(0.10 * 94) = 9 != 10, and letting each language recompute that rounding
    # guarantees PC/MCU divergence.
    "context_frames": 5,
    "n_contexts": 94,
    "top_k": 10,
    # Bottleneck width. Measured equivalent to 16 (margin 78.9x vs 81.9x) while being
    # 1,032 parameters smaller, so 8 is the MVP default. Kept as a named setting -- it is
    # the knob to reach for when the export report warns that the score spread has sunk
    # towards the quantisation noise floor -- but it is NOT exposed in the teaching UI.
    "bottleneck_dim": 8,
    # Autoencoder training. See the notes in the accompanying handover for why these three
    # differ from AUDIO_DEFAULTS: the training items here are 200-dim context vectors
    # (94 per clip), not whole spectrograms, so a clip contributes ~94 rows rather than 1.
    "epochs": 60,
    "batch_size": 64,
    "learning_rate": 0.001,
    # Denoising: train input = clip(x + N(0, denoise_sigma), 0, 1), target = the clean x.
    "denoise_sigma": 0.01,
    # YAMNet one-class scorer. The 3.7M pretrained encoder remains frozen; Train fits these
    # normal-distribution statistics and the held-out threshold from the user's recordings.
    "yamnet_embedding_dim": 1024,
    "yamnet_scorer": "robust_diagonal",
    "yamnet_scale_shrinkage": 0.50,
    "yamnet_scale_floor": 0.0001,
    # Decision policy. See SENSITIVITY_PRESETS for the alpha / vote values.
    "sensitivity": "balanced",
    "hop_seconds": 0.5,
    # Level branch tolerance floor, in dB. T_level = max(empirical quantile, this floor).
    # Without a floor a very stable room drives T_level towards zero and every ordinary
    # loudness wobble alarms.
    "level_tolerance_floor_db": 3.0,
    # Data readiness gates. Below "train" the project refuses to train and says why;
    # between "train" and "calibrated" it is score-only / low confidence and must say so.
    "minimum_train_sessions": 3,
    "minimum_train_seconds": 60,
    "minimum_calibration_sessions": 6,
    "minimum_calibration_seconds": 120,
    "minimum_heldout_sessions": 2,
    "minimum_heldout_clips": 40,
}

# --------------------------------------------------------------------------------------
# Known Sound (closed-set multi-label classifier, kind="known_sound")
#
# Frozen YAMNet 1024-D encoder + a trainable Dense(n_classes, sigmoid) head. Unlike
# abnormal_sound this DOES name the sound, and unlike audio it uses the official YAMNet
# frontend (96 x 64 log-mel patch) rather than the 98 x 40 relative-dB spectrogram.
#
# Multi-label on purpose: two target sounds can occur at once (a dog barking while glass
# breaks), so every class gets an INDEPENDENT sigmoid score. The scores do NOT sum to 1
# and must never be presented as probabilities or as a pie chart.
#
# Design document: docs/KNOWN_SOUND_zh-TW.md
# --------------------------------------------------------------------------------------

# The head is the only trainable part: embedding_dim * n_classes + n_classes parameters.
# The encoder stays frozen because a few hundred clips cannot fine-tune 3.2M parameters
# without destroying the pretrained representation.
KNOWN_SOUND_EMBEDDING_DIM = 1024  # width at full depth

# How many of YAMNet's 14 blocks to keep. 14 is the full official encoder.
#
# This is the ONLY knob that meaningfully changes the exported model's size, because
# YAMNet is MobileNetV1-shaped and its parameters are heavily back-loaded: block 14 alone
# is ~33% of the encoder, blocks 13-14 together ~49%. Measured on a NuMaker-GestureAI-M55M1
# target (Vela 5.1.0, ethos-u55-256, Shared_Sram, Ethos_U55_High_End_Embedded), post-Vela
# flash for a 3-class head:
#
#   depth 14  2,974,368 B      depth 11  1,308,864 B      depth 8    537,600 B
#   depth 13  2,066,928 B      depth 10  1,051,952 B      depth 7    279,792 B
#   depth 12  1,563,648 B      depth  9    795,680 B      depth 6    144,912 B
#
# Tensor-arena SRAM is ~151 KB at EVERY depth -- the peak activation is in the early
# high-resolution blocks that all depths share -- so depth trades flash, not RAM.
KNOWN_SOUND_DEFAULT_ENCODER_DEPTH = 14
KNOWN_SOUND_MIN_ENCODER_DEPTH = 2
KNOWN_SOUND_MAX_ENCODER_DEPTH = 14

# Validation splits by recording_session_id, never by clip. Clips sliced out of one 20 s
# take are near-duplicates; letting them straddle the boundary is what makes the existing
# audio kind's accuracy look better than it is. Same ceil-not-round rule as
# VAL_SESSION_FRACTION above, and deliberately a module constant so the split rule cannot
# drift per project.
KNOWN_SOUND_VAL_SESSION_FRACTION = 0.25

# Decision threshold for "detected". Per-class overrides live in project settings.
KNOWN_SOUND_DEFAULT_THRESHOLD = 0.5

KNOWN_SOUND_DEFAULTS = {
    # Frozen pretrained encoder. Named explicitly so a known_sound project can never fall
    # through to the from-scratch audio CNN.
    "encoder_backend": "yamnet_embedding",
    # Collection-side audio frontend. Byte-identical to the first nine AUDIO_DEFAULTS
    # entries, for exactly the same reason abnormal_sound copies them: ProjectStore's
    # add_audio_bytes() drives clip slicing and the spectrogram THUMBNAILS through
    # audio_frontend.config_from_mapping(). Without these keys, recording would fail.
    #
    # These are the display/slicing frontend and are NOT what the model eats. The model
    # frontend is the official YAMNet contract (64 mel bins over 125-7500 Hz,
    # log(mel + 0.001), 96 x 64 patch) implemented in tm_local/yamnet_model.py.
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "window_ms": 25.0,
    "hop_ms": 10.0,
    "fft_size": 512,
    "mel_bins": 40,
    "fmin": 20.0,
    "fmax": 8000.0,
    "db_floor": -80.0,
    # Inference window geometry, matching the collected clip length.
    "hop_seconds": 0.5,
    # How many YAMNet blocks to keep. See the note above; 14 is the full encoder.
    # The embedding width is DERIVED from this, so it is not a separate setting -- the
    # trained value is recorded in the training report under encoder.embedding_dim.
    "encoder_depth": KNOWN_SOUND_DEFAULT_ENCODER_DEPTH,
    # Head training. The head is ~4K parameters on cached embeddings, so a high epoch
    # count still finishes in seconds. The encoder forward pass runs exactly once.
    "epochs": 120,
    "batch_size": 32,
    "learning_rate": 0.001,
    "early_stopping": True,
    # Multi-label decision policy.
    "detection_threshold": KNOWN_SOUND_DEFAULT_THRESHOLD,
    # Preview peak-hold, in seconds. A gunshot lasts 100-200 ms; without a hold the bar
    # spikes and vanishes between frames. This replaces abnormal_sound's 5-window vote,
    # which would actively SUPPRESS a transient event.
    "preview_peak_hold_seconds": 1.5,
    # Mixup augmentation. Two clips from DIFFERENT classes are summed so the model sees
    # genuine simultaneous events, which single-label collection cannot provide.
    # Expressed as a multiple of the training clip count; 0 disables it.
    "mixup_ratio": 0.5,
    # Data readiness gates. Fewer than 2 independent sessions per class makes
    # session-disjoint validation impossible, so the reported accuracy would be
    # meaningless. Train refuses and names exactly what is missing, per class.
    "minimum_sessions_per_class": 2,
    "minimum_clips_per_class": 20,
    # Tunable-parameters plan (2026-09-12), defaults equal to current behaviour.
    "class_thresholds": {},
    "background_weight": 1.0,
    "background_class_id": "",
    "head_dropout": 0.0,
    "waveform_augment_level": "off",
    "deployment_target": "pc",
}

# Sensitivity presets. votes_required out of window_count consecutive 0.5 s-hop windows.
# Sensitive is 2-of-5 on purpose: with a 1 s window and a 0.5 s hop a single point-like
# impact lands in exactly 2 windows, so 3-of-5 can never fire on one knock.
# Fewer than window_count windows since start => UNCERTAIN (warming up), never an alarm.
SENSITIVITY_PRESETS = {
    "sensitive": {"alpha": 0.10, "votes_required": 2, "window_count": 5},
    "balanced": {"alpha": 0.05, "votes_required": 3, "window_count": 5},
    "low_false_alarm": {"alpha": 0.02, "votes_required": 3, "window_count": 5},
}

SENSITIVITY_PRESET_LABELS = {
    "sensitive": "Sensitive",
    "balanced": "Balanced",
    "low_false_alarm": "Low false alarm",
}

DEFAULT_SENSITIVITY = "balanced"

# --------------------------------------------------------------------------------------
# Project kind registry
#
# Dispatch used to be an implicit two-way branch (`if kind == "image" ... else audio`).
# Route new code through these instead so a fourth kind is one entry, not a grep.
# --------------------------------------------------------------------------------------

PROJECT_KINDS = ("image", "audio", "abnormal_sound", "known_sound")

# Kinds whose samples are audio clips and whose settings drive audio_frontend.
# Use this for "does this project record/upload audio", NOT for "is this a classifier":
# abnormal_sound is audio-like but is a one-class detector, not a classifier.
AUDIO_LIKE_KINDS = frozenset({"audio", "abnormal_sound", "known_sound"})

# Kinds that emit one score per project class, i.e. whose model output length must equal
# the class count. This is the contract app._prediction_response() enforces. It is NOT the
# same question as is_audio_like(): abnormal_sound is audio-like but emits a single scalar.
CLASSIFIER_KINDS = frozenset({"image", "audio", "known_sound"})

# Classifier kinds whose per-class scores are INDEPENDENT sigmoids rather than a softmax
# distribution. Scores from these kinds do not sum to 1, so the UI must not show a total.
MULTI_LABEL_KINDS = frozenset({"known_sound"})

DEFAULTS_BY_KIND = {
    "image": IMAGE_DEFAULTS,
    "audio": AUDIO_DEFAULTS,
    "abnormal_sound": ABNORMAL_SOUND_DEFAULTS,
    "known_sound": KNOWN_SOUND_DEFAULTS,
}

PROJECT_KIND_LABELS = {
    "image": "Image Project",
    "audio": "Audio Project",
    "abnormal_sound": "Abnormal Sound Project",
    "known_sound": "Known Sound Project",
}

# MCU deployment registry. Values are the firmware application ids used by tm_local.mcu;
# None means the kind stays desktop-only (abnormal_sound has no device runtime).
MCU_APPLICATION_BY_KIND = {
    "image": "imgclass",
    "audio": "kws",
    "known_sound": "known_sound",
    "abnormal_sound": None,
}
MCU_DEPLOYABLE_KINDS = frozenset(k for k, v in MCU_APPLICATION_BY_KIND.items() if v)
DEPLOYMENT_TARGETS = ("pc", "NuMaker-M55M1", "NuGestureAI-M55M1", "NuMaker-VoiceAI-M55M1")

# --------------------------------------------------------------------------------------
# Tunable-parameters plan (2026-09-12): student-facing training knobs plus the board
# lock constants a firmware build depends on. This module is the single source for the
# lock constants -- tm_local/mcu/contract.py imports them from here (keeping its old
# names as aliases) rather than defining its own copies.
# --------------------------------------------------------------------------------------

AUGMENTATION_LEVELS = ("off", "light", "medium", "strong")

# Image sizes the on-device camera pipeline supports. Enforced only when
# deployment_target != "pc"; a pc-only project may still pick any size in [96, 320].
MCU_IMAGE_SIZES = (96, 128, 160, 192, 224)

# The audio front-end geometry a KWS firmware build is compiled against. Enforced
# byte-for-byte (within floating point tolerance) whenever deployment_target != "pc".
MCU_AUDIO_FRONTEND_LOCK = {
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "window_ms": 25.0,
    "hop_ms": 10.0,
    "fft_size": 512,
    "fmin": 20.0,
    "fmax": 8000.0,
    "db_floor": -80.0,
}

# The subset of MCU_AUDIO_FRONTEND_LOCK the Advanced panel actually writes. The other
# five (clip_seconds / window_ms / hop_ms / fmin / db_floor) have no control in ANY kind's
# panel, so a project that violates them can only have come from an archive import or a
# direct API call -- and telling that student to 「改回預設值」 would point at a field that
# does not exist on their screen. validate_audio_settings() picks the advice accordingly.
MCU_AUDIO_FRONTEND_PANEL_KEYS = ("sample_rate", "fft_size", "fmax")

# mel_bins is the one audio front-end knob that is genuinely configurable: the training
# CNN and the exported mel-filterbank tables only ever exist for these two widths, so the
# restriction applies to every deployment_target, not only a board build.
MCU_MEL_BINS = (40, 64)

# Firmware known_sound flash budget caps, keyed by board label. Duplicates
# mcu_toolkit/boards.json's "known_sound_max_depth": this dict is the UI/validation
# source (fast, TensorFlow-free, importable from config.py), boards.json stays the
# firmware registry consumed by tm_local.mcu.boards. tests/test_tunables_config.py
# asserts the two agree.
#
# The NuMaker-M55M1 (X board) and NuGestureAI numbers come from MEASURED Vela output, not from
# guesswork (2026-09-12 fix round): a 3-class known_sound model exported through the real
# strict-INT8 path and compiled by Vela costs, in flash, d11 1,308,784 B / d12 1,563,552 B /
# d13 2,066,832 B / d14 2,974,272 B. deploy_service.py gates every board build with a flat
# FIRMWARE_CODE_BASELINE_BYTES = 256 KiB (262,144 B) firmware-code allowance on top of the
# Vela flash size -- not a per-board margin -- so all three boards share the same 2 MiB
# internal-flash arithmetic and d13 cannot fit on any of them:
#   d12: 1,563,552 + 262,144 = 1,825,696 <= 2,097,152  OK
#   d13: 2,066,832 + 262,144 = 2,328,976 >  2,097,152  no
#
# SUPERSEDED (2026-09-13, see the NuMaker-VoiceAI-M55M1 paragraph below): this comment used to
# say NuGestureAI stays at the spec's 11 because its UVC/CDC firmware is the larger one, so it
# has less than the flat 256 KiB of headroom the arithmetic above assumes. The VoiceAI
# measurement below shows that claim does not hold -- GestureAI's own measured firmware
# (157,712 B) leaves 1,563,552 + 157,712 = 1,721,264 B, still under the 2,097,152 B budget, so
# d12 would fit by this same arithmetic using GestureAI's real firmware size. GestureAI's cap
# stays 11 regardless (unchanged, out of scope for this task); it is just not derived from this
# headroom argument -- see below for what it actually is (a deliberate judgement, not this
# arithmetic's result).
#
# So: the X board's and NuGestureAI's caps are this flash-budget arithmetic over Vela's reported
# sizes; NuMaker-VoiceAI-M55M1's cap (added 2026-09-13) is NOT -- see the paragraph below for
# what it actually is. NOTHING on this branch has been flashed to a board: no measurement here,
# for any of the three, reflects a device that actually ran.
#
# NuMaker-VoiceAI-M55M1（2026-09-13 實測，Task 9）：對這塊板實際建出 known_sound 與 KWS
# 韌體（真的 arm-none-eabi-gcc 連結，不是估算），量出韌體本身（.bin 扣掉內嵌的
# `*_vela.tflite` 模型陣列）的大小：known_sound 150,080 B、KWS 238,584 B。兩者都遠低於
# FIRMWARE_CODE_BASELINE_BYTES 的 262,144 B，256 KiB 對這塊板同樣是高估（比 X 板與
# GestureAI 都更寬裕）。
#
# 把這塊板自己的算式套進同一條公式（d12: 1,563,552 + 262,144 = 1,825,696 <= 2,097,152 OK；
# d13: 2,066,832 + 262,144 = 2,328,976 > 2,097,152 no），算出來的上限是 12，跟 X 板一樣——
# 用這塊板自己實測的韌體大小算也是同一個答案（d12: 1,563,552 + 150,080 = 1,713,632 <=
# 2,097,152 OK；d13 一樣超）。但把同一條公式套回 NuGestureAI-M55M1 自己已經量過的數字
# （157,712 B）一樣會得到 12，而不是它實際出貨的 11——換句話說，這條公式本來就無法還原
# GestureAI 現有的選擇，11 從來不是這條算式算出來的（2026-09-12 那次修正的舊註解就已經
# 沒對上，見上方 GestureAI 那段）。這個落差本任務不處理、也不改 GestureAI。
#
# 因此 VoiceAI 這裡定為 11，不是算式的直接結果，是刻意的判斷：VoiceAI 借用
# NuGestureAI-M55M1 的樣板、共享同一個限制輪廓（無 LCD、無相機、console 走 USB CDC），
# 是 GestureAI 這個「樣板出借方」最近的近親，而這塊板還沒有人實際燒錄驗證過。給借樣板的一
# 方比出樣板的一方更高的上限，會是下一個人要解釋的不一致；兩塊 CDC-only 板先對齊在 11，
# 代價是比算式保守一層，換到的是說得通的故事。11 仍然比先前暫用的 10 多一層。
# 之後如果真的拿板子燒錄驗證過 known_sound 在 d12 也沒問題，這個值可以再往上調。
KNOWN_SOUND_BOARD_MAX_DEPTH = {
    "NuMaker-M55M1": 12,
    "NuGestureAI-M55M1": 11,
    "NuMaker-VoiceAI-M55M1": 11,
}

# 每塊板支援的 kind。鏡射自 mcu_toolkit/boards.json 的 has_camera：沒有相機連接器的板子
# 不可能跑 image 韌體。這裡放一份鏡射而不是 import mcu_toolkit，是為了維持 create_app()
# 不碰 MCU 的既有分界；tests/test_tunables_config.py 綁住兩邊一致。
MCU_BOARD_KINDS = {
    "NuMaker-M55M1": frozenset({"image", "audio", "known_sound"}),
    "NuGestureAI-M55M1": frozenset({"image", "audio", "known_sound"}),
    "NuMaker-VoiceAI-M55M1": frozenset({"audio", "known_sound"}),
}

# Firmware-side KWS tunables. tm_local.mcu.contract only passes through keys that
# already exist here, so a new tunable becomes reachable from project settings by being
# added to this dict and nowhere else. Consumed by tm_local.mcu.kws_codegen.kws_tokens().
KWS_RUNTIME_DEFAULTS = {
    # Seconds between consecutive inferences; the analysis window itself stays 1 clip
    # long, so 0.25 s means each keyword is scored by ~4 overlapping windows.
    "inference_hop_seconds": 0.25,
    "smooth_windows": 4,
    "required_hits": 2,
    "trigger_margin": 0.20,
    "rearm_score": 0.40,
    "rms_gate_dbfs": -55.0,
    # Power-spectrum floor before 10*log10(), identical to the 1e-10 that
    # audio_frontend.log_mel_spectrogram() applies during training. kws_codegen.py
    # requires this to stay exactly 1e-10.
    "log_epsilon": 1e-10,
}

# Union of every settings key of every kind. This is exactly the drop-in replacement for
# `set(IMAGE_DEFAULTS) | set(AUDIO_DEFAULTS)` in ProjectStore.update_project(): it keeps
# the current behaviour for image/audio byte-for-byte and merely stops abnormal_sound
# settings from being silently discarded. Prefer allowed_setting_keys(kind) when the
# project kind is in hand -- it is strictly tighter.
ALL_SETTING_KEYS = frozenset().union(*(frozenset(values) for values in DEFAULTS_BY_KIND.values()))


def normalize_kind(kind: object) -> str:
    """Canonical spelling of a project kind. Does not validate."""
    return str(kind).strip().lower()


def defaults_for_kind(kind: object) -> dict:
    """Fresh, mutable copy of the default settings for ``kind``.

    Raises ValueError for an unknown kind -- an unknown kind must never fall through to
    the audio defaults the way the old two-way branch did.
    """
    normalized = normalize_kind(kind)
    if normalized not in DEFAULTS_BY_KIND:
        known = ", ".join(PROJECT_KINDS)
        raise ValueError(f"Unknown project kind {normalized!r}. Expected one of: {known}.")
    return deepcopy(DEFAULTS_BY_KIND[normalized])


def is_known_kind(kind: object) -> bool:
    """True when ``kind`` is one of PROJECT_KINDS."""
    return normalize_kind(kind) in DEFAULTS_BY_KIND


def is_audio_like(kind: object) -> bool:
    """True for kinds whose samples are audio clips.

    ``audio``, ``abnormal_sound`` and ``known_sound``. Says nothing about whether the kind
    is a classifier -- use is_classifier() for that.
    """
    return normalize_kind(kind) in AUDIO_LIKE_KINDS


def is_classifier(kind: object) -> bool:
    """True when the model emits one score per project class."""
    return normalize_kind(kind) in CLASSIFIER_KINDS


def is_multi_label(kind: object) -> bool:
    """True when per-class scores are independent sigmoids that do not sum to 1."""
    return normalize_kind(kind) in MULTI_LABEL_KINDS


def allowed_setting_keys(kind: object | None = None) -> frozenset[str]:
    """Settings keys ProjectStore.update_project() should accept.

    With a kind, only that kind's keys (tighter, recommended). Without one, the union over
    every kind, which is the behaviour-preserving drop-in for the current expression.
    """
    if kind is None:
        return ALL_SETTING_KEYS
    normalized = normalize_kind(kind)
    if normalized not in DEFAULTS_BY_KIND:
        known = ", ".join(PROJECT_KINDS)
        raise ValueError(f"Unknown project kind {normalized!r}. Expected one of: {known}.")
    return frozenset(DEFAULTS_BY_KIND[normalized])


def mcu_application(kind: object) -> str | None:
    return MCU_APPLICATION_BY_KIND.get(normalize_kind(kind))


def is_mcu_deployable(kind: object) -> bool:
    return normalize_kind(kind) in MCU_DEPLOYABLE_KINDS


def sensitivity_preset(name: object) -> dict:
    """Fresh copy of a sensitivity preset. Raises ValueError for an unknown name."""
    normalized = str(name).strip().lower()
    if normalized not in SENSITIVITY_PRESETS:
        known = ", ".join(SENSITIVITY_PRESETS)
        raise ValueError(f"Unknown sensitivity {normalized!r}. Expected one of: {known}.")
    return dict(SENSITIVITY_PRESETS[normalized])


def abnormal_sound_context_count(settings: dict) -> int:
    """Contexts per clip implied by the frontend geometry: frame_count - context_frames + 1.

    Only for cross-checking the stored absolute ``n_contexts``; the stored integer stays
    the contract so PC and MCU cannot disagree about a rounding.
    """
    from .audio_frontend import config_from_mapping

    frames = config_from_mapping(settings).frame_count
    context_frames = int(settings.get("context_frames", ABNORMAL_SOUND_DEFAULTS["context_frames"]))
    if context_frames < 1:
        raise ValueError("context_frames must be at least 1.")
    if frames < context_frames:
        raise ValueError(
            f"clip_seconds yields only {frames} frames, fewer than context_frames={context_frames}."
        )
    return frames - context_frames + 1


# --------------------------------------------------------------------------------------
# Shared validation helpers for the tunable-parameters plan. All error messages are
# Traditional Chinese and name the setting plus its allowed value(s): they are shown to
# students verbatim (ProjectStore wraps ValueError into ProjectError, which app.py maps
# to HTTP 400).
# --------------------------------------------------------------------------------------


def _require_enum(settings: dict, key: str, allowed: tuple, label: str) -> str:
    value = str(settings.get(key, allowed[0]))
    if value not in allowed:
        raise ValueError(f"{label} 必須是 {'／'.join(allowed)} 之一，收到 {value!r}")
    return value


def _require_number(
    settings: dict, key: str, low: float, high: float, default: float, *, integer: bool = False
):
    raw = settings.get(key, default)
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必須是數字") from exc
    if integer:
        # A non-integer where an integer is required is an input error, not a request to
        # round: ``int(2.9) == 2`` would silently train a different model from the one the
        # caller asked for (fine_tune_blocks 2.9 -> 2, image_size 224.7 -> 224), and every
        # other branch of these validators raises rather than repairing. ``is_integer()``
        # rather than ``number == int(number)`` because int(nan)/int(inf) raise on their
        # own, with a message that says nothing about ``key``.
        if not number.is_integer():
            raise ValueError(f"{key} 必須是整數，收到 {raw!r}")
        value: float = int(number)
    else:
        value = number
    if not low <= value <= high:
        raise ValueError(f"{key} 必須介於 {low} 與 {high} 之間，收到 {value}")
    return value


def _require_bool(settings: dict, key: str, default: bool) -> bool:
    raw = settings.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    raise ValueError(f"{key} 必須是 true 或 false")


def _require_target(settings: dict, kind: str) -> str:
    """The board this project will be built for, refused early if it cannot run this kind.

    The deploy-time guard in tm_local/mcu/contract.py catches the same mismatch, but only
    after training: a project may be trained for "pc" and pick a board afterwards. Refusing
    here as well means a student who picks the board first never trains a model the board
    cannot run.
    """
    target = _require_enum(settings, "deployment_target", DEPLOYMENT_TARGETS, "deployment_target")
    if target == "pc":
        return target
    supported = MCU_BOARD_KINDS.get(target)
    if supported is not None and kind not in supported:
        usable = ", ".join(name for name, kinds in MCU_BOARD_KINDS.items() if kind in kinds)
        raise ValueError(f"{target} 不支援這種專案；請把「目標裝置」改成 {usable} 或「電腦」")
    return target


def validate_image_settings(settings: dict) -> dict:
    """Validate an image settings mapping. Returns a normalized copy; raises ValueError.

    Unlike the audio/known_sound validators (which historically mutated their argument in
    place), this merges onto IMAGE_DEFAULTS and returns a fresh dict so a caller always
    gets every key filled in, matching validate_audio_settings.

    Every ``_require_*`` result is written back into ``result`` (not just discarded for
    its side-effecting raise): the API boundary still accepts a string boolean
    ("true"/"false", any case) or a numeric string for convenience, but the value this
    function returns -- the one project_store.py persists -- must always be the real
    JSON type. Reading a stored ``"false"`` string back with ``bool(...)`` is truthy, so
    a discarded normalisation here would silently invert every boolean setting.
    """

    result = {**IMAGE_DEFAULTS, **settings}
    target = _require_target(result, "image")
    image_size = _require_number(result, "image_size", 96, 320, 224, integer=True)
    # MobileNetV2/small_cnn are only validated at 32 px steps (96, 128, ..., 320); every
    # MCU_IMAGE_SIZES entry is one of these, so this is the general PC-side guard and the
    # MCU check below is strictly tighter, not a separate axis.
    if image_size % 32 != 0:
        raise ValueError(f"image_size 必須是 32 的倍數（96、128、...、320），收到 {image_size}")
    if target != "pc" and image_size not in MCU_IMAGE_SIZES:
        raise ValueError(f"部署到開發板時 image_size 只能是 {MCU_IMAGE_SIZES}，收到 {image_size}")
    epochs = _require_number(result, "epochs", 1, 300, 30, integer=True)
    batch_size = _require_number(result, "batch_size", 1, 128, 16, integer=True)
    learning_rate = _require_number(result, "learning_rate", 1e-6, 0.1, 0.001)
    validation_split = _require_number(result, "validation_split", 0.0, 0.45, 0.2)
    backbone = _require_enum(result, "backbone", ("mobilenet_v2", "small_cnn"), "backbone")
    alpha = _require_number(result, "mobilenet_alpha", 0.35, 1.0, 0.35)
    if alpha not in (0.35, 0.5, 0.75, 1.0):
        raise ValueError("mobilenet_alpha 只能是 0.35、0.5、0.75 或 1.0")
    augmentation_level = _require_enum(result, "augmentation_level", AUGMENTATION_LEVELS, "augmentation_level")
    dropout = _require_number(result, "dropout", 0.0, 0.6, 0.2)
    fine_tune_blocks = _require_number(result, "fine_tune_blocks", 0, 4, 0, integer=True)
    early_stopping = _require_bool(result, "early_stopping", True)
    result["image_size"] = image_size
    result["deployment_target"] = target
    result["epochs"] = epochs
    result["batch_size"] = batch_size
    result["learning_rate"] = learning_rate
    result["validation_split"] = validation_split
    result["backbone"] = backbone
    result["mobilenet_alpha"] = alpha
    result["augmentation_level"] = augmentation_level
    result["dropout"] = dropout
    result["fine_tune_blocks"] = fine_tune_blocks
    result["early_stopping"] = early_stopping
    return result


def validate_audio_settings(settings: dict) -> dict:
    """Validate an audio settings mapping. Returns a normalized copy; raises ValueError.

    Every ``_require_*`` result is written back into ``result``: the API boundary still
    accepts a string boolean ("true"/"false", any case) or a numeric string for
    convenience, but the returned value -- the one project_store.py persists -- must
    always be the real JSON type, since a stored ``"false"`` string reads back truthy
    through ``bool(...)``.
    """

    from .audio_frontend import config_from_mapping  # lazy: audio_frontend pulls numpy/scipy/PIL

    result = {**AUDIO_DEFAULTS, **settings}
    target = _require_target(result, "audio")
    frontend = config_from_mapping(result)  # raises ValueError on fft/window conflicts

    # mel_bins only ever exists as one of these two widths: the training CNN, the
    # exported mel-filterbank tables and the MCU firmware all assume 40 or 64, so this is
    # not gated on deployment_target the way the rest of MCU_AUDIO_FRONTEND_LOCK is.
    mel_bins = int(result.get("mel_bins", 40))
    if mel_bins not in MCU_MEL_BINS:
        raise ValueError(f"mel_bins 只能是 {MCU_MEL_BINS} 之一，收到 {mel_bins}")
    if target != "pc":
        for key, expected in MCU_AUDIO_FRONTEND_LOCK.items():
            if abs(float(result.get(key, expected)) - float(expected)) > 1e-6:
                if key in MCU_AUDIO_FRONTEND_PANEL_KEYS:
                    advice = "請在進階設定把它改回預設值再訓練"
                else:
                    advice = (
                        "這個設定在畫面上沒有欄位，表示這個專案是從匯入檔或 API 建立的；"
                        "請把「目標裝置」改回「電腦」，或另開一個新專案重新收音，"
                        "才能部署到開發板"
                    )
                raise ValueError(
                    f"部署到開發板時 {key} 必須是 {expected}，收到 {result.get(key)}；{advice}"
                )
    epochs = _require_number(result, "epochs", 1, 300, 40, integer=True)
    batch_size = _require_number(result, "batch_size", 1, 128, 16, integer=True)
    learning_rate = _require_number(result, "learning_rate", 1e-6, 0.1, 0.001)
    validation_split = _require_number(result, "validation_split", 0.0, 0.45, 0.2)
    augmentation_level = _require_enum(result, "augmentation_level", AUGMENTATION_LEVELS, "augmentation_level")
    result["spec_augment"] = _require_bool(result, "spec_augment", False)
    result["session_disjoint_validation"] = _require_bool(result, "session_disjoint_validation", True)
    # early_stopping predates this plan (it was never coerced anywhere before), but
    # audio_pipeline.py reads it with the same bool(settings.get(...)) pattern that made
    # a stored "false" string read back truthy for the other booleans above, so it gets
    # the same normalisation.
    result["early_stopping"] = _require_bool(result, "early_stopping", True)
    background_weight = _require_number(result, "background_weight", 0.25, 4.0, 1.0)
    dropout = _require_number(result, "dropout", 0.0, 0.6, 0.2)
    threshold = _require_number(result, "detection_threshold", 0.05, 0.99, 0.5)
    result["detection_threshold"] = threshold
    result["background_class_id"] = str(result.get("background_class_id", "") or "")
    result["mel_bins"] = mel_bins
    result["sample_rate"] = frontend.sample_rate
    result["deployment_target"] = target
    result["epochs"] = epochs
    result["batch_size"] = batch_size
    result["learning_rate"] = learning_rate
    result["validation_split"] = validation_split
    result["augmentation_level"] = augmentation_level
    result["background_weight"] = background_weight
    result["dropout"] = dropout
    return result


def validate_abnormal_sound_settings(settings: dict) -> dict:
    """Validate an abnormal_sound settings mapping. Returns it unchanged; raises ValueError.

    Call this from ProjectStore wherever ``config_from_mapping`` is already called for the
    audio kind: the frontend keys are validated the same way, plus the scorer invariants
    that would otherwise only surface as a silently wrong score.
    """
    backend = str(settings.get("detector_backend", "yamnet_embedding")).strip().lower()
    if backend != "yamnet_embedding":
        raise ValueError(
            "This Abnormal Sound project supports detector_backend=yamnet_embedding only."
        )
    else:
        if int(settings.get("sample_rate", 0)) != 16000:
            raise ValueError("YAMNet requires mono audio resampled to exactly 16000 Hz.")
        clip_seconds = float(settings.get("clip_seconds", 0.0))
        if clip_seconds != 1.0:
            raise ValueError(
                "This YAMNet workflow requires clip_seconds=1.0 so collection, Train, "
                "Preview, and exported runners use the same window geometry."
            )
        if int(settings.get("yamnet_embedding_dim", 0)) != 1024:
            raise ValueError("This YAMNet backend requires yamnet_embedding_dim=1024.")
        if str(settings.get("yamnet_scorer", "")).strip() != "robust_diagonal":
            raise ValueError("This build implements yamnet_scorer=robust_diagonal.")
        shrinkage = float(settings.get("yamnet_scale_shrinkage", 0.50))
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("yamnet_scale_shrinkage must be between 0 and 1.")
        scale_floor = float(settings.get("yamnet_scale_floor", 0.0001))
        if not scale_floor > 0.0:
            raise ValueError("yamnet_scale_floor must be greater than 0.")
    sensitivity_preset(settings.get("sensitivity", DEFAULT_SENSITIVITY))
    hop_seconds = float(settings.get("hop_seconds", ABNORMAL_SOUND_DEFAULTS["hop_seconds"]))
    clip_seconds = float(settings.get("clip_seconds", ABNORMAL_SOUND_DEFAULTS["clip_seconds"]))
    if hop_seconds != 0.5:
        raise ValueError(
            "This YAMNet workflow requires hop_seconds=0.5 so browser and exported "
            "temporal voting use the same window geometry."
        )
    default_floor_db = ABNORMAL_SOUND_DEFAULTS["level_tolerance_floor_db"]
    floor_db = float(settings.get("level_tolerance_floor_db", default_floor_db))
    if floor_db <= 0:
        raise ValueError("level_tolerance_floor_db must be greater than 0.")
    denoise_sigma = float(settings.get("denoise_sigma", ABNORMAL_SOUND_DEFAULTS["denoise_sigma"]))
    if denoise_sigma < 0:
        raise ValueError("denoise_sigma must not be negative.")
    return settings


def validate_known_sound_settings(settings: dict) -> dict:
    """Validate a known_sound settings mapping. Mutates ``settings`` in place and
    returns it (the same object, not a merged copy the way validate_image_settings/
    validate_audio_settings work); raises ValueError.

    The frontend geometry is pinned rather than configurable: collection, Train, Preview
    and the exported runner must all agree with the official YAMNet contract, and a
    project that drifts from it would silently produce embeddings the head never saw.
    Every key this function validates is also normalised (coerced to its real JSON type)
    into ``settings`` before it is returned, not merely validated -- callers must persist
    the return value, not their original argument, even though the two happen to be the
    same object.
    """
    backend = str(settings.get("encoder_backend", "yamnet_embedding")).strip().lower()
    if backend != "yamnet_embedding":
        raise ValueError(
            "This Known Sound project supports encoder_backend=yamnet_embedding only."
        )
    if int(settings.get("sample_rate", 0)) != 16000:
        raise ValueError("YAMNet requires mono audio resampled to exactly 16000 Hz.")
    if float(settings.get("clip_seconds", 0.0)) != 1.0:
        raise ValueError(
            "This YAMNet workflow requires clip_seconds=1.0 so collection, Train, "
            "Preview, and exported runners use the same window geometry."
        )
    if float(settings.get("hop_seconds", 0.0)) != 0.5:
        raise ValueError("This YAMNet workflow requires hop_seconds=0.5.")
    # These three are normalised here rather than with the rest of the write-back at the
    # bottom: the config_from_mapping() call below builds an AudioFrontendConfig straight
    # out of this mapping, and a string that satisfied the checks above ("16000") is then
    # multiplied as a sequence inside it -- a TypeError, which project_store turns into a
    # student-facing ProjectError only for ValueError, so it would escape as a 500.
    settings["sample_rate"] = 16000
    settings["clip_seconds"] = 1.0
    settings["hop_seconds"] = 0.5
    depth = int(
        settings.get("encoder_depth", KNOWN_SOUND_DEFAULTS["encoder_depth"])
    )
    if not KNOWN_SOUND_MIN_ENCODER_DEPTH <= depth <= KNOWN_SOUND_MAX_ENCODER_DEPTH:
        raise ValueError(
            f"encoder_depth must be between {KNOWN_SOUND_MIN_ENCODER_DEPTH} and "
            f"{KNOWN_SOUND_MAX_ENCODER_DEPTH}, got {depth}."
        )
    threshold = float(
        settings.get("detection_threshold", KNOWN_SOUND_DEFAULTS["detection_threshold"])
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("detection_threshold must be strictly between 0 and 1.")
    mixup_ratio = float(settings.get("mixup_ratio", KNOWN_SOUND_DEFAULTS["mixup_ratio"]))
    if mixup_ratio < 0.0:
        raise ValueError("mixup_ratio must not be negative.")
    hold = float(
        settings.get(
            "preview_peak_hold_seconds", KNOWN_SOUND_DEFAULTS["preview_peak_hold_seconds"]
        )
    )
    if hold < 0.0:
        raise ValueError("preview_peak_hold_seconds must not be negative.")
    sessions = int(
        settings.get(
            "minimum_sessions_per_class", KNOWN_SOUND_DEFAULTS["minimum_sessions_per_class"]
        )
    )
    if sessions < 2:
        raise ValueError(
            "minimum_sessions_per_class must be at least 2; session-disjoint validation "
            "is impossible with a single recording session per class."
        )
    clips = int(
        settings.get("minimum_clips_per_class", KNOWN_SOUND_DEFAULTS["minimum_clips_per_class"])
    )
    if clips < 1:
        raise ValueError("minimum_clips_per_class must be at least 1.")
    counts: dict[str, int] = {}
    for key in ("epochs", "batch_size"):
        counts[key] = int(settings.get(key, KNOWN_SOUND_DEFAULTS[key]))
        if counts[key] < 1:
            raise ValueError(f"{key} must be at least 1.")
    learning_rate = float(settings.get("learning_rate", KNOWN_SOUND_DEFAULTS["learning_rate"]))
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0.")
    # The collection-side frontend still has to be a valid audio_frontend config, because
    # add_audio_bytes() slices clips and renders thumbnails through it. Imported lazily,
    # matching the other helpers in this module.
    from .audio_frontend import config_from_mapping

    config_from_mapping(settings)

    # Tunable-parameters plan (2026-09-12): board lock plus the new per-class/head knobs.
    # Reuses the ``depth`` already read and range-checked above -- no need to re-parse
    # encoder_depth a second time.
    target = _require_target(settings, "known_sound")
    cap = KNOWN_SOUND_BOARD_MAX_DEPTH.get(target)
    if cap is not None and depth > cap:
        raise ValueError(f"{target} 的 encoder_depth 上限是 {cap}，收到 {depth}")
    thresholds = settings.get("class_thresholds")
    if thresholds is None:
        thresholds = {}
    # A non-None, non-dict value (0, False, "", [] ...) is a real input error, not
    # "no thresholds" -- must raise rather than being silently swallowed by a falsy
    # fallback the way ``settings.get("class_thresholds", {}) or {}`` would.
    if not isinstance(thresholds, dict):
        # ValueError (not TypeError) on purpose: project_store._validate_kind_settings
        # only catches ValueError and turns it into a student-facing ProjectError.
        raise ValueError("class_thresholds 必須是 {類別 id: 門檻} 的物件")  # noqa: TRY004
    cleaned: dict[str, float] = {}
    for key, value in thresholds.items():
        # The messages lead with the UI label, not with ``key``: ``key`` is the internal
        # 32-hex class id, which the student has never seen and cannot match to a row.
        # The id is kept in parentheses because the log and the developer still need it.
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"進階設定「每類門檻」有一個類別填的不是數字（{value!r}）"
                f"；class_thresholds 類別 id {key}"
            ) from exc
        if not 0.0 < number < 1.0:
            raise ValueError(
                f"進階設定「每類門檻」有一個類別填了 {number}，必須是 0 與 1 之間（不含）的數字。"
                "把欄位清空也算 0，想用預設值請填回「偵測門檻（預設值）」的那個數字"
                f"；class_thresholds 類別 id {key}"
            )
        cleaned[str(key)] = number
    background_weight = _require_number(settings, "background_weight", 0.25, 4.0, 1.0)
    head_dropout = _require_number(settings, "head_dropout", 0.0, 0.5, 0.0)
    waveform_augment_level = _require_enum(
        settings, "waveform_augment_level", AUGMENTATION_LEVELS, "waveform_augment_level"
    )
    # early_stopping predates this plan, but known_sound_pipeline.py reads it with the
    # same bool(settings.get(...)) pattern that made a stored "false" string read back
    # truthy for the keys above, so it gets the same normalisation.
    early_stopping = _require_bool(settings, "early_stopping", True)
    # Every value this function coerced is written back (not just discarded for its
    # side-effecting raise): the stored settings dict is what project_store.py persists,
    # and a stored numeric-as-string value would read back wrong wherever a pipeline does
    # its own int()/float() coercion downstream. That includes the keys that predate the
    # tunable-parameters plan -- they used to be coerced into a local and thrown away, so
    # PATCH {"settings": {"detection_threshold": "0.7"}} persisted the *string* "0.7"
    # while the sibling validate_audio_settings stored a float for the same key.
    settings["encoder_backend"] = backend
    settings["encoder_depth"] = depth
    settings["detection_threshold"] = threshold
    settings["mixup_ratio"] = mixup_ratio
    settings["preview_peak_hold_seconds"] = hold
    settings["minimum_sessions_per_class"] = sessions
    settings["minimum_clips_per_class"] = clips
    settings["epochs"] = counts["epochs"]
    settings["batch_size"] = counts["batch_size"]
    settings["learning_rate"] = learning_rate
    settings["class_thresholds"] = cleaned
    settings["background_weight"] = background_weight
    settings["head_dropout"] = head_dropout
    settings["waveform_augment_level"] = waveform_augment_level
    settings["early_stopping"] = early_stopping
    settings["background_class_id"] = str(settings.get("background_class_id", "") or "")
    settings["deployment_target"] = target
    return settings


def ensure_workspace() -> None:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
