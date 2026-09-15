"""Deploy contract: everything an MCU firmware build needs from a trained project.

``collect()`` reads a project's persisted training/export state (project.json plus the
JSON sidecars written into ``models/``) and assembles a single validated
``DeployContract``. ``validate()`` then checks that contract against the fixed rules the
firmware build depends on (integer I/O, a supported image size, the locked audio/YAMNet
frontend, the board's known_sound encoder-depth cap, ...).

Both functions raise ``DeployError`` with Traditional-Chinese, student-facing messages
that name the project setting to change -- these strings are shown to students verbatim,
so keep them specific and actionable. Neither function ever lets a bare KeyError,
IndexError, TypeError, ValueError or json.JSONDecodeError escape: project.json and the
``models/`` sidecars are files a student (or a hand-edited export) can leave malformed or
truncated, and every path built from them is treated as untrusted input, joined with
``utils.path_within()`` rather than raw ``/``.

Stays cheap to import: no TensorFlow at module level. Tensor I/O info comes from the
conversion report that export_service/tflite_export already wrote to disk (itself
produced by ``tflite_export.inspect_tflite()`` at export time), so this module never
needs to load TensorFlow or re-inspect the ``.tflite`` file itself -- ``sha256_file()`` is
plain hashlib. ``yamnet_model`` is imported lazily inside ``validate()`` for the same
reason, even though the module itself is TensorFlow-free at import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import (
    KWS_RUNTIME_DEFAULTS,
    MCU_AUDIO_FRONTEND_LOCK,
    MCU_IMAGE_SIZES,
    MCU_MEL_BINS,
    mcu_application,
    normalize_kind,
)
from ..utils import path_within, read_json, sha256_file
from .boards import Board
from .errors import DeployError

# Board/frontend lock constants and firmware-side KWS tunables now live in tm_local.config
# (the single source both the UI validators and this module read from). These names stay
# as aliases so kws_codegen.py, tm_local/mcu/apps/*.py and the existing MCU tests keep
# importing them unchanged.
MCU_AUDIO_FRONTEND = MCU_AUDIO_FRONTEND_LOCK
DEFAULT_KWS_RUNTIME = KWS_RUNTIME_DEFAULTS

# Sidecar/report parsing errors that a malformed or hand-edited artifact can raise.
# DeployError is never one of these, so wrapping a block in this tuple can never
# accidentally swallow a DeployError raised inside it.
_PARSE_ERRORS = (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError)

# Firmware compares a quantised int8 score against a threshold baked into generated C at
# deploy time. Below 0.05 quantisation noise alone can trip every frame; above 0.99 an int8
# score (max representable value ~0.996) could never reach it and the board would look dead.
# Mirrors kws_codegen.py's TRIGGER_THRESHOLD clamp so every path that bakes a threshold into
# firmware -- audio/KWS's single detection_threshold and known_sound's per-class
# class_thresholds alike -- applies the same firmware-safe bounds, regardless of what a
# hand-edited training_report.json settings block claims.
_THRESHOLD_MIN = 0.05
_THRESHOLD_MAX = 0.99


def _clamp_threshold(value: float) -> float:
    return min(_THRESHOLD_MAX, max(_THRESHOLD_MIN, value))


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    scale: float
    zero_point: int


@dataclass
class DeployContract:
    kind: str
    application: str
    board: Board
    project_id: str
    project_name: str
    labels: list[str]
    models_dir: Path
    int8_path: Path
    model_sha256: str
    input: TensorInfo
    output: TensorInfo
    image_size: int | None
    audio_frontend: dict[str, Any] | None
    yamnet_frontend: dict[str, Any] | None
    encoder_depth: int | None
    detection_threshold: float
    class_thresholds: list[float]
    hop_seconds: float
    clip_seconds: float
    peak_hold_seconds: float
    kws_runtime: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_KWS_RUNTIME))
    # The background/"nothing was said" class the project TRAINED against, or None when it
    # has none. Defaulted so every existing constructor call stays valid. It exists because
    # the student can name that class anything (`background_class_id` in the settings), and
    # kws_codegen's label matcher would then pick a different class than training weighted
    # -- the board would suppress the wrong keyword and measure its trigger margin against
    # the wrong baseline.
    background_index: int | None = None


def _tensor(entry: Any, *, source: str, label: str) -> TensorInfo:
    if not isinstance(entry, dict):
        raise DeployError(f"{source} 的{label}張量資訊格式錯誤，請重新 Export")
    try:
        return TensorInfo(
            name=str(entry.get("name", "")),
            shape=tuple(int(x) for x in entry["shape"]),
            dtype=str(entry["dtype"]),
            scale=float(entry["scale"]),
            zero_point=int(entry["zero_point"]),
        )
    except _PARSE_ERRORS as exc:
        raise DeployError(f"{source} 的{label}張量資訊缺漏或格式錯誤，請重新 Export") from exc


def _first_tensor_entry(items: Any, *, source: str, label: str) -> Any:
    if not isinstance(items, list) or not items:
        raise DeployError(f"{source} 缺少{label}張量資訊，請重新 Export")
    return items[0]


def _read_json_sidecar(path: Path, description: str) -> dict[str, Any]:
    """``read_json()`` a project sidecar, converting a broken file into a DeployError."""

    try:
        data = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DeployError(f"{description} 損毀或格式錯誤，請重新 Export") from exc
    if not isinstance(data, dict):
        raise DeployError(f"{description} 格式錯誤，請重新 Export")
    return data


def collect(store: Any, project_id: str, board: Board) -> DeployContract:
    """Gather a validated-shape ``DeployContract`` from a trained, exported project.

    Raises ``DeployError`` (never a bare KeyError/IndexError/TypeError/ValueError/
    json.JSONDecodeError) whenever the project cannot be deployed yet -- wrong kind, not
    trained, not exported to Strict INT8, or a corrupted/hand-edited artifact -- naming
    the screen/setting/file the student needs to fix next.
    """

    raw = store.get_raw_project(project_id)
    kind = normalize_kind(raw.get("kind"))
    application = mcu_application(kind)
    if application is None:
        raise DeployError(f"{kind} 專案不支援部署到開發板（只支援 image、audio、known_sound）")

    training = raw.get("training") or {}
    if training.get("state") != "trained":
        raise DeployError(
            "請先完成 Train Model，再到 Export Model 產生 Strict INT8 TFLite 後才能部署到開發板"
        )
    artifacts = training.get("artifacts") or {}
    if not artifacts.get("int8"):
        raise DeployError("請先在 Export Model 產生 Strict INT8 TFLite，再部署到開發板")

    models_dir = store.project_dir(project_id) / "models"
    # artifacts["int8"] is a leaf name from project.json, which a hand-edited file could
    # point outside models_dir (e.g. "../../../../Windows/win.ini"); resolve it through
    # path_within() rather than a raw "/" join so that can never escape the project.
    try:
        int8_path = path_within(models_dir, str(artifacts["int8"]))
    except ValueError as exc:
        raise DeployError("INT8 模型檔路徑不合法，請重新 Export") from exc
    if not int8_path.is_file():
        raise DeployError("找不到 INT8 模型檔，請重新 Export")

    report = training.get("report") or {}
    conversion = (report.get("conversion") or {}).get("models") or {}
    entry = conversion.get("int8")
    if not entry:
        conv_path = models_dir / "conversion_report.json"
        if conv_path.is_file():
            entry = (_read_json_sidecar(conv_path, "conversion_report.json").get("models") or {}).get(
                "int8"
            )
    if not entry:
        raise DeployError("conversion_report.json 缺少 int8 模型資訊，請重新 Export")
    if not entry.get("strict_full_integer", False):
        raise DeployError("INT8 模型不是全整數圖，無法部署到 Ethos-U")

    # Recompute the checksum from the actual file on disk (TF-free: plain hashlib) rather
    # than trusting the recorded value outright; a mismatch means the file and the export
    # record have drifted (e.g. a stale copy, or a hand-edited report).
    computed_sha256 = sha256_file(int8_path)
    recorded_sha256 = entry.get("sha256")
    if recorded_sha256 and str(recorded_sha256) != computed_sha256:
        raise DeployError("INT8 模型檔與匯出紀錄不符，請重新 Export")

    input_tensor = _tensor(
        _first_tensor_entry(entry.get("inputs"), source="conversion_report.json", label="輸入"),
        source="conversion_report.json",
        label="輸入",
    )
    output_tensor = _tensor(
        _first_tensor_entry(entry.get("outputs"), source="conversion_report.json", label="輸出"),
        source="conversion_report.json",
        label="輸出",
    )

    labels = [str(item["name"]) for item in raw.get("classes") or []]
    settings = report.get("settings") or {}
    if not isinstance(settings, dict):
        settings = {}

    audio_frontend = None
    if kind == "audio":
        path = models_dir / "audio_frontend.json"
        if not path.is_file():
            raise DeployError("找不到 audio_frontend.json，請重新 Export")
        audio_frontend = _read_json_sidecar(path, "audio_frontend.json")

    yamnet_frontend = None
    encoder_depth = None
    if kind == "known_sound":
        path = models_dir / "yamnet_frontend.json"
        if not path.is_file():
            raise DeployError("找不到 yamnet_frontend.json，請重新訓練")
        yamnet_frontend = _read_json_sidecar(path, "yamnet_frontend.json")
        depth_value = (report.get("encoder") or {}).get("depth")
        if depth_value is None:
            depth_value = settings.get("encoder_depth")
        if depth_value is None:
            raise DeployError("known_sound 訓練報告缺少 encoder_depth，請重新訓練並匯出")
        try:
            encoder_depth = int(depth_value)
        except _PARSE_ERRORS as exc:
            raise DeployError("training_report.json 的 encoder_depth 格式錯誤，請重新訓練並匯出") from exc

    try:
        image_size = (
            int(settings["image_size"]) if kind == "image" and settings.get("image_size") else None
        )
        threshold = _clamp_threshold(float(settings.get("detection_threshold", 0.5)))
        hop_seconds = float(settings.get("hop_seconds", 0.5))
        clip_seconds = float(settings.get("clip_seconds", 1.0))
        peak_hold_seconds = float(settings.get("preview_peak_hold_seconds", 1.5))
    except _PARSE_ERRORS as exc:
        raise DeployError("training_report.json 的 settings 欄位格式錯誤，請重新訓練並匯出") from exc

    # class_thresholds is a {class_id: 0..1} map (default {}), per-entry falling back to
    # detection_threshold -- see docs/MCU_DEPLOY_zh-TW.md. Resolved here into one float per
    # label, in project class order, so the rest of the pipeline only ever deals with a
    # plain list.
    raw_thresholds = settings.get("class_thresholds")
    class_thresholds: list[float] = []
    for class_item in raw.get("classes") or []:
        value = threshold
        if isinstance(raw_thresholds, dict):
            candidate = raw_thresholds.get(str(class_item.get("id")))
            if candidate is not None:
                try:
                    value = _clamp_threshold(float(candidate))
                except _PARSE_ERRORS:
                    value = threshold
        class_thresholds.append(value)

    kws_runtime = dict(DEFAULT_KWS_RUNTIME)
    raw_kws_runtime = settings.get("kws_runtime")
    if isinstance(raw_kws_runtime, dict):
        kws_runtime.update({k: v for k, v in raw_kws_runtime.items() if k in kws_runtime})

    # Anything but a real in-range int (a bool, a float, a string, an index left over from
    # a since-deleted class) falls back to None, i.e. to kws_codegen's label matcher --
    # baking a nonsense index into the generated C would be far worse than that fallback.
    raw_background = settings.get("background_class_index")
    background_index: int | None = None
    if (
        isinstance(raw_background, int)
        and not isinstance(raw_background, bool)
        and 0 <= raw_background < len(labels)
    ):
        background_index = int(raw_background)

    return DeployContract(
        kind=kind,
        application=application,
        board=board,
        project_id=str(raw["id"]),
        project_name=str(raw.get("name", "project")),
        labels=labels,
        models_dir=models_dir,
        int8_path=int8_path,
        model_sha256=computed_sha256,
        input=input_tensor,
        output=output_tensor,
        image_size=image_size,
        audio_frontend=audio_frontend,
        yamnet_frontend=yamnet_frontend,
        encoder_depth=encoder_depth,
        detection_threshold=threshold,
        class_thresholds=class_thresholds,
        hop_seconds=hop_seconds,
        clip_seconds=clip_seconds,
        peak_hold_seconds=peak_hold_seconds,
        kws_runtime=kws_runtime,
        background_index=background_index,
    )


def validate(contract: DeployContract) -> None:
    """Reject a ``DeployContract`` that cannot actually run on its target board."""

    board = contract.board
    if contract.input.dtype != "int8" or contract.output.dtype != "int8":
        raise DeployError("開發板只接受 int8 輸入／輸出的模型，請用 Strict INT8 匯出")
    if not contract.labels:
        raise DeployError("專案沒有任何類別")
    if contract.output.shape[-1] != len(contract.labels):
        raise DeployError(
            f"模型輸出 {contract.output.shape[-1]} 個分數，但專案有 {len(contract.labels)} "
            "個類別數；請重新訓練並匯出"
        )
    if contract.kind == "image":
        if not board.has_camera:
            raise DeployError(f"{board.label} 沒有相機，無法部署 image 專案")
        if contract.image_size not in MCU_IMAGE_SIZES:
            raise DeployError(
                f"開發板只支援 image_size {MCU_IMAGE_SIZES}，目前是 {contract.image_size}；"
                "請改設定後重新訓練"
            )
        if contract.input.shape[1:] != (contract.image_size, contract.image_size, 3):
            raise DeployError("模型輸入形狀與 image_size 不一致，請重新訓練並匯出")
    elif contract.kind == "audio":
        frontend = contract.audio_frontend
        if not isinstance(frontend, dict):
            raise DeployError("找不到 audio_frontend.json，請重新 Export")
        for key, expected in MCU_AUDIO_FRONTEND.items():
            if key not in frontend:
                raise DeployError(f"audio_frontend.json 缺少 {key}，請重新 Export")
            try:
                actual = float(frontend[key])
            except _PARSE_ERRORS as exc:
                raise DeployError(f"audio_frontend.json 的 {key} 格式錯誤，請重新 Export") from exc
            if abs(actual - float(expected)) > 1e-6:
                raise DeployError(
                    f"開發板需要 {key} = {expected}，目前是 {frontend.get(key)}；請改訓練設定後重新訓練"
                )
        if "mel_bins" not in frontend:
            raise DeployError("audio_frontend.json 缺少 mel_bins，請重新 Export")
        try:
            mel_bins = int(frontend["mel_bins"])
        except _PARSE_ERRORS as exc:
            raise DeployError("audio_frontend.json 的 mel_bins 格式錯誤，請重新 Export") from exc
        if mel_bins not in MCU_MEL_BINS:
            raise DeployError(f"開發板只支援 mel_bins {MCU_MEL_BINS}")
        if "frame_count" not in frontend:
            raise DeployError("audio_frontend.json 缺少 frame_count，請重新 Export")
        try:
            frame_count = int(frontend["frame_count"])
        except _PARSE_ERRORS as exc:
            raise DeployError("audio_frontend.json 的 frame_count 格式錯誤，請重新 Export") from exc
        expected_shape = (1, frame_count, mel_bins, 1)
        if contract.input.shape != expected_shape:
            raise DeployError("模型輸入形狀與 audio_frontend.json 不一致，請重新訓練並匯出")
    elif contract.kind == "known_sound":
        from ..yamnet_model import frontend_contract

        expected = frontend_contract()
        actual = contract.yamnet_frontend or {}
        for key, value in expected.items():
            if actual.get(key) != value:
                raise DeployError(f"yamnet_frontend.json 的 {key} 與內建契約不同，請重新訓練")
        if contract.encoder_depth is None or contract.encoder_depth > board.known_sound_max_depth:
            raise DeployError(
                f"{board.label} 的 encoder_depth 上限是 {board.known_sound_max_depth}，"
                f"目前是 {contract.encoder_depth}；請降低後重新訓練"
            )
        # Derived from the same frontend_contract() looked up above (CLAUDE.md decision 11:
        # every MCU-side constant is generated from the contract, never hand-synced) rather
        # than hardcoding (1, 96, 64) -- model_input_shape is a 2-element [frames, mel_bins]
        # list, so the batch dimension is prepended here.
        expected_input_shape = (1, *expected["model_input_shape"])
        if contract.input.shape != expected_input_shape:
            raise DeployError(f"known_sound 模型輸入必須是 {list(expected_input_shape)}")
    if len(contract.class_thresholds) != len(contract.labels):
        raise DeployError("每類門檻數量與類別數不符")
