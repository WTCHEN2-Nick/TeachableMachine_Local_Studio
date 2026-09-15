"""Shared helpers for the reference scripts.

Thin on purpose: every real decision lives in ``tm_local/``. These scripts only open the
same ProjectStore the Studio opens, print what is about to happen, and then call the same
function the Studio's job worker calls.

Nothing here imports TensorFlow. ``--dry-run`` must stay instant, so the heavy imports are
all inside the functions that genuinely need them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Same three environment decisions app.py and scripts/start_local.py make, and for the
# same reason: TF_USE_LEGACY_KERAS has to be set BEFORE tensorflow is imported anywhere.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The Windows console already speaks UTF-8; a redirected stdout would fall back to the
# ANSI code page and blow up on the Traditional Chinese in these scripts.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):  # pragma: no cover - non-reconfigurable
        pass


# Which question a setting answers, in the four classes the README tables use:
#
#   training          只影響這次訓練怎麼學；模型輸入輸出形狀不變，開發板不用重做表
#   model-shape       改變模型的輸入或大小；上板要重新 Vela 編譯與重新燒錄
#   frontend-contract 改變「聲音怎麼變成數字」；PC 與 MCU 的前處理必須同時改
#   runtime           只影響判定門檻／平滑，不改模型本身
#   lock              選了開發板就會反過來限制其他設定的合法範圍
EFFECT = {
    # --- training -------------------------------------------------------------------
    "epochs": "training",
    "batch_size": "training",
    "learning_rate": "training",
    "validation_split": "training",
    "early_stopping": "training",
    "augmentation_level": "training",
    "dropout": "training",
    "fine_tune_blocks": "training",
    "spec_augment": "training",
    "session_disjoint_validation": "training",
    "background_weight": "training",
    "background_class_id": "training",
    "mixup_ratio": "training",
    "head_dropout": "training",
    "waveform_augment_level": "training",
    "denoise_sigma": "training",
    "yamnet_scale_shrinkage": "training",
    "yamnet_scale_floor": "training",
    "minimum_samples_per_class": "training",
    "minimum_sessions_per_class": "training",
    "minimum_clips_per_class": "training",
    "minimum_train_sessions": "training",
    "minimum_train_seconds": "training",
    "minimum_calibration_sessions": "training",
    "minimum_calibration_seconds": "training",
    "minimum_heldout_sessions": "training",
    "minimum_heldout_clips": "training",
    # --- model-shape ----------------------------------------------------------------
    "image_size": "model-shape",
    "backbone": "model-shape",
    "mobilenet_alpha": "model-shape",
    "encoder_depth": "model-shape",
    "encoder_backend": "model-shape",
    "detector_backend": "model-shape",
    "bottleneck_dim": "model-shape",
    "context_frames": "model-shape",
    "n_contexts": "model-shape",
    "yamnet_embedding_dim": "model-shape",
    # --- frontend-contract ------------------------------------------------------------
    "sample_rate": "frontend-contract",
    "clip_seconds": "frontend-contract",
    "window_ms": "frontend-contract",
    "hop_ms": "frontend-contract",
    "fft_size": "frontend-contract",
    "mel_bins": "frontend-contract",
    "fmin": "frontend-contract",
    "fmax": "frontend-contract",
    "db_floor": "frontend-contract",
    # --- runtime ----------------------------------------------------------------------
    "detection_threshold": "runtime",
    "class_thresholds": "runtime",
    "preview_peak_hold_seconds": "runtime",
    "hop_seconds": "runtime",
    "sensitivity": "runtime",
    "level_tolerance_floor_db": "runtime",
    "top_k": "runtime",
    "yamnet_scorer": "runtime",
    # --- lock ---------------------------------------------------------------------------
    "deployment_target": "lock",
}


def studio_root() -> Path:
    """The Studio root, i.e. the folder that holds ``tm_local/`` and ``02_START.bat``."""

    return ROOT


def open_store():
    """The very same ProjectStore the running Studio uses (``workspace/projects/``)."""

    from tm_local.project_store import default_store

    return default_store()


def find_project(store, key: str) -> dict[str, Any]:
    """Look a project up by id or by name; list what exists and exit(2) when it is absent."""

    projects = store.list_projects()
    for item in projects:
        if item["id"] == key or item["name"] == key:
            return store.get_raw_project(item["id"])
    print(f"找不到專案 {key!r}。目前可用的專案：", file=sys.stderr)
    for item in projects:
        print(f"  {item['id']}  {item['kind']:15s}  {item['name']}", file=sys.stderr)
    if not projects:
        print("  （workspace/projects/ 內沒有任何專案，請先在 Studio 建立）", file=sys.stderr)
    raise SystemExit(2)


def require_kind(project: dict[str, Any], kind: str) -> None:
    """Refuse to run a kind's script against another kind's project."""

    actual = project.get("kind")
    if actual != kind:
        print(
            f"專案 {project.get('name')!r} 是 {actual} 專案，這支腳本只處理 {kind}；"
            f"請改用 reference/<kind>/ 內對應的腳本。",
            file=sys.stderr,
        )
        raise SystemExit(2)


def merged_settings(kind: str, settings: dict[str, Any]) -> dict[str, Any]:
    """``defaults_for_kind(kind)`` with the project's own settings layered on top."""

    from tm_local.config import defaults_for_kind

    return {**defaults_for_kind(kind), **(settings or {})}


def print_settings(kind: str, settings: dict[str, Any]) -> None:
    """Print the settings table the Studio would train with, and where each one bites."""

    from tm_local.config import defaults_for_kind

    defaults = defaults_for_kind(kind)
    merged = {**defaults, **(settings or {})}
    print(f"== {kind} 專案設定（來源：tm_local/config.py 的 {kind.upper()}_DEFAULTS + 專案設定）==")
    print(f"{'setting':34s} {'value':>16s}  {'default?':9s} effect")
    print("-" * 82)
    for key in sorted(merged):
        value = merged[key]
        is_default = "yes" if defaults.get(key) == value else "no"
        print(f"{key:34s} {str(value)[:16]:>16s}  {is_default:9s} {EFFECT.get(key, '-')}")
    print("-" * 82)
    print(
        "effect：training=只影響學習過程／model-shape=改變模型形狀（上板要重做）／"
        "frontend-contract=改變前處理（PC 與 MCU 必須一致）／runtime=只改判定門檻／"
        "lock=選了開發板就限制其他設定"
    )


def parse_bool(value: str) -> bool:
    """argparse ``type=`` for a true/false flag (settings must stay real JSON booleans)."""

    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"必須是 true 或 false，收到 {value!r}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True, help="專案 id 或名稱")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列印將使用的設定就結束，不訓練（不載入 TensorFlow）",
    )


def add_tunable_args(parser: argparse.ArgumentParser, tunables: dict[str, dict]) -> None:
    """Turn a ``{setting_key: {"type": ..., "choices": ..., "help": ...}}`` map into flags.

    Every flag defaults to ``None`` so an omitted flag means "leave the project's stored
    setting alone" rather than "reset it to the default".
    """

    for name, spec in tunables.items():
        options: dict[str, Any] = {
            "dest": name,
            "default": None,
            "type": spec.get("type", str),
            "help": spec.get("help", ""),
        }
        if "choices" in spec:
            options["choices"] = spec["choices"]
        if "metavar" in spec:
            options["metavar"] = spec["metavar"]
        parser.add_argument("--" + name.replace("_", "-"), **options)


def collect_overrides(args: argparse.Namespace, tunables: dict[str, dict]) -> dict[str, Any]:
    return {key: getattr(args, key) for key in tunables if getattr(args, key, None) is not None}


def print_progress(value: float, message: str) -> None:
    print(f"[{value * 100:5.1f}%] {message}", flush=True)


def run_training(store, project: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Train exactly the way the Studio's job worker does (tm_local/app.py 的 train job）.

    Order matters and mirrors app.py: persist the overrides first (so Preview and Export
    can never read different settings than the model was trained with), then mark the
    project as training, then dispatch. Export/INT8 is a SEPARATE job -- this function
    stops at the ``.keras`` model, exactly like the Studio's Train button.
    """

    project_id = project["id"]
    if options:
        store.update_project(project_id, {"settings": dict(options)})
        project = store.get_raw_project(project_id)
    store.set_training_started(project_id)
    try:
        from tm_local.training_dispatch import train_project  # TensorFlow loads in here

        result = train_project(store, project_id, project["kind"], dict(options), print_progress)
    except Exception as exc:
        store.set_training_failed(project_id, str(exc))
        raise
    store.set_training_result(
        project_id, report=result["report"], artifacts=result["artifacts"]
    )
    print("訓練完成。Preview 已可使用；INT8 .tflite 要另外按 Export Model（是獨立的 job）。")
    evaluation = (result.get("report") or {}).get("evaluation")
    if evaluation is not None:
        print("evaluation:", json.dumps(evaluation, ensure_ascii=False, indent=2))
    return result


# ----------------------------------------------------------------------------------------
# TFLite helpers for the run_tflite.py scripts. Identical arithmetic to the runners the
# Studio writes into the export ZIP (image_pipeline.write_image_runner and friends).
# ----------------------------------------------------------------------------------------


def load_interpreter(model_path: Path):
    """Allocate a ``tf.lite.Interpreter``; returns (interpreter, input_detail, output_detail)."""

    import tensorflow as tf

    if not Path(model_path).is_file():
        print(f"找不到模型檔 {model_path}", file=sys.stderr)
        raise SystemExit(2)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def describe_tensors(input_detail: dict, output_detail: dict) -> None:
    """Print the INT8 contract the firmware also reads out of conversion_report.json."""

    for label, detail in (("input", input_detail), ("output", output_detail)):
        scale, zero_point = detail["quantization"]
        print(
            f"{label:6s} shape={tuple(int(v) for v in detail['shape'])} "
            f"dtype={detail['dtype'].__name__} scale={scale!r} zero_point={zero_point!r}"
        )


def quantize_input(value, detail: dict):
    """float32 -> the model's input dtype, using the exported scale/zero point."""

    import numpy as np

    value = np.asarray(value, dtype=np.float32)
    if not np.issubdtype(detail["dtype"], np.integer):
        return value.astype(detail["dtype"])
    scale, zero_point = detail["quantization"]
    info = np.iinfo(detail["dtype"])
    quantized = np.round(value / float(scale)) + float(zero_point)
    return np.clip(quantized, info.min, info.max).astype(detail["dtype"])


def dequantize_output(raw, detail: dict):
    """The model's output dtype -> float32 scores."""

    import numpy as np

    raw = np.asarray(raw)
    if not np.issubdtype(detail["dtype"], np.integer):
        return raw.astype(np.float32)
    scale, zero_point = detail["quantization"]
    return (raw.astype(np.float32) - float(zero_point)) * float(scale)


def read_labels(path: Path | None) -> list[str]:
    """Read an exported ``labels.txt`` ("0 Class 1" per line, output order)."""

    if path is None:
        print("需要 --labels（匯出 ZIP 內的 labels.txt）", file=sys.stderr)
        raise SystemExit(2)
    labels: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.append(line.split(" ", 1)[1].strip() if " " in line else line.strip())
    return labels


def print_scores(labels: list[str], scores, *, independent: bool, thresholds=None) -> None:
    """Print one line per class, strongest first.

    ``independent=True`` is the multi-label case (known_sound): the numbers are per-class
    信心分數 that never add up to 100%.
    """

    import numpy as np

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(labels) != scores.size:
        print(
            f"labels.txt 有 {len(labels)} 個類別，模型卻輸出 {scores.size} 個分數；"
            "labels.txt 與模型不是同一次匯出的。",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if independent:
        print("每類的信心分數是獨立的，不會加總成 100%：")
    else:
        print("每類分數（softmax，總和為 1）：")
    for index in np.argsort(scores)[::-1]:
        line = f"  {labels[index]:<28s} {float(scores[index]):7.2%}"
        if thresholds is not None:
            threshold = float(thresholds[index])
            mark = "  <= 偵測到" if scores[index] >= threshold else ""
            line += f"  (門檻 {threshold:.0%}){mark}"
        print(line)
