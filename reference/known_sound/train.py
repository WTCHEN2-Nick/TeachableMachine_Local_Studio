"""用 Studio 內建的同一套程式，從命令列訓練一個 Known Sound 專案。

路徑：tm_local/training_dispatch.py 的 train_project() -> tm_local/known_sound_pipeline.py
的 train_known_sound_project()。結構是「凍結的 YAMNet encoder + Dense(n, sigmoid) head」，
每個類別各自輸出獨立的信心分數，不會加總成 1。

驗證一律以 recording_session_id 切分（split_sessions()），同一段錄音切出來的片段永遠不會
同時出現在訓練與驗證，報出來的分數才誠實。訓練只產出 .keras；INT8 是另一個 Export job。

    .venv\\Scripts\\python.exe reference\\known_sound\\train.py --project <名稱或 id> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REFERENCE_ROOT = Path(__file__).resolve().parents[1]
KIND = "known_sound"


def bootstrap() -> None:
    """讓「直接用檔案路徑執行」也 import 得到 reference/_common.py（它再加上 Studio 根目錄）。"""

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))


def tunables() -> dict[str, dict]:
    """可調參數；合法範圍與開發板上限一律取自 tm_local/config.py。"""

    from _common import parse_bool

    from tm_local.config import (
        AUGMENTATION_LEVELS,
        DEPLOYMENT_TARGETS,
        KNOWN_SOUND_BOARD_MAX_DEPTH,
        KNOWN_SOUND_MAX_ENCODER_DEPTH,
        KNOWN_SOUND_MIN_ENCODER_DEPTH,
    )

    caps = "／".join(f"{board} {cap}" for board, cap in KNOWN_SOUND_BOARD_MAX_DEPTH.items())
    return {
        "epochs": {"type": int, "help": "head 的訓練回合數（encoder 凍結，所以很快）"},
        "batch_size": {"type": int, "help": "每批片段數"},
        "learning_rate": {"type": float, "help": "學習率"},
        "encoder_depth": {
            "type": int,
            "help": (
                f"保留幾個 YAMNet block（{KNOWN_SOUND_MIN_ENCODER_DEPTH}-"
                f"{KNOWN_SOUND_MAX_ENCODER_DEPTH}）；模型大小幾乎只由它決定，"
                f"開發板上限：{caps}"
            ),
        },
        "head_dropout": {
            "type": float,
            "help": "Dense 前的 Dropout（0 ~ 0.5）；只影響訓練，不改參數量",
        },
        "waveform_augment_level": {
            "type": str,
            "choices": AUGMENTATION_LEVELS,
            "help": "在波形上做增益／時移／加噪（只增強訓練 session）",
        },
        "mixup_ratio": {
            "type": float,
            "help": "把兩個不同類別的片段相加，做出「同時發生」的訓練樣本；0 關閉",
        },
        "detection_threshold": {
            "type": float,
            "help": "沒有個別設定的類別共用的判定門檻（0 ~ 1，不含端點）",
        },
        "background_weight": {
            "type": float,
            "help": "背景類在加權 BCE 內的倍率（0.25 ~ 4）；誤報多就調高",
        },
        "background_class_id": {"type": str, "help": "指定哪個類別是背景／沒有目標聲音"},
        "preview_peak_hold_seconds": {
            "type": float,
            "help": "Preview 的峰值保持秒數；短促的聲音才看得到",
        },
        "early_stopping": {
            "type": parse_bool,
            "choices": (True, False),
            "metavar": "{true,false}",
            "help": "驗證分數不再進步就提早停止",
        },
        "deployment_target": {
            "type": str,
            "choices": DEPLOYMENT_TARGETS,
            "help": "目標平台；選了開發板就會限制 encoder_depth",
        },
    }


def parse_class_thresholds(pairs: list[str] | None) -> dict[str, float]:
    """``--class-threshold <class_id>=0.6``（可重複）-> ``class_thresholds`` 設定值。"""

    thresholds: dict[str, float] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--class-threshold 要寫成 <class_id>=0.6，收到 {item!r}")
        class_id, _, raw = item.partition("=")
        try:
            thresholds[class_id.strip()] = float(raw)
        except ValueError as exc:
            raise SystemExit(f"--class-threshold 的門檻必須是數字，收到 {raw!r}") from exc
    return thresholds


def main(argv: list[str] | None = None) -> int:
    bootstrap()
    from _common import (
        add_common_args,
        add_tunable_args,
        collect_overrides,
        find_project,
        open_store,
        print_settings,
        require_kind,
        run_training,
    )

    specs = tunables()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    add_tunable_args(parser, specs)
    parser.add_argument(
        "--class-threshold",
        action="append",
        metavar="CLASS_ID=0.6",
        help=(
            "單一類別的判定門檻，可重複；會用這裡列出的類別整批取代 class_thresholds，"
            "沒列出的類別會改用 detection_threshold，不是保留原本各自的門檻"
        ),
    )
    args = parser.parse_args(argv)

    store = open_store()
    project = find_project(store, args.project)
    require_kind(project, KIND)

    overrides = collect_overrides(args, specs)
    thresholds = parse_class_thresholds(args.class_threshold)
    if thresholds:
        overrides["class_thresholds"] = thresholds
        known = {item["id"]: item["name"] for item in project.get("classes", [])}
        for class_id in thresholds:
            if class_id not in known:
                print(f"警告：{class_id!r} 不是這個專案的類別 id", file=sys.stderr)
        print("每類門檻：", {known.get(key, key): value for key, value in thresholds.items()})

    print_settings(KIND, {**project.get("settings", {}), **overrides})
    if args.dry_run:
        print("（--dry-run：只列出設定，沒有訓練）")
        return 0
    run_training(store, project, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
