"""用 Studio 內建的同一套程式，從命令列訓練一個 Audio（關鍵字辨識 KWS）專案。

路徑：tm_local/training_dispatch.py 的 train_project() -> tm_local/audio_pipeline.py 的
train_audio_project()。模型吃的是 log-mel 頻譜（不是原始 PCM），規格在
tm_local/audio_frontend.py。訓練只產出 .keras；INT8 .tflite 是另一個 Export job。

log-mel 的幾何（sample_rate／window_ms／hop_ms／fft_size…）故意不開成旗標：要部署到開發板
時它們被 config.MCU_AUDIO_FRONTEND_LOCK 鎖死，改了 PC 與 MCU 就對不起來。唯一可調的是
mel_bins（40 或 64），它會一併改變韌體的 C 表。

    .venv\\Scripts\\python.exe reference\\audio_kws\\train.py --project <專案名稱或 id> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REFERENCE_ROOT = Path(__file__).resolve().parents[1]
KIND = "audio"


def bootstrap() -> None:
    """讓「直接用檔案路徑執行」也 import 得到 reference/_common.py（它再加上 Studio 根目錄）。"""

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))


def tunables() -> dict[str, dict]:
    """可調參數；合法範圍一律取自 tm_local/config.py，不在這裡寫死第二份。"""

    from _common import parse_bool

    from tm_local.config import AUGMENTATION_LEVELS, DEPLOYMENT_TARGETS, MCU_MEL_BINS

    return {
        "epochs": {"type": int, "help": "訓練回合數（1-300）"},
        "batch_size": {"type": int, "help": "每批片段數（1-128）"},
        "learning_rate": {"type": float, "help": "學習率（1e-6 ~ 0.1）"},
        "validation_split": {"type": float, "help": "驗證集比例（0 ~ 0.45）"},
        "mel_bins": {
            "type": int,
            "choices": MCU_MEL_BINS,
            "help": "mel 頻帶數；模型輸入形狀與 MCU 的 mel 表都會跟著變",
        },
        "augmentation_level": {
            "type": str,
            "choices": AUGMENTATION_LEVELS,
            "help": "log-mel 上的時移／增益／加噪強度；medium 與舊版行為相同",
        },
        "spec_augment": {
            "type": parse_bool,
            "choices": (True, False),
            "metavar": "{true,false}",
            "help": "在頻譜上隨機遮一小段時間與頻帶（各約 1/10）",
        },
        "session_disjoint_validation": {
            "type": parse_bool,
            "choices": (True, False),
            "metavar": "{true,false}",
            "help": "驗證片段不與訓練片段來自同一段錄音；關掉會讓準確率虛高",
        },
        "background_weight": {
            "type": float,
            "help": "背景類的 class_weight 倍率（0.25 ~ 4）；誤觸發太多就調高",
        },
        "background_class_id": {"type": str, "help": "指定哪個類別代表「沒有人說話」"},
        "dropout": {"type": float, "help": "分類頭 Dropout（0 ~ 0.6）"},
        "detection_threshold": {
            "type": float,
            "help": "觸發門檻（0.05 ~ 0.99）；只影響判定，不會改模型權重",
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
            "help": "目標平台；選了開發板就會鎖住前處理幾何",
        },
    }


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
    args = parser.parse_args(argv)

    store = open_store()
    project = find_project(store, args.project)
    require_kind(project, KIND)

    overrides = collect_overrides(args, specs)
    print_settings(KIND, {**project.get("settings", {}), **overrides})
    if args.dry_run:
        print("（--dry-run：只列出設定，沒有訓練）")
        return 0
    run_training(store, project, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
