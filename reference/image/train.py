"""用 Studio 內建的同一套程式，從命令列訓練一個 Image 專案。

跟按下 Studio 的 Train Model 完全同一條路徑：tm_local/training_dispatch.py 的
train_project() -> tm_local/image_pipeline.py 的 train_image_project()。
訓練只產出 .keras（Preview 立刻可用）；INT8 .tflite 是另一個 Export job。

    .venv\\Scripts\\python.exe reference\\image\\train.py --project <專案名稱或 id> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REFERENCE_ROOT = Path(__file__).resolve().parents[1]
KIND = "image"


def bootstrap() -> None:
    """讓「直接用檔案路徑執行」也 import 得到 reference/_common.py（它再加上 Studio 根目錄）。"""

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))


def tunables() -> dict[str, dict]:
    """可調參數；合法範圍一律取自 tm_local/config.py，不在這裡寫死第二份。"""

    from _common import parse_bool

    from tm_local.config import AUGMENTATION_LEVELS, DEPLOYMENT_TARGETS, MCU_IMAGE_SIZES

    return {
        "epochs": {"type": int, "help": "訓練回合數（1-300）；太多會過擬合，看訓練報告的驗證曲線"},
        "batch_size": {"type": int, "help": "每批張數（1-128）"},
        "learning_rate": {"type": float, "help": "學習率（1e-6 ~ 0.1）"},
        "validation_split": {"type": float, "help": "驗證集比例（0 ~ 0.45）"},
        "image_size": {
            "type": int,
            "help": f"輸入邊長，32 的倍數（96-320）；要部署到開發板只能是 {MCU_IMAGE_SIZES}",
        },
        "backbone": {
            "type": str,
            "choices": ("mobilenet_v2", "small_cnn"),
            "help": "模型骨幹；small_cnn 不支援 fine_tune_blocks",
        },
        "mobilenet_alpha": {"type": float, "help": "MobileNetV2 寬度：0.35、0.5、0.75 或 1.0"},
        "augmentation_level": {
            "type": str,
            "choices": AUGMENTATION_LEVELS,
            "help": "資料增強強度；medium 與舊版行為完全相同",
        },
        "dropout": {"type": float, "help": "分類頭 Dropout（0 ~ 0.6）；資料少、過擬合時調高"},
        "fine_tune_blocks": {
            "type": int,
            "help": "解凍 MobileNetV2 最後 N 個 block 再微調（0-4）；small_cnn 會忽略此設定",
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
            "help": "目標平台；選了開發板就會鎖住 image_size 等設定",
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
