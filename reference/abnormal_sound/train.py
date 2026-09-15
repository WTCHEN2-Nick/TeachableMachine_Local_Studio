"""用 Studio 內建的同一套程式，從命令列訓練一個 Abnormal Sound 專案。

路徑：tm_local/training_dispatch.py 的 train_project() ->
tm_local/yamnet_anomaly_pipeline.py 的 train_yamnet_abnormal_sound_project()。

這是 open-set 偏離偵測，不是分類器：只收 Normal 音訊，訓練就是「量出正常長什麼樣子」，
輸出只有偏離程度，永遠不會說出那是什麼聲音。它也**沒有開發板版本**（tm_local/config.py 的
MCU_APPLICATION_BY_KIND 把 abnormal_sound 對應到 None），只在 PC 上跑。

注意：這個 pipeline 只認 detector_backend=yamnet_embedding，而且傳進來的 options 必須
先寫進專案設定（pipeline 會自己檢查兩邊一致），所以本腳本一律先存設定再訓練。

    .venv\\Scripts\\python.exe reference\\abnormal_sound\\train.py --project <名稱或 id> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REFERENCE_ROOT = Path(__file__).resolve().parents[1]
KIND = "abnormal_sound"


def bootstrap() -> None:
    """讓「直接用檔案路徑執行」也 import 得到 reference/_common.py（它再加上 Studio 根目錄）。"""

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))


def tunables() -> dict[str, dict]:
    """YAMNet 流程真正會讀到的可調參數（取自 tm_local/config.py）。"""

    from tm_local.config import SENSITIVITY_PRESETS

    return {
        "sensitivity": {
            "type": str,
            "choices": tuple(SENSITIVITY_PRESETS),
            "help": "判定寬嚴：alpha 與「連續幾個視窗要投票通過」的預設組合",
        },
        "level_tolerance_floor_db": {
            "type": float,
            "help": "音量分支的容忍下限（dB）；房間太安靜時避免一點音量變化就報警",
        },
        "yamnet_scale_shrinkage": {
            "type": float,
            "help": "正常統計量的收縮係數（0-1）；樣本少時調高比較穩",
        },
        "yamnet_scale_floor": {"type": float, "help": "每一維標準差的下限，避免除以 0"},
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
    print(
        "提醒：epochs／batch_size／learning_rate／denoise_sigma／bottleneck_dim 屬於原始的"
        " Dense 自編碼器基準線，YAMNet 流程不會用到，所以這支腳本不開那些旗標。"
    )
    if args.dry_run:
        print("（--dry-run：只列出設定，沒有訓練）")
        return 0
    run_training(store, project, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
