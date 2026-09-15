"""把一個 WAV 檔變成 YAMNet encoder 真正吃到的 log-mel patch，並印出形狀與數值範圍。

與 Known Sound 用的是同一份官方 YAMNet 前處理（tm_local/yamnet_model.py 的
waveform_to_log_mel_patches()）：16 kHz、64 個 mel（125-7500 Hz）、log(mel + 0.001)、
96 x 64 的 patch。專案設定裡那組 98 x 40 的 log-mel 參數只用來切片與畫縮圖，
模型吃的不是它。

偏離分數有兩個分支：形狀（嵌入向量）與音量（dBFS），所以這裡也把第一秒的音量印出來。
之後的步驟（1024 維嵌入 -> 與正常基準比較 -> 偏離比值）在 run_tflite.py。

    .venv\\Scripts\\python.exe reference\\abnormal_sound\\preprocess.py --wav room.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REFERENCE_ROOT = Path(__file__).resolve().parents[1]


def bootstrap() -> None:
    """讓「直接用檔案路徑執行」也 import 得到 reference/_common.py（它再加上 Studio 根目錄）。"""

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))


def main(argv: list[str] | None = None) -> int:
    bootstrap()
    from _common import studio_root

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wav", required=True, help="要分析的 .wav")
    parser.add_argument("--png", help="把第一張 patch 存成 PNG（會先線性拉伸到 0-1 才上色）")
    args = parser.parse_args(argv)

    import numpy as np

    from tm_local.audio_frontend import (
        AudioFrontendConfig,
        read_wav_file,
        rms_dbfs,
        spectrogram_thumbnail,
    )
    from tm_local.yamnet_model import frontend_contract, waveform_to_log_mel_patches

    contract = frontend_contract()
    rate, signal = read_wav_file(Path(args.wav), int(contract["sample_rate"]))
    patches = waveform_to_log_mel_patches(np.asarray(signal, dtype=np.float32))
    print(f"Studio 根目錄：{studio_root()}")
    print(f"讀入 {args.wav}：{signal.size} samples @ {rate} Hz")
    print(
        f"patches shape={patches.shape}（張數, {contract['patch_frames']} frames, "
        f"{contract['mel_bands']} mel）"
        f" min={float(patches.min()):.4f} max={float(patches.max()):.4f}"
    )
    print(f"第一秒音量 {rms_dbfs(signal, AudioFrontendConfig()):.1f} dBFS（偏離分數的音量分支）")
    print("YAMNet 前處理契約（tm_local/yamnet_model.py frontend_contract()）：")
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    if args.png:
        patch = patches[0]
        span = float(patch.max() - patch.min()) or 1.0
        spectrogram_thumbnail((patch - float(patch.min())) / span, Path(args.png))
        print("已存檔：", args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
