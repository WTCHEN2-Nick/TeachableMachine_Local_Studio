"""把一個 WAV 檔變成模型真正吃到的 log-mel 張量，並印出形狀與數值範圍。

用的就是訓練／Preview／量化校正同一支函式：tm_local/audio_frontend.py 的
log_mel_spectrogram()。MCU 端的 C 實作必須逐位元對齊這份規格（見
docs/MCU_AUDIO_INT8_zh-TW.md 與 reference/audio_kws/mcu_tables.py）。

模型的輸入不是原始 PCM，而是這裡印出來的 (frames, mel_bins, 1) 張量。

    .venv\\Scripts\\python.exe reference\\audio_kws\\preprocess.py --wav yes.wav --png yes.png
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
    parser.add_argument("--wav", required=True, help="要分析的 .wav（16 kHz mono 最準）")
    parser.add_argument("--frontend", help="audio_frontend.json；省略則用 Studio 預設值")
    parser.add_argument("--png", help="把頻譜存成 PNG（和樣本縮圖同一支繪圖函式）")
    args = parser.parse_args(argv)

    from tm_local.audio_frontend import (
        AudioFrontendConfig,
        config_from_mapping,
        log_mel_spectrogram,
        read_wav_file,
        spectrogram_thumbnail,
    )

    if args.frontend:
        config = config_from_mapping(json.loads(Path(args.frontend).read_text(encoding="utf-8")))
    else:
        config = AudioFrontendConfig()
    rate, signal = read_wav_file(Path(args.wav), config.sample_rate)
    feature = log_mel_spectrogram(signal, config)
    print(f"Studio 根目錄：{studio_root()}")
    print(f"讀入 {args.wav}：{signal.size} samples @ {rate} Hz")
    print(
        f"log-mel shape={feature.shape} min={float(feature.min()):.4f} "
        f"max={float(feature.max()):.4f} mean={float(feature.mean()):.4f}"
    )
    print(
        f"幾何：frames={config.frame_count} x mel={config.mel_bins}｜"
        f"window={config.window_samples} samples（{config.window_ms} ms）｜"
        f"hop={config.hop_samples} samples（{config.hop_ms} ms）｜fft={config.fft_size}"
    )
    print(
        f"數值：10*log10(mel_power) 先減掉自身最大值，截斷到 [{config.db_floor}, 0]，"
        "再線性映射到 [0, 1]。"
    )
    if args.png:
        spectrogram_thumbnail(feature, Path(args.png))
        print("已存檔：", args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
