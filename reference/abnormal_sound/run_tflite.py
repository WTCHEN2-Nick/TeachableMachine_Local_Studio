"""用匯出的 INT8 .tflite 對一段 WAV 算出 YAMNet 嵌入向量，並（可選）算出偏離比值。

模型本身只輸出 1024 維嵌入，沒有類別；判定要靠匯出 ZIP 內的 yamnet_scorer.json：
裡面有每個 runtime（keras／float32／dynamic／int8／uint8）各自校正過的正常中心、尺度與門檻。
算式與匯出 ZIP 內的 run_model.py 相同（tm_local/yamnet_anomaly_pipeline.py 的
write_yamnet_runner()），這裡只跑一個 1 秒視窗，不做多視窗投票。

這個專案種類只回答「偏離正常多少」，永遠不會說出那是什麼聲音，也沒有開發板版本。

    .venv\\Scripts\\python.exe reference\\abnormal_sound\\run_tflite.py ^
        --model abnormal_sound_yamnet_embedding_int8.tflite ^
        --scorer yamnet_scorer.json --input room.wav
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
    from _common import (
        dequantize_output,
        describe_tensors,
        load_interpreter,
        quantize_input,
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, help="abnormal_sound_yamnet_embedding_int8.tflite"
    )
    parser.add_argument("--input", help="要分析的 .wav")
    parser.add_argument("--scorer", help="匯出 ZIP 內的 yamnet_scorer.json")
    parser.add_argument("--runtime", default="int8", help="yamnet_scorer.json 內的 runtime 名稱")
    parser.add_argument("--dry-run", action="store_true", help="只印張量資訊，不需要 --input")
    args = parser.parse_args(argv)

    interpreter, input_detail, output_detail = load_interpreter(Path(args.model))
    describe_tensors(input_detail, output_detail)
    if args.dry_run:
        print("輸入是 YAMNet log-mel patch（96 x 64），輸出是 1024 維嵌入，不是類別分數。")
        return 0
    if not args.input:
        print("需要 --input（一段 .wav），或改用 --dry-run", file=sys.stderr)
        return 2

    import numpy as np

    from tm_local.audio_frontend import AudioFrontendConfig, fix_length, read_wav_file, rms_dbfs
    from tm_local.yamnet_model import waveform_to_log_mel_patches

    _rate, signal = read_wav_file(Path(args.input), 16000)
    window = fix_length(np.asarray(signal, dtype=np.float32), 16000)
    outputs = []
    for patch in waveform_to_log_mel_patches(window):
        value = patch[None, ...].astype(np.float32)
        interpreter.set_tensor(input_detail["index"], quantize_input(value, input_detail))
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0]
        outputs.append(dequantize_output(raw, output_detail))
    embedding = np.mean(np.stack(outputs), axis=0).astype(np.float64)
    embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
    print(f"嵌入向量：{embedding.size} 維，已正規化成單位長度。")

    if not args.scorer:
        print("（沒有 --scorer 就只到嵌入向量為止；判定需要 yamnet_scorer.json）")
        return 0
    entry = json.loads(Path(args.scorer).read_text(encoding="utf-8"))["runtimes"][args.runtime]
    reference, threshold = entry["reference"], entry["threshold"]
    center = np.asarray(reference["center"], dtype=np.float64)
    scale = np.asarray(reference["scale"], dtype=np.float64)
    shape_ratio = float(np.mean(((embedding - center) / scale) ** 2)) / float(threshold["t_shape"])
    level_dbfs = rms_dbfs(window, AudioFrontendConfig())
    level_ratio = abs(level_dbfs - float(threshold["c_level_db"])) / float(threshold["t_level_db"])
    ratio = max(shape_ratio, level_ratio)
    print(f"形狀分支比值 {shape_ratio:.3f}｜音量分支比值 {level_ratio:.3f}（{level_dbfs:.1f} dBFS）")
    print(f"偏離比值 {ratio:.3f} -> 這個視窗{'偏離正常' if ratio > 1.0 else '在正常範圍內'}")
    print("注意：它只說偏離程度，不會、也不能說出那是什麼聲音。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
