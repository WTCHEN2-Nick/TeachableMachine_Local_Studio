"""用匯出的 INT8 .tflite 對一段 WAV 做一次推論，印出每一類的信心分數。

與匯出 ZIP 內的 run_model.py 同一套算式（tm_local/known_sound_pipeline.py 的
write_known_sound_runner()）：每 0.48 秒一張 YAMNet patch（每張涵蓋 0.96 秒），取各類在整段音訊上的最大值。

這是 multi-label 模型：每一類都是各自獨立的 sigmoid，分數不會加總成 100%。
把 training_report.json 傳給 --report，就會用訓練當下的每類門檻來判定。

    .venv\\Scripts\\python.exe reference\\known_sound\\run_tflite.py ^
        --model known_sound_yamnet_classifier_int8.tflite --labels labels.txt --input dog.wav
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
        print_scores,
        quantize_input,
        read_labels,
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="known_sound_yamnet_classifier*_int8.tflite")
    parser.add_argument("--labels", help="匯出 ZIP 內的 labels.txt")
    parser.add_argument("--input", help="要辨識的 .wav")
    parser.add_argument("--report", help="training_report.json；用來取得每類判定門檻")
    parser.add_argument("--dry-run", action="store_true", help="只印張量資訊，不需要 --input")
    args = parser.parse_args(argv)

    interpreter, input_detail, output_detail = load_interpreter(Path(args.model))
    describe_tensors(input_detail, output_detail)
    if args.dry_run:
        print("輸入是 YAMNet log-mel patch（96 x 64），輸出是每類獨立的 sigmoid 信心分數。")
        return 0
    if not args.input:
        print("需要 --input（一段 .wav），或改用 --dry-run", file=sys.stderr)
        return 2

    import numpy as np

    from tm_local.audio_frontend import read_wav_file
    from tm_local.known_sound_pipeline import thresholds_for_labels
    from tm_local.yamnet_model import waveform_to_log_mel_patches

    labels = read_labels(Path(args.labels) if args.labels else None)
    _rate, signal = read_wav_file(Path(args.input), 16000)
    patches = waveform_to_log_mel_patches(np.asarray(signal, dtype=np.float32))
    rows = []
    for patch in patches:
        value = patch[None, ...].astype(np.float32)
        interpreter.set_tensor(input_detail["index"], quantize_input(value, input_detail))
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0]
        rows.append(dequantize_output(raw, output_detail))
    # 取整段的最大值：只響 150 ms 的敲擊聲若取平均就被稀釋掉了。
    scores = np.max(np.stack(rows, axis=0), axis=0)

    thresholds = None
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        thresholds = thresholds_for_labels(report.get("settings") or {}, len(labels))
    print(f"{len(patches)} 張 patch，取每類在整段音訊上的最大值：")
    print_scores(labels, scores, independent=True, thresholds=thresholds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
