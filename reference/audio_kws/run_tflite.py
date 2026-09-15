"""用匯出的 INT8 .tflite 對一段 WAV 做一次推論，並印出每類分數。

與匯出 ZIP 內的 run_model.py 同一套算式（tm_local/audio_pipeline.py 的
write_audio_runner()）：先用 audio_frontend 做 log-mel，再依 scale/zero_point 量化。
開發板上的 kws 韌體做的是同一件事，只是前處理換成 C（見 mcu_tables.py）。

    .venv\\Scripts\\python.exe reference\\audio_kws\\run_tflite.py ^
        --model audio_classifier_spectrogram_int8.tflite --labels labels.txt --input yes.wav
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
    parser.add_argument("--model", required=True, help="audio_classifier_spectrogram_int8.tflite")
    parser.add_argument("--labels", help="匯出 ZIP 內的 labels.txt")
    parser.add_argument("--input", help="要辨識的 .wav")
    parser.add_argument("--frontend", help="匯出 ZIP 內的 audio_frontend.json")
    parser.add_argument("--dry-run", action="store_true", help="只印張量資訊，不需要 --input")
    args = parser.parse_args(argv)

    interpreter, input_detail, output_detail = load_interpreter(Path(args.model))
    describe_tensors(input_detail, output_detail)
    if args.dry_run:
        print("輸入是 log-mel 張量（frames, mel_bins, 1），不是原始 PCM。")
        return 0
    if not args.input:
        print("需要 --input（一段 .wav），或改用 --dry-run", file=sys.stderr)
        return 2

    from tm_local.audio_frontend import (
        AudioFrontendConfig,
        config_from_mapping,
        log_mel_spectrogram,
        read_wav_file,
    )

    if args.frontend:
        config = config_from_mapping(json.loads(Path(args.frontend).read_text(encoding="utf-8")))
    else:
        config = AudioFrontendConfig()
    labels = read_labels(Path(args.labels) if args.labels else None)
    _rate, signal = read_wav_file(Path(args.input), config.sample_rate)
    value = log_mel_spectrogram(signal, config)[None, ...]
    interpreter.set_tensor(input_detail["index"], quantize_input(value, input_detail))
    interpreter.invoke()
    scores = dequantize_output(interpreter.get_tensor(output_detail["index"])[0], output_detail)
    print_scores(labels, scores, independent=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
