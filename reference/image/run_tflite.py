"""用匯出的 INT8 .tflite 對一張圖片做一次推論，並印出每類分數。

與匯出 ZIP 內的 run_model.py 同一套算式（tm_local/image_pipeline.py 的
write_image_runner()）：一樣的縮放、一樣的量化 scale/zero_point。
開發板上的 imgclass 韌體吃的也是這個檔，只是把圖片換成相機畫面。

    .venv\\Scripts\\python.exe reference\\image\\run_tflite.py ^
        --model image_classifier_int8.tflite --labels labels.txt --input cat.jpg
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--model", required=True, help="image_classifier_int8.tflite")
    parser.add_argument("--labels", help="匯出 ZIP 內的 labels.txt")
    parser.add_argument("--input", help="要辨識的 .jpg／.png")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只印張量資訊（dtype／shape／量化參數），不需要圖片",
    )
    args = parser.parse_args(argv)

    interpreter, input_detail, output_detail = load_interpreter(Path(args.model))
    describe_tensors(input_detail, output_detail)
    image_size = int(input_detail["shape"][1])
    channels = int(input_detail["shape"][3])
    print(
        f"模型吃 {image_size}x{image_size}x{channels} 的影像"
        "（Studio 匯出的是 RGB 0-255 float，再依 scale 量化）。"
    )
    if args.dry_run:
        return 0
    if not args.input:
        print("需要 --input（一張圖片），或改用 --dry-run", file=sys.stderr)
        return 2

    from tm_local.image_pipeline import load_image_for_prediction

    labels = read_labels(Path(args.labels) if args.labels else None)
    value = load_image_for_prediction(Path(args.input), image_size)[None, ...]
    interpreter.set_tensor(input_detail["index"], quantize_input(value, input_detail))
    interpreter.invoke()
    scores = dequantize_output(interpreter.get_tensor(output_detail["index"])[0], output_detail)
    print_scores(labels, scores, independent=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
