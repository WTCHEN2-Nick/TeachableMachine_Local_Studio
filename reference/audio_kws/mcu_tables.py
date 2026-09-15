"""印出開發板韌體實際內嵌的 C 表：Hann 視窗與稀疏 mel 濾波器組。

表的來源就是訓練用的那一份（tm_local/audio_frontend.py 的 mel_filterbank()），由
tm_local/mcu/kws_codegen.py 的 kws_tables() 轉成韌體的排列方式，所以 PC 與 MCU 不可能
算出不同的頻譜。部署時 Studio 會自己重新產生一次，這支腳本只是讓你看得見內容。

    .venv\\Scripts\\python.exe reference\\audio_kws\\mcu_tables.py ^
        --frontend <匯出ZIP>\\audio_frontend.json --out kws_tables.h
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


def define(name: str, value: str) -> str:
    """韌體端的排版：巨集本體對齊到第 32 欄（與 mcu_toolkit 的 main.cpp.in 相同）。"""

    return f"#define {name:<22} ({value})"


def main(argv: list[str] | None = None) -> int:
    bootstrap()
    from _common import studio_root

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frontend", required=True, help="匯出 ZIP 內的 audio_frontend.json")
    parser.add_argument("--out", help="輸出 .h 檔；省略就直接印到畫面")
    args = parser.parse_args(argv)

    from tm_local.audio_frontend import config_from_mapping
    from tm_local.mcu.kws_codegen import DEFAULT_LOG_EPSILON, c_array, float_literal, kws_tables

    source = Path(args.frontend)
    config = config_from_mapping(json.loads(source.read_text(encoding="utf-8")))
    tables = kws_tables(config)
    header = (
        f"/* 由 reference/audio_kws/mcu_tables.py 從 {source.name} 產生"
        f"（Studio：{studio_root().name}）。\n"
        f" * {config.sample_rate} Hz，{config.frame_count} x {config.mel_bins}。來源："
        "tm_local/audio_frontend.py mel_filterbank()\n"
        " * 與 tm_local/mcu/kws_codegen.py kws_tables()。 */"
    )
    lines = [
        header,
        "#pragma once",
        "#include <stdint.h>",
        "",
        define("NUML_SAMPLE_RATE", f"{config.sample_rate}U"),
        define("NUML_CLIP_SAMPLES", f"{config.clip_samples}U"),
        define("NUML_FRAME_LENGTH", f"{config.window_samples}U"),
        define("NUML_FRAME_STEP", f"{config.hop_samples}U"),
        define("NUML_FRAME_COUNT", f"{config.frame_count}U"),
        define("NUML_FFT_LENGTH", f"{config.fft_size}U"),
        define("NUML_MEL_BINS", f"{config.mel_bins}U"),
        define("NUML_MEL_NNZ", f"{len(tables.mel_weights)}U"),
        define("NUML_DB_FLOOR", float_literal(config.db_floor)),
        define("NUML_LOG_EPSILON", float_literal(DEFAULT_LOG_EPSILON)),
        "",
        "/* 稀疏三角形 mel 濾波器：第 m 個 mel 蓋住頻譜 bin",
        " * [s_melStart[m], s_melStart[m] + s_melCount[m])，權重放在",
        " * s_melWeights[s_melOffset[m] ...]。 */",
        "static const uint16_t s_melStart[NUML_MEL_BINS] = {",
        c_array(tables.mel_start, str),
        "};",
        "static const uint16_t s_melCount[NUML_MEL_BINS] = {",
        c_array(tables.mel_count, str),
        "};",
        "static const uint16_t s_melOffset[NUML_MEL_BINS] = {",
        c_array(tables.mel_offset, str),
        "};",
        "static const float s_melWeights[NUML_MEL_NNZ] = {",
        c_array(tables.mel_weights, float_literal),
        "};",
        "/* 對稱 Hann（np.hanning），與訓練前處理同一條 */",
        "static const float s_hannTable[NUML_FRAME_LENGTH] = {",
        c_array(tables.hann.tolist(), float_literal),
        "};",
        "",
    ]
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(
            f"已存檔：{args.out}（{len(tables.mel_weights)} 個 mel 權重、"
            f"{tables.hann.size} 點 Hann）"
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
