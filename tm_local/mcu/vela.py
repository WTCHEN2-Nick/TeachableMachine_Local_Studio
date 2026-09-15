from __future__ import annotations

import csv
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .errors import DeployError
from .gcc_build import Toolchain, run_tool, sanitized_env
from .paths import VELA_EXE, VELA_INI

DEFAULT_ARENA_BUDGET_BYTES = 716800
ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class VelaResult:
    vela_tflite: Path
    summary_csv: Path
    sram_bytes: int
    flash_bytes: int
    arena_bytes: int
    optimise: str
    npu_ops: int
    cpu_ops: int
    log_tail: str


def arena_size_from_sram(sram_bytes: int) -> int:
    """Tensor arena size the firmware must reserve for a given Vela SRAM estimate.

    Verbatim port of NuML's own formula so `ACTIVATION_BUF_SZ` matches what NuML's tool
    would provision for the same model: 20% headroom, +1024-byte pad, then FLOOR to the
    next 1024-byte boundary. See `add_activation_size_section()` in
    `mcu_toolkit/NuML_TFLM_Tool/generic_codegen/main_cpp_codegen.py:9-12` (the
    `imgclass_codegen` copy is byte-identical). Do not "fix" this to round up -- NuML
    floors on purpose and this must match it exactly.
    """
    return (int(int(sram_bytes) * 1.2) + 1024) & ~1023


def vela_command(
    model: Path, out_dir: Path, optimise: str, arena_budget_bytes: int | None
) -> list[str]:
    """Build the Vela CLI invocation. `out_dir` is only used as the subprocess cwd by the
    caller (never embedded as an absolute path here) -- `--output-dir=.` writes into it."""
    cmd = [
        str(VELA_EXE),
        str(model),
        "--accelerator-config=ethos-u55-256",
        f"--optimise={optimise}",
        f"--config={VELA_INI}",
        "--memory-mode=Shared_Sram",
        "--system-config=Ethos_U55_High_End_Embedded",
        "--verbose-cycle-estimate",
    ]
    if arena_budget_bytes:
        cmd += ["--arena-cache-size", str(int(arena_budget_bytes))]
    cmd.append("--output-dir=.")
    return cmd


def parse_summary(csv_path: Path) -> tuple[int, int]:
    """Read Vela's `<stem>_summary_<system_config>.csv` and return (sram_bytes, flash_bytes).

    Raises `DeployError` (never a raw csv/KeyError/ValueError) if the file is empty,
    missing a data row, missing one of the two expected columns, or has a non-numeric cell
    -- any of which means Vela's summary format changed underneath us.
    """
    csv_path = Path(csv_path)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        sram_bytes = int(float(row["sram_memory_used"]) * 1024)
        flash_bytes = int(float(row["off_chip_flash_memory_used"]) * 1024)
    except StopIteration as exc:
        raise DeployError(f"Vela 摘要檔格式不符：{csv_path}（檔案是空的或缺少資料列）") from exc
    except KeyError as exc:
        raise DeployError(f"Vela 摘要檔格式不符：{csv_path}（缺少欄位 {exc.args[0]}）") from exc
    except ValueError as exc:
        raise DeployError(f"Vela 摘要檔格式不符：{csv_path}（欄位不是數字：{exc}）") from exc
    return sram_bytes, flash_bytes


def _find_output(out_dir: Path, stem: str) -> tuple[Path | None, Path | None]:
    """Locate Vela's `<stem>_vela.tflite` and `<stem>_summary_*.csv` robustly by globbing
    on the model stem, rather than assuming the exact `<system_config>` suffix."""
    vela_tflite = next(iter(sorted(out_dir.glob(f"{stem}_vela.tflite"))), None)
    summary_csv = next(iter(sorted(out_dir.glob(f"{stem}_summary_*.csv"))), None)
    return vela_tflite, summary_csv


def _count_ops(vela_tflite: Path) -> tuple[int, int, str]:
    """Count ethos-u NPU custom ops vs remaining CPU ops in subgraph 0.

    Uses the TFLite flatbuffers schema bundled with the project's own TensorFlow
    dependency (`tensorflow.lite.python.schema_py_generated`) rather than
    `tf.lite.Interpreter`, because the interpreter's op resolver cannot load a
    Vela-compiled model (it does not know the "ethos-u" custom op) and would raise
    before we ever get to inspect it.

    Op counts are informational only (shown to the student, never load-bearing for the
    deploy decision), so any failure here -- the private schema module moving, a subgraph
    shape we don't expect -- returns `(0, 0, <note>)` instead of failing the whole compile.
    Returns `(npu_ops, cpu_ops, note)`; `note` is `""` on success.
    """
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb

        model = schema_fb.Model.GetRootAsModel(vela_tflite.read_bytes(), 0)
        graph = model.Subgraphs(0)
        npu = cpu = 0
        for i in range(graph.OperatorsLength()):
            code = model.OperatorCodes(graph.Operators(i).OpcodeIndex())
            custom = code.CustomCode()
            if custom and custom.decode("utf-8", "replace") == "ethos-u":
                npu += 1
            else:
                cpu += 1
        return npu, cpu, ""
    except Exception as exc:  # noqa: BLE001 - informational only, must never fail the deploy
        return 0, 0, f"（無法統計 NPU/CPU op 數量：{exc}）"


def compile_model(
    int8_model: Path,
    out_dir: Path,
    *,
    arena_budget_bytes: int = DEFAULT_ARENA_BUDGET_BYTES,
    timeout_seconds: int = 600,
    progress: ProgressFn | None = None,
) -> VelaResult:
    """Compile a strict-INT8 `.tflite` for the Ethos-U55 NPU with Vela.

    Tries `--optimise=Performance` first; if the resulting tensor arena would exceed
    `arena_budget_bytes`, retries once with `--optimise=Size`. Still over budget after
    that raises `DeployError` telling the student what to change.

    `timeout_seconds` is the *overall* budget across both the Performance run and the
    Size retry (mirrors `gcc_build.build()`'s `guarded_run` deadline), not a fresh budget
    handed to each subprocess call.
    """
    if not VELA_EXE.is_file():
        raise DeployError(f"找不到 Vela 編譯器 {VELA_EXE}；mcu_toolkit 不完整，請重新執行 01_INSTALL.bat")
    model = Path(int8_model).resolve()
    if not model.is_file():
        raise DeployError(f"找不到要編譯的 TFLite 模型：{model}")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = sanitized_env(Toolchain(bin_dir=VELA_EXE.parent, version="vela"))
    stem = model.stem
    last_log = ""
    arena = 0
    deadline = time.monotonic() + timeout_seconds

    def guarded_run(cmd: list[str]) -> subprocess.CompletedProcess:
        """`run_tool()` honoring the overall deadline, not a fresh per-call budget: if the
        deadline has already passed before this attempt even starts, fail it as a timeout
        immediately instead of handing `subprocess.run()` a manufactured positive value."""
        time_left = deadline - time.monotonic()
        if time_left <= 0:
            raise subprocess.TimeoutExpired(cmd, timeout_seconds)
        return run_tool(cmd, out_dir, env, max(1, int(time_left)))

    for optimise in ("Performance", "Size"):
        if progress:
            progress(0.0 if optimise == "Performance" else 0.5, f"Vela（{optimise}）編譯中…")
        cmd = vela_command(model, out_dir, optimise, arena_budget_bytes)
        try:
            proc = guarded_run(cmd)
        except subprocess.TimeoutExpired as exc:
            raise DeployError(f"Vela 逾時（{timeout_seconds} 秒）") from exc
        last_log = proc.stdout[-6000:]
        if proc.returncode != 0:
            raise DeployError("Vela 編譯失敗：\n" + proc.stdout[-2000:])
        vela_tflite, summary = _find_output(out_dir, stem)
        if vela_tflite is None or summary is None:
            raise DeployError("Vela 沒有產生預期的輸出檔（_vela.tflite / summary csv）")
        sram, flash = parse_summary(summary)
        arena = arena_size_from_sram(sram)
        if arena <= arena_budget_bytes:
            npu, cpu, op_count_note = _count_ops(vela_tflite)
            log_tail = last_log + ("\n" + op_count_note if op_count_note else "")
            return VelaResult(vela_tflite, summary, sram, flash, arena, optimise, npu, cpu, log_tail)
    raise DeployError(
        f"模型的 tensor arena 需要 {arena:,} bytes，超過板子預算 {arena_budget_bytes:,} bytes；"
        "請降低 image_size／MobileNet alpha／encoder_depth 後重新訓練與匯出，或改用其他開發板"
    )
