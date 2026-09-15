"""Drive one project through Export -> Vela -> assemble -> GCC -> packaged firmware.

Runs inside the JobManager's single worker thread. Everything student-facing leaves here as a
Traditional-Chinese `DeployError` naming the setting to change; every intermediate file lives in
a throwaway work dir under `paths.MCU_TEMP_ROOT` that is deleted on every exit path (the build
log is copied out first so a failed deploy is still diagnosable).

Import stays cheap -- no TensorFlow, no yaml: `export_service` is imported lazily inside the
`ensure_export_artifacts()` shim below, which also keeps that name patchable on this module.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import time
import uuid
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import MCU_APPLICATION_BY_KIND, MCU_DEPLOYABLE_KINDS, mcu_application, normalize_kind
from ..utils import atomic_write_json, safe_identifier, sha256_file, utc_now_iso
from . import contract as contract_mod
from . import gcc_build, project_builder, vela
from . import paths as mcu_paths  # module reference: tests monkeypatch MCU_TEMP_ROOT
from .boards import Board, board_for
from .errors import TOOLCHAIN_MISSING_MESSAGE, DeployError, long_path_reason
from .reports import REPORT_NAME, deploy_dir, read_deploy_reports
from .toolchain import find_toolchain

__all__ = [
    "DEPLOY_TIMEOUT_SECONDS",
    "REPORT_NAME",
    "DeployOutcome",
    "deploy_dir",
    "ensure_export_artifacts",
    "flash_instructions",
    "read_deploy_reports",
    "run_deploy",
]

ProgressFn = Callable[[float, str], None]
DEPLOY_TIMEOUT_SECONDS = max(300, int(os.environ.get("TM_LOCAL_DEPLOY_TIMEOUT", "1200")))
REPORT_SCHEMA_VERSION = 1
ZIP_MEMBERS = (
    "firmware.bin",
    "firmware.elf",
    "firmware.map",
    "build.log",
    REPORT_NAME,
    "labels.txt",
    "README_FLASH_zh-TW.txt",
    "source.zip",
)
_LOG_TAIL_CHARS = 3000
LOGGER = logging.getLogger("tm_local.mcu.deploy")

# GNU ld reports an over-full memory region in one of two shapes, both naming the region:
#   ld.exe: region `FLASH' overflowed by 123456 bytes
#   ld.exe: <elf> section `.sram01_hyperram' will not fit in region `SRAM01_HYPERRAM'
# The linker script declares five regions (ITCM, FLASH, SRAM2, DTCM, SRAM_NONCACHEABLE/
# SRAM01_HYPERRAM), so the region name -- not merely the presence of the word "overflowed" --
# decides which resource the student actually ran out of.
_REGION_OVERFLOWED = re.compile(
    r"region\s+[`'\"]?(?P<name>\w+)['\"]?\s+overflowed by\s+(?P<bytes>\d+)\s+bytes"
)
_REGION_WONT_FIT = re.compile(r"will not fit in region\s+[`'\"]?(?P<name>\w+)")
_SRAM01_REGIONS = {"SRAM01", "SRAM01_HYPERRAM", "SRAM_NONCACHEABLE"}
_FLASH_REGIONS = {"FLASH"}

# Flash the firmware costs WITHOUT its model, measured on this branch (2026-09-12) from the
# FINAL tree as the built firmware.bin minus the embedded `*_vela.tflite`, on both boards:
#   known_sound  153,144 B (X) / 157,712 B (GestureAI)
#   image        160,636 B (X) / 161,428 B (GestureAI)
#   kws          241,648 B (X) / 246,184 B (GestureAI)   <- the largest, and the constant below
# Rounded up to 256 KiB so `_check_flash_budget()` below never waves through a model that then
# blows up in the linker two minutes later. It is deliberately an over-estimate of the code and
# an under-estimate of nothing: Vela's `flash_bytes` is itself ~0.4% below the embedded array's
# real size, and the rounding absorbs that too.
#
# This flat 256 KiB is what the gate below actually enforces, for every kind and both boards;
# the per-app rows above are only the evidence that it is an over-estimate.
# `config.KNOWN_SOUND_BOARD_MAX_DEPTH` was derived from the same measurements (with a 64 KiB
# margin instead of this rounding) and lands on the same verdict for the X board's 2 MiB:
# depth 12 passes here (1,563,552 + 262,144 = 1,825,696), depth 13 does not
# (2,066,832 + 262,144 = 2,328,976).
FIRMWARE_CODE_BASELINE_BYTES = 256 * 1024


@dataclass
class DeployOutcome:
    board: str
    bin_path: Path
    zip_path: Path
    report: dict[str, Any]


def ensure_export_artifacts(
    store: Any, project_id: str, formats: Iterable[str], progress: ProgressFn | None = None
) -> dict[str, Any]:
    """Lazy shim over `export_service.ensure_export_artifacts` (which imports TensorFlow).

    Importing `tm_local.mcu.deploy_service` must stay free of the training stack, and tests
    replace this name on the module to run a deploy without TensorFlow.
    """
    from ..export_service import ensure_export_artifacts as _impl

    return _impl(store, project_id, formats, progress)


def _result_text(board: Board, kind: str) -> str:
    """"What you should see" after flashing -- which is NOT the same for the three apps.

    Only the image app has a picture: an LCD label on the X board, a USB camera window with
    an overlaid label on the GestureAI board. The two audio apps have no picture at all --
    GestureAI enumerates a plain virtual COM port (PID 0x1105, no camera window), the X board
    prints to the Nu-Link VCOM UART and leaves the LCD alone -- so a student told to open the
    camera app is being told to stare at a black screen.
    """
    if kind == "image":
        if board.has_lcd:
            return "結果觀看：LCD 會顯示辨識標籤；Nu-Link 的虛擬 COM port（115200 8N1）會印出文字。\n"
        return (
            "結果觀看：Windows 相機 app 可看到 USB 攝影機影像與左上角的辨識標籤；"
            "裝置管理員會多一個 COM port，用 115200 8N1 可看到文字輸出。\n"
        )
    port = (
        "板上 Nu-Link 的虛擬 COM port"
        if board.has_lcd
        else "板子列舉出的 USB 虛擬 COM port（PID 0x1105；音訊韌體不會開相機，不會有影像視窗）"
    )
    if kind == "known_sound":
        expect = (
            "開機時每一類會印一行 INFO threshold[類別名] 0.50 -> output code >= …，"
            "之後每 0.5 秒一行 live rms   -42.1 dBFS | … | 類別名=0.123 …（每類一個獨立信心分數）。"
        )
    else:
        expect = (
            "之後每 0.25 秒一行 KWS hop=12 rms=-42.1 dBFS top=標籤 0.876 | …；"
            "判定成立時會多印一行 KWS DETECTED: label=…。"
        )
    return (
        "結果觀看：這個專案沒有畫面，只有文字。請在 Windows「裝置管理員 → 連接埠 (COM 和 LPT)」"
        f"找出{port}，用終端機軟體（PuTTY／Tera Term）以 115200 8N1 連線。\n"
        f"{expect}\n"
    )


def flash_instructions(board: Board, kind: str = "image", method: str | None = None) -> str:
    """Flashing steps for `board` plus what `kind`'s firmware then shows.

    `kind` defaults to "image" so the older two-argument-less callers (app.py's flash-failure
    fallback, which only ever explains the steps) keep working; `run_deploy()` always passes
    the project's real kind, and that is the text the report and the UI carry.

    `method` defaults to `board.flash_method` (the board's first/default method). Boards that
    support more than one way to flash (NuGestureAI-M55M1: MSC bootloader and Nu-Link) let the
    student pick, so this branches on the RESOLVED method, not on the board's default -- a
    preflight failure for the method actually requested must describe that method's steps, not
    always the default's.
    """
    resolved = method or board.flash_method
    if resolved == "msc":
        return (
            f"{board.label} 燒錄步驟：\n"
            "1. 按住板上 User 按鈕（PA.4），同時按一下 Reset，再放開 User。\n"
            "2. 電腦會出現名稱為 M55M1 的隨身碟。\n"
            "3. 把 firmware.bin 拖放到該隨身碟。\n"
            "4. 拖放完成後按一下 Reset，新韌體就會啟動。\n" + _result_text(board, kind)
        )
    if resolved == "nulink":
        return (
            f"{board.label} 燒錄步驟：\n"
            "1. 用 USB 線接板上的 Nu-Link 埠。\n"
            "2. 若電腦已安裝 Nu-Link Command Tool，Studio 的「燒錄」按鈕會直接寫入 APROM 並重置。\n"
            "3. 也可用 Keil 的 Download，或 Nu-Link Command Tool："
            "NuLink.exe -C、-W APROM firmware.bin 1、-S。\n" + _result_text(board, kind)
        )
    if resolved == "pyocd":
        return (
            f"{board.label} 燒錄步驟：\n"
            "1. 這塊板沒有板載 Nu-Link。把外接 Nu-Link2（CMSIS-DAP）接到板上的 J2"
            "（PF.0 ICE_DAT / PF.1 ICE_CLK）。\n"
            "2. 執行一次 scripts\\setup_voiceai_flash.py 安裝 pyocd 與晶片支援檔，"
            "之後按「燒錄」就會自動寫入。\n"
            "3. 或用 Keil 的 Download：Nu-Link 的 Chip Select 必須選 M5531，選 M55M1 會永遠跑不完。\n"
            "   （本工具包附的 NuLink Command Tool 不認得這顆晶片，會回 "
            "Target Chip is Not Supported，所以 Studio 不會用它。）\n" + _result_text(board, kind)
        )
    raise DeployError(f"不支援的燒錄方式：{resolved!r}")


def _staging_dir(target: Path) -> Path:
    """`models/mcu/<board>.tmp` -- where a rebuild is assembled before it replaces `<board>`.

    A sibling of the real folder so the final swap is a rename on the same volume, and named
    with a suffix `read_deploy_reports()` never mistakes for a board (it only keys folders that
    already hold a deploy_report.json, and this one only ever holds one for the microseconds
    between the last write and the rename).
    """
    return target.with_name(f"{target.name}.tmp")


# --- Windows swap: rename, not rmtree-then-rename -------------------------------------------
# Measured on this machine (2026-09-14), reproducibly: the swap below can fail with
# `PermissionError: [WinError 5]`, and it is the SOURCE directory that is locked, not the
# destination. Renaming `staging` to a brand-new, never-used name fails identically, and
# `target.exists()` is False immediately after a `shutil.rmtree(target)`, so this is NOT
# Windows' delete-pending state on the destination -- an earlier fix assumed exactly that and
# renamed the old `target` aside first, which does not touch the actual cause and left
# `staging.rename(target)` still unprotected.
#
# The lock sits on exactly two files this module just finished writing inside `staging`
# (`source.zip` and the `<project>_<board>_firmware.zip` bundle) -- `CreateFileW(...,
# dwShareMode=0)` on those two (never on firmware.bin/.elf/.map/.hex) returns
# ERROR_SHARING_VIOLATION (32) because the on-access antivirus (Trend Micro Apex One on the
# machine this was measured on; Windows Defender was off) is unpacking them to scan, and holds
# the handle for 0.5-5 seconds after the file closes. Windows refuses to rename a directory
# while any file inside its subtree is open. A 10 ms retry loop cleared it after ~420 ms
# (30-32 attempts), consistently across many rounds -- but NOT deterministically on the first
# vs. second attempt: a first-ever deploy has been observed to succeed while another first-ever
# deploy fails, so this is a sub-second race against the scanner's queue, not a fixed pattern.
_RENAME_RETRY_INTERVAL_SECONDS = 0.01
_RENAME_RETRY_BUDGET_SECONDS = 10.0


def _rename_with_retry(
    src: Path, dst: Path, *, budget_seconds: float = _RENAME_RETRY_BUDGET_SECONDS
) -> None:
    """`src.rename(dst)`, retried through a transient Windows sharing violation.

    See the module comment above this function for the measured cause. The ~10 second default
    budget is comfortably past the ~420 ms worst case actually observed; raises the last error
    once the budget is exhausted.
    """
    deadline = time.monotonic() + budget_seconds
    while True:
        try:
            src.rename(dst)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_RENAME_RETRY_INTERVAL_SECONDS)


def _antivirus_lock_message(target: Path) -> str:
    return (
        f"部署已完成建置，但把新版韌體搬進「{target}」時被 Windows 拒絕存取（PermissionError）。"
        "最可能的原因是防毒軟體（例如 Trend Micro Apex One）正在掃描剛寫入的壓縮檔，"
        "暫時鎖住了檔案幾秒鐘；先前的韌體並未受影響。"
        "請直接再按一次「建置」重新部署即可；若持續發生，"
        "可到防毒軟體的即時掃描設定中，把 Studio 的 workspace 資料夾加入排除清單。"
    )


def _swap_staging_into_place(staging: Path, target: Path) -> None:
    """Replace `target` with `staging`, without ever losing the student's previous build.

    Order matters: the previous `target` is renamed aside to a retired sibling BEFORE the
    fallible rename that puts the new build in place -- never deleted first -- so a failure in
    that second rename still has something to restore. If `staging.rename(target)` fails (see
    `_rename_with_retry()` for why), the retired copy is renamed straight back to `target`, so a
    failed deploy leaves the student with exactly the firmware they had before it, and a Chinese
    `DeployError` is raised in place of the raw `PermissionError`.

    In the extremely unlikely case that even the restore fails, the retired copy is left on disk
    (never swept) and named in the `DeployError`, because it is the only surviving copy of the
    previous build -- the caller's cleanup must not delete it.
    """
    retired = target.with_name(f"{target.name}.old-{os.getpid()}")
    shutil.rmtree(retired, ignore_errors=True)
    target_existed = target.exists()
    if target_existed:
        _rename_with_retry(target, retired)
    try:
        _rename_with_retry(staging, target)
    except OSError as exc:
        if not target_existed:
            raise DeployError(_antivirus_lock_message(target)) from exc
        try:
            _rename_with_retry(retired, target)
        except OSError as restore_exc:
            LOGGER.error(
                "MCU 部署交換失敗且無法還原先前的韌體：%s 仍留在 %s，請手動改名回 %s",
                target, retired, target,
            )
            raise DeployError(
                "部署韌體時發生嚴重錯誤：新版韌體卡在暫存狀態，且系統一併拒絕把先前的韌體還原回"
                f"原位。先前的韌體並未遺失，目前留在資料夾「{retired}」，"
                f"請手動將它改名回「{target}」以取回，並把這個訊息回報給維護者。"
                "這通常也是防毒軟體暫時鎖住檔案所致，可先關閉即時掃描或稍候幾秒再重試。"
            ) from restore_exc
        raise DeployError(_antivirus_lock_message(target)) from exc
    else:
        if target_existed:
            shutil.rmtree(retired, ignore_errors=True)


def _tail(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL_CHARS:]
    except OSError:
        return ""


def _zip_dir(src: Path, out: Path, *, exclude_dirs: set[str]) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(src.rglob("*")):
            relative = path.relative_to(src)
            if any(part in exclude_dirs for part in relative.parts):
                continue
            if path.is_file():
                archive.write(path, relative.as_posix())


def _sub(progress: ProgressFn, start: float, end: float) -> ProgressFn:
    def forward(value: float, message: str) -> None:
        progress(start + (end - start) * max(0.0, min(1.0, float(value))), message)

    return forward


def parse_link_overflow(text: str) -> tuple[str, int | None] | None:
    """`(region_name, overflow_bytes | None)` for the first over-full region ld reported."""
    match = _REGION_OVERFLOWED.search(text)
    if match:
        return match.group("name"), int(match.group("bytes"))
    match = _REGION_WONT_FIT.search(text)
    if match:
        return match.group("name"), None
    return None


def _shrink_advice(kind: str, board: Board) -> str:
    """The knob THIS student can actually turn to make the firmware smaller.

    The three apps have three different ones, and an image-only sentence sends a Known Sound
    student hunting for an `image_size` setting their project does not have. Ends without
    punctuation so callers can place it inside their own sentence.
    """
    if kind == "known_sound":
        cap = board.known_sound_max_depth
        return (
            f"請把 encoder_depth 降到 {cap} 以下（{board.label} 的上限就是 {cap}）"
            "後重新訓練與匯出，或改用其他開發板"
        )
    if kind == "image":
        return (
            "請降低 image_size／MobileNet alpha，或減少類別數後重新訓練與匯出，"
            "或改用其他開發板"
        )
    # audio (KWS): the model is a few KB of the ~250 KB firmware, so no project setting will
    # shrink this meaningfully -- something in the firmware itself grew, and that is a bug.
    return (
        "audio 專案的模型只有幾 KB，超出的是韌體程式碼本身，調整專案設定不會讓它變小；"
        r"請把 logs\LATEST.log 與這段訊息回報給維護者"
    )


def _check_flash_budget(vela_result: vela.VelaResult, board: Board, kind: str) -> None:
    """Refuse an over-sized model straight after Vela, BEFORE the minutes-long GCC build.

    Both boards execute from the same 2 MiB internal flash, so whether the firmware will fit
    is already decidable here: the Vela-compiled model plus `FIRMWARE_CODE_BASELINE_BYTES`.
    Without this, a default-depth Known Sound project spends two minutes compiling only to
    fail in `ld` -- in English, about a region name, with no idea which setting to change.
    """
    needed = int(vela_result.flash_bytes) + FIRMWARE_CODE_BASELINE_BYTES
    if needed <= board.internal_flash_bytes:
        return
    raise DeployError(
        f"模型經 Vela 編譯後佔用 {vela_result.flash_bytes:,} bytes，加上韌體程式碼約 "
        f"{FIRMWARE_CODE_BASELINE_BYTES:,} bytes，超過 {board.label} 的內部 flash "
        f"（{board.internal_flash_bytes:,} bytes）。{_shrink_advice(kind, board)}。"
    )


def _raise_build_failure(
    result: gcc_build.BuildResult, board: Board, arena_bytes: int, kind: str
) -> None:
    tail = _tail(result.log_path)
    overflow = parse_link_overflow(tail)
    if overflow is None:
        raise DeployError(f"{result.message}\n{tail}")
    region, over_by = overflow
    over_text = f"，超出 {over_by:,} bytes" if over_by is not None else ""
    if region in _SRAM01_REGIONS:
        # The SRAM01 window (1 MiB) holds both the non-cacheable buffers and the tensor arena,
        # so an over-sized model fails at LINK rather than at Vela's arena-budget check.
        raise DeployError(
            f"韌體連結失敗：模型的 tensor arena 需要 {arena_bytes:,} bytes，加上非快取緩衝區後"
            f"超過 {board.label} 的 SRAM01 空間（{board.sram01_bytes:,} bytes）{over_text}。"
            f"{_shrink_advice(kind, board)}。\n{tail}"
        )
    if region in _FLASH_REGIONS:
        raise DeployError(
            f"韌體連結失敗：程式碼加上模型權重超過 {board.label} 的 flash "
            f"（{board.internal_flash_bytes:,} bytes）{over_text}。"
            f"{_shrink_advice(kind, board)}。\n{tail}"
        )
    raise DeployError(f"{result.message}（記憶體區段 {region}）\n{tail}")


def _check_limits(result: gcc_build.BuildResult, board: Board, kind: str) -> None:
    if result.flash_used > board.internal_flash_bytes:
        raise DeployError(
            f"韌體 {result.flash_used:,} bytes 超過 {board.label} 的 flash "
            f"{board.internal_flash_bytes:,} bytes；{_shrink_advice(kind, board)}"
        )
    # The linker script maps .sram01_hyperram and the non-cacheable buffers onto the same 1 MiB
    # window, so ld already refuses a combined overflow; this is the post-link belt-and-braces.
    if result.sram01_used > board.sram01_bytes:
        raise DeployError(
            f"SRAM01 使用 {result.sram01_used:,} bytes 超過 {board.label} 的 "
            f"{board.sram01_bytes:,} bytes；{_shrink_advice(kind, board)}"
        )


def _build_report(
    contract: Any,
    board: Board,
    toolchain: gcc_build.Toolchain,
    vela_result: vela.VelaResult,
    result: gcc_build.BuildResult,
    bin_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "board": board.name,
        "kind": contract.kind,
        "application": contract.application,
        "built_at": utc_now_iso(),
        "toolchain": {"bin_dir": str(toolchain.bin_dir), "version": toolchain.version},
        "vela": {
            "optimise": vela_result.optimise,
            "sram_bytes": vela_result.sram_bytes,
            "flash_bytes": vela_result.flash_bytes,
            "arena_bytes": vela_result.arena_bytes,
            "npu_ops": vela_result.npu_ops,
            "cpu_ops": vela_result.cpu_ops,
        },
        "model": {"file": contract.int8_path.name, "sha256": contract.model_sha256},
        "labels": list(contract.labels),
        "image": {
            "text": result.text,
            "data": result.data,
            "bss": result.bss,
            "flash_used": result.flash_used,
            "flash_limit": board.internal_flash_bytes,
            "sram01_used": result.sram01_used,
            "sram01_limit": board.sram01_bytes,
            "noncacheable_used": result.noncacheable_used,
        },
        "bin": {
            "file": "firmware.bin",
            "bytes": bin_path.stat().st_size,
            "sha256": sha256_file(bin_path),
        },
        "flash_method": board.flash_method,
        "flash_methods": list(board.flash_methods),
        # Carried in the report (and therefore in the job result, project.deploy[board] and the
        # ZIP) so the UI never drifts from flash_instructions(): web/app.js renders this text
        # and only falls back to its own copy for reports that predate the field.
        "flash_instructions": flash_instructions(board, contract.kind),
        "seconds": round(result.seconds, 1),
    }


def run_deploy(store: Any, project_id: str, board_name: str, progress: ProgressFn) -> DeployOutcome:
    """Build and package firmware for `board_name`; returns where the artifacts landed.

    Progress bands: 0.00-0.35 Strict INT8 export, 0.35-0.40 contract + toolchain,
    0.40-0.50 Vela, 0.50-0.55 assemble, 0.55-0.95 GCC, 0.95-1.00 checks and packaging.
    """
    board = board_for(board_name)
    # Both refusals are cheap and come first, because everything after them costs minutes:
    # a Studio installed too deep for Windows MAX_PATH (gcc fails on the deepest BSP header
    # with an English "No such file or directory"), and a kind whose firmware app module does
    # not exist in this build (project_builder.assemble() raises the very same message, but
    # only after a full Strict INT8 export and a Vela compile).
    too_long = long_path_reason(mcu_paths.MCU_TOOLKIT_ROOT)
    if too_long:
        raise DeployError(too_long)
    kind = normalize_kind((store.get_raw_project(project_id) or {}).get("kind"))
    application = mcu_application(kind)
    if application is None:
        # Named kinds are filtered through supported_applications() -- the same check
        # project_builder.app_module() below performs for a *registered* kind whose app module
        # has not landed yet -- so this never lists a kind the student cannot actually deploy.
        # safe=True: a real app module raising anything other than DeployError here must not
        # crash this refusal message into an unhandled 500 (Round 2 defect).
        supported_apps = set(project_builder.supported_applications(safe=True))
        supported_kinds = sorted(
            k for k in MCU_DEPLOYABLE_KINDS if MCU_APPLICATION_BY_KIND[k] in supported_apps
        )
        raise DeployError(f"{kind} 專案不支援部署到開發板（只支援 {'、'.join(supported_kinds)}）")
    project_builder.app_module(application)

    progress(0.0, "檢查 Strict INT8 匯出…")
    ensure_export_artifacts(store, project_id, ["int8"], _sub(progress, 0.0, 0.35))
    progress(0.35, "讀取部署契約…")
    contract = contract_mod.collect(store, project_id, board)
    contract_mod.validate(contract)
    toolchain = find_toolchain()
    if toolchain is None:
        raise DeployError(TOOLCHAIN_MISSING_MESSAGE)

    # Resolved before the work dir exists: deploy_dir() validates the board name and can raise,
    # and a raise between mkdir() and the try/finally would leak the directory.
    target = deploy_dir(contract.models_dir, board.name)
    work = Path(mcu_paths.MCU_TEMP_ROOT) / uuid.uuid4().hex[:8]
    work.mkdir(parents=True, exist_ok=True)
    try:
        progress(0.40, "Vela 編譯中…")
        vela_result = vela.compile_model(
            contract.int8_path, work / "vela", progress=_sub(progress, 0.40, 0.50)
        )
        # Decidable now, and only now: Vela has said what the model costs in flash. Doing it
        # here rather than after the build turns a two-minute wait plus an English linker
        # error into an immediate Chinese sentence naming this kind's setting.
        _check_flash_budget(vela_result, board, contract.kind)
        progress(0.50, "組裝韌體專案…")
        assembled = project_builder.assemble(contract, vela_result, work)
        progress(0.55, "GCC 編譯中…")
        result = gcc_build.build(
            assembled.manifest,
            toolchain,
            progress=_sub(progress, 0.55, 0.95),
            timeout_seconds=DEPLOY_TIMEOUT_SECONDS,
        )
        # The new firmware is assembled in a sibling `<board>.tmp` and swapped in only once
        # everything succeeded: a failed rebuild must leave the previous (working) firmware,
        # ZIP and report exactly where the student left them. Only build.log is overwritten in
        # place on a failure -- it is what diagnoses *this* attempt.
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.log_path, target / "build.log")
        if not result.ok:
            _raise_build_failure(result, board, vela_result.arena_bytes, contract.kind)
        _check_limits(result, board, contract.kind)

        progress(0.96, "打包產物…")
        staging = _staging_dir(target)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        bin_path = staging / "firmware.bin"
        shutil.copy2(result.log_path, staging / "build.log")
        shutil.copy2(result.bin, bin_path)
        shutil.copy2(result.elf, staging / "firmware.elf")
        shutil.copy2(result.map, staging / "firmware.map")
        shutil.copy2(result.hex, staging / "firmware.hex")
        (staging / "labels.txt").write_text(
            "".join(f"{i} {name}\n" for i, name in enumerate(contract.labels)),
            encoding="utf-8",
            newline="\n",
        )
        (staging / "README_FLASH_zh-TW.txt").write_text(
            flash_instructions(board, contract.kind), encoding="utf-8", newline="\n"
        )
        _zip_dir(assembled.app_dir, staging / "source.zip", exclude_dirs={"build"})

        report = _build_report(contract, board, toolchain, vela_result, result, bin_path)
        # deploy_report.json is written LAST, from the same staged copy the ZIP carries, because
        # its presence is what marks this board's deploy complete (read_deploy_reports() keys on
        # it). A failure while writing the ZIP must not leave a report advertising a missing file.
        staged_report = work / REPORT_NAME
        atomic_write_json(staged_report, report)
        zip_name = f"{safe_identifier(contract.project_name)}_{board.name}_firmware.zip"
        with zipfile.ZipFile(staging / zip_name, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ZIP_MEMBERS:
                archive.write(staged_report if name == REPORT_NAME else staging / name, name)
        atomic_write_json(staging / REPORT_NAME, report)
        # See `_swap_staging_into_place()` above: renames the previous `target` aside before the
        # fallible rename that puts `staging` in place, and restores it if that fails, so a
        # transient Windows sharing violation (antivirus scanning the archives just written into
        # `staging`) can never cost the student the build they already had.
        _swap_staging_into_place(staging, target)
        progress(1.0, "部署產物已完成")
        return DeployOutcome(
            board=board.name,
            bin_path=target / "firmware.bin",
            zip_path=target / zip_name,
            report=report,
        )
    finally:
        # `_staging_dir(target)` no longer exists once `_swap_staging_into_place()` has renamed
        # it into `target`, so this is a no-op on every success path (and after a failure that
        # was cleanly rolled back); it only ever deletes real garbage -- a staging dir left by a
        # failure earlier in this function, or one whose swap failed and was rolled back.
        shutil.rmtree(_staging_dir(target), ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
        if work.exists():
            # ignore_errors swallowed something -- usually a file still held open by antivirus
            # or an editor. Name the path so the leftover is diagnosable from logs/LATEST.log.
            LOGGER.warning("無法刪除 MCU 部署暫存資料夾：%s", work)
        # A crash between the two renames inside `_swap_staging_into_place()` (an EARLIER deploy
        # of this board, killed mid-swap) would strand `<board>.old-*` next to `<board>` forever;
        # sweep it here so every ordinary exit path leaves at most the current build on disk.
        # Guarded on `target.exists()`: when it does not, `_swap_staging_into_place()` just
        # failed catastrophically (both the forward rename AND the restore failed) and
        # deliberately left a retired copy on disk as the student's last surviving build -- named
        # in the DeployError it raised -- so sweeping indiscriminately here would delete the very
        # thing that message just told them to go recover by hand.
        if target.exists() and target.parent.is_dir():
            for leftover in sorted(target.parent.glob(f"{target.name}.old-*")):
                shutil.rmtree(leftover, ignore_errors=True)
