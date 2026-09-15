"""Assemble a buildable firmware project from the vendored template + this project's model.

Pure assembly: copy the template, drop the Vela-compiled model and the generated sources in,
let the app module render its `main.cpp` and stage its board-specific files, then turn the
template's progen records into a `BuildManifest`. Nothing here runs a compiler or touches the
project store -- `deploy_service` owns the orchestration, progress and packaging.
"""
from __future__ import annotations

import importlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from . import codegen, records
from .boards import board_source_dir
from .contract import DeployContract
from .errors import DeployError
from .paths import BSP_ROOT, MCU_TOOLKIT_ROOT, project_name_for, template_dir
from .vela import VelaResult

LOGGER = logging.getLogger("tm_local.mcu.project_builder")

# tflite2cpp's naming, matching the spike build: it must NOT be `Model.cpp`, because the
# ml-embedded-evaluation-kit already compiles `api/common/source/Model.cc` -> `Model.o` and
# `BuildManifest.check_object_collisions()` would (rightly) refuse the duplicate object name.
MODEL_DATA_SOURCE = "NN_Model_INT8.tflite.cpp"

# The vendored imgclass templates hard-code a 1 MiB tensor arena as a compile-time `-D` in
# progen's `toolchain_settings.yaml` (the NuMaker entry even has a trailing space). Leaving it
# means every translation unit but `main.cpp` sees 0x100000; dropping it outright makes
# ml-embedded-evaluation-kit's BufAttributes.hpp `#warning` in each of them instead. So the
# manifest *replaces* it with the real Vela-derived arena, which is exactly what the rendered
# `main.cpp` `#undef`s and re-`#define`s -- the two always agree.
ARENA_DEFINE = "ACTIVATION_BUF_SZ"

_IGNORED_TEMPLATE_NAMES = {"KEIL", "__pycache__"}
_APP_MODULES = {"imgclass": "image", "kws": "kws", "known_sound": "known_sound"}


@dataclass
class AssembledProject:
    app_dir: Path
    project_name: str
    manifest: records.BuildManifest
    template_dir: Path


def app_module(application: str) -> ModuleType:
    """The `tm_local.mcu.apps.*` module that owns `main.cpp` and the record overlay."""
    name = _APP_MODULES.get(str(application))
    if name is None:
        raise DeployError(f"未知的韌體應用 {application!r}")
    target = f"{__package__}.apps.{name}"
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as exc:
        # Only the app module itself being absent is a "not implemented yet" case; a missing
        # dependency *inside* an existing app module must keep its own traceback.
        if exc.name != target:
            raise
        raise DeployError(f"此版本尚未支援 {application} 韌體應用") from exc


def application_supported(application: object) -> bool:
    """True when firmware support for `application` actually exists in this build.

    Cheap on purpose (a lazy `importlib.import_module` of a stdlib-only app module, no
    TensorFlow / Vela / toolchain): `config.is_mcu_deployable()` only says the *kind* has an
    application id, so this is what keeps a deploy whose `apps/<name>.py` is not written yet
    from running a full Strict INT8 export and a Vela compile before failing at assemble time.
    A kind becomes deployable again automatically the moment its app module lands.
    """
    try:
        app_module(str(application))
    except DeployError:
        return False
    return True


def supported_applications(*, safe: bool = False) -> list[str]:
    """Every firmware application id this build can actually assemble, registry order.

    `safe=True` wraps each application's `application_supported()` probe in its own broad
    `except Exception` -- not just the `DeployError` `application_supported()` itself catches.
    A real `apps/<name>.py` can fail with anything (a broken transitive import, say), and a
    caller building a student-facing "what can you deploy" message from this list must not
    let that raise straight through it (that was Round 2's defect: `deploy_service.run_deploy()`
    called the unsafe form here to build its refusal message, so a bad app module 500'd a
    deploy request instead of producing the 400 it was building). The default stays unsafe so
    a genuinely broken app module still fails loudly wherever this is called without asking
    for the safe form.
    """
    if not safe:
        return [name for name in _APP_MODULES if application_supported(name)]
    supported: list[str] = []
    for name in _APP_MODULES:
        try:
            if application_supported(name):
                supported.append(name)
        except Exception:
            LOGGER.debug("application_supported(%r) failed", name, exc_info=True)
    return supported


def _ignore_template(directory: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _IGNORED_TEMPLATE_NAMES or n.endswith(".bak")}


def _overlay_kwargs(
    module: ModuleType, contract: DeployContract, app_dir: Path, arena_bytes: int,
    board_dir: Path,
) -> dict:
    """Normalize an app module's overlay into `records.apply_overlay()` keyword arguments.

    App modules return `add_sources` AND `add_includes` relative to the app dir whenever they
    name something the module itself staged there (`Frontend/numl_dmic.c`, `Frontend` as an
    include root); both lists are joined here so `apply_overlay()` only ever sees real paths.
    An absolute entry (a BSP driver, the BSP's `StdDriver/src` include dir) passes through
    untouched. `apply_overlay()` de-duplicates the include list, so an app may repeat a
    directory the template already provides without pushing a second `-I` at every compile.

    A board that borrows another board's template (Task 3's VoiceAI borrowing GestureAI)
    contributes its own `extra_sources` / `remove_source_basenames` into these SAME two keys
    as the app module -- not a parallel path. That way `_check_removals_matched()`, which only
    ever looks at the merged list, refuses a board's unmatched removal exactly the way it
    already refuses an app's: by naming the file, not by silently doing nothing. A board that
    declares neither (today's two boards) leaves both keys byte-for-byte as the app produced.
    """
    overlay = dict(module.overlay(contract))
    unknown = set(overlay) - {"add_sources", "remove_source_basenames", "add_includes",
                              "add_defines", "remove_defines"}
    if unknown:
        raise DeployError(f"{contract.application} overlay 有未知欄位：{sorted(unknown)}")

    def _under_app(entries: object) -> list[Path]:
        paths = [Path(p) for p in entries or ()]
        return [p if p.is_absolute() else app_dir / p for p in paths]

    def _under_board(entries: object) -> list[Path]:
        paths = [Path(p) for p in entries or ()]
        return [p if p.is_absolute() else board_dir / p for p in paths]

    return {
        "add_sources": [
            *_under_app(overlay.get("add_sources")),
            *_under_board(contract.board.extra_sources),
        ],
        "remove_source_basenames": [
            *overlay.get("remove_source_basenames", []),
            *contract.board.remove_source_basenames,
        ],
        "add_includes": _under_app(overlay.get("add_includes")),
        # add_defines is applied last and replaces any same-named entry, so the template's
        # 0x100000 arena is gone either way; remove_defines states the intent explicitly.
        "add_defines": [
            *overlay.get("add_defines", []),
            *contract.board.extra_defines,
            f"{ARENA_DEFINE}={int(arena_bytes)}",
        ],
        "remove_defines": [
            *overlay.get("remove_defines", []),
            ARENA_DEFINE,
            *contract.board.remove_defines,
        ],
    }


def _check_removals_matched(
    before: list[records.SourceFile], removed: list[str], template: Path
) -> None:
    """A removal that matched nothing means the template moved the file we meant to drop.

    For the LCD-less board that is a link-breaking condition rather than a no-op: the staged
    `numl_cdc_retarget.c` textually `#include`s the BSP's `retarget.c`, so if the BSP copy is
    still compiled under a different name, `_write()`/`stdout_putchar()` are defined twice.
    """
    present = {s.path.name.lower() for s in before}
    missing = [name for name in removed if name.lower() not in present]
    if missing:
        raise DeployError(
            f"韌體樣板 {template} 已沒有要移除的來源檔 {missing}；mcu_toolkit 版本不相容，"
            "請重新執行 01_INSTALL.bat 取得相符的樣板"
        )


def assemble(
    contract: DeployContract, vela_result: VelaResult, work_root: Path
) -> AssembledProject:
    """Copy the template into `work_root` and return it plus its ready-to-build manifest."""
    module = app_module(contract.application)  # resolved first: fail before any copying
    tdir = template_dir(contract.board.numl_template_board, contract.application)
    project_name = project_name_for(contract.application)
    src_app = tdir / project_name
    if not src_app.is_dir():
        raise DeployError(f"找不到韌體樣板 {src_app}；mcu_toolkit 不完整，請重新執行 01_INSTALL.bat")

    app_dir = Path(work_root) / project_name
    if app_dir.exists():
        shutil.rmtree(app_dir)
    shutil.copytree(src_app, app_dir, ignore=_ignore_template)

    gcc_dir = app_dir / "GCC"
    gcc_dir.mkdir(exist_ok=True)
    linker_source = tdir / "link_script" / "gcc" / "gcc.ld"
    if not linker_source.is_file():
        raise DeployError(f"找不到 linker script {linker_source}")
    shutil.copy2(linker_source, gcc_dir / "gcc.ld")

    model_dir = app_dir / "Model"
    model_dir.mkdir(exist_ok=True)
    for stale in sorted(model_dir.glob("*.tflite*")):
        stale.unlink()
    codegen.write_model_cpp(vela_result.vela_tflite.read_bytes(), model_dir / MODEL_DATA_SOURCE)
    # NuML's codegen also writes its own main.cpp and a numeric-placeholder Labels.cpp; the
    # app module's prepare() and write_labels() below deliberately run after it and win.
    codegen.generate_model_sources(
        contract.application, vela_result.vela_tflite, app_dir, vela_result.summary_csv
    )
    codegen.write_labels(list(contract.labels), model_dir)
    module.prepare(contract, vela_result, app_dir)

    manifest = records.load_records(tdir, app_dir, BSP_ROOT)
    overlay = _overlay_kwargs(
        module, contract, app_dir, vela_result.arena_bytes,
        board_source_dir(MCU_TOOLKIT_ROOT, contract.board.name),
    )
    _check_removals_matched(manifest.sources, overlay["remove_source_basenames"], tdir)
    manifest = records.apply_overlay(manifest, **overlay)
    manifest.check_object_collisions()
    return AssembledProject(
        app_dir=app_dir, project_name=project_name, manifest=manifest, template_dir=tdir
    )
