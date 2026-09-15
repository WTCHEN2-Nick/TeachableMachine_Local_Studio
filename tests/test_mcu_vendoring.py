from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tm_local.mcu import paths

ROOT = Path(__file__).resolve().parents[1]
TOOLKIT = ROOT / "mcu_toolkit"

pytestmark = pytest.mark.skipif(not paths.toolkit_available(), reason="mcu_toolkit not vendored")


def _rel_files() -> list[Path]:
    # __pycache__ directories and *.pyc files are excluded from this walk entirely: they are
    # transient bytecode-cache noise a stray import can leave behind on a student machine (e.g.
    # tm_local.mcu.codegen importing the vendored NuML codegen packages), not a vendoring defect,
    # so they must never fail the forbidden-suffix check below or the manifest's byte-exact
    # file-set comparison in test_vela_and_manifest_present.
    return [
        p
        for p in TOOLKIT.rglob("*")
        if p.is_file() and p.suffix.lower() != ".pyc" and "__pycache__" not in p.parts
    ]


def test_no_forbidden_files_in_toolkit() -> None:
    bad = [p for p in _rel_files() if p.suffix.lower() in {".bat", ".uvprojx", ".uvoptx", ".scatter"}]
    assert bad == []
    assert not (TOOLKIT / "NuML_TFLM_Tool" / "templates" / "M55M1BSP" / ".git").exists()
    assert not list(TOOLKIT.rglob("KEIL"))


def test_bsp_subset_has_required_libraries_and_headers() -> None:
    bsp = paths.BSP_ROOT
    for rel in (
        "Library/CMSIS/Lib/GCC/libCMSIS_DSP.a", "Library/CMSIS/Lib/GCC/libCMSIS_NN.a",
        "ThirdParty/tflite_micro/Lib/libtflu.a", "ThirdParty/openmv/omv/Lib/libomv.a",
        "Library/StdDriver/src/dmic.c", "Library/StdDriver/src/lppdma.c", "Library/StdDriver/src/pmc.c",
        "Library/StdDriver/src/hsusbd.c", "Library/StdDriver/src/retarget.c",
        "Library/Device/Nuvoton/M55M1/Source/GCC/retarget_GCC.c",
        "ThirdParty/tflite_micro/signal/micro/kernels/rfft.h",
        "ThirdParty/tflite_micro/_deps/tensorflow-gemlowp-src/fixedpoint/fixedpoint.h",
        "ThirdParty/ml-embedded-evaluation-kit/source/application/api/common/source/Classifier.cc",
        "ThirdParty/ml-embedded-evaluation-kit/source/profiler/include/Profiler.hpp",
        "ThirdParty/FatFs/source/ff.h", "LICENSE", "NOTICE",
    ):
        assert (bsp / rel).is_file(), rel
    assert not (bsp / "ThirdParty/FatFs/source/ffconf_M55M1.h").exists()
    assert not (bsp / "ThirdParty/tflite_micro/tensorflow/lite/micro/examples").exists()
    assert not (bsp / "Library/PowerDeliveryLib").exists()


def test_bsp_patch_profiler_applied() -> None:
    patched = paths.BSP_ROOT / "ThirdParty/ml-embedded-evaluation-kit/source/profiler/include/Profiler.hpp"
    source = paths.TEMPLATES_ROOT / "BSP_patch/ThirdParty/ml-embedded-evaluation-kit/source/profiler/include/Profiler.hpp"
    assert patched.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("board", ["NuMaker-M55M1", "NuGestureAI-M55M1"])
@pytest.mark.parametrize("app", ["imgclass", "generic"])
def test_templates_are_patched(board: str, app: str) -> None:
    tdir = paths.template_dir(board, app)
    proj = tdir / paths.project_name_for(app)
    ffconf = (proj / "ffconf_M55M1.h").read_text(encoding="utf-8")
    assert "#define FFCONF_DEF\t80386" in ffconf or "#define FFCONF_DEF 80386" in ffconf
    assert "FF_USE_LFN\t1" in ffconf.replace(" ", "\t") or "FF_USE_LFN 1" in ffconf
    authoritative_ld = tdir / "link_script" / "gcc" / "gcc.ld"
    ld = authoritative_ld.read_text(encoding="utf-8")
    assert "SRAM_NONCACHEABLE" in ld and "__sram_noncacheable_start__" in ld
    assert "KEEP(*(nn_model))" in ld and "*(ITCM)" in ld
    # progen toolchain_settings `linker_file` is `<proj>\GCC\gcc.ld`, so the in-project copy is the
    # one actually linked; it must be byte-identical to the authoritative patched script.
    assert (proj / "GCC" / "gcc.ld").read_bytes() == authoritative_ld.read_bytes()
    if board == "NuMaker-M55M1":
        assert "SRAM01_HYPERRAM (rw) : ORIGIN = 0x81F00000, LENGTH = 0x00100000" in ld
        board_init = (proj / "BoardInit.cpp").read_text(encoding="utf-8")
        for stmt in ("HyperRAM_PinConfig(HYPERRAM_SPIM_PORT);", "HyperRAM_Init(HYPERRAM_SPIM_PORT);",
                     "SPIM_HYPER_EnterDirectMapMode(HYPERRAM_SPIM_PORT);"):
            assert f"// {stmt}" in board_init
        assert not (proj / "Device" / "HyperRAM").exists()
        device_yaml = (tdir / "progen/tools/records/Device/Device.yaml").read_text(encoding="utf-8")
        assert "hyperram_code.c" not in device_yaml
        if app == "imgclass":
            display = (proj / "Device/include/Display.h").read_text(encoding="utf-8")
            assert "//#define CONFIG_DISP_USE_PDMA" in display
    assert not (tdir / "link_script" / "armcc").exists()
    assert not (tdir / "progen" / "tools" / "records" / "templates").exists()
    assert (tdir / "progen" / "project.yaml").is_file()


def test_numl_codegen_patched_for_studio() -> None:
    numl = paths.NUML_ROOT
    for rel in ("generic_codegen/generic_codegen.py", "imgclass_codegen/imgclass_codegen.py"):
        text = (numl / rel).read_text(encoding="utf-8")
        assert "os.path.dirname(os.path.abspath(__file__))" in text
        assert "Modified for Teachable Machine Local Studio" in text
    for rel in ("generic_codegen/main_cpp_codegen.py", "imgclass_codegen/main_cpp_codegen.py"):
        text = (numl / rel).read_text(encoding="utf-8")
        assert "import pandas" not in text and "import csv" in text
    assert (numl / "tflite" / "__init__.py").is_file()
    assert not (numl / "project_generate.py").exists()
    assert not (numl / "objdet_codegen").exists()


def test_vela_and_manifest_present() -> None:
    assert paths.VELA_EXE.is_file() and paths.VELA_INI.is_file()
    manifest = json.loads(paths.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["sources"]["m55m1bsp_tag"] == "V3.01.005"
    files = manifest["files"]
    assert len(files) > 1000
    # The manifest covers every file under mcu_toolkit/ except manifest.json itself (the rule in
    # vendor_mcu_toolkit.write_manifest()); nothing may be shipped unhashed or listed but absent.
    on_disk = {p.relative_to(TOOLKIT).as_posix() for p in _rel_files() if p.name != "manifest.json"}
    assert on_disk == set(files)
    for rel, expected in files.items():
        assert hashlib.sha256((TOOLKIT / rel).read_bytes()).hexdigest() == expected, rel
    assert (TOOLKIT / "README_zh-TW.md").is_file()
    assert (TOOLKIT / "NOTICE_third_party.md").is_file()
    assert (TOOLKIT / "boards.json").is_file()
