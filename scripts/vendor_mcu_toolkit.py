"""Maintainer tool: build mcu_toolkit/ from a NuML_Toolkit checkout (+ App Builder for Task 9/10).

Usage:
  .venv\\Scripts\\python.exe scripts\\vendor_mcu_toolkit.py toolkit --numl-root C:\\...\\NuML_Toolkit
  .venv\\Scripts\\python.exe scripts\\vendor_mcu_toolkit.py cdc --app-builder-root C:\\...\\NuML_App_Builder_v1.0
  .venv\\Scripts\\python.exe scripts\\vendor_mcu_toolkit.py image-templates --numl-root ... --app-builder-root ...
  .venv\\Scripts\\python.exe scripts\\vendor_mcu_toolkit.py known-sound --app-builder-root C:\\...\\NuML_App_Builder_v1.0
  .venv\\Scripts\\python.exe scripts\\vendor_mcu_toolkit.py kws-template --app-builder-root C:\\...\\NuML_App_Builder_v1.0
  .venv\\Scripts\\python.exe scripts\\vendor_mcu_toolkit.py board-init --app-builder-root C:\\...\\NuML_App_Builder_v1.0
  .venv\\Scripts\\python.exe scripts\\vendor_mcu_toolkit.py licenses [--numl-root C:\\...\\NuML_Toolkit]
  .venv\\Scripts\\python.exe scripts\\vendor_mcu_toolkit.py manifest
Students never run this; the output is shipped inside the Studio zip.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEST = PROJECT_ROOT / "mcu_toolkit"
PATCHES = PROJECT_ROOT / "scripts" / "vendor_patches"
BOARDS = ("NuMaker-M55M1", "NuGestureAI-M55M1")
APPS = ("generic_template", "imgclass_template")
PROJECT_NAMES = {"generic_template": "NN_ModelInference", "imgclass_template": "NN_ImgClassInference"}
SOURCES = {
    "numl_toolkit_commit": "3126344",
    "m55m1bsp_tag": "V3.01.005",
    "m55m1bsp_commit": "adb58c85",
    "app_builder_version": "0.1.8",
}
STUDIO_MARK = "Modified for Teachable Machine Local Studio (see mcu_toolkit/README_zh-TW.md)."

BSP_DIRS = [
    "Library/CMSIS/Core/Include", "Library/CMSIS/DSP/Include", "Library/CMSIS/NN/Include",
    "Library/Device/Nuvoton/M55M1/Include", "Library/Device/Nuvoton/M55M1/Source/GCC",
    "Library/StdDriver/inc", "Library/StdDriver/src/npu",
    "ThirdParty/tflite_micro/tensorflow", "ThirdParty/tflite_micro/signal",
    "ThirdParty/tflite_micro/_deps/tensorflow-flatbuffers-src/include",
    "ThirdParty/tflite_micro/_deps/tensorflow-gemlowp-src/fixedpoint",
    "ThirdParty/tflite_micro/_deps/tensorflow-gemlowp-src/internal",
    "ThirdParty/tflite_micro/_deps/tensorflow-gemlowp-src/profiling",
    "ThirdParty/tflite_micro/_deps/tensorflow-gemlowp-src/public",
    "ThirdParty/FatFs/source",
    "ThirdParty/ml-embedded-evaluation-kit/source/application/api/common",
    "ThirdParty/ml-embedded-evaluation-kit/source/log/include",
    "ThirdParty/ml-embedded-evaluation-kit/source/math",
    "ThirdParty/ml-embedded-evaluation-kit/source/application/main/include",
    "ThirdParty/ml-embedded-evaluation-kit/source/profiler",
]
BSP_FILES = [
    "LICENSE", "NOTICE", "README.md",
    "Library/CMSIS/Lib/GCC/libCMSIS_DSP.a", "Library/CMSIS/Lib/GCC/libCMSIS_NN.a",
    "Library/CMSIS/Compiler/LICENSE", "Library/CMSIS/DSP/LICENSE", "Library/CMSIS/NN/LICENSE",
    "Library/Device/Nuvoton/M55M1/Source/startup_M55M1.c",
    "Library/Device/Nuvoton/M55M1/Source/system_M55M1.c",
    "Library/Storage/diskio_SDH.c",
    "ThirdParty/tflite_micro/LICENSE", "ThirdParty/tflite_micro/README.md", "ThirdParty/tflite_micro/AUTHORS",
    "ThirdParty/tflite_micro/Lib/libtflu.a",
    "ThirdParty/openmv/LICENSE.txt", "ThirdParty/openmv/README.md",
    "ThirdParty/openmv/omv/Lib/imlib_config.h", "ThirdParty/openmv/omv/Lib/omv_boardconfig.h",
    "ThirdParty/openmv/omv/Lib/libomv.a",
] + [
    f"Library/StdDriver/src/{name}.c"
    for name in ("uart", "retarget", "ccap", "clk", "sys", "spim_hyper", "gpio", "ebi", "spi", "sdh",
                 "hsusbd", "pdma", "dmic", "lppdma", "pmc")
]
BSP_GLOB_FILES = [
    "ThirdParty/tflite_micro/_deps/tensorflow-flatbuffers-src/LICENSE*",
    "ThirdParty/tflite_micro/_deps/tensorflow-gemlowp-src/LICENSE*",
]
BSP_EXCLUDE_SUBDIRS = [
    "ThirdParty/tflite_micro/tensorflow/lite/micro/integration_tests",
    "ThirdParty/tflite_micro/tensorflow/lite/micro/examples",
    "ThirdParty/tflite_micro/tensorflow/lite/micro/tools",
    "ThirdParty/tflite_micro/tensorflow/lite/micro/models",
    "ThirdParty/tflite_micro/tensorflow/lite/micro/docs",
    "ThirdParty/tflite_micro/tensorflow/lite/python",
]
# P2: _syscalls.c carries a GPL notice (uOS++, Liviu Ionescu) inside a BSP directory the NOTICE
# calls Apache-2.0. It is never #included or otherwise referenced anywhere in this repository --
# unlike its sibling retarget_GCC.c, which numl_cdc_retarget.c #includes for every LCD-less board
# build (image/known_sound/kws apps all stage it via tm_local/mcu/apps/_cdc.py), so the directory
# itself stays in BSP_DIRS and only these two files are dropped. semihosting.h is _syscalls.c's
# only #include-r and becomes dead code the moment it is gone.
BSP_EXCLUDE_FILES = [
    "Library/Device/Nuvoton/M55M1/Source/GCC/_syscalls.c",
    "Library/Device/Nuvoton/M55M1/Source/GCC/semihosting.h",
]
# This OpenMV imlib/alloc/common subset is vendored header-only: no .c file from these three
# directories is compiled or linked. gcc_build.py assembles an explicit source list, never a glob
# or directory walk, and no app overlay names one, so only the headers the vendored tree's own C
# files #include are copied; see the header_only_ignore() loop in vendor_toolkit().
# Licensing, measured rather than assumed: of the 70 .c files upstream ships in these directories,
# exactly one -- agast.c -- carries a copyleft header (GNU GPL v3-or-later, Copyright (C) 2010
# Elmar Mair). lsd.c, zbar.c, ini.c and the other 66 carry OpenMV's own MIT header, so the MIT
# line in NOTICE.md / NOTICE_third_party.md describes them correctly.
# This exclusion says nothing about ThirdParty/openmv/omv/Lib/libomv.a: that is a prebuilt archive
# which imgclass_template puts on the link line (-lomv, image firmware only -- generic_template,
# used by known_sound and kws, does not), and agast.o is one of its 62 members. Dropping a .c file
# here does not remove its object from that archive.
BSP_HEADER_ONLY_DIRS = [
    "ThirdParty/openmv/omv/imlib", "ThirdParty/openmv/omv/alloc", "ThirdParty/openmv/omv/common",
]
# ...which is why agast.c is the one .c kept despite that rule. We ship the object (inside
# libomv.a, linked by Image firmware), so we ship its source: GPL v3 section 4 asks a distributor
# to hand over the source and a copy of the license, not to hide the source. Keeping it is also
# inert for the build -- the only reference anywhere in the vendored tree is the agast_detect()
# prototype in imlib.h, no progen record or overlay names the file, and gcc_build.py compiles an
# explicit list rather than a directory walk, so this file is distributed and never built here.
# NOT full GPL compliance on its own: the Corresponding Source for libomv.a also includes the
# recipe Nuvoton used to build the archive, which this repo does not have. See NOTICE.md.
BSP_HEADER_ONLY_KEEP = {"ThirdParty/openmv/omv/imlib/agast.c"}
FORBIDDEN_SUFFIXES = {".bat", ".uvprojx", ".uvoptx", ".pyc", ".scatter"}
FORBIDDEN_DIRS = {"__pycache__", ".git", "KEIL"}

# --- license texts -----------------------------------------------------------------------
# NuML_Toolkit's root LICENSE is the Apache-2.0 text; it covers NuML_TFLM_Tool and, per
# NOTICE_third_party.md, the App Builder-derived sources under apps/ as well. It is copied into
# the vendored tree (NuML_TFLM_Tool/LICENSE) and re-used as LICENSES/Apache-2.0.txt, which the
# NOTICE points at for the components that ship no license file of their own here (the Arm ML
# Embedded Evaluation Kit subset and the Vela executable).
NUML_LICENSE_REL = "NuML_TFLM_Tool/LICENSE"
LICENSES_DIR = "LICENSES"
# sha256 of https://www.gnu.org/licenses/gpl-3.0.txt as fetched 2026-09-13 (35,149 bytes).
GPL3_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
# FatFs ships no LICENSE file at all -- upstream's M55M1BSP ThirdParty/FatFs holds only
# `documents/` and `source/` -- so the terms live in a comment block at the top of every source
# file. LICENSES/FatFs.txt is that block, copied out byte for byte.
FATFS_HEADER_REL = "NuML_TFLM_Tool/templates/M55M1BSP/ThirdParty/FatFs/source/ff.h"
FATFS_BLOCK_END = b"/----------------------------------------------------------------------------*/"


def _copy_tree(src: Path, dst: Path, *, exclude_dirs: set[str] = frozenset()) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        skip = {n for n in names if n in FORBIDDEN_DIRS or n in exclude_dirs}
        skip |= {n for n in names if Path(n).suffix.lower() in FORBIDDEN_SUFFIXES}
        return skip

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _replace_once(path: Path, old: str, new: str, *, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != count:
        raise SystemExit(f"patch anchor mismatch in {path}: expected {count} of {old!r}, found {text.count(old)}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def vendor_toolkit(numl_root: Path) -> None:
    tflm = numl_root / "NuML_TFLM_Tool"
    dest_tflm = DEST / "NuML_TFLM_Tool"
    if dest_tflm.exists():
        shutil.rmtree(dest_tflm)
    if (DEST / "vela").exists():
        shutil.rmtree(DEST / "vela")

    for pkg in ("generic_codegen", "imgclass_codegen", "tflite"):
        _copy_tree(tflm / pkg, dest_tflm / pkg)
    _copy_tree(tflm / "templates" / "M55M1" / "BSP_patch", dest_tflm / "templates" / "M55M1" / "BSP_patch")
    for board in BOARDS:
        for app in APPS:
            src = tflm / "templates" / "M55M1" / board / app
            dst = dest_tflm / "templates" / "M55M1" / board / app
            _copy_tree(src, dst, exclude_dirs={"armcc", "templates"})

    bsp_src = tflm / "templates" / "M55M1BSP"
    bsp_dst = dest_tflm / "templates" / "M55M1BSP"
    excluded_abs = {bsp_src / rel for rel in BSP_EXCLUDE_SUBDIRS} | {
        bsp_src / rel for rel in BSP_EXCLUDE_FILES
    }
    for rel in BSP_DIRS:
        def ignore(directory: str, names: list[str], _root=Path) -> set[str]:
            here = Path(directory)
            skip = {n for n in names if n in FORBIDDEN_DIRS or Path(n).suffix.lower() in FORBIDDEN_SUFFIXES}
            skip |= {n for n in names if (here / n) in excluded_abs}
            return skip
        shutil.copytree(bsp_src / rel, bsp_dst / rel, ignore=ignore, dirs_exist_ok=True)
    keep_abs = {(bsp_src / rel).resolve() for rel in BSP_HEADER_ONLY_KEEP}
    for rel in BSP_HEADER_ONLY_DIRS:
        def header_only_ignore(directory: str, names: list[str]) -> set[str]:
            here = Path(directory)
            skip = {n for n in names if n in FORBIDDEN_DIRS}
            skip |= {
                n
                for n in names
                if not (here / n).is_dir()
                and Path(n).suffix.lower() != ".h"
                and (here / n).resolve() not in keep_abs
            }
            return skip
        shutil.copytree(bsp_src / rel, bsp_dst / rel, ignore=header_only_ignore, dirs_exist_ok=True)
    missing_keep = [rel for rel in BSP_HEADER_ONLY_KEEP if not (bsp_dst / rel).is_file()]
    if missing_keep:
        # A silent miss here would quietly drop the GPL source we promise in NOTICE.md, so it is
        # an error: either the upstream path moved or BSP_HEADER_ONLY_KEEP no longer matches it.
        raise SystemExit(f"BSP_HEADER_ONLY_KEEP entries were not copied: {missing_keep}")
    for rel in BSP_FILES:
        _copy_file(bsp_src / rel, bsp_dst / rel)
    for pattern in BSP_GLOB_FILES:
        for found in bsp_src.glob(pattern):
            _copy_file(found, bsp_dst / found.relative_to(bsp_src))
    # BSP_patch overlay (Profiler.hpp) is applied once here, exactly like project_generate.py did.
    _copy_tree(dest_tflm / "templates" / "M55M1" / "BSP_patch", bsp_dst)

    (DEST / "vela").mkdir(parents=True)
    _copy_file(numl_root / "vela" / "vela-5_1_0.exe", DEST / "vela" / "vela-5_1_0.exe")
    _copy_file(numl_root / "vela" / "default_vela.ini", DEST / "vela" / "default_vela.ini")

    apply_template_patches(dest_tflm)
    patch_numl_codegen(dest_tflm)
    write_licenses(numl_root)
    write_docs()
    write_manifest()


def apply_template_patches(dest_tflm: Path) -> None:
    for board in BOARDS:
        for app in APPS:
            tdir = dest_tflm / "templates" / "M55M1" / board / app
            proj = tdir / PROJECT_NAMES[app]
            shutil.copy2(PATCHES / "ffconf_M55M1.h", proj / "ffconf_M55M1.h")
            ld_dst = tdir / "link_script" / "gcc" / "gcc.ld"
            shutil.copy2(PATCHES / "gcc_ld" / board / "gcc.ld", ld_dst)
            ld = ld_dst.read_text(encoding="utf-8")
            if "*(ITCM)" not in ld:
                # `NVT_ITCM` in system_M55M1.h is `__attribute__((section("ITCM")))`, so the .itcm
                # output section must collect `*(ITCM)` as well as libomv's text. Anchor on the two
                # real lines: upstream opens the section as `.itcm : AT (__etext)` with `{` on the
                # next line, so a `\.itcm\s*:\s*\{` pattern never matches.
                anchor = "    __itcm_text_start__ = .;\n    *libomv.a:(.text*)\n"
                if ld.count(anchor) != 1:
                    raise SystemExit(f"{ld_dst}: .itcm anchor for *(ITCM) not found exactly once")
                ld = ld.replace(anchor, "    __itcm_text_start__ = .;\n    *(ITCM)\n    *libomv.a:(.text*)\n")
                if "*(ITCM)" not in ld:
                    raise SystemExit(f"could not insert *(ITCM) into {ld_dst}")
            if "KEEP(*(nn_model))" not in ld:
                raise SystemExit(f"{ld_dst} lacks KEEP(*(nn_model)); refresh scripts/vendor_patches/gcc_ld")
            if board == "NuMaker-M55M1":
                # Upstream pads the region name/attribute columns ("SRAM01_HYPERRAM  (rw)  :");
                # normalise to single spaces so the shipped line is byte-stable for the guard below.
                x_region = "SRAM01_HYPERRAM (rw) : ORIGIN = 0x81F00000, LENGTH = 0x00100000"
                old_region = "SRAM01_HYPERRAM  (rw)  : ORIGIN = 0x81F00000, LENGTH = 0x02100000"
                if x_region not in ld:
                    if ld.count(old_region) != 1:
                        raise SystemExit(f"{ld_dst}: SRAM01_HYPERRAM MEMORY anchor not found exactly once")
                    ld = ld.replace(old_region, x_region)
                if x_region not in ld:
                    raise SystemExit("X-board gcc.ld SRAM01_HYPERRAM anchor not found")
            ld_dst.write_text(ld, encoding="utf-8", newline="\n")
            # progen's toolchain_settings `linker_file` is `<proj>\GCC\gcc.ld`, so the in-project
            # copy -- not link_script/gcc/gcc.ld -- is what actually gets linked. It must always be
            # overwritten; upstream's copy carries a different memory map (ITCM 128 KiB, SRAM012,
            # 32 KiB stack/heap) and silently skipping it would link the unpatched map.
            in_project_ld = proj / "GCC" / "gcc.ld"
            if not in_project_ld.is_file():
                raise SystemExit(
                    f"in-project linker script {in_project_ld} is missing; progen "
                    "toolchain_settings linker_file points at it, so it must exist and stay in "
                    "sync with link_script/gcc/gcc.ld -- check whether upstream moved it"
                )
            shutil.copy2(ld_dst, in_project_ld)
            if board == "NuMaker-M55M1":
                board_init = proj / "BoardInit.cpp"
                text = board_init.read_text(encoding="utf-8")
                for stmt in ('#include "hyperram_code.h"', "HyperRAM_PinConfig(HYPERRAM_SPIM_PORT);",
                             "HyperRAM_Init(HYPERRAM_SPIM_PORT);", "SPIM_HYPER_EnterDirectMapMode(HYPERRAM_SPIM_PORT);"):
                    text, n = re.subn(r"(?m)^([ \t]*)" + re.escape(stmt) + r"[ \t]*$",
                                      r"\1// " + stmt + "  /* internal flash deployment: HyperRAM unused */", text)
                    if n != 1:
                        raise SystemExit(f"BoardInit.cpp anchor {stmt!r} found {n} times in {board_init}")
                board_init.write_text(text, encoding="utf-8", newline="\n")
                hyper = proj / "Device" / "HyperRAM"
                if hyper.exists():
                    shutil.rmtree(hyper)
                device_yaml = tdir / "progen" / "tools" / "records" / "Device" / "Device.yaml"
                lines = [ln for ln in device_yaml.read_text(encoding="utf-8").splitlines()
                         if "hyperram_code.c" not in ln]
                device_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
                display = proj / "Device" / "include" / "Display.h"
                if display.exists():
                    _replace_once(display, "#define CONFIG_DISP_USE_PDMA",
                                  "//#define CONFIG_DISP_USE_PDMA  /* stable no-heap LCD path */")


def patch_numl_codegen(dest_tflm: Path) -> None:
    header = f"# {STUDIO_MARK}\n"
    for pkg in ("generic_codegen", "imgclass_codegen"):
        entry = dest_tflm / pkg / f"{pkg}.py"
        _replace_once(entry, f"template_path = '{pkg}'",
                      "template_path = os.path.dirname(os.path.abspath(__file__))")
        entry.write_text(header + entry.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        main_gen = dest_tflm / pkg / "main_cpp_codegen.py"
        text = main_gen.read_text(encoding="utf-8")
        text = text.replace("import pandas\n", "import csv\n")
        old = ("    usecols = ['sram_memory_used', 'off_chip_flash_memory_used']\n"
               "    df = pandas.read_csv(summary_file, usecols=usecols)\n"
               "    return df.iloc[0,0]*1024, df.iloc[0,1]*1024 ")
        new = ("    with open(summary_file, newline='') as stream:\n"
               "        row = next(csv.DictReader(stream))\n"
               "    return float(row['sram_memory_used']) * 1024, float(row['off_chip_flash_memory_used']) * 1024")
        if old not in text:
            raise SystemExit(f"pandas anchor not found in {main_gen}")
        main_gen.write_text(header + text.replace(old, new), encoding="utf-8", newline="\n")
    for pkg in ("generic_codegen", "imgclass_codegen"):
        for py in (dest_tflm / pkg).glob("*.py"):
            text = py.read_text(encoding="utf-8")
            if "import pandas" in text:
                raise SystemExit(f"{py} still imports pandas; extend patch_numl_codegen")


def write_licenses(numl_root: Path | None = None) -> None:
    """Materialise `NuML_TFLM_Tool/LICENSE` and `LICENSES/` from sources already in the tree.

    Split out of `vendor_toolkit()` so it can be re-run on its own: the full toolkit vendoring
    rmtree's and re-copies 93 MB, which is a poor way to add three text files. `numl_root` is
    optional and only needed to (re)fetch the NuML LICENSE; without it the already-vendored copy
    is the input, and its absence is an error rather than a silent skip -- a NOTICE that points
    at a file which is not there is worse than no NOTICE.
    """
    license_path = DEST / NUML_LICENSE_REL
    if numl_root is not None:
        _copy_file(numl_root / "LICENSE", license_path)
    if not license_path.is_file():
        raise SystemExit(
            f"{license_path} is missing; re-run the `toolkit` subcommand, or pass --numl-root "
            "to `licenses`"
        )
    apache = license_path.read_bytes()
    for marker in (b"Apache License", b"Version 2.0, January 2004"):
        if marker not in apache:
            raise SystemExit(f"{license_path} is not the Apache-2.0 text ({marker!r} missing)")

    header = (DEST / FATFS_HEADER_REL).read_bytes()
    end = header.find(FATFS_BLOCK_END)
    if not header.startswith(b"/*---") or end < 0:
        raise SystemExit(f"{DEST / FATFS_HEADER_REL}: FatFs license block not found at the top")
    fatfs = header[: end + len(FATFS_BLOCK_END)]
    for marker in (b"Copyright (C) 2025, ChaN", b"Redistribution and use of FatFs"):
        if marker not in fatfs:
            raise SystemExit(f"FatFs license block lacks {marker!r}; upstream header changed")

    # The GPL v3 text is the one license here that no vendored or upstream file carries in full:
    # agast.c has the copyleft notice and a gnu.org URL, not the terms, and the only complete v3
    # text anywhere upstream is the LESSER GPL, a different license. So the canonical text is kept
    # in-repo at scripts/vendor_patches/gpl-3.0.txt and copied out, with its digest checked -- a
    # license file that has drifted is worse than one that is absent, because it still looks
    # authoritative. Refresh it from https://www.gnu.org/licenses/gpl-3.0.txt if the FSF ever
    # reissues the text, and update GPL3_SHA256 in the same commit.
    gpl_src = Path(__file__).resolve().parent / "vendor_patches" / "gpl-3.0.txt"
    if not gpl_src.is_file():
        raise SystemExit(f"{gpl_src} is missing; it is the source for LICENSES/GPL-3.0.txt")
    gpl = gpl_src.read_bytes()
    digest = hashlib.sha256(gpl).hexdigest()
    if digest != GPL3_SHA256:
        raise SystemExit(
            f"{gpl_src} does not match the recorded GNU GPL v3 text (sha256 {digest}, "
            f"expected {GPL3_SHA256})"
        )

    out = DEST / LICENSES_DIR
    out.mkdir(parents=True, exist_ok=True)
    # Byte-for-byte, line endings included: these are license texts, not generated prose.
    (out / "Apache-2.0.txt").write_bytes(apache)
    (out / "FatFs.txt").write_bytes(fatfs)
    (out / "GPL-3.0.txt").write_bytes(gpl)


def write_docs() -> None:
    (DEST / "README_zh-TW.md").write_text(
        "# mcu_toolkit\n\n"
        "本資料夾由 `scripts/vendor_mcu_toolkit.py` 從 Nuvoton NuML_Toolkit（commit "
        f"{SOURCES['numl_toolkit_commit']}）、M55M1BSP {SOURCES['m55m1bsp_tag']}（{SOURCES['m55m1bsp_commit']}）"
        f"與 NuML App Builder v{SOURCES['app_builder_version']} 產生，只保留 GCC 建置需要的部分。\n\n"
        "## 相對上游的修改\n\n"
        "- `NuML_TFLM_Tool/*/*_codegen.py`：樣板路徑改為相對 `__file__`；檔頭加註 Studio 修改標記。\n"
        "- `NuML_TFLM_Tool/*/main_cpp_codegen.py`：Vela summary CSV 改用標準函式庫 `csv` 依**欄名**"
        "（`sram_memory_used`、`off_chip_flash_memory_used`）讀取，取代 pandas 的位置索引 "
        "`df.iloc[0,0]`／`df.iloc[0,1]`；Vela 的欄序即為此兩欄，結果與上游相同。\n"
        "- `templates/M55M1/*/*/NN_*/ffconf_M55M1.h`：重生為 FatFs R0.16（FFCONF_DEF 80386）。\n"
        "- `templates/M55M1/*/*/link_script/gcc/gcc.ld`：新增 SRAM_NONCACHEABLE 區段與 `__sram_noncacheable_*` 符號、"
        "`KEEP(*(nn_model))`、`*(ITCM)`；X 板 SRAM01_HYPERRAM 縮為 1 MiB（internal-only）。\n"
        "- `templates/M55M1/*/*/NN_*/GCC/gcc.ld`：**整份覆寫**為上面那份修補後的 `link_script/gcc/gcc.ld`。"
        "progen `toolchain_settings.yaml` 的 `linker_file` 指向的是這份專案內副本，實際連結用的也是它，"
        "所以兩者必須逐位元相同。相對上游的**專案內**副本，記憶體配置因此改變："
        "ITCM 由 0x00020000（128 KiB）改為 0x00010000（64 KiB）、"
        "SRAM012（0x00150000）改為 SRAM2（0x00050000）視窗、stack/heap 由 0x8000（32 KiB）改為 0xa000（40 KiB）。"
        "拿本資料夾與全新的 NuML checkout 做 diff 時，這份檔案的差異是預期的。\n"
        "- X 板 `BoardInit.cpp`：HyperRAM 初始化註解掉；`Device/HyperRAM` 移除；`Display.h` 的 PDMA 關閉。\n"
        "- `templates/M55M1BSP`：只含建置需要的子集，已套用 `BSP_patch/Profiler.hpp`。\n"
        "- `apps/`：Studio 自己的每 kind 韌體來源（部分改寫自 App Builder）。\n"
        "- `boards/NuMaker-VoiceAI-M55M1/BoardInit_VoiceAI.cpp`：由 App Builder "
        "`known_sound/device/BoardInit_VoiceAI.cpp` 原封不動複製，未作任何修改。VoiceAI 板沒有自己的 "
        "NuML 樣板，專案借用 NuGestureAI-M55M1 樣板產生，該樣板的除錯主控台在 UART5；VoiceAI 板實際的主控台是 "
        "UART4（J3 排針），而 BSP 的 `SetDebugUartMFP()` 沒有 UART4 分支，所以這個板子必須自帶 "
        "`SetDebugUartMFP()`／`SetDebugUartCLK()`／`InitDebugUart()` 三個函式——這是整支檔案的差異，不是編譯期 "
        "`-D` 定義能表達的。Task 3 的 `boards.json` 會把它登記為該板的 `extra_sources`。\n"
        "- `LICENSES/`：本資料夾額外補上的授權全文（`Apache-2.0.txt`、`FatFs.txt`、`GPL-3.0.txt`），"
        "上游沒有隨附授權檔的元件在 `NOTICE_third_party.md` 指向這裡。`GPL-3.0.txt` 對應的是 "
        "`omv/Lib/libomv.a` 內的 `agast.o`，其原始碼 `omv/imlib/agast.c` 也一併隨附（只散布、不編譯）。\n\n"
        "重新產生：`.venv\\Scripts\\python.exe scripts\\vendor_mcu_toolkit.py toolkit --numl-root <NuML_Toolkit>`，"
        "再執行 `cdc`、`image-templates`、`known-sound`、`kws-template`、`board-init`、`manifest` 子命令。\n"
        "`licenses` 子命令只重寫授權全文與本文件，不會重跑 93 MB 的 `toolkit`。\n",
        encoding="utf-8",
    )
    (DEST / "NOTICE_third_party.md").write_text(
        "# Third-party notices for mcu_toolkit\n\n"
        "License texts shipped here: `NuML_TFLM_Tool/LICENSE` (NuML_Toolkit, Apache-2.0), "
        "`LICENSES/Apache-2.0.txt` (the same text, for the components below that carry no "
        "license file of their own in this subset) and `LICENSES/FatFs.txt` (the FatFs terms, "
        "copied verbatim from the block at the top of "
        "`NuML_TFLM_Tool/templates/M55M1BSP/ThirdParty/FatFs/source/ff.h`; FatFs ships no "
        "separate license file upstream). Components that do carry their own file keep it in "
        "place — see `templates/M55M1BSP/LICENSE`, `.../CMSIS/*/LICENSE`, "
        "`.../tflite_micro/LICENSE`, `.../openmv/LICENSE.txt`.\n\n"
        "- NuML_Toolkit — Copyright Nuvoton Technology Corp. Apache License 2.0 "
        "(NuML_TFLM_Tool/LICENSE). `NuML_TFLM_Tool/tflite/` is TensorFlow's official FlatBuffers-"
        "generated Python schema bindings v2.10.0, Copyright The TensorFlow Authors, Apache "
        "License 2.0 (unmodified upstream code, imported by the codegen scripts above it).\n"
        f"- M55M1 Series BSP {SOURCES['m55m1bsp_tag']} — Copyright 2023 Nuvoton Technology Corp. All rights reserved. "
        "Apache License 2.0 (LICENSE, NOTICE in templates/M55M1BSP). `Library/StdDriver/*/gdma/dma350_*` are Arm "
        "Limited, BSD-3-Clause.\n"
        "- CMSIS 6.3.0 / CMSIS-DSP 1.16.2 / CMSIS-NN 6.0.0 — Copyright Arm Limited and affiliates. Apache License 2.0; "
        "CMSIS-DSP ComputeLibrary files are MIT.\n"
        "- TensorFlow Lite for Microcontrollers — Copyright The TensorFlow Authors. Apache License 2.0. "
        "Bundled FlatBuffers and gemmlowp (Google LLC, Apache-2.0).\n"
        "- Arm ML Embedded Evaluation Kit — Copyright (c) 2021-2022 Arm Limited and affiliates. Apache License 2.0 "
        "(LICENSES/Apache-2.0.txt).\n"
        "- FatFs R0.16 — Copyright (C) 2025, ChaN, all right reserved. Redistribution permitted per the notice in "
        "source headers, reproduced verbatim in LICENSES/FatFs.txt.\n"
        "- OpenMV imlib — The MIT License (MIT), Copyright (c) 2013-2021 Ibrahim Abdelkader and Kwabena W. Agyeman; "
        "`omv/imlib/lodepng.h` is zlib-licensed, Copyright 2005-2022 Lode Vandevenne. Only headers are vendored as "
        "source, with one deliberate exception described next. The prebuilt archive `omv/Lib/libomv.a`, which Image "
        "firmware links (-lomv) and Known Sound and KWS firmware do not, contains the member `agast.o`, built "
        "upstream from AGAST corner-detection code licensed GNU GPL v3-or-later, Copyright (C) 2010 Elmar Mair. "
        "Because that object is distributed here, its source is distributed with it: "
        "`templates/M55M1BSP/ThirdParty/openmv/omv/imlib/agast.c`, byte-for-byte from upstream, together with the "
        "full license text in `LICENSES/GPL-3.0.txt`. The file is shipped, never compiled -- no build record names "
        "it and the GCC driver compiles an explicit source list. This is a good-faith source offer, not a complete "
        "Corresponding Source package: the recipe Nuvoton used to build `libomv.a` is upstream and is not vendored "
        "here. Anyone redistributing this toolkit should either obtain that recipe, or drop `agast.o` from the "
        "archive and re-verify both boards' Image builds.\n"
        "- Arm Vela compiler 5.1.0 (vela/vela-5_1_0.exe) — Copyright 2020-2024 Arm Limited and/or its affiliates. "
        "Apache License 2.0 (LICENSES/Apache-2.0.txt). The executable is a PyInstaller bundle containing CPython "
        "3.10 (PSF-2.0), NumPy (BSD-3-Clause), FlatBuffers (Apache-2.0) and the PyInstaller bootloader (GPL-2.0 "
        "with bootloader exception).\n"
        f"- Sources under `apps/**` and `boards/**` derived from NuML App Builder "
        f"v{SOURCES['app_builder_version']} — Copyright Nuvoton Technology Corp. The App Builder "
        "distribution carries no license file of its own; these sources are redistributed here "
        "under the same Apache License 2.0 terms as NuML_Toolkit (NuML_TFLM_Tool/LICENSE, also copied to "
        "LICENSES/Apache-2.0.txt). `boards/NuMaker-VoiceAI-M55M1/BoardInit_VoiceAI.cpp` is "
        "`known_sound/device/BoardInit_VoiceAI.cpp` from the same App Builder release, copied unmodified "
        "and carrying its own Apache-2.0 SPDX header already.\n\n"
        "Not included: Arm GNU Toolchain (GPL-3.0 with runtime library exception; installed separately), "
        "Nuvoton Nu-Link Command Tool (proprietary; detected if installed), Keil MDK (not required).\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    # Checked first, never after: a manifest is a statement that the tree is shippable, so a tree
    # that still holds a forbidden file or directory must not get one written for it.
    verify_forbidden()
    files: dict[str, str] = {}
    for path in sorted(DEST.rglob("*")):
        # Same exemption as verify_forbidden() and tests/test_mcu_vendoring.py: bytecode cache is
        # not vendored content.  Listing it would make the manifest disagree with that test, which
        # compares the manifest keys against an on-disk walk that ignores it.
        if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
            continue
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(DEST).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sources": SOURCES,
        "files": files,
    }
    (DEST / "manifest.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def verify_forbidden() -> None:
    # __pycache__ directories and *.pyc files are exempt, matching the rule that
    # tests/test_mcu_vendoring.py already applies: they are transient bytecode-cache noise that a
    # stray import leaves behind (the Studio's host-frontend test imports apps/known_sound/
    # host_tests/harness.py), not something the vendoring produced.  write_manifest() below skips
    # them for the same reason, so an on-disk .pyc can never desynchronise the manifest either.
    for path in DEST.rglob("*"):
        if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
            continue
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            raise SystemExit(f"forbidden directory vendored: {path}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise SystemExit(f"forbidden file vendored: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_toolkit = sub.add_parser("toolkit")
    p_toolkit.add_argument("--numl-root", required=True, type=Path)
    p_cdc = sub.add_parser("cdc")
    p_cdc.add_argument("--app-builder-root", required=True, type=Path)
    p_img = sub.add_parser("image-templates")
    p_img.add_argument("--numl-root", required=True, type=Path)
    p_img.add_argument("--app-builder-root", required=True, type=Path)
    p_ks = sub.add_parser("known-sound")
    p_ks.add_argument("--app-builder-root", required=True, type=Path)
    p_kws = sub.add_parser("kws-template")
    p_kws.add_argument("--app-builder-root", required=True, type=Path)
    p_bi = sub.add_parser("board-init")
    p_bi.add_argument("--app-builder-root", required=True, type=Path)
    p_lic = sub.add_parser("licenses")
    p_lic.add_argument("--numl-root", type=Path)
    sub.add_parser("manifest")
    args = parser.parse_args(argv)
    DEST.mkdir(exist_ok=True)
    if args.command == "toolkit":
        vendor_toolkit(args.numl_root.resolve())
    elif args.command == "cdc":
        from scripts.vendor_cdc import vendor_cdc  # Task 9

        vendor_cdc(args.app_builder_root.resolve(), DEST)
        write_manifest()
    elif args.command == "image-templates":
        from scripts.vendor_image_templates import vendor_image_templates  # Task 10

        vendor_image_templates(args.numl_root.resolve(), args.app_builder_root.resolve(), DEST)
        write_manifest()
    elif args.command == "known-sound":
        from scripts.vendor_known_sound import vendor_known_sound  # Plan 2 Task 2

        vendor_known_sound(args.app_builder_root.resolve(), DEST)
        write_manifest()
    elif args.command == "kws-template":
        from scripts.vendor_kws_template import vendor_kws_template  # Plan 2 Task 6

        vendor_kws_template(args.app_builder_root.resolve(), DEST)
        write_manifest()
    elif args.command == "board-init":
        from scripts.vendor_board_init import vendor_board_init  # Plan 2 Task 2 (VoiceAI board)

        vendor_board_init(args.app_builder_root.resolve(), DEST)
        write_manifest()
    elif args.command == "licenses":
        write_licenses(args.numl_root.resolve() if args.numl_root else None)
        write_docs()
        write_manifest()
    elif args.command == "manifest":
        # The license texts and the NOTICE that indexes them are cheap to regenerate and must
        # never drift from the tree the manifest is about to hash, so refresh them first.
        write_licenses()
        write_docs()
        write_manifest()
    # Every branch ends in write_manifest(), which runs verify_forbidden() before writing.
    print(f"[OK] mcu_toolkit ready at {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
