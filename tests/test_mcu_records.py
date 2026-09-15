from __future__ import annotations

import json
from pathlib import Path

import pytest

from tm_local.mcu import records
from tm_local.mcu.errors import DeployError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "mcu"
RECIPE = json.loads((FIXTURES / "recipe_x_imgclass.json").read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_template(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A miniature NuML template + BSP + app dir that exercises every record rule."""
    bsp = tmp_path / "M55M1BSP"
    app = tmp_path / "work" / "NN_ImgClassInference"
    tdir = tmp_path / "templates" / "NuMaker-M55M1" / "imgclass_template"
    progen = tdir / "progen"
    _write(progen / "project.yaml", """
modules:
    Application: &module_Application
        - tools/records/Application/Application.yaml
    Driver: &module_Driver
        - tools/records/Driver/Driver.yaml
    Lib: &module_Lib
        - tools/records/Lib/Lib.yaml
    Model: &module_Model
        - tools/records/Model/Model.yaml
    toolchain_settings: &toolchain_settings
        - tools/records/toolchain_settings/toolchain_settings.yaml
projects:
    NN_ImgClassInference:
        - *module_Application
        - *module_Driver
        - *module_Lib
        - *module_Model
        - *toolchain_settings
""")
    _write(progen / "tools/records/Application/Application.yaml", """
common:
  target:
      - numaker-m55m1
  includes:
    - NN_ImgClassInference
  sources:
      Application:
      - NN_ImgClassInference
""")
    _write(progen / "tools/records/Driver/Driver.yaml", """
common:
  includes:
    - ..\\..\\Library\\StdDriver\\inc
    - ..\\..\\Library\\StdDriver\\inc\\
  sources:
    Driver:
    - ..\\..\\Library\\StdDriver\\src\\uart.c
    - ..\\..\\Library\\StdDriver\\src\\retarget.c
    - ..\\..\\Library\\StdDriver\\src\\LCD\\lcd.c
""")
    _write(progen / "tools/records/Lib/Lib.yaml", """
common:
  includes:
    - ..\\..\\ThirdParty\\evk\\source\\math\\
    - ..\\..\\Library\\StdDriver\\inc
tool_specific:
  gcc_arm:
    sources:
      Lib:
      - ..\\..\\ThirdParty\\tflite_micro\\Lib\\libtflu.a
      - ..\\..\\Library\\CMSIS\\Lib\\GCC\\libCMSIS_DSP.a
  uvision5_armc6:
    sources:
      Lib:
      - ..\\..\\Library\\CMSIS\\Lib\\KEIL\\cmsis_dsp.lib
""")
    _write(progen / "tools/records/Model/Model.yaml", """
common:
  includes:
    - NN_ImgClassInference\\Model\\include
  sources:
    Model:
    - NN_ImgClassInference\\Model
""")
    _write(progen / "tools/records/toolchain_settings/toolchain_settings.yaml", """
tool_specific:
  make_gcc_arm:
    macros:
      - NVT_VECTOR_ON_FLASH
      - ACTIVATION_BUF_SZ=0x100000
    misc:
      standard_libraries:
        - m
        - gcc
      common_flags:
        - -mfloat-abi=hard
        - -O1
      ld_flags:
        - -specs=nosys.specs
      cxx_flags:
        - -std=c++14
      c_flags:
        - -std=c11
    linker_file:
      - NN_ImgClassInference\\GCC\\gcc.ld
""")
    for rel in ("Library/StdDriver/src/uart.c", "Library/StdDriver/src/retarget.c",
                "Library/StdDriver/src/LCD/lcd.c", "ThirdParty/tflite_micro/Lib/libtflu.a",
                "Library/CMSIS/Lib/GCC/libCMSIS_DSP.a"):
        _write(bsp / rel, "")
    (bsp / "Library/StdDriver/inc").mkdir(parents=True)
    (bsp / "ThirdParty/evk/source/math").mkdir(parents=True)
    for rel in ("main.cpp", "BoardInit.cpp", "ModelFileReader.c", "Model/Labels.cpp",
                "Model/Thing.cc", "Model/include/x.hpp", "GCC/gcc.ld", "notes.txt"):
        _write(app / rel, "")
    (app / "Model" / "sub").mkdir()
    _write(app / "Model" / "sub" / "ignored.c", "")  # non-recursive expansion must skip this
    return tdir, app, bsp


def test_progen_relpath_and_resolution(tmp_path: Path) -> None:
    _tdir, app, bsp = _make_template(tmp_path)
    assert records.progen_relpath("..\\..\\Library\\StdDriver\\inc\\") == "..\\..\\..\\..\\Library\\StdDriver\\inc"
    assert records.progen_relpath("NN_ImgClassInference\\Model") == "..\\..\\NN_ImgClassInference\\Model"
    assert records.resolve_record_path("..\\..\\Library\\StdDriver\\src\\uart.c", "NN_ImgClassInference", app, bsp) == (bsp / "Library/StdDriver/src/uart.c").resolve()
    assert records.resolve_record_path("NN_ImgClassInference\\Model", "NN_ImgClassInference", app, bsp) == (app / "Model").resolve()
    assert records.resolve_record_path("NN_ImgClassInference", "NN_ImgClassInference", app, bsp) == app.resolve()
    with pytest.raises(DeployError):
        records.resolve_record_path("C:\\evil\\abs.c", "NN_ImgClassInference", app, bsp)


def test_load_records_builds_manifest(tmp_path: Path) -> None:
    tdir, app, bsp = _make_template(tmp_path)
    m = records.load_records(tdir, app, bsp)
    assert m.project_name == "NN_ImgClassInference"
    # includes: module order, first occurrence wins, trailing backslash normalised
    assert [p.name for p in m.includes] == ["NN_ImgClassInference", "inc", "math", "include"]
    assert m.includes[1] == (bsp / "Library/StdDriver/inc").resolve()
    # macros stripped of trailing whitespace
    assert m.defines == ["NVT_VECTOR_ON_FLASH", "ACTIVATION_BUF_SZ=0x100000"]
    # sources: directory expansion is non-recursive and extension-filtered
    names = sorted(s.path.name for s in m.sources)
    assert names == sorted(["main.cpp", "BoardInit.cpp", "ModelFileReader.c", "Labels.cpp", "Thing.cc",
                            "uart.c", "retarget.c", "lcd.c"])
    kinds = {s.path.name: s.kind for s in m.sources}
    assert kinds["Thing.cc"] == "cxx" and kinds["ModelFileReader.c"] == "c"
    # progen ordering: sorted by backslash relpath, uppercase before lowercase
    c_names = [s.path.name for s in m.c_sources()]
    assert c_names.index("lcd.c") < c_names.index("retarget.c") < c_names.index("uart.c")
    assert c_names[-1] == "ModelFileReader.c"  # ..\..\NN_ImgClass... sorts after ..\..\..\..\Library
    # libs: sorted by relpath, name minus lib/.a, one -L per lib (duplicates kept)
    assert m.libs == ["CMSIS_DSP", "tflu"]
    assert m.lib_dirs == [(bsp / "Library/CMSIS/Lib/GCC").resolve(), (bsp / "ThirdParty/tflite_micro/Lib").resolve()]
    assert m.common_flags == ["-mfloat-abi=hard", "-O1"]
    assert m.c_flags == ["-std=c11"] and m.cxx_flags == ["-std=c++14"]
    assert m.ld_flags == ["-specs=nosys.specs"] and m.standard_libraries == ["m", "gcc"]
    assert m.linker_script == (app / "GCC/gcc.ld").resolve()


def test_cpp_record_maps_to_cc_on_disk(tmp_path: Path) -> None:
    tdir, app, bsp = _make_template(tmp_path)
    _write(tdir / "progen/tools/records/Driver/Driver.yaml", """
common:
  sources:
    Driver:
    - ..\\..\\ThirdParty\\evk\\Model.cpp
""")
    _write(bsp / "ThirdParty/evk/Model.cc", "")
    m = records.load_records(tdir, app, bsp)
    cc = [s for s in m.sources if s.path.name == "Model.cc"]
    assert len(cc) == 1 and cc[0].kind == "cxx"
    assert cc[0].progen_relpath.endswith("Model.cpp")


def test_missing_source_raises(tmp_path: Path) -> None:
    tdir, app, bsp = _make_template(tmp_path)
    (bsp / "Library/StdDriver/src/uart.c").unlink()
    with pytest.raises(DeployError, match="uart.c"):
        records.load_records(tdir, app, bsp)


def test_object_collision_detected(tmp_path: Path) -> None:
    tdir, app, bsp = _make_template(tmp_path)
    _write(app / "Model" / "uart.c", "")
    m = records.load_records(tdir, app, bsp)
    with pytest.raises(DeployError, match="uart"):
        m.check_object_collisions()


def test_apply_overlay_is_pure(tmp_path: Path) -> None:
    tdir, app, bsp = _make_template(tmp_path)
    m = records.load_records(tdir, app, bsp)
    extra = app / "Device" / "CDC" / "numl_cdc.c"
    _write(extra, "")
    m2 = records.apply_overlay(
        m,
        add_sources=[extra],
        remove_source_basenames=["retarget.c"],
        add_includes=[bsp / "Library/StdDriver/src"],
        add_defines=["BOARD_X=1", "ACTIVATION_BUF_SZ=4096"],
        remove_defines=["ACTIVATION_BUF_SZ"],
    )
    assert any(s.path.name == "retarget.c" for s in m.sources)
    assert not any(s.path.name == "retarget.c" for s in m2.sources)
    assert any(s.path == extra.resolve() for s in m2.sources)
    assert m2.includes[-1] == (bsp / "Library/StdDriver/src").resolve()
    assert m2.defines == ["NVT_VECTOR_ON_FLASH", "BOARD_X=1", "ACTIVATION_BUF_SZ=4096"]
    assert m.defines == ["NVT_VECTOR_ON_FLASH", "ACTIVATION_BUF_SZ=0x100000"]


def test_apply_overlay_deduplicates_includes(tmp_path: Path) -> None:
    """One `-I` per directory, however many overlay entries name it.

    The known_sound app asks for the BSP's `StdDriver/src` (numl_cdc_retarget.c textually
    #includes retarget.c from there) on a board whose template may already provide it, and
    project_builder resolves app-relative entries before they get here -- so the same
    directory can arrive twice, spelled two ways, and must still be added once.
    """
    tdir, app, bsp = _make_template(tmp_path)
    m = records.load_records(tdir, app, bsp)
    already = bsp / "Library" / "StdDriver" / "inc"  # the template already lists this one
    assert already.resolve() in m.includes
    extra = app / "Frontend"
    m2 = records.apply_overlay(m, add_includes=[extra, already, extra, app, extra])
    assert m2.includes.count(extra.resolve()) == 1
    assert m2.includes.count(already.resolve()) == 1
    assert m2.includes.count(app.resolve()) == 1  # the template already lists the app dir
    assert len(m2.includes) == len(m.includes) + 1  # only Frontend is new


def test_recipe_fixture_ordering_matches_progen_rules() -> None:
    """The spike recipe is the ground truth for progen's include/source/link ordering.

    NOTE: the fixture's `sources_c`/`sources_cxx` entries are the raw strings `make -n -B`
    handed to the compiler -- VPATH directory (backslashes, as progen wrote them in the
    Makefile) plus a `/` before the basename (added by make's own path functions). So the
    natural sort key is the path with every separator normalised to a single character --
    specifically backslash, matching `records.progen_relpath()` and the key `BuildManifest`
    itself sorts sources by -- not the raw mixed-separator string. `libs` similarly carries
    the linker's `-l`/`-Wl,--start-group` decoration; the bare project library names (this
    test's concern) live under `libs_origin`.
    """
    includes = RECIPE["includes"]
    assert includes[0] == "."
    assert len(includes) == len(set(includes))
    c_sources = RECIPE["sources_c"]
    assert c_sources == sorted(c_sources, key=lambda s: s.replace("/", "\\"))
    cxx_sources = RECIPE["sources_cxx"]
    assert cxx_sources == sorted(cxx_sources, key=lambda s: s.replace("/", "\\"))
    project_libs = [lib.removeprefix("-l")
                    for lib in RECIPE["libs_origin"]["project_libs_from_Lib.yaml_gcc_arm_sources"]]
    assert project_libs == ["CMSIS_DSP", "CMSIS_NN", "omv", "tflu"]
    assert RECIPE["defines"] == [
        "NVT_VECTOR_ON_FLASH", "TF_LITE_STATIC_MEMORY", "ARM_MATH_DSP", "ARM_NPU", "ACTIVATION_BUF_SZ=0x100000",
    ]


def test_load_records_fills_file_prefix_map(tmp_path: Path) -> None:
    tdir, app, bsp = _make_template(tmp_path)
    m = records.load_records(tdir, app, bsp)
    assert m.file_prefix_map == [(str(bsp.resolve()), "BSP"), (str(app.resolve()), "APP")]
