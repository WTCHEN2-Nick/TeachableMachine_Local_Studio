from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tm_local.mcu import paths
from tm_local.mcu.errors import DeployError

CDC = paths.APPS_ROOT / "common" / "cdc"
pytestmark = pytest.mark.skipif(not (CDC / "uvc_composite").is_dir(), reason="cdc sources not vendored")

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@pytest.mark.parametrize(
    "variant, files",
    [
        ("uvc_composite", ["numl_cdc.h", "numl_cdc.c", "numl_cdc_retarget.c", "numl_overlay.h", "numl_overlay.c"]),
        ("cdc_only", ["numl_cdc.h", "numl_usbd.h", "numl_cdc.c", "numl_usbd.c", "numl_cdc_retarget.c"]),
    ],
)
def test_files_present(variant: str, files: list[str]) -> None:
    for name in files:
        assert (CDC / variant / name).is_file(), name


def test_interface_numbers() -> None:
    """uvc_composite renders decimal interface numbers (CDC beside UVC on 2/3);
    cdc_only renders them as zero-padded hex (CDC owns 0/1 with nothing else
    present) -- this is how the App Builder's own two writers format the
    @@COMM_IF@@/@@DATA_IF@@ tokens, verified against tools/usb_cdc_runtime.py
    and tools/usb_cdc_device.py rather than assumed.
    """
    composite = (CDC / "uvc_composite" / "numl_cdc.h").read_text(encoding="utf-8")
    only = (CDC / "cdc_only" / "numl_cdc.h").read_text(encoding="utf-8")
    assert re.search(r"NUML_CDC_COMM_INTERFACE\s+2\b", composite)
    assert re.search(r"NUML_CDC_DATA_INTERFACE\s+3\b", composite)
    assert re.search(r"NUML_CDC_COMM_INTERFACE\s+0x00\b", only)
    assert re.search(r"NUML_CDC_DATA_INTERFACE\s+0x01\b", only)


@pytest.mark.parametrize("variant", ["uvc_composite", "cdc_only"])
def test_retarget_seam_patched(variant: str) -> None:
    text = (CDC / variant / "numl_cdc_retarget.c").read_text(encoding="utf-8")
    assert '#include "retarget.c"' in text
    assert "../../../../" not in text
    gnu = text.index("defined(__GNUC__)")
    include_gcc = text.index("retarget_GCC.c")
    undef = text.index("#undef stdout_putchar", gnu)
    assert gnu < undef < include_gcc
    assert "Modified for Teachable Machine Local Studio" in text


@pytest.mark.parametrize("variant", ["uvc_composite", "cdc_only"])
def test_gcc_write_binds_to_tee(variant: str, tmp_path: Path) -> None:
    """Compile the retarget shim with the real BSP and prove _write() reaches the CDC tee."""
    gcc = shutil.which("arm-none-eabi-gcc")
    if not gcc or not paths.toolkit_available():
        pytest.skip("toolchain or toolkit missing")
    bsp = paths.BSP_ROOT
    src = CDC / variant / "numl_cdc_retarget.c"
    obj = tmp_path / "retarget.o"
    cmd = [
        gcc, "-mcpu=cortex-m55", "-mthumb", "-mfloat-abi=hard", "-O1", "-std=c11", "-c", str(src), "-o", str(obj),
        f"-I{CDC / variant}", f"-I{bsp / 'Library/StdDriver/src'}", f"-I{bsp / 'Library/StdDriver/inc'}",
        f"-I{bsp / 'Library/Device/Nuvoton/M55M1/Include'}", f"-I{bsp / 'Library/CMSIS/Core/Include'}",
        "-DNVT_VECTOR_ON_FLASH",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, cwd=tmp_path, check=False,
        creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    assert proc.returncode == 0, proc.stderr
    nm_exe = str(Path(gcc).parent / "arm-none-eabi-nm")
    nm = subprocess.run(
        [nm_exe, "-C", str(obj)], capture_output=True, text=True, timeout=60, cwd=tmp_path, check=False,
        creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    symbols = nm.stdout
    assert re.search(r"\bT _write\b", symbols), symbols
    assert re.search(r"\bT stdout_putchar\b", symbols), symbols
    objdump_exe = str(Path(gcc).parent / "arm-none-eabi-objdump")
    objdump = subprocess.run(
        [objdump_exe, "-d", str(obj)], capture_output=True, text=True, timeout=60, cwd=tmp_path, check=False,
        creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    write_body = objdump.stdout.split("<_write>:")[1].split("\n\n")[0]
    assert "stdout_putchar" in write_body, write_body


def test_stage_cdc_places_every_variants_files_where_the_template_expects_them(
    tmp_path: Path,
) -> None:
    """One staging helper for all three apps; each variant's five files land in two dirs."""
    from tm_local.mcu.apps._cdc import CDC_VARIANTS, stage_cdc

    for variant, (headers, sources) in CDC_VARIANTS.items():
        app_dir = tmp_path / variant
        app_dir.mkdir()
        stage_cdc(app_dir, variant)
        for name in headers:
            staged = app_dir / "Device" / "include" / name
            assert staged.read_bytes() == (CDC / variant / name).read_bytes(), name
        for name in sources:
            staged = app_dir / "Device" / "CDC" / name
            assert staged.read_bytes() == (CDC / variant / name).read_bytes(), name
        assert len(headers) + len(sources) == 5


def test_stage_cdc_reports_a_missing_source_in_chinese(tmp_path: Path, monkeypatch) -> None:
    from tm_local.mcu.apps import _cdc

    monkeypatch.setattr(_cdc, "APPS_ROOT", tmp_path / "not-vendored")
    with pytest.raises(DeployError, match="找不到 USB CDC 來源檔"):
        _cdc.stage_cdc(tmp_path / "app", "cdc_only")


def test_stage_cdc_rejects_an_unknown_variant(tmp_path: Path) -> None:
    from tm_local.mcu.apps._cdc import stage_cdc

    with pytest.raises(DeployError, match="未知的 USB CDC 版本"):
        stage_cdc(tmp_path / "app", "no_such_variant")


def test_app_modules_share_the_one_staging_helper() -> None:
    """Three verbatim copies of this loop is how they drift; one import each keeps them honest."""
    from tm_local.mcu.apps import image, known_sound, kws

    assert image._CDC_VARIANT == "uvc_composite"
    assert known_sound.CDC_VARIANT == kws.CDC_VARIANT == "cdc_only"
    assert known_sound.CDC_HEADERS == kws.CDC_HEADERS == ("numl_cdc.h", "numl_usbd.h")
    assert known_sound.CDC_SOURCES == kws.CDC_SOURCES == (
        "numl_cdc.c",
        "numl_usbd.c",
        "numl_cdc_retarget.c",
    )
    assert image._CDC_HEADERS == ("numl_cdc.h", "numl_overlay.h")
    assert image._CDC_SOURCES == ("numl_cdc.c", "numl_cdc_retarget.c", "numl_overlay.c")
    for module in (image, known_sound, kws):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "stage_cdc(" in source
        # The copy loop that used to be duplicated here, identified by its own local names.
        assert "include_dir.mkdir" not in source, module.__name__
