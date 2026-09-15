from __future__ import annotations

from pathlib import Path

import pytest

from tm_local.mcu import paths, vela
from tm_local.mcu.errors import DeployError

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mcu" / "vww4_128_128_INT8.tflite"


def test_arena_formula_matches_numl() -> None:
    # NuML's add_activation_size_section() floors: int(sram*1.2)+1024, then & ~1023.
    # known_sound d10 example (151,360 B): int(151360*1.2)=181632; +1024=182656;
    # floor to 1024 -> 182272 (verified against main_cpp_codegen.py, not the earlier
    # 183296 figure -- that was a transcription error in an earlier draft of this test).
    assert vela.arena_size_from_sram(147 * 1024 + 832) == 182272
    assert vela.arena_size_from_sram(0) == 1024
    assert vela.arena_size_from_sram(1000) % 1024 == 0
    # Pin the "+1024 then floor" behaviour: int(1707*1.2) == 2048 exactly (already a
    # 1024 multiple), so if the +1024 pad were dropped this would wrongly stay at 2048.
    assert vela.arena_size_from_sram(1707) == 3072


def test_vela_command_shape(tmp_path: Path) -> None:
    cmd = vela.vela_command(Path("m.tflite"), tmp_path, "Performance", 716800)
    assert cmd[0] == str(paths.VELA_EXE)
    assert "--accelerator-config=ethos-u55-256" in cmd
    assert "--optimise=Performance" in cmd
    assert f"--config={paths.VELA_INI}" in cmd
    assert "--memory-mode=Shared_Sram" in cmd
    assert "--system-config=Ethos_U55_High_End_Embedded" in cmd
    assert cmd[cmd.index("--arena-cache-size") + 1] == "716800"
    assert cmd[-1] == "--output-dir=."


def test_parse_summary(tmp_path: Path) -> None:
    csv_path = tmp_path / "x_summary_Ethos_U55_High_End_Embedded.csv"
    csv_path.write_text(
        "experiment,network,accelerator_configuration,system_config,memory_mode,core_clock,"
        "arena_cache_size,sram_memory_used,off_chip_flash_memory_used\n"
        "default,x,Ethos_U55_256,Ethos_U55_High_End_Embedded,Shared_Sram,500000000.0,716800,128.36,309.92\n",
        encoding="utf-8",
    )
    sram, flash = vela.parse_summary(csv_path)
    assert sram == int(128.36 * 1024) and flash == int(309.92 * 1024)


def test_parse_summary_empty_file_raises_deploy_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty_summary.csv"
    csv_path.write_text("", encoding="utf-8")
    with pytest.raises(DeployError, match="摘要檔格式不符"):
        vela.parse_summary(csv_path)


def test_parse_summary_header_only_raises_deploy_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "header_only_summary.csv"
    csv_path.write_text("experiment,network,sram_memory_used,off_chip_flash_memory_used\n", encoding="utf-8")
    with pytest.raises(DeployError, match="摘要檔格式不符"):
        vela.parse_summary(csv_path)


def test_parse_summary_missing_column_raises_deploy_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_column_summary.csv"
    csv_path.write_text("experiment,network\ndefault,x\n", encoding="utf-8")
    with pytest.raises(DeployError, match="sram_memory_used"):
        vela.parse_summary(csv_path)


def test_parse_summary_non_numeric_raises_deploy_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "non_numeric_summary.csv"
    csv_path.write_text(
        "experiment,network,sram_memory_used,off_chip_flash_memory_used\n"
        "default,x,not-a-number,309.92\n",
        encoding="utf-8",
    )
    with pytest.raises(DeployError, match="摘要檔格式不符"):
        vela.parse_summary(csv_path)


def test_compile_real_model(tmp_path: Path) -> None:
    if not paths.VELA_EXE.is_file():
        pytest.skip("vela exe not vendored")
    result = vela.compile_model(FIXTURE, tmp_path)
    assert result.vela_tflite.is_file() and result.vela_tflite.name == "vww4_128_128_INT8_vela.tflite"
    assert result.summary_csv.is_file()
    assert 100_000 < result.sram_bytes < 400_000
    assert result.arena_bytes == vela.arena_size_from_sram(result.sram_bytes)
    assert result.optimise == "Performance"
    assert result.npu_ops >= 1


def test_budget_exceeded_raises(tmp_path: Path) -> None:
    if not paths.VELA_EXE.is_file():
        pytest.skip("vela exe not vendored")
    with pytest.raises(DeployError, match="arena"):
        vela.compile_model(FIXTURE, tmp_path, arena_budget_bytes=16 * 1024)
