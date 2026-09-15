from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import install_local
from tm_local import runtime_config
from tm_local.mcu import toolchain


def _fake_gcc(bin_dir: Path, version_line: str = "arm-none-eabi-gcc (Fake) 14.2.1 20241119") -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    # A .cmd stub is enough for --version probing on Windows.
    (bin_dir / "arm-none-eabi-gcc.cmd").write_text(f"@echo {version_line}\n", encoding="ascii")
    (bin_dir / "arm-none-eabi-gcc.exe").write_bytes(b"")  # existence check only


def test_env_override_wins(tmp_path: Path, monkeypatch) -> None:
    bin_dir = tmp_path / "tc" / "bin"
    _fake_gcc(bin_dir)
    monkeypatch.setenv("TM_ARM_GCC_BIN", str(bin_dir))
    dirs = toolchain.candidate_bin_dirs()
    assert dirs[0] == ("env", bin_dir.resolve())


def test_local_runtime_dir_is_a_candidate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TM_ARM_GCC_BIN", raising=False)
    local = tmp_path / "arm-gnu-toolchain"
    _fake_gcc(local / "bin")
    _fake_gcc(local / "arm-gnu-toolchain-14.2.rel1" / "bin")
    monkeypatch.setattr(toolchain, "LOCAL_TOOLCHAIN_ROOT", local)
    labels = [label for label, _ in toolchain.candidate_bin_dirs()]
    assert labels.count("local") == 2


def test_probe_toolchain_reports_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TM_ARM_GCC_BIN", raising=False)
    monkeypatch.setattr(toolchain, "candidate_bin_dirs", lambda: [("env", tmp_path / "nope")])
    info = toolchain.probe_toolchain()
    assert info["available"] is False and info["bin_dir"] is None
    assert info["checked"] == [str(tmp_path / "nope")]


def test_probe_toolchain_real_if_installed() -> None:
    info = toolchain.probe_toolchain()
    if not info["available"]:
        pytest.skip("no arm-none-eabi-gcc on this machine")
    assert "arm-none-eabi-gcc" in info["version"]
    tc = toolchain.find_toolchain()
    assert tc is not None and tc.tool("gcc").exists()


def test_probe_nulink_env(tmp_path: Path, monkeypatch) -> None:
    exe = tmp_path / "M55M1_M5531" / "NuLink.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setenv("TM_NULINK_EXE", str(exe))
    assert toolchain.probe_nulink() == {"available": True, "path": str(exe.resolve())}
    monkeypatch.setenv("TM_NULINK_EXE", str(tmp_path / "missing.exe"))
    assert toolchain.probe_nulink()["available"] is False


def test_probe_all_shape(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(toolchain, "probe_toolchain", lambda: {"available": False, "bin_dir": None, "version": None, "source": None, "checked": []})
    monkeypatch.setattr(toolchain, "probe_nulink", lambda: {"available": False, "path": None})
    monkeypatch.setattr(toolchain, "probe_pyocd", lambda: {"available": False, "path": None})
    monkeypatch.setattr(toolchain, "probe_numicro_pack", lambda: {"available": False, "path": None})
    info = toolchain.probe_all()
    assert set(info) == {
        "available", "toolchain", "nulink", "pyocd", "numicro_pack", "vela", "toolkit", "probed_at",
    }
    assert info["available"] is False


def test_probe_nulink_env_is_authoritative_over_program_files(tmp_path: Path, monkeypatch) -> None:
    # A Program Files install that would otherwise match must not override an explicit
    # (and here, missing) TM_NULINK_EXE.
    program_files = tmp_path / "Program Files (x86)"
    real = program_files / "Nuvoton Tools" / "NuLink Command Tool" / "M55M1_M5531" / "NuLink.exe"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"")
    monkeypatch.setenv("ProgramFiles(x86)", str(program_files))
    monkeypatch.setenv("TM_NULINK_EXE", str(tmp_path / "missing.exe"))
    assert toolchain.probe_nulink() == {"available": False, "path": None}


def test_probe_pyocd_env(tmp_path: Path, monkeypatch) -> None:
    exe = tmp_path / "pyocd.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("TM_PYOCD_EXE", str(exe))
    assert toolchain.probe_pyocd() == {"available": True, "path": str(exe.resolve())}
    monkeypatch.setenv("TM_PYOCD_EXE", str(tmp_path / "missing.exe"))
    assert toolchain.probe_pyocd() == {"available": False, "path": None}


def test_the_newest_installed_pack_wins(tmp_path) -> None:
    """Arm's pack manager keeps every version it has ever downloaded side by side."""
    for version in ("1.0.1", "3.1.4-rc.3", "2.9.0"):
        (tmp_path / f"Nuvoton.NuMicroM55_DFP.{version}.pack").write_bytes(b"PK\x03\x04")
    found = toolchain.probe_numicro_pack(roots=[tmp_path])
    assert found["available"] is True
    assert found["path"].endswith("Nuvoton.NuMicroM55_DFP.3.1.4-rc.3.pack")


def test_a_release_pack_outranks_the_same_version_release_candidate(tmp_path) -> None:
    """3.1.4 is newer than 3.1.4-rc.3, and a plain string sort gets this backwards."""
    for version in ("3.1.4-rc.3", "3.1.4"):
        (tmp_path / f"Nuvoton.NuMicroM55_DFP.{version}.pack").write_bytes(b"PK\x03\x04")
    found = toolchain.probe_numicro_pack(roots=[tmp_path])
    assert found["path"].endswith("Nuvoton.NuMicroM55_DFP.3.1.4.pack")


def test_no_pack_installed_is_reported_not_raised(tmp_path) -> None:
    """A missing pack is an ordinary 'not available', the same shape probe_nulink() returns."""
    assert toolchain.probe_numicro_pack(roots=[tmp_path]) == {"available": False, "path": None}


def test_probe_numicro_pack_env(tmp_path: Path, monkeypatch) -> None:
    pack = tmp_path / "Nuvoton.NuMicroM55_DFP.3.1.4.pack"
    pack.write_bytes(b"PK\x03\x04")
    monkeypatch.setenv("TM_NUMICRO_PACK", str(pack))
    assert toolchain.probe_numicro_pack() == {"available": True, "path": str(pack.resolve())}
    monkeypatch.setenv("TM_NUMICRO_PACK", str(tmp_path / "missing.pack"))
    assert toolchain.probe_numicro_pack() == {"available": False, "path": None}


def test_local_numicro_pack_root_is_a_search_location(tmp_path: Path, monkeypatch) -> None:
    """`runtime/numicro-pack/` is `scripts/setup_voiceai_flash.py`'s download destination, so
    it must be searched unconditionally -- like `LOCAL_TOOLCHAIN_ROOT` is for
    `probe_toolchain()` -- without depending on `TM_NUMICRO_PACK` or a `runtime_config` write,
    and it must keep working after a `runtime_config` reset.
    """
    monkeypatch.delenv("TM_NUMICRO_PACK", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    local = tmp_path / "numicro-pack"
    local.mkdir()
    pack = local / "Nuvoton.NuMicroM55_DFP.3.1.4-rc.3.pack"
    pack.write_bytes(b"PK\x03\x04")
    monkeypatch.setattr(toolchain, "LOCAL_NUMICRO_PACK_ROOT", local)
    monkeypatch.setattr(runtime_config, "load_runtime_config", lambda: {"mcu": {"available": False}})
    assert toolchain.probe_numicro_pack() == {"available": True, "path": str(pack.resolve())}


@pytest.mark.parametrize(
    "stdout",
    [b"null", b'"boom"', b"", b"Traceback (most recent call last):\n  ...\nImportError: x\n"],
)
def test_probe_mcu_toolchain_survives_bad_probe_output(monkeypatch, stdout: bytes) -> None:
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(install_local, "run", fake_run)
    info = install_local.probe_mcu_toolchain({})
    assert isinstance(info, dict)
    assert info["available"] is False


def test_runtime_config_keeps_mcu_block() -> None:
    payload = runtime_config.normalize_runtime_config({"mcu": {"available": True, "toolchain": {"bin_dir": "X"}}})
    assert payload["mcu"] == {"available": True, "toolchain": {"bin_dir": "X"}}
    assert payload["selected_backend"] == "windows_tensorflow_cpu"
    public = runtime_config.public_runtime_info(payload)
    assert public["mcu"]["available"] is True
    default = runtime_config.normalize_runtime_config({})
    assert default["mcu"] == {"available": False}
