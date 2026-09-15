"""Flashing helpers (`tm_local/mcu/flash.py`) and `POST /api/projects/{id}/deploy-mcu/flash`.

`find_msc_drives`/`_volume_label` are monkeypatched everywhere so nothing here needs real
hardware, a real USB MSC drive, or a real Nu-Link Command Tool -- only `tmp_path` directories
and a fake `.cmd`/`.exe` stub path that `run_tool` never actually executes (it, too, is
monkeypatched). Non-Windows behavior is reached through the `flash._is_windows` seam rather than
by mutating the real `sys.platform`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from tm_local import app as app_module
from tm_local.jobs import JobManager
from tm_local.mcu import flash
from tm_local.model_runtime import ModelRuntime
from tm_local.project_store import ProjectStore


def test_copy_to_msc_checks_label(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "firmware.bin"
    src.write_bytes(b"\x01" * 16)
    drive = tmp_path / "drive"
    drive.mkdir()
    monkeypatch.setattr(flash, "_volume_label", lambda root: "M55M1")
    result = flash.copy_to_msc(src, str(drive))
    assert result["ok"] and (drive / "firmware.bin").read_bytes() == b"\x01" * 16
    monkeypatch.setattr(flash, "_volume_label", lambda root: "OTHER")
    assert flash.copy_to_msc(src, str(drive))["ok"] is False


def test_copy_to_msc_unreadable_label_does_not_say_none(tmp_path: Path, monkeypatch) -> None:
    """When the label can't be read at all, the message must say so -- not print `None`."""
    monkeypatch.setattr(flash, "_volume_label", lambda root: None)
    result = flash.copy_to_msc(tmp_path / "firmware.bin", str(tmp_path))
    assert result["ok"] is False
    assert "None" not in result["message"]
    assert "無法讀取卷標" in result["message"]


def test_find_msc_drives_non_windows_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(flash, "_is_windows", lambda: False)
    assert flash.find_msc_drives() == []


def test_nulink_program_parses_failure_text(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, env, timeout):
        calls.append(cmd)
        text = "Target Chip is Not Supported" if "-C" in cmd else "ok"
        return subprocess.CompletedProcess(cmd, 0, stdout=text)

    monkeypatch.setattr(flash, "run_tool", fake_run)
    exe = tmp_path / "M55M1_M5531" / "NuLink.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    result = flash.nulink_program(tmp_path / "f.bin", exe)
    assert result["ok"] is False and "Not Supported" in result["message"]
    assert calls[0][1] == "-C" and len(calls) == 1


def test_nulink_program_ignores_lowercase_fail_substring(tmp_path: Path, monkeypatch) -> None:
    """A benign log line containing lowercase "fail" must not be treated as a failure."""
    monkeypatch.setattr(
        flash,
        "run_tool",
        lambda cmd, cwd, env, timeout: subprocess.CompletedProcess(
            cmd, 0, stdout="0 failures, connection ok"
        ),
    )
    exe = tmp_path / "NuLink.exe"
    exe.write_bytes(b"")
    result = flash.nulink_program(tmp_path / "f.bin", exe)
    assert result["ok"] is True


def test_nulink_program_runs_steps_in_order_with_correct_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run(cmd, cwd, env, timeout):
        calls.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, 0, stdout="done")

    monkeypatch.setattr(flash, "run_tool", fake_run)
    exe = tmp_path / "NuLink.exe"
    exe.write_bytes(b"")
    bin_path = tmp_path / "f.bin"
    result = flash.nulink_program(bin_path, exe)
    assert result["ok"] and "APROM" in result["log"]
    assert [cmd[1] for cmd, _cwd in calls] == ["-C", "-W", "-S"]
    assert calls[0][0] == [str(exe), "-C"]
    assert calls[1][0] == [str(exe), "-W", "APROM", str(bin_path.resolve()), "1"]
    assert calls[2][0] == [str(exe), "-S"]
    assert all(cwd == exe.parent for _cmd, cwd in calls)


def test_nulink_program_timeout(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, cwd, env, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(flash, "run_tool", fake_run)
    exe = tmp_path / "NuLink.exe"
    exe.write_bytes(b"")
    result = flash.nulink_program(tmp_path / "f.bin", exe, timeout_seconds=1)
    assert result["ok"] is False
    assert "逾時" in result["message"]


def test_the_pyocd_command_targets_the_pack_device_name(tmp_path: Path) -> None:
    """App Builder verified this exact command on hardware; the pieces must not drift."""
    cmd = flash.pyocd_command(
        tmp_path / "firmware.bin", tmp_path / "pyocd.exe", tmp_path / "NuMicroM55_DFP.pack"
    )
    assert cmd[1] == "load"
    assert "--pack" in cmd
    # The pack declares M55M1R2LJAE; M55M1R2LJC7E is the ordering part number and is not in it.
    assert "M55M1R2LJAE" in cmd
    assert "--base-address" in cmd and "0x00100000" in cmd
    # pyocd reset -m hw leaves nRESET asserted on a Nu-Link2, so nothing here may reset.
    assert "reset" not in cmd


def test_pyocd_program_runs_a_single_load_command(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run(cmd, cwd, env, timeout):
        calls.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, 0, stdout="Erase done\nProgramming done\n")

    monkeypatch.setattr(flash, "run_tool", fake_run)
    exe = tmp_path / "pyocd.exe"
    exe.write_bytes(b"")
    pack = tmp_path / "Nuvoton.NuMicroM55_DFP.3.1.4.pack"
    pack.write_bytes(b"PK\x03\x04")
    bin_path = tmp_path / "f.bin"
    bin_path.write_bytes(b"\x00")
    result = flash.pyocd_program(bin_path, exe, pack)
    assert result["ok"] is True
    assert len(calls) == 1
    assert calls[0][0][:2] == [str(exe), "load"]
    assert calls[0][1] == exe.parent


def test_pyocd_program_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        flash,
        "run_tool",
        lambda cmd, cwd, env, timeout: subprocess.CompletedProcess(cmd, 1, stdout="error: no probe"),
    )
    exe = tmp_path / "pyocd.exe"
    exe.write_bytes(b"")
    pack = tmp_path / "Nuvoton.NuMicroM55_DFP.3.1.4.pack"
    pack.write_bytes(b"PK\x03\x04")
    result = flash.pyocd_program(tmp_path / "f.bin", exe, pack)
    assert result["ok"] is False
    assert "no probe" in result["message"]


def test_pyocd_program_timeout(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, cwd, env, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(flash, "run_tool", fake_run)
    exe = tmp_path / "pyocd.exe"
    exe.write_bytes(b"")
    pack = tmp_path / "Nuvoton.NuMicroM55_DFP.3.1.4.pack"
    pack.write_bytes(b"PK\x03\x04")
    result = flash.pyocd_program(tmp_path / "f.bin", exe, pack, timeout_seconds=1)
    assert result["ok"] is False
    assert "逾時" in result["message"]


def _client(store: ProjectStore, jobs: JobManager | None = None) -> TestClient:
    return TestClient(
        app_module.create_app(
            store=store, jobs=jobs or JobManager(max_workers=1), runtime=ModelRuntime()
        )
    )


def _prepare_build(store: ProjectStore, board: str) -> tuple[str, Path]:
    from tm_local.mcu.boards import board_for

    project = store.create_project("image", "Img")
    target = store.project_dir(project["id"]) / "models" / "mcu" / board
    target.mkdir(parents=True)
    (target / "firmware.bin").write_bytes(b"\x02" * 32)
    (target / "deploy_report.json").write_text(
        json.dumps(
            {
                "board": board,
                "flash_method": board_for(board).flash_method,
                "bin": {
                    "file": "firmware.bin",
                    "bytes": 32,
                    "sha256": hashlib.sha256(b"\x02" * 32).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return project["id"], target


def test_flash_endpoint_msc(store: ProjectStore, tmp_path: Path, monkeypatch) -> None:
    pid, _target = _prepare_build(store, "NuGestureAI-M55M1")
    drive = tmp_path / "M55M1"
    drive.mkdir()
    monkeypatch.setattr(flash, "find_msc_drives", lambda label="M55M1": [str(drive)])
    monkeypatch.setattr(flash, "_volume_label", lambda root: "M55M1")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc"},
        )
    assert response.status_code == 200, response.text
    assert (drive / "firmware.bin").read_bytes() == b"\x02" * 32
    assert response.json()["ok"] is True


def test_flash_endpoint_msc_with_explicit_drive_skips_auto_detect(
    store: ProjectStore, monkeypatch
) -> None:
    """A well-formed `drive` in the request must be validated, then used as-is (no scan)."""
    pid, _ = _prepare_build(store, "NuGestureAI-M55M1")
    calls: list[str] = []

    def fake_copy(bin_path: Path, drive: str) -> dict:
        calls.append(drive)
        return {"ok": True, "message": "已複製", "target": str(Path(drive) / "firmware.bin")}

    def fail_if_called(label: str = "M55M1") -> list[str]:
        raise AssertionError("explicit drive must skip auto-detection")

    monkeypatch.setattr(flash, "copy_to_msc", fake_copy)
    monkeypatch.setattr(flash, "find_msc_drives", fail_if_called)
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc", "drive": "D:"},
        )
    assert response.status_code == 200, response.text
    assert calls == ["D:"]


def test_flash_endpoint_strips_and_validates_drive(store: ProjectStore, monkeypatch) -> None:
    """A drive with surrounding whitespace is stripped before the format check runs."""
    pid, _ = _prepare_build(store, "NuGestureAI-M55M1")
    calls: list[str] = []
    monkeypatch.setattr(
        flash,
        "copy_to_msc",
        lambda bin_path, drive: calls.append(drive) or {"ok": True, "message": "ok", "target": ""},
    )
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc", "drive": " D:\\ "},
        )
    assert response.status_code == 200, response.text
    assert calls == ["D:\\"]


def test_flash_endpoint_rejects_malformed_drive(store: ProjectStore, monkeypatch) -> None:
    pid, _ = _prepare_build(store, "NuGestureAI-M55M1")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc", "drive": "C:\\stuff\\evil"},
        )
    assert response.status_code == 400
    assert "磁碟機代號" in response.json()["detail"]


def test_flash_endpoint_rejects_drive_with_trailing_newline(
    store: ProjectStore, monkeypatch
) -> None:
    """`re.fullmatch` (not `match`) so a trailing newline cannot sneak a bad drive past `$`."""
    pid, _ = _prepare_build(store, "NuGestureAI-M55M1")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc", "drive": "D:\nrm -rf /"},
        )
    assert response.status_code == 400
    assert "磁碟機代號" in response.json()["detail"]


def test_flash_endpoint_rejects_wrong_method_and_missing_drive(
    store: ProjectStore, monkeypatch
) -> None:
    """NuGestureAI-M55M1 accepts msc AND nulink (both verified on hardware), so the mismatch
    case here has to name a method the board genuinely does not list -- pyocd, which only
    NuMaker-VoiceAI-M55M1 supports."""
    pid, _ = _prepare_build(store, "NuGestureAI-M55M1")
    monkeypatch.setattr(flash, "find_msc_drives", lambda label="M55M1": [])
    with _client(store) as client:
        method_mismatch = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "pyocd"},
        )
        assert method_mismatch.status_code == 400
        assert "燒錄方式" in method_mismatch.json()["detail"]

        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc"},
        )
        assert response.status_code == 400 and "M55M1" in response.json()["detail"]


def test_flash_endpoint_gestureai_accepts_nulink_as_well(
    store: ProjectStore, tmp_path: Path, monkeypatch
) -> None:
    """The user verified Nu-Link also works on their NuGestureAI-M55M1; both listed methods
    must actually flash, not just msc (the documented default)."""
    pid, _ = _prepare_build(store, "NuGestureAI-M55M1")
    from tm_local.mcu import toolchain

    exe = tmp_path / "NuLink.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(toolchain, "probe_nulink", lambda: {"available": True, "path": str(exe)})
    monkeypatch.setattr(
        flash, "run_tool", lambda cmd, cwd, env, timeout: subprocess.CompletedProcess(cmd, 0, stdout="ok")
    )
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "nulink"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True


def test_flash_endpoint_detects_tampered_bin(store: ProjectStore, monkeypatch) -> None:
    pid, target = _prepare_build(store, "NuGestureAI-M55M1")
    (target / "firmware.bin").write_bytes(b"\x03" * 32)
    monkeypatch.setattr(flash, "find_msc_drives", lambda label="M55M1": ["X:\\"])
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc"},
        )
    assert response.status_code == 400
    assert "部署紀錄損毀或不符" in response.json()["detail"]


def test_flash_endpoint_detects_corrupt_report(store: ProjectStore, monkeypatch) -> None:
    """A hand-edited/truncated deploy_report.json must 400, not 500."""
    pid, target = _prepare_build(store, "NuGestureAI-M55M1")
    (target / "deploy_report.json").write_text("not json at all {{{", encoding="utf-8")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc"},
        )
    assert response.status_code == 400
    assert "部署紀錄損毀或不符" in response.json()["detail"]


def test_flash_endpoint_detects_non_object_report(store: ProjectStore, monkeypatch) -> None:
    """Valid JSON that isn't an object (e.g. a bare list) must also 400, not 500."""
    pid, target = _prepare_build(store, "NuGestureAI-M55M1")
    (target / "deploy_report.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc"},
        )
    assert response.status_code == 400
    assert "部署紀錄損毀或不符" in response.json()["detail"]


def test_flash_endpoint_no_build_yet(store: ProjectStore) -> None:
    project = store.create_project("image", "Img")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{project['id']}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc"},
        )
    assert response.status_code == 400
    assert "韌體" in response.json()["detail"]


def test_flash_endpoint_is_409_while_another_job_runs(
    store: ProjectStore, monkeypatch
) -> None:
    """Flashing during a running deploy must 409, not read a half-written firmware.bin."""
    pid, _ = _prepare_build(store, "NuGestureAI-M55M1")
    jobs = JobManager(max_workers=1)
    release = threading.Event()

    def blocking(progress: object) -> dict:
        release.wait(10)
        return {}

    jobs.submit("deploy", pid, blocking)
    try:
        with _client(store, jobs) as client:
            response = client.post(
                f"/api/projects/{pid}/deploy-mcu/flash",
                json={"board": "NuGestureAI-M55M1", "method": "msc"},
            )
    finally:
        release.set()
    assert response.status_code == 409
    assert response.json()["detail"] == "此專案已有訓練、匯出或部署工作進行中。"


def test_flash_endpoint_nulink_unavailable(store: ProjectStore, monkeypatch) -> None:
    pid, _ = _prepare_build(store, "NuMaker-M55M1")
    from tm_local.mcu import toolchain

    monkeypatch.setattr(toolchain, "probe_nulink", lambda: {"available": False, "path": None})
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuMaker-M55M1", "method": "nulink"},
        )
    assert response.status_code == 400
    assert "Nu-Link" in response.json()["detail"]


def test_flash_endpoint_nulink_success(store: ProjectStore, tmp_path: Path, monkeypatch) -> None:
    pid, _ = _prepare_build(store, "NuMaker-M55M1")
    from tm_local.mcu import toolchain

    exe = tmp_path / "NuLink.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(toolchain, "probe_nulink", lambda: {"available": True, "path": str(exe)})
    calls: list[tuple[list[str], Path]] = []

    def fake_run(cmd, cwd, env, timeout):
        calls.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr(flash, "run_tool", fake_run)
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuMaker-M55M1", "method": "nulink"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    assert [cmd[1] for cmd, _cwd in calls] == ["-C", "-W", "-S"]
    assert all(cwd == exe.parent for _cmd, cwd in calls)


def test_flash_endpoint_nulink_failure(store: ProjectStore, tmp_path: Path, monkeypatch) -> None:
    pid, _ = _prepare_build(store, "NuMaker-M55M1")
    from tm_local.mcu import toolchain

    exe = tmp_path / "NuLink.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(toolchain, "probe_nulink", lambda: {"available": True, "path": str(exe)})

    def fake_run(cmd, cwd, env, timeout):
        text = "Target Chip is Not Supported" if "-C" in cmd else "ok"
        return subprocess.CompletedProcess(cmd, 0, stdout=text)

    monkeypatch.setattr(flash, "run_tool", fake_run)
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuMaker-M55M1", "method": "nulink"},
        )
    assert response.status_code == 400
    assert "Not Supported" in response.json()["detail"]


def test_flash_endpoint_pyocd_unavailable(store: ProjectStore, monkeypatch) -> None:
    pid, _ = _prepare_build(store, "NuMaker-VoiceAI-M55M1")
    from tm_local.mcu import toolchain

    monkeypatch.setattr(toolchain, "probe_pyocd", lambda: {"available": False, "path": None})
    monkeypatch.setattr(
        toolchain, "probe_numicro_pack", lambda: {"available": True, "path": "C:\\pack.pack"}
    )
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuMaker-VoiceAI-M55M1", "method": "pyocd"},
        )
    assert response.status_code == 400
    assert "Nu-Link2" in response.json()["detail"]


def test_flash_endpoint_pyocd_pack_unavailable(store: ProjectStore, monkeypatch) -> None:
    """pyocd itself being present is not enough -- the DFP pack must also be found."""
    pid, _ = _prepare_build(store, "NuMaker-VoiceAI-M55M1")
    from tm_local.mcu import toolchain

    monkeypatch.setattr(
        toolchain, "probe_pyocd", lambda: {"available": True, "path": "C:\\pyocd.exe"}
    )
    monkeypatch.setattr(toolchain, "probe_numicro_pack", lambda: {"available": False, "path": None})
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuMaker-VoiceAI-M55M1", "method": "pyocd"},
        )
    assert response.status_code == 400
    assert "Nu-Link2" in response.json()["detail"]


def test_flash_endpoint_pyocd_success(store: ProjectStore, tmp_path: Path, monkeypatch) -> None:
    pid, _ = _prepare_build(store, "NuMaker-VoiceAI-M55M1")
    from tm_local.mcu import toolchain

    pyocd_exe = tmp_path / "pyocd.exe"
    pyocd_exe.write_bytes(b"")
    pack = tmp_path / "Nuvoton.NuMicroM55_DFP.3.1.4.pack"
    pack.write_bytes(b"PK\x03\x04")
    monkeypatch.setattr(
        toolchain, "probe_pyocd", lambda: {"available": True, "path": str(pyocd_exe)}
    )
    monkeypatch.setattr(
        toolchain, "probe_numicro_pack", lambda: {"available": True, "path": str(pack)}
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, env, timeout):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="done")

    monkeypatch.setattr(flash, "run_tool", fake_run)
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuMaker-VoiceAI-M55M1", "method": "pyocd"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    assert len(calls) == 1 and calls[0][1] == "load"


def test_flash_endpoint_rejects_pyocd_method_for_a_nulink_board(
    store: ProjectStore, monkeypatch
) -> None:
    """The method-vs-board mismatch check must catch a pyocd request for a Nu-Link board too."""
    pid, _ = _prepare_build(store, "NuMaker-M55M1")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuMaker-M55M1", "method": "pyocd"},
        )
    assert response.status_code == 400
    assert "燒錄方式" in response.json()["detail"]


def test_flash_import_stays_lazy() -> None:
    """`import tm_local.app` must not pull `tm_local.mcu.flash` into sys.modules."""
    import subprocess
    import sys

    code = (
        "import sys; import tm_local.app as app;"
        " print('tm_local.mcu.flash' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", proc.stdout
