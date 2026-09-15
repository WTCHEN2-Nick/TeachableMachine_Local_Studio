"""HTTP surface for MCU deploy: status probe, deploy job, firmware download.

Every test drives the real FastAPI app through TestClient with a real ProjectStore in
tmp_path. The build pipeline itself is replaced (`app.run_mcu_deploy`) and the toolchain probe
is monkeypatched, so these run on a machine with no Arm toolchain installed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from tm_local import app as app_module
from tm_local.jobs import JobManager
from tm_local.mcu import boards as boards_mod
from tm_local.mcu import deploy_service, housekeeping, toolchain
from tm_local.mcu import paths as mcu_paths
from tm_local.mcu.errors import TOOLCHAIN_MISSING_MESSAGE, DeployError, long_path_reason
from tm_local.model_runtime import ModelRuntime
from tm_local.project_store import ProjectStore
from tm_local.tflite_export import ExportError
from tm_local.utils import sha256_file

BOARD = "NuGestureAI-M55M1"
BUSY_MESSAGE = "此專案已有訓練、匯出或部署工作進行中。"

READY_PROBE = {
    "available": True,
    "toolchain": {"available": True, "bin_dir": "C:/gcc/bin", "version": "gcc 14.2"},
    "nulink": {"available": False, "path": None},
    "vela": {"available": True, "path": "vela.exe"},
    "toolkit": {"available": True, "bsp_tag": "v3.00.001"},
    "probed_at": "2026-09-12T00:00:00+00:00",
}


def _client(store: ProjectStore, jobs: JobManager | None = None) -> TestClient:
    return TestClient(
        app_module.create_app(
            store=store, jobs=jobs or JobManager(max_workers=1), runtime=ModelRuntime()
        )
    )


def _ready(monkeypatch: MonkeyPatch) -> None:
    """Everything a deploy needs is present -- including a short enough install path.

    A Studio checked out past Windows' MAX_PATH budget makes `app.mcu_unavailable_reason()`
    refuse every deploy with the long-path message, so each case below would fail on a
    sentence about folder names instead of the thing it is testing. Skip with that reason.
    """
    reason = long_path_reason(mcu_paths.MCU_TOOLKIT_ROOT)
    if reason:
        pytest.skip(reason)
    monkeypatch.setattr(toolchain, "probe_all", lambda: dict(READY_PROBE))


def _fake_outcome(store: ProjectStore, project_id: str, board: str) -> deploy_service.DeployOutcome:
    """Write what a successful `run_deploy` leaves behind, minus the real firmware."""
    models = store.project_dir(project_id) / "models"
    target = deploy_service.deploy_dir(models, board)
    target.mkdir(parents=True, exist_ok=True)
    bin_path = target / "firmware.bin"
    bin_path.write_bytes(b"\x00" * 64)
    zip_path = target / f"Img_{board}_firmware.zip"
    zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    report = {
        "schema_version": 1,
        "board": board,
        "bin": {"file": "firmware.bin", "bytes": 64, "sha256": "x"},
        "image": {
            "flash_used": 64,
            "flash_limit": 2097152,
            "sram01_used": 1,
            "sram01_limit": 1048576,
        },
        "flash_method": "msc",
        "labels": ["a", "b"],
    }
    (target / "deploy_report.json").write_text(json.dumps(report), encoding="utf-8")
    return deploy_service.DeployOutcome(
        board=board, bin_path=bin_path, zip_path=zip_path, report=report
    )


def _trained_image_project(store: ProjectStore, name: str = "Img") -> str:
    """A project in the state the deploy endpoint requires: trained, not yet exported."""
    project = store.create_project("image", name)
    keras_path = store.project_dir(project["id"]) / "models" / "image_classifier.keras"
    keras_path.write_bytes(b"fake keras")
    store.set_training_result(
        project["id"],
        report={
            "settings": {"image_size": 224},
            "dataset": {"calibration_samples": 0},
            "evaluation": {},
            "conversion": {"state": "pending", "models": {}},
        },
        artifacts={"keras": keras_path.name},
    )
    return project["id"]


def _wait(client: TestClient, job_id: str) -> dict:
    for _ in range(500):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in {"completed", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_status_reports_probe_boards_and_kinds(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        toolchain,
        "probe_all",
        lambda: {
            "available": False,
            "toolchain": {"available": False},
            "nulink": {"available": False},
            "vela": {"available": True},
            "toolkit": {"available": True},
            "probed_at": "x",
        },
    )
    with _client(store) as client:
        payload = client.get("/api/mcu/status").json()
    assert payload["available"] is False
    assert payload["toolchain"] == {"available": False}
    assert payload["probed_at"] == "x"
    assert [b["name"] for b in payload["boards"]] == [
        "NuMaker-M55M1",
        "NuGestureAI-M55M1",
        "NuMaker-VoiceAI-M55M1",
    ]
    assert set(payload["boards"][0]) == {
        "name",
        "label",
        "has_lcd",
        "flash_method",
        "flash_methods",
        "known_sound_max_depth",
    }
    # Only kinds whose firmware app module exists in this build -- today all three of
    # them. Registry order, not alphabetical: PROJECT_KINDS puts `audio` between `image`
    # and `known_sound`, and `abnormal_sound` has no firmware application at all.
    assert payload["deployable_kinds"] == ["image", "audio", "known_sound"]


def test_status_probes_live_on_every_request(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    calls: list[int] = []

    def probe() -> dict:
        calls.append(1)
        return dict(READY_PROBE)

    monkeypatch.setattr(toolchain, "probe_all", probe)
    with _client(store) as client:
        assert client.get("/api/mcu/status").json()["available"] is True
        assert client.get("/api/mcu/status").json()["available"] is True
    assert len(calls) == 2


def test_deploy_job_success_and_download(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    _ready(monkeypatch)
    pid = _trained_image_project(store)
    assert store.get_project(pid)["deploy"] == {}
    monkeypatch.setattr(
        app_module,
        "run_mcu_deploy",
        lambda s, project_id, board, progress: _fake_outcome(s, project_id, board),
    )
    with _client(store) as client:
        response = client.post(f"/api/projects/{pid}/deploy-mcu", json={"board": BOARD})
        assert response.status_code == 200, response.text
        job = _wait(client, response.json()["id"])
        assert job["state"] == "completed", job
        assert job["job_type"] == "deploy"
        result = job["result"]
        assert result["board"] == BOARD
        assert result["filename"] == f"Img_{BOARD}_firmware.zip"
        assert result["report"]["bin"]["bytes"] == 64
        assert result["project"]["deploy"][BOARD]["bin"]["bytes"] == 64

        download = client.get(result["download_url"])
        assert download.status_code == 200
        assert download.content.startswith(b"PK")
        assert client.get(result["download_url"]).status_code == 404  # one-time token

        original = deploy_service.deploy_dir(store.project_dir(pid) / "models", BOARD)
        original_zip = original / f"Img_{BOARD}_firmware.zip"
        assert original_zip.is_file(), "download must serve a copy, not the stored firmware"

        again = client.post(f"/api/projects/{pid}/deploy-mcu/download", json={"board": BOARD})
        assert again.status_code == 200
        assert again.json()["filename"] == f"Img_{BOARD}_firmware.zip"
        assert client.get(again.json()["download_url"]).status_code == 200
        assert original_zip.is_file()

        fetched = client.get(f"/api/projects/{pid}").json()
        assert fetched["deploy"][BOARD]["board"] == BOARD


def test_deploy_rejects_unknown_board_and_kind(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    _ready(monkeypatch)
    abnormal = store.create_project("abnormal_sound", "Ab")
    image_id = _trained_image_project(store)
    with _client(store) as client:
        bad_kind = client.post(
            f"/api/projects/{abnormal['id']}/deploy-mcu", json={"board": "NuMaker-M55M1"}
        )
        assert bad_kind.status_code == 400
        assert "abnormal_sound" in bad_kind.json()["detail"]
        bad_board = client.post(
            f"/api/projects/{image_id}/deploy-mcu", json={"board": "M467"}
        )
        assert bad_board.status_code == 400
        assert "M467" in bad_board.json()["detail"]


def test_status_survives_a_broken_board_registry(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """A truncated/hand-edited boards.json must not 500 the endpoint that explains it."""
    from tm_local.mcu import boards

    def broken() -> dict:
        raise KeyError("label")

    monkeypatch.setattr(toolchain, "probe_all", lambda: dict(READY_PROBE))
    monkeypatch.setattr(boards, "load_boards", broken)
    with _client(store) as client:
        response = client.get("/api/mcu/status")
    assert response.status_code == 200, response.text
    assert response.json()["boards"] == []
    assert response.json()["deployable_kinds"] == ["image", "audio", "known_sound"]


def _without_app_module(monkeypatch: MonkeyPatch, application: str) -> None:
    """Simulate a build in which `tm_local/mcu/apps/<name>.py` does not exist yet.

    Every deployable kind now ships its app module, so "this kind has no firmware support"
    can only be produced by patching the lookup. `application_supported()` reaches
    `app_module()` through the module global, so this one patch covers both the status
    listing and the deploy-time capability gate.
    """
    from tm_local.mcu import project_builder

    real_app_module = project_builder.app_module

    def _lookup(name: object):
        if str(name) == application:
            raise DeployError(f"此版本尚未支援 {name} 韌體應用")
        return real_app_module(name)

    monkeypatch.setattr(project_builder, "app_module", _lookup)


def test_status_follows_the_app_modules_that_exist(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """A kind is deployable exactly while its firmware app module is importable."""
    _ready(monkeypatch)
    _without_app_module(monkeypatch, "kws")
    with _client(store) as client:
        payload = client.get("/api/mcu/status").json()
    assert payload["deployable_kinds"] == ["image", "known_sound"]


def test_status_treats_a_broken_app_module_probe_as_merely_unsupported(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """R3: application_supported() raising something other than DeployError (a bad import
    inside a future apps/<name>.py, say) must not 500 the endpoint that explains what a
    student can deploy today -- it must report only that one kind as unsupported."""
    _ready(monkeypatch)
    from tm_local.mcu import project_builder

    real_application_supported = project_builder.application_supported

    def _flaky(application: object) -> bool:
        if application == "imgclass":
            raise RuntimeError("boom: a bad import inside apps/imgclass.py")
        return real_application_supported(application)

    monkeypatch.setattr(project_builder, "application_supported", _flaky)
    with _client(store) as client:
        response = client.get("/api/mcu/status")
    assert response.status_code == 200, response.text
    deployable = response.json()["deployable_kinds"]
    assert "image" not in deployable  # the one whose probe raised
    assert "known_sound" in deployable  # unrelated kinds still reported correctly


def test_supported_applications_reflects_every_importable_app_module(
    monkeypatch: MonkeyPatch,
) -> None:
    """`project_builder.supported_applications()` backs the "which kinds can deploy today"
    message in app.py -- it must actually track apps/<name>.py, not a fixed list."""
    from tm_local.mcu import project_builder

    assert project_builder.supported_applications() == ["imgclass", "kws", "known_sound"]
    _without_app_module(monkeypatch, "kws")
    assert project_builder.supported_applications() == ["imgclass", "known_sound"]


def test_deploy_of_a_kind_without_firmware_support_is_400_before_any_job(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """F1: a kind with no app module must be refused up front, not after INT8 + Vela."""
    _ready(monkeypatch)
    _without_app_module(monkeypatch, "kws")
    project = store.create_project("audio", "Aud")
    jobs = JobManager(max_workers=1)
    calls: list[tuple] = []
    monkeypatch.setattr(app_module, "run_mcu_deploy", lambda *a: calls.append(a))
    try:
        with _client(store, jobs) as client:
            response = client.post(
                f"/api/projects/{project['id']}/deploy-mcu", json={"board": BOARD}
            )
    finally:
        jobs._executor.shutdown(wait=False)
    assert response.status_code == 400
    assert response.json()["detail"] == "此版本尚未支援 kws 韌體應用"
    assert jobs.recent() == []
    assert calls == []


def test_deploy_of_an_audio_project_passes_the_capability_gate(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """The same request gets past the capability gate now that apps/kws.py is importable."""
    _ready(monkeypatch)
    project = store.create_project("audio", "Aud")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{project['id']}/deploy-mcu", json={"board": BOARD}
        )
    # Past the kind gate; now refused by the *next* check instead.
    assert response.status_code == 400
    assert response.json()["detail"] == "請先完成訓練，再部署到開發板。"


def test_deploy_refuses_an_install_path_too_long_for_windows(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """F4: a deep install root fails gcc's include resolution with a Chinese message."""
    _ready(monkeypatch)
    pid = _trained_image_project(store)
    monkeypatch.setattr(
        mcu_paths, "MCU_TOOLKIT_ROOT", Path("C:/" + "verylongfolder/" * 12 + "mcu_toolkit")
    )
    jobs = JobManager(max_workers=1)
    try:
        with _client(store, jobs) as client:
            response = client.post(f"/api/projects/{pid}/deploy-mcu", json={"board": BOARD})
    finally:
        jobs._executor.shutdown(wait=False)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail.startswith("Studio 資料夾路徑太長（")
    assert "C:\\TM_Studio" in detail or r"C:\TM_Studio" in detail
    assert jobs.recent() == []


def test_startup_sweeps_orphaned_firmware_temp_files(tmp_path: Path) -> None:
    """F3: at startup every firmware ZIP copy and work dir under TEMP_ROOT is an orphan."""
    stale_zip = tmp_path / "tm_local_firmware_abcd.zip"
    stale_zip.write_bytes(b"PK\x05\x06")
    work = tmp_path / "mcu" / "deadbeef"
    (work / "NN_ImgClassInference" / "GCC").mkdir(parents=True)
    (work / "NN_ImgClassInference" / "main.cpp").write_text("int main(){}", encoding="utf-8")
    keep_model = tmp_path / "tm_local_model_1234.zip"
    keep_model.write_bytes(b"PK")
    keep_project = tmp_path / "tm_local_project_1234.zip"
    keep_project.write_bytes(b"PK")

    removed = housekeeping.sweep_temp_root(tmp_path)

    assert sorted(p.name for p in removed) == ["deadbeef", "tm_local_firmware_abcd.zip"]
    assert not stale_zip.exists() and not work.exists()
    assert (tmp_path / "mcu").is_dir()  # the parent stays; only the work dirs go
    assert keep_model.is_file() and keep_project.is_file()  # other exports are not ours
    assert housekeeping.sweep_temp_root(tmp_path) == []  # idempotent
    assert housekeeping.sweep_temp_root(tmp_path / "does-not-exist") == []  # never raises


def test_create_app_sweeps_the_temp_root_once(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(housekeeping, "sweep_temp_root", lambda *a, **k: calls.append(1) or [])
    app_module.create_app(store=store, jobs=JobManager(max_workers=1), runtime=ModelRuntime())
    assert len(calls) == 1


def test_deploy_requires_a_trained_project(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """Untrained -> Chinese 400 up front, instead of an English ExportError inside the job."""
    _ready(monkeypatch)
    project = store.create_project("image", "Img")
    jobs = JobManager(max_workers=1)
    try:
        with _client(store, jobs) as client:
            response = client.post(
                f"/api/projects/{project['id']}/deploy-mcu", json={"board": BOARD}
            )
    finally:
        jobs._executor.shutdown(wait=False)
    assert response.status_code == 400
    assert response.json()["detail"] == "請先完成訓練，再部署到開發板。"
    assert jobs.recent() == []


def test_deploy_rejects_when_toolchain_missing(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        toolchain,
        "probe_all",
        lambda: {**READY_PROBE, "available": False, "toolchain": {"available": False}},
    )
    pid = _trained_image_project(store)
    with _client(store) as client:
        response = client.post(f"/api/projects/{pid}/deploy-mcu", json={"board": BOARD})
    assert response.status_code == 400
    assert "Arm GNU Toolchain" in response.json()["detail"]
    assert response.json()["detail"] == TOOLCHAIN_MISSING_MESSAGE


def test_deploy_is_409_while_another_job_runs(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    _ready(monkeypatch)
    pid = _trained_image_project(store)
    jobs = JobManager(max_workers=1)
    release = threading.Event()

    def blocking(progress: object) -> dict:
        release.wait(10)
        return {}

    jobs.submit("train", pid, blocking)
    try:
        with _client(store, jobs) as client:
            response = client.post(f"/api/projects/{pid}/deploy-mcu", json={"board": BOARD})
    finally:
        release.set()
        jobs._executor.shutdown(wait=False)  # this JobManager is the test's own, not the app's
    assert response.status_code == 409
    assert response.json()["detail"] == BUSY_MESSAGE


@pytest.mark.parametrize(
    ("error", "message"),
    # run_deploy() runs the Strict INT8 export first, so a project that was never exported
    # fails with ExportError, not DeployError. Neither is caught on the way out.
    [(DeployError, "請先匯出 INT8"), (ExportError, "INT8 conversion failed")],
)
def test_deploy_error_surfaces_as_failed_job(
    store: ProjectStore, monkeypatch: MonkeyPatch, error: type[Exception], message: str
) -> None:
    _ready(monkeypatch)
    pid = _trained_image_project(store)

    def boom(*args: object, **kwargs: object) -> None:
        raise error(message)

    monkeypatch.setattr(app_module, "run_mcu_deploy", boom)
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu", json={"board": "NuMaker-M55M1"}
        )
        assert response.status_code == 200, response.text
        job = _wait(client, response.json()["id"])
    assert job["state"] == "failed"
    assert message in job["error"]
    assert error.__name__ in job["error"]


def test_download_without_build_is_400(store: ProjectStore) -> None:
    project = store.create_project("image", "Img")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{project['id']}/deploy-mcu/download", json={"board": "NuMaker-M55M1"}
        )
    assert response.status_code == 400
    assert "韌體" in response.json()["detail"]


def test_importing_app_does_not_pull_mcu_build_dependencies() -> None:
    """create_app() must stay free of the MCU build stack (yaml/jinja2/deploy_service)."""
    code = (
        "import sys; import tm_local.app as app; app.create_app();"
        " print(sorted(m for m in"
        " ('tm_local.mcu.deploy_service', 'yaml', 'jinja2', 'tensorflow')"
        " if m in sys.modules))"
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
    assert proc.stdout.strip() == "[]", proc.stdout


def _seed_firmware(store: ProjectStore, kind: str, board: str, flash_method: str) -> str:
    """A project with a finished build on disk, ready for POST .../deploy-mcu/flash.

    The flash endpoint never rebuilds: it checks the recorded sha256 and then runs the
    method's preflight, so a 64-byte stand-in firmware.bin is enough.
    """
    project = store.create_project(kind, "Snd")
    target = deploy_service.deploy_dir(store.project_dir(project["id"]) / "models", board)
    target.mkdir(parents=True, exist_ok=True)
    bin_path = target / "firmware.bin"
    bin_path.write_bytes(b"\x00" * 64)
    report = {
        "schema_version": 1,
        "board": board,
        "kind": kind,
        "flash_method": flash_method,
        "bin": {"file": "firmware.bin", "bytes": 64, "sha256": sha256_file(bin_path)},
        "labels": ["a", "b"],
    }
    (target / "deploy_report.json").write_text(json.dumps(report), encoding="utf-8")
    return project["id"]


@pytest.mark.parametrize(
    ("kind", "board", "method", "expected"),
    [
        ("known_sound", "NuGestureAI-M55M1", "msc", "INFO threshold["),
        ("audio", "NuMaker-M55M1", "nulink", "KWS DETECTED"),
    ],
)
def test_flash_preflight_failure_describes_this_kinds_output(
    kind: str, board: str, method: str, expected: str,
    store: ProjectStore, monkeypatch: MonkeyPatch,
) -> None:
    """A failed flash hands over the flashing steps -- which end with what to look at.

    For an audio build that is a COM port and nothing else: no camera window opens and the LCD
    is never written. Both preflights (no MSC drive, no Nu-Link tool) took the `kind` default,
    so a Known Sound student was told to watch a window that would stay black.
    """
    _ready(monkeypatch)
    from tm_local.mcu import flash as flash_mod
    from tm_local.mcu import toolchain as toolchain_mod

    monkeypatch.setattr(flash_mod, "find_msc_drives", list)  # no M55M1 drive present
    monkeypatch.setattr(toolchain_mod, "probe_nulink", lambda: {"available": False, "path": None})
    pid = _seed_firmware(store, kind, board, method)
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash", json={"board": board, "method": method}
        )
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    # The board label names its LCD/camera, so the claim is about the result paragraph alone.
    result_text = detail.split("結果觀看")[1]
    assert expected in result_text and "COM port" in result_text
    # (the GestureAI wording does say there is NO camera window -- the opposite instruction)
    assert "相機 app" not in result_text and "攝影機影像" not in result_text
    assert "LCD" not in result_text
    # ...and it is exactly what the server would write into README_FLASH_zh-TW.txt.
    assert detail == deploy_service.flash_instructions(boards_mod.board_for(board), kind)


def test_flash_preflight_failure_still_describes_the_camera_for_an_image_build(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """The image wording is not merely gone -- it still reaches the kind that needs it."""
    _ready(monkeypatch)
    from tm_local.mcu import flash as flash_mod

    monkeypatch.setattr(flash_mod, "find_msc_drives", list)  # no M55M1 drive present
    pid = _seed_firmware(store, "image", "NuGestureAI-M55M1", "msc")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{pid}/deploy-mcu/flash",
            json={"board": "NuGestureAI-M55M1", "method": "msc"},
        )
    assert response.status_code == 400, response.text
    assert "相機 app" in response.json()["detail"].split("結果觀看")[1]
