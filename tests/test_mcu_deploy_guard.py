"""Round 2 residual: the "which kinds can deploy" message must not 500 when a real app
module's `application_supported()` probe raises something other than `DeployError`.

Kept in its own file (not `tests/test_mcu_deploy_api.py`, being edited concurrently elsewhere)
because it exercises exactly the one regression the Round 2 re-review found: `app.py`'s
`_validate_deploy_request()` (via `_currently_deployable_kind_names()`) and
`deploy_service.run_deploy()`'s twin block both used to call
`project_builder.supported_applications()` with no exception guard at all, so a broken app
module (an `ImportError` from a bad transitive import, say) raised straight through the 400
response the kind gate is supposed to produce -- an unhandled 500 on `POST .../deploy-mcu`.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from tm_local import app as app_module
from tm_local.jobs import JobManager
from tm_local.mcu import deploy_service as svc
from tm_local.mcu import paths as mcu_paths
from tm_local.mcu import project_builder
from tm_local.mcu.errors import DeployError, long_path_reason
from tm_local.model_runtime import ModelRuntime
from tm_local.project_store import ProjectStore


@pytest.fixture(autouse=True)
def _skip_when_installed_too_deep() -> None:
    """A Studio installed past Windows' MAX_PATH budget refuses EVERY deploy with the
    long-path message, which is correct but has nothing to do with the kind gate under test:
    both cases below would then fail on a message about folder names. Skip, with the reason.
    """
    reason = long_path_reason(mcu_paths.MCU_TOOLKIT_ROOT)
    if reason:
        pytest.skip(reason)


def _client(store: ProjectStore) -> TestClient:
    return TestClient(
        app_module.create_app(store=store, jobs=JobManager(max_workers=1), runtime=ModelRuntime())
    )


def _flaky_application_supported():
    """A stand-in for `application_supported` that raises for `imgclass` -- simulating a
    broken transitive import inside a real `apps/image.py` -- and otherwise delegates to the
    real function, so a genuinely supported application (`known_sound`, vendored in this repo)
    still shows up normally."""
    real = project_builder.application_supported

    def _flaky(application: object) -> bool:
        if application == "imgclass":
            raise ImportError("boom: a broken transitive import inside apps/image.py")
        return real(application)

    return _flaky


def test_deploy_of_an_undeployable_kind_is_400_even_if_an_app_module_probe_raises(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """abnormal_sound is refused by the kind gate (`is_mcu_deployable()` is False for it)
    before anything toolchain-related runs, so this reaches `_validate_deploy_request()` ->
    `_currently_deployable_kind_names()` with a real app module's probe raising."""
    monkeypatch.setattr(project_builder, "application_supported", _flaky_application_supported())

    project = store.create_project("abnormal_sound", "Ab")
    with _client(store) as client:
        response = client.post(
            f"/api/projects/{project['id']}/deploy-mcu", json={"board": "NuGestureAI-M55M1"}
        )
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "abnormal_sound" in detail
    assert "不支援部署到開發板" in detail
    assert "known_sound" in detail  # a real, unaffected app module is still listed
    assert "image" not in detail  # the one whose probe raised is not listed


def test_run_deploy_of_an_undeployable_kind_is_a_deploy_error_even_if_a_probe_raises(
    store: ProjectStore, monkeypatch: MonkeyPatch
) -> None:
    """The `deploy_service.py` twin of the fix above: `run_deploy()` must still raise its own
    Chinese `DeployError` (not propagate the app module's own exception) when building the
    "只支援 ..." message for a kind with no application id at all."""
    monkeypatch.setattr(project_builder, "application_supported", _flaky_application_supported())

    project = store.create_project("abnormal_sound", "Ab")
    with pytest.raises(DeployError) as excinfo:
        svc.run_deploy(store, project["id"], "NuGestureAI-M55M1", lambda value, message: None)
    detail = str(excinfo.value)
    assert "abnormal_sound" in detail
    assert "只支援" in detail
    assert "known_sound" in detail
    assert "image" not in detail
