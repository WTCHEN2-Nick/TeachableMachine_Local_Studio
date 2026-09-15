"""R1: the port must be checked before the FastAPI app is created.

`_create_application()` runs `create_app()`, which sweeps `workspace/tmp/mcu/<hex>` work dirs
and pending firmware download copies (`tm_local.mcu.housekeeping.sweep_temp_root()`) on the
assumption that this is the only Local Studio instance starting up. A student double-clicking
`02_START.bat` while another instance is already running (or mid-deploy) must be told the port
is taken *before* that sweep runs -- not after it has already deleted the live instance's
in-progress work dir and pending firmware download copies.

`--check` is the one mode that must keep creating the app with no port involved at all: that is
the whole point of `--check` (import/build the app, then exit).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import start_local


class _FakeSessionLog:
    latest_path = Path("FAKE.log")


@pytest.fixture(autouse=True)
def _no_real_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep main() from writing real log files or touching the real runtime environment.

    `main()` imports these with a fresh `from ... import ...` on every call, so patching the
    attribute on the source module is what that local import picks up.
    """
    from tm_local import logging_utils, runtime_config

    monkeypatch.setattr(
        logging_utils, "setup_session_logging", lambda log_root: _FakeSessionLog()
    )
    monkeypatch.setattr(runtime_config, "apply_runtime_environment", lambda: object())
    monkeypatch.setattr(
        runtime_config, "public_runtime_info", lambda cfg: {"backend_label": "test"}
    )


def test_port_in_use_stops_before_the_app_is_created(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(start_local, "_port_is_in_use", lambda host, port: True)
    calls: list[None] = []
    monkeypatch.setattr(start_local, "_create_application", lambda: calls.append(None))

    exit_code = start_local.main(["--port", "8765", "--no-browser"])

    assert exit_code == 3
    assert calls == []  # create_app()'s temp sweep must never have run


def test_check_mode_creates_the_app_without_checking_the_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected(host: str, port: int) -> bool:
        raise AssertionError("--check must not consult the port at all")

    monkeypatch.setattr(start_local, "_port_is_in_use", _unexpected)
    calls: list[None] = []

    def _fake_create_application():
        calls.append(None)
        return object()

    monkeypatch.setattr(start_local, "_create_application", _fake_create_application)

    exit_code = start_local.main(["--check"])

    assert exit_code == 0
    assert calls == [None]  # --check still builds the app


def test_invalid_port_is_also_rejected_before_the_app_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[None] = []
    monkeypatch.setattr(start_local, "_create_application", lambda: calls.append(None))

    exit_code = start_local.main(["--port", "0"])

    assert exit_code == 2
    assert calls == []
