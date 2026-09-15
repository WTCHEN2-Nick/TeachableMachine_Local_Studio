"""The Windows rename race in `deploy_service._swap_staging_into_place()`.

Measured on the maintainer's machine (2026-09-14), reproducibly: `staging.rename(target)` can
fail with `PermissionError: [WinError 5]` because the on-access antivirus is still unpacking
`source.zip` / the firmware ZIP that `staging` just gained, to scan them -- the SOURCE directory
is what is locked, not the destination, and it clears in well under a second. A previous fix
guessed the destination was in a delete-pending state and renamed the old `target` aside first,
which is harmless but does not address the actual cause: the rename that moves `staging` into
place was, and still needs to be, retried.

These tests never touch a real antivirus: they monkeypatch `pathlib.Path.rename` to fail a
controlled number of times (or forever) for a chosen source path, so the retry/restore logic in
`deploy_service._rename_with_retry()` and `_swap_staging_into_place()` is exercised the same way
on every machine, deterministically and in milliseconds.
"""
from __future__ import annotations

import os
import pathlib
from pathlib import Path

import pytest

from tm_local.mcu import deploy_service as svc
from tm_local.mcu.errors import DeployError

_REAL_RENAME = pathlib.Path.rename


def _fail_n_times_then_succeed(target_path: Path, fail_times: int):
    """`Path.rename` replacement: raises for the first `fail_times` renames whose SOURCE is
    `target_path`, then delegates to the real rename (for every source, always)."""
    remaining = {"n": fail_times}

    def _rename(self: Path, dst):
        if str(self) == str(target_path) and remaining["n"] > 0:
            remaining["n"] -= 1
            raise PermissionError(
                f"[WinError 5] 存取被拒。: '{self}' -> '{dst}'"
            )
        return _REAL_RENAME(self, dst)

    return _rename


def _always_fail_for(*locked_paths: Path):
    """`Path.rename` replacement: raises forever for any of `locked_paths` as the SOURCE,
    delegates to the real rename for everything else."""
    locked = {str(p) for p in locked_paths}

    def _rename(self: Path, dst):
        if str(self) in locked:
            raise PermissionError(
                f"[WinError 5] 存取被拒。: '{self}' -> '{dst}'"
            )
        return _REAL_RENAME(self, dst)

    return _rename


def _fast_budget(monkeypatch: pytest.MonkeyPatch, budget_seconds: float) -> None:
    """Make every retry loop inside the module give up in `budget_seconds` instead of the
    production ~10s default, so a permanently-locked-path test finishes in milliseconds."""
    real = svc._rename_with_retry

    def _wrapped(src: Path, dst: Path, *, budget_seconds: float = budget_seconds) -> None:
        return real(src, dst, budget_seconds=budget_seconds)

    monkeypatch.setattr(svc, "_rename_with_retry", _wrapped)


def _seed_target_and_staging(tmp_path: Path) -> tuple[Path, Path]:
    """A previous build at `target` and a freshly-built replacement at its `.tmp` staging dir."""
    target = tmp_path / "mcu" / "BoardX"
    staging = target.with_name(f"{target.name}.tmp")
    target.mkdir(parents=True)
    (target / "firmware.bin").write_text("old-build", encoding="utf-8")
    staging.mkdir(parents=True)
    (staging / "firmware.bin").write_text("new-build", encoding="utf-8")
    return target, staging


# --- _rename_with_retry() itself --------------------------------------------------------------


def test_rename_with_retry_retries_through_transient_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_text("payload", encoding="utf-8")
    monkeypatch.setattr(pathlib.Path, "rename", _fail_n_times_then_succeed(src, fail_times=4))

    svc._rename_with_retry(src, dst, budget_seconds=1.0)

    assert not src.exists()
    assert dst.read_text(encoding="utf-8") == "payload"


def test_rename_with_retry_gives_up_after_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_text("payload", encoding="utf-8")
    monkeypatch.setattr(pathlib.Path, "rename", _always_fail_for(src))

    with pytest.raises(PermissionError):
        svc._rename_with_retry(src, dst, budget_seconds=0.05)

    assert src.exists()  # never moved
    assert not dst.exists()


# --- _swap_staging_into_place() -----------------------------------------------------------------


def test_swap_completes_once_the_transient_lock_on_staging_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured failure mode: `staging.rename(target)` fails a handful of times (the
    antivirus still has `source.zip` open) and then succeeds on its own -- the swap must finish
    with the NEW build in place and no leftover `.tmp` / `.old-*` directories."""
    target, staging = _seed_target_and_staging(tmp_path)
    monkeypatch.setattr(pathlib.Path, "rename", _fail_n_times_then_succeed(staging, fail_times=5))

    svc._swap_staging_into_place(staging, target)

    assert (target / "firmware.bin").read_text(encoding="utf-8") == "new-build"
    assert not staging.exists()
    retired = target.with_name(f"{target.name}.old-{os.getpid()}")
    assert not retired.exists()


def test_swap_restores_the_previous_build_when_the_lock_never_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock outlasts the whole retry budget: the swap must give up, put the STUDENT'S
    PREVIOUS BUILD back exactly where it was, and raise a `DeployError` with Chinese text that
    tells them what happened and what to do next -- not a raw `PermissionError`."""
    target, staging = _seed_target_and_staging(tmp_path)
    monkeypatch.setattr(pathlib.Path, "rename", _always_fail_for(staging))
    _fast_budget(monkeypatch, budget_seconds=0.05)

    with pytest.raises(DeployError) as excinfo:
        svc._swap_staging_into_place(staging, target)

    text = str(excinfo.value)
    assert "防毒軟體" in text
    assert "再按一次" in text and "建置" in text
    assert isinstance(excinfo.value.__cause__, PermissionError)

    # The previous build is intact, in place, under its ORIGINAL name -- not stranded aside.
    assert target.is_dir()
    assert (target / "firmware.bin").read_text(encoding="utf-8") == "old-build"
    retired = target.with_name(f"{target.name}.old-{os.getpid()}")
    assert not retired.exists()


def test_swap_preserves_the_retired_copy_when_even_the_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doubly unlucky: the forward rename never clears AND renaming the retired copy back also
    fails. The previous build must not be lost -- it stays on disk under its retired name, named
    in the error, and the caller must be told this is serious enough for manual recovery."""
    target, staging = _seed_target_and_staging(tmp_path)
    retired = target.with_name(f"{target.name}.old-{os.getpid()}")
    monkeypatch.setattr(pathlib.Path, "rename", _always_fail_for(staging, retired))
    _fast_budget(monkeypatch, budget_seconds=0.05)

    with pytest.raises(DeployError) as excinfo:
        svc._swap_staging_into_place(staging, target)

    text = str(excinfo.value)
    assert "嚴重錯誤" in text
    assert "手動" in text
    assert str(retired) in text

    # Not lost: sitting under the retired name, exactly as it was.
    assert not target.exists()
    assert retired.is_dir()
    assert (retired / "firmware.bin").read_text(encoding="utf-8") == "old-build"
    # The failed candidate is untouched too -- `_swap_staging_into_place()` never deletes it;
    # that is the caller's (`run_deploy`'s `finally`) job.
    assert (staging / "firmware.bin").read_text(encoding="utf-8") == "new-build"
