"""The reference/ scripts stay runnable, thin and honest.

They are teaching material: a student must be able to run every one of them from the
Studio root with the project ``.venv``. These tests keep them importable (``py_compile``),
self-describing (``--help``), TensorFlow-free on the dry-run path, and free of the two
wordings the architecture forbids (``convert_all(`` inside a training path, and calling a
known_sound score a probability).
"""

from __future__ import annotations

import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"
SCRIPTS = sorted(p for p in REF.rglob("*.py") if p.name != "_common.py")

KIND_FOLDERS = ("image", "audio_kws", "known_sound", "abnormal_sound")
# folder -> (project kind, settings keys the dry-run table must show)
DRY_RUN_CASES = {
    "image": ("image", ("augmentation_level", "fine_tune_blocks", "image_size")),
    "audio_kws": ("audio", ("augmentation_level", "session_disjoint_validation", "mel_bins")),
    "known_sound": ("known_sound", ("encoder_depth", "head_dropout", "waveform_augment_level")),
    "abnormal_sound": ("abnormal_sound", ("sensitivity", "bottleneck_dim", "top_k")),
}


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        check=False,
    )


def test_reference_layout() -> None:
    for folder in KIND_FOLDERS:
        assert (REF / folder / "README_zh-TW.md").is_file()
        assert (REF / folder / "train.py").is_file()
        assert (REF / folder / "run_tflite.py").is_file()
    for folder in ("audio_kws", "known_sound", "abnormal_sound"):
        assert (REF / folder / "preprocess.py").is_file()
    assert (REF / "audio_kws" / "mcu_tables.py").is_file()
    assert (REF / "README_zh-TW.md").is_file()
    assert (REF / "_common.py").is_file()
    assert not list(REF.rglob("*.bat"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_scripts_compile_and_help(script: Path) -> None:
    py_compile.compile(str(script), doraise=True)
    proc = _run([sys.executable, str(script), "--help"])
    assert proc.returncode == 0, proc.stderr
    assert (
        "--project" in proc.stdout
        or "--wav" in proc.stdout
        or "--model" in proc.stdout
        or "--frontend" in proc.stdout
    )


@pytest.mark.parametrize("folder", KIND_FOLDERS)
def test_train_dry_run_prints_settings(tmp_path: Path, folder: str) -> None:
    from tm_local.project_store import ProjectStore

    kind, expected_keys = DRY_RUN_CASES[folder]
    workspace = tmp_path / "ws"
    store = ProjectStore(workspace / "projects", workspace / "tmp")
    project = store.create_project(kind, f"Ref {folder}")
    env = {
        "TM_LOCAL_WORKSPACE": str(workspace),
        "PYTHONUTF8": "1",
        "TF_USE_LEGACY_KERAS": "1",
    }
    proc = _run(
        [sys.executable, str(REF / folder / "train.py"), "--project", project["id"], "--dry-run"],
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    for key in expected_keys:
        assert key in proc.stdout, f"{key} missing from {folder} dry-run table"
    # The dry run must stay TensorFlow-free: it exists so a student can check settings in
    # a second rather than waiting for the whole backend to import.
    assert "tensorflow" not in proc.stderr.lower()


def test_train_dry_run_finds_project_by_name(tmp_path: Path) -> None:
    from tm_local.project_store import ProjectStore

    workspace = tmp_path / "ws"
    store = ProjectStore(workspace / "projects", workspace / "tmp")
    store.create_project("image", "Named Image Project")
    env = {"TM_LOCAL_WORKSPACE": str(workspace), "PYTHONUTF8": "1"}
    proc = _run(
        [
            sys.executable,
            str(REF / "image" / "train.py"),
            "--project",
            "Named Image Project",
            "--dry-run",
        ],
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "image_size" in proc.stdout

    missing = _run(
        [sys.executable, str(REF / "image" / "train.py"), "--project", "nope", "--dry-run"],
        env=env,
    )
    assert missing.returncode == 2
    assert "Named Image Project" in missing.stderr


def test_train_rejects_the_wrong_kind(tmp_path: Path) -> None:
    from tm_local.project_store import ProjectStore

    workspace = tmp_path / "ws"
    store = ProjectStore(workspace / "projects", workspace / "tmp")
    project = store.create_project("audio", "Wrong Kind")
    env = {"TM_LOCAL_WORKSPACE": str(workspace), "PYTHONUTF8": "1"}
    proc = _run(
        [sys.executable, str(REF / "image" / "train.py"), "--project", project["id"], "--dry-run"],
        env=env,
    )
    assert proc.returncode == 2
    assert "audio" in proc.stderr


def test_mcu_tables_script_prints_header(tmp_path: Path) -> None:
    frontend = ROOT / "tests" / "fixtures" / "mcu" / "audio_frontend_default.json"
    out = tmp_path / "kws_tables.h"
    proc = _run(
        [
            sys.executable,
            str(REF / "audio_kws" / "mcu_tables.py"),
            "--frontend",
            str(frontend),
            "--out",
            str(out),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    text = out.read_text(encoding="utf-8")
    assert "s_hannTable" in text and "s_melWeights" in text and "NUML_MEL_NNZ" in text
    assert "s_melStart" in text and "s_melCount" in text and "s_melOffset" in text
    # The header keeps the firmware's own macro layout (body aligned at column 32).
    assert re.search(r"#define NUML_MEL_BINS\s+\(40U\)", text)
    assert re.search(r"#define NUML_FRAME_LENGTH\s+\(400U\)", text)


def test_reference_tree_keeps_the_architecture_rules() -> None:
    """No TFLite conversion inside a training path, and no 「機率」 for known_sound."""

    for path in SCRIPTS + [REF / "_common.py"]:
        text = path.read_text(encoding="utf-8")
        assert "convert_all(" not in text, f"{path} converts inside a training path"
    for path in sorted(REF.rglob("*.md")) + SCRIPTS:
        if "known_sound" not in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        assert "機率" not in text, f"{path} calls a known_sound score a probability"
        assert "信心分數" in text, f"{path} must call the score 信心分數"
