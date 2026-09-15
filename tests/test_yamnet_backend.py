from __future__ import annotations

import math
import json
import importlib.util
import py_compile
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from tm_local.audio_frontend import encode_wav_bytes
from tm_local.export_service import create_model_export, ensure_export_artifacts
from tm_local.model_runtime import ModelRuntime
from tm_local.project_store import ProjectStore
from tm_local.tflite_export import ExportError, inspect_tflite
from tm_local.training_common import TrainingError
from tm_local.yamnet_anomaly_pipeline import (
    fit_normal_reference,
    score_wav_bytes,
    train_yamnet_abnormal_sound_project,
    write_yamnet_frontend_reference,
    write_yamnet_runner,
)
from tm_local.yamnet_model import (
    YAMNET_EMBEDDING_DIM,
    build_yamnet_models,
    frontend_contract,
    require_yamnet_weights,
    waveform_to_log_mel_patches,
)


def test_yamnet_frontend_contract_and_patch_shape() -> None:
    contract = frontend_contract()
    assert contract["sample_rate"] == 16000
    assert contract["mel_bands"] == 64
    assert contract["model_input_shape"] == [96, 64]
    assert contract["inference_window_seconds"] == 1.0
    assert contract["inference_hop_seconds"] == 0.5
    patches = waveform_to_log_mel_patches(np.zeros(16000, dtype=np.float32))
    assert patches.shape == (1, 96, 64)
    np.testing.assert_allclose(patches, np.log(0.001), rtol=0, atol=1e-6)
    timeline = np.arange(16000, dtype=np.float32) / 16000.0
    waveform = (
        0.2 * np.sin(2 * math.pi * 440.0 * timeline)
        + 0.05 * np.sin(2 * math.pi * 1234.0 * timeline)
    ).astype(np.float32)
    golden = waveform_to_log_mel_patches(waveform)
    np.testing.assert_allclose(
        [
            golden[0, 0, 0],
            golden[0, 0, 10],
            golden[0, 10, 20],
            golden[0, 47, 30],
            golden[0, 95, 63],
            golden.mean(),
            golden.std(),
        ],
        [-5.0287123, 0.8744472, -4.1487508, -6.0187101, -5.8744473, -4.5563855, 2.7907772],
        rtol=0,
        atol=2e-4,
    )


def test_official_yamnet_asset_loads_exact_encoder_topology() -> None:
    assert require_yamnet_weights().is_file()
    full, encoder = build_yamnet_models()
    assert full.count_params() == 3_751_369
    assert encoder.count_params() == 3_217_344
    output = encoder.predict(
        np.full((1, 96, 64), np.log(0.001), dtype=np.float32), verbose=0
    )
    assert output.shape == (1, YAMNET_EMBEDDING_DIM)
    assert np.all(np.isfinite(output))
    timeline = np.arange(16000, dtype=np.float32) / 16000.0
    waveform = (
        0.2 * np.sin(2 * math.pi * 440.0 * timeline)
        + 0.05 * np.sin(2 * math.pi * 1234.0 * timeline)
    ).astype(np.float32)
    embedding = encoder.predict(waveform_to_log_mel_patches(waveform), verbose=0)[0]
    np.testing.assert_allclose(
        embedding[[20, 40, 55, 59, 80, 89, 95, 134]],
        [1.2282885, 0.2848041, 0.4527029, 0.5354860, 0.6472375, 2.0526726, 0.5563542, 1.0053122],
        rtol=0,
        atol=2e-3,
    )
    assert np.linalg.norm(embedding) == pytest.approx(12.106001, abs=2e-3)


def test_normal_reference_is_finite_and_session_balanced() -> None:
    rng = np.random.default_rng(20260901)
    embeddings = rng.normal(size=(12, YAMNET_EMBEDDING_DIM)).astype(np.float32)
    sessions = ["session-a"] * 4 + ["session-b"] * 4 + ["session-c"] * 4
    reference = fit_normal_reference(embeddings, sessions)
    scores = reference.score(embeddings)
    assert reference.center.shape == (YAMNET_EMBEDDING_DIM,)
    assert reference.scale.shape == (YAMNET_EMBEDDING_DIM,)
    assert np.all(reference.scale > 0)
    assert scores.shape == (12,)
    assert np.all(np.isfinite(scores))


def test_exported_runner_uses_explicit_uint8_runtime_and_compiles(tmp_path: Path) -> None:
    frontend_path = write_yamnet_frontend_reference(tmp_path)
    runner = write_yamnet_runner(
        tmp_path,
        "abnormal_sound_yamnet_embedding_uint8.tflite",
        "uint8",
    )
    source = runner.read_text(encoding="utf-8")
    assert "RUNTIME_NAME = 'uint8'" in source
    assert "if name in MODEL_FILENAME" not in source
    assert "def audio_windows" in source
    assert "temporal_verdict" in source
    assert "final_start" not in source
    assert "EXPECTED_FRONTEND" in source
    py_compile.compile(str(runner), doraise=True)
    spec = importlib.util.spec_from_file_location("exported_yamnet_frontend", frontend_path)
    assert spec is not None and spec.loader is not None
    exported_frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exported_frontend)
    stereo = np.column_stack(
        [
            np.full(32, 16384, dtype=np.int16),
            np.full(32, 8192, dtype=np.int16),
        ]
    )
    stereo_path = tmp_path / "stereo.wav"
    wavfile.write(stereo_path, 16000, stereo)
    np.testing.assert_allclose(exported_frontend.read_wav(stereo_path), 0.375, atol=1e-7)
    uint8_path = tmp_path / "uint8.wav"
    wavfile.write(uint8_path, 16000, np.asarray([0, 128, 255], dtype=np.uint8))
    np.testing.assert_allclose(
        exported_frontend.read_wav(uint8_path),
        np.asarray([-1.0, 0.0, 127.0 / 128.0], dtype=np.float32),
        atol=1e-7,
    )


def test_yamnet_training_saves_frozen_encoder_and_normal_reference(tmp_path: Path) -> None:
    store = ProjectStore(
        tmp_path / "workspace" / "projects",
        tmp_path / "workspace" / "tmp",
    )
    project = store.create_project("abnormal_sound", "YAMNet smoke")
    normal_id = project["classes"][0]["id"]
    sample_rate = 16000
    seconds = 20
    timeline = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    rng = np.random.default_rng(601)
    for session_index, frequency in enumerate((180.0, 240.0, 315.0), start=1):
        modulation = 0.65 + 0.20 * np.sin(2 * math.pi * 0.23 * timeline + session_index)
        signal = (
            0.10 * modulation * np.sin(2 * math.pi * frequency * timeline)
            + 0.035 * np.sin(2 * math.pi * (frequency * 2.7) * timeline)
            + 0.006 * rng.normal(size=timeline.size)
        ).astype(np.float32)
        store.add_audio_bytes(
            project["id"],
            normal_id,
            encode_wav_bytes(signal, sample_rate),
            source_name=f"normal_{session_index}.wav",
            recording_session_id=f"session-{session_index}",
        )
    result = train_yamnet_abnormal_sound_project(store, project["id"], {}, None)
    assert result["report"]["detector_backend"] == "yamnet_embedding"
    assert result["report"]["dataset"]["normal_clips"] == 60
    assert result["report"]["history"]["encoder_training_epochs"] == 0
    models = store.project_dir(project["id"]) / "models"
    assert (models / result["artifacts"]["keras"]).is_file()
    assert (models / result["artifacts"]["yamnet_scorer"]).is_file()
    assert (models / result["artifacts"]["yamnet_frontend"]).is_file()
    store.set_training_result(
        project["id"], report=result["report"], artifacts=result["artifacts"]
    )
    runtime = ModelRuntime()
    preview = score_wav_bytes(
        project_dir=store.project_dir(project["id"]),
        artifact_name=result["artifacts"]["keras"],
        runtime_name="keras",
        payload=encode_wav_bytes(signal[-sample_rate:], sample_rate),
        settings=store.get_raw_project(project["id"])["settings"],
        predict=runtime.predict,
    )
    assert preview["verdict"] == "uncertain"
    assert preview["window_verdict"] in {"normal", "abnormal"}
    assert preview["confidence"] == "low"
    assert np.isfinite(preview["anomaly_ratio"])
    assert set(preview["components"]) == {"embedding", "level"}
    exported = ensure_export_artifacts(store, project["id"], ["int8"])
    current = exported["project"]
    assert current["training"]["report"]["dataset"]["calibration_samples"] == 40
    int8_path = models / current["training"]["artifacts"]["int8"]
    audit = inspect_tflite(int8_path)
    assert audit["strict_full_integer"] is True
    assert audit["inputs"][0]["shape"] == [1, 96, 64]
    assert audit["outputs"][0]["shape"] == [1, 1024]
    comparison = current["training"]["report"]["conversion"]["models"]["int8"][
        "comparison"
    ]
    assert comparison["sample_count"] >= 1
    assert comparison["mean_cosine_similarity"] >= 0.95
    assert comparison["minimum_cosine_similarity"] >= 0.80
    assert comparison["mean_relative_l2_error"] <= 0.35
    scorer = json.loads((models / "yamnet_scorer.json").read_text(encoding="utf-8"))
    assert "int8" in scorer["runtimes"]
    for runtime_name in ("keras", "int8"):
        runtime_entry = scorer["runtimes"][runtime_name]
        assert runtime_entry["confidence"]["fpr_upper_bound_95"] == pytest.approx(
            runtime_entry["threshold"]["fpr_upper_bound_95"]
        )
        assert runtime_entry["confidence"]["observed_exceedances"] == runtime_entry[
            "threshold"
        ]["observed_exceedances"]
    original_scorer = json.dumps(scorer, ensure_ascii=False, indent=2) + "\n"
    scorer["runtimes"]["keras"]["reference"]["center"][0] += 0.25
    (models / "yamnet_scorer.json").write_text(
        json.dumps(scorer, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TrainingError, match="binding"):
        score_wav_bytes(
            project_dir=store.project_dir(project["id"]),
            artifact_name=result["artifacts"]["keras"],
            runtime_name="keras",
            payload=encode_wav_bytes(signal[-sample_rate:], sample_rate),
            settings=store.get_raw_project(project["id"])["settings"],
            predict=runtime.predict,
        )
    with pytest.raises(ExportError, match="binding"):
        ensure_export_artifacts(store, project["id"], ["keras"])
    (models / "yamnet_scorer.json").write_text(original_scorer, encoding="utf-8")
    output_zip = tmp_path / "yamnet_int8.zip"
    create_model_export(
        store=store,
        project_id=project["id"],
        selected=["int8"],
        include_c_header=False,
        output_zip=output_zip,
    )
    with zipfile.ZipFile(output_zip) as archive:
        names = set(archive.namelist())
        assert "abnormal_sound_yamnet_embedding_int8.tflite" in names
        assert "yamnet_scorer.json" in names
        assert "yamnet_frontend.json" in names
        assert "yamnet_frontend_reference.py" in names
        assert "run_model.py" in names
        assert "labels.txt" not in names
        runner_dir = tmp_path / "runner"
        archive.extractall(runner_dir)
    long_wav = runner_dir / "four_seconds.wav"
    wavfile.write(long_wav, sample_rate, np.tile(signal[-sample_rate:], 4))
    completed = subprocess.run(
        [sys.executable, str(runner_dir / "run_model.py"), str(long_wav)],
        cwd=runner_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    runner_result = json.loads(completed.stdout)
    assert runner_result["verdict"] == "uncertain"
    assert len(runner_result["windows"]) == 7
    assert runner_result["windows"][-1]["start_seconds"] == pytest.approx(3.0)
    off_grid_wav = runner_dir / "four_and_quarter_seconds.wav"
    wavfile.write(
        off_grid_wav,
        sample_rate,
        np.concatenate([np.tile(signal[-sample_rate:], 4), signal[-sample_rate // 4 :]]),
    )
    off_grid_completed = subprocess.run(
        [sys.executable, str(runner_dir / "run_model.py"), str(off_grid_wav)],
        cwd=runner_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert off_grid_completed.returncode == 0, (
        off_grid_completed.stdout + off_grid_completed.stderr
    )
    off_grid_result = json.loads(off_grid_completed.stdout)
    assert len(off_grid_result["windows"]) == 7
    assert off_grid_result["windows"][-1]["start_seconds"] == pytest.approx(3.0)
