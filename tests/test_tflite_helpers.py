from __future__ import annotations

from pathlib import Path

from tm_local.tflite_export import (
    convert_selected,
    inspect_tflite,
    representative_samples_sha256,
    validate_tflite_format_report,
    write_c_array,
)


def test_write_c_array_is_portable(tmp_path: Path) -> None:
    model = tmp_path / "model.tflite"
    model.write_bytes(bytes(range(32)))
    header = write_c_array(model, tmp_path / "model_data.h", "g_test_model")
    text = header.read_text(encoding="utf-8")
    assert "TM_LOCAL_ALIGNAS(16)" in text
    assert "defined(_MSC_VER)" in text
    assert "g_test_model_len = 32u" in text
    assert "0x00, 0x01, 0x02" in text


def test_build_model_download_contains_selected_files(tmp_path: Path) -> None:
    import json
    import zipfile

    from tm_local.tflite_export import build_model_download

    project_dir = tmp_path / "project"
    models = project_dir / "models"
    models.mkdir(parents=True)
    (models / "image_classifier_int8.tflite").write_bytes(b"TFL3" + bytes(range(20)))
    (models / "image_classifier_float32.tflite").write_bytes(b"TFL3float")
    (models / "conversion_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (models / "training_report.json").write_text(json.dumps({"accuracy": 1.0}), encoding="utf-8")
    project = {
        "id": "project-id",
        "name": "Demo",
        "kind": "image",
        "classes": [{"name": "A"}, {"name": "B"}],
        "training": {
            "state": "trained",
            "artifacts": {
                "int8": "image_classifier_int8.tflite",
                "float32": "image_classifier_float32.tflite",
            },
        },
    }
    output = build_model_download(
        project=project,
        project_dir=project_dir,
        selected=("int8", "float32"),
        include_c_header=True,
        output_zip=tmp_path / "model.zip",
    )
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "image_classifier_int8.tflite" in names
        assert "image_classifier_float32.tflite" in names
        assert "model_data.h" in names
        assert "labels.txt" in names
        assert "metadata.json" in names


def test_build_model_download_rejects_missing_selected_artifact(tmp_path: Path) -> None:
    import pytest

    from tm_local.tflite_export import ExportError, build_model_download

    project_dir = tmp_path / "project"
    models = project_dir / "models"
    models.mkdir(parents=True)
    project = {
        "id": "project-id",
        "name": "Demo",
        "kind": "image",
        "classes": [{"name": "A"}, {"name": "B"}],
        "training": {
            "state": "trained",
            "artifacts": {"keras": "image_classifier.keras"},
        },
    }
    (models / "image_classifier.keras").write_bytes(b"keras")
    with pytest.raises(ExportError, match="were not generated"):
        build_model_download(
            project=project,
            project_dir=project_dir,
            selected=["int8"],
            include_c_header=True,
            output_zip=tmp_path / "export.zip",
        )


def test_build_model_download_rejects_missing_abnormal_support(tmp_path: Path) -> None:
    import pytest

    from tm_local.tflite_export import ExportError, build_model_download

    project_dir = tmp_path / "project"
    models = project_dir / "models"
    models.mkdir(parents=True)
    (models / "abnormal_sound_yamnet.keras").write_bytes(b"keras")
    (models / "yamnet_scorer.json").write_text("{}", encoding="utf-8")
    (models / "training_report.json").write_text("{}", encoding="utf-8")
    project = {
        "id": "project-id",
        "name": "Abnormal",
        "kind": "abnormal_sound",
        "classes": [
            {"name": "Normal"},
            {"name": "Anomaly Examples"},
        ],
        "training": {
            "state": "trained",
            "artifacts": {"keras": "abnormal_sound_yamnet.keras"},
        },
    }
    with pytest.raises(ExportError, match="yamnet_frontend.json"):
        build_model_download(
            project=project,
            project_dir=project_dir,
            selected=["keras"],
            include_c_header=False,
            output_zip=tmp_path / "abnormal.zip",
        )


def test_fake_cached_int8_is_reconverted_and_freshly_audited(tmp_path: Path) -> None:
    import numpy as np
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(4,), dtype=tf.float32),
            tf.keras.layers.Dense(3, activation="relu"),
            tf.keras.layers.Dense(2),
        ]
    )
    output_dir = tmp_path / "models"
    output_dir.mkdir()
    cached = output_dir / "probe_int8.tflite"
    cached.write_bytes(b"not a tflite model but larger than sixteen bytes")
    samples = [np.linspace(-1, 1, 4, dtype=np.float32) + index / 20 for index in range(12)]
    artifacts, report = convert_selected(
        model,
        samples,
        output_dir,
        "probe",
        ["int8"],
        reuse_existing=True,
        source_model_sha256="a" * 64,
    )
    assert artifacts["int8"] == cached.name
    audit = inspect_tflite(cached)
    validate_tflite_format_report(audit, "int8")
    assert audit["strict_full_integer"] is True
    assert report["models"]["int8"]["reused_existing_file"] is False
    _, reused_report = convert_selected(
        model,
        samples,
        output_dir,
        "probe",
        ["int8"],
        reuse_existing=True,
        source_model_sha256="a" * 64,
    )
    assert reused_report["models"]["int8"]["reused_existing_file"] is True
    assert reused_report["models"]["int8"]["source_model_sha256"] == "a" * 64
    assert reused_report["models"]["int8"]["representative_sha256"] == (
        representative_samples_sha256(samples)
    )
    changed_samples = [np.asarray(sample).copy() for sample in samples]
    changed_samples[0][0] += 0.25
    _, changed_report = convert_selected(
        model,
        changed_samples,
        output_dir,
        "probe",
        ["int8"],
        reuse_existing=True,
        source_model_sha256="a" * 64,
    )
    assert changed_report["models"]["int8"]["reused_existing_file"] is False
    assert changed_report["models"]["int8"]["representative_sha256"] == (
        representative_samples_sha256(changed_samples)
    )


def test_prediction_survives_an_xnnpack_workspace_failure(tmp_path: Path) -> None:
    """A failed XNNPACK start must degrade to the builtin kernels, not fail the prediction.

    The delegate needs a runtime workspace, and that allocation can fail in a process that has
    been alive a long time -- the Studio server after many Previews, or the whole test suite in
    one interpreter -- with "failed to create XNNPACK runtime". Nothing about the model is wrong
    when that happens, so a prediction that dies there is a bug in us, not in the model.
    """

    import numpy as np
    import tensorflow as tf

    from tm_local import tflite_export

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(4,), dtype=tf.float32),
            tf.keras.layers.Dense(3, activation="relu"),
            tf.keras.layers.Dense(2),
        ]
    )
    saved = tflite_export.export_inference_saved_model(model, tmp_path / "saved")
    model_path = tmp_path / "probe_float32.tflite"
    model_path.write_bytes(tflite_export.convert_float32(saved))
    batch = np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 4)

    healthy = tflite_export.predict_tflite(model_path, batch)

    real_builder = tflite_export._interpreter
    calls: list[bool] = []

    def flaky_builder(path, *, delegates):
        calls.append(delegates)
        if delegates:
            raise RuntimeError(
                "failed to create XNNPACK runtimeNode number 2 (TfLiteXNNPackDelegate) "
                "failed to prepare."
            )
        return real_builder(path, delegates=False)

    tflite_export._interpreter = flaky_builder
    try:
        degraded = tflite_export.predict_tflite(model_path, batch)
    finally:
        tflite_export._interpreter = real_builder

    assert calls == [True, False], "the delegated attempt must come first, then the fallback"
    np.testing.assert_allclose(degraded, healthy, rtol=1e-5, atol=1e-5)


def test_a_non_xnnpack_allocation_error_is_not_swallowed(tmp_path: Path) -> None:
    """Only the XNNPACK workspace failure degrades; a real model problem must still raise."""

    import numpy as np
    import pytest

    from tm_local import tflite_export

    model_path = tmp_path / "broken_float32.tflite"
    model_path.write_bytes(b"not a tflite model but larger than sixteen bytes")
    real_builder = tflite_export._interpreter

    def always_broken(path, *, delegates):
        raise RuntimeError("this model has a genuinely broken graph")

    tflite_export._interpreter = always_broken
    try:
        with pytest.raises(RuntimeError, match="genuinely broken graph"):
            tflite_export.predict_tflite(model_path, np.zeros((1, 4), dtype=np.float32))
    finally:
        tflite_export._interpreter = real_builder
