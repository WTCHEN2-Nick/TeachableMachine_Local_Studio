from __future__ import annotations

from pathlib import Path

import pytest

from tm_local.training_common import TrainingError, save_native_keras_model


ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def save(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.calls.append((args, kwargs))
        Path(str(args[0])).write_bytes(b"fake keras archive")


def test_native_keras_save_uses_no_legacy_format_keywords(tmp_path: Path) -> None:
    model = FakeModel()
    destination = tmp_path / "classifier.keras"
    result = save_native_keras_model(model, destination)

    assert result == destination
    assert destination.is_file()
    assert model.calls == [((str(destination),), {})]


def test_native_keras_save_rejects_wrong_extension(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match=r"must end in \.keras"):
        save_native_keras_model(FakeModel(), tmp_path / "classifier.h5")


def test_image_and_audio_pipelines_do_not_pass_include_optimizer() -> None:
    for relative in ("tm_local/image_pipeline.py", "tm_local/audio_pipeline.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "include_optimizer=False" not in source
        assert "save_native_keras_model(model, keras_path)" in source


def test_installer_contains_native_training_save_probe() -> None:
    source = (ROOT / "scripts/verify_install.py").read_text(encoding="utf-8")
    assert 'model.save(str(keras_path))' in source
    assert 'tf.keras.models.load_model' in source
    assert 'TFLiteConverter.from_saved_model' in source
