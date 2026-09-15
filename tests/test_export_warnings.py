from __future__ import annotations

import pytest

from tm_local import export_service
from tm_local.project_store import ProjectStore


def _report(agreement: float | str | None) -> dict:
    entry = {
        "path": "x",
        "comparison": {"top1_agreement": agreement} if agreement is not None else {},
    }
    return {"models": {"int8": entry}}


def test_low_agreement_warns_for_image_and_audio() -> None:
    for kind in ("image", "audio"):
        warnings = export_service.int8_agreement_warnings(kind, _report(0.912))
        assert len(warnings) == 1 and "91.2%" in warnings[0] and "95%" in warnings[0]
    assert export_service.int8_agreement_warnings("image", _report(0.97)) == []
    assert export_service.int8_agreement_warnings("image", _report(None)) == []
    assert export_service.int8_agreement_warnings("known_sound", _report(0.5)) == []


def test_agreement_at_minimum_boundary_does_not_warn() -> None:
    # >= minimum is acceptance, not a warning: exactly 0.95 must not warn.
    assert export_service.int8_agreement_warnings("image", _report(0.95)) == []


def test_malformed_agreement_is_not_a_warning() -> None:
    # A malformed comparison is "no evidence", not a warning: must not raise ValueError.
    assert export_service.int8_agreement_warnings("image", _report("bad")) == []
    assert export_service.int8_agreement_warnings("audio", _report("bad")) == []
    assert export_service.int8_agreement_warnings("image", _report([0.9])) == []


def test_minimum_override() -> None:
    # A stricter custom minimum warns on an agreement the default would accept.
    warnings = export_service.int8_agreement_warnings("image", _report(0.97), minimum=0.99)
    assert len(warnings) == 1 and "97.0%" in warnings[0] and "99%" in warnings[0]
    # And a looser custom minimum accepts an agreement the default would warn on.
    assert export_service.int8_agreement_warnings("audio", _report(0.912), minimum=0.90) == []


# ----------------------------------------------------------------------------------
# Task 7 Part A group 4: integration coverage for ensure_export_artifacts().
#
# The tests above prove int8_agreement_warnings() itself is correct in isolation. They
# do not prove that the real Export Model job -- ensure_export_artifacts() -- actually
# calls it with the conversion report a real conversion just produced, or that the
# result is persisted where a student re-opening the project would see it
# (int8_agreement_warnings()'s own docstring names the path:
# training.report.conversion.warnings, plus the same list mirrored under
# conversion.models.int8.warnings). build_model_download() later reads that
# already-persisted list rather than recomputing it, so if ensure_export_artifacts()
# ever stopped attaching it, a student would silently never see the warning at all.
#
# The subprocess conversion itself is replaced with a stand-in that returns a
# below-threshold top1_agreement: forcing a real 90%-vs-95% disagreement through actual
# training would be slow and non-deterministic, and the arithmetic that decides "is this
# a warning" is exactly what the tests above already cover -- only the plumbing between
# the conversion report and the persisted project is new here.
# ----------------------------------------------------------------------------------


def _fake_conversion_worker(agreement: float):
    def run(*, keras_path, representative, models_dir, prefix, selected, progress):
        generated = {name: f"{prefix}_{name}.tflite" for name in selected}
        conversion_report = {
            "models": {
                name: {"path": generated[name], "comparison": {"top1_agreement": agreement}}
                for name in selected
            }
        }
        return generated, conversion_report

    return run


def _trained_project_for_export(store: ProjectStore, kind: str) -> str:
    """A project in the "trained" state with a placeholder Keras file on disk.

    ensure_export_artifacts() only checks that the keras artifact FILE exists before
    handing off to the conversion worker (faked here) -- it never loads or validates its
    contents itself -- so a placeholder is enough to drive the real function.
    """

    project = store.create_project(kind, f"Warn {kind}")
    project_id = project["id"]
    models_dir = store.project_dir(project_id) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    keras_name = f"{kind}_classifier.keras"
    (models_dir / keras_name).write_bytes(b"placeholder")
    store.set_training_result(
        project_id, report={"settings": {}}, artifacts={"keras": keras_name}
    )
    return project_id


def test_ensure_export_artifacts_persists_the_agreement_warning_for_image_and_audio(
    store: ProjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    for kind in ("image", "audio"):
        monkeypatch.setattr(
            export_service, "_run_conversion_worker", _fake_conversion_worker(0.90)
        )
        project_id = _trained_project_for_export(store, kind)

        export_service.ensure_export_artifacts(store, project_id, ["int8"], lambda *_: None)

        project = store.get_raw_project(project_id)
        conversion = project["training"]["report"]["conversion"]
        assert len(conversion["warnings"]) == 1
        assert "90.0%" in conversion["warnings"][0] and "95%" in conversion["warnings"][0]
        assert conversion["models"]["int8"]["warnings"] == conversion["warnings"]

    # known_sound is multi-label and out of scope for this warning by ruling (see
    # test_low_agreement_warns_for_image_and_audio above, which already covers this
    # input directly). Cheap direct check per the brief, reusing the same
    # forced-low-agreement shape used for image/audio above so the two are directly
    # comparable, rather than a full integration drive-through: known_sound's real
    # ensure_export_artifacts path also runs _validate_known_sound_parity() before the
    # warning is computed, which expects a real detection-agreement comparison shape
    # this stand-in does not attempt to fabricate.
    conversion_report = {"models": {"int8": {"comparison": {"top1_agreement": 0.90}}}}
    assert export_service.int8_agreement_warnings("known_sound", conversion_report) == []
