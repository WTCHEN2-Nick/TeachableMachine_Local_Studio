"""Tunable-parameters plan (2026-09-12), Task 1: settings keys, validators, lock constants.

Covers ``tm_local/config.py`` (new *_DEFAULTS keys, validate_image_settings,
validate_audio_settings, the extended validate_known_sound_settings, and the board/KWS
lock constants) and ``tm_local/project_store.py`` (class_ids-aware
``_validate_kind_settings`` reached through the public ``ProjectStore`` API).

Note: the store helper calls below use the real ``ProjectStore`` API
(``create_project(kind, name=None)``, ``update_project(project_id, changes)`` with a
``{"settings": {...}}`` payload, ``add_class`` returning the full project dict) rather
than the shape sketched in the task brief.
"""

from __future__ import annotations

import json

import pytest

from tm_local import config
from tm_local.archive import zip_directory
from tm_local.project_store import ProjectError, ProjectStore


def test_defaults_contain_new_keys() -> None:
    assert config.IMAGE_DEFAULTS["augmentation_level"] == "medium"
    assert config.IMAGE_DEFAULTS["fine_tune_blocks"] == 0
    assert config.AUDIO_DEFAULTS["session_disjoint_validation"] is True
    assert config.AUDIO_DEFAULTS["detection_threshold"] == 0.5
    assert config.KNOWN_SOUND_DEFAULTS["class_thresholds"] == {}
    assert config.KNOWN_SOUND_DEFAULTS["head_dropout"] == 0.0
    for kind in ("image", "audio", "known_sound"):
        assert "deployment_target" in config.allowed_setting_keys(kind)
    assert "deployment_target" not in config.allowed_setting_keys("abnormal_sound")
    assert config.KNOWN_SOUND_BOARD_MAX_DEPTH == {
        "NuMaker-M55M1": 12,
        "NuGestureAI-M55M1": 11,
        "NuMaker-VoiceAI-M55M1": 11,
    }
    # The PC default stays 14: the board cap filters the choices per deployment_target,
    # it does not shrink the model a student trains for their own computer.
    assert config.KNOWN_SOUND_DEFAULT_ENCODER_DEPTH == 14
    assert config.KWS_RUNTIME_DEFAULTS["inference_hop_seconds"] == 0.25


def test_known_sound_board_max_depth_matches_boards_json() -> None:
    """config's UI/validation source must agree with mcu_toolkit/boards.json's registry."""

    from tm_local.mcu.boards import load_boards

    boards = load_boards()
    for name, depth in config.KNOWN_SOUND_BOARD_MAX_DEPTH.items():
        assert boards[name].known_sound_max_depth == depth


def test_every_board_depth_cap_is_in_a_sane_range() -> None:
    """Catches a forgotten value, a 0, and the PC default copied into the board table."""
    for name, depth in config.KNOWN_SOUND_BOARD_MAX_DEPTH.items():
        assert 1 <= depth <= 14, f"{name} 的深度上限 {depth} 不合理"


@pytest.mark.parametrize(
    "bad",
    [
        {"image_size": 200},
        {"augmentation_level": "extreme"},
        {"dropout": 0.9},
        {"fine_tune_blocks": 7},
        {"deployment_target": "M467"},
        {"mobilenet_alpha": 0.6},
        {"backbone": "resnet"},
        {"epochs": 0},
    ],
)
def test_validate_image_settings_rejects(bad: dict) -> None:
    with pytest.raises(ValueError):
        config.validate_image_settings({**config.IMAGE_DEFAULTS, **bad})


def test_validate_image_settings_mcu_lock() -> None:
    ok = config.validate_image_settings(
        {**config.IMAGE_DEFAULTS, "image_size": 160, "deployment_target": "NuGestureAI-M55M1"}
    )
    assert ok["image_size"] == 160
    with pytest.raises(ValueError, match="image_size"):
        config.validate_image_settings(
            {**config.IMAGE_DEFAULTS, "image_size": 320, "deployment_target": "NuMaker-M55M1"}
        )
    assert config.validate_image_settings({**config.IMAGE_DEFAULTS, "image_size": 320})["image_size"] == 320


@pytest.mark.parametrize(
    "bad",
    [
        {"sample_rate": 44100, "deployment_target": "NuMaker-M55M1"},
        {"mel_bins": 80},
        {"augmentation_level": "x"},
        {"background_weight": 9},
        {"detection_threshold": 1.0},
        {"spec_augment": "yes"},
        {"fft_size": 128},
    ],
)
def test_validate_audio_settings_rejects(bad: dict) -> None:
    with pytest.raises(ValueError):
        config.validate_audio_settings({**config.AUDIO_DEFAULTS, **bad})


def test_validate_audio_settings_accepts_pc_44k_and_mcu_64mel() -> None:
    assert (
        config.validate_audio_settings(
            {**config.AUDIO_DEFAULTS, "sample_rate": 44100, "fft_size": 2048, "fmax": 22050.0}
        )["sample_rate"]
        == 44100
    )
    assert (
        config.validate_audio_settings(
            {**config.AUDIO_DEFAULTS, "mel_bins": 64, "deployment_target": "NuGestureAI-M55M1"}
        )["mel_bins"]
        == 64
    )


def test_validate_known_sound_settings_new_keys() -> None:
    base = dict(config.KNOWN_SOUND_DEFAULTS)
    ok = config.validate_known_sound_settings(
        {
            **base,
            "class_thresholds": {"abc": 0.7},
            "head_dropout": 0.3,
            "waveform_augment_level": "light",
            "background_weight": 2.0,
        }
    )
    assert ok["class_thresholds"] == {"abc": 0.7}
    for bad in (
        {"class_thresholds": {"abc": 1.5}},
        {"head_dropout": 0.9},
        {"waveform_augment_level": "loud"},
        {"encoder_depth": 12, "deployment_target": "NuGestureAI-M55M1"},
    ):
        with pytest.raises(ValueError):
            config.validate_known_sound_settings({**base, **bad})
    assert (
        config.validate_known_sound_settings(
            {**base, "encoder_depth": 11, "deployment_target": "NuGestureAI-M55M1"}
        )["encoder_depth"]
        == 11
    )


def test_store_rejects_unknown_class_ids_and_prunes_on_delete(store: ProjectStore) -> None:
    project = store.create_project("known_sound", "KS")
    pid = project["id"]
    updated = store.add_class(pid, "dog")
    class_id = updated["classes"][-1]["id"]

    store.update_project(
        pid,
        {"settings": {"class_thresholds": {class_id: 0.8}, "background_class_id": class_id}},
    )
    assert store.get_raw_project(pid)["settings"]["class_thresholds"] == {class_id: 0.8}

    with pytest.raises(ProjectError):
        store.update_project(pid, {"settings": {"class_thresholds": {"nope": 0.8}}})

    store.delete_class(pid, class_id)
    settings = store.get_raw_project(pid)["settings"]
    assert settings["class_thresholds"] == {}
    assert settings["background_class_id"] == ""


def test_image_settings_now_validated_by_store(store: ProjectStore) -> None:
    project = store.create_project("image", "Img")
    with pytest.raises(ProjectError):
        store.update_project(project["id"], {"settings": {"image_size": 999}})


# ----------------------------------------------------------------------------------
# Fix round 1 (P1): import must remap class-scoped settings through class_id_map, not
# copy the archive's pre-remap ids verbatim -- otherwise the very next settings PATCH
# (e.g. every Train click, which revalidates settings against the project's live
# classes) rejects them as belonging to no class.
# ----------------------------------------------------------------------------------


def _write_archive_with_raw_settings(tmp_path, *, kind: str, class_ids: list[str], settings: dict):
    """Hand-build a minimal, sample-free project archive so the class-scoped-settings
    remap can be tested against an archive whose ids are known in advance, including one
    id (in ``settings``) that is NOT one of ``class_ids`` -- the "unknown old id" case
    import_project_archive must drop rather than carry over."""

    source = tmp_path / f"src-{kind}"
    source.mkdir()
    payload = {
        "schema_version": 2,
        "id": "11111111-1111-1111-1111-111111111111",
        "name": "Stale IDs",
        "kind": kind,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "classes": [{"id": cid, "name": f"Class {i}"} for i, cid in enumerate(class_ids, start=1)],
        "samples": {},
        "settings": settings,
        "training": {"state": "untrained", "trained_at": None, "report": None, "artifacts": {}},
    }
    (source / "project.json").write_text(json.dumps(payload), encoding="utf-8")
    archive_path = tmp_path / f"{kind}.zip"
    return zip_directory(source, archive_path)


def test_import_remaps_class_scoped_settings_and_next_patch_succeeds(
    store: ProjectStore, tmp_path
) -> None:
    project = store.create_project("known_sound", "KS")
    pid = project["id"]
    updated = store.add_class(pid, "dog")
    old_class_id = updated["classes"][-1]["id"]
    store.update_project(
        pid,
        {
            "settings": {
                "class_thresholds": {old_class_id: 0.8},
                "background_class_id": old_class_id,
            }
        },
    )

    archive_path = tmp_path / "export.zip"
    store.export_project_archive(pid, archive_path)
    imported = store.import_project_archive(archive_path)
    new_pid = imported["id"]
    new_class_id = imported["classes"][-1]["id"]
    assert new_class_id != old_class_id

    raw = store.get_raw_project(new_pid)
    assert raw["settings"]["class_thresholds"] == {new_class_id: 0.8}
    assert raw["settings"]["background_class_id"] == new_class_id

    # The next settings PATCH must not reject the just-imported, correctly remapped ids.
    result = store.update_project(new_pid, {"settings": {"head_dropout": 0.1}})
    assert result["settings"]["head_dropout"] == 0.1
    assert result["settings"]["class_thresholds"] == {new_class_id: 0.8}


def test_import_drops_class_thresholds_for_an_id_the_archive_does_not_have(
    store: ProjectStore, tmp_path
) -> None:
    archive_path = _write_archive_with_raw_settings(
        tmp_path,
        kind="known_sound",
        class_ids=["c1", "c2"],
        settings={
            **config.KNOWN_SOUND_DEFAULTS,
            "class_thresholds": {"c1": 0.6, "c-not-in-this-archive": 0.9},
            "background_class_id": "c2",
        },
    )
    imported = store.import_project_archive(archive_path)
    new_c1, new_c2 = (item["id"] for item in imported["classes"])
    settings = store.get_raw_project(imported["id"])["settings"]
    assert settings["class_thresholds"] == {new_c1: 0.6}
    assert settings["background_class_id"] == new_c2


# ----------------------------------------------------------------------------------
# Fix round 1 (P2): _validate_kind_settings must persist the NORMALIZED dict the
# validators compute, not the caller's original settings -- otherwise a string boolean
# like "false" survives into storage and reads back truthy through the
# bool(settings.get(key, True)) pattern every training pipeline uses.
# ----------------------------------------------------------------------------------


def test_settings_patch_normalizes_string_booleans_before_storage(store: ProjectStore) -> None:
    image_project = store.create_project("image", "Img2")
    result = store.update_project(image_project["id"], {"settings": {"early_stopping": "false"}})
    assert result["settings"]["early_stopping"] is False
    stored = store.get_raw_project(image_project["id"])["settings"]["early_stopping"]
    assert stored is False
    # This is exactly the failure mode being guarded against: bool("false") is True.
    assert bool(stored) is False

    audio_project = store.create_project("audio", "Audio2")
    result = store.update_project(
        audio_project["id"], {"settings": {"session_disjoint_validation": "false"}}
    )
    assert result["settings"]["session_disjoint_validation"] is False
    stored = store.get_raw_project(audio_project["id"])["settings"]["session_disjoint_validation"]
    assert stored is False
    assert bool(stored) is False


def test_settings_patch_normalizes_early_stopping_for_audio_and_known_sound(
    store: ProjectStore,
) -> None:
    """Task 7 Part A group 1: the image/session_disjoint_validation coverage above never
    exercised ``early_stopping`` for audio or known_sound. Both validators coerce it
    through the exact same ``_require_bool`` as every other plan-era boolean (see
    validate_audio_settings/validate_known_sound_settings docstrings), so a browser PATCH
    or an imported legacy project's string boolean must normalize the same way image's
    already does -- never surviving into storage as a string that reads back truthy via
    ``bool(...)`` regardless of its value.
    """

    audio_project = store.create_project("audio", "AudioEarlyStop")
    result = store.update_project(audio_project["id"], {"settings": {"early_stopping": "FALSE"}})
    assert result["settings"]["early_stopping"] is False
    stored = store.get_raw_project(audio_project["id"])["settings"]["early_stopping"]
    assert stored is False
    assert bool(stored) is False

    result = store.update_project(audio_project["id"], {"settings": {"early_stopping": "True"}})
    assert result["settings"]["early_stopping"] is True
    stored = store.get_raw_project(audio_project["id"])["settings"]["early_stopping"]
    assert stored is True

    known_sound_project = store.create_project("known_sound", "KnownSoundEarlyStop")
    result = store.update_project(
        known_sound_project["id"], {"settings": {"early_stopping": "false"}}
    )
    assert result["settings"]["early_stopping"] is False
    stored = store.get_raw_project(known_sound_project["id"])["settings"]["early_stopping"]
    assert stored is False
    assert bool(stored) is False

    result = store.update_project(
        known_sound_project["id"], {"settings": {"early_stopping": "true"}}
    )
    assert result["settings"]["early_stopping"] is True
    stored = store.get_raw_project(known_sound_project["id"])["settings"]["early_stopping"]
    assert stored is True


@pytest.mark.parametrize("bad", [" true", "true ", "yes", "1", "0", "on", "off", ""])
def test_require_bool_rejects_whitespace_and_non_boolean_strings_for_early_stopping(
    bad: str,
) -> None:
    """``_require_bool`` (tm_local/config.py) is case-insensitive on an exact "true"/
    "false" match only -- it does not strip whitespace, and every other string (including
    the common "1"/"0"/"yes"/"on" spellings a hand-written import or a different UI widget
    might send) is a real input error, not a value it silently falls back on."""

    with pytest.raises(ValueError, match="early_stopping"):
        config.validate_audio_settings({**config.AUDIO_DEFAULTS, "early_stopping": bad})
    with pytest.raises(ValueError, match="early_stopping"):
        config.validate_known_sound_settings(
            {**config.KNOWN_SOUND_DEFAULTS, "early_stopping": bad}
        )


def _rewrite_raw_settings(store: ProjectStore, project_id: str, *, drop=(), patch=None) -> None:
    """Directly rewrite project.json's settings, bypassing ProjectStore's own validation,
    to simulate either a project saved before this plan (missing keys) or one with
    legacy un-normalized data already on disk (a string boolean)."""

    path = store.project_dir(project_id) / "project.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in drop:
        payload["settings"].pop(key, None)
    if patch:
        payload["settings"].update(patch)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_add_audio_bytes_revalidates_and_normalizes_stale_settings(
    store: ProjectStore, wav_bytes: bytes
) -> None:
    project = store.create_project("audio", "Audio3")
    pid = project["id"]
    class_id = project["classes"][0]["id"]
    _rewrite_raw_settings(store, pid, patch={"spec_augment": "true"})
    assert store.get_raw_project(pid)["settings"]["spec_augment"] == "true"

    store.add_audio_bytes(pid, class_id, wav_bytes, source_name="clip.wav")

    assert store.get_raw_project(pid)["settings"]["spec_augment"] is True


# ----------------------------------------------------------------------------------
# Fix round 1 (P3): backward compatibility with settings shapes from before this plan
# (commit fe82328), where none of the new keys existed yet, plus the audio store-level
# mel_bins rejection the review flagged as missing.
# ----------------------------------------------------------------------------------

_NEW_KEYS_BY_KIND = {
    "image": ("augmentation_level", "dropout", "fine_tune_blocks", "deployment_target"),
    "audio": (
        "augmentation_level",
        "spec_augment",
        "session_disjoint_validation",
        "background_weight",
        "background_class_id",
        "dropout",
        "detection_threshold",
        "deployment_target",
    ),
    "known_sound": (
        "class_thresholds",
        "background_weight",
        "background_class_id",
        "head_dropout",
        "waveform_augment_level",
        "deployment_target",
    ),
    "abnormal_sound": (),  # this plan added no new keys to abnormal_sound
}

_VALIDATE_BY_KIND = {
    "image": config.validate_image_settings,
    "audio": config.validate_audio_settings,
    "known_sound": config.validate_known_sound_settings,
    "abnormal_sound": config.validate_abnormal_sound_settings,
}

# Global Constraints: every new default equals current (pre-plan) behaviour, except
# audio's session_disjoint_validation=True (spec Sec.8.1's one deliberate change).
_EXPECTED_DEFAULTS_BY_KIND = {
    "image": {
        "augmentation_level": "medium",
        "dropout": 0.2,
        "fine_tune_blocks": 0,
        "deployment_target": "pc",
    },
    "audio": {
        "augmentation_level": "medium",
        "spec_augment": False,
        "session_disjoint_validation": True,
        "background_weight": 1.0,
        "background_class_id": "",
        "dropout": 0.2,
        "detection_threshold": 0.5,
        "deployment_target": "pc",
    },
    "known_sound": {
        "class_thresholds": {},
        "background_weight": 1.0,
        "background_class_id": "",
        "head_dropout": 0.0,
        "waveform_augment_level": "off",
        "deployment_target": "pc",
    },
    "abnormal_sound": {},
}


@pytest.mark.parametrize("kind", ["image", "audio", "known_sound", "abnormal_sound"])
def test_validators_backward_compat_with_pre_plan_sparse_settings(kind: str) -> None:
    """Every validator must still accept a settings mapping with none of this plan's new
    keys (as every project saved before commit fe82328 has) and backfill exactly the
    documented, behaviour-preserving defaults."""

    defaults = config.defaults_for_kind(kind)
    new_keys = _NEW_KEYS_BY_KIND[kind]
    sparse = {k: v for k, v in defaults.items() if k not in new_keys}
    assert not (set(new_keys) & set(sparse))

    result = _VALIDATE_BY_KIND[kind](sparse)
    for key, expected in _EXPECTED_DEFAULTS_BY_KIND[kind].items():
        assert result[key] == expected, f"{kind}.{key}"


@pytest.mark.parametrize("kind", ["image", "audio", "known_sound"])
def test_store_update_project_backfills_new_defaults_for_a_pre_plan_project(
    store: ProjectStore, kind: str
) -> None:
    project = store.create_project(kind, f"Old {kind}")
    pid = project["id"]
    new_keys = _NEW_KEYS_BY_KIND[kind]
    _rewrite_raw_settings(store, pid, drop=new_keys)
    stripped = store.get_raw_project(pid)["settings"]
    assert not (set(new_keys) & set(stripped))

    # Any settings PATCH revalidates the whole settings dict -- e.g. every Train click --
    # and must backfill the missing keys with their documented defaults rather than
    # raising or silently leaving them absent.
    bump = {"epochs": int(stripped["epochs"]) + 1}
    result = store.update_project(pid, {"settings": bump})
    settings = result["settings"]
    assert settings["epochs"] == bump["epochs"]
    for key, expected in _EXPECTED_DEFAULTS_BY_KIND[kind].items():
        assert settings[key] == expected, f"{kind}.{key}"


def test_store_rejects_invalid_mel_bins_for_audio(store: ProjectStore) -> None:
    project = store.create_project("audio", "AudioMel")
    with pytest.raises(ProjectError):
        store.update_project(project["id"], {"settings": {"mel_bins": 80}})


def test_an_image_project_cannot_target_a_board_without_a_camera() -> None:
    """Refuse at settings time: the student has not spent an hour training yet."""

    settings = dict(config.IMAGE_DEFAULTS)
    settings["deployment_target"] = "NuMaker-VoiceAI-M55M1"
    with pytest.raises(ValueError, match="NuMaker-VoiceAI-M55M1"):
        config.validate_image_settings(settings)


def test_sound_projects_may_target_the_camera_less_board() -> None:
    """The same board is the whole point for these two kinds."""

    for defaults, validate in (
        (config.AUDIO_DEFAULTS, config.validate_audio_settings),
        (config.KNOWN_SOUND_DEFAULTS, config.validate_known_sound_settings),
    ):
        settings = dict(defaults)
        settings["deployment_target"] = "NuMaker-VoiceAI-M55M1"
        # KNOWN_SOUND_DEFAULTS's encoder_depth (14, the PC default) exceeds this board's
        # depth cap (config.KNOWN_SOUND_BOARD_MAX_DEPTH == 11); that is a
        # separate, pre-existing guard in validate_known_sound_settings, not the one this
        # test is about, so pin a depth the board can actually hold.
        cap = config.KNOWN_SOUND_BOARD_MAX_DEPTH.get("NuMaker-VoiceAI-M55M1")
        if cap is not None and "encoder_depth" in settings:
            settings["encoder_depth"] = min(settings["encoder_depth"], cap)
        assert validate(settings)["deployment_target"] == "NuMaker-VoiceAI-M55M1"


def test_board_kinds_mirror_the_boards_json_camera_flag() -> None:
    """MCU_BOARD_KINDS is a mirror; a board that grows a camera must update both."""

    from tm_local.mcu import boards

    for name, board in boards.load_boards().items():
        assert ("image" in config.MCU_BOARD_KINDS[name]) is board.has_camera
