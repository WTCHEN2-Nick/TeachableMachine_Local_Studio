from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_distribution_files_exist() -> None:
    required = [
        "01_INSTALL.bat",
        "02_START.bat",
        "README_zh-TW.md",
        "requirements.txt",
        "scripts/install_local.py",
        "scripts/verify_install.py",
        "scripts/prefetch_assets.py",
        "tm_local/yamnet_model.py",
        "tm_local/yamnet_anomaly_pipeline.py",
        "tm_local/known_sound_pipeline.py",
        "tm_local/mcu/deploy_service.py",
        "mcu_toolkit/boards.json",
        "docs/ABNORMAL_SOUND_YAMNET_zh-TW.md",
        "docs/KNOWN_SOUND_zh-TW.md",
        "docs/MCU_DEPLOY_zh-TW.md",
        "reference/README_zh-TW.md",
        "web/index.html",
        "web/app.js",
        "web/style.css",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    assert not missing


def test_version_strings_agree() -> None:
    """The five places that state the tool's version number must state the same one.

    Each comparison below is exact equality, not substring containment: `"2.1.0" in
    "2.1.0-mcu-3"` is true, so a naive `in` check would never catch a stray `-mcu-N` suffix
    or an `-rc1` prerelease tag riding along on one of them. Extract the value with a regex
    at each location and compare with `==` against the canonical TOOL_VERSION instead.
    """
    import tomllib

    config_source = (ROOT / "tm_local/config.py").read_text(encoding="utf-8")
    config_match = re.search(r'TOOL_VERSION = "([^"]+)"', config_source)
    assert config_match, "tm_local/config.py must declare TOOL_VERSION"
    version = config_match.group(1)
    assert version == "2.1.0"

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == version

    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == version

    installer = (ROOT / "01_INSTALL.bat").read_text(encoding="utf-8")
    installer_match = re.search(r"Teachable Machine Local Studio v(\S+)", installer)
    assert installer_match, "01_INSTALL.bat banner must state the version"
    assert installer_match.group(1) == version

    index_html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    script_match = re.search(r'<script src="/app\.js\?v=([^"]+)"></script>', index_html)
    assert script_match, "web/index.html must load /app.js with a version query string"
    assert script_match.group(1) == version


def test_distribution_exposes_only_two_student_bat_files() -> None:
    relative_names = sorted(
        str(path.relative_to(ROOT)).replace("/", "\\")
        for path in ROOT.rglob("*.bat")
        if ".venv" not in path.parts
    )
    assert relative_names == ["01_INSTALL.bat", "02_START.bat"]


def test_distribution_uses_project_venv() -> None:
    bats = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in ROOT.glob("*.bat"))
    assert ".venv\\Scripts\\python.exe" in bats
    assert "conda activate" not in bats.lower()
    assert "Python 3.13" in (ROOT / "01_INSTALL.bat").read_text(encoding="utf-8")


def test_installer_is_native_windows_cpu_only() -> None:
    installer = (ROOT / "scripts/install_local.py").read_text(encoding="utf-8")
    dispatch = (ROOT / "tm_local/training_dispatch.py").read_text(encoding="utf-8")
    runtime = (ROOT / "tm_local/runtime_config.py").read_text(encoding="utf-8")
    combined = "\n".join((installer, dispatch, runtime))
    assert "windows_tensorflow_cpu" in combined
    assert "Windows CPU" in combined
    assert "wsl.exe" not in combined
    assert "wsl --install" not in combined
    assert "tensorflow[and-cuda]" not in combined
    assert "nvidia-smi" not in combined
    assert not (ROOT / "requirements-wsl-gpu.txt").exists()
    assert not (ROOT / "scripts/training_worker.py").exists()
    assert not (ROOT / "docs/GPU_SETUP_zh-TW.md").exists()


def test_runtime_config_passes_mcu_block_through() -> None:
    source = (ROOT / "tm_local/runtime_config.py").read_text(encoding="utf-8")
    assert '"mcu"' in source
    installer = (ROOT / "scripts/install_local.py").read_text(encoding="utf-8")
    assert "probe_mcu_toolchain" in installer


def test_web_contains_image_audio_training_preview_and_int8_export() -> None:
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    for term in ("Image Project", "Audio Project", "Train Model", "Preview", "Quantized INT8"):
        assert term in html
    assert "Abnormal Sound Project" in html
    assert "frozen YAMNet 1024-D embedding" in html
    for term in ("getUserMedia", "predictCameraFrame", "predictRecentAudio", "downloadModel"):
        assert term in js
    assert "下載 Log" in html
    assert "/api/diagnostics/download" in js
    assert "/predict/${endpointKind}" in js


def test_abnormal_sound_has_explicit_dispatch_route_and_pinned_asset() -> None:
    app_source = (ROOT / "tm_local/app.py").read_text(encoding="utf-8")
    dispatch = (ROOT / "tm_local/training_dispatch.py").read_text(encoding="utf-8")
    asset = (ROOT / "tm_local/yamnet_model.py").read_text(encoding="utf-8")
    assert '@app.post("/api/projects/{project_id}/predict/abnormal-sound")' in app_source
    assert 'normalized_kind == "abnormal_sound"' in dispatch
    assert "train_yamnet_abnormal_sound_project" in dispatch
    assert "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744" in asset


def test_known_sound_has_explicit_dispatch_route_and_multi_label_contract() -> None:
    """The fourth kind must be routed explicitly, never via an implicit else branch."""
    app_source = (ROOT / "tm_local/app.py").read_text(encoding="utf-8")
    dispatch = (ROOT / "tm_local/training_dispatch.py").read_text(encoding="utf-8")
    pipeline = (ROOT / "tm_local/known_sound_pipeline.py").read_text(encoding="utf-8")
    config_source = (ROOT / "tm_local/config.py").read_text(encoding="utf-8")
    store_source = (ROOT / "tm_local/project_store.py").read_text(encoding="utf-8")

    assert '@app.post("/api/projects/{project_id}/predict/known-sound")' in app_source
    assert '@app.post("/api/projects/{project_id}/audio-sanity")' in app_source
    assert 'normalized_kind == "known_sound"' in dispatch
    assert "train_known_sound_project" in dispatch

    # Registered in the kind registry rather than bolted on with string comparisons.
    assert '"known_sound"' in config_source
    assert "KNOWN_SOUND_DEFAULTS" in config_source
    assert "MULTI_LABEL_KINDS" in config_source
    # create_project must branch explicitly; an implicit else silently made every unknown
    # kind an abnormal_sound project.
    assert 'elif kind == "known_sound":' in store_source
    assert 'elif kind == "abnormal_sound":' in store_source

    # The class map is a DOWNLOADED asset like yamnet.h5, so it is pinned by digest in
    # both the installer and the loader rather than shipped in the repo.
    prefetch = (ROOT / "scripts/prefetch_assets.py").read_text(encoding="utf-8")
    asset = (ROOT / "tm_local/yamnet_model.py").read_text(encoding="utf-8")
    class_map_sha = "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"
    assert class_map_sha in prefetch
    assert class_map_sha in asset
    assert "yamnet_class_map.csv" in prefetch

    # Multi-label, not softmax: independent sigmoids that must not be called probabilities.
    assert 'activation="sigmoid"' in pipeline
    assert 'loss="binary_crossentropy"' in pipeline
    assert "softmax" not in pipeline
    # Train must not convert to TFLite, and conversion must not use from_keras_model.
    assert "convert_all(" not in pipeline
    assert "from_keras_model(" not in pipeline
    # The encoder stays frozen; fine-tuning 3.2M parameters on a few clips is not the
    # design. This pins the MECHANISM -- every layer but the head is frozen. The behaviour
    # itself is asserted in test_known_sound.py by counting trainable parameters.
    assert 'if layer.name != "class_scores":' in pipeline
    assert "layer.trainable = False" in pipeline


def test_known_sound_split_is_session_disjoint_not_clip_level() -> None:
    """The leakage fix is the whole point of the kind; lock it against a 'tidy' rewrite."""
    pipeline = (ROOT / "tm_local/known_sound_pipeline.py").read_text(encoding="utf-8")
    assert "def split_sessions(" in pipeline
    assert "recording_session_id" in pipeline
    # stratified_path_split is the clip-level splitter that causes the leakage.
    assert "stratified_path_split" not in pipeline


def test_web_offers_known_sound_project_with_independent_scores() -> None:
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    assert 'data-create-kind="known_sound"' in html
    assert "Known Sound Project" in html
    assert "isKnownSoundProject" in js
    assert "applyKnownSoundPeakHold" in js
    assert "runAudioSanityCheck" in js
    assert "/audio-sanity" in js
    # The UI must state that the scores are independent and do not total 100%.
    assert "不會加總成 100%" in js


def _js_code_only(source: str) -> str:
    """`source` with its whole-line `//` comments removed.

    Every check in this file is a substring match, and web/app.js carries long Chinese
    comments that name the very settings keys those checks are meant to pin:
    `fine_tune_blocks`, `waveform_augment_level`, `class_thresholds`, `augmentation_level`
    and `split_warning` each appear in prose as well as in code, so deleting the control
    would have left the assertion green. Strip the prose and the match has to be code.

    Whole-line comments only: web/app.js has no `/* */` blocks, and a block-comment regex
    would swallow everything from the `'image/*'` accept string onwards.
    """

    return re.sub(r"(?m)^[ \t]*//.*$", "", source)


def test_training_panel_exposes_tunables() -> None:
    """Every student-tunable training setting must have a control in the Advanced panel.

    The server already validates these keys; a setting nobody can reach from the browser is
    the same as a setting that does not exist, so the panel is locked by name here.

    The names are matched against comment-stripped code, and the settings keys against the
    body of readTrainingOptions() alone. A whole-file `term in js` loop passed on the
    comments that discuss these keys, which is no pin at all.
    """
    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    code = _js_code_only(js)
    reader = _js_code_only(js.split("function readTrainingOptions")[1].split("\nfunction ")[0])
    for control_id in (
        "optDeploymentTarget", "optAugmentationLevel", "optFineTuneBlocks", "optSpecAugment",
        "optSessionDisjoint", "optBackgroundWeight", "optBackgroundClass", "optDropout",
        "opt-class-threshold", "optHeadDropout", "optWaveformAugment", "optEncoderDepth",
        "optDetectionThreshold", "optImageSize", "optSampleRate", "optMelBins",
    ):
        # Read back when the student presses Train...
        assert control_id in reader, control_id
        # ...and rendered somewhere too. An id that only the reader mentions is a $() that
        # returns null, i.e. a TypeError on the Train click.
        assert code.count(control_id) >= 2, control_id
    # The settings keys they round-trip through readTrainingOptions(). This is the half the
    # server sees: a control whose value never reaches a key changes nothing.
    for key in (
        "deployment_target", "augmentation_level", "fine_tune_blocks", "spec_augment",
        "session_disjoint_validation", "background_weight", "background_class_id",
        "class_thresholds", "head_dropout", "waveform_augment_level",
    ):
        assert key in reader, key
    # Per-class thresholds, the chosen background class and the split warning must be
    # presented, not swallowed. These come back from training.report rather than from a
    # control, so they belong to the whole (comment-stripped) file, not to the reader.
    for field in (
        "class_thresholds_by_label", "background_class_index", "split_warning",
        "fine_tune_epochs_completed",
    ):
        assert field in code, field
    # Changing the deployment target re-renders the panel and must keep the filled values.
    # Slice the handler itself rather than "anywhere after wireGlobalEvents": the previous
    # assertion passed for any later occurrence of the call anywhere in the file, so moving
    # the re-render out of the handler would not have failed anything.
    handler = js.split("elements.trainingOptions.addEventListener('change'")[1].split("});")[0]
    for term in (
        "optDeploymentTarget", "optBackbone", "readTrainingOptions()", "renderTrainingOptions(",
    ):
        assert term in handler, term
    # ...into a throwaway copy. state.project is the server's answer; writing the panel's
    # unsaved values back into it makes the in-memory project diverge until Train.
    assert "state.project.settings = " not in js


def test_training_panel_keeps_unsaved_edits_across_a_project_re_render() -> None:
    """renderProject() must not throw away what the student has typed but not yet trained.

    Typed values live only in the DOM until startTraining() reads them, and renderProject()
    rebuilds #trainingOptions wholesale from the server's settings. It runs after every
    ordinary sample-collection action -- add class, rename class, upload images, record
    audio, delete a sample -- so "set the options, then collect a few more samples" silently
    reverted every field, and the reverted values are the ones the next Train really uses.
    known_sound is the worst case: `data-touched` is DOM-only and is recomputed from
    *persisted* overrides, so the edit and the evidence of it died together and
    readTrainingOptions() then truthfully reported that nothing had been changed.

    Three conditions come with the fix, and each one is a worse bug than the original if it
    is dropped, so all three are pinned here: no draft may cross a project switch, the panel
    must not be read before it exists, and Train's persisted settings must win afterwards.
    """
    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    for name in ("trainingOptionsDraft", "discardTrainingOptionsDraft"):
        assert f"function {name}()" in js, name

    # The bare call is the bug itself. Slice renderProject() so that moving the re-render
    # somewhere else cannot keep this green.
    render_project = js.split("\nfunction renderProject()")[1].split("\nfunction ")[0]
    assert "renderTrainingOptions();" not in render_project
    assert "renderTrainingOptions(trainingOptionsDraft())" in render_project

    draft = js.split("\nfunction trainingOptionsDraft()")[1].split("\n}")[0]
    # Same shape as the deployment-target handler: a throwaway merge, never a write back.
    assert "...project.settings, ...readTrainingOptions()" in draft
    # Condition 1 + 2: a draft belongs to one project, and only after that project's panel
    # has actually been rendered -- readTrainingOptions() would $() into an empty panel.
    assert "state.trainingOptionsProjectId !== project.id" in draft
    assert "return null" in draft
    rendered = js.split("function renderTrainingOptions(")[1].split("\nfunction ")[0]
    assert "state.trainingOptionsProjectId = state.project?.id" in rendered

    # Condition 3: once Train has PATCHed the settings, the server's copy is authoritative.
    start_training = js.split("async function startTraining()")[1].split("\nfunction ")[0]
    assert "discardTrainingOptionsDraft()" in start_training
    # ...and a project switch/reload starts from the server's settings alone.
    load_project = js.split("async function loadProject(")[1].split("\nfunction ")[0]
    assert "discardTrainingOptionsDraft()" in load_project



def _js_frozen_body(js: str, name: str) -> str:
    """Source text inside `const <name> = Object.freeze([...])`."""

    match = re.search(rf"const {name} = Object\.freeze\(\[(.*?)\]\);", js, re.DOTALL)
    assert match, f"web/app.js must declare {name}"
    return match.group(1)


def test_training_panel_constants_match_config() -> None:
    """app.js's copies of the config.py option lists must not drift.

    Every one of these is hand-copied into web/app.js so the panel can render before any
    API call lands. CLAUDE.md decision 11 ("MCU 端的常數一律生成，禁止手動同步") is
    exactly about this: drift here means the panel offers a value the server answers 400
    for, or hides a legal one, and nobody finds out until a student presses Train.
    """
    from tm_local.config import (
        AUGMENTATION_LEVELS,
        DEPLOYMENT_TARGETS,
        MCU_AUDIO_FRONTEND_LOCK,
        MCU_IMAGE_SIZES,
        MCU_MEL_BINS,
    )

    js = (ROOT / "web/app.js").read_text(encoding="utf-8")

    # [value, label] pairs -- only the values are the contract, the labels are Chinese UI text.
    targets = re.findall(r"\['([^']+)',", _js_frozen_body(js, "DEPLOYMENT_TARGET_OPTIONS"))
    assert targets == list(DEPLOYMENT_TARGETS)
    levels = re.findall(r"\['([^']+)',", _js_frozen_body(js, "AUGMENTATION_LEVEL_OPTIONS"))
    assert levels == list(AUGMENTATION_LEVELS)
    # The shared level list is reused by two settings whose defaults differ
    # (augmentation_level=medium, waveform_augment_level=off), so "（預設）" must be applied
    # per caller by augmentationSelect(), never baked into the list.
    assert "（預設）" not in _js_frozen_body(js, "AUGMENTATION_LEVEL_OPTIONS")

    # Plain number lists; the panel orders them largest-first for the dropdown.
    mcu_sizes = [int(value) for value in re.findall(r"\d+", _js_frozen_body(js, "MCU_IMAGE_SIZES"))]
    assert sorted(mcu_sizes) == sorted(MCU_IMAGE_SIZES)
    mel_bins = [int(value) for value in re.findall(r"\d+", _js_frozen_body(js, "MEL_BIN_CHOICES"))]
    assert sorted(mel_bins) == sorted(MCU_MEL_BINS)

    rate = re.search(r"const MCU_SAMPLE_RATE = (\d+);", js)
    assert rate, "web/app.js must declare MCU_SAMPLE_RATE"
    assert int(rate.group(1)) == MCU_AUDIO_FRONTEND_LOCK["sample_rate"]


def test_pc_image_sizes_are_all_accepted_by_the_server() -> None:
    """PC_IMAGE_SIZES is a UI-side subset choice, not a mirror of one config constant.

    validate_image_settings accepts any multiple of 32 in [96, 320]; the panel offers a
    hand-picked eight of those. So the invariant to pin is not equality with a constant but
    "the server accepts everything the dropdown offers", plus "switching 目標裝置 from a
    board back to 電腦 never silently changes the student's size" -- which needs every
    board-legal size to also be in the PC list.
    """
    from tm_local.config import MCU_IMAGE_SIZES, validate_image_settings

    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    sizes = [int(value) for value in re.findall(r"\d+", _js_frozen_body(js, "PC_IMAGE_SIZES"))]
    assert sizes, "web/app.js must declare PC_IMAGE_SIZES"
    for size in sizes:
        normalized = validate_image_settings({"image_size": size, "deployment_target": "pc"})
        assert normalized["image_size"] == size
    assert set(MCU_IMAGE_SIZES) <= set(sizes)


def test_known_sound_class_thresholds_only_emit_rows_the_student_edited() -> None:
    """Regression guard for the silent settings-loss path Task 6's review found.

    The rows are rendered with `value = class_thresholds[id] ?? detection_threshold`, so a
    project with no overrides shows every class at the current default. Emitting every row
    whose value merely *differed* from the default meant that raising 偵測門檻 from 0.5 to
    0.7 sent `{A:0.5, B:0.5, C:0.5}`: the control the student just moved did nothing, and
    the wrong per-class dict flowed on to Preview, the exported runner and
    tm_local/mcu/contract.py -- i.e. into firmware -- with no error anywhere.

    Only a row the student actually edited (or one that was already a stored override) may
    become an override, so the flag that records that is pinned by name. This repo has no
    JS DOM harness, so a string assertion is the only thing standing between a refactor and
    a silent return of the bug.
    """
    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    assert 'data-touched="' in js
    reader = js.split("function readTrainingOptions")[1].split("\nfunction ")[0]
    assert "dataset.touched !== '1'" in reader
    # The float-equality filter this replaced must not come back: an untouched row now emits
    # nothing at all, so no two thresholds are ever compared with === again.
    assert "value === threshold" not in reader
    # ...and something has to set the flag, or every row stays untouched for ever.
    handler = js.split("function wireGlobalEvents")[1]
    assert "opt-class-threshold" in handler
    assert "dataset.touched = '1'" in handler


def test_known_sound_board_depth_cap_matches_config() -> None:
    """app.js's fallback copy of the flash-budget cap must not drift from config.py.

    The panel prefers the live cap from /api/mcu/status, but a student who opens Advanced
    before that request lands sees the frozen literal, and offering a depth the server
    rejects would only surface as a 400 at Train time.
    """
    from tm_local.config import KNOWN_SOUND_BOARD_MAX_DEPTH

    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    match = re.search(r"const KNOWN_SOUND_BOARD_MAX_DEPTH = Object\.freeze\(\{([^}]*)\}\)", js)
    assert match, "web/app.js must declare KNOWN_SOUND_BOARD_MAX_DEPTH"
    parsed = {name: int(value) for name, value in re.findall(r"'([^']+)':\s*(\d+)", match.group(1))}
    assert parsed == dict(KNOWN_SOUND_BOARD_MAX_DEPTH)


def test_web_has_mcu_deploy_tab() -> None:
    """The Export modal must carry a second tab that builds M55M1 firmware."""
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    for term in (
        "部署到開發板",
        "data-export-tab",
        'data-export-tab="mcu"',
        "exportTabTflite",
        "exportTabMcu",
        "mcuBoardSelect",
        "mcuToolchainNotice",
        "mcuKindNotice",
        "mcuErrorNotice",
        "deployMcuButton",
        "deployLog",
        "mcuResultSummary",
        "mcuFlashInstructions",
        "mcuDownloadButton",
        "mcuFlashButton",
        "mcuFlashStatus",
        "app.js?v=2.1.0",
    ):
        assert term in html, term
    for term in (
        "startMcuDeploy",
        "pollDeployJob",
        "switchExportTab",
        "renderMcuTab",
        "refreshMcuStatus",
        "/api/mcu/status",
        "/deploy-mcu",
        "/deploy-mcu/download",
        "/deploy-mcu/flash",
        "job_type === 'deploy'",
        "renderDeployLog",
        "此專案類型無法部署到開發板",
        "模型已變更，請重新建置",
        # Flashing overwrites the board: name the file, its size and the board first.
        "bytes）燒錄到",
        "確定要繼續？",
    ):
        assert term in js, term
    css = (ROOT / "web/style.css").read_text(encoding="utf-8")
    for term in (".deploy-log", ".mcu-last-build", ".mcu-result", ".mcu-actions"):
        assert term in css, term
    # The TensorFlow Lite tab keeps working exactly as before.
    assert "Quantized INT8" in html
    assert "downloadModel" in js


def test_mcu_flash_fallback_text_is_kind_aware() -> None:
    """The UI's own copy of the flashing advice must not send an audio student to the camera.

    `deploy_report.json` carries the server's `flash_instructions`, but reports built before
    that field fall back to these strings -- and the image app's "open the camera app / watch
    the LCD" is wrong for both audio kinds, whose firmware only ever prints to a COM port.
    """
    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    assert "MCU_FLASH_STEPS" in js and "MCU_FLASH_RESULT" in js
    assert "report?.kind" in js
    for term in ("known_sound:", "audio:", "KWS DETECTED", "INFO threshold[", "115200 8N1"):
        assert term in js, term
    # The steps table must no longer carry the image-only result sentence.
    steps = js.split("const MCU_FLASH_STEPS")[1].split("});")[0]
    assert "相機" not in steps and "LCD" not in steps


def test_prediction_percentage_bars_have_visible_block_fill() -> None:
    css = (ROOT / "web/style.css").read_text(encoding="utf-8")
    js = (ROOT / "web/app.js").read_text(encoding="utf-8")
    assert ".prediction-fill { display:block;" in css
    assert "--prediction-width" in css
    assert "prediction-meta" in js
    assert "prediction-score" in js


def test_javascript_syntax_when_node_is_available() -> None:
    if shutil.which("node") is None:
        return
    result = subprocess.run(
        ["node", "--check", str(ROOT / "web/app.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_startup_does_not_use_fragile_app_string_import() -> None:
    source = (ROOT / "scripts/start_local.py").read_text(encoding="utf-8")
    assert 'uvicorn.run("app:app"' not in source
    assert "sys.path.insert(0, str(PROJECT_ROOT))" in source
    assert "application = _create_application()" in source
    assert "uvicorn.run(\n            application," in source
    assert "setup_session_logging" in source
    assert "/?session={time_ns()}" in source


def test_startup_import_check_works_from_unrelated_working_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/start_local.py"), "--check"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] Local Studio application import/create check" in result.stdout


def test_training_defers_tflite_until_export_and_converter_uses_saved_model() -> None:
    image_source = (ROOT / "tm_local/image_pipeline.py").read_text(encoding="utf-8")
    audio_source = (ROOT / "tm_local/audio_pipeline.py").read_text(encoding="utf-8")
    export_source = (ROOT / "tm_local/tflite_export.py").read_text(encoding="utf-8")
    app_source = (ROOT / "tm_local/app.py").read_text(encoding="utf-8")
    assert "convert_all(" not in image_source
    assert "convert_all(" not in audio_source
    assert "from_keras_model(" not in export_source
    assert "TFLiteConverter.from_saved_model" in export_source
    assert '@app.post("/api/projects/{project_id}/export-model")' in app_source
    assert "conversion_worker" in (ROOT / "tm_local/export_service.py").read_text(encoding="utf-8")


def test_single_diagnostic_download_includes_runtime_and_both_latest_logs() -> None:
    source = (ROOT / "tm_local/diagnostics.py").read_text(encoding="utf-8")
    assert "runtime_config" in source
    assert "LATEST.log" in source
    assert "LATEST_INSTALL.log" in source
    assert "no images/audio included" in source

def test_windows_batch_entry_points_are_ascii_crlf() -> None:
    for name in ("01_INSTALL.bat", "02_START.bat"):
        payload = (ROOT / name).read_bytes()
        assert not payload.startswith(b"\xef\xbb\xbf")
        payload.decode("ascii")
        assert b"\r\n" in payload
        assert b"\n" not in payload.replace(b"\r\n", b"")


def test_mcu_toolkit_and_reference_have_no_bat_or_keil_files() -> None:
    for folder in ("mcu_toolkit", "reference"):
        root = ROOT / folder
        if not root.is_dir():
            continue
        offenders = [p for p in root.rglob("*") if p.suffix.lower() in {".bat", ".uvprojx", ".uvoptx"}]
        assert offenders == []


def test_studio_never_touches_keil_or_make() -> None:
    combined = "\n".join(
        (ROOT / "tm_local" / name).read_text(encoding="utf-8")
        for name in ("app.py", "export_service.py", "tflite_export.py")
    ) + "\n".join(
        p.read_text(encoding="utf-8")
        for p in [
            *(ROOT / "tm_local" / "mcu").glob("*.py"),
            *(ROOT / "tm_local" / "mcu" / "apps").glob("*.py"),
        ]
    )
    for forbidden in ("UV4.exe", ".uvprojx", "make.exe", "project_generator", "numl_tool.py", "os.chdir("):
        assert forbidden not in combined, forbidden
    assert '@app.post("/api/projects/{project_id}/deploy-mcu")' in (ROOT / "tm_local/app.py").read_text(encoding="utf-8")


def test_mcu_registry_in_config() -> None:
    config_source = (ROOT / "tm_local/config.py").read_text(encoding="utf-8")
    for term in ("MCU_APPLICATION_BY_KIND", "MCU_DEPLOYABLE_KINDS", "DEPLOYMENT_TARGETS", "def mcu_application("):
        assert term in config_source


def test_audio_mcu_runtimes_are_vendored_and_studio_sourced() -> None:
    for rel in ("mcu_toolkit/apps/known_sound/main.cpp.in",
                "mcu_toolkit/apps/audio_kws/main.cpp.in",
                "tm_local/mcu/apps/kws.py", "tm_local/mcu/apps/known_sound.py",
                "tm_local/mcu/kws_codegen.py"):
        assert (ROOT / rel).is_file(), rel
    codegen = (ROOT / "tm_local/mcu/kws_codegen.py").read_text(encoding="utf-8")
    assert "mel_filterbank" in codegen and "_mel_filters" not in codegen
    template = (ROOT / "mcu_toolkit/apps/audio_kws/main.cpp.in").read_text(encoding="utf-8")
    assert "arm_cos_f32" not in template and "s_hannTable" in template
