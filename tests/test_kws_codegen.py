from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tm_local.audio_frontend import AudioFrontendConfig, log_mel_spectrogram, mel_filterbank
from tm_local.mcu import boards, contract, kws_codegen
from tm_local.mcu.errors import DeployError

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mcu"


def _tensor(entry: dict) -> contract.TensorInfo:
    return contract.TensorInfo(
        entry["name"],
        tuple(entry["shape"]),
        entry["dtype"],
        float(entry["scale"]),
        int(entry["zero_point"]),
    )


def _kws_contract(tmp_path: Path, *, board: str = "NuGestureAI-M55M1") -> contract.DeployContract:
    entry = json.loads((FIXTURES / "kws3_int8.report.json").read_text(encoding="utf-8"))
    frontend = json.loads((FIXTURES / "audio_frontend_default.json").read_text(encoding="utf-8"))
    return contract.DeployContract(
        kind="audio",
        application="kws",
        board=boards.board_for(board),
        project_id="p",
        project_name="n",
        labels=["Background Noise", "yes", "no"],
        models_dir=tmp_path,
        int8_path=tmp_path / "m.tflite",
        model_sha256="x",
        input=_tensor(entry["inputs"][0]),
        output=_tensor(entry["outputs"][0]),
        image_size=None,
        audio_frontend=frontend,
        yamnet_frontend=None,
        encoder_depth=None,
        detection_threshold=0.6,
        class_thresholds=[0.6] * 3,
        hop_seconds=0.5,
        clip_seconds=1.0,
        peak_hold_seconds=0.0,
    )


@pytest.mark.parametrize("mel_bins", [40, 64])
def test_tables_match_studio_filterbank(mel_bins: int) -> None:
    config = AudioFrontendConfig(mel_bins=mel_bins)
    tables = kws_codegen.kws_tables(config)
    assert tables.hann.shape == (400,)
    assert np.array_equal(tables.hann, np.hanning(400).astype(np.float32))
    dense = kws_codegen.dense_from_tables(tables, mel_bins, config.fft_size // 2 + 1)
    reference = mel_filterbank(
        config.sample_rate, config.fft_size, mel_bins, config.fmin, config.fmax
    )
    assert np.array_equal(dense, reference)
    assert len(tables.mel_start) == mel_bins and all(c >= 1 for c in tables.mel_count)
    assert tables.mel_offset[-1] + tables.mel_count[-1] == len(tables.mel_weights)


@pytest.mark.parametrize("mel_bins", [40, 64])
def test_emulated_frontend_matches_training_within_one_lsb(mel_bins: int) -> None:
    config = AudioFrontendConfig(mel_bins=mel_bins)
    tables = kws_codegen.kws_tables(config)
    rng = np.random.default_rng(42)
    scale, zp = 0.00392157, -128
    t = np.arange(config.clip_samples) / config.sample_rate
    clips = [
        (rng.uniform(-0.5, 0.5, config.clip_samples) * 32767).astype(np.int16),
        (0.6 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16),
        (0.05 * rng.standard_normal(config.clip_samples) * 32767).astype(np.int16),
    ]
    for pcm in clips:
        emulated = kws_codegen.emulate_frontend(pcm, config, tables, scale, zp)
        reference = log_mel_spectrogram(pcm.astype(np.float32) / 32768.0, config)[..., 0]
        quantized = np.clip(np.rint(reference / scale) + zp, -128, 127).astype(np.int8)
        assert emulated.shape == (config.frame_count, mel_bins)
        diff = np.abs(emulated.astype(np.int32) - quantized.astype(np.int32))
        assert np.max(diff) <= 1
        assert float(np.mean(diff)) < 0.01


def test_emulate_frontend_rejects_wrong_clip_length() -> None:
    config = AudioFrontendConfig()
    tables = kws_codegen.kws_tables(config)
    with pytest.raises(DeployError):
        kws_codegen.emulate_frontend(np.zeros(100, dtype=np.int16), config, tables, 0.0039, -128)


def test_background_index_and_literals() -> None:
    assert kws_codegen.background_index(["yes", "Background Noise", "no"]) == 1
    assert kws_codegen.background_index(["yes", "no"]) == 0
    assert kws_codegen.float_literal(0.5) == "0.5f" and kws_codegen.float_literal(-80.0) == "-80.0f"
    assert kws_codegen.float_literal(1e-10) == "1e-10f"
    assert kws_codegen.c_array([1, 2, 3], str) == "    1, 2, 3"


def test_has_background_distinguishes_fallback_from_real_background() -> None:
    # background_index() returns 0 either way; only has_background() tells the two apart,
    # and the firmware needs that because index 0 is both "never trigger" and the
    # trigger-margin baseline.
    assert kws_codegen.background_index(["yes", "no"]) == 0
    assert kws_codegen.has_background(["yes", "no"]) is False
    assert kws_codegen.has_background(["yes", "Background Noise"]) is True
    assert kws_codegen.has_background([]) is False


@pytest.mark.parametrize(
    "label",
    ["Background Noise", "background", "SILENCE", "Noise", "背景", "背景音", "環境音", "安靜",
     "無聲", "雜音", "教室背景音"],
)
def test_background_hints_cover_english_and_chinese(label: str) -> None:
    assert kws_codegen.has_background([label, "yes"]) is True
    assert kws_codegen.background_index([label, "yes"]) == 0


def test_background_hints_do_not_match_ordinary_keywords() -> None:
    for label in ["yes", "no", "開燈", "關燈", "你好", "上", "下"]:
        assert kws_codegen.has_background([label]) is False


def test_has_background_token(tmp_path: Path) -> None:
    c = _kws_contract(tmp_path)
    assert kws_codegen.kws_tokens(c, 1024)["HAS_BACKGROUND"] == "1"
    c.labels = ["yes", "no"]
    c.class_thresholds = [0.6, 0.6]
    tokens = kws_codegen.kws_tokens(c, 1024)
    assert tokens["HAS_BACKGROUND"] == "0" and tokens["BACKGROUND_INDEX"] == "0"
    c.labels = ["背景音", "yes"]
    tokens = kws_codegen.kws_tokens(c, 1024)
    assert tokens["HAS_BACKGROUND"] == "1" and tokens["BACKGROUND_INDEX"] == "0"


def test_emulate_frontend_rejects_clip_shorter_than_window() -> None:
    config = AudioFrontendConfig(clip_seconds=0.01)
    tables = kws_codegen.kws_tables(config)
    pcm = np.zeros(config.clip_samples, dtype=np.int16)
    with pytest.raises(DeployError, match="window_samples"):
        kws_codegen.emulate_frontend(pcm, config, tables, 0.0039, -128)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("smooth_windows", 0),
        ("smooth_windows", 2.5),
        ("smooth_windows", "4"),
        ("required_hits", -1),
        ("trigger_margin", 1.5),
        ("trigger_margin", float("nan")),
        ("rearm_score", -0.1),
        ("rms_gate_dbfs", 3.0),
        ("inference_hop_seconds", 0.0),
        ("inference_hop_seconds", 2.0),
        ("log_epsilon", 0.0),
        ("log_epsilon", None),
        # Not a tunable: any value other than the training front-end's own floor is refused.
        ("log_epsilon", 1e-9),
        ("log_epsilon", 1e-12),
    ],
)
def test_kws_tokens_rejects_bad_runtime_values(tmp_path: Path, key: str, value: object) -> None:
    c = _kws_contract(tmp_path)
    c.kws_runtime = {**contract.DEFAULT_KWS_RUNTIME, key: value}
    with pytest.raises(DeployError, match=key):
        kws_codegen.kws_tokens(c, 1024)


def test_kws_tokens_rejects_required_hits_above_smooth_windows(tmp_path: Path) -> None:
    c = _kws_contract(tmp_path)
    c.kws_runtime = {**contract.DEFAULT_KWS_RUNTIME, "smooth_windows": 2, "required_hits": 3}
    with pytest.raises(DeployError, match="required_hits"):
        kws_codegen.kws_tokens(c, 1024)


def test_kws_tokens_accepts_tuned_runtime(tmp_path: Path) -> None:
    c = _kws_contract(tmp_path)
    c.kws_runtime = {
        **contract.DEFAULT_KWS_RUNTIME,
        "inference_hop_seconds": 0.5,
        "smooth_windows": 3,
        "required_hits": 3,
        "trigger_margin": 0.0,
        "rearm_score": 1.0,
        "rms_gate_dbfs": 0.0,
    }
    tokens = kws_codegen.kws_tokens(c, 1024)
    assert tokens["INFERENCE_HOP"] == "8000" and tokens["COOLDOWN_HOPS"] == "2"
    assert tokens["TRIGGER_MARGIN"] == "0.0f" and tokens["REARM_SCORE"] == "1.0f"
    assert tokens["RMS_GATE_DBFS"] == "0.0f"


def test_c_array_wraps_rows_at_width() -> None:
    rendered = kws_codegen.c_array(list(range(12)), str, width=5)
    assert rendered == "    0, 1, 2, 3, 4,\n    5, 6, 7, 8, 9,\n    10, 11"


def test_float_literals_round_trip() -> None:
    for value in (0.0, 1.0, -80.0, 0.00392152089625597, 1e-10, 0.2, 0.4, -55.0):
        assert float(kws_codegen.float_literal(value).rstrip("f")) == pytest.approx(value, rel=1e-9)


def test_kws_tokens(tmp_path: Path) -> None:
    c = _kws_contract(tmp_path)
    tokens = kws_codegen.kws_tokens(c, 102400)
    assert tokens["ARENA"] == "102400"
    assert tokens["SAMPLE_RATE"] == "16000" and tokens["FRAME_COUNT"] == "98"
    assert tokens["MEL_BINS"] == "40"
    assert tokens["CLIP_SAMPLES"] == "16000" and tokens["FFT_LENGTH"] == "512"
    assert tokens["FRAME_LENGTH"] == "400" and tokens["FRAME_STEP"] == "160"
    assert tokens["INFERENCE_HOP"] == "4000" and tokens["COOLDOWN_HOPS"] == "4"
    assert tokens["BACKGROUND_INDEX"] == "0" and tokens["LABEL_COUNT"] == "3"
    assert tokens["HAS_BACKGROUND"] == "1"
    assert tokens["TRIGGER_THRESHOLD"] == "0.6f" and tokens["DB_FLOOR"] == "-80.0f"
    assert tokens["LOG_EPSILON"] == "1e-10f"
    assert tokens["SMOOTH_WINDOWS"] == "4" and tokens["REQUIRED_HITS"] == "2"
    assert tokens["TRIGGER_MARGIN"] == "0.2f" and tokens["REARM_SCORE"] == "0.4f"
    assert tokens["RMS_GATE_DBFS"] == "-55.0f"
    assert tokens["INPUT_ZERO_POINT"] == "-128" and tokens["OUTPUT_ZERO_POINT"] == "-128"
    assert tokens["OUTPUT_SCALE"] == "0.00390625f"
    assert tokens["DMIC_CLK_MACRO"] == "SET_DMIC1_CLK_PB2"
    assert tokens["DMIC_DAT_MACRO"] == "SET_DMIC1_DAT_PB3"
    assert tokens["DMIC_CHANNEL_MASK"] == "DMIC_CTL_CHEN2_Msk"
    assert tokens["HANN"].count(",") == 399 and int(tokens["MEL_NNZ"]) > 40
    assert '"Background Noise"' in tokens["LABELS"]
    assert contract.DEFAULT_KWS_RUNTIME["inference_hop_seconds"] == 0.25
    assert contract.DEFAULT_KWS_RUNTIME["log_epsilon"] == 1e-10


def test_kws_tokens_uses_x_board_dmic(tmp_path: Path) -> None:
    tokens = kws_codegen.kws_tokens(_kws_contract(tmp_path, board="NuMaker-M55M1"), 102400)
    assert tokens["DMIC_CLK_MACRO"] == "SET_DMIC0_CLK_PB4"
    assert tokens["DMIC_DAT_MACRO"] == "SET_DMIC0_DAT_PB5"
    assert tokens["DMIC_CHANNEL_MASK"] == "DMIC_CTL_CHEN0_Msk"


def test_kws_tokens_clamps_threshold_and_requires_frontend(tmp_path: Path) -> None:
    c = _kws_contract(tmp_path)
    c.detection_threshold = 0.999
    assert kws_codegen.kws_tokens(c, 1024)["TRIGGER_THRESHOLD"] == "0.99f"
    c.detection_threshold = 0.0
    assert kws_codegen.kws_tokens(c, 1024)["TRIGGER_THRESHOLD"] == "0.05f"
    c.audio_frontend = None
    with pytest.raises(DeployError, match="audio_frontend"):
        kws_codegen.kws_tokens(c, 1024)


def test_render_kws_main(tmp_path: Path) -> None:
    c = _kws_contract(tmp_path)
    names = sorted(kws_codegen.kws_tokens(c, 102400))
    template = tmp_path / "main.cpp.in"
    body = "\n".join(f"{name} = @@{name}@@;" for name in names)
    template.write_text(body, encoding="utf-8")
    rendered = kws_codegen.render_kws_main(c, 102400, template)
    assert "@@" not in rendered
    assert "SAMPLE_RATE = 16000;" in rendered
    assert "DMIC_CHANNEL_MASK = DMIC_CTL_CHEN2_Msk;" in rendered

    bad = tmp_path / "bad.cpp.in"
    bad.write_text("@@NOT_A_TOKEN@@", encoding="utf-8")
    with pytest.raises(DeployError, match="@@NOT_A_TOKEN@@"):
        kws_codegen.render_kws_main(c, 102400, bad)


def test_log_epsilon_is_pinned_to_the_training_frontend(tmp_path: Path) -> None:
    """The firmware's `10*log10()` floor must be the one `audio_frontend.py` applies.

    It is not a knob: shifting it moves every quiet mel bin on the board relative to the
    spectrograms the model was trained on, and nothing downstream would notice.
    """
    source = (
        Path(__file__).resolve().parents[1] / "tm_local" / "audio_frontend.py"
    ).read_text(encoding="utf-8")
    assert "np.maximum(mel_power, 1e-10)" in source
    assert kws_codegen.DEFAULT_LOG_EPSILON == 1e-10
    assert contract.DEFAULT_KWS_RUNTIME["log_epsilon"] == kws_codegen.DEFAULT_LOG_EPSILON

    c = _kws_contract(tmp_path)
    c.kws_runtime = {**contract.DEFAULT_KWS_RUNTIME, "log_epsilon": 1e-8}
    with pytest.raises(DeployError, match="log_epsilon 必須與訓練前處理一致"):
        kws_codegen.kws_tokens(c, 1024)


TEMPLATE = Path(__file__).resolve().parents[1] / "mcu_toolkit" / "apps" / "audio_kws" / "main.cpp.in"


def test_explicit_background_index_overrides_the_label_matcher(tmp_path: Path) -> None:
    """`background_class_id` picked in the UI must beat the firmware's name hints.

    Otherwise a project whose background class is called "教室" would train with one
    background class and trigger against another on the board.
    """
    c = _kws_contract(tmp_path)
    c.labels = ["red", "green", "blue"]
    c.class_thresholds = [0.6] * 3
    c.background_index = 2
    tokens = kws_codegen.kws_tokens(c, 1024)
    assert tokens["BACKGROUND_INDEX"] == "2" and tokens["HAS_BACKGROUND"] == "1"

    # The explicit index also wins over a label that WOULD match.
    c.labels = ["Background Noise", "green", "blue"]
    assert kws_codegen.kws_tokens(c, 1024)["BACKGROUND_INDEX"] == "2"


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="KWS template not vendored")
def test_explicit_background_index_reaches_the_firmware_define(tmp_path: Path) -> None:
    c = _kws_contract(tmp_path)
    c.labels = ["red", "green", "blue"]
    c.class_thresholds = [0.6] * 3
    c.background_index = 2
    rendered = kws_codegen.render_kws_main(c, 102400, TEMPLATE)
    assert "#define NUML_BACKGROUND_INDEX  (2U)" in rendered
    assert "#define NUML_HAS_BACKGROUND    (1)" in rendered


def test_background_index_none_keeps_the_label_matcher(tmp_path: Path) -> None:
    c = _kws_contract(tmp_path)
    assert c.background_index is None
    tokens = kws_codegen.kws_tokens(c, 1024)
    assert tokens["BACKGROUND_INDEX"] == "0" and tokens["HAS_BACKGROUND"] == "1"
    c.labels = ["yes", "no"]
    c.class_thresholds = [0.6, 0.6]
    tokens = kws_codegen.kws_tokens(c, 1024)
    assert tokens["BACKGROUND_INDEX"] == "0" and tokens["HAS_BACKGROUND"] == "0"


def test_out_of_range_background_index_is_refused(tmp_path: Path) -> None:
    c = _kws_contract(tmp_path)
    c.background_index = 7
    with pytest.raises(DeployError, match="background_class_id"):
        kws_codegen.kws_tokens(c, 1024)
