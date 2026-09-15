from __future__ import annotations

import json

import pytest

from tm_local import config
from tm_local.mcu import boards, contract
from tm_local.mcu.errors import DeployError


def test_kind_registry_maps_three_deployable_kinds() -> None:
    assert config.MCU_APPLICATION_BY_KIND == {
        "image": "imgclass",
        "audio": "kws",
        "known_sound": "known_sound",
        "abnormal_sound": None,
    }
    assert config.MCU_DEPLOYABLE_KINDS == frozenset({"image", "audio", "known_sound"})
    assert config.mcu_application("image") == "imgclass"
    assert config.mcu_application("abnormal_sound") is None
    assert config.is_mcu_deployable("known_sound") is True
    assert config.is_mcu_deployable("abnormal_sound") is False
    assert config.DEPLOYMENT_TARGETS == (
        "pc",
        "NuMaker-M55M1",
        "NuGestureAI-M55M1",
        "NuMaker-VoiceAI-M55M1",
    )


def test_boards_json_loads_three_boards() -> None:
    table = boards.load_boards()
    assert set(table) == {"NuMaker-M55M1", "NuGestureAI-M55M1", "NuMaker-VoiceAI-M55M1"}
    x = table["NuMaker-M55M1"]
    g = table["NuGestureAI-M55M1"]
    voiceai = table["NuMaker-VoiceAI-M55M1"]
    assert x.has_lcd is True and x.display_output == "lcd" and x.flash_method == "nulink"
    assert g.has_lcd is False and g.display_output == "uvc_cdc" and g.flash_method == "msc"
    # NuMaker-M55M1 only has a board-mounted Nu-Link; NuGestureAI-M55M1's documented default is
    # its USB bootloader, but the user has verified Nu-Link also works on that board, so it lists
    # both -- msc first (it stays the default `flash_method`).
    assert x.flash_methods == ("nulink",)
    assert g.flash_methods == ("msc", "nulink")
    assert x.audio_dmic.channel_mask == "DMIC_CTL_CHEN0_Msk"
    assert g.audio_dmic.channel_mask == "DMIC_CTL_CHEN2_Msk"
    # Both boards run from the same 2 MiB internal flash; the caps are the measured
    # largest YAMNet depth that still fits with the firmware around it (see
    # config.KNOWN_SOUND_BOARD_MAX_DEPTH and deploy_service.FIRMWARE_CODE_BASELINE_BYTES).
    assert x.known_sound_max_depth == 12 and g.known_sound_max_depth == 11
    assert x.sram01_bytes == 1048576 and x.internal_flash_bytes == 2097152
    # 借用 GestureAI 的 template：這塊板在 NuML 裡沒有自己的目錄。
    assert voiceai.numl_template_board == "NuGestureAI-M55M1"
    # 沒有相機連接器，所以 image 韌體對它沒有意義（contract.validate() 會擋）。
    assert voiceai.has_camera is False
    assert voiceai.flash_method == "pyocd"
    assert voiceai.flash_methods == ("pyocd",)
    assert voiceai.audio_dmic.clk_macro == "SET_DMIC0_CLK_PA4"
    assert voiceai.audio_dmic.channel == 0
    assert voiceai.remove_source_basenames == ("BoardInit.cpp",)


def test_board_for_unknown_raises_deploy_error() -> None:
    with pytest.raises(DeployError):
        boards.board_for("NuMaker-M467HJ")


@pytest.mark.parametrize(
    "clk, dat, channel",
    [
        ("SET_DMIC0_CLK_PB4", "SET_DMIC1_DAT_PB3", 0),  # mixed pin pairs
        ("SET_DMIC1_CLK_PB2", "SET_DMIC1_DAT_PB3", 0),  # DMIC1 pins carry channels 2/3
        ("SET_DMIC0_CLK_PB4", "SET_DMIC0_DAT_PB5", 5),  # channel out of range
        ("SET_UART5_RXD_PB4", "SET_DMIC0_DAT_PB5", 0),  # not a DMIC macro
    ],
)
def test_invalid_dmic_wiring_rejected(clk: str, dat: str, channel: int) -> None:
    with pytest.raises(DeployError):
        boards.AudioDmic(clk_macro=clk, dat_macro=dat, channel=channel).validate()


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("not_json", "{ truncated"),
        ("not_an_object", "[1, 2, 3]"),
        ("entry_not_an_object", '{"NuMaker-M55M1": "oops"}'),
        ("missing_key", '{"NuMaker-M55M1": {"label": "x"}}'),
        (
            "wrong_type",
            (
            '{"NuMaker-M55M1": {"label": "x", "numl_template_board": "NuMaker-M55M1",'
            ' "has_lcd": true, "has_camera": true, "display_output": "lcd",'
            ' "audio_dmic": {"clk_macro": "SET_DMIC0_CLK_PB4",'
            ' "dat_macro": "SET_DMIC0_DAT_PB5", "channel": 0},'
            ' "known_sound_max_depth": 14, "internal_flash_bytes": "two megabytes",'
            ' "sram01_bytes": 1048576, "flash_method": "nulink"}}'
            ),
        ),
    ],
)
def test_corrupt_boards_json_raises_deploy_error(name: str, payload: str, tmp_path) -> None:
    """A hand-edited/truncated registry must reach app.py as a 400, never a raw 500."""
    path = tmp_path / f"{name}.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(DeployError) as excinfo:
        boards.load_boards(path)
    assert str(path) in str(excinfo.value)
    assert "01_INSTALL.bat" in str(excinfo.value)


def test_load_boards_from_custom_json(tmp_path) -> None:
    payload = {
        "NuMaker-M55M1": {
            "label": "x", "numl_template_board": "NuMaker-M55M1", "has_lcd": True,
            "has_camera": True, "display_output": "lcd",
            "audio_dmic": {"clk_macro": "SET_DMIC0_CLK_PB4", "dat_macro": "SET_DMIC0_DAT_PB5", "channel": 0},
            "known_sound_max_depth": 14, "internal_flash_bytes": 2097152, "sram01_bytes": 1048576,
            "flash_method": "nulink", "extra_defines": ["FOO=1"], "remove_defines": [],
        }
    }
    path = tmp_path / "boards.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    table = boards.load_boards(path)
    assert table["NuMaker-M55M1"].extra_defines == ("FOO=1",)
    # The v2.1.0 singular "flash_method" field must still load -- a hand-edited or older
    # boards.json that has not been migrated to the plural array yet is not broken by this.
    assert table["NuMaker-M55M1"].flash_methods == ("nulink",)
    assert table["NuMaker-M55M1"].flash_method == "nulink"


def test_load_boards_accepts_the_new_plural_flash_methods_array(tmp_path) -> None:
    """A board that genuinely supports more than one way to flash lists them, ordered --
    the first entry stays the default that older callers (`board.flash_method`) still see."""
    board = _minimal_board()
    del board["flash_method"]
    board["flash_methods"] = ["msc", "nulink"]
    path = _write_boards(tmp_path, board)
    loaded = boards.load_boards(path)["Probe-M55M1"]
    assert loaded.flash_methods == ("msc", "nulink")
    assert loaded.flash_method == "msc"


@pytest.mark.parametrize(
    "flash_methods",
    [
        (),  # empty: a board must declare at least one way to flash it
        ("msc", "msc"),  # duplicate
        ("keil_uv4",),  # not one of the three methods Studio actually implements
        ("msc", "keil_uv4"),  # one valid, one not -- still refused
    ],
)
def test_flash_methods_validation_rejects_empty_duplicate_and_unknown(
    flash_methods: tuple[str, ...],
) -> None:
    board = boards.Board(
        name="Probe-M55M1",
        label="x",
        numl_template_board="NuGestureAI-M55M1",
        has_lcd=False,
        has_camera=False,
        display_output="usb_cdc",
        audio_dmic=boards.AudioDmic("SET_DMIC0_CLK_PA4", "SET_DMIC0_DAT_PA5", 0),
        known_sound_max_depth=10,
        internal_flash_bytes=2097152,
        sram01_bytes=1048576,
        flash_methods=flash_methods,
    )
    with pytest.raises(DeployError, match="flash_methods"):
        board.validate()


def _minimal_board(**overrides) -> dict:
    """A board dict that load_boards() accepts, so a test can vary one field at a time."""
    board = {
        "label": "Probe 板",
        "numl_template_board": "NuGestureAI-M55M1",
        "has_lcd": False,
        "has_camera": False,
        "display_output": "usb_cdc",
        "audio_dmic": {
            "clk_macro": "SET_DMIC0_CLK_PA4",
            "dat_macro": "SET_DMIC0_DAT_PA5",
            "channel": 0,
        },
        "known_sound_max_depth": 10,
        "internal_flash_bytes": 2097152,
        "sram01_bytes": 1048576,
        "flash_method": "pyocd",
    }
    board.update(overrides)
    return board


def _write_boards(tmp_path, board: dict, name: str = "Probe-M55M1"):
    path = tmp_path / "boards.json"
    path.write_text(json.dumps({name: board}), encoding="utf-8")
    return path


def test_usb_cdc_and_pyocd_are_accepted_board_values(tmp_path) -> None:
    """The VoiceAI board needs a display_output and a flash_method neither older board uses."""
    path = _write_boards(tmp_path, _minimal_board())
    board = boards.load_boards(path)["Probe-M55M1"]
    assert board.display_output == "usb_cdc"
    assert board.flash_method == "pyocd"


def test_a_board_may_declare_its_own_sources(tmp_path) -> None:
    """A board with no NuML template of its own swaps files out of the template it borrows."""
    (tmp_path / "boards" / "Probe-M55M1").mkdir(parents=True)
    (tmp_path / "boards" / "Probe-M55M1" / "BoardInit_Probe.cpp").write_text(
        "void x(void) {}\n", encoding="utf-8"
    )
    path = _write_boards(
        tmp_path,
        _minimal_board(
            extra_sources=["BoardInit_Probe.cpp"],
            remove_source_basenames=["BoardInit.cpp"],
        ),
    )
    board = boards.load_boards(path)["Probe-M55M1"]
    assert board.extra_sources == ("BoardInit_Probe.cpp",)
    assert board.remove_source_basenames == ("BoardInit.cpp",)


def test_a_board_source_that_does_not_exist_is_refused_at_load_time(tmp_path) -> None:
    """Naming a missing file must fail here, not halfway through assembling a build."""
    path = _write_boards(tmp_path, _minimal_board(extra_sources=["Nowhere.cpp"]))
    with pytest.raises(DeployError, match="Nowhere.cpp"):
        boards.load_boards(path)


def test_a_board_with_no_sources_of_its_own_needs_no_directory(tmp_path) -> None:
    """The two existing boards declare nothing, and must not be made to create an empty dir."""
    path = _write_boards(tmp_path, _minimal_board())
    board = boards.load_boards(path)["Probe-M55M1"]
    assert board.extra_sources == ()
    assert board.remove_source_basenames == ()


def test_deploying_an_image_model_to_the_camera_less_board_is_refused(tmp_path) -> None:
    """This guard already exists in contract.validate(); this proves it covers the new board."""

    board = boards.board_for("NuMaker-VoiceAI-M55M1")
    c = contract.DeployContract(
        kind="image", application="imgclass", board=board, project_id="p", project_name="n",
        labels=["a", "b"], models_dir=tmp_path, int8_path=tmp_path / "m.tflite",
        model_sha256="x",
        input=contract.TensorInfo("in", (1, 224, 224, 3), "int8", 1.0, -128),
        output=contract.TensorInfo("out", (1, 2), "int8", 0.0039, -128),
        image_size=224, audio_frontend=None, yamnet_frontend=None, encoder_depth=None,
        detection_threshold=0.5, class_thresholds=[0.5, 0.5], hop_seconds=0.5,
        clip_seconds=1.0, peak_hold_seconds=1.5,
        kws_runtime=dict(contract.DEFAULT_KWS_RUNTIME),
    )
    with pytest.raises(DeployError, match="沒有相機"):
        contract.validate(c)
