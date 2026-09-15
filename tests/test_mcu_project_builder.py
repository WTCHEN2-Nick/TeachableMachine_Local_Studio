from __future__ import annotations


def _contract_with_board(**board_overrides):
    from types import SimpleNamespace

    from tm_local.mcu.boards import AudioDmic, Board

    board = Board(
        name="Probe-M55M1", label="Probe 板", numl_template_board="NuGestureAI-M55M1",
        has_lcd=False, has_camera=False, display_output="usb_cdc",
        audio_dmic=AudioDmic("SET_DMIC0_CLK_PA4", "SET_DMIC0_DAT_PA5", 0),
        known_sound_max_depth=10, internal_flash_bytes=2097152, sram01_bytes=1048576,
        flash_methods=("pyocd",), **board_overrides,
    )
    return SimpleNamespace(application="known_sound", board=board)


def test_a_board_contributes_its_own_sources_to_the_overlay(tmp_path) -> None:
    """A borrowed template's BoardInit.cpp comes out; the board's own goes in."""
    from tm_local.mcu import project_builder

    board_dir = tmp_path / "boards" / "Probe-M55M1"
    board_dir.mkdir(parents=True)
    (board_dir / "BoardInit_Probe.cpp").write_text("void x(void) {}\n", encoding="utf-8")

    class _Module:
        @staticmethod
        def overlay(contract):
            return {"add_sources": ["Frontend/numl_dmic.c"], "add_defines": ["APP_ONE=1"]}

    contract = _contract_with_board(
        extra_sources=("BoardInit_Probe.cpp",),
        remove_source_basenames=("BoardInit.cpp",),
        extra_defines=("DEBUG_PORT_UART_IDX=4",),
        remove_defines=("DEBUG_PORT=UART5",),
    )
    kwargs = project_builder._overlay_kwargs(
        _Module, contract, tmp_path / "app", 102400, board_dir
    )

    assert board_dir / "BoardInit_Probe.cpp" in kwargs["add_sources"]
    assert "BoardInit.cpp" in kwargs["remove_source_basenames"]
    # 板子的來源不得取代 app 的，兩者都要在。
    assert (tmp_path / "app" / "Frontend" / "numl_dmic.c") in kwargs["add_sources"]
    # defines 的既有行為不得改變。
    assert "APP_ONE=1" in kwargs["add_defines"]
    assert "DEBUG_PORT_UART_IDX=4" in kwargs["add_defines"]
    assert "DEBUG_PORT=UART5" in kwargs["remove_defines"]


def test_a_board_with_no_sources_changes_nothing(tmp_path) -> None:
    """The two existing boards must produce exactly the overlay they produce today."""
    from tm_local.mcu import project_builder

    class _Module:
        @staticmethod
        def overlay(contract):
            return {"add_sources": ["Frontend/numl_dmic.c"]}

    contract = _contract_with_board()
    kwargs = project_builder._overlay_kwargs(
        _Module, contract, tmp_path / "app", 102400, tmp_path / "boards" / "Probe-M55M1"
    )
    assert kwargs["add_sources"] == [tmp_path / "app" / "Frontend" / "numl_dmic.c"]
    assert kwargs["remove_source_basenames"] == []


def test_a_board_removing_a_file_the_template_lacks_is_refused(tmp_path) -> None:
    """The existing removal check must cover board declarations, not just app ones."""
    import pytest

    from tm_local.mcu import project_builder, records
    from tm_local.mcu.errors import DeployError

    board_dir = tmp_path / "boards" / "Probe-M55M1"
    board_dir.mkdir(parents=True)
    (board_dir / "BoardInit_Probe.cpp").write_text("void x(void) {}\n", encoding="utf-8")

    class _Module:
        @staticmethod
        def overlay(contract):
            return {}

    contract = _contract_with_board(
        extra_sources=("BoardInit_Probe.cpp",),
        remove_source_basenames=("BoardInit.cpp",),
    )
    kwargs = project_builder._overlay_kwargs(
        _Module, contract, tmp_path / "app", 102400, board_dir
    )
    # A template that has since renamed BoardInit.cpp -- the board's removal now matches nothing.
    before = [
        records.SourceFile(path=tmp_path / "Other.cpp", kind="cxx", progen_relpath="Other.cpp")
    ]
    with pytest.raises(DeployError, match="BoardInit.cpp"):
        project_builder._check_removals_matched(
            before, kwargs["remove_source_basenames"], tmp_path / "template"
        )
