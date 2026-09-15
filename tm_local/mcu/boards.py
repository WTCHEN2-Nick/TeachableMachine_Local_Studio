from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .errors import DeployError
from .paths import BOARDS_JSON

_DMIC_MACRO = re.compile(r"^SET_DMIC([01])_(CLK|DAT)_P[A-H]\d{1,2}$")
_DISPLAY_OUTPUTS = ("lcd", "uvc_cdc", "usb_cdc")
_FLASH_METHODS = ("nulink", "msc", "pyocd")


@dataclass(frozen=True)
class AudioDmic:
    clk_macro: str
    dat_macro: str
    channel: int

    @property
    def channel_mask(self) -> str:
        return f"DMIC_CTL_CHEN{self.channel}_Msk"

    def validate(self) -> AudioDmic:
        clk = _DMIC_MACRO.match(self.clk_macro)
        dat = _DMIC_MACRO.match(self.dat_macro)
        if not clk or clk.group(2) != "CLK" or not dat or dat.group(2) != "DAT":
            raise DeployError(f"DMIC 腳位巨集無效：{self.clk_macro} / {self.dat_macro}")
        if clk.group(1) != dat.group(1):
            raise DeployError("DMIC CLK 與 DAT 必須使用同一組腳位（都是 DMIC0 或都是 DMIC1）")
        if not 0 <= int(self.channel) <= 3:
            raise DeployError(f"DMIC channel 必須是 0..3，收到 {self.channel}")
        pair = clk.group(1)
        expected = ("0", "1") if pair == "0" else ("2", "3")
        if str(self.channel) not in expected:
            raise DeployError(
                f"DMIC{pair} 腳位只能對應 channel {expected[0]}/{expected[1]}，收到 {self.channel}"
            )
        return self


@dataclass(frozen=True)
class Board:
    name: str
    label: str
    numl_template_board: str
    has_lcd: bool
    has_camera: bool
    display_output: str
    audio_dmic: AudioDmic
    known_sound_max_depth: int
    internal_flash_bytes: int
    sram01_bytes: int
    # Ordered; the first entry is the default -- what the UI preselects and what run_deploy()'s
    # report describes when nobody picks a method. A board may genuinely support more than one
    # (NuGestureAI-M55M1 has both its USB bootloader and a working on-board Nu-Link), so this is
    # a tuple, not a single string; see the `flash_method` property below for the singular view.
    flash_methods: tuple[str, ...]
    extra_defines: tuple[str, ...] = ()
    remove_defines: tuple[str, ...] = ()
    # 板子專屬的韌體來源。借用別塊板 template 的板子（VoiceAI 借 GestureAI）用這兩個欄位
    # 換掉 template 裡不適用的檔案；檔名相對於 mcu_toolkit/boards/<board key>/。
    extra_sources: tuple[str, ...] = ()
    remove_source_basenames: tuple[str, ...] = ()

    @property
    def flash_method(self) -> str:
        """The default method -- what the UI preselects and what older readers still ask for."""
        return self.flash_methods[0]

    def validate(self) -> Board:
        if self.display_output not in _DISPLAY_OUTPUTS:
            raise DeployError(f"{self.name}: display_output 無效 {self.display_output!r}")
        if not self.flash_methods:
            raise DeployError(f"{self.name}: flash_methods 不可為空")
        if len(set(self.flash_methods)) != len(self.flash_methods):
            raise DeployError(f"{self.name}: flash_methods 有重複：{self.flash_methods!r}")
        invalid = [m for m in self.flash_methods if m not in _FLASH_METHODS]
        if invalid:
            raise DeployError(f"{self.name}: flash_methods 無效 {invalid!r}")
        self.audio_dmic.validate()
        return self


def board_source_dir(root: Path, board_name: str) -> Path:
    """Where a board keeps the firmware sources it substitutes into a borrowed template."""
    return root / "boards" / board_name


def _board_from_dict(name: str, raw: dict, root: Path) -> Board:
    dmic = raw["audio_dmic"]
    # Ordered array is the current shape; a hand-edited or older registry may still carry the
    # v2.1.0 singular "flash_method" string instead, and that keeps loading too.
    raw_methods = raw.get("flash_methods")
    if raw_methods is None:
        raw_methods = [raw["flash_method"]]  # KeyError here if neither key is present
    board = Board(
        name=name,
        label=str(raw["label"]),
        numl_template_board=str(raw["numl_template_board"]),
        has_lcd=bool(raw["has_lcd"]),
        has_camera=bool(raw["has_camera"]),
        display_output=str(raw["display_output"]),
        audio_dmic=AudioDmic(str(dmic["clk_macro"]), str(dmic["dat_macro"]), int(dmic["channel"])),
        known_sound_max_depth=int(raw["known_sound_max_depth"]),
        internal_flash_bytes=int(raw["internal_flash_bytes"]),
        sram01_bytes=int(raw["sram01_bytes"]),
        flash_methods=tuple(str(x) for x in raw_methods),
        extra_defines=tuple(str(x) for x in raw.get("extra_defines", [])),
        remove_defines=tuple(str(x) for x in raw.get("remove_defines", [])),
        extra_sources=tuple(str(x) for x in raw.get("extra_sources", [])),
        remove_source_basenames=tuple(str(x) for x in raw.get("remove_source_basenames", [])),
    ).validate()
    for source in board.extra_sources:
        if not (board_source_dir(root, name) / source).is_file():
            raise DeployError(
                f"{name}: 板子宣告的韌體來源不存在：{source}"
                f"（應位於 {root.name}/boards/{name}/）"
            )
    return board


@lru_cache(maxsize=4)
def _load_cached(path: str) -> dict[str, Board]:
    """Parse boards.json into Boards, or raise DeployError naming the file.

    boards.json is a plain text file inside the student's install, so it can be truncated by a
    half-finished copy or hand-edited into the wrong shape. Every such failure has to arrive as
    a DeployError (400 + Chinese) rather than a raw KeyError/TypeError/JSONDecodeError, which
    app.py would turn into a 500 the student cannot act on.
    """
    target = Path(path)
    advice = "請重新執行 01_INSTALL.bat 取回原始檔"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DeployError(f"板子註冊表 {target} 讀取失敗：{exc}；{advice}") from exc
    except ValueError as exc:  # json.JSONDecodeError
        raise DeployError(f"板子註冊表 {target} 不是有效的 JSON：{exc}；{advice}") from exc
    if not isinstance(payload, dict):
        raise DeployError(f"板子註冊表 {target} 的內容不是物件；{advice}")
    table: dict[str, Board] = {}
    for name, raw in payload.items():
        if not isinstance(raw, dict):
            raise DeployError(f"板子註冊表 {target} 的 {name} 設定不是物件；{advice}")
        try:
            table[str(name)] = _board_from_dict(str(name), raw, Path(path).parent)
        except DeployError:
            raise  # already student-facing (validate() names the offending field)
        except (KeyError, TypeError, ValueError) as exc:
            raise DeployError(
                f"板子註冊表 {target} 的 {name} 設定不正確"
                f"（{type(exc).__name__}: {exc}）；{advice}"
            ) from exc
    return table


def load_boards(path: Path | None = None) -> dict[str, Board]:
    target = Path(path) if path is not None else BOARDS_JSON
    if not target.is_file():
        raise DeployError(f"找不到板子註冊表 {target}；mcu_toolkit 尚未安裝")
    return dict(_load_cached(str(target.resolve())))


def board_names(path: Path | None = None) -> list[str]:
    return list(load_boards(path).keys())


def board_for(name: str, path: Path | None = None) -> Board:
    table = load_boards(path)
    if name not in table:
        raise DeployError(f"不支援的板子 {name!r}；可用：{', '.join(table)}")
    return table[name]
