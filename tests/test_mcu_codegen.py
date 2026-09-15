from __future__ import annotations

from pathlib import Path

import pytest

from tm_local.mcu import codegen, paths
from tm_local.mcu.errors import DeployError

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mcu"


def test_render_tokens_replaces_and_rejects_leftovers() -> None:
    assert codegen.render_tokens("a=@@A@@;b=@@B_2@@", {"A": "1", "B_2": "x"}) == "a=1;b=x"
    with pytest.raises(DeployError, match="@@MISSING@@"):
        codegen.render_tokens("@@MISSING@@", {})


def test_cpp_escape() -> None:
    assert codegen.cpp_escape('a"b\\c\n') == 'a\\"b\\\\c\\n'
    assert codegen.cpp_escape("貓") == "貓"


def test_cpp_escape_control_chars_use_octal_not_hex() -> None:
    # \xNN would be wrong here: C's hex escape is greedy and keeps consuming hex digits, so
    # "\x011" would parse as a single (invalid, 3-digit) hex escape rather than \x01 + "1".
    assert codegen.cpp_escape("\x01\x7f") == "\\001\\177"
    assert codegen.cpp_escape("a\x00b") == "a\\000b"


def test_model_cpp_text_layout() -> None:
    text = codegen.model_cpp_text(bytes(range(20)))
    assert "namespace arm {" in text and "namespace app {" in text and "namespace nn {" in text
    assert "static const uint8_t nn_model[] MODEL_TFLITE_ATTRIBUTE =" in text
    assert "0x00, 0x01, 0x02" in text
    assert "const uint8_t * GetModelPointer()" in text and "size_t GetModelLen()" in text
    assert text.count("\n") > 5 and "0x13," in text


def test_model_cpp_text_rejects_empty_bytes() -> None:
    # A zero-byte Vela output must never silently become a firmware with an empty model array.
    with pytest.raises(DeployError):
        codegen.model_cpp_text(b"")


def test_labels_cpp_and_hpp(tmp_path: Path) -> None:
    codegen.write_labels(["cat", 'dog "x"', "背景"], tmp_path)
    cpp = (tmp_path / "Labels.cpp").read_text(encoding="utf-8")
    hpp = (tmp_path / "include" / "Labels.hpp").read_text(encoding="utf-8")
    assert '"cat",' in cpp and '"dog \\"x\\"",' in cpp and '"背景",' in cpp
    assert "LABELS_ATTRIBUTE" in cpp
    for symbol in ("size_t GetLabelCount(void)", "const char *GetLabelByIndex(size_t index)",
                   "bool GetLabelsVector(std::vector<std::string> &labels)"):
        assert symbol in cpp and symbol.split("(")[0].split()[-1].lstrip("*") in hpp


def test_activation_define() -> None:
    assert codegen.activation_define(183296) == "#define ACTIVATION_BUF_SZ (183296)"


def test_generate_model_sources_with_vendored_numl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not paths.toolkit_available():
        pytest.skip("mcu_toolkit not vendored")
    from tm_local.mcu import vela

    if not paths.VELA_EXE.is_file():
        pytest.skip("vela exe not vendored")
    # The real deploy pipeline compiles Vela's output under MCU_TEMP_ROOT (outside app_dir);
    # point MCU_TEMP_ROOT at the same folder used for the Vela compile below so this test can
    # actually exercise the R2 scrub of that path out of numl_codegen.log.
    vela_root = tmp_path / "vela"
    monkeypatch.setattr(codegen.paths, "MCU_TEMP_ROOT", vela_root)
    result = vela.compile_model(FIXTURES / "vww4_128_128_INT8.tflite", vela_root)
    app = tmp_path / "NN_ImgClassInference"
    (app / "Model" / "include").mkdir(parents=True)
    codegen.generate_model_sources("imgclass", result.vela_tflite, app, result.summary_csv)
    hpp = (app / "Model" / "include" / "MobileNetModel.hpp").read_text(encoding="utf-8")
    cpp = (app / "Model" / "MobileNetModel.cpp").read_text(encoding="utf-8")
    assert "ms_maxOpCnt" in hpp
    assert "AddEthosU" in cpp
    assert (app / "main.cpp").is_file()  # NuML also renders its main; callers overwrite it
    log_path = app / "numl_codegen.log"
    assert log_path.is_file()  # captured stdout, written on success too
    log = log_path.read_text(encoding="utf-8")
    assert str(app) not in log
    assert str(vela_root) not in log


class _RaisingCodegen:
    """Stand-in for a vendored NuML codegen class whose `code_gen()` raises -- e.g. an
    unsupported opcode (`ValueError`) or a malformed Vela summary (`KeyError`)."""

    @classmethod
    def from_args(cls, *args: object, **kwargs: object) -> _RaisingCodegen:
        return cls()

    def code_gen(self) -> None:
        raise ValueError("Unknown custom opcode 999")


class _SilentCodegen:
    """Stand-in for a vendored NuML codegen class that returns success but writes nothing --
    `code_gen()`'s return value alone is not proof the expected files exist."""

    @classmethod
    def from_args(cls, *args: object, **kwargs: object) -> _SilentCodegen:
        return cls()

    def code_gen(self) -> None:
        return None


def test_generate_model_sources_wraps_vendor_exception_as_deploy_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = tmp_path / "NN_ImgClassInference"
    (app / "Model" / "include").mkdir(parents=True)
    monkeypatch.setattr(codegen, "_import_numl_codegen", lambda application: _RaisingCodegen)
    with pytest.raises(DeployError, match="韌體模型程式碼產生失敗"):
        codegen.generate_model_sources(
            "imgclass", tmp_path / "model_vela.tflite", app, tmp_path / "summary.csv"
        )


def test_generate_model_sources_raises_when_expected_outputs_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = tmp_path / "NN_ImgClassInference"
    (app / "Model" / "include").mkdir(parents=True)
    monkeypatch.setattr(codegen, "_import_numl_codegen", lambda application: _SilentCodegen)
    with pytest.raises(DeployError, match="缺少預期輸出"):
        codegen.generate_model_sources(
            "imgclass", tmp_path / "model_vela.tflite", app, tmp_path / "summary.csv"
        )


def test_model_pointer_declaration_is_const_in_every_app_template() -> None:
    """One return type for `GetModelPointer()` across every translation unit that sees it.

    `model_cpp_text()` above defines it as `const uint8_t *` (the model table lives in flash).
    Nuvoton's upstream wording declares it without the const, which is a formal ODR violation
    no compiler has to diagnose -- it happens to link on this ABI. The vendoring scripts
    rewrite the declaration, so the templates must carry only the const form.
    `arm::app::Model::Init()` takes `const uint8_t* nnModelAddr`, so the call sites need no cast.
    """
    if not paths.toolkit_available():
        pytest.skip("mcu_toolkit not vendored")
    templates = [
        paths.APPS_ROOT / "known_sound" / "main.cpp.in",
        paths.APPS_ROOT / "audio_kws" / "main.cpp.in",
        paths.APPS_ROOT / "image" / "NuMaker-M55M1" / "main.cpp.in",
        paths.APPS_ROOT / "image" / "NuGestureAI-M55M1" / "main.cpp.in",
    ]
    for template in templates:
        if not template.is_file():
            pytest.skip(f"{template.name} not vendored")
        text = template.read_text(encoding="utf-8")
        assert text.count("extern const uint8_t *GetModelPointer();") == 1, template
        assert "extern uint8_t *GetModelPointer();" not in text, template
        assert "arm::app::nn::GetModelPointer()" in text, template
    init = (
        paths.BSP_ROOT
        / "ThirdParty/ml-embedded-evaluation-kit/source/application/api/common/include/Model.hpp"
    ).read_text(encoding="utf-8")
    assert "const uint8_t* nnModelAddr" in init
