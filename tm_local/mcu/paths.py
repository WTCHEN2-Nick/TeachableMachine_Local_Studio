from __future__ import annotations

from pathlib import Path

from ..config import PROJECT_ROOT, RUNTIME_ROOT, TEMP_ROOT

MCU_TOOLKIT_ROOT = PROJECT_ROOT / "mcu_toolkit"
NUML_ROOT = MCU_TOOLKIT_ROOT / "NuML_TFLM_Tool"
BSP_ROOT = NUML_ROOT / "templates" / "M55M1BSP"
TEMPLATES_ROOT = NUML_ROOT / "templates" / "M55M1"
APPS_ROOT = MCU_TOOLKIT_ROOT / "apps"
VELA_EXE = MCU_TOOLKIT_ROOT / "vela" / "vela-5_1_0.exe"
VELA_INI = MCU_TOOLKIT_ROOT / "vela" / "default_vela.ini"
BOARDS_JSON = MCU_TOOLKIT_ROOT / "boards.json"
MANIFEST_JSON = MCU_TOOLKIT_ROOT / "manifest.json"
LOCAL_TOOLCHAIN_ROOT = RUNTIME_ROOT / "arm-gnu-toolchain"
LOCAL_NUMICRO_PACK_ROOT = RUNTIME_ROOT / "numicro-pack"
MCU_TEMP_ROOT = TEMP_ROOT / "mcu"


def toolkit_available() -> bool:
    return NUML_ROOT.is_dir() and BOARDS_JSON.is_file() and TEMPLATES_ROOT.is_dir()


def template_dir(board_template_name: str, application: str) -> Path:
    """`imgclass` uses the imgclass template; every other application uses generic."""
    folder = "imgclass_template" if application == "imgclass" else "generic_template"
    return TEMPLATES_ROOT / board_template_name / folder


def project_name_for(application: str) -> str:
    return "NN_ImgClassInference" if application == "imgclass" else "NN_ModelInference"
