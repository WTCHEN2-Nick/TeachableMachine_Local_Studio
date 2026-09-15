from __future__ import annotations


class DeployError(Exception):
    """Student-facing MCU deploy failure; mapped to HTTP 400 by app.py."""


# One wording for "no Arm toolchain", wherever the student meets it: the pre-flight check in
# app.mcu_unavailable_reason() and the late check inside deploy_service.run_deploy(). Keeping
# two copies let their punctuation drift, which reads like two different problems.
TOOLCHAIN_MISSING_MESSAGE = (
    "找不到 Arm GNU Toolchain（arm-none-eabi-gcc）。請安裝後重新執行 01_INSTALL.bat，"
    r"或把 .zip 版解壓到 runtime\arm-gnu-toolchain\ 之後重新啟動 Studio。"
)


# --- Windows MAX_PATH pre-flight -------------------------------------------------------------
# The deepest compile-relevant header under a Studio checkout measures 152 characters relative
# to `mcu_toolkit/` (the ml-embedded-evaluation-kit headers under
# NuML_TFLM_Tool\templates\M55M1BSP\ThirdParty\...). 160 is that measurement plus a small margin
# for the build dir's own object/dep names; 250 keeps 10 characters of headroom below Windows'
# 260-character MAX_PATH for gcc's temporaries. Past that budget gcc fails to resolve includes
# with an English "No such file or directory" that means nothing to a student, so both the
# pre-flight in app.mcu_unavailable_reason() and the top of deploy_service.run_deploy() refuse
# the build with the advice below instead.
MCU_LONGEST_RELATIVE_PATH_CHARS = 160
MCU_MAX_PATH_BUDGET = 250


def long_path_reason(root: object) -> str | None:
    """Chinese advice when `root` is too deep for a Windows firmware build, else None."""
    length = len(str(root))
    if length + MCU_LONGEST_RELATIVE_PATH_CHARS <= MCU_MAX_PATH_BUDGET:
        return None
    return (
        f"Studio 資料夾路徑太長（{length} 字元），請把整個資料夾搬到較短的路徑"
        r"（例如 C:\TM_Studio）再重新啟動"
    )
