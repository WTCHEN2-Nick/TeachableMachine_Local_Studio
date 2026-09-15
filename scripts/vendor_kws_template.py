"""Vendor the App Builder's live Audio-KWS runtime into mcu_toolkit/apps/audio_kws/.

Run by the maintainer through ``vendor_mcu_toolkit.py kws-template``; students never run it.

The App Builder renders ``main.cpp`` from one project's contract, which means the numbers are
baked in. The Studio needs the opposite: one static file with 35 ``@@TOKEN@@`` placeholders
that ``tm_local.mcu.kws_codegen.render_kws_main()`` fills per deploy, so a deploy needs neither
the App Builder nor jinja2. This script renders upstream **once**, with obviously-fake probe
values, and rewrites every baked-in number into its token.

Two transformations go beyond "swap a number for a token", and both are deliberate:

* **The mel filterbank is replaced, not parameterised.** Upstream computes its own triangular
  filters (``_mel_filters()``: ``floor((fft+1)*hz/rate)`` bin edges, ``s_melLeft/Center/Right``
  plus an ``s_melInverseSum`` normaliser). That is a *different* filterbank from
  ``tm_local.audio_frontend.mel_filterbank()``, which is what training, Preview and the INT8
  calibration all use. Shipping it would give the board a front-end that is plausible and
  systematically wrong, so the whole block -- and the ``NuML_MelPower()`` that walks it -- is
  deleted and replaced by a sparse (start, count, offset, weights) accumulation driven by the
  Studio's own exported tables. Those weights are already row-normalised, hence no inverse-sum
  multiply. The Hann window is likewise embedded as a table instead of being recomputed with
  ``arm_cos_f32``, because upstream's phase scale happens to match ``np.hanning`` only for the
  symmetric case and nothing would catch it drifting.
* **``NUML_BACKGROUND_INDEX`` is gated behind ``#if NUML_HAS_BACKGROUND``.** Upstream assumes
  every KWS project has a background/silence class. A Studio project may not (``["yes","no"]``
  is a perfectly ordinary two-keyword project), and there the index falls back to 0 -- which
  ungated would make "yes" permanently untriggerable *and* its own trigger-margin baseline.
  :func:`assert_background_gated` refuses to write a template where any read of that macro
  escaped the gate.

Every anchor is checked for an exact match count; a silently skipped edit would ship firmware
that looks right and listens with the wrong filterbank, so a miss is a hard ``SystemExit``
naming the anchor rather than a warning.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import ModuleType

APP_BUILDER_VERSION = "0.1.8"
EXPECTED_RUNTIME_MARKER = "NUML_APP_BUILDER_AUDIO_KWS_RUNTIME_V0_1_8"
STUDIO_MARKER = "TEACHABLE_MACHINE_LOCAL_STUDIO_AUDIO_KWS_RUNTIME_V1"

#: Probe values fed to ``_render_main()``. They exist only to be replaced: the arena and the
#: labels are deliberately un-C-like so a surviving one fails the post-checks loudly.
PROBE_ARENA = 999999
PROBE_CONFIG = {
    "sample_rate": 16000,
    "clip_samples": 16000,
    "inference_hop": 4000,
    "frame_length": 400,
    "frame_step": 160,
    "frame_count": 98,
    "fft_length": 512,
    "mel_bins": 40,
    "db_floor": -80.0,
    "log_epsilon": 1e-10,
    "labels": ["__L0__", "__L1__"],
    "background_index": 0,
    "trigger_threshold": 0.65,
    "cooldown_hops": 4,
    "input_scale": 0.00392157,
    "input_zero_point": -128,
    "output_scale": 0.00390625,
    "output_zero_point": -128,
    "frequency_min": 20.0,
    "frequency_max": 8000.0,
}
PROBE_DMIC = {"clk_macro": "SET_DMIC0_CLK_PB4", "dat_macro": "SET_DMIC0_DAT_PB5", "channel": 0}

#: ``#define`` lines that become tokens, in the order upstream emits them. The value text is
#: the full parenthesised body, so the unsigned suffix travels with the macro that needs it.
DEFINE_TOKENS = (
    ("NUML_SAMPLE_RATE", "@@SAMPLE_RATE@@U"),
    ("NUML_CLIP_SAMPLES", "@@CLIP_SAMPLES@@U"),
    ("NUML_INFERENCE_HOP", "@@INFERENCE_HOP@@U"),
    ("NUML_FRAME_LENGTH", "@@FRAME_LENGTH@@U"),
    ("NUML_FRAME_STEP", "@@FRAME_STEP@@U"),
    ("NUML_FRAME_COUNT", "@@FRAME_COUNT@@U"),
    ("NUML_FFT_LENGTH", "@@FFT_LENGTH@@U"),
    ("NUML_MEL_BINS", "@@MEL_BINS@@U"),
    ("NUML_LABEL_COUNT", "@@LABEL_COUNT@@U"),
    ("NUML_BACKGROUND_INDEX", "@@BACKGROUND_INDEX@@U"),
    # The four tunables below are hard-coded upstream; as tokens they make the contract's
    # kws_runtime settings actually reach the board instead of sitting inert in project.json.
    ("NUML_SMOOTH_WINDOWS", "@@SMOOTH_WINDOWS@@U"),
    ("NUML_REQUIRED_HITS", "@@REQUIRED_HITS@@U"),
    ("NUML_COOLDOWN_HOPS", "@@COOLDOWN_HOPS@@U"),
    ("NUML_TRIGGER_THRESHOLD", "@@TRIGGER_THRESHOLD@@"),
    ("NUML_TRIGGER_MARGIN", "@@TRIGGER_MARGIN@@"),
    ("NUML_REARM_SCORE", "@@REARM_SCORE@@"),
    ("NUML_RMS_GATE_DBFS", "@@RMS_GATE_DBFS@@"),
    ("NUML_DB_FLOOR", "@@DB_FLOOR@@"),
    ("NUML_LOG_EPSILON", "@@LOG_EPSILON@@"),
    ("NUML_INPUT_SCALE", "@@INPUT_SCALE@@"),
    ("NUML_INPUT_ZERO_POINT", "@@INPUT_ZERO_POINT@@"),
    ("NUML_OUTPUT_SCALE", "@@OUTPUT_SCALE@@"),
    ("NUML_OUTPUT_ZERO_POINT", "@@OUTPUT_ZERO_POINT@@"),
)

#: The complete set ``kws_codegen.kws_tokens()`` supplies; ``render_tokens()`` raises on any
#: leftover ``@@...@@``, so the template must carry exactly these -- no more, no fewer.
#: ``tests/test_mcu_kws_build.py`` cross-checks this list against the real token table.
DEPLOY_TOKENS = (
    "ARENA",
    "SAMPLE_RATE",
    "CLIP_SAMPLES",
    "INFERENCE_HOP",
    "FRAME_LENGTH",
    "FRAME_STEP",
    "FRAME_COUNT",
    "FFT_LENGTH",
    "MEL_BINS",
    "MEL_NNZ",
    "LABEL_COUNT",
    "BACKGROUND_INDEX",
    "HAS_BACKGROUND",
    "SMOOTH_WINDOWS",
    "REQUIRED_HITS",
    "COOLDOWN_HOPS",
    "TRIGGER_THRESHOLD",
    "TRIGGER_MARGIN",
    "REARM_SCORE",
    "RMS_GATE_DBFS",
    "DB_FLOOR",
    "LOG_EPSILON",
    "INPUT_SCALE",
    "INPUT_ZERO_POINT",
    "OUTPUT_SCALE",
    "OUTPUT_ZERO_POINT",
    "LABELS",
    "HANN",
    "MEL_START",
    "MEL_COUNT",
    "MEL_OFFSET",
    "MEL_WEIGHTS",
    "DMIC_CLK_MACRO",
    "DMIC_DAT_MACRO",
    "DMIC_CHANNEL_MASK",
)

#: Nothing from the probe render, and nothing from upstream's own mel filterbank, may survive.
FORBIDDEN = (
    str(PROBE_ARENA),
    "__L0__",
    "__L1__",
    "s_melLeft",
    "s_melCenter",
    "s_melRight",
    "s_melInverseSum",
    "arm_cos_f32",
    "s_hann[",
    "phaseScale",
    "SET_DMIC0_CLK_PB4",
    "SET_DMIC0_DAT_PB5",
    "DMIC_CTL_CHEN0_Msk",
    # arm-none-eabi-g++ cannot deduce std::max(int, int32_t); see the "int8 saturation clamp"
    # anchor. Leaving this form behind would break the build, not merely the semantics.
    "std::max(-128,",
    # The non-const declaration upstream ships; see the "GetModelPointer const" anchor.
    "extern uint8_t *GetModelPointer();",
    EXPECTED_RUNTIME_MARKER,
)

REQUIRED = (
    STUDIO_MARKER,
    "extern const uint8_t *GetModelPointer();",
    "NUML_USE_USB_CDC",
    "s_hannTable",
    "s_melWeights[NUML_MEL_NNZ]",
    "BOARD_LOG_INIT();",
    "BOARD_LOG_PUMP();",
    "BOARD_LOG_WAIT();",
    "NVT_NONCACHEABLE",
    # Re-arming must not depend on NUML_REARM_SCORE alone: see BACKGROUND_GATE.
    "const bool released = !scorePass || !marginPass;",
    "if (quiet || background || released)",
)

#: A vendored file carrying one of these leaks the maintainer's machine into the student zip.
PRIVATE_PATH_RE = re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}|/Users/|\.venv", re.IGNORECASE)

HEADER = (
    f"/* {STUDIO_MARKER}: generated by scripts/vendor_kws_template.py from NuML App Builder "
    f"v{APP_BUILDER_VERSION} audio_kws_runtime._render_main; tables come from "
    "tm_local.audio_frontend. */"
)

CDC_BLOCK = """
#if defined(NUML_USE_USB_CDC)
#include "numl_cdc.h"
#include "numl_usbd.h"
#define BOARD_LOG_INIT()   NuML_USBD_Init()
#define BOARD_LOG_PUMP()   NuML_CDC_Pump()

/* Bounded wait for a terminal to open the virtual COM port, mirroring known_sound's
 * BoardConfig.h. The banner and the model-contract self-test print exactly once, and a host
 * needs a second or two to enumerate the port; without this the student's first lines are
 * already gone by the time PuTTY connects. It gives up after NUML_BOARD_LOG_WAIT_MS rather
 * than refusing to boot with no terminal attached.
 *
 * 10 ms per step on purpose: CLK_SysTickDelay counts a 24-bit SysTick, which at 220 MHz tops
 * out around 76 ms, so a single long delay would silently wrap. */
#ifndef NUML_BOARD_LOG_WAIT_MS
#define NUML_BOARD_LOG_WAIT_MS 10000
#endif
static inline void NuML_BoardLogWait(void)
{
    for (int step = 0; step < (NUML_BOARD_LOG_WAIT_MS / 10); step++)
    {
        if (NuML_CDC_IsOpen()) { return; }

        CLK_SysTickDelay(10000);
    }
}
#define BOARD_LOG_WAIT()   NuML_BoardLogWait()
#else
#define BOARD_LOG_INIT()   ((void)0)
#define BOARD_LOG_PUMP()   ((void)0)
#define BOARD_LOG_WAIT()   ((void)0)
#endif
"""

MEL_TABLES = """\
/* Sparse mel filterbank exported from tm_local.audio_frontend.mel_filterbank(): row-normalised
 * triangular filters as (start bin, count, offset into s_melWeights). */
static const uint16_t s_melStart[NUML_MEL_BINS] = {
@@MEL_START@@
};
static const uint16_t s_melCount[NUML_MEL_BINS] = {
@@MEL_COUNT@@
};
static const uint16_t s_melOffset[NUML_MEL_BINS] = {
@@MEL_OFFSET@@
};
static const float s_melWeights[NUML_MEL_NNZ] = {
@@MEL_WEIGHTS@@
};
/* Symmetric Hann (np.hanning) exported from the training frontend. */
static const float s_hannTable[NUML_FRAME_LENGTH] = {
@@HANN@@
};
"""

MEL_POWER = """\
static float NuML_MelPower(uint32_t melIndex)
{
    const uint32_t start = s_melStart[melIndex];
    const uint32_t count = s_melCount[melIndex];
    const float *weights = &s_melWeights[s_melOffset[melIndex]];
    float sum = 0.0f;
    for (uint32_t index = 0U; index < count; ++index)
    {
        sum += s_power[start + index] * weights[index];
    }
    return sum;
}
"""

BACKGROUND_GATE = """\
    const bool scorePass = average[best] >= NUML_TRIGGER_THRESHOLD;
#if NUML_HAS_BACKGROUND
    /* The background class means "nothing was said": it must never fire an action, and it is
     * the baseline the winner has to beat by NUML_TRIGGER_MARGIN. */
    const bool background = best == NUML_BACKGROUND_INDEX;
    const bool marginPass = average[best] >= average[NUML_BACKGROUND_INDEX] + NUML_TRIGGER_MARGIN;
    /* Upstream's re-arm rule, unchanged: the keyword has faded when its smoothed score drops
     * back below NUML_REARM_SCORE. */
    const bool released = average[best] < NUML_REARM_SCORE;
#else
    /* This project has no background/silence class (labels like ["yes", "no"]), so every label
     * may trigger and NUML_BACKGROUND_INDEX is a meaningless 0 fallback that must not be read.
     * The margin baseline becomes the runner-up score: with a baseline of 0 the margin test
     * would reduce to "average[best] >= NUML_TRIGGER_MARGIN", which NUML_TRIGGER_THRESHOLD
     * already subsumes, and the tunable would silently do nothing. */
    const bool background = false;
    float runnerUp = 0.0f;
    for (uint32_t label = 0U; label < NUML_LABEL_COUNT; ++label)
    {
        if ((label != best) && (average[label] > runnerUp)) { runnerUp = average[label]; }
    }
    const bool marginPass = average[best] >= runnerUp + NUML_TRIGGER_MARGIN;
    /* ...and the re-arm rule cannot be NUML_REARM_SCORE either. The scores are one softmax, so
     * with two labels the winner is ALWAYS >= 0.5, i.e. always above any sensible re-arm score:
     * the only thing that could ever re-arm the trigger would be the RMS gate, so the app would
     * fire once per silence rather than once per spoken word. Release the trigger as soon as
     * the conditions that fired it stop holding -- the keyword is no longer being said. */
    const bool released = !scorePass || !marginPass;
#endif
"""

README = """\
# audio_kws — live DMIC → Log-Mel → Ethos-U KWS firmware template

`main.cpp.in` is generated once by `scripts/vendor_kws_template.py` from NuML App Builder
v0.1.8 `tools/audio_kws_runtime.py::_render_main`, then filled per deploy by
`tm_local/mcu/kws_codegen.py::render_kws_main()` (35 `@@TOKEN@@` placeholders) and staged by
`tm_local/mcu/apps/kws.py`. Nothing here is edited by hand: re-run the vendoring instead.

What differs from upstream, and why:

* The mel filterbank and the Hann window are **not** computed on the board. Both are exported
  from `tm_local/audio_frontend.py` — the same objects training, Preview and the INT8
  calibration use — as `s_melStart/s_melCount/s_melOffset/s_melWeights` (row-normalised, so
  there is no inverse-sum normaliser) and `s_hannTable`. Upstream's own `_mel_filters()` uses
  different bin edges; keeping it would give the board a plausible but systematically wrong
  front-end.
* `NUML_BACKGROUND_INDEX` is only read inside `#if NUML_HAS_BACKGROUND`. A project without a
  background/silence class (`["yes", "no"]`) sets the flag to 0; there every label may trigger
  and the trigger-margin baseline is the runner-up score instead of the background score.
* **Re-arming is per arm.** After a detection the trigger stays disarmed until `released` goes
  true. With a background class that is upstream's rule (the smoothed winner falls back below
  `NUML_REARM_SCORE`). Without one it cannot be: the scores are one softmax, so with two labels
  the winner never drops below 0.5 and only the RMS gate could ever re-arm — the firmware would
  fire once per silence instead of once per word. The no-background arm therefore re-arms as
  soon as the score or margin test stops holding.
* The USB-CDC build waits (bounded, `NUML_BOARD_LOG_WAIT_MS`, default 10 s) for a terminal to
  open the port before the banner prints, so the student does not lose the first lines while
  Windows enumerates. The UART build compiles that away to `((void)0)`.
* The smoothing window, trigger margin, re-arm score and RMS gate are tokens, not constants,
  so the project's `kws_runtime` settings actually reach the firmware.
* `printf()` goes over USB CDC when `NUML_USE_USB_CDC` is defined (the LCD-less
  NuGestureAI-M55M1 board); `tm_local/mcu/apps/kws.py` stages
  `mcu_toolkit/apps/common/cdc/cdc_only/` and adds that define.

The DMIC pin macros and channel mask come from `mcu_toolkit/boards.json` via
`boards.AudioDmic`, never from this file.
"""


def _fail(message: str) -> None:
    raise SystemExit(f"vendor_kws_template: {message}")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    found = text.count(old)
    if found != 1:
        _fail(f"KWS template anchor {label!r} found {found} times, expected 1")
    return text.replace(old, new)


def _sub_exact(
    text: str, pattern: str, new: str, label: str, *, flags: int = 0, count: int = 1
) -> str:
    """Replace every match of `pattern` with the literal `new`, insisting on `count` matches.

    Literal, not `re.sub()`: the replacements carry `\\`-free but group-reference-looking text
    and a silent backreference expansion would corrupt the firmware rather than fail loudly.
    """
    matches = list(re.finditer(pattern, text, flags))
    if len(matches) != count:
        _fail(f"KWS template anchor {label!r} matched {len(matches)} times, expected {count}")
    parts: list[str] = []
    last = 0
    for match in matches:
        parts.append(text[last : match.start()])
        parts.append(new)
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def _define_line(name: str, value: str) -> str:
    """Upstream aligns every `NUML_*` macro body at column 32; keep the file diffable."""
    return f"#define {name:<22} ({value})"


def import_audio_kws_runtime(app_builder_root: Path) -> ModuleType:
    """Import the App Builder runtime module without writing into its (read-only) tree."""
    module_path = app_builder_root / "tools" / "audio_kws_runtime.py"
    if not module_path.is_file():
        _fail(f"App Builder Audio KWS runtime not found at {module_path}")
    if str(app_builder_root) not in sys.path:
        sys.path.insert(0, str(app_builder_root))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from tools import audio_kws_runtime  # App Builder module (read-only use)
    finally:
        sys.dont_write_bytecode = previous
    marker = getattr(audio_kws_runtime, "AUDIO_KWS_RUNTIME_MARKER", "")
    if marker != EXPECTED_RUNTIME_MARKER:
        _fail(
            f"App Builder runtime marker is {marker!r}, expected {EXPECTED_RUNTIME_MARKER!r}; "
            f"this template claims App Builder v{APP_BUILDER_VERSION} -- re-check every anchor "
            "in this script before bumping it"
        )
    return audio_kws_runtime


def code_lines(text: str) -> list[str]:
    """`text` line by line with `/* */` and `//` comment bodies blanked, line count preserved.

    The gate assertion below is a statement about code: the template's own comments have to
    name `NUML_BACKGROUND_INDEX` in order to explain why it is gated, and matching those would
    make the check unusable. Blanking rather than dropping keeps the reported line numbers
    pointing at the real file.
    """
    out: list[str] = []
    in_block = False
    for line in text.split("\n"):
        kept: list[str] = []
        index = 0
        while index < len(line):
            if in_block:
                end = line.find("*/", index)
                if end < 0:
                    break
                in_block = False
                index = end + 2
                continue
            start_block = line.find("/*", index)
            start_line = line.find("//", index)
            if start_line >= 0 and (start_block < 0 or start_line < start_block):
                kept.append(line[index:start_line])
                break
            if start_block >= 0:
                kept.append(line[index:start_block])
                in_block = True
                index = start_block + 2
                continue
            kept.append(line[index:])
            break
        out.append("".join(kept))
    return out


def assert_background_gated(text: str) -> None:
    """Every read of `NUML_BACKGROUND_INDEX` must sit inside `#if NUML_HAS_BACKGROUND`.

    Task 5 can emit the `HAS_BACKGROUND` flag but cannot enforce that the firmware honours it;
    this is where that is enforced. An ungated read makes a no-background project's first class
    untriggerable and its own margin baseline -- firmware that compiles, boots and never fires.
    """
    lines = code_lines(text)
    depth = 0
    opened_at = -1
    gate_blocks = 0
    has_else = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#if"):
            depth += 1
            if opened_at < 0 and stripped.startswith("#if NUML_HAS_BACKGROUND"):
                opened_at = depth
                gate_blocks += 1
            continue
        if stripped.startswith("#endif"):
            if opened_at == depth:
                opened_at = -1
            depth -= 1
            continue
        if stripped.startswith("#else") and opened_at == depth:
            opened_at = -1
            has_else = True
            continue
        read_outside = (
            "NUML_BACKGROUND_INDEX" in line
            and not line.startswith("#define NUML_BACKGROUND_INDEX")
            and opened_at < 0
        )
        if read_outside:
            _fail(
                f"NUML_BACKGROUND_INDEX is read outside #if NUML_HAS_BACKGROUND at line "
                f"{number}: {line.strip()!r}"
            )
    if gate_blocks != 1:
        _fail(f"expected exactly one #if NUML_HAS_BACKGROUND block, found {gate_blocks}")
    if not has_else:
        _fail("the #if NUML_HAS_BACKGROUND block has no #else branch for no-background projects")


def build_main_template(runtime: ModuleType) -> str:
    """Render upstream once with probe values and rewrite it into the token template."""
    text = runtime._render_main(dict(PROBE_CONFIG), PROBE_ARENA, dict(PROBE_DMIC))

    # 1. Provenance marker (upstream's own marker must not survive: this file is no longer it).
    text = _replace_once(text, f"/* {EXPECTED_RUNTIME_MARKER} */", HEADER, "runtime marker")

    # 2. Every baked-in constant becomes a token, one whole `#define` line at a time.
    text = _sub_exact(
        text,
        r"^#define ACTIVATION_BUF_SZ\s+\(.*\)$",
        "#define ACTIVATION_BUF_SZ (@@ARENA@@)",
        "ACTIVATION_BUF_SZ",
        flags=re.MULTILINE,
    )
    for name, value in DEFINE_TOKENS:
        text = _sub_exact(
            text,
            rf"^#define {re.escape(name)}\s+\(.*\)$",
            _define_line(name, value),
            f"#define {name}",
            flags=re.MULTILINE,
        )
    # MEL_NNZ has no upstream counterpart: the sparse weight table's length varies with the
    # mel-bin count (454 at 40 bins, 437 at 64), so it can never be derived from NUML_MEL_BINS.
    mel_bins_line = _define_line("NUML_MEL_BINS", "@@MEL_BINS@@U")
    text = _replace_once(
        text,
        mel_bins_line + "\n",
        mel_bins_line + "\n" + _define_line("NUML_MEL_NNZ", "@@MEL_NNZ@@U") + "\n",
        "NUML_MEL_NNZ insertion point",
    )
    background_line = _define_line("NUML_BACKGROUND_INDEX", "@@BACKGROUND_INDEX@@U")
    text = _replace_once(
        text,
        background_line + "\n",
        background_line
        + "\n"
        + "/* 1 when one of this project's classes really is a background/silence class; 0 when\n"
        + " * none is, and then NUML_BACKGROUND_INDEX is a meaningless fallback that must not be\n"
        + " * read -- see the #if NUML_HAS_BACKGROUND block in NuML_ProcessKwsScores(). */\n"
        + _define_line("NUML_HAS_BACKGROUND", "@@HAS_BACKGROUND@@")
        + "\n",
        "NUML_HAS_BACKGROUND insertion point",
    )

    # 2b. The model-data declaration must match the definition `tm_local/mcu/codegen.py`
    #     generates, which returns `const uint8_t *` (the table lives in flash). Upstream
    #     declares it without the const; two declarations of one function with different
    #     return types is an ODR violation no compiler has to diagnose. `Model::Init()` takes
    #     `const uint8_t* nnModelAddr`, so the call site below needs no cast.
    text = _replace_once(
        text,
        "extern uint8_t *GetModelPointer();",
        "extern const uint8_t *GetModelPointer();",
        "GetModelPointer const",
    )

    # 3. printf() transport: USB CDC on an LCD-less board, a no-op otherwise.
    text = _replace_once(
        text, '#include "log_macros.h"\n', '#include "log_macros.h"\n' + CDC_BLOCK, "log_macros"
    )

    # 4. Labels and the front-end tables become initialiser bodies the deploy step fills in.
    text = _sub_exact(
        text,
        r"static const char \*const s_labels\[NUML_LABEL_COUNT\] = \{.*?\n\};",
        "static const char *const s_labels[NUML_LABEL_COUNT] = {\n@@LABELS@@\n};",
        "s_labels",
        flags=re.DOTALL,
    )
    text = _sub_exact(
        text,
        r"static const uint16_t s_melLeft\[NUML_MEL_BINS\] = \{.*?"
        r"static const float s_melInverseSum\[NUML_MEL_BINS\] = \{.*?\n\};\n",
        MEL_TABLES,
        "upstream mel tables",
        flags=re.DOTALL,
    )

    # 5. The runtime-computed Hann window goes away entirely; the exported table replaces it.
    text = _replace_once(
        text, "static float s_hann[NUML_FRAME_LENGTH] NUML_AUDIO_BUFFER_ATTRIBUTE;\n", "", "s_hann"
    )
    text = _replace_once(
        text,
        "    const float phaseScale = 2.0f * 3.14159265358979323846f / "
        "(float)(NUML_FRAME_LENGTH - 1U);\n"
        "    for (uint32_t index = 0U; index < NUML_FRAME_LENGTH; ++index)\n"
        "    {\n"
        "        s_hann[index] = 0.5f - 0.5f * arm_cos_f32(phaseScale * (float)index);\n"
        "    }\n",
        "",
        "arm_cos_f32 window",
    )
    text = _replace_once(
        text,
        "            s_fftInput[index] = pcm * s_hann[index];",
        "            s_fftInput[index] = pcm * s_hannTable[index];",
        "window application",
    )

    # 5b. Upstream's int8 saturation clamp does not compile under arm-none-eabi-g++: there
    #     `int32_t` is `long int`, so `std::max(-128, quantized)` has to deduce `const _Tp&`
    #     from both `int` and `long int` and fails. The App Builder only ever builds this file
    #     with armclang, where `int32_t` is plain `int`. Casting the literals keeps the
    #     arithmetic identical and makes the deduction unambiguous on both compilers.
    text = _replace_once(
        text,
        "        quantized = std::min(127, std::max(-128, quantized));",
        "        quantized = std::min((int32_t)127, std::max((int32_t)-128, quantized));",
        "int8 saturation clamp",
    )

    # 6. Mel accumulation now walks the sparse, already-normalised Studio tables.
    text = _sub_exact(
        text,
        r"static float NuML_MelPower\(uint32_t melIndex\)\n\{.*?\n\}\n",
        MEL_POWER,
        "NuML_MelPower",
        flags=re.DOTALL,
    )

    # 7. DMIC wiring comes from boards.json, not from the App Builder's default board.
    text = _replace_once(
        text, "    SET_DMIC0_CLK_PB4();", "    @@DMIC_CLK_MACRO@@();", "DMIC clock macro"
    )
    text = _replace_once(
        text, "    SET_DMIC0_DAT_PB5();", "    @@DMIC_DAT_MACRO@@();", "DMIC data macro"
    )
    text = _sub_exact(
        text, r"DMIC_CTL_CHEN0_Msk", "@@DMIC_CHANNEL_MASK@@", "DMIC channel mask", count=2
    )

    # 8. The background class is optional in the Studio; upstream assumed it always exists.
    text = _replace_once(
        text,
        "static uint32_t s_candidateIndex = NUML_BACKGROUND_INDEX;\n",
        "/* Any valid label index: s_candidateHits starts at 0, so the initial value is inert.\n"
        " * It deliberately does NOT use NUML_BACKGROUND_INDEX, which is meaningless when the\n"
        " * project has no background class. */\n"
        "static uint32_t s_candidateIndex = 0U;\n",
        "s_candidateIndex",
    )
    # `scorePass` moves INTO this replacement (it is computed the same way in both arms, and
    # the no-background arm's re-arm rule is written in terms of it).
    text = _replace_once(
        text,
        "    const bool background = best == NUML_BACKGROUND_INDEX;\n"
        "    const bool marginPass = average[best] >= average[NUML_BACKGROUND_INDEX] + "
        "NUML_TRIGGER_MARGIN;\n"
        "    const bool scorePass = average[best] >= NUML_TRIGGER_THRESHOLD;\n",
        BACKGROUND_GATE,
        "background / margin decision",
    )
    # The re-arm test now asks each arm's own `released`. In the background arm this is
    # upstream's `average[best] < NUML_REARM_SCORE` verbatim; in the no-background arm the
    # whole condition collapses to `quiet || released`, because `background` is a compile-time
    # false there. See BACKGROUND_GATE for why the score-based rule cannot work without a
    # background class.
    text = _replace_once(
        text,
        "        if (quiet || background || average[best] < NUML_REARM_SCORE)\n",
        "        if (quiet || background || released)\n",
        "re-arm condition",
    )

    # 9. Pump the CDC endpoint while idling and after every scored hop, so the log keeps up.
    text = _replace_once(
        text,
        "            __WFI();\n            continue;",
        "            BOARD_LOG_PUMP();\n            __WFI();\n            continue;",
        "idle pump",
    )
    text = _replace_once(
        text,
        "        NuML_ProcessKwsScores(scores, inferenceHop, rmsDbfs);\n",
        "        NuML_ProcessKwsScores(scores, inferenceHop, rmsDbfs);\n"
        "        BOARD_LOG_PUMP();\n",
        "score pump",
    )
    text = _replace_once(
        text,
        "        return 1;\n    }\n",
        "        return 1;\n    }\n\n    BOARD_LOG_INIT();\n    BOARD_LOG_WAIT();\n",
        "BoardInit guard",
    )

    _check(text)
    return text


def _check(text: str) -> None:
    for forbidden in FORBIDDEN:
        if forbidden in text:
            _fail(f"{forbidden!r} still present after transformation")
    for required in REQUIRED:
        if required not in text:
            _fail(f"{required!r} missing after transformation")
    survivors = set(re.findall(r"@@([A-Z0-9_]+)@@", text))
    if survivors != set(DEPLOY_TOKENS):
        missing = sorted(set(DEPLOY_TOKENS) - survivors)
        extra = sorted(survivors - set(DEPLOY_TOKENS))
        _fail(f"token mismatch after transformation: missing={missing} unexpected={extra}")
    assert_background_gated(text)
    leak = PRIVATE_PATH_RE.search(text)
    if leak is not None:
        _fail(f"generated template leaks a local path: {leak.group(0)!r}")


def vendor_kws_template(app_builder_root: Path, dest_root: Path) -> None:
    runtime = import_audio_kws_runtime(app_builder_root)
    text = build_main_template(runtime)
    out = dest_root / "apps" / "audio_kws"
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.cpp.in").write_text(text, encoding="utf-8", newline="\n")
    (out / "README.md").write_text(README, encoding="utf-8", newline="\n")
    print(f"[OK] audio_kws: wrote {out / 'main.cpp.in'} ({len(DEPLOY_TOKENS)} tokens)")
