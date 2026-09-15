"""The vendored known_sound C frontend must reproduce the TensorFlow log-mel patch.

The firmware and this test compile the *same* translation units in
``mcu_toolkit/apps/known_sound/frontend/``; only the compiler differs. That is what makes a
host-side check meaningful: a drift caught here is a drift the board would have had, without a
flash cycle per iteration.

Acceptance is stated in units the int8 model can represent. The float comparisons (Hann window,
mel matrix, |STFT|, log-mel) exist to guarantee the one that matters -- the quantised 96x64 patch
must land within one LSB of the reference -- rather than leaving it to luck.

Building needs MSVC (Visual Studio Build Tools). Where it is absent the module skips; it never
fails, because a student machine is not required to carry a C compiler.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from tm_local.mcu import paths

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
HOST = paths.APPS_ROOT / "known_sound" / "host_tests"
GOLDEN_DIR = HOST / "golden"
FIXTURES = ROOT / "tests" / "fixtures" / "mcu"
#: Git-ignored (``workspace/tmp/*``): the DLL and its stamp must never land inside mcu_toolkit/,
#: whose file set is asserted byte-for-byte against manifest.json by tests/test_mcu_vendoring.py.
BUILD_DIR = ROOT / "workspace" / "tmp" / "mcu_host"

pytestmark = pytest.mark.skipif(
    not (HOST / "harness.py").is_file(), reason="known_sound host tests not vendored"
)


def _harness():
    """Import the vendored harness without leaving a __pycache__ inside mcu_toolkit/.

    That tree is hashed file-for-file by manifest.json; a .pyc dropped there by this import is
    exempted by the manifest walk, but not writing one at all is cheaper than exempting it.

    The two vendored test modules beside it are deliberately NOT imported: they resolve
    ``ROOT/golden`` and ``ROOT/model`` against the App Builder's own layout, where ``golden/``
    sits next to ``tests/`` and a private ``model/`` export exists. Their assertions are ported
    into this file instead, reading from ``host_tests/golden/`` and the Studio's INT8 fixture.
    """
    if str(HOST) not in sys.path:
        sys.path.insert(0, str(HOST))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import harness  # type: ignore[import-not-found]  # vendored, not a package
    finally:
        sys.dont_write_bytecode = previous
    return harness


@pytest.fixture(scope="module")
def lib() -> ctypes.CDLL:
    harness = _harness()
    if harness.find_vcvars() is None:
        pytest.skip("MSVC Build Tools not installed (no TM_VCVARS64, vswhere or default vcvars64)")
    return harness.frontend(BUILD_DIR)


@pytest.fixture(scope="module")
def golden() -> tuple[object, dict]:
    if not (GOLDEN_DIR / "golden.npz").is_file():
        pytest.skip("known_sound golden vectors not vendored")
    data = np.load(GOLDEN_DIR / "golden.npz")
    meta = json.loads((GOLDEN_DIR / "golden_meta.json").read_text(encoding="utf-8"))
    return data, meta


def _floats(count: int):
    return (ctypes.c_float * count)()


def _as_array(buffer, shape=None):
    values = np.ctypeslib.as_array(buffer).astype(np.float32, copy=True)
    return values.reshape(shape) if shape else values


def _input_scale(meta: dict) -> float:
    """The int8 input scale: never hard-coded, because every re-export changes it."""
    scale = (meta.get("input") or {}).get("scale")
    if scale:
        return float(scale)
    report = json.loads((FIXTURES / "known_sound3_int8.report.json").read_text(encoding="utf-8"))
    return float(report["inputs"][0]["scale"])


def _cases(data) -> list[str]:
    return sorted(key.split("/", 1)[1] for key in data.files if key.startswith("pcm/"))


def test_hann_and_mel_matrix_bit_exact(lib: ctypes.CDLL, golden) -> None:
    """The mel filterbank is exported, so equality is the bar; the window is computed, so it is not.

    ``NuML_Yamnet_MelMatrix`` decompresses a table lifted straight out of TensorFlow. Nothing is
    recomputed, so nothing may differ -- which is what removes the four documented ways a
    hand-rolled filterbank goes wrong (HTK versus Slaney, linear-in-mel edges, area
    normalisation, the zeroed DC row) in one assertion.

    ``NuML_Yamnet_Window`` evaluates ``0.5f - 0.5f*cosf(2*pi*i/400)`` in single precision, so it
    lands one float32 ULP (measured: 5.96e-8 on 40 of the 400 taps) from TensorFlow's own
    double-precision construction. Upstream's bound of 1e-6 is kept, and it is still three orders
    of magnitude tighter than the error this check exists to catch: a *symmetric* Hann -- the
    classic off-by-one -- ends at exactly 0 where the periodic one ends at 6.16e-5, asserted
    directly below so the denominator cannot be 399.
    """
    data, _meta = golden

    window = _floats(400)
    lib.NuML_Yamnet_Window(window)
    taps = _as_array(window)
    assert np.max(np.abs(taps - data["hann_window"].astype(np.float32))) <= 1e-6
    assert taps[-1] > 1e-5, "periodic Hann: a symmetric window would end at exactly 0"

    matrix = _floats(257 * 64)
    lib.NuML_Yamnet_MelMatrix(matrix)
    assert np.array_equal(_as_array(matrix, (257, 64)), data["mel_matrix"].astype(np.float32))


def test_golden_cases(lib: ctypes.CDLL, golden) -> None:
    """Magnitude, ln(mel + 0.001) and the 96x64 patch, case by case.

    The float log-mel budget is half an input LSB, which is exactly what guarantees the
    quantised patch cannot drift by more than one step. A tighter float bound is not reachable:
    a pure tone puts one bin at 50 while its neighbours sit near 1e-7, a dynamic range of 1e9
    against float32's ~1e7, so there both FFTs report their own rounding noise.
    """
    data, meta = golden
    scale = _input_scale(meta)
    half_lsb = scale / 2.0
    cases = _cases(data)
    assert cases

    for case in cases:
        pcm = data[f"pcm/{case}"].astype(np.float32)
        pcm_buf = (ctypes.c_float * pcm.size)(*pcm.tolist())

        magnitude = _floats(98 * 257)
        lib.NuML_Yamnet_Magnitude(pcm_buf, magnitude)
        assert (
            np.max(np.abs(_as_array(magnitude, (98, 257)) - data[f"magnitude/{case}"])) <= 1e-4
        ), case

        logmel = _floats(98 * 64)
        lib.NuML_Yamnet_LogMel(pcm_buf, logmel)
        assert (
            np.max(np.abs(_as_array(logmel, (98, 64)) - data[f"log_mel/{case}"])) <= half_lsb
        ), case

        patch = _floats(96 * 64)
        lib.NuML_Yamnet_Patch(pcm_buf, patch)
        q_c = np.rint(_as_array(patch, (96, 64)).astype(np.float64) / scale)
        q_ref = np.rint(data[f"patch/{case}"].astype(np.float64) / scale)
        assert np.max(np.abs(q_c - q_ref)) <= 1, case


def test_threshold_code_and_scores(lib: ctypes.CDLL) -> None:
    """The output stage: independent per-class sigmoid scores, and an integer threshold.

    ``known_sound`` is multi-label -- the scores do not sum to one and nothing here may take an
    argmax. The hot path compares int8 codes instead of dequantising every class every hop, which
    is only legitimate if the code is the *ceiling*: rounding fires one step early whenever
    threshold/scale lands between two integers.
    """
    lib.NuML_KnownSound_ThresholdCode.restype = ctypes.c_int
    lib.NuML_KnownSound_ThresholdCode.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int]
    scale, zp = 0.00390625, -128

    assert lib.NuML_KnownSound_ThresholdCode(0.5, scale, zp) == math.ceil(0.5 / scale) + zp

    codes = (ctypes.c_int8 * 3)(-128, 0, 127)
    out = _floats(3)
    lib.NuML_KnownSound_Scores(codes, 3, ctypes.c_float(scale), ctypes.c_int(zp), out)
    assert np.allclose(_as_array(out), [0.0, 0.5, 255 * scale])


def test_scores_match_the_recorded_model_output(lib: ctypes.CDLL) -> None:
    """The same dequantisation against what the exported model actually produced.

    ``known_sound`` is multi-label, so this also states the property a future "fix" would break:
    the per-class scores are independent sigmoids and at least one case sums to well over 1. A
    softmax or a normalise-to-100% would have to change these reference numbers to pass.
    """
    scores_npz = GOLDEN_DIR / "scores.npz"
    if not scores_npz.is_file():
        pytest.skip("known_sound score vectors not vendored")
    data = np.load(scores_npz)
    meta = json.loads((GOLDEN_DIR / "scores_meta.json").read_text(encoding="utf-8"))
    scale = float(meta["output"]["scale"])
    zero_point = int(meta["output"]["zero_point"])
    classes = len(meta["labels"])
    cases = tuple(meta["cases"])
    assert cases

    totals = []
    for case in cases:
        raw = data[f"output_int8/{case}"]
        assert raw.shape == (classes,), case
        codes = (ctypes.c_int8 * classes)(*(int(value) for value in raw))
        out = _floats(classes)
        lib.NuML_KnownSound_Scores(
            codes, ctypes.c_int(classes), ctypes.c_float(scale), ctypes.c_int(zero_point), out
        )
        actual = _as_array(out)
        expected = data[f"scores/{case}"]
        assert np.max(np.abs(actual - expected)) <= 1e-7, case
        assert np.all(actual >= 0.0) and np.all(actual <= 1.0), case
        totals.append(float(actual.sum()))

    assert any(total > 1.01 for total in totals), "scores are independent, not a distribution"


def test_find_vcvars_prefers_the_env_override(tmp_path: Path, monkeypatch) -> None:
    """TM_VCVARS64 wins over discovery, so a machine with several toolsets can pin one."""
    harness = _harness()
    stub = tmp_path / "vcvars64.bat"
    stub.write_text("@echo off\n", encoding="ascii")
    monkeypatch.setenv("TM_VCVARS64", str(stub))

    assert harness.find_vcvars() == stub


def test_find_vcvars_returns_none_without_msvc(tmp_path: Path, monkeypatch) -> None:
    """No override, no vswhere, no fixed path -> None, which is what makes the module skip.

    This is the path every machine without the Build Tools takes, including the student laptops
    this Studio actually ships to, so it is asserted rather than assumed. Note what it must NOT
    do: raise. `_compile` still raises, but only once a caller has ignored the None.
    """
    harness = _harness()
    monkeypatch.delenv("TM_VCVARS64", raising=False)
    empty = tmp_path / "no-visual-studio"
    empty.mkdir()
    monkeypatch.setenv("ProgramFiles(x86)", str(empty))
    monkeypatch.setenv("ProgramFiles", str(empty))
    monkeypatch.setattr(harness, "_VCVARS", tmp_path / "absent" / "vcvars64.bat")

    assert harness.find_vcvars() is None


def test_stamp_is_source_content_not_mtime(tmp_path: Path, monkeypatch) -> None:
    """Rebuild on sha256 of the C sources: a fresh checkout rewrites mtimes but changes nothing.

    The upstream harness stamped ``st_mtime_ns``, which re-runs the whole MSVC build after every
    clone, re-vendor or branch switch -- and, worse, would skip a rebuild for an edit that
    happened to preserve the timestamp.
    """
    harness = _harness()
    monkeypatch.setattr(harness, "FRONTEND", tmp_path)
    (tmp_path / "a.c").write_bytes(b"int a(void){return 1;}\n")
    (tmp_path / "a.h").write_bytes(b"int a(void);\n")
    (tmp_path / "notes.txt").write_bytes(b"ignored\n")

    stamp = harness._stamp()
    assert [name for name, _ in stamp] == ["a.c", "a.h"]
    assert stamp == tuple(
        (name, hashlib.sha256((tmp_path / name).read_bytes()).hexdigest())
        for name in ("a.c", "a.h")
    )

    before = stamp
    (tmp_path / "a.c").write_bytes(b"int a(void){return 2;}\n")
    assert harness._stamp() != before


def test_build_writes_only_into_workspace_tmp_and_no_batch_file(lib: ctypes.CDLL) -> None:
    """Two guardrails the host build must not break, asserted right after it has run.

    1. Nothing lands in ``mcu_toolkit/``. tests/test_mcu_vendoring.py asserts the on-disk file
       set there equals the manifest keys, so a DLL or an .obj beside the sources would turn a
       passing frontend check into a failing vendoring check.
    2. No batch file is generated anywhere -- not even inside the git-ignored build directory.
       tests/test_distribution.py scans the *whole repository* and requires exactly two of them
       (01_INSTALL.bat, 02_START.bat), which is why this harness drives cl.exe directly from a
       captured vcvars environment instead of writing a build script.
    """
    assert not list(paths.MCU_TOOLKIT_ROOT.rglob("*.bat"))
    assert not (paths.APPS_ROOT / "known_sound" / "build_host").exists()
    assert (BUILD_DIR / "numl_frontend.dll").is_file()
    strays = [p.name for p in BUILD_DIR.rglob("*") if p.suffix.lower() in {".bat", ".cmd"}]
    assert strays == []
