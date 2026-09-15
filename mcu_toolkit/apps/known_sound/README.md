Known Sound device runtime vendored from NuML App Builder v0.1.8 (Apache-2.0) by
`scripts/vendor_mcu_toolkit.py known-sound`; never edit by hand.

frontend/: portable YAMNet 96x64 log-mel C (bit-checked against TensorFlow by host_tests/).
device/: DMIC + LPPDMA capture and per-board BoardConfig.h.
host_tests/: the App Builder's build harness plus the golden vectors the Studio's own
tests/test_known_sound_frontend_host.py compares against; it builds the frontend with
MSVC, so those checks are skipped where MSVC is absent. The App Builder's own test
modules and the source .wav recordings are not vendored: nothing here can run or read
them.
harness.py is the one host file this script patches (see HARNESS_PATCHES): the build goes
to workspace/tmp/mcu_host/ instead of into this tree, and find_vcvars() returns None rather
than raising when MSVC is absent. tests/test_known_sound_frontend_host.py drives it.
main.cpp.in: token template rendered at deploy time (per-class thresholds).

Tokens: @@ARENA@@ @@CLASS_COUNT@@ @@CLIP_SAMPLES@@ @@HOP_SAMPLES@@ @@HOLD_HOPS@@ @@LABELS@@ @@THRESHOLDS@@
Board is chosen by a compile-time define -- BOARD_NUMAKER_X_M55M1D or
BOARD_NUGESTUREAI_M55M1 -- which is what selects the DMIC pins and channel in
BoardConfig.h; on the GestureAI board BoardConfig.h also pulls in numl_cdc.h/numl_usbd.h,
so apps/common/cdc/cdc_only must be on the include path there.
