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
