# Teachable Machine Local Studio v2.1.0 — Windows Only

A local browser-based Image, Audio, and Abnormal Sound training studio for Windows. The Abnormal Sound workflow uses a pinned frozen YAMNet embedding encoder with a normal-only scorer; it reports deviation from the user's normal baseline rather than naming an anomaly. The application supports webcam/microphone sample collection, native TensorFlow training, live PC preview, and Float32, Dynamic Range, strict INT8, and strict UINT8 TensorFlow Lite export.

This release intentionally uses one native Windows Python 3.13 `.venv` and the TensorFlow 2.21 Windows CPU runtime. It does not install or use WSL, Linux, CUDA, or a second Python environment.

Student entry points:

```text
01_INSTALL.bat
02_START.bat
```

Project data stays under `workspace`. Diagnostic information can be downloaded from the top-right **Download Log** action; images and audio are not included in that diagnostic file.

This is an independent local implementation and is not an official Google product. See `NOTICE.md` and `LICENSE.txt`.

See `README_zh-TW.md` for the user guide and `docs/ABNORMAL_SOUND_YAMNET_zh-TW.md` for the YAMNet data, calibration, preview, and export contract.
