# Technical sources

- TensorFlow pip installation and platform support:
  https://www.tensorflow.org/install/pip

- TensorFlow native Windows build notes, including native Windows GPU support ending after TensorFlow 2.10:
  https://www.tensorflow.org/install/source_windows

- TensorFlow 2.21 Python 3.13 Windows wheel metadata:
  https://pypi.org/project/tensorflow/2.21.0/

- TensorFlow Lite post-training integer quantization:
  https://www.tensorflow.org/model_optimization/guide/quantization/post_training

- TensorFlow Lite converter API:
  https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter

The v2.1.0 distribution intentionally chooses the native Windows CPU path and does not provision a Linux runtime.

## M55M1 firmware deployment

- Arm GNU Toolchain downloads (arm-none-eabi; installed by the user, not bundled):
  https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads
- Pinned release used by `scripts/download_arm_toolchain.py` (verified against Arm's own
  `.sha256asc` checksum file): Arm GNU Toolchain 14.2.rel1,
  `arm-gnu-toolchain-14.2.rel1-mingw-w64-i686-arm-none-eabi.zip`
- Vendored firmware templates, BSP subset and the Arm Vela 5.1.0 compiler ship in
  `mcu_toolkit/`; component versions are recorded in `mcu_toolkit/manifest.json` and the
  license summary in `mcu_toolkit/NOTICE_third_party.md`.

## YAMNet abnormal-sound feature extractor

- TensorFlow Models YAMNet model, frontend specification, and Apache-2.0 source:
  https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- Pinned official HDF5 weights:
  https://storage.googleapis.com/audioset/yamnet.h5
- Weight SHA-256:
  `13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744`
- Official transfer-learning tutorial:
  https://www.tensorflow.org/tutorials/audio/transfer_learning_audio

Local Studio uses YAMNet only as a frozen 1024-D feature extractor. The normal-only scorer,
threshold calibration, RMS branch, export metadata, and user interface are original Local
Studio code. It does not use the official 521-class head to claim open-set anomaly detection.
