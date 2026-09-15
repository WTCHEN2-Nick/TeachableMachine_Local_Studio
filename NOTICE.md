# Notice

Teachable Machine Local Studio is an independent local implementation created for education and local model deployment. It is not sponsored, endorsed, maintained by, or affiliated with Google.

“Teachable Machine”, “TensorFlow”, “Google”, and related marks belong to their respective owners. The descriptive project title is used to explain compatibility and the familiar workflow. This package does not include Google logos, proprietary artwork, Google account integration, cloud storage, telemetry, or the official hosted application.

Portions of the design approach were informed by the Apache-2.0 licensed `googlecreativelab/teachablemachine-community` source. Modified and newly written files in this distribution carry this notice and are distributed under Apache-2.0.

The optional Abnormal Sound feature extractor reconstructs the Apache-2.0 licensed YAMNet
MobileNetV1 topology and frontend constants from `tensorflow/models`, and uses the official
YAMNet HDF5 weights downloaded from Google Storage. Local Studio verifies the pinned weight
SHA-256 before use. TensorFlow and YAMNet remain projects of their respective authors; this
distribution is not an official Google product.

## MCU deployment components

The firmware build under `mcu_toolkit/` redistributes third-party sources and one prebuilt
tool. Full license texts ship alongside them: `mcu_toolkit/NuML_TFLM_Tool/LICENSE`,
`mcu_toolkit/LICENSES/` and each component's own file inside the vendored tree. The
per-component summary, including the exact vendored versions, is
`mcu_toolkit/NOTICE_third_party.md`; `mcu_toolkit/manifest.json` records the source
revisions and a SHA-256 for every vendored file.

- **NuML_Toolkit** (commit `3126344`) — Copyright Nuvoton Technology Corp., Apache License
  2.0. Only the GCC-relevant subset is redistributed. Modified files carry a "Modified by
  Teachable Machine Local Studio" header, and `mcu_toolkit/README_zh-TW.md` lists the
  modifications, as Apache-2.0 requires. `NuML_TFLM_Tool/tflite/` is TensorFlow's official
  FlatBuffers-generated Python schema bindings, version 2.10.0, Copyright The TensorFlow
  Authors, Apache License 2.0 — unmodified upstream code, imported by the codegen scripts
  next to it.
- **M55M1 Series BSP V3.01.005** — Copyright 2023 Nuvoton Technology Corp., Apache License
  2.0; the upstream `LICENSE` and `NOTICE` files are kept inside the vendored BSP.
  `Library/StdDriver/*/gdma/dma350_*` are Arm Limited, BSD-3-Clause.
- **CMSIS 6.3.0, CMSIS-DSP 1.16.2, CMSIS-NN 6.0.0** — Copyright Arm Limited and affiliates,
  Apache License 2.0; the CMSIS-DSP ComputeLibrary files are MIT.
- **TensorFlow Lite for Microcontrollers** — Copyright The TensorFlow Authors, Apache
  License 2.0, including the bundled FlatBuffers and gemmlowp (Google LLC, Apache-2.0).
- **Arm ML Embedded Evaluation Kit** — Copyright (c) 2021-2022 Arm Limited and affiliates,
  Apache License 2.0.
- **FatFs R0.16** — Copyright (C) 2025 ChaN, all rights reserved; redistributed under the
  BSD-style notice in its source headers, reproduced in `mcu_toolkit/LICENSES/FatFs.txt`.
- **OpenMV imlib** — The MIT License, Copyright (c) 2013-2021 Ibrahim Abdelkader and
  Kwabena W. Agyeman; `omv/imlib/lodepng.h` is zlib-licensed, Copyright 2005-2022 Lode
  Vandevenne. Only headers are vendored as source, with one deliberate exception described
  below. The prebuilt archive `omv/Lib/libomv.a`, which Image firmware links (`-lomv`) and
  Known Sound and KWS firmware do not, contains the member `agast.o`, built upstream from
  AGAST corner-detection code licensed GNU GPL v3-or-later, Copyright (C) 2010 Elmar Mair.
- **AGAST corner detection, GNU GPL v3-or-later** — Copyright (C) 2010 Elmar Mair. Because
  this project distributes `agast.o` inside the archive above, it distributes the source it
  was built from alongside it:
  `mcu_toolkit/NuML_TFLM_Tool/templates/M55M1BSP/ThirdParty/openmv/omv/imlib/agast.c`,
  byte-for-byte from upstream, with the full license text in
  `mcu_toolkit/LICENSES/GPL-3.0.txt`. That file is shipped but never compiled: no build
  record names it, and the GCC driver compiles an explicit source list rather than walking
  directories. Treat this as a good-faith source offer rather than a complete Corresponding
  Source package — the recipe Nuvoton used to build `libomv.a` lives upstream and is not
  vendored here. Before redistributing this toolkit outside your organisation, either obtain
  that build recipe, or remove `agast.o` from the archive and re-verify the Image firmware
  builds for both boards.
- **Arm Vela compiler 5.1.0** (`mcu_toolkit/vela/vela-5_1_0.exe`) — Copyright 2020-2024 Arm
  Limited and/or its affiliates, Apache License 2.0. The executable is a PyInstaller bundle
  containing CPython 3.10 (PSF-2.0), NumPy (BSD-3-Clause), FlatBuffers (Apache-2.0) and the
  PyInstaller bootloader (GPL-2.0 with the bootloader exception).
- **NuML App Builder v0.1.8 sources** under `mcu_toolkit/apps/**` — Copyright Nuvoton
  Technology Corp.; that distribution carries no license file of its own, so these sources
  are redistributed under the same Apache License 2.0 terms as NuML_Toolkit.

**Not included in this distribution:** the **Arm GNU Toolchain** (GPL-3.0 with the GCC
runtime library exception) is downloaded and installed by the user, never bundled; the
**Nuvoton Nu-Link Command Tool** (proprietary) is only detected if the user installed it;
**Keil MDK** is not required and no Keil project is produced; **pyocd** (Apache License 2.0)
is installed by the user, on request, into this project's own virtual environment by
`scripts/setup_voiceai_flash.py` (needed only for the NuMaker-VoiceAI-M55M1 board), and is
never bundled; the **Nuvoton NuMicroM55 DFP** CMSIS-Pack that same script fetches from
Nuvoton's own upstream, verifying its SHA-256 before use, is likewise not redistributed by
this project.
