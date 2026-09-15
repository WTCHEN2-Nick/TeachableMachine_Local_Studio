# Teachable Machine Local Studio

**v2.1.0 — Windows only.** A teaching tool that runs entirely on a local Windows PC: collect
samples with a webcam or microphone, train a model, preview it, export TensorFlow Lite, and build
firmware for a Nuvoton M55M1 dev board — without an account, and without a single photo or
recording leaving the machine.

一套完全在本機 Windows 電腦執行的教學式機器學習工具：收集樣本 → 訓練 → Preview → 匯出 TensorFlow Lite
→ 建成 Nuvoton M55M1 開發板韌體。不需要帳號，圖片與錄音都不會離開你的電腦。

```text
Create classes → Collect samples (webcam / microphone) → Train → Preview
  → Export Float32 / Dynamic / INT8 / UINT8 TensorFlow Lite
  → Build M55M1 firmware and flash it
```

![The Local Studio home page — four project cards, the language toggle, and recent local projects](docs/screenshots/01_home.png)

---

## 📖 Documentation / 文件

| | |
|---|---|
| **[使用手冊_zh-TW.md](使用手冊_zh-TW.md)** | 完整使用手冊（繁體中文）— 安裝、四種專案、每一個訓練參數、燒錄開發板、尚未驗證的事 |
| **[USER_GUIDE_en.md](USER_GUIDE_en.md)** | Full user guide (English) — installation, the four project kinds, every training setting, flashing a board, and what is not verified |
| [README_zh-TW.md](README_zh-TW.md) | 中文版總覽（技術重點與版本差異） |
| [docs/](docs/) | Architecture, MCU deployment, audio frontend spec, troubleshooting, release notes |

**New here?** Start with the user guide in your language. It assumes no machine-learning or
embedded background.

---

## 🚀 Quick start

1. Install **64-bit CPython 3.13.x** (not the free-threaded `3.13t` build), ticking
   *Add python.exe to PATH*.
2. Unzip or clone into a **short path** — `C:\TM_Local_Studio` is ideal. Long paths break the
   firmware build weeks later, not now.
3. Double-click **`01_INSTALL.bat`** — builds `.venv` and verifies the environment with a real
   train → export → strict-INT8 round trip.
4. Double-click **`02_START.bat`** — opens `http://127.0.0.1:8765`.

Those two BAT files are the only entry points. Everything else is internal.

If the environment ever breaks: close Studio, delete **only** `.venv`, and re-run
`01_INSTALL.bat`. **Never delete `workspace/`** — it holds every project, photo, recording and
trained model, and there is no backup.

---

## 🧩 The four project kinds

The kind is fixed when the project is created and cannot be changed afterwards.

| Kind | What it answers | Scores | Dev board |
|---|---|---|---|
| **Image** | Which of these things is it? | Sum to 100% | X board, GestureAI |
| **Audio** (keyword spotting) | Which short sound or keyword is it? | Sum to 100% | X board, GestureAI |
| **Known Sound** | Which named sounds are present? (several at once) | Independent per class, **do not** sum to 100% | X board, GestureAI, VoiceAI |
| **Abnormal Sound** | How far from normal is this? (never names the sound) | One deviation score | — none |

**Known Sound and Abnormal Sound are not interchangeable.** Known Sound *names* sounds and needs
every class collected equally plus a Background class; Abnormal Sound collects only Normal and
never names anything. Known Sound scores are called **confidence scores**, never "probabilities".

---

## 🔌 Dev-board deployment

Image, Audio and Known Sound projects can be built into Nuvoton M55M1 firmware straight from the
Export Model dialog's second tab.

| Board | Camera | Output | Flashing |
|---|---|---|---|
| NuMaker-M55M1 (X board) | Yes | Built-in LCD + serial | Nu-Link Command Tool |
| NuMaker-GestureAI-M55M1 | Yes | USB camera overlay + serial | USB mass storage (default) or Nu-Link |
| NuMaker-VoiceAI-M55M1 | No | Serial only | pyocd + external Nu-Link2 |

Firmware templates, the M55M1 BSP subset and the Vela NPU compiler all ship in
[`mcu_toolkit/`](mcu_toolkit/) — no separate Nuvoton toolkit download, and **no Keil required**.
The one thing you install yourself is the free Arm GNU toolchain:

```bat
.venv\Scripts\python.exe scripts\download_arm_toolchain.py
```

Details, memory limits and every error message: [docs/MCU_DEPLOY_zh-TW.md](docs/MCU_DEPLOY_zh-TW.md).

---

## 🏗️ Design decisions worth knowing

- **Windows CPU only, deliberately.** Even with an NVIDIA GPU present, training and export run on
  the Windows CPU runtime. No WSL, no CUDA, no second Python environment — so every classroom PC
  behaves identically and there is one `.venv` to manage.
- **Train and Export are separate jobs.** Training produces only `.keras`, so Preview unlocks
  immediately; TensorFlow Lite conversion runs later, in its own subprocess, and a failure there
  cannot damage the trained model.
- **Strict integer quantization is a hard requirement.** After INT8/UINT8 conversion the graph is
  inspected for integer I/O, leftover float tensors and Flex ops. If any check fails the export
  fails — a mixed-precision model is never handed over labelled INT8.
- **Any sample or setting change invalidates the trained model,** including the built firmware.
  There is no confirmation dialog and no undo.
- **MCU constants are generated, never hand-synced.** Labels, model arrays, arena size,
  quantization parameters and the mel/Hann tables are all derived at deploy time from the training
  and conversion reports, so the PC and the board cannot compute different spectrograms.
- **The diagnostic log contains metadata only** — never image or audio content.

---

## 📁 Repository layout

```text
01_INSTALL.bat / 02_START.bat   the only two student entry points
tm_local/                       FastAPI app, project store, training, export, MCU pipeline
web/                            browser UI (zh/en toggle)
mcu_toolkit/                    vendored firmware templates, M55M1 BSP subset, Vela compiler
reference/                      runnable reference scripts for all four project kinds
scripts/                        install, toolchain download, vendoring utilities
tests/                          pytest suite, incl. architecture guardrails
docs/                           documentation and screenshots
workspace/                      student projects, samples and models (git-ignored)
```

---

## 🛠️ Development

```bat
.venv\Scripts\python.exe -m pytest                   all tests
.venv\Scripts\python.exe -m pytest -m "not slow"     skip real training / encoder builds
.venv\Scripts\python.exe scripts\start_local.py --check
node --check web\app.js
```

`tests/test_distribution.py` is an architectural guardrail rather than a logic test: it asserts the
design decisions above as strings (two BAT files only, Train/Export separation, the SavedModel
conversion path, no WSL/CUDA traces). It will fail first when you change the architecture — confirm
the change is intended, then update the assertions.

Python 3.13, `from __future__ import annotations`, ruff line-length 100.

---

## 📄 License

Apache License 2.0 — see [LICENSE.txt](LICENSE.txt).

This is an independent local implementation for education. It is **not** sponsored by, endorsed by,
or affiliated with Google, and it is not an official Google product. "Teachable Machine",
"TensorFlow" and "Google" belong to their respective owners. Third-party components — YAMNet, the
Nuvoton NuML toolkit and M55M1 BSP, tflite-micro, CMSIS, and the Arm Vela compiler — retain their
own licenses; see [NOTICE.md](NOTICE.md) and
[mcu_toolkit/NOTICE_third_party.md](mcu_toolkit/NOTICE_third_party.md) for the per-component
summary, vendored versions and full license texts.
