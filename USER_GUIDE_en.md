# Teachable Machine Local Studio — User Guide

> Version v2.1.0 — Windows only. This guide assumes you have no machine-learning or
> embedded-development background. If you can double-click a `.bat` file, you can follow it.
>
> 繁體中文版：**[使用手冊_zh-TW.md](使用手冊_zh-TW.md)**

> **About the screenshots:** the figures in this guide show the Chinese interface, because that
> is what they were captured in. Click the **中文／EN** toggle in the top-right corner and you get
> exactly the same layout with English labels — nothing moves, only the wording changes. The one
> thing the toggle does *not* translate is messages reported by the server: those are always in
> Traditional Chinese, in both languages.

---

## Contents

1. [What this tool is](#1-what-this-tool-is)
2. [Installation](#2-installation)
3. [Choosing among the four project kinds](#3-choosing-among-the-four-project-kinds)
4. [The everyday workflow](#4-the-everyday-workflow)
5. [What the settings mean](#5-what-the-settings-mean)
6. [Flashing to a dev board](#6-flashing-to-a-dev-board)
7. [When something goes wrong](#7-when-something-goes-wrong)
8. [What has not been verified yet](#8-what-has-not-been-verified-yet)

---

## 1. What this tool is

Teachable Machine Local Studio is a teaching tool that runs entirely on your own Windows PC. In
your browser you create a *project*, collect teaching material with a webcam or a microphone,
press **Train** so the computer learns to tell that material apart, and then test the result
immediately in the same page (**Preview**). If you need to, you can convert the trained model to
TensorFlow Lite and download it, or go one step further and compile it into firmware you flash
onto a supported Nuvoton M55M1 dev board, so the model runs on a small board with no computer
attached. No account is required, and no photo or recording is ever uploaded anywhere — all of
your data stays on your machine.

It is not a general-purpose deep-learning platform, and it is not a cloud service. It supports
exactly four teaching scenarios (classify photos, classify short sounds, recognise several named
sounds, detect abnormal sound), runs only on Windows, and trains only on the CPU — even if your
machine has a discrete graphics card. That last part is deliberate: it keeps every classroom PC
behaving identically. It also supports exactly three dev boards. You cannot use it to train large
general-purpose models; datasets here are typically tens to a few hundred samples and training
finishes in minutes. That *is* the design goal — small-class teaching, not research.

To use it you need: a 64-bit Windows 10 or 11 PC; a clean, correct-version Python (see the next
section); a folder for the tool whose path is short enough (the reason is in the next section —
get this wrong and it bites you weeks later, at the "flash to a dev board" step); an internet
connection for the first install (training, Preview and export mostly work offline afterwards);
and, if you want to use a dev board, one extra free install of the Arm toolchain plus the flashing
tool for your board.

---

## 2. Installation

### 2.1 Before you start

**Check your Python version first.** This is the one thing that, if wrong, makes installation
impossible: you need **64-bit, regular (not the free-threaded 3.13t experimental build) CPython
3.13.x**. Both 3.12 and 3.14 are rejected. When installing Python, tick "Add python.exe to PATH".
If your PC already has several Pythons (Anaconda, for example), that is fine — the installer finds
the one that qualifies. An Anaconda-installed 64-bit Python 3.13 works too; the installer never
starts conda and always builds its own dedicated environment.

**No Python 3.13 on the machine yet?** `0_Python_3.13.15\python-3.13.15-amd64.exe` in the Studio
folder is the official installer for the right version (digitally signed by the Python Software
Foundation). Double-click it, and tick "Add python.exe to PATH" on the first screen.

**Put the folder on a short path.** After you unzip, the length of the folder's own path affects a
feature you will not use for *weeks* — building firmware for a dev board. Training, Preview and
TensorFlow Lite export are unaffected, but the firmware source tree is deeply nested, and Windows'
260-character path limit will stop the compiler from finding some header files. The rule of thumb
is to keep the Studio folder's own path under 78 characters. A Desktop path (such as
`C:\Users\yourname\Desktop\TeachableMachine_Local_Studio_v3`) is often already close to that
limit; prefer something like `C:\TM_Local_Studio` or `C:\TM_Studio`. If you miss this and the path
is too long, pressing **Build firmware** on the "Deploy to a Dev Board" tab shows a message
telling you to move the whole folder somewhere shorter. Move it, restart, and carry on — there is
no need to reinstall.

**The first install needs internet access.** Most later operations work offline — see "The
offline trap" below for the exceptions.

### 2.2 The two BAT files

In the whole folder there are only two files you ever touch:

| File | What it does |
|---|---|
| `01_INSTALL.bat` | Creates a dedicated Python environment (`.venv`) inside the folder, installs the packages, and verifies the environment with one real save/load, export and INT8 quantization run |
| `02_START.bat` | Starts the local site at `http://127.0.0.1:8765` and opens your browser |

Double-click `01_INSTALL.bat` and it does the following, printing progress into the black window:

1. Creates (or, where necessary, rebuilds) the isolated `.venv` Python environment. An existing
   `.venv` of the right version is reused; one with the wrong version, or one left half-finished
   by an interrupted install, is deleted and rebuilt. That is safe — it never touches any project
   data you have collected.
2. Installs the packages and checks that they are compatible with each other.
3. Actually builds a small model, saves it, loads it back, converts it to a SavedModel, quantizes
   it to strict INT8 and inspects the result — confirming your machine really can run the whole
   "train → export" path, rather than just importing the libraries and declaring success.
4. Tries to pre-download a few pretrained weights the models use (YAMNet, MobileNetV2). Failure
   here does **not** fail the install; it only prints a warning (the consequences are described
   below).
5. Checks whether the Arm toolchain needed for dev boards is present. Its absence does not fail
   the install either — it only means you cannot build firmware yet.

**How you know it worked:** the last lines in the window should read, word for word:

```text
[PASS] Native .keras save/load
[PASS] Strict INT8 TensorFlow Lite conversion
[PASS] Local Studio installation verification completed
[DONE] Installation completed
Runtime      : Windows CPU
Linux/WSL    : disabled
Next        : double-click 02_START.bat
```

If you see `Runtime : Windows CPU` and `Linux/WSL : disabled`, the install succeeded. If
`MCU toolchain: missing` appears somewhere in the middle, **that is not a failure** — it only
means you have not installed the Arm toolchain, so dev-board features are not available yet.
Everything else works normally.

Once installed, double-click `02_START.bat`: it sets the environment variables, confirms `.venv`
exists, starts the site, and opens your browser at `http://127.0.0.1:8765` about 1.3 seconds
later. The black window shows:

```text
Local URL : http://127.0.0.1:8765/?session=...
Workspace : <your folder>\workspace\projects
Runtime   : Windows CPU
Stop      : press Ctrl+C
```

**That black window is the server itself — do not close it while you are working.** To stop
cleanly, press `Ctrl+C` inside it. All of your projects (photos, recordings, trained models) live
in the folder shown on the `Workspace` line, and that is the only copy.

The top-right corner of the page has a **中文／EN** toggle that switches the whole interface to
English. Your choice is remembered in this PC's browser, so the next session opens in the language
you last picked. Note that **only interface text changes — messages reported by the server are
always in Traditional Chinese**, so if something goes wrong after switching to English, the error
you see will still be Chinese.

> **What the EN interface looks like** (no screenshot in this version, so here it is in words):
> the layout is identical to the Chinese one — nothing moves or resizes. Only the interface text
> changes: 「匯入專案」becomes Import Project, 「建立新專案」becomes Create a new project,
> 「最近的本機專案」becomes Recent local projects, and the four project cards get English titles
> and descriptions. The project names and creation dates listed below **stay in Chinese**, because
> that is data you typed in and it is never translated. The runtime badge in the top-right corner
> (Windows CPU) was already English, so it looks unchanged.

### 2.3 What to do when the environment breaks (remember this one)

**Delete only the `.venv` folder. Never delete the `workspace` folder.**

1. Close Local Studio (the black window).
2. Delete only `.venv` inside the Studio folder.
3. Keep `workspace` — it holds all of your projects, photos, recordings and trained models.
4. Double-click `01_INSTALL.bat` again.

`.venv` is pure software; if it breaks you can rebuild it any time and lose nothing. `workspace`
is your data, and deleting it really is gone — no recycle bin, no backup. Whenever installation or
startup shows an error, the first thought should always be "rebuild `.venv`", never "delete the
whole folder and unzip again".

### 2.4 Common installation failures and what to do

| What you see | What it means | What to do |
|---|---|---|
| `[ERROR] Compatible Python was not found.` | No qualifying Python (64-bit, regular CPython 3.13.x) on the machine | Install 3.13.x (64-bit) from python.org and tick "Add python.exe to PATH". Do not pick free-threaded 3.13t |
| Double-clicking `02_START.bat` throws an English Python traceback and the window freezes | `.venv` exists but is incomplete (an earlier install was interrupted) | This does **not** mean the program is broken: close the window and run `01_INSTALL.bat` again |
| `[ERROR] Port 8765 已被其他程式使用。` | Usually Local Studio is already running (perhaps opened twice), or another program holds the port | Look for an existing browser tab first. To confirm, run `netstat -ano \| findstr :8765` at a command prompt — no output means the port is actually free |
| The network drops during install and `pip install` fails | Packages could not be downloaded | Check your connection and run `01_INSTALL.bat` again. Even with an existing `.venv`, reinstalling needs the network (the first step always checks for a pip update) |
| The install prints `[WARN] 無法安裝 YAMNet` or a MobileNetV2 warning | These two are optional pretrained weights; failing to fetch them does not fail the install, but it affects later training (see "The offline trap") | Reconnect and run `01_INSTALL.bat` once more — it skips what is already installed and fetches only what is missing |

**The offline trap — you need to know this one.** The MobileNetV2 weights pre-downloaded during
installation exist in only one size (one specific internal configuration used by Image projects),
but a default Image project uses a *different* size. In other words: even when `01_INSTALL.bat`
finishes cleanly with no warnings, **creating an Image project with default settings and pressing
Train for the first time still needs an internet connection** to download the weights actually
used. If you happen to be offline at that moment, training does not fail — it quietly switches to
a much simpler small network, accuracy drops noticeably, and the only on-screen hint is one line
of English text that flashes past at around 8% progress. The summary line after training then
contains the words `small_cnn_fallback`. If you see that, the model you trained is not the
MobileNetV2 you expected; reconnect and train again.

Likewise, the YAMNet weights that Known Sound and Abnormal Sound projects need: if they failed to
download during install, training is blocked outright with a message telling you to reconnect and
re-run `01_INSTALL.bat`. That one fails loudly — it never silently downgrades.

---

## 3. Choosing among the four project kinds

The home page offers four cards for four genuinely different kinds of project. **The kind is fixed
the moment the project is created and can never be changed**; if you pick wrong, your only option
is to start a new project.

![The Local Studio home page — four project cards, the language toggle and the runtime badge](docs/screenshots/01_home.png)

*Figure 1: the home page. This is what the browser opens after you double-click `02_START.bat` in
section 2.2. The four cards match the four kinds in the table below, and a single click creates
the project immediately. The top-right corner switches between 中文 and EN and shows the current
runtime (Windows CPU); below that are the projects already created on this machine.*

| Project kind | In one line | What you collect | How the computer answers | Can it go on a board? |
|---|---|---|---|---|
| **Image Project** | Teach the computer to tell a few things apart, using a webcam or images | Photos of each class | One answer at a time; all class percentages sum to 100% | Yes (X board, GestureAI) |
| **Audio Project** (keyword spotting) | Teach the computer to tell a few short sounds or keywords apart, using a microphone | Short recordings per class (including a "background" class) | One winner at a time; all class percentages sum to 100% | Yes (X board, GestureAI) |
| **Known Sound Project** | Teach the computer to recognise each sound, each with its own independent confidence score | Every sound you care about, plus a "Background" class | One 0–1 confidence score per class, which **does not sum to 100%**; several sounds can be detected at once | Yes (X board, GestureAI, VoiceAI) |
| **Abnormal Sound Project** | Collect only "normal" sound; anything that deviates gets flagged — but it **never says what the sound is** | Normal only (you may add "anomaly examples", but those are only scored, never learned from) | One of Normal / Abnormal / Uncertain, plus a deviation number | **No** — this is the only kind with no dev-board firmware |

### 3.1 How to choose

Think in this order:

1. Photos or sound? Photos → Image Project, done.
2. For sound, do you need the computer to *name* which sound it is?
   - Yes, and several sounds may occur at once → **Known Sound**.
   - Yes, but they are just a few short keywords and only one at a time → **Audio Project
     (keyword spotting)**.
   - No, you only want to know "that sounded wrong" without knowing what it was → **Abnormal
     Sound**.
3. If the end goal is a dev board, Abnormal Sound is out; and the VoiceAI board, which has no
   camera, cannot run Image.

### 3.2 Known Sound and Abnormal Sound are the easiest pair to confuse — keep them straight

These two are often treated as the same thing, but they are deliberately built on opposite logic:

- **They answer different questions.** Known Sound answers "which sound is this?"; Abnormal Sound
  only answers "how far from normal is this?" and never gives a name.
- **They need different data.** Known Sound needs every sound collected equally (plus a Background
  class — without it the model guesses wildly at any sound). Abnormal Sound collects only Normal;
  the "anomaly example" groups you add are used purely to check afterwards how well the detector
  works, and are never used for training, for threshold selection, or in the exported model.
- **When an unseen sound appears.** Known Sound gives every class a low score (exactly why the
  Background class cannot be skipped); Abnormal Sound raises an alarm — which is actually its
  strength, since it never needed to know in advance what an anomaly sounds like.
- **Their outputs differ in shape.** Known Sound gives you n independent scores; Abnormal Sound
  gives you a single deviation number.
- **Simultaneous events.** Known Sound can detect several sounds at once; Abnormal Sound has no
  such concept.

A simple rule: **use Abnormal Sound for "alarm on anything odd", and Known Sound for "tell me
which one". They can happily coexist in the same setting as two separate projects.**

One real lesson is worth recording here. Someone once believed they had recorded 62 "abnormal
sound" samples; checking later against an official sound-recognition tool revealed that all 62
were people talking, with nothing to do with gunshots, dog barks or breaking glass (those scored
zero). The problem is that **an Abnormal Sound detector, which reports only a deviation number, is
structurally incapable of noticing that you recorded the wrong thing** — it happily reported
"62/62 detected" and everything looked fine. This is why a small "recording health check" now pops
up automatically after each recording, using an official model to tell you what the clip sounds
like. If what it says bears no resemblance to what you thought you recorded, you probably captured
the wrong source and should record again. (Right now this check only fires after *live microphone
recording*. Uploading an existing audio file does not trigger it, so check uploads yourself.)

### 3.3 Known Sound scores are confidence, not probability

Every Known Sound class has its own independent 0–1 score, and those scores **do not add up to
100%**, because each class is judged independently rather than all classes competing for one
shared 100%. That is intentional: two sounds genuinely can happen at once (a dog barking *while*
glass breaks), so they should not have to steal score from each other. On screen and in the
exported files these numbers are always called **confidence scores**, never "probabilities": a
model trained on a small number of recordings is inherently prone to overconfidence, so the number
reflects how sure the model is, not a real-world probability. Seeing one class at 80% and another
also at 80% is normal, not a bug to be fixed by making them sum to 100%.

![Known Sound Preview — three classes with independent confidence bars and threshold ticks](docs/screenshots/19_known_sound_preview.png)

*Figure 2: Known Sound Preview (the demo data is synthetic audio). The three scores are 93.4%,
13.3% and 2.9%, summing to 109.6% — not an arithmetic error, but the result of judging each class
independently. The dashed tick on each bar is the threshold that must be exceeded to count as
detected; all three share 50.0% here, but if you set per-class thresholds the ticks sit at
different places (see 5.5).*

---

## 4. The everyday workflow

All four kinds follow the same steps: **create the project → add classes → collect samples → train
the model → test in Preview → export the model**. Two rules run through the whole process, and you
should understand them first:

> **Rule 1: any time you touch a sample or a setting, the already-trained model is thrown away
> immediately.**
> **Rule 2: the moment you press Train, the old model, the old TensorFlow Lite files and any
> firmware you already built are cleared — whether or not the new training eventually succeeds.**

Taken together: **from the moment you change anything, assume the model is gone**, until the next
training run succeeds. There is no confirmation dialog and no undo button.

### 4.1 Creating a project and adding classes

Click one of the four cards on the home page and the project is created immediately, with the
default class names for that kind (Image gives "Class 1" and "Class 2"; Audio gives "Background
Noise" and "Class 2"; Known Sound gives "Class 1", "Class 2" and "Background"). Project and class
names can be changed at any time, and **renaming does not affect an already-trained model**.

However: **adding or deleting a class, or clearing a class's samples, destroys the trained model
immediately** (Rule 1). Each project allows at most 20 classes. Image, Audio and Known Sound must
keep at least 2 classes (deleting down to 1 is blocked). Abnormal Sound's "Normal" container is
locked — it can be neither renamed nor deleted — but the extra "anomaly example" groups can all be
removed.

### 4.2 Collecting samples

**Image:** open a class card and hold the webcam record button; the system keeps taking a photo
every 260 ms (about 3.8 per second) until you let go. You can also use "Upload" to import many
images at once. The card shows only the most recent 240 thumbnails, but training uses **all**
samples — trust the "N Image Samples" text for the count, not the number of thumbnails.

![Image project page — class cards with thumbnails on the left, the training card in the middle, Preview on the right](docs/screenshots/02_image_project.png)

*Figure 3: an Image project (the demo data is synthetic imagery). The page has three columns: each
card on the left is a class, and the "Webcam" and "Upload" buttons on it are the two ways to add
samples — the thumbnails show only the last 240, so read the real count from the "N Image Samples"
line. The middle column is the training card (section 4.4 presses the button here; the model has
already been trained in this shot, so it reads "Retrain Model"). The right column is Preview and
Export Model (sections 4.5 and 4.6).*

**Audio (including Known Sound and Abnormal Sound):** there is one concept here that is easy to
get wrong — **"one recording session" and "the number of clips cut from one file" are not the same
thing**. One press of "Record 20 Seconds" (which stops automatically at 20 s) is cut into 20
one-second clips, but that whole recording counts as **one** independent session, not 20
independent examples. Known Sound and Abnormal Sound both check the session count and the clip
count separately, and during validation they deliberately keep every clip cut from the same
recording on the same side (all in training, or all in validation) so that the reported accuracy
is not inflated. In other words: **record each sound many separate times** (different moments,
distances, volumes, positions) rather than one long file. Uploads work the same way — one file is
one session, and a file longer than 120 seconds is truncated to its first 120 seconds, silently,
with no warning.

![A Known Sound class card — clip count and independent-session count shown as two separate numbers](docs/screenshots/20_recording_sessions.png)

*Figure 4: a Known Sound project where the first two classes have been recorded once each and
Background not at all. The "6 Audio Samples / 20 minimum" line and the "1 / 2 independent
recordings" line below it are two separate thresholds, each followed by how far you still have to
go. This is exactly why the previous paragraph asks you to record many separate times: one long
recording only grows the top number, while the bottom one stays at 1.*

**Collecting with the board's own microphone or camera (optional):** if the model will end up on a
board, collecting samples through that *same* microphone or camera keeps the model from learning
the sound and picture quality of a laptop mic or webcam instead of what the board actually
receives. The firmware in `1_Collect_Firmware_bin` does exactly this: once flashed, the board shows
up on the PC as a USB microphone or a USB camera, and you pick it from the microphone / camera menu
while collecting samples in Studio.

| Board folder | File | The board becomes | Flashable `.bin` inside the zip |
|---|---|---|---|
| `NuMaker-X-M55M1D` | `DMIC_UAC_Codec_Monitor.zip` | USB microphone (shown on the PC as `M55M1 DMIC Mic`), plus live headphone monitoring | `Keil\release\DMIC_UAC_Codec_Monitor.bin` |
| `NuMaker-X-M55M1D` | `HSUSBD_Video_CAM.zip` | USB camera (HM1055 sensor) | **None** — source only; build it yourself in Keil |
| `NuMaker-GestureAI-M55M1` | `DMIC_UAC_NuMaker-GestureAI-M55M1.zip` | USB microphone | `Keil\release\DMIC_UAC_Codec_Monitor.bin` |
| `NuMaker-GestureAI-M55M1` | `HSUSBD_Video_CAM_GC0308.zip` | USB camera (GC0308 sensor) | `KEIL\Objects\HSUSBD_Video_CAM.bin` |
| `NuMaker-VoiceAI-M55M1(Chip select M5531)` | `DMIC_UAC_NuMaker-VoiceAI-M55M1.zip` | USB microphone | `VSCode\out\DMIC_UAC_Codec_Monitor\ARMCLANG\Release\DMIC_UAC_Codec_Monitor.bin` |

Each zip is a complete project source tree. Unzip it, find the `.bin` listed above, and flash it
using your board's method from section 6.4. Remember that a board runs one firmware at a time: when
you later flash the model firmware built by Studio, it replaces this collection firmware, so flash
the collection firmware again whenever you want to collect more samples.

### 4.3 How many samples are enough (these are floors, not targets)

Each kind has a minimum before Train becomes available. Below it, the button stays grey and the
card tells you what is still missing:

| Project kind | Minimum |
|---|---|
| Image | at least 5 images per class |
| Audio (keyword) | at least 8 clips per class |
| Known Sound | at least 20 clips per class **and** at least 2 independent recording sessions |
| Abnormal Sound | Normal needs at least 3 independent sessions **and** at least 60 seconds total; to upgrade the system from "score only" to a confident Normal/Abnormal verdict, you need at least 6 sessions and over 120 seconds |

**These are the floors for starting training, not recommendations for getting an accurate model.**
For example, an Image project at the minimum (5 per class) with the default validation split
leaves exactly 1 validation photo per class — so the reported accuracy can only be 0%, 50% or
100%, which tells you very little. A real data point you can work from: a Known Sound project with
3 classes and 122 clips in total (spread across several recording sessions) reached a macro
precision of 0.969. So **a few dozen per class, spread over several sessions** is a realistic
classroom target; you do not need hundreds.

![Two class cards short of samples, with a greyed-out Train Model button](docs/screenshots/17_not_enough_samples.png)

*Figure 5: what it looks like when samples are short. Each card states how many more it needs, the
train button stays grey and unclickable, and the line underneath explains which threshold is
blocking you — you never have to come back to the table above and work it out.*

### 4.4 Training the model

Pressing **Train Model** first sends the values from the "Advanced" panel to the server to be
saved, then starts training. **Training produces only the native `.keras` model — no TensorFlow
Lite file at all.** That is deliberate: earlier versions quantized during training, which on a slow
PC left progress stuck at 71% for minutes and looked like a crash. Now you can Preview as soon as
training finishes, and quantization (the INT8/UINT8 conversion) happens only when you press
**Export Model** — in a separate background job with its own progress bar and elapsed-time
display, which aborts on timeout (20 minutes by default) without damaging the trained model.

![Training in progress — a percentage progress bar and the current stage message](docs/screenshots/04_training_progress.png)

*Figure 6: training in progress. The bar and the message run all the way to 100%. Pausing at some
percentage for tens of seconds is normal — especially the first time pretrained weights have to be
loaded — and is not a crash.*

While training runs, every edit to that project (settings, adding or deleting samples, renaming)
is blocked with the message "此專案已有訓練、匯出或部署工作進行中。" ("this project already has a
training, export or deploy job running"). Wait for it to finish.

![Training finished — a validation-accuracy summary and a retrain button](docs/screenshots/05_training_done.png)

*Figure 7: training finished. The summary line gives the validation accuracy and which model
version was actually used — seeing `small_cnn_fallback` means you were offline and got downgraded
(see "The offline trap" in 2.4). Known Sound is the exception: it shows no accuracy figure at all
when it finishes (see 5.5).*

### 4.5 Preview

Once training finishes, the Preview panel unlocks. Turn on the webcam or microphone switch and
scores appear live; you can also use "choose an image / audio file" to test a single file without
opening a camera or microphone. A dropdown at the top switches between "Float (trained) model",
"INT8 quantized model" and "UINT8 quantized model" — the latter two only work once you have
exported that format, and before that Preview always uses the Float version.

![The Preview panel — confidence bars per class after uploading an image](docs/screenshots/06_preview.png)

*Figure 8: Preview (the demo data is synthetic imagery). After uploading an image, each class gets
a score bar; the dropdown above is where you switch between Float, INT8 and UINT8.*

One thing to remember: **an Audio (keyword) project's "detection threshold" setting has no effect
in Preview at all** — Preview always shows the model's raw scores. That threshold is applied only
by the exported runner script and by firmware flashed onto a board. If you set the threshold to
0.9 and still see a 0.4 score light up in Preview, that is not a bug. (Known Sound's per-class
thresholds behave differently — those *are* reflected in Preview, drawn as a tick mark on each
class's bar.)

### 4.6 Exporting the model

The **Export Model** dialog has two tabs: TensorFlow Lite (download model files) and "Deploy to a
Dev Board" (see section 6). The TensorFlow Lite tab offers five formats: Quantized INT8 (ticked by
default, and the recommended one), Quantized UINT8, Float32, Dynamic Range and Keras model. There
is also an option to generate a `model_data.h` C array alongside them.

![The Export Model dialog, TensorFlow Lite tab — five format checkboxes and the C header option](docs/screenshots/07_export_tflite.png)

*Figure 9: the TensorFlow Lite tab of the Export Model dialog. Of the five formats, only
Quantized INT8 is ticked by default, and the "also generate model_data.h C array" box at the
bottom is ticked too. The other tab, "Deploy to a Dev Board", is section 6.*

INT8/UINT8 conversion is **strict**: afterwards the system checks that input and output really are
integers, that no floating-point tensor remains, and that no unsupported operator is used. If any
one of those fails, the export **fails outright** and tells you why, rather than quietly handing
you a mixed-precision model labelled INT8. If the INT8 model agrees with the original Float model
on fewer than 95% of predictions (this check applies only to the Image and Audio classifiers), a
yellow warning appears but the export still succeeds — it means quantization cost you real
accuracy, and it is worth revisiting your data volume or training settings.

If you tick only "Keras model", the ZIP contains just the `.keras` file plus labels and
configuration, with no runnable example. Tick any TensorFlow Lite format and the ZIP also includes
a ready-to-run `run_model.py`.

---

## 5. What the settings mean

This is the most important section of the guide. Opening "Advanced" in the training area below the
class cards reveals a set of controls that differ by project kind. Before looking at individual
settings, three things need to be clear:

![The expanded Advanced panel for an Image project, target device set to PC](docs/screenshots/16_training_options_pc.png)

*Figure 10: the "Advanced" panel expanded. This is an Image project with the target device left at
its default ("電腦", PC), so nothing is restricted yet. Other project kinds have different
controls — see 5.4 to 5.6 — and 5.7 shows what happens once you pick a board.*

**Until you press Train Model, the numbers in this panel are only a draft held in your browser;
they have no effect.** Only pressing Train sends them to the server to be saved and used. So you
can open the panel and look around freely with no consequences — but equally, if you nudge a
number by accident and then press Train, the old model is cleared instantly, whether or not the
new training succeeds.

**The real gatekeeper is the server, not the sliders and number boxes, and it always answers with
the English setting key.** Most input boxes show a min/max hint, but what actually rejects an
unreasonable value is the server-side check. Submitting an illegal value shows a red message such
as "`epochs` 必須介於 1 與 300 之間，收到 500" — note that the message names the English setting
key (`epochs`), not the label shown on screen. Every setting below is listed with both its
on-screen label and its key, so you can match up error messages.

**Picking a board locks some settings.** Firmware is compiled for one fixed input format, so
changing that format afterwards leaves the firmware unable to read the new model. Which settings
get locked, and to what, is summarised at the end of this section.

### 5.1 Target device (`deployment_target`) — on every deployable project

This choice decides which board the project will eventually be flashed to, and **it is a training
setting, not something you pick at export time**. Once a board is chosen, other settings are
constrained to what that board supports. Abnormal Sound has no such option at all, because it has
no firmware.

The default is "電腦" (PC). Only boards that support this project kind appear in the dropdown —
the camera-less VoiceAI board, for instance, is simply absent from an Image project's menu. The
advice is to **decide whether you want a board before collecting samples and setting training
parameters**; choosing a board afterwards often reveals that the settings you trained with (image
size, for example) are incompatible, forcing a retrain.

### 5.2 The five basic settings shared by Image and Audio

| Label (key) | Default | Allowed range | When to change it | What you see if it is wrong |
|---|---|---|---|---|
| Epochs (`epochs`) | Image 30 / Audio 40 | 1–300 (integer) | Early stopping is on by default and derives its patience from the epoch count (roughly a quarter of it, between 3 and 10), so raising epochs does not necessarily mean training longer. If you really want more, first check whether `epochs_completed` in the training report equals the `epochs` you set (meaning it never stopped early and was still improving), then consider 60–80. For a fast classroom demo, 10–15 works | Out of range: "`epochs` 必須介於 1 與 300 之間，收到 500"; a decimal: "`epochs` 必須是整數，收到 12.5" |
| Batch Size (`batch_size`) | 16 | 1–128 (integer) | At classroom data volumes (tens to hundreds of samples) the default is fine and almost never needs changing. Reduce it only if you raise image resolution above 224 and training reports a memory error | Out of range is rejected. Setting it larger than the whole training set is legal but means one batch per epoch, so the model barely learns and accuracy stalls near 1 ÷ number-of-classes — with no error message explaining why |
| Learning Rate (`learning_rate`) | 0.001 | 0.000001–0.1 | Best left alone. If validation loss oscillates wildly, try 0.0005 or 0.0001. Above 0.003, combined with a frozen MobileNetV2, usually hurts. **The up/down arrows on the input box step by a tiny amount and do not start from the default** — type the number instead | Zero or negative: "`learning_rate` 必須介於 1e-06 與 0.1 之間，收到 0.0"; too high: training completes but accuracy stays near chance, with nothing on screen pointing at the learning rate |
| Validation Split (`validation_split`) | 0.20 | 0.0–0.45 | Raise to 0.3 when you have plenty of data and want more confidence in the accuracy figure. **Never set it to 0**: that is technically legal, but "evaluation accuracy" then means grading the model on photos it trained on, which usually inflates to near 100% and means nothing | Out of range is rejected. At Image's minimum (5 per class), a 0.20 split leaves 1 validation image per class, so accuracy can only be 0%, 50% or 100% |
| Early stopping (`early_stopping`, checkbox) | on | on / off | Leave it on. Turn it off only to demonstrate what a full training curve looks like; without it the model ends at the last epoch rather than the best-validating one, which is usually worse | A non-boolean is rejected: "`early_stopping` 必須是 true 或 false" |

### 5.3 Image-only settings

| Label (key) | Default | Allowed range | When to change it | What you see if it is wrong |
|---|---|---|---|---|
| Model / backbone (`backbone`) | MobileNetV2 transfer learning | `mobilenet_v2` or `small_cnn` | Almost never choose Small CNN deliberately — it is a small network trained from scratch and clearly loses to a pretrained MobileNetV2 at classroom data volumes. It exists as the fallback for offline failures | Choosing Small CNN greys out the fine-tune control entirely. **If the MobileNetV2 pretrained weights fail to download** (usually offline), training does not error — it switches to Small CNN, and the summary line afterwards reads `small_cnn_fallback`. Only those words mean you were really downgraded; a line like `mobilenet_v2_alpha_0.35` is normal (every successful MobileNetV2 run names the version it used) and is not a warning |
| MobileNet Size (`mobilenet_alpha`) | 0.35 | one of 0.35 / 0.5 / 0.75 / 1.0 | 0.35 is this tool's original default and usually the best choice. Consider 0.5 or 0.75 only for PC-only projects where classes look very similar and you have plenty of photos. **If you intend to flash a board, anything above 0.35 will almost certainly be rejected at deploy time for lack of memory** | A value outside the list (0.6, say): "`mobilenet_alpha` 只能是 0.35、0.5、0.75 或 1.0". **Offline trap**: the install pre-downloads only the alpha=1.0 (224) weights, while a default project uses alpha=0.35 — so even after a clean install, the first training run with default settings still needs the network to fetch a different weights file, and offline it silently falls back to Small CNN (see the row above). When it will not fit on a board you see "tensor arena 超過板子預算" at the firmware-build stage; lower this or Image Size |
| Image Size (`image_size`) | 224 | PC: 96–320 in multiples of 32 (96/128/…/320); **once a board is chosen, only 96 / 128 / 160 / 192 / 224** | Raise it when the thing you want to recognise is small in frame (hand gestures, for example); lower it when the board is short of memory or training is slow. **256 / 288 / 320 have no pretrained weights of their own and borrow the 224 set**, costing noticeably more time and memory, usually without being more accurate — to improve accuracy, add samples or raise augmentation before raising resolution | Not a multiple of 32: "`image_size` 必須是 32 的倍數（96、128、...、320），收到 200"; a board plus 320: "部署到開發板時 `image_size` 只能是 (96, 128, 160, 192, 224)，收到 320". Switching target device from PC to a board automatically resets an illegal size to 224 and tells you so |
| Fine-tune blocks (`fine_tune_blocks`) | 0 (no fine-tuning) | 0–4 (integer) | Try 1–2 when default training plateaus around 80–90% and you already have 50+ photos per class. This is the only way to let MobileNetV2 itself adapt to your photos, at a cost of roughly 50% more training time. With few samples it tends instead to push training accuracy up while validation accuracy falls (overfitting) | Out of range: "`fine_tune_blocks` 必須介於 0 與 4 之間，收到 7". With Small CNN selected the control is locked at zero, because Small CNN has no pretrained weights to fine-tune |
| Augmentation (`augmentation_level`) | medium | off / light / medium / strong | With few samples, or when every photo was shot in the same scene and lighting, raise this to "strong" before raising Image Size. Only avoid the flipping levels (every level except "off" flips horizontally) when left/right has meaning — telling "left hand" from "right hand", say — since flipping there teaches the model the wrong answer | An unknown level: "`augmentation_level` 必須是 off／light／medium／strong 之一，收到 …" |
| Dropout (`dropout`) | 0.2 | 0.0–0.6 | Raise to 0.3–0.4 when training accuracy far exceeds validation accuracy (rote memorisation); lower to 0–0.1 if even training accuracy will not climb | Out of range is rejected. At 0.6 with little data, accuracy may never climb at all, and nothing on screen blames Dropout |

### 5.4 Audio (keyword spotting) settings

![The Audio keyword-spotting project page and its specific training settings](docs/screenshots/12_audio_project.png)

*Figure 11: an Audio (keyword spotting) project and its settings. Sample rate, Mel Bins and
SpecAugment exist only for this kind; the background class and detection threshold are shared with
Known Sound. To collect samples you must first press "Mic" on the card before "Record 20 Seconds"
appears — see section 4.2 for what a "recording session" means.*

| Label (key) | Default | Allowed range | When to change it | What you see if it is wrong |
|---|---|---|---|---|
| Sample Rate (`sample_rate`) | 16000 Hz | 16 kHz or 44.1 kHz on PC; **once a board is chosen, only 16 kHz**, and the menu locks | Not recommended. 44.1 kHz suits only PC-only projects needing higher-frequency content, and **switching to it also changes the FFT length and top frequency — the entire spectrogram spec — so the model must be retrained from scratch** | 44100 in board mode: "部署到開發板時 `sample_rate` 必須是 16000，收到 44100". Switching the target device to a board resets it to 16 kHz and warns you that the spectrogram spec changed and the model needs retraining |
| Mel Bins (`mel_bins`) | 40 | only 40 or 64, board or not | 64 resolves finer timbral differences at a slightly larger model; 40 is the smallest and the default. This limit does not relax for PC-only projects, because the training CNN, the exported filter table and the firmware's C constant tables only line up at these two widths | 80: "`mel_bins` 只能是 (40, 64) 之一，收到 80" |
| Augmentation (`augmentation_level`) | medium | off / light / medium / strong | Raise to "strong" (which adds noise) when every recording came from the same quiet room and the same microphone; use "off" when exact timing within a clip must be preserved | An unknown level is rejected; an unknown value read from an imported file silently degrades to medium |
| SpecAugment (`spec_augment`) | off | on / off | Turn on when clips per class are few — the masking is deliberately small (at most a tenth of each axis), so the risk is low | A non-boolean is rejected |
| Session-disjoint validation (`session_disjoint_validation`) | **on** | on / off | Leave it on. This is what guarantees validation clips are not neighbours of training clips cut from the same recording; turning it off makes accuracy look prettier and mean less. When one class's samples all come from a single session, that class falls back to per-clip splitting automatically and is named in the post-training note, telling you "accuracy for these classes may be inflated — record again at a different time and place" | — |
| Background class (`background_class_id`) | blank (auto-detect) | any class in the project, or blank to let the system match by name (it looks for background / noise / 背景 / 環境音 / 安靜 and similar) | Simplest is to name a class something like "Background Noise" and let auto-detection find it; set it manually if that picks the wrong one | A deleted class ID: "`background_class_id` … 不是這個專案的類別" |
| Background weight (`background_weight`) | 1.0 | 0.25–4.0 | Raise it when the device keeps false-triggering on silence or noise (fewer false alarms, but real keywords get missed more easily); lower it if the device is too sluggish | Out of range is rejected. **Setting it in a project with no background class is not an error but has no effect whatsoever** — one line flashes past during training, and the training report records that the requested weight differs from the applied one (1.0) |
| Detection threshold (`detection_threshold`) | 0.5 | 0.05–0.99 | Raise it when the device mistakes one keyword for another; lower it when nothing triggers. **This setting takes no part in training and Preview never applies it** (Preview always shows raw scores) — only the exported `run_model.py` and the firmware on a board actually use it | Out of range is rejected. **Two-class projects hide a trap**: a board additionally requires the winner to beat the runner-up by at least 0.2, and two softmax scores always sum to 1, so **no matter how low you set the threshold, the effective threshold on a board never drops below 0.6**. The real fix is to add another class (a background class, for instance), not to keep dragging this slider |

### 5.5 Known Sound settings

Known Sound uses a fixed, already-trained YAMNet as its "ear" (that part is never retrained); what
you actually train is a very small classification layer bolted on behind it. This is why most of
the settings below are so cheap to change — the time-consuming step is the first pass where YAMNet
converts all your recordings into internal features, not the repeated training of the layer after
it.

![The Known Sound project page — per-class threshold fields, encoder depth and other settings](docs/screenshots/13_known_sound.png)

*Figure 12: a Known Sound project and its specific settings. Each class has its own threshold field
(leaving it blank counts as 0, and 0 is illegal), and encoder depth only ever needs lowering to fit
on a board. Scores here are always "confidence scores", independent per class and not summing to
100% (see 3.3).*

| Label (key) | Default | Allowed range | When to change it | What you see if it is wrong |
|---|---|---|---|---|
| Encoder depth (`encoder_depth`) | 14 (full) | 2–14 (the UI offers 14 / 12 / 11 / 10 / 9 / 8 / 7 / 6) | **Only ever lower it to fit the board's flash; on a PC there is no reason to leave 14.** Fewer layers means a smaller model but a weaker ability to tell similar sounds apart. Depth does not affect RAM — every depth uses the same scratch memory (about 151 KB); what you save is flash | See the depth table below |
| Detection threshold (default, `detection_threshold`) / per-class thresholds (`class_thresholds`) | 0.5 | strictly between 0 and 1 (neither endpoint allowed) | Raise a class's own threshold when it false-triggers; lower it for a class that is rare but must never be missed. Only classes you actually changed are stored as exceptions; the rest always follow the default | Blank or non-numeric: "進階設定「每類門檻」有一個類別填的不是數字"; 0 or 1: "必須是 0 與 1 之間（不含）的數字。把欄位清空也算 0" |
| Waveform augmentation (`waveform_augment_level`) | **off** (note: unlike Image/Audio, whose default is medium) | off / light / medium / strong | Raise it when clips per class are few, or when everything was recorded in one place. This makes randomly gain-shifted, time-shifted and noise-added copies of the raw waveform *before* YAMNet ever sees it, so YAMNet sees more variation | An unknown level is rejected |
| Head dropout (`head_dropout`) | 0.0 | 0.0–0.5 | Raise it only when samples are few and validation scores sit clearly below training scores. Because the layer actually being trained is tiny (a few thousand parameters), overfitting risk is lower than for Image, so this is rarely the first knob to reach for | Out of range is rejected |
| Mixup (`mixup_ratio`) | 0.5 (the UI offers only 0 / 0.5 / 1.0) | the server only requires it not be negative | Keep 0.5 — this deliberately synthesises "two sounds at once" training data, giving the multi-label capability something real to learn from instead of hoping the model generalises there by itself. **Turn it off only when you are certain sounds never overlap in practice. It compounds with waveform augmentation: with both on, the number of mixed samples is original clips × waveform-augmented copies × mixup ratio, so each extra augmentation multiplies the training work** | A value outside the three options is silently reset to 0.5 with a notice |
| Epochs / Batch Size / Learning Rate | 120 / 32 / 0.001 | **the server imposes almost no ceiling on these three** (epochs and batch size must be at least 1, learning rate above 0) — far looser than Image/Audio | Leave them alone: the layer being trained is tiny and works on precomputed features, so even 120 epochs takes seconds. The training report gives one concrete data point: with too few epochs (20, say) the post-quantization error against the original model grows noticeably | 0 epochs is rejected; a very large number (5000 by a typo) is not blocked and really does run, though early stopping usually ends it sooner |

**How encoder depth affects model size** (measured values, to help you decide whether to lower it):

| Depth | Internal feature dim | Size after Vela (approx.) | Board limit |
|---:|---:|---:|---|
| 14 (default) | 1024 | ~2.97 MB | fits on none of the three boards |
| 13 | 1024 | ~2.07 MB | fits on none of the three boards |
| 12 | 512 | ~1.56 MB | X board only |
| 11 | 512 | ~1.31 MB | X board, GestureAI and VoiceAI all fit |
| 10 | 512 | ~1.05 MB | fits |
| 9 | 512 | ~0.80 MB | fits |
| 8 | 512 | ~0.54 MB | fits |
| 7 | 512 | ~0.28 MB | fits |
| 6 | 256 | ~0.14 MB | fits |

In the one accuracy test that was actually run (3 classes, 122 clips, split by recording session),
depths 9–11 were nearly as accurate as depth 14, and below depth 7 accuracy clearly degraded. Depth
12, however, produced an unexplained anomaly in that test (recall fell to 0.66, reproducibly across
three runs), **so depth 12 is not recommended** even though it happens to be the X board's ceiling.
This table **reflects only that one dataset** — after changing depth, always re-read the validation
report for your own data rather than assuming these numbers transfer.

**Known Sound differs from the other three kinds in one respect: no accuracy figure is shown when
training finishes**, only "模型已可預覽與匯出" ("model ready to preview and export"). Per-class
precision and recall exist only in the exported files (`training_report.json`, or the export ZIP);
nothing in the web UI currently draws them, so you have to open the file yourself.

The panel also shows two read-only lines: the model is always "YAMNet embedding (frozen) + sigmoid
head", and preprocessing is always "16 kHz · 64 mel · 0.96 s patch". That is YAMNet's official
spec and is a different thing from this project's own sample-rate and mel settings (which are only
used for slicing clips and drawing thumbnails); there is no UI for changing it.

### 5.6 Abnormal Sound settings: one knob

![The Abnormal Sound Advanced panel — a single sensitivity option](docs/screenshots/14_abnormal_sound.png)

*Figure 13: Abnormal Sound's Advanced panel — genuinely just one "sensitivity" knob. Everything
else (how the threshold is computed, which statistical method is used) is deliberately not exposed.*

The only adjustable setting is sensitivity:

| Sensitivity | Statistical tolerance | Votes needed among the voting windows | When to use it |
|---|---|---|---|
| Sensitive (detects more readily) | looser | 2 of 5 | When missing an anomaly costs more than a false alarm (security monitoring, say) |
| Balanced (default) | medium | 3 of 5 | General use |
| Low false alarm | stricter | 3 of 5 | Long unattended monitoring where nobody is watching and false alarms are a nuisance |

Detection uses a 1-second window sliding every 0.5 seconds, and **the first 4 windows after
startup always report "Uncertain" (warming up) and never raise an alarm**. All other design details
(threshold computation, statistical method, feature dimensionality) are deliberately withheld from
the teaching UI. Changing sensitivity invalidates the trained model just like any other setting,
because the sensitivity parameters feed into the internal threshold calibration.

### 5.7 Which settings lock once you pick a board

Firmware is compiled against a fixed input format (image size, every number in the audio frontend,
model depth); if training settings and firmware disagree, the model and the firmware cannot read
each other. The system checks at three separate moments: when settings are saved; when firmware is
built (in case you trained on "PC" first and picked a board later); and after Vela compiles, when
size is estimated. Every rejection names the setting you need to change.

| Project kind | Locked once a board is chosen |
|---|---|
| Image | Image Size must be one of 96 / 128 / 160 / 192 / 224 |
| Audio | Sample rate must be 16000 Hz; FFT length, window length, hop, min/max frequency and dB floor are all pinned to fixed values (most have no UI control at all — if you received such a project by import or through the API, there is nowhere on screen to change them, so switch the target device back to PC or start a fresh project and record again) |
| Known Sound | Encoder depth: 12 on the X board, 11 on GestureAI and VoiceAI (independent of the board: Mel Bins is always 40 or 64) |
| All three | The VoiceAI board has no camera, so Image projects cannot select it |

Advice: **if you know a board is the destination, pick it before collecting samples and tuning
settings** — it saves a lot of backtracking.

![The Advanced panel after selecting the GestureAI board, with Image Size at 96 × 96](docs/screenshots/03_training_options.png)

*Figure 14: the same panel after choosing NuMaker-GestureAI-M55M1 as the target device. Compared
side by side with the one at the start of section 5 (target device = PC), the locking described
above is visible: the help text under Image Size changes from "224 is the default; 256, 288,
320…" to "the board firmware only compiled these sizes". The menu is collapsed here, but expanded
it lists only the sizes in the table above.*

---

## 6. Flashing to a dev board

"Deploy to a Dev Board" is the second tab of the Export Model dialog. Only Image, Audio (keyword)
and Known Sound can use it; Abnormal Sound has no firmware at all.

### 6.1 How the three boards differ

| Board (as named on screen) | Camera | How results are shown | Known Sound depth limit | Flashing method | Projects it can run |
|---|---|---|---|---|---|
| NuMaker-M55M1 (X board: LCD, on-board Nu-Link) | Yes | Built-in LCD | 12 | Nu-Link Command Tool (on-board, or download via Keil) | Image / Audio / Known Sound |
| NuMaker-GestureAI-M55M1 (no LCD: USB camera + COM port) | Yes | No LCD; use the Windows "Camera" app to view the USB camera | 11 | Default: drag-and-drop to a USB mass-storage mode; Nu-Link also works (a user-reported path, not one documented officially) | Image / Audio / Known Sound |
| NuMaker-VoiceAI-M55M1 (no camera: audio only, USB COM port) | No | No display at all, text only (USB virtual COM port) | 11 | pyocd plus an external Nu-Link2 debugger | Audio / Known Sound only (no camera, so no Image) |

All three boards have the same internal flash (2 MiB) and the same SRAM. They differ only in
camera, LCD, microphone pin configuration, and the flashing methods listed above.

### 6.2 What you need to install first

**The only thing a student must install separately is the Arm GNU toolchain**
(`arm-none-eabi-gcc`). The firmware templates, board BSP and the Vela NPU compiler all ship inside
the Studio folder — no separate download. Licensing prevents bundling the toolchain, so install it
once:

```bat
.venv\Scripts\python.exe scripts\download_arm_toolchain.py
```

This downloads the pinned official version (14.2.rel1), verifies its SHA-256 to confirm it is
neither corrupt nor tampered with, and unpacks it into `runtime\arm-gnu-toolchain\`. There is no
need to restart Studio — switching to the "Deploy to a Dev Board" tab re-detects it immediately.
If you would rather not use the script, Arm's own installer works, as does unpacking it yourself
to the same place; all that matters is that `runtime\arm-gnu-toolchain\bin\arm-none-eabi-gcc.exe`
can be found.

**Flashing tools are a separate matter, and in most cases you can skip them** — building firmware
and downloading the ZIP need none of them. They are needed only for the one-click "flash to board"
button inside Studio:

- **X board:** requires the Nu-Link Command Tool (Nuvoton's official tool; Studio finds it at the
  usual install locations). The installer ships in
  `2_Compiler and Download Tool Driver\en-us--Nu-Link_Command_Tool_V3.23.7973r.zip` — unzip it and
  run the installer inside.
- **GestureAI:** the default USB mass-storage mode needs nothing installed.
- **VoiceAI:** run `.venv\Scripts\python.exe scripts\setup_voiceai_flash.py` once. It installs
  pyocd into Studio's own environment and downloads Nuvoton's official device description pack
  (also verified by size and hash). This board has no built-in debugger, so you must supply an
  external Nu-Link2.

Even with no flashing tool installed at all, you can still press "download firmware ZIP", get
`firmware.bin`, and flash it with whatever tool you prefer.

If you open the "Deploy to a Dev Board" tab before installing the Arm toolchain, the **Build
firmware** button is greyed out and the text beside it names what is missing, for example
"還不能部署：缺少 Arm GNU Toolchain（arm-none-eabi-gcc）。" After installing, you do not need to
restart Studio — switch tabs once and it re-detects.

![The Deploy to a Dev Board tab — the board dropdown and the blue Build firmware button](docs/screenshots/08_mcu_board.png)

*Figure 15: the second tab of the Export dialog, "Deploy to a Dev Board". The board dropdown at the
top shows the currently selected board when collapsed (NuMaker-GestureAI-M55M1 here) and, when
expanded, lists only boards that support this project kind. The blue "Build firmware" button below
means the toolchain has been detected; when it has not, the button is grey and the text beside it
names what is missing.*

### 6.3 What happens during the build

Pressing **Build firmware** starts a background job that moves through several stages, with a
progress percentage on screen:

1. **Confirm a strict INT8 export exists** — if you never pressed Export, this step does it for you.
2. **Check the deployment contract** — labels, quantization parameters, preprocessing spec and
   per-class thresholds are re-checked against the selected board's limits (you may have trained on
   "PC" and chosen a board afterwards).
3. **Compile the model for the NPU with Vela**, reporting the flash and scratch memory the model
   needs. If the first attempt (optimised for speed) exceeds budget, it automatically retries with
   a size-optimised configuration.
4. **Check flash capacity early** — using Vela's computed size, before compiling any firmware, so
   that if it will not fit you are told which setting to change (usually image size, MobileNet
   size, or Known Sound's encoder depth) rather than waiting two minutes for a mid-compile failure.
5. **Assemble the full firmware project, call `arm-none-eabi-gcc` to compile and link, and package
   the ZIP.**

**How long it takes:** the project holds no precise timing record of a full build, only the
timeouts (20 minutes for the whole deploy job by default, 10 minutes maximum for the Vela stage).
What is certain is that the GCC compile-and-link stage dominates and runs on the order of minutes;
how many seconds it took on *your* machine is recorded in `deploy_report.json` after the build. Do
not expect a fixed number.

![Firmware build in progress — a progress bar and a scrolling deploy log](docs/screenshots/09_mcu_building.png)

*Figure 16: a firmware build in progress. The log below the bar shows line by line which of the
five stages above it has reached; sitting on the GCC stage for several minutes is normal.*

When the build finishes, everything lands in `models\mcu\<board name>\` under the project folder:
`firmware.bin`, a source snapshot, flashing instructions, the build record, and a ZIP of the lot.
**Retraining, or changing any sample or setting, wipes this folder too** — the firmware you built
disappears, and the page simply says "模型已變更，請重新建置。" ("the model changed; rebuild").

![The finished build — flash and memory usage summary, and a download firmware ZIP button](docs/screenshots/10_mcu_result.png)

*Figure 17: a completed build. The log ends at "部署產物已完成", and the summary below lists how
much flash and scratch memory this firmware occupies (Flash 653.9 KB / 2.00 MB, SRAM01 206.0 KB /
1.00 MB here). Below that are the flashing method and its steps, and at the bottom left, "download
firmware ZIP" — which works even with no flashing tool installed, so you can take `firmware.bin`
and flash it yourself.*

### 6.4 Choosing a flashing method

Only the GestureAI board supports two methods; the other two have one each:

- **GestureAI — USB mass-storage mode (default, nothing to install):** hold the board's User
  button, tap Reset, then release User. A drive named `M55M1` appears on your PC; drag
  `firmware.bin` onto it (or press "flash to board" in Studio, which finds the drive and copies it
  for you), then tap Reset on the board.
- **GestureAI — Nu-Link (alternative):** the same as the X board.
- **X board — Nu-Link Command Tool:** requires Nuvoton's official tool installed separately.
  Studio runs connect, write and reset in order, stopping and reporting if any step fails. You can
  also flash the same `.bin` with Keil's download function.
- **VoiceAI — pyocd plus an external Nu-Link2:** this board has no built-in debugger, so you must
  attach a Nu-Link2 to the board's J2 header. After flashing it **does not reset the board
  automatically** — that is deliberate, because the auto-reset command on this debugger tends to
  leave the board looking hung — so the page reminds you to press Reset by hand.

![After switching the flashing method to Nu-Link, the steps below change to the Nu-Link version](docs/screenshots/11_mcu_flash_method.png)

*Figure 18: the same GestureAI board with the flashing method switched from mass storage to
Nu-Link — the steps underneath are replaced wholesale by the Nu-Link version, as a comparison with
Figure 17 (USB mass-storage bootloader, four steps) shows. GestureAI is the only board with two
options; on the other two this menu has a single entry.*

Before "flash to board" runs, Studio shows a confirmation dialog naming the file size and the
target board, and re-verifies the file's hash immediately before acting, so a corrupted or altered
file is never written.

### 6.5 What you see after a successful flash

**This depends on the project kind, not the board:** only Image projects produce a picture; Audio
and Known Sound always produce text only.

- **Image:** on the X board, which has an LCD, the result (class name and score, e.g.
  "開燈:0.9876") appears on the LCD along with a frame-rate line. GestureAI has no LCD, so the same
  text is overlaid on the top-left of the USB camera image — open the built-in Windows "Camera"
  app and you will see the overlay (it is only drawn while a PC is actually reading that camera).
  Both boards also send the same line to the serial port.
- **Audio (keyword) and Known Sound:** neither board opens a camera; output goes to the serial
  port as text only. Even on the camera-equipped GestureAI board, an audio project shows no picture
  at all. That is normal — do not go looking in the camera app.

**Opening a serial console:** find the new COM port under Windows "Device Manager → Ports (COM &
LPT)", and connect with a terminal such as PuTTY or Tera Term at **115200 baud, 8 data bits, no
parity, 1 stop bit (115200 8N1)**. **Boards without an LCD (GestureAI, VoiceAI) wait at most 10
seconds for a terminal after power-up**, so open and connect your terminal *first*, then press
Reset on the board — otherwise the first burst of boot messages (model, quantization parameters
and so on) is gone before you can read it, leaving only the lines that keep refreshing.

**Reading Known Sound output:** one line every 0.5 seconds by default, in this shape:

```text
live rms  -42.1 dBFS | front   14 ms | npu    9 ms | ovr 0 | drop 0 | 槍聲=0.031  狗叫聲=0.008  玻璃破裂=0.742*
```

- `rms`: the volume of this one second in decibels; a quiet room sits around -90.
- `front` / `npu`: milliseconds spent on preprocessing and on model inference.
- `ovr`: cumulative count of lost microphone data. It should stay at 0.
- `drop`: cumulative bytes of truncated text output. It should stay at 0. (**Note: on the X board
  this goes through a real debug UART rather than a USB virtual COM port, so it is always 0 there
  and cannot be used to infer anything else.**)
- The number after each class is a **confidence score** (independent, not summing to 100%), and
  `*` marks a class that exceeded its own threshold at that instant. **The score itself is the peak
  over the last few instants (a brief "peak hold", 1.5 s by default), but the asterisk reflects
  only the current instant's raw decision** — so a high number with no asterisk is normal, not a
  bug.

**Reading Audio (keyword) output:** one line every 0.25 seconds by default, in this shape:

```text
KWS hop=12 rms=-38.4 dBFS top=開燈 0.876 frontend=28658 us inference=643 us dma_errors=0 drop=0 | 開燈=0.876* 關燈=0.101  background=0.023
```

Unlike Known Sound, **there is only ever one asterisk** (winner-takes-all), and the scores shown
are averaged over the last few instants rather than a single raw instant. When a keyword really
triggers, an extra line appears: "KWS DETECTED: label=開燈 score=0.912 hop=57".

---

## 7. When something goes wrong

| Symptom | Likely cause | What to do |
|---|---|---|
| `01_INSTALL.bat` says no compatible Python was found | Python 3.13 (64-bit, not free-threaded) is not installed | Install it from python.org and tick "Add to PATH" |
| `02_START.bat` throws an English traceback | `.venv` is incomplete | Run `01_INSTALL.bat` again. Do not delete `workspace` |
| A message says port 8765 is in use | Local Studio is already running, or another program holds the port | Look for an already-open tab; confirm with `netstat -ano \| findstr :8765` |
| The Image training summary says `small_cnn_fallback` | You were offline, so the MobileNetV2 weights could not be downloaded | Reconnect and train again |
| Known Sound / Abnormal Sound training is blocked, saying the YAMNet weights are missing | The network was down during install, so the YAMNet weights never arrived | Reconnect and re-run `01_INSTALL.bat` |
| Validation Split will not stay at 0 | A known quirk of the redraw: 0 is treated as empty and reverts to 0.2 | Leave it at 0.2 or higher. Do not force it to 0 — accuracy at 0 is meaningless anyway |
| Raising an Audio project's detection threshold changes nothing in Preview | That setting never affects Preview, only the exported script and the firmware | Export and test on a board; do not use Preview to check this setting |
| A two-class Audio project seems stuck at about 0.6 no matter how the threshold is set | The board also enforces a hidden "winner must beat runner-up by 0.2" rule, and two softmax scores always sum to 1 | Add another class (a background class, say) and the problem disappears |
| Known Sound rejects a per-class threshold you filled in | A blank field counts as 0, and both 0 and 1 are illegal (it must be strictly between them) | To use the default, clear the field and type the current default value back in — do not leave it blank |
| The firmware build says "tensor arena 超過板子預算" or that internal flash is exceeded | The model is too big (Image: size or MobileNet version; Known Sound: encoder depth; Audio: the firmware is close to full on its own, and changing settings rarely helps) | Lower the setting the message names, then retrain and re-export |
| The firmware build is refused immediately, saying the folder path is too long | The Studio folder sits on a deeply nested path | Move the whole folder somewhere short (`C:\TM_Studio`, say). No reinstall needed |
| "Flash to board" says it cannot find the drive or the tool | The matching flashing tool is not installed, or the board is not in the right mode | The 400 error message *is* the complete set of manual flashing steps — follow it, or do the one-time setup in section 6.2 |
| Occasional "file in use / PermissionError" during a build or flash | Antivirus briefly locked a freshly written file | Studio retries and usually recovers on its own; if it keeps failing, add the whole `workspace` folder to your antivirus exclusions |
| You want evidence to hand over for any problem | — | Press "下載 Log" (Download Log) in the top-right corner and give the resulting `TM_Local_Studio_Diagnostic.txt` to your teacher or maintainer. `logs\LATEST.log` and `logs\LATEST_INSTALL.log` are always kept locally too |

![The top-right toolbar — language toggle, connection status, import project and download log](docs/screenshots/18_header_tools.png)

*Figure 19: the "Download Log" mentioned in the last table row lives in this row at the top right,
alongside the language toggle, connection status and "import project". Inside a project, this row
also gains a "download project" button.*

---

## 8. What has not been verified yet

This section has to be stated plainly, because it determines whether a teacher can safely rely on
these features in class.

### 8.1 No firmware built by this Studio has ever run on a real board

What *has* been verified is that it compiles and fits. Image, Audio (keyword) and Known Sound, on
the X board and GestureAI (six combinations in total), are recompiled, relinked and size-checked
with a real `arm-none-eabi-gcc` on every test run — and they pass. But if your machine does not
have that toolchain installed, those tests are skipped silently, with nothing announcing that this
part went unverified.

**What has *not* been verified is that any of it works on the board:** whether photos are really
captured, whether the microphone really picks up sound, whether the LCD or the USB camera overlay
really displays, whether the serial port really prints, whether Nu-Link / USB mass storage / pyocd
really write the firmware — none of this has been confirmed by anyone with a physical board.
Specific unverified details include: the GestureAI second microphone (DMIC1) pin configuration;
whether GestureAI's composite USB camera + serial device enumerates correctly; the X board's debug
serial pin configuration; keyword re-trigger behaviour when there is no background class; the
10-second boot-time wait for a serial connection; and Known Sound depth-12 firmware on the X board.

### 8.2 NuMaker-VoiceAI-M55M1 needs particular caution

This board is not even in the automated test list. It has one manual build record (not using a
genuinely trained model — only to confirm that the template code compiles), was never added to the
automated tests, and re-running the tests does not re-check it. Its flashing flow
(`setup_voiceai_flash.py`) has never been executed end to end — including the download URL for
Nuvoton's official device description pack, because that file already existed locally throughout
development and nobody ever watched it download (the file-integrity verification itself is sound;
it is "the URL really resolves" that was never confirmed). **No Studio-produced firmware has ever
been written to a physical VoiceAI board.**

(One easily confused point deserves spelling out: during development, a completely separate small
verification program — not built by this Studio and never installed by it — was connected to a
physical VoiceAI board over a debug cable and confirmed the board's chip model and NPU were
healthy. That proves only that the board exists and its hardware works; it says nothing about
whether Studio-built firmware runs on it. Do not conflate the two.)

### 8.3 Known Sound's depth limits are calculated, not measured

The limits — 12 on the X board, 11 on GestureAI and VoiceAI — all come from one formula: whether
the Vela-compiled size plus a fixed firmware space reservation stays under 2 MiB of flash.
**Not one of them was measured by actually flashing a board.** The same formula applied to
GestureAI actually yields 12, but the conservative 11 was kept; VoiceAI's 11 is likewise not the
formula's answer (which is also 12) but a conservative choice to stay consistent with GestureAI.
In other words, if someone eventually tests depth 12 on a real board and it works, these limits
may rise — but as of today nobody has done that.

The one "depth vs accuracy" test that was actually run used a single dataset (3 classes, 122 clips,
a validation set of only 61 clips, and just one recording session per class). It found depths 9–11
nearly as accurate as depth 14, but depth 12 showed an unexplained anomaly (recall down to 0.66,
reproducibly across three runs) — which is exactly why depth 12 is not recommended. That table
reflects only that one dataset; it is not a universal conclusion.

### 8.4 General advice that has never been measured

Suggestions in this guide of the form "raise augmentation", "raise Dropout", "fine-tune one more
block" are reasonable directions derived from the design intent and comments in the code. They are
**not** the results of accuracy comparisons actually run in this project. Apart from the Known
Sound depth test in 8.3 and the INT8-vs-Float agreement measured at export time, this guide
contains no quantified evidence that changing a given parameter moves accuracy by any particular
number of points. Treat the advice as informed direction, not as a guarantee.

### 8.5 Other gaps worth knowing about

- **Browsers:** only Chrome was confirmed working during verification. Camera and microphone
  capture in Edge and Firefox were never separately confirmed.
- **The recording health check fires only for live recording, not for uploaded files** — which does
  not quite match some documentation describing it as running "after recording or upload". Trust
  "live recording only".
- **If Local Studio is closed mid-training** (a crash, or closing the black window): Image, Audio
  and Abnormal Sound projects detect and keep a surviving model file on the next launch, **but
  Known Sound projects are always marked as failed**, even when the model file did reach disk, and
  must be trained again. Samples themselves are never lost in any case.
- **The screenshots in this guide cover only the browser**, and use synthetic demo data rather than
  real classroom samples. Command-prompt windows (`01_INSTALL.bat` / `02_START.bat` running,
  installation failure messages), Explorer folders, physical boards and Nu-Link wiring, and serial
  terminal output all lack figures in this version — follow the text for those. Section 6.5, "what
  you see on the board", cannot even be photographed yet; see 8.1 for why.

---

If anything in this guide disagrees with what you actually see, trust the on-screen message and the
contents of `logs\LATEST.log` / `logs\LATEST_INSTALL.log` first, and use "Download Log" to send the
diagnostic file to the maintainer.
