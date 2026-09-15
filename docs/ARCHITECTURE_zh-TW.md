# Teachable Machine Local Studio v2.1.0 架構

## 整體流程

```text
Chrome / Edge
  ├─ Webcam / microphone capture
  ├─ Sample management
  ├─ Training controls
  ├─ Preview
  └─ Export controls
          │ HTTP 127.0.0.1:8765
          ▼
FastAPI service in Windows .venv
  ├─ ProjectStore
  ├─ Image training pipeline
  ├─ Audio log-mel training pipeline
  ├─ Abnormal Sound frozen-YAMNet pipeline
  ├─ Keras preview runtime
  ├─ TFLite conversion worker
  ├─ MCU firmware build (tm_local\mcu\)
  └─ Diagnostic log builder
          │                         │
          │                         └─▶ mcu_toolkit\  （內附的韌體樣板、BSP 子集、Vela 5.1.0）
          ▼                              + Arm GNU Toolchain（使用者自行安裝，不打包）
workspace\projects\<project-id>
  ├─ samples
  ├─ thumbnails
  ├─ project.json
  └─ models
       └─ mcu\<board>\  firmware.bin / deploy_report.json / source.zip
```

## Runtime

v2.1.0 固定使用：

```text
Windows 64-bit CPython 3.13
專案內 .venv
TensorFlow 2.21 Windows CPU runtime
```

`01_INSTALL.bat` 不會建立或呼叫 Linux 環境。`runtime/runtime_config.json` 只會選擇：

```json
{
  "selected_backend": "windows_tensorflow_cpu",
  "backend_label": "Windows CPU",
  "windows_only": true
}
```

為避免較舊教室電腦完全沒有回應，安裝器會依邏輯核心數設定 TensorFlow thread 上限，通常最多使用 8 threads，並在可行時保留一個核心給 Windows。

## Image training

```text
JPEG/PNG/WebP
  → resize / augmentation
  → MobileNetV2 transfer learning 或 small CNN
  → native .keras
  → Preview
```

## Audio training

```text
Browser-decoded mono WAV
  → resample 16 kHz
  → 1-second windows
  → 25 ms framing / 10 ms hop
  → 512-point RFFT
  → 40-bin log-mel spectrogram
  → CNN classifier
  → native .keras
  → Preview
```

## Abnormal Sound training

```text
Browser-decoded mono WAV + recording_session_id
  → resample 16 kHz / one-second stored clips
  → exact 25 ms / 10 ms / 64-mel YAMNet frontend
  → 96 × 64 patch
  → frozen 3,217,344-param YAMNet embedding encoder
  → one 1024-D mean embedding per clip
  → session-balanced robust diagonal Normal reference
  → held-out Normal threshold + independent RMS threshold
  → Keras Preview or per-runtime recalibrated TFLite Preview
```

`normal_train` 與 `anomaly_eval` 是不同資料角色。後者只能產生 evaluation report，
不能進入 reference、threshold、contamination 統計或 representative dataset。同一個
`recording_session_id` 的 clips 不會跨 train／held-out boundary。

Live Preview 每 0.5 秒評估最近一秒，browser 保存最近 5 個 window 並依 sensitivity
做 2-of-5 或 3-of-5 投票。未暖機或 calibration confidence 不足時一律回報
`Uncertain`，仍保留 raw anomaly ratio 供檢查。

## Train／Export 分離

訓練只負責 `.keras`：

```text
fit → evaluate → save .keras → 100% → Preview
```

TFLite 只在 Export 時建立：

```text
.keras
  → explicit SavedModel serving signature
  → separate conversion worker process
  → Float32 / Dynamic / INT8 / UINT8
  → validation and report
  → downloadable ZIP
```

TensorFlow Lite 的單次 `convert()` 沒有 UI 細部進度 callback，因此同一格式轉換期間百分比可能暫時維持不變。主服務會回報經過時間與 worker heartbeat，並在超時時終止 conversion worker；已訓練 `.keras` 與樣本不受影響。

## 部署到開發板（MCU 韌體）

Deploy 是第三種 job（`train` / `export` / `deploy`），和前兩者共用同一個 JobManager，
所以同一個專案同時只會有一個工作在跑。

```text
Export modal「部署到開發板」頁籤
  └─ POST /api/projects/{id}/deploy-mcu {board}
       → jobs.submit('deploy') → tm_local\mcu\deploy_service.run_deploy()

          ensure_export_artifacts(['int8'])   確保有 Strict INT8（0–35%）
            → contract.collect() / validate() 讀契約、檢查板子限制（35–40%）
            → vela.compile_model()            Performance，arena 超標改 Size（40–50%）
            → flash 預檢                       Vela flash + 256 KiB ≤ 2 MiB
            → project_builder.assemble()      樣板 + codegen + 每 kind 的 app 模組（50–55%）
            → gcc_build.build()               arm-none-eabi-gcc/ld/objcopy（55–95%）
            → package()                       firmware.bin / source.zip / ZIP（95–100%）
```

```text
tm_local\mcu\
  boards.py         讀 mcu_toolkit\boards.json 並驗證 DMIC 腳位
  toolchain.py      偵測 arm-none-eabi-gcc 與 Nu-Link Command Tool
  contract.py       DeployContract：標籤、量化、frontend、深度、每類門檻
  vela.py           Ethos-U55 編譯與 arena 預算
  codegen.py        model .cpp、Labels.cpp、arena 常數
  kws_codegen.py    KWS 的 Hann／mel 表與 main.cpp token
  project_builder.py／gcc_build.py／flash.py／deploy_service.py
  apps\{image,known_sound,kws}.py   每種 kind 的 main.cpp 與來源清單
```

板子相關的 API：`GET /api/mcu/status`（即時偵測工具鏈與板子清單）、
`POST /api/projects/{id}/deploy-mcu`、`.../deploy-mcu/download`、`.../deploy-mcu/flash`。
流程、燒錄與限制見 `docs\MCU_DEPLOY_zh-TW.md`。

## 可調參數與驗證

v2.1.0 把一組會影響準確率的訓練參數開放到 Advanced 面板。每一個控制項對應**恰好一個**
專案設定鍵，由 `tm_local/config.py` 的 `validate_image_settings()` /
`validate_audio_settings()` / `validate_known_sound_settings()` 驗證，並寫進
`training_report.settings`：

```text
image        augmentation_level / dropout / fine_tune_blocks / image_size
audio        augmentation_level / spec_augment / session_disjoint_validation /
             background_weight / background_class_id / dropout / detection_threshold / mel_bins
known_sound  waveform_augment_level / head_dropout / class_thresholds /
             background_weight / background_class_id / encoder_depth
全部          deployment_target（pc / NuMaker-M55M1 / NuGestureAI-M55M1 / NuMaker-VoiceAI-M55M1）
```

`deployment_target` 選了板子之後會**鎖住**會改變韌體輸入契約的設定（影像尺寸、音訊前端
幾何、encoder 深度上限），面板同時只顯示合法選項，伺服器也會再驗一次；部署時
`contract.validate()` 第三次檢查同樣的條件。

驗證誠實度：

- `session_disjoint_validation`（預設開啟）讓 audio 專案也以 `recording_session_id` 切分
  train／validation，同一段錄音切出的片段不會兩邊都出現。某一類只有一個錄音場次時，
  該類自動退回片段切分，並在報告的 `dataset.split_warning` 說明。
- `known_sound` 一律 session-disjoint，沒有開關。
- Image／Audio 匯出後若 INT8 與 Float 模型的 top-1 一致率低於 95%，`conversion_report`
  會加上警告並顯示在匯出結果上（不阻擋匯出）；`known_sound` 既有的硬性 parity gate 不變。

`reference\` 底下每種 kind 都有一份可執行的精簡訓練／前處理／推論腳本，
用來說明這些參數各自影響什麼。

## Log

```text
logs\LATEST_INSTALL.log  最近一次安裝
logs\LATEST.log          最近一次服務／訓練／匯出
```

網頁「下載 Log」將上述資料、Runtime、版本與最近 jobs 整合成一個文字檔，不包含圖片或錄音內容。
