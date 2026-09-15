# v2.1.0 發行說明 — Windows Only

## 新增：部署到 Nuvoton M55M1 開發板

- Export Model 視窗新增「部署到開發板」頁籤：把 Strict INT8 模型建成韌體 `firmware.bin`，
  可下載 ZIP 或直接燒錄。建置是和 Train／Export 同一套機制的背景工作。
- 支援 **Image**、**Audio（關鍵字／聲音分類）** 與 **Known Sound** 三種專案，
  目標板為 **NuMaker-M55M1（X 板）**、**NuGestureAI-M55M1** 與 **NuMaker-VoiceAI-M55M1** 三塊。
  VoiceAI 沒有相機也沒有 LCD，只支援 **Audio** 與 **Known Sound** 兩種聲音專案；
  Image 專案仍只能選 X 板或 GestureAI。
  `Abnormal Sound` 維持電腦上使用，沒有開發板韌體。
- 建置**只用 Arm GNU Toolchain（GCC）**：不需要 Keil、不需要 GNU make、不需要
  project-generator。韌體樣板、M55M1 BSP 子集與 Arm Vela 5.1.0 都內附在 `mcu_toolkit\`。
- **Arm GNU Toolchain 不隨 Studio 打包**，請自行安裝一次；`scripts\download_arm_toolchain.py`
  會下載釘死的 14.2.rel1 並用 Arm 官方的 `.sha256asc` 驗證後解壓到 `runtime\arm-gnu-toolchain\`。
  `01_INSTALL.bat` 會探測工具鏈並寫進 `runtime_config.json`，缺少時只停用部署，不會讓安裝失敗。
- **Nu-Link Command Tool 不打包**，只偵測你自己安裝的版本（X 板燒錄用）。
  NuGestureAI 用 USB 隨身碟 bootloader 拖放燒錄；VoiceAI 沒有板載 Nu-Link，改用外接
  Nu-Link2 走 `pyocd load`（不含 reset 步驟，燒完請自己按一下 Reset），一次性設定見
  `scripts\setup_voiceai_flash.py`。三種燒錄方式都不需要額外購買 Nuvoton 的付費工具。
- 韌體裡所有跟模型有關的常數（標籤、tensor arena、量化參數、KWS 的 mel／Hann 表、
  Known Sound 的每類門檻）都由匯出契約檔生成，不靠人工同步。
- Known Sound 的 `encoder_depth` 上限：**X 板 12、GestureAI 11、VoiceAI 11**（X 板與 GestureAI
  是量測後的 flash 預算結果；VoiceAI 的 11 是刻意保守的判斷值，這塊板從未實際燒錄驗證過）；
  depth 13 以上三塊板都放不下，只能留在電腦上跑。

## 新增：更多可調訓練參數

- 共同：**目標裝置**（電腦／X 板／GestureAI／VoiceAI）。選了板子之後，會改變韌體輸入契約的設定
  （Image Size、音訊前處理幾何、Encoder 深度上限）會被鎖成韌體支援的值。
- Image：資料增強強度、Dropout、MobileNetV2 微調層數（微調會讓訓練時間大約多 50%）。
- Audio：資料增強強度、SpecAugment、**Session-disjoint 驗證（預設開啟）**、背景類別與其權重、
  Dropout、偵測門檻、Mel Bins（40／64）。
- Known Sound：波形增強、分類頭 Dropout、**每一類獨立門檻**、背景類別與其權重、Encoder 深度。
- Image／Audio 匯出後若 INT8 與 Float 模型的 top-1 一致率低於 95%，會顯示黃色提醒（不阻擋匯出）。

## 新增：`reference\` 參考程式資料夾

四種專案各有一份可直接執行的精簡 Python 腳本（訓練／前處理／用匯出的 INT8 模型推論），
Audio 另有 `mcu_tables.py` 可印出韌體實際使用的 mel／Hann 表。
總覽與參數說明見 `reference\README_zh-TW.md`。

## 已知限制與尚未實測的項目

- Image 兩塊板（X、GestureAI）＋ Audio／Known Sound 三塊板（X、GestureAI、VoiceAI）共 8 種
  組合都已在電腦上建置、連結並通過大小檢查，**但都還沒有在真實硬體上跑過**。
- 待實測：非快取記憶體區段的修補、GestureAI 的 DMIC1 channel 2 接線、
  GestureAI 的 UVC + CDC 複合裝置、X 板的除錯 UART 埠位、沒有背景類別時的 KWS 重新武裝規則、
  USB CDC 開機等待、X 板 depth 12 的 Known Sound 韌體、VoiceAI 的 DMIC0（PA.4／PA.5）收音與
  pyocd 燒錄。
- 本版不做：SD 卡 + HyperRAM 載入模型（因此 `encoder_depth ≥ 13` 無法部署）、
  物件偵測（YOLOX-Nano）、Keil 專案輸出、手動指定 tensor arena。
- 詳見 `docs\MCU_DEPLOY_zh-TW.md`。

## 從 v2.0.7 更新

可以保留 `.venv` 與 `workspace`，重新執行 `01_INSTALL.bat` 即可（安裝步驟最後會多一步
MCU 工具鏈探測）。既有專案的行為不變：所有新參數的預設值都等於原本的做法，只有 Audio 的
**Session-disjoint 驗證預設開啟**會讓驗證分數變得比較誠實（通常會比舊版低一些）。

---

# v2.0.7 發行說明 — Windows Only

## 新增 Known Sound Project（YAMNet 分類器）

- 新增第四種 `known_sound` project：frozen YAMNet 1024-D encoder + 可訓練的
  `Dense(n, sigmoid)` head，**會說出聽到的是哪一種聲音**（例如槍聲／狗叫／玻璃破裂）。
- **多標籤**：每類一個獨立分數，**不會加總成 100%**，兩種聲音可以同時被偵測到。
  一律稱「信心分數」，不稱「機率」。
- 驗證改為 **session-disjoint**：同一段錄音切出的 clips 永不跨越 train／validation，
  避免既有 Audio Project 的 clip-level leakage 讓 accuracy 虛高。
- 每類至少 2 個獨立錄音場次、20 個片段，不足會擋下訓練並逐類說明還差什麼。
- **混音增強**：訓練時把兩個不同類別的波形相加，讓模型真的學到同時發生的事件。
- Preview 採 **峰值保持**（預設 1.5 秒）而非時間投票——槍聲只持續 100–200 ms，
  投票會抑制正確偵測。
- 匯出為**單一** TFLite（`96×64 log-mel patch → n 個獨立分數`）加 `labels.txt`，
  strict INT8／UINT8 維持零 float tensor、零 Flex、零 custom op，
  並新增以 `detection_agreement` 為主的分類器 parity gate。
- 新增**錄音健檢**：錄音／上傳後用官方 YAMNet 521 類 head 回報 top-3 標籤，
  純資訊性，用來在收資料當下就抓出「錄到的不是你以為的東西」。
- 新增 **`encoder_depth`**（Training 面板可選）：保留 YAMNet 14 個 block 的前 N 個。
  YAMNet 參數嚴重後段集中，因此這是唯一能有效縮小模型的旋鈕 ——
  完整 14 層 Vela 後約 2.97 MB，depth 10 只有約 1.05 MB，可放進 M55M1 的 2 MB 內部 flash。
  tensor arena 在每個深度都約 151 KB，所以換的是 flash 不是 RAM。
  預設仍是 14，既有專案行為完全不變；非完整深度的匯出檔名會帶 `_dN`。
  各深度的實測大小與準確率見 `docs/KNOWN_SOUND_zh-TW.md`。

## 新增 Abnormal Sound Project（YAMNet）

- 新增第三種 `abnormal_sound` project，不會再落入一般 Audio classifier 分支。
- 使用 SHA-256 固定的官方 YAMNet 權重、frozen 1024-D embedding 與 Normal-only scorer。
- `Normal baseline` 依 recording session 分割；`anomaly_eval` 僅評估，禁止進入訓練、threshold 與量化代表資料。
- 新增 GestureAI／DMIC 麥克風選擇、device settings 與 session metadata 保存。
- Preview 顯示 embedding／RMS anomaly ratio，採 5-window temporal vote；未達 calibration gate 時保持 `Uncertain`。
- Export 為每個 Keras／TFLite runtime 分別建立 reference、threshold、實測 FPR confidence 與 SHA binding。
- Strict INT8／UINT8 會重新檢查實際檔案，拒絕 Float、Flex、custom op、錯誤 I/O dtype 或無來源綁定的舊快取。

## 安裝流程簡化

- 移除所有 Linux／WSL GPU 安裝與提示。
- 不會要求重開機後啟動其他作業系統。
- 不會執行 `wsl.exe`、不會安裝 CUDA，也不會建立 Linux virtual environment。
- 只使用 Windows 64-bit CPython 3.13 與專案內 `.venv`。
- Runtime 固定顯示 `Windows CPU`。

## 移除的檔案

```text
requirements-wsl-gpu.txt
scripts/training_worker.py
docs/GPU_SETUP_zh-TW.md
```

## 保留功能

- Image／Audio 資料收集與訓練。
- PC 即時 Preview。
- 清楚的分類百分比和彩色 bar。
- Float32／Dynamic／strict INT8／strict UINT8 TFLite 匯出。
- Train Model 與 Export Model 分離。
- 單一「下載 Log」。
- 根目錄只有 `01_INSTALL.bat` 與 `02_START.bat`。
- Windows BAT 固定使用純 ASCII、CRLF、無 BOM。

## 舊版更新注意

從 v2.0.5／v2.0.6 更新時，可保留：

```text
.venv
workspace
```

重新執行 `01_INSTALL.bat` 後，舊的 GPU runtime 設定會自動覆寫成 Windows CPU，不需要刪除學生專案。
