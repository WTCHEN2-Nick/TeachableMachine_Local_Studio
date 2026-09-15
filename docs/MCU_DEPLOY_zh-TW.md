# 部署到 M55M1 開發板

把在 Studio 訓練好的模型，建成可以直接燒進 Nuvoton M55M1 開發板的韌體 `firmware.bin`。
整個過程都在本機完成：Studio 內附韌體樣板與 Vela 編譯器，學生只要自己裝一次
**Arm GNU Toolchain**（免費），不需要 Keil，也不需要另外下載 Nuvoton 的工具包。

> 這份文件描述的是 v2.1.0 的實際行為。每一個數字、訊息與檔名都對照過程式碼；
> 尚未在真實板子上跑過的部分，一律寫在最後一節「已知限制與待實測項」。

---

## 1. 支援的板子與專案類型

| 專案類型（kind） | NuMaker-M55M1（X 板） | NuGestureAI-M55M1 | NuMaker-VoiceAI-M55M1 | 韌體應用 |
|---|---|---|---|---|
| Image Project | ✅ 結果顯示在 LCD | ✅ 結果疊在 USB 相機畫面上 | ❌ 不支援（沒有相機） | `imgclass` |
| Audio Project（關鍵字／聲音分類，KWS） | ✅ 結果印在 COM port | ✅ 結果印在 COM port | ✅ 結果印在 COM port | `kws` |
| Known Sound Project | ✅ 結果印在 COM port | ✅ 結果印在 COM port | ✅ 結果印在 COM port | `known_sound` |
| Abnormal Sound Project | ❌ 不支援 | ❌ 不支援 | ❌ 不支援 | 無 |

`Abnormal Sound` 沒有開發板韌體（`tm_local/config.py` 的
`MCU_APPLICATION_BY_KIND["abnormal_sound"] = None`）。這種專案打開部署頁籤時會看到
「此專案類型無法部署到開發板。」而且按鈕是停用的；真的送出請求也會被擋下：

```text
abnormal_sound 專案不支援部署到開發板（只有 image / audio / known_sound 可以）。
```

`NuMaker-VoiceAI-M55M1` 沒有相機（`boards.json` 的 `has_camera: false`），選它的 Image
專案會被擋下**兩次**，訊息不一樣但道理相同：

1. 存訓練設定時，`tm_local/config.py` 的 `validate_image_settings()` 經 `MCU_BOARD_KINDS`
   判斷，回報「`NuMaker-VoiceAI-M55M1 不支援這種專案；請把「目標裝置」改成
   NuMaker-M55M1, NuGestureAI-M55M1 或「電腦」`」。
2. 萬一專案是先訓練完才改選這塊板、真的送出部署請求時，`tm_local/mcu/contract.py` 的
   `validate()` 再擋一次：「`NuMaker-VoiceAI-M55M1（無相機：僅聲音，USB COM port） 沒有相機，
   無法部署 image 專案`」。

網頁的開發板下拉選單也不會把它列給 Image 專案選，但那只是省得學生選錯，不是這裡真正的防線。

### 三塊板子的差別

| 項目 | NuMaker-M55M1（X 板） | NuGestureAI-M55M1 | NuMaker-VoiceAI-M55M1 |
|---|---|---|---|
| 板子說明字串 | NuMaker-M55M1（X 板：LCD、板載 Nu-Link） | NuMaker-GestureAI-M55M1（無 LCD：USB 相機 + COM port） | NuMaker-VoiceAI-M55M1（無相機：僅聲音，USB COM port） |
| 螢幕／相機 | 有 LCD | 沒有 LCD，改用 USB UVC 影像 | 都沒有，這塊板只能跑 Audio／Known Sound |
| 燒錄方式 | Nu-Link（`nulink`） | USB 隨身碟 bootloader（`msc`） | pyocd（`pyocd`）；板上沒有 Nu-Link |
| 文字輸出 | 板載 Nu-Link 的虛擬 COM port | 板子自己列舉的 USB 虛擬 COM port（PID 0x1105） | 板子自己列舉的 USB 虛擬 COM port（PID 0x1105，與 GestureAI 共用同一份 cdc_only 韌體） |
| 數位麥克風 | DMIC0（PB.4 / PB.5），channel 0 | DMIC1（PB.2 / PB.3），channel 2 | DMIC0（PA.4 / PA.5），channel 0 |
| 內部 flash | 2,097,152 bytes | 2,097,152 bytes | 2,097,152 bytes |
| SRAM01 | 1,048,576 bytes | 1,048,576 bytes | 1,048,576 bytes |
| Known Sound 深度上限 | `encoder_depth ≤ 12` | `encoder_depth ≤ 11` | `encoder_depth ≤ 11`（判斷值，非量測，見第 5 節） |

以上全部來自 `mcu_toolkit\boards.json`，程式端由 `tm_local/mcu/boards.py` 讀取與驗證。

### 板子借用樣板：原始檔抽換

`NuMaker-VoiceAI-M55M1` 沒有自己的 NuML template，直接借用 `boards.json` 的
`numl_template_board` 指到 `NuGestureAI-M55M1`；但兩塊板不是每個檔案都通用——腳位、UART 對應
不一樣的 `BoardInit.cpp` 就不能照抄。這正是 `boards.json` 的 `extra_sources` /
`remove_source_basenames` 兩個欄位存在的理由：**借用 template 的板子要換掉不適用的檔案。**
VoiceAI 用 `extra_sources` 帶一份自己的 `BoardInit_VoiceAI.cpp`
（放在 `mcu_toolkit/boards/NuMaker-VoiceAI-M55M1/`），再用 `remove_source_basenames` 把樣板原本
的 `BoardInit.cpp` 從編譯清單裡拿掉。`tm_local/mcu/project_builder.py` 的 `_overlay_kwargs()`
把這兩個欄位併進韌體 app 自己的來源清單（同一組 `add_sources` / `remove_source_basenames`，
不是另開一條路徑）；宣告要移除的檔案如果在清單裡找不到，會直接報錯，不會悄悄什麼都不做。
今天只有 VoiceAI 用到這個機制，另外兩塊板的這兩個欄位都是空的。

---

## 2. 安裝 Arm GNU Toolchain（只要做一次）

韌體是用 `arm-none-eabi-gcc` 編譯的。Studio **不附**這套工具鏈（它是 GPL-3.0 with
runtime library exception，由使用者自行安裝），所以第一次部署前要先裝。三種方式任選：

### 方式 A：用 Studio 附的下載腳本（最省事）

```bat
.venv\Scripts\python.exe scripts\download_arm_toolchain.py
```

- 下載釘死的版本 **Arm GNU Toolchain 14.2.rel1**（`mingw-w64-i686-arm-none-eabi` 的 zip）。
- 下載後會抓 Arm 官方的 `.sha256asc` 校驗檔比對 SHA-256，**不符就中止並刪掉下載檔**。
- 預設安裝到專案內的 `runtime\arm-gnu-toolchain\`，Studio 會自己找到，不必設 PATH。
- 想裝到別的地方：`--dest D:\tools\arm-gnu-toolchain`；想保留壓縮檔：`--keep-zip`。
- 這支腳本**只有學生自己執行時才會連網**，`01_INSTALL.bat` 不會自動跑它。

### 方式 B：官方安裝器

到 <https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads> 下載
`arm-none-eabi` 的 Windows 安裝器並安裝。Studio 會自動搜尋
`C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\<版本>\bin`。

### 方式 C：手動解壓 zip

把官方 zip 解開，讓 `arm-none-eabi-gcc.exe` 落在：

```text
<Studio 資料夾>\runtime\arm-gnu-toolchain\bin\arm-none-eabi-gcc.exe
```

### Studio 找工具鏈的順序

`tm_local/mcu/toolchain.py` 依序檢查，第一個能跑出 `--version` 的就用：

1. 環境變數 `TM_ARM_GCC_BIN`
2. `runtime\runtime_config.json` 裡 `mcu.toolchain.bin_dir`（安裝時記錄的）
3. 系統 `PATH` 上的 `arm-none-eabi-gcc`
4. `runtime\arm-gnu-toolchain\bin`（含 `runtime\arm-gnu-toolchain\*\bin`）
5. `C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\*\bin`（新版優先）

### 讓安裝紀錄跟著更新

重新執行 `01_INSTALL.bat`，最後一步會做一次探測：

```text
[5/5] MCU toolchain probe (optional; never fails the install)
[MCU ] Arm GNU Toolchain : <gcc --version 的第一行> (<bin 資料夾>)
[MCU ] Nu-Link Command Tool: not found (optional, NuMaker-M55M1 only)
```

沒找到時會印出下載網址與解壓目的地，**而且不會讓安裝失敗**——只是暫時不能建置韌體。
結果寫進 `runtime\runtime_config.json` 的 `mcu` 區塊（`toolchain` / `nulink` / `vela` /
`toolkit` / `probed_at`），也會出現在「下載 Log」的診斷檔裡。

### `/api/mcu/status` 與網頁上的狀態列

網頁的 Export Model →「部署到開發板」頁籤每次開啟時都會呼叫 `GET /api/mcu/status`，
**當場**跑一次 `arm-none-eabi-gcc --version`（不是讀安裝時的快照）。所以工具鏈是裝完之後
才補上的，只要切換一次頁籤就會變成可用，不必重跑安裝。缺東西時按鈕會停用並顯示：

```text
還不能部署：缺少 Arm GNU Toolchain（arm-none-eabi-gcc）。請先安裝 Arm GNU Toolchain（arm-none-eabi-gcc），再重新執行 01_INSTALL.bat。
```

同一個端點也回報板子清單、每塊板的 `known_sound_max_depth`，以及這個版本真正能部署的
kind（`deployable_kinds`）——訓練面板的深度選單就是拿這個值來限制的。

---

## 3. 部署流程

### 學生看到的步驟

```text
Train Model
  → Export Model
  → 勾選 Quantized INT8 並匯出一次（或直接進下一步，部署時會自動補做）
  → Export modal 切到「部署到開發板」頁籤
  → 選開發板
  → 按「建置韌體」
  → 進度條 + 編譯訊息
  → 「下載韌體 ZIP」或「燒錄到板子」
```

建置是一個和 Train／Export 同一套機制的背景工作（`job_type = "deploy"`），同一個專案
同時只能有一個工作在跑，否則會得到：

```text
此專案已有訓練、匯出或部署工作進行中。
```

關掉 Export 視窗不會中斷建置；重新打開就能接回進度。

### 伺服器實際做的事

`tm_local/mcu/deploy_service.py` 的 `run_deploy()`，括號內是進度條的區間：

| 步驟 | 做什麼 |
|---|---|
| 1. 確保 Strict INT8 匯出（0–35%） | 呼叫既有的 `export_service.ensure_export_artifacts(["int8"])`，缺就重做一次 |
| 2. 讀部署契約（35–40%） | `tm_local/mcu/contract.py` 收集標籤、量化參數、frontend 契約、每類門檻、背景類別索引、mel bins、encoder 深度，並檢查板子限制 |
| 3. Vela 編譯（40–50%） | `mcu_toolkit\vela\vela-5_1_0.exe`，先 `--optimise=Performance`，arena 超出預算再用 `--optimise=Size` 重跑一次 |
| 4. flash 預檢 | Vela 報的 flash 用量加上韌體程式碼基準值，超過 2 MiB 就**在編譯前**擋下並說明要調哪個設定 |
| 5. 組裝專案（50–55%） | `project_builder.py` 複製對應板子的韌體樣板，生成 `Model/` 下的模型與 `Labels.cpp`，再由 `tm_local/mcu/apps/{image,known_sound,kws}.py` 寫出 `main.cpp` 並掛上該板需要的驅動／USB CDC 來源 |
| 6. GCC 編譯（55–95%） | `tm_local/mcu/gcc_build.py` 直接呼叫 `arm-none-eabi-gcc/ld/objcopy/size`，不用 make、不用 Keil、不用 project-generator |
| 7. 打包（95–100%） | 產生 `firmware.bin`、原始碼快照 `source.zip`、燒錄說明與 `deploy_report.json`，再包成一個可下載的 ZIP |

逾時上限由 `TM_LOCAL_DEPLOY_TIMEOUT` 控制，預設 1200 秒。
建置暫存資料夾在 `workspace\tmp\mcu\<8 碼>\`，不論成敗都會刪掉（`build.log` 會先複製出來）。

### 產物放在哪裡

```text
workspace\projects\<專案 id>\models\mcu\<板子名稱>\
  firmware.bin                    要燒進板子的檔案
  firmware.elf / firmware.map / firmware.hex
  build.log                       完整編譯紀錄（路徑已去識別化）
  labels.txt                      「索引 類別名」一行一個
  README_FLASH_zh-TW.txt          這塊板子的燒錄步驟與「會看到什麼」
  source.zip                      這次建置的韌體原始碼快照
  deploy_report.json              板子、模型 sha256、Vela 結果、flash/SRAM 用量、bin sha256
  <專案名>_<板子>_firmware.zip     按「下載韌體 ZIP」拿到的就是它
```

下載的 ZIP 裡有：`firmware.bin`、`firmware.elf`、`firmware.map`、`build.log`、
`deploy_report.json`、`labels.txt`、`README_FLASH_zh-TW.txt`、`source.zip`。

> **重新訓練會刪掉韌體。** 任何樣本或設定變動都會作廢已訓練模型並清空整個 `models\`，
> 韌體與 `deploy_report.json` 一起消失。網頁這時會顯示「模型已變更，請重新建置。」
> 這是刻意的：留著一份對不上目前模型的 `.bin` 才危險。

---

## 4. 燒錄與看結果

燒錄前網頁會先確認一次，內容包含檔名、大小與板子：

```text
即將把 firmware.bin（<實際位元組數> bytes）燒錄到 NuMaker-GestureAI-M55M1（無 LCD：USB 相機 + COM port），確定要繼續？
```

> 上面的位元組數只是示意用的佔位符：實際大小會隨模型、設定與工具鏈版本而變動，
> 請以網頁當下顯示的數字與 `deploy_report.json` 為準，不要拿本文件的數字去對答案。

### NuGestureAI-M55M1：USB 隨身碟 bootloader

1. 按住板上 **User 按鈕（PA.4）**，同時按一下 **Reset**，再放開 User。
2. 電腦會出現名稱為 **M55M1** 的隨身碟。
3. 把 `firmware.bin` 拖放到該隨身碟（或直接按 Studio 的「燒錄到板子」——它會自動找卷標為
   `M55M1` 的可移除磁碟並複製過去）。
4. 拖放完成後按一下 **Reset**，新韌體就會啟動。

找不到隨身碟時，Studio 會把上面這段步驟原樣顯示出來，而不是只說「失敗」。

### NuMaker-M55M1（X 板）：Nu-Link

1. 用 USB 線接板上的 **Nu-Link 埠**。
2. 若電腦已安裝 **Nu-Link Command Tool**，按「燒錄到板子」就會依序執行
   `NuLink.exe -C`（連接）→ `-W APROM firmware.bin 1`（寫入）→ `-S`（重置）。
3. 也可以用 Keil 的 Download 功能燒同一個 `.bin`。

Nu-Link Command Tool **不包含在 Studio 裡**，只會偵測你自己安裝的版本，預設路徑是
`%ProgramFiles(x86)%\Nuvoton Tools\NuLink Command Tool\M55M1_M5531\NuLink.exe`；
裝在別處可用環境變數 `TM_NULINK_EXE` 指定。沒有安裝也不影響建置與下載 ZIP。

### NuMaker-VoiceAI-M55M1：pyocd

1. 這塊板**沒有板載 Nu-Link**。把外接 **Nu-Link2**（CMSIS-DAP）接到板上的 **J2**
   （PF.0 ICE_DAT / PF.1 ICE_CLK）。
2. 執行一次安裝腳本，裝好 pyocd 與晶片支援檔：
   ```bat
   .venv\Scripts\python.exe scripts\setup_voiceai_flash.py
   ```
   之後按 Studio 的「燒錄到板子」就會自動寫入。
3. 也可以用外接 Nu-Link2 走 Keil 的 Download：Nu-Link 的 **Chip Select 必須選 M5531**，
   選 M55M1 會永遠跑不完。（本工具包附的 Nu-Link Command Tool 不認得這顆晶片，會回
   `Target Chip is Not Supported`，所以 Studio 不會用它燒這塊板。）

`scripts\setup_voiceai_flash.py` 只有真的擁有這塊板的學生需要執行：它把 **pyocd**
（Apache License 2.0）安裝進專案自己的 `.venv`，並把 Nuvoton 的 `NuMicroM55_DFP` CMSIS-Pack
下載到 `runtime\numicro-pack\`（先驗證 SHA-256 才落地）；`01_INSTALL.bat` 不會自動執行它，
Studio 的基本安裝也不隨附 pyocd 或這個 pack，這個 pack 也不會被這個專案轉發散布。

裝好之後 Studio 執行的燒錄指令固定是
`pyocd load --pack <pack 路徑> -t M55M1R2LJAE --base-address 0x00100000 firmware.bin`，
**刻意不含 reset 步驟**：`pyocd reset -m hw` 在 Nu-Link2 上會讓 nRESET 卡在 assert 狀態，
看起來就像板子當機了，所以燒完後請自己按一下 **Reset**。

沒有偵測到 pyocd 或 device pack 其中之一時，Studio 會把上面完整的燒錄步驟（1～3 步，包含
第 2 步要跑的安裝腳本）原樣顯示出來，而不是只說「失敗」：`tm_local/app.py` 的燒錄端點在這種
情況下直接把 `flash_instructions()` 回傳的整段文字當成錯誤訊息，並不會只挑其中幾步。

### 燒完之後會看到什麼

這一段**看專案類型**，不是看板子：只有 Image 專案有畫面，而 VoiceAI 完全沒有相機，
永遠只有文字輸出。

| 專案類型 | X 板 | GestureAI | VoiceAI |
|---|---|---|---|
| Image | LCD 顯示辨識標籤；Nu-Link 虛擬 COM port 印出 `INFO - 標籤:0.9876` | Windows「相機」app 看到 USB 攝影機影像，左上角疊上辨識標籤；裝置管理員會多一個 COM port 同步印文字 | ❌ 這塊板沒有相機，不支援 Image 專案 |
| Audio（KWS） | 只有文字：Nu-Link 虛擬 COM port | 只有文字：板子列舉的 USB 虛擬 COM port（PID 0x1105，不會有影像視窗） | 只有文字：板子列舉的 USB 虛擬 COM port（PID 0x1105；這塊板沒有相機，不會有影像視窗） |
| Known Sound | 同上 | 同上 | 同上 |

COM port 一律是 **115200 8N1**：在 Windows「裝置管理員 → 連接埠 (COM 和 LPT)」找出對應的
COM 編號，用 PuTTY／Tera Term 連線。

Known Sound 韌體開機時每一類先印一行門檻，之後每 0.5 秒印一行分數：

```text
INFO threshold[玻璃破裂] 0.50 -> output code >= 13
live rms  -42.1 dBFS | front   14 ms | npu    9 ms | ovr 0 | drop 0 | 槍聲=0.031  狗叫聲=0.008  玻璃破裂=0.742*
```

每一類是**獨立的信心分數**，不會加總成 1；分數後面的 `*` 表示這一類超過了它自己的門檻。
`ovr` 是麥克風 DMA 溢位次數、`drop` 是來不及送出的日誌行數，兩個都應該是 0。

Audio（KWS）韌體每 0.25 秒印一行，判定成立時多印一行：

```text
INFO - KWS hop=12 rms=-38.4 dBFS top=開燈 0.876 frontend=28658 us inference=643 us dma_errors=0 drop=0 | 開燈=0.876* 關燈=0.101  background=0.023
INFO - KWS DETECTED: label=開燈 score=0.912 hop=57
```

這一行裡有兩個數字每次都該是 **0**：`dma_errors` 是麥克風 DMA 錯誤次數，`drop` 是日誌來不及送出而被丟掉的位元組數。`drop` 不是 0 就表示你看到的文字可能缺字——數字本身就是缺了多少。

分數後面的 `*` 標出這個 hop 判給哪一類。KWS 是單一贏家（argmax），所以永遠只有一個 `*`；這和 Known Sound 不同，那邊每一類各有自己的門檻，可以同時好幾個 `*`。

> 使用 USB 虛擬 COM port 的板子，韌體開機後會**最多等 10 秒**讓終端機連上再開始印字
> （`NUML_BOARD_LOG_WAIT_MS`），所以開機訊息不會因為 PuTTY 還沒開好而消失；沒有終端機
> 也不會卡住不開機。

---

## 5. 記憶體、大小與常見錯誤

### 硬性上限

| 資源 | 數值 | 誰在檢查 |
|---|---|---|
| 內部 flash（韌體＋模型共用） | 2,097,152 bytes | Vela 之後的預檢、連結器、連結後複檢 |
| SRAM01（tensor arena 與非快取緩衝區共用） | 1,048,576 bytes | 連結器與連結後複檢 |
| Vela tensor arena 預算 | 716,800 bytes | `vela.compile_model()` |
| 韌體程式碼預留（估計值，三塊板共用） | 262,144 bytes（256 KiB） | `deploy_service.FIRMWARE_CODE_BASELINE_BYTES` |

Vela 先用 `--optimise=Performance`；如果算出來的 arena 超過 716,800 bytes，就自動改用
`--optimise=Size` 再編一次（換速度換空間）。兩次都超過才會失敗。

flash 預檢的算式很單純：**Vela 後的模型大小 + 262,144 ≤ 2,097,152**。
這個檢查放在 GCC 之前，就是為了不要讓學生等兩分鐘才看到一句英文的連結器錯誤。

### 常見錯誤訊息與處置

| 訊息（開頭） | 意思 | 怎麼辦 |
|---|---|---|
| `找不到 Arm GNU Toolchain（arm-none-eabi-gcc）。` | 工具鏈還沒裝好 | 見第 2 節；裝好後重跑 `01_INSTALL.bat` 或重新啟動 Studio |
| `請先完成訓練，再部署到開發板。` | 還沒訓練 | 先按 Train Model |
| `INT8 模型不是全整數圖，無法部署到 Ethos-U` | 匯出的 INT8 不是嚴格整數 | 重新 Export；嚴格 INT8 是硬性條件 |
| `模型經 Vela 編譯後佔用 … bytes，加上韌體程式碼約 262,144 bytes，超過 … 的內部 flash` | 模型太大 | 照訊息後半段調整：Known Sound 降 `encoder_depth`、Image 降 `image_size`／MobileNet alpha 或減類別 |
| `模型的 tensor arena 需要 … bytes，超過板子預算 716,800 bytes` | 活化記憶體太大 | 同上；通常是 `image_size` 太大 |
| `韌體連結失敗：模型的 tensor arena 需要 … 超過 … 的 SRAM01 空間（1,048,576 bytes）` | 連結期才爆的 SRAM | 同上 |
| `韌體連結失敗：程式碼加上模型權重超過 … 的 flash` | 連結期才爆的 flash | 同上 |
| `NuMaker-M55M1（X 板：LCD、板載 Nu-Link） 的 encoder_depth 上限是 12，目前是 14；請降低後重新訓練` | 深度超過該板上限 | 在訓練面板把「Encoder 深度」調到上限以內，重新訓練與匯出 |
| `Studio 資料夾路徑太長（… 字元），請把整個資料夾搬到較短的路徑（例如 C:\TM_Studio）再重新啟動` | Windows MAX_PATH | 把整個 Studio 資料夾搬到短路徑 |
| `此專案已有訓練、匯出或部署工作進行中。` | 同專案已有工作 | 等它跑完 |

失敗時 `build.log` 會留在 `models\mcu\<板子>\build.log`，網頁的建置訊息框也會顯示最後一段。
再搭配右上角「下載 Log」即可回報。

### Known Sound 的深度怎麼挑

`encoder_depth` 是唯一能有效縮小模型的旋鈕（YAMNet 的參數嚴重集中在最後幾個 block）。
2026-09-12 用真正的 Strict INT8 匯出重新量測，Vela 後的 flash 佔用：

| depth | Vela 後 flash | 加上 262,144 韌體基準 | 2 MiB flash |
|---:|---:|---:|---|
| 11 | 1,308,784 | 1,570,928 | ✅ |
| 12 | 1,563,552 | 1,825,696 | ✅ |
| 13 | 2,066,832 | 2,328,976 | ❌ 放不下 |
| 14 | 2,974,272 | 3,236,416 | ❌ 放不下 |

所以：

- **`depth ≥ 13` 三塊板都不能部署。** 要用完整 14 層就只能留在電腦上跑（PC 端預設仍是 14）。
  把模型放到 SD 卡再載入 HyperRAM 的做法不在這一版的範圍內。
- X 板上限 **12**，GestureAI 上限 **11**（兩個上限都是依照 flash 預算換算出來的，尚未上板
  實測）。**已被推翻的舊說法：** 早期認為 GestureAI 的 11 是因為「它的 UVC／CDC 韌體比較大，
  headroom 較少」——這個理由已被下面的推算否定：把同一條算式套在 GestureAI 自己量到的韌體
  大小上，得到的其實也是 12，不是 11。GestureAI 的上限**維持 11 不變**（不在這次任務調整
  範圍內），只是不再用「韌體比較大」這個理由解釋。VoiceAI 上限也是 **11**，但這個數字不是
  同一條算式的直接產物：把算式套在 VoiceAI 自己量到的韌體大小上其實推得出 12，但反過來把
  算式套回 GestureAI 已經量過的數字，也還原不出 GestureAI 現有的 11——算式本來就沒辦法反推
  出樣板出借方（GestureAI）既有的選擇。與其相信一個連既有答案都推不出來的公式，不如讓借樣板
  的一方（VoiceAI）對齊出樣板的一方，兩塊 CDC-only 板先停在同一個數字。之後真的拿 VoiceAI 板
  實測過 depth 12 沒問題，這個上限可以再調高。
- tensor arena 在每個深度都約 151 KB，**深度換的是 flash，不是 RAM**。

（`docs/KNOWN_SOUND_zh-TW.md` 裡那張表是較早一輪的量測，數字與上表相差約 100 bytes，
結論相同。準確率的比較也在那份文件裡。）

---

## 6. 哪些訓練設定會被開發板鎖住

訓練面板最上面的 **「目標裝置」**（`deployment_target`）就是這個鎖的開關：
選「電腦（不限制）」時什麼都不限制；選了一塊板子之後，面板只會列出該韌體支援的選項，
伺服器也會在存設定時擋下不合法的值（`tm_local/config.py` 的 `validate_*_settings()`）。

| 專案類型 | 選了板子之後被鎖住的設定 |
|---|---|
| Image | `image_size` 只能是 96、128、160、192、224 |
| Audio（KWS） | `sample_rate` 16000、`clip_seconds` 1.0、`window_ms` 25、`hop_ms` 10、`fft_size` 512、`fmin` 20、`fmax` 8000、`db_floor` −80 |
| Known Sound | `encoder_depth ≤` 該板上限（X 板 12；GestureAI、VoiceAI 皆為 11） |

`mel_bins` 只能是 **40 或 64**，這一條**不分目標裝置**：訓練的 CNN、匯出的濾波器組與韌體
的 C 表都只有這兩種寬度。

部署時 `contract.validate()` 會**再檢查一次**同樣的條件——因為專案可能是在「電腦」目標下
訓練完，之後才改選板子的。這時錯誤訊息會直接指出要改哪個設定並重新訓練。

### 板子上的常數一律「生成」，不手動同步

韌體裡所有跟模型有關的常數都是部署當下從匯出檔算出來的，沒有任何一個是人工填的：

| 韌體裡的東西 | 來源 |
|---|---|
| `Model/Labels.cpp`（類別名稱與數量） | 專案的 classes 順序 |
| 模型陣列 `nn_model[]` | Vela 編譯後的 `.tflite` |
| `ACTIVATION_BUF_SZ`（tensor arena） | Vela summary CSV 的 SRAM 用量 |
| 量化 scale／zero point、輸入形狀 | `models\conversion_report.json` |
| KWS 的 Hann 視窗與稀疏 mel 濾波器表 | `models\audio_frontend.json` → `tm_local/mcu/kws_codegen.py` |
| KWS 觸發參數（門檻、平滑、margin、RMS gate） | 訓練報告的 `settings` |
| Known Sound 每一類的門檻 | 訓練設定的 `class_thresholds`（沒設的類別用預設門檻） |
| 背景／環境音類別索引 | 訓練設定的 `background_class_id`（留空則依名稱自動判斷） |
| YAMNet 96×64 前處理契約 | `models\yamnet_frontend.json`，與內建契約逐項比對 |

想看韌體裡那些 C 表長什麼樣，可以直接印出來：

```bat
.venv\Scripts\python.exe reference\audio_kws\mcu_tables.py --frontend <匯出ZIP>\audio_frontend.json
```

其他 kind 的訓練／前處理／推論參考程式都在 `reference\` 底下，
總覽見 `reference\README_zh-TW.md`。

---

## 7. 目前驗證到哪裡

三種 kind × 兩塊板（共 6 種組合）都已經用真實的 Arm GNU Toolchain 建置成功並通過連結期的
記憶體檢查，也就是說「編得出來、放得下」是驗證過的。
**但是沒有任何一種組合在真實硬體上跑過。**

| 組合 | 已驗證（在電腦上） | 尚未驗證（要上板才知道） |
|---|---|---|
| Image × X 板 | GCC 建置、連結、大小檢查 | LCD 顯示、相機影像、UART 文字 |
| Image × GestureAI | GCC 建置、連結、大小檢查；CDC retarget 的符號綁定 | UVC 列舉、疊字是否看得到、CDC 文字 |
| Audio KWS × X 板 | GCC 建置；log-mel 表與訓練前端以 numpy 逐元素比對、量化後差異 ≤ 1 LSB | DMIC 收音、UART 埠位、觸發行為 |
| Audio KWS × GestureAI | 同上 | DMIC1 channel 2 接線、USB CDC 文字 |
| Known Sound × X 板 | GCC 建置；C 前端與 Python 前端的 golden 向量比對（需要 MSVC，否則跳過） | DMIC 收音、UART 埠位、門檻行為 |
| Known Sound × GestureAI | 同上 | DMIC1 channel 2 接線、USB CDC 文字 |

### NuMaker-VoiceAI-M55M1（2026-09-13 加入，驗證方式不同、也更輕）

上面六種組合都被 `tests/test_mcu_known_sound_build.py` 等測試檔在每次跑 pytest 時，用真實
`arm-none-eabi-gcc` 重新建置一次；VoiceAI **不在**這份自動化清單裡
（該檔的 `BOARD_NAMES` 目前只有 `NuMaker-M55M1` 與 `NuGestureAI-M55M1`）。它目前只有：

- **一次性的手動建置驗證（2026-09-13）**：用真實 `arm-none-eabi-gcc` 手動建置並連結成功
  known_sound 與 KWS 韌體（沒有實際訓練模型，只是驗證借來的樣板換上
  `BoardInit_VoiceAI.cpp` 之後編得起來、連得起來）。量到的韌體本身大小（`.bin` 扣掉內嵌的
  模型陣列）：known_sound 150,080 B、KWS 238,584 B，都遠低於 262,144 B 的韌體基準值。這次
  建置**沒有**被收進自動測試套件，重跑 `pytest` 不會重新驗證它。
- **mock 過的燒錄邏輯單元測試**（`tests/test_mcu_flash.py`）：只驗證 `pyocd load` 的指令組
  得對、找不到 pyocd／device pack 時擋得下來；從來沒有真的呼叫過 pyocd，也沒有連過任何板子。

**沒有任何東西燒進過實體 VoiceAI 板。** DMIC0（PA.4 / PA.5）收音、USB CDC 是否真的印得出字、
pyocd 燒錄是否真的能把 `firmware.bin` 寫進板子，都還沒有人拿真板子確認過。

---

## 8. 已知限制與待實測項

### 還沒在硬體上確認的事

1. **`gcc.ld` 的 `SRAM_NONCACHEABLE` 修補只做過連結驗證。** DMA 用的非快取緩衝區位址正確
   與否，要實際跑 DMIC／相機才知道。
2. **GestureAI 的 DMIC1 channel 2 接線**只來自上游的板子設定檔，上游自己也註明沒有在該板
   實測過。兩個音訊 kind 在 GestureAI 上收不到聲音的話，第一個要查的就是這裡。
3. **GestureAI 的 UVC + CDC 複合裝置**在 GCC 版是第一次組出來：列舉、疊字可見度、COM port
   是否穩定出現，都還沒實測。
4. **X 板的除錯 UART 埠位**沿用上游現況（BSP 預設 UART0 / PB.12、PB.13，樣板註解寫 UART6）。
   若虛擬 COM port 沒有輸出，要對照板子的原理圖確認。
5. **沒有背景類別的 KWS 專案**（例如只有 `yes` / `no` 兩類）改用「贏家與第二名的差距」當
   margin、條件不成立就立刻解除觸發；這條規則只有邏輯推導與單元測試，沒有實際說話測過。
6. **USB CDC 開機等待**（最多 10 秒）沒有在真實 Windows 列舉流程上測過。
7. **X 板 depth 12 的 Known Sound 韌體**是照公式判定放得下，尚未在板子上跑過。
8. **VoiceAI 板從來沒有實際燒錄驗證過。** GCC 建置與連結是真的做過一次（2026-09-13，手動
   驗證，見第 7 節），但 DMIC0（PA.4 / PA.5）收音、USB CDC 是否真的印得出字、pyocd 燒錄是否
   真的能把韌體寫進板子，都還沒有人拿真板子試過。
9. **`scripts\setup_voiceai_flash.py` 的下載網址從來沒有被真的連線過。** 這個開發環境連不到
   GitHub，至今每一次執行（開發期間與自動測試）都因為本機已經有這個 pack 而直接走「已安裝，
   略過下載」那條路徑，`_fetch_pack()` 裡真正發出 HTTP 請求的那一段程式碼從未被實際執行過。
   `PACK_URL`、`PACK_SHA256`、`PACK_BYTES` 三個值是照著開發機上已安裝的那份 pack 抄錄、
   再對照 Arm 官方 pack 索引裡這個版本的描述資料核對過的，不是靠一次成功的下載驗證出來的。
   SHA-256 檢查本身沒有被放鬆：真的下載失敗或檔案不符，`_fetch_pack()` 一樣會擋下來。

### 這一版**不做**的事

- **不支援 SD 卡 + HyperRAM 載入模型**：模型一律放在內部 flash，因此 `encoder_depth ≥ 13`
  無法部署。沒有「放不下就自動改用 SD」的選項。
- **VoiceAI 不支援 Image 專案**：這塊板沒有相機，Image 專案在存設定與部署時都會被擋下
  （見第 1 節）；Audio（KWS）與 Known Sound 不受影響。
- **不支援物件偵測（YOLOX-Nano）**：Studio 沒有物件偵測的專案類型。
- **不輸出 Keil 專案**：只走 GCC。Keil 可以用來燒 `.bin`，但不是必需品，Studio 也不會產生
  `.uvprojx`。
- **不能手動指定 tensor arena 大小**：arena 一律由 Vela 的結果算出來。
- **沒有 A／B／C／D 記憶體方案表**：改用「Vela arena 預算 → flash 預檢 → 連結器 → 連結後
  複檢」四道依序把關；哪一道擋下來，訊息就會說要調哪個設定。
- **韌體的輸出內容固定**：Image 只報 Top-1，音訊類印出全部類別分數；沒有 Top-K 或顯示項目
  的開關。

---

## 9. 相關檔案

| 用途 | 位置 |
|---|---|
| 板子註冊表 | `mcu_toolkit\boards.json` |
| 韌體樣板與第三方元件 | `mcu_toolkit\`（授權見 `mcu_toolkit\NOTICE_third_party.md`） |
| 部署流程 | `tm_local\mcu\deploy_service.py` |
| 契約與板子限制 | `tm_local\mcu\contract.py` |
| Vela / GCC / 燒錄 | `tm_local\mcu\vela.py`、`gcc_build.py`、`flash.py` |
| KWS 的 C 表生成 | `tm_local\mcu\kws_codegen.py` |
| 音訊前處理的 MCU 規格 | `docs\MCU_AUDIO_INT8_zh-TW.md` |
| Known Sound 的資料規則與深度取捨 | `docs\KNOWN_SOUND_zh-TW.md` |
| 可執行的參考程式 | `reference\README_zh-TW.md` |
