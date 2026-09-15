# v2.1.0 最終整合報告：MCU 部署、可調參數、reference 資料夾

分支 `mcu-deploy`，基準 `main` @ `fe82328`（v2.0.7）。本文件是合併前的驗收依據。

---

## 1. 結論

這個分支把原本「Studio 匯出 → NuML_App_Builder → NuML_TFLM_Tool → Keil」的四段流程，
收斂成 Studio 內的一個分頁。學生按「部署到開發板」就拿到 `firmware.bin`，
不需要下載 NuML_Toolkit、不需要選 `UV4.exe`、不需要 Keil。

四個實作計畫全部完成，整分支審查的 43 項意見全部結案（36 項修正、7 項判定範圍外）。
分支已於 2026-09-13 以 fast-forward 併入 `main`。合併後又處理了兩件事：
唯一那個測試失敗查出是真缺陷並修好（第 5 節），`agast.o` 的授權改為隨附原始碼（第 7 節）。
全套測試 584 passed / 0 failed。

---

## 2. 對照你當初的六項要求

| # | 你的要求 | 交付狀態 |
|---|---|---|
| 1 | 把 NuML_TFLM_Tool 必要的部分放進 Studio，路徑固定 | `mcu_toolkit/`（vendored 子集，`manifest.json` 逐檔雜湊）。學生不再下載或選路徑。 |
| 2 | 確認 `.bin` 不用 Keil 也能編譯 | 是。純 Arm GNU Toolchain（GCC），Python 建置驅動取代 progen + make。六個 `.bin` 已實際產出。 |
| 3 | 三種 kind × 兩塊板子 | image、known_sound、audio KWS × NuMaker-M55M1、NuGestureAI-M55M1，全部可部署。 |
| 4 | GestureAI 的 UVC 疊字 + USB CDC 移植到 GCC | 已移植。image 走 UVC 複合裝置（疊字 + CDC），兩種音訊走 CDC-only。 |
| 5 | 每種 kind 一個 reference 資料夾 | `reference/<kind>/`：`train.py`、`run_tflite.py`，音訊三種另有 `preprocess.py`，KWS 另有 `mcu_tables.py`，每個資料夾一份繁中 README。 |
| 6 | 可調參數，且要自動帶到開發板 | 見第 4 節。參數經由契約檔（`training_report.json`、`audio_frontend.json`、`labels.txt`）驅動韌體生成，沒有任何手動同步點。 |

`NuML_App_Builder_v1.0` 的功能吸收情況見第 9 節。

---

## 3. 交付物

韌體位於 `workspace/deliverables/`（git 忽略，不進版本庫）：

| 檔案 | 大小 (bytes) | SHA-256 前 16 碼 |
|---|---:|---|
| `image_vww4_NuMaker-M55M1.bin` | 488,252 | `2c30c014903f4509` |
| `image_vww4_NuGestureAI-M55M1.bin` | 489,044 | `cda813f4c6be90f9` |
| `known_sound3_NuMaker-M55M1.bin` | 158,024 | `9bf9e584d6adc145` |
| `known_sound3_NuGestureAI-M55M1.bin` | 162,592 | `3a33530f8984f2b0` |
| `kws3_NuMaker-M55M1.bin` | 251,120 | `4a61f113c8cd548e` |
| `kws3_NuGestureAI-M55M1.bin` | 255,656 | `f85b35061b4a136c` |

程式碼規模：相對 `main` 為 1,750 個檔案、562,025 行新增，其中 1,616 個檔案在 `mcu_toolkit/`
（vendored 第三方樹）。Studio 自身的變更集中在 `tm_local/`（31 檔）、`tests/`（37 檔）、
`reference/`（18 檔）、`docs/`（17 檔）、`scripts/`（13 檔）、`web/`（3 檔）。

部署流程（`tm_local/mcu/deploy_service.py::run_deploy`）：
匯出 Strict INT8 → 契約 → Vela（Performance，arena 超預算改 Size）→ flash 預檢 → 組專案 → GCC → 打包。
產物含 `firmware.bin`、`source.zip`、`README_FLASH_zh-TW.txt`、`deploy_report.json`。

---

## 4. 學生可調的參數

`deployment_target`（`pc` / `NuMaker-M55M1` / `NuGestureAI-M55M1`）是總開關：選了板子，
會被鎖住的欄位自動只剩板子吃得下的值。

| kind | 影響準確率的參數 | 會被板子鎖住的 |
|---|---|---|
| image | `augmentation_level`（off/light/medium/strong）、`dropout`、`fine_tune_blocks`（僅 MobileNetV2，約 +50% 訓練時間） | `image_size` 限 96/128/160/192/224 |
| audio KWS | `augmentation_level`、`spec_augment`、`session_disjoint_validation`（預設開）、`background_class_id` + `background_weight`、`detection_threshold` | `sample_rate` 鎖 16 kHz；`mel_bins` 限 40 或 64（會改變 MCU 的 C 表） |
| known_sound | `head_dropout`、`waveform_augment_level`、每類別 `class_thresholds`、`background_class_id` + `background_weight`、`mixup_ratio` | `encoder_depth` 上限：X 板 12、GestureAI 11 |

abnormal_sound 維持桌機專用，不提供 MCU 部署。

三項與準確率有關的行為變更值得注意：

- audio 的 `session_disjoint_validation` 預設為 True，同一段錄音的切片不會同時進訓練與驗證。
  這是唯一會改變舊專案預設行為的鍵，代價是回報的 accuracy 會比舊版低而誠實。
- 匯出時若 INT8 與 Float 的 top-1 一致率低於 95%，會顯示繁中警示（image / audio）。
- known_sound 的每類別門檻會被烘進韌體，Preview 的判定與板子一致。

**深度上限 12 是實測結果，不是估計**：Vela 編譯後 flash 用量在深度 12 為 1,563,552 bytes、
深度 13 為 2,066,832 bytes；加上韌體本體基線 262,144 bytes 後，深度 13 超過 2,097,152 bytes 的內部 flash。
規格書原本寫「深度 14 在 X 板驗證過」是錯的，已在程式與文件中更正。PC 端預設仍是 14。
深度 13 以上需要 SD 卡 + HyperRAM 載入，那是規格明訂的範圍外項目。

---

## 5. 驗證結果

在合併後的 `main` 上實際執行，輸出留存於 `.superpowers/sdd/final-verify/`：

| 項目 | 結果 |
|---|---|
| `pytest` 全套（584 項，含兩塊板子的真實 GCC 韌體建置） | **584 passed / 0 failed**，708 秒 |
| `node --check web/app.js` | 通過 |
| `scripts/start_local.py --check` | 通過 |
| 版本字串（`config.TOOL_VERSION`、`pyproject.toml`、`VERSION`、`01_INSTALL.bat`、`index.html`） | 五處一致為 2.1.0 |
| 學生入口 BAT 檔數量 | 2 個（`01_INSTALL.bat`、`02_START.bat`），符合架構護欄 |
| `ruff check tm_local scripts tests` | 164 項，基準 `main`（v2.0.7）為 158 項 |

### 那個 XNNPACK 失敗已經修掉，不是「接受它」

合併當下全套是 581 passed / 1 failed，那一項
（`test_fake_cached_int8_is_reconverted_and_freshly_audited`）先前被當成環境相依的
flake。追查後發現它指向一個真實缺陷，已在合併後修正：

建立 TFLite interpreter 時，`allocate_tensors()` 會順帶配置 XNNPACK 的工作記憶體。
在存活很久的行程裡這塊配置會失敗（`failed to create XNNPACK runtime`），而失敗會落在
當下剛好要建立 interpreter 的任何地方。第一次修正把稽核路徑的 delegate 拿掉之後，
失敗就從稽核那行移到 parity 比對那行——這正好證明問題不在單一位置。

兩項修正：

1. `inspect_tflite()` 不再載入 delegate。它只讀取 tensor 與運算子資訊、從不執行推論，
   delegate 對它毫無用處。順帶讓稽核更正確：XNNPACK 會改寫圖並藏起被它吸收的 tensor
   （`kws3_int8.tflite` 上是 int8 10／int32 3 對真實的 int8 12／int32 6，運算子清單還多一個
   合成的 `DELEGATE`）。嚴格關卡問的是「這個檔案有沒有 float tensor 或 Flex 運算子」，
   所以它必須看到檔案宣告的每一個 tensor。兩種看法下 `strict_full_integer` 與
   `float_tensors` 都相同，判定沒變，只是證據變完整。
2. `predict_tflite()` 遇到 XNNPACK 那個特定錯誤時退回內建 kernel 並記警告，其他錯誤照樣往上拋。
   正常情況仍然使用 XNNPACK，所以浮點輸出與由它算出的 top-1 一致率完全不變。
   這條路徑也涵蓋 Preview（`model_runtime.py`）與 `yamnet_anomaly_pipeline`，
   也就是長時間執行的服務行程真正會累積 interpreter 的兩個地方。

修正後全套 584 passed（含兩個新增的退回機制測試），耗時從 1,027 秒降到 708 秒——
每次稽核少建一個 XNNPACK 執行環境的直接效果。退回警告在該次執行中一次也沒有觸發，
表示第一項修正已經把壓力降到不會觸發第二項的程度，第二項是保險。

**對學生的影響**：匯出本來就不受影響（轉換跑在每次重開的子行程裡），但 Preview 跑在
一直開著的主服務行程內，長時間使用後理論上會遇到同一個錯誤。這兩項修正把它從
「操作失敗」變成「慢一點但會完成」。

**ruff 的差異已逐項查清**：相對 `main` 多出的 6 項全部落在本分支新增的
`tests/test_mcu_known_sound_frontend_contract.py`，且全是 RUF046（`int(round(x))` 在 `round`
已回傳 int 時多餘）。RUF046 不在本專案 `pyproject.toml` 啟用的規則集內（它來自這台機器上的
全域 ruff 設定），而該處的轉型是刻意的防禦寫法，因此保留。分支原本另有一項 F401（未使用的
import），那條規則屬於 ruff 預設集，已在 `9d7cb2c` 移除。基準 `main` 自身的 158 項是既有債務，
本分支未增加也未處理。

---

## 6. 43 項審查意見的最終狀態

整分支審查由四位審查者分四個視角對 `31652bf` 進行，共回報 2 Critical / 13 Important /
24 Minor，加上前一輪保留的 4 項 Minor，合計 **43 項**。修正分三位實作者一輪完成（檔案互斥），
之後一次範圍限定的再審查判定 **37 addressed / 0 not addressed / 0 regressed**。

| 處置 | 數量 |
|---|---:|
| 已修正 | 36 |
| 判定範圍外（不修） | 7 |
| **合計** | **43** |

其中 2 項 Critical 都與學生直接相關：未存檔的進階設定被重繪吃掉（`A1`），
以及 vendored OpenMV 的授權聲明與實際檔案不符（`P1`）。

下表逐項列出。`A4` 是 `B1` 的另一半（說明文字那半分派給不同實作者），
`B2 half 1/2` 是同一項意見的兩半，因此 45 列對應 43 項。

| 編號 | 嚴重度 | 問題 | 處置 | commit |
|---|---|---|---|---|
| A1 | Critical | 進階訓練設定還沒按下訓練，任何一次專案重繪就會把它吃掉 | 已修正 | `b8e6286` |
| P1 | Critical | vendored 的 OpenMV 原始碼授權與 NOTICE 的「MIT」說法不符 | 擴大範圍後修正 | `1de25c8<br>3df6361` |
| A2 | Important | /train 用的是原始請求內容，不是它剛剛驗證過的設定 | 已修正 | `54020ae` |
| A3 | Important | 三條架構護欄斷言比對到的是註解，不是程式碼 | 已修正 | `b8e6286` |
| B1 | Important | audio 的偵測門檻設定在 PC 端完全沒有作用 | 已修正 | `b39e2d7` |
| B2 half 1 | Important | 指稱儲存時會接受已失效的 background class id | 審查有誤，撤銷 | `—` |
| B2 half 2 | Important | 找不到 background 類別時 background_weight 被忽略，報告卻仍說已套用 | 已修正 | `70847dd` |
| B3 | Important | KWS 的觸發策略沒有寫進匯出的 metadata.json | 已修正 | `b39e2d7` |
| B4 | Important | audio 的 INT8 校正取樣取到了保留給驗證的片段 | 已修正 | `282217b` |
| B5 | Important | 三個吃 TensorFlow 的音訊測試違反 pytest -m "not slow" 的約定 | 已修正 | `67ffa8b` |
| P2 | Important | BSP 內夾帶帶 GPL 標頭的 newlib syscalls 檔，NOTICE 卻稱整個 BSP 為 Apache-2.0 | 已修正 | `1de25c8` |
| P3 | Important | known_sound 的 YAMNet C 前端與 Python 契約之間沒有任何綁定 | 已修正 | `ac12ec4` |
| P4 | Important | 偵測門檻與每類別門檻未夾限就進入生成的韌體 C 碼 | 已修正 | `294214b` |
| P5 | Important | 建置驗證記錄寫的測試數量與同一份文件自相矛盾 | 已修正 | `b16ad3f<br>e4dd393` |
| lens A I2 | Important | 板子鎖定的錯誤訊息叫學生去改五個根本沒有控制項的音訊設定 | 已修正 | `8e800bf` |
| lens A I4 | Important | 每類別門檻的錯誤訊息報出學生從未見過的 32 位十六進位類別 id | 已修正 | `8e800bf` |
| A4 | Minor | audio 偵測門檻的說明文字承諾了 PC 端 Preview 做不到的行為 | 已修正 | `49440c7` |
| A5 M1 | Minor | 音訊的板子鎖定改寫是唯一沒有「哪裡被改了」提示的 | 已修正 | `49440c7` |
| A5 M4 | Minor | 已停用的微調下拉仍被讀取，切到 Small CNN 後 fine_tune_blocks 還留著 | 已修正 | `49440c7` |
| A5 M5 | Minor | 手寫的 mixup 比例下拉，遇到清單外的既存值會靜靜把 mixup 關掉 | 已修正 | `49440c7` |
| B1 M1 | Minor | SpecAugment 遮罩輔助函式重複了 tf.data 的抽樣，極短輸入時行為分歧 | 判定範圍外 | `—` |
| B1 M2 | Minor | image 訓練報告記錄的微調區塊數可能根本沒有執行過 | 已修正 | `70847dd` |
| B1 M3 | Minor | Preview 失敗時叫學生重新匯出 | 已修正 | `8e800bf` |
| B1 M4 | Minor | 每類別門檻的解析沒有實施它自己訊息中引用的 0 到 1 範圍 | 已修正 | `8e800bf` |
| B1 M5 | Minor | 波形增強強度遇到無法辨識的值時，靜靜退回不增強 | 判定範圍外 | `—` |
| B2-10 | Minor | 實際推論 hop 會被 DMA 區塊大小量化，但沒有說明 | 判定範圍外 | `—` |
| B2-11 | Minor | 板子的 log 等待常數默默假設每步 10 ms | 判定範圍外 | `—` |
| B2-3 | Minor | known_sound 的模型輸入形狀寫死，沒有從契約讀取 | 已修正 | `ac12ec4` |
| B2-4 | Minor | known_sound 的 clip/hop 拒絕發生在 Vela 之後，而非設定儲存時 | 判定範圍外 | `—` |
| B2-5 | Minor | 兩類別的 KWS 專案有一個隱藏的 0.60 實際觸發下限，沒有任何說明 | 擴大範圍後修正 | `49440c7` |
| B2-6 | Minor | preview_peak_hold_seconds 同時決定板子的保持時間，reference 的表格卻否認 | 改址後修正 | `7a58432` |
| B2-7 | Minor | Ethos-U 加速器設定寫死在 Vela 步驟，而不是板子屬性 | 判定範圍外 | `—` |
| B2-8 | Minor | 部署規格書仍記載無法連結成功的 known_sound 深度上限 14 | 已修正 | `7a58432` |
| B2-9 | Minor | 渲染後主程式的門檻斷言只檢查存在，沒有檢查順序 | 判定範圍外 | `—` |
| Plan4 M1 | Minor | Known Sound reference 執行腳本的 docstring 寫錯 patch 速率 | 已修正 | `7a58432` |
| Plan4 M2 | Minor | 安裝程式缺少工具鏈的提示，從未提到隨附的下載腳本 | 已修正 | `1e30bf0` |
| Plan4 M3 | Minor | 生成的 NOTICE 寫死了 sources 表格已經記載的 vendored 版本號 | 已修正 | `1de25c8` |
| Plan4 M4 | Minor | 共用的 reference 輔助模組沒有出現在任何 README 的清單列 | 已修正 | `7a58432` |
| Plan4 M5 | Minor | 燒錄確認視窗的範例引用了對不上任何實際產物的位元組數 | 已修正 | `7a58432` |
| leftover M1 | Minor | Known Sound 的 reference 訓練腳本，每類別門檻旗標是取代而非合併 | 已修正 | `7a58432` |
| leftover M2 | Minor | 沒有提醒不要對伺服器正開著的專案執行 reference 腳本 | 已修正 | `7a58432` |
| leftover M3 | Minor | 程式註解聲稱某個板子深度上限已在硬體上驗證過 | 已修正 | `8e800bf` |
| leftover M4 | Minor | 一條失效的 noqa 指示，是本分支唯一自己引入的 ruff 問題 | 已修正 | `67ffa8b` |
| lens A M2 | Minor | known_sound 驗證器只寫回新增的鍵，其餘鍵以未正規化的樣子存下 | 已修正 | `e450919` |
| lens A M3 | Minor | integer=True 把非整數值截斷，而不是拒絕 | 已修正 | `e450919` |

### 修正過程中另外發現，不屬於這 43 項

| 編號 | 嚴重度 | 問題 | 處置 | commit |
|---|---|---|---|---|
| correction #5 | Important | P1 點名的四個檔案中有三個授權判斷錯誤（只有 agast.c 是 copyleft） | 已修正 | `3df6361` |
| re-review new: libomv.a | Important | 預編譯的 OpenMV 函式庫仍夾帶 GPL 目的檔，NOTICE 卻寫 MIT | 保留給維護者 | `3df6361` |
| bonus (impl2) | Minor | known_sound 收到字串型 sample_rate 時回 HTTP 500，而非學生看得懂的 400 | 已修正 | `e450919` |
| correction #4 | Minor | 建置驗證檔宣稱有一次全綠的完整測試，但那次從未發生 | 已修正 | `e4dd393` |

### 審查本身被推翻的四處

這一輪有四次「審查說的不對」，都經查證後推翻，記錄下來是因為它們各自代表一種查證太窄的錯誤：

1. **`B2 half 1`**：說儲存時不會檢查失效的 background class id。實際上 `project_store.py`
   早就會擋並回報繁中訊息。審查只看了 `config.py`，而驗證器結構上看不到類別清單。
2. **`B2-6` 的位置**：說是瀏覽器標籤的問題。實際上那個鍵在 `web/` 只有一處讀取、沒有控制項；
   真正的錯誤在 reference 的參數表把它標成「只影響 PC 畫面」，而它其實就是板子的 peak-hold 視窗。
   改址後修正，比原本報的問題更嚴重。
3. **修正簡報自己的 `P2` 指示**：要求刪掉整個 BSP 的 GCC 目錄，理由是「沒有任何檔案被編譯或連結」。
   實際上 `retarget_GCC.c` 被三個檔案以 `#include` 當作源碼文字引入。照指示做會讓兩塊板子的
   CDC 建置與 StdDriver 全部壞掉。實作者偏離指示、只刪 GPL 的 `_syscalls.c`，並用兩塊板子的
   真實建置證明。
4. **`P1` 的授權前提**：說四個檔案是 copyleft。逐一讀標頭後發現只有 `agast.c` 是 GPL，
   其餘三個帶 OpenMV 的 MIT 標頭。刪除動作仍然正確（那 69 個檔案確實沒被引用），但理由錯了，
   已改寫為實測結果。

共同的教訓：這四次都是 grep 的範圍比結論窄。有效的檢查方式是先問「如果這個結論是錯的，
它會長什麼樣子」，然後去 grep 那個樣子。

### 判定範圍外的 7 項

`B1 M1`、`B1 M5`、`B2-4`、`B2-7`、`B2-9`、`B2-10`、`B2-11`。
每一項的理由是：在目前這個設定下不可能觸發、對所有已註冊的板子都正確、
或是已經在上一層被鎖住。共同的代價是若「鎖定的前端規格」或「板子註冊表」日後變動，
這些潛在分歧會重新出現——而那正好是會動到它們所在程式的時候。完整理由見
`docs/superpowers/handoff/final-review/final-fix-brief.md` 的 §4。

### 一項既有問題，建議排進後續

`fft_size`、`window_ms`、`sample_rate`、`fmax`、`db_floor` 這幾個鍵若以字串值送進 HTTP API，
會得到 500 而不是學生看得懂的 400。我以同一支探測程式對 `main` 與本分支各跑一次，
行為完全相同，所以這是 v2.0.7 既有的，不是本分支造成的。本分支其實修好了同類的一個實例
（known_sound 的字串 `sample_rate`，`e450919`）。瀏覽器面板送的是正確的 JSON 型別，
所以學生碰不到；用 HTTP API 的老師才會遇到。

---

## 7. `libomv.a` 內的 `agast.o`：已隨附原始碼

vendored 的 OpenMV 子集裡，預編譯靜態函式庫 `libomv.a` 含有 `agast.o`，
其原始碼是 GNU GPL v3-or-later 的 AGAST 角點偵測程式碼（Copyright (C) 2010 Elmar Mair）。
這個 `.a` 只有 image 韌體會連結（`imgclass_template` 才列入，known_sound 與 KWS 不會）。

**處置（2026-09-13 完成）**：既然散布了那個目的檔，就把它的原始碼一起散布。

- `agast.c` 逐位元從上游複製回 vendored 樹
  （`mcu_toolkit/NuML_TFLM_Tool/templates/M55M1BSP/ThirdParty/openmv/omv/imlib/agast.c`，
  sha256 `ff6cdcd1…`）。
- GPL v3 全文放在 `mcu_toolkit/LICENSES/GPL-3.0.txt`。樹裡與上游都沒有這份文字
  （上游唯一的 v3 全文是 LESSER GPL，那是不同的授權），因此標準文字保存在
  `scripts/vendor_patches/gpl-3.0.txt`，由 `licenses` 子命令複製出去並每次核對 sha256。
- vendoring 腳本以 `BSP_HEADER_ONLY_KEEP` 記下這個例外，重跑不會再把原始碼刪掉；
  若上游路徑變動會直接報錯而非默默略過。
- 兩份 NOTICE 都改寫，說明這是什麼、以及不是什麼。

**只散布、不編譯**：整個樹裡唯一提到它的地方是 `imlib.h` 的 `agast_detect()` 函式原型，
沒有任何 progen 記錄或 app overlay 指名它，而 GCC 驅動編譯的是明確的檔案清單、不掃描目錄。

**這不等於完整合規，NOTICE 也照實這樣寫**：GPL v3 要求的 Corresponding Source 還包含
Nuvoton 用來建造 `libomv.a` 的配方，那在上游、不在這個 repo 裡。要把這套工具散布到組織外之前，
仍應取得那份配方，或把 `agast.o` 從函式庫移除並重驗兩塊板子的 image 建置。
對內部教學使用，目前的狀態已從「僅揭露」提升為「附上原始碼與授權全文」。

補充一點未經證明的事實：連結器只會抽出解得開未定義符號的成員，而影像分類器沒有理由參照
AGAST 角點偵測，所以 `agast.o` 很可能根本不會進到 `firmware.bin`。要證明需要一份 link map。

## 8. 尚未上板實測的項目

程式面全部通過自動化驗證，但下列項目只做到連結或主機端驗證，需要你在實機上確認：

1. `gcc.ld` 的 `SRAM_NONCACHEABLE` 修補：只做過連結驗證。X 板的 SD 讀取與兩板的 DMIC DMA 要實測。
2. GestureAI 的 DMIC1 channel 2 接線：來源是 App Builder 的 profile，該專案自己也註明未在此板實測。
3. GestureAI 的 UVC + CDC 複合裝置在 GCC 版的列舉、疊字可見度、COM port 文字。
4. X 板 debug UART：BSP 預設 UART0（PB12/PB13），樣板註解寫 UART6。若 VCOM 無輸出要對照原理圖。
5. KWS 的 log-mel C 實作只以 numpy 重演驗證，沒有主機端編譯的 golden 比對。
   上板後請用 `reference/audio_kws/preprocess.py` 的輸出對照。
6. KWS 在沒有 background 類別時的重新待命（re-arm）邏輯。
7. CDC 終端機等待（`BOARD_LOG_WAIT`）是否真的讓開機訊息不漏字。
8. X 板深度 12 的 known_sound 實機建置與推論。
9. 未簽章的 `vela.exe`：學校電腦若啟用 AppLocker 可能被擋。

known_sound 的 YAMNet 前端有主機端 MSVC golden 比對，且新增了一個永不 skip 的綁定測試，
確保 C 的 `#define` 與 Python 契約不會悄悄漂移。

---

## 9. 與 NuML_App_Builder 的功能對照

唯讀稽核共 238 項功能：已移植 93、部分移植 66、未移植 4、改走 GCC 後不再需要 48、規格明訂排除 27。
完整清單在 `docs/superpowers/handoff/app-builder-gap-audit.md`。

完全未移植的 4 項：手動指定 Tensor Arena 大小、「開啟輸出資料夾」按鈕、
附加參考圖片／WAV 到 bundle、Known Sound 板上 golden 自測韌體。

規格明訂排除、若要追加需另開一輪規格的：SD 卡 + HyperRAM 模型載入、NuMaker-VoiceAI-M55M1 板、
Object Detection YOLOX-Nano、Keil 專案輸出、隱藏的 Motion / Generic 類型。

部分移植中，學生比較容易感覺到的差異：沒有建置前的 A/B/C/D 記憶體方案表
（改成建置流程中的連續硬性關卡，放不下會直接擋並給繁中建議）、Vela SRAM 預算與板子記憶體
不可在 UI 調整、image 沒有 Top-K 與分數門檻控制、LCD/UART/FPS 輸出一律開啟。

---

## 10. 合併狀態

`mcu-deploy` 已於 2026-09-13 以 fast-forward 併入 `main`（`main` 當時仍停在分歧點 `fe82328`，
所以沒有任何衝突），合併後在 `main` 上重跑全套確認，分支標籤隨後刪除。
分支上的每個 commit 都保留在 `main` 的歷史中。若需要還原分支標籤：

```bash
git branch mcu-deploy 616149d8c04100c0acca76857851e9487e49764c
```

合併之後的兩項修正（XNNPACK 與 GPL 原始碼）在 `post-merge-fixes` 分支上，同樣是 fast-forward：

```bash
cd <Studio 資料夾>
git checkout main
git merge --ff-only post-merge-fixes
.venv/Scripts/python.exe -m pytest -q          # 合併後再跑一次
```

本機沒有設定 remote，所以沒有 PR 流程可走。

接下來建議的第一件事，是照第 8 節把兩塊板子各燒一次三種韌體。
`workspace/` 與 `workspace/deliverables/` 都在 `.gitignore` 內，學生的樣本與既有韌體不受影響。

決策記錄（每一條 ruling、四份審查判定、修正簡報、再審查判定）在 `docs/superpowers/handoff/`，
已隨分支進入 main。原始 SDD 工作區在 `.superpowers/` 之下，是 git 忽略的暫存區，不會合併。

---

## 11. 相關文件

| 文件 | 內容 |
|---|---|
| `docs/MCU_DEPLOY_zh-TW.md` | 學生用的部署與燒錄指南 |
| `docs/MCU_AUDIO_INT8_zh-TW.md` | 音訊前處理的 MCU 端規格 |
| `docs/KNOWN_SOUND_zh-TW.md` | Known Sound 專案（含部署小節） |
| `reference/README_zh-TW.md` | reference 腳本總覽與參數表 |
| `NOTICE.md` | 第三方元件授權（含 MCU 部署元件與 `agast.o` 揭露） |
| `docs/superpowers/HANDOFF_2026-09-12_mcu-deploy.md` | 開發過程交接說明 |
| `docs/superpowers/handoff/` | 四份 ledger、四份審查判定、修正簡報、再審查判定、App Builder 對照 |
