# Known Sound Project：命名式聲音分類契約

## 產品語意

這個 project **會說出聽到的是哪一種聲音**（例如槍聲、狗叫聲、玻璃破裂），
與 `abnormal_sound` 只回答「偏離正常多少」正好相反。

```text
每類的 1 秒 clips
  → (混音增強：兩個不同類別的波形相加 → 兩類都標 1)
  → 官方 YAMNet frontend (96 × 64 log-mel patch)
  → frozen encoder（3,217,344 參數，完全不訓練）
  → 1024-D embedding      ← 整個資料集只前向一次
  → Dense(n, sigmoid)     ← 唯一可訓練層
  → 每類一個獨立分數
```

**分數彼此獨立，不會加總成 1。** 兩種聲音可以同時被偵測到。
UI、匯出 metadata 與 runner 一律稱「信心分數」，**不得**稱為「機率」——
未經 calibration 的 sigmoid 在小資料上系統性過度自信。

## 與 Abnormal Sound 的分工

| | `known_sound` | `abnormal_sound` |
|---|---|---|
| 回答什麼 | 這是**哪一種**聲音 | 跟正常差多少 |
| 要收什麼 | **每一種**聲音都要收，含 Background | 只收 Normal |
| 沒學過的聲音 | 全部低分（所以要有 Background 類） | 會警報（這是它的優點） |
| 輸出 | n 個獨立分數 | 一個純量 ratio |
| 同時事件 | 支援 | 不適用 |

要「任何怪聲都警報」用 `abnormal_sound`；要「指出是哪一種」用 `known_sound`。
兩者可以並存：異常偵測器負責未知事件，分類器負責命名已知事件。

## 資料規則（最重要的一節）

### 每類至少 2 個獨立錄音場次、至少 20 個片段

不足會**擋下訓練**並逐類說明還差什麼。這不是保守，是必要：
少於 2 個場次就無法做 session-disjoint 驗證，算出來的 accuracy 沒有意義。

### 一段 20 秒錄音切成 20 片 ≠ 20 個例子

`known_sound_pipeline.split_sessions()` 以 `recording_session_id` 分組，
**同一段錄音切出的 clips 永不跨越 train / validation**。
既有 `audio` kind 用的 `training_common.stratified_path_split()` 是 clip-level 切分，
相鄰的近重複片段會同時進訓練與驗證，報出來的 accuracy 系統性虛高。
新的音訊 kind 不要再用它。

### Background / Other 不是可選的

多標籤模型若沒有負例，會對任何聲音都給高分。
`Background` 類要收使用者真實環境的底噪：冷氣、講話、鍵盤、椅子、腳步、關門。
這一類收得好不好，直接決定系統會不會整天亂報。

### 喇叭播放的 domain gap

用喇叭播出來再錄音，模型會同時學到「喇叭的頻率響應 + 這個房間的殘響 + 播放音量」，
可能學到的是「喇叭在響」而不是「槍聲」。緩解：
不同音量／距離／房間位置各錄幾次，並盡量用「上傳檔案」加入真實錄音。

## 錄音健檢

`POST /api/projects/{id}/audio-sanity` 在錄音／上傳完成後，
用官方 YAMNet 521 類 head 回報 top-3 標籤，例如
「這段聽起來像 Speech 99% · Inside, small room 1%」。

**純資訊性**：不影響訓練、門檻、representative set 或任何已存 metadata。
缺少 `yamnet_class_map.csv` 時只有這個功能停用，其他一切照常。

它存在的理由是一次真實事故：使用者收了 62 個「異常聲音」樣本，
經官方 head 檢測**全部 62 個都是人聲**，槍聲／狗叫／玻璃三個家族分數皆為 0.000。
只回報一個純量 ratio 的異常偵測器在結構上不可能發現這件事——
它反而回報 62/62「全部偵測到」，讓人以為一切正常。

## Encoder 深度（唯一能有效縮小模型的旋鈕）

YAMNet 是 MobileNetV1 形狀，參數嚴重**後段集中**：第 14 塊自己就佔 encoder 約 33%，
第 13+14 塊合計約 49%。所以砍掉後面幾塊，模型縮小的幅度遠大於等比例。

`encoder_depth` 設定保留 YAMNet 14 個 block 的前 N 個。輸入幾何不變（仍是 96×64 patch），
所以 **frontend 契約、已收集的音訊、representative set 全部不受影響**；只有 embedding 寬度與模型大小改變。
預設 14（完整），既有專案行為不變。

### 實測：大小

以 NuMaker-GestureAI-M55M1 為目標，用 Vela 5.1.0
（`ethos-u55-256` / `Shared_Sram` / `Ethos_U55_High_End_Embedded` / arena 700 KiB）實際編譯，3 類 head：

| depth | embedding | 原始 .tflite | **Vela 後 flash** | tensor arena SRAM |
|---:|---:|---:|---:|---:|
| 14（完整） | 1024 | 3,513,960 | **2,974,368** | 151,360 |
| 13 | 1024 | 2,397,480 | 2,066,928 | 151,360 |
| 12 | 512 | 1,822,704 | 1,563,648 | 151,360 |
| 11 | 512 | 1,525,936 | 1,308,864 | 151,360 |
| 10 | 512 | 1,229,168 | 1,051,952 | 151,360 |
| 9 | 512 | 932,392 | 795,680 | 151,360 |
| 8 | 512 | 635,640 | 537,600 | 151,360 |
| 7 | 512 | 338,880 | 279,792 | 151,360 |
| 6 | 256 | 181,704 | 144,912 | 151,360 |

**SRAM 在每個深度都是 151,360 B**——峰值活化在最前面的高解析度 block，所有深度共用。
所以深度換的是 **flash，不是 RAM**。

M55M1 內部 flash 是 2,097,152 B，且**韌體與模型共用同一份 image**。
Studio 建置韌體時用的實際預留值是 262,144 B（256 KiB），對應的上限是
**X 板 depth ≤ 12、GestureAI depth ≤ 11**——算式與量測見下面的「部署到開發板」一節。

### 實測：準確率

在一組真實使用者錄音上（3 類、122 clips、session-disjoint 驗證、3 seeds 平均）：

| depth | macro precision | macro recall | 三類全對 |
|---:|---:|---:|---:|
| 14 | 0.969 | 0.983 | 0.967 |
| 11 | 0.985 | 0.983 | 0.984 |
| 10 | 0.965 | 0.983 | 0.962 |
| 9 | 0.965 | 0.983 | 0.962 |
| 8 | 0.912 | 0.983 | 0.896 |
| 7 | 0.899 | 0.983 | 0.880 |
| 6 | 0.779 | 0.956 | 0.727 |

**depth 9–11 與完整版在該資料上分不出差別**，而 depth 11 只有完整版的 44% 大小。
depth ≤ 7 開始明顯掉。

誠實標注：那個驗證集只有 61 clips、每類 1 個 session，**0.02 的差距在雜訊範圍內**，
不能宣稱「切了更準」。另外 depth 12 在該次量測出現異常（recall 掉到 0.66，三個 seed 皆然），
原因未查明，因此不建議 12。**換深度後一定要用自己的資料重新看驗證報告**，不要照抄這張表。

### 匯出檔名

非完整深度的匯出檔名會帶 `_dN`（例如 `known_sound_yamnet_classifier_d10_int8.tflite`），
避免不同深度的匯出在磁碟上分不出來。

## 部署到開發板

Known Sound 可以直接建成 M55M1 韌體（NuMaker-M55M1 X 板、NuGestureAI-M55M1 與
NuMaker-VoiceAI-M55M1 都支援）。
完整步驟見 `docs/MCU_DEPLOY_zh-TW.md`；這裡只講跟這個 kind 有關的三件事。

### 1. 深度上限：X 板 12、GestureAI 11、VoiceAI 11

2026-09-12 以真正的 Strict INT8 匯出重新量測，Vela 後的 flash 佔用是
depth 11 = 1,308,784 B、12 = 1,563,552 B、13 = 2,066,832 B、14 = 2,974,272 B。
建置前的檢查是「Vela 後大小 + 262,144 B 韌體程式碼 ≤ 2,097,152 B 內部 flash」，所以：

| depth | 加上韌體基準 | 2 MiB flash |
|---:|---:|---|
| 11 | 1,570,928 | ✅（GestureAI／VoiceAI 上限） |
| 12 | 1,825,696 | ✅（X 板上限） |
| 13 | 2,328,976 | ❌ |
| 14 | 3,236,416 | ❌ |

**depth 13 與 14 三塊板子都放不下**，只能留在電腦上跑（PC 端預設仍是 14）；
把模型放到 SD 卡再載入 HyperRAM 的做法不在這一版的範圍內。
NuGestureAI 的上限是 **11**；VoiceAI 借用同一份樣板的限制輪廓，上限同樣是 **11**，但這是
刻意保守的判斷值，不是量測結果——這塊板從未實際燒錄驗證過。「GestureAI 的 UVC／CDC 韌體比較
大、headroom 較少」這個舊說法已被實測數字推翻，完整算式與說明見
`docs\MCU_DEPLOY_zh-TW.md` 第 5 節。

選了「目標裝置」之後，訓練面板的深度選單只會列出該板放得下的值；設定存檔時
`validate_known_sound_settings()` 也會擋，訊息是
`NuGestureAI-M55M1 的 encoder_depth 上限是 11，收到 14`。

上面那張較早的量測表（`Vela 後 flash` 欄）與這裡的數字相差約 100 bytes，是兩輪不同的
量測，結論相同。深度 12 在準確率量測中出現過異常，換深度後**一定**要重看自己的驗證報告。

### 2. 每一類的門檻會被烘進韌體

訓練面板可以給每一類獨立門檻（`class_thresholds`，沒設的類別沿用「偵測門檻（預設值）」）。
部署時 `tm_local/mcu/contract.py` 會依照專案的類別順序把它們展開成一個陣列，
韌體再把每個門檻換算成 int8 輸出碼比較，不會在板子上重新做浮點運算。

改門檻要重新訓練與匯出（改設定會作廢模型），韌體才會跟著更新。

### 3. 結果只在 COM port，不會有畫面

Known Sound 韌體沒有任何顯示輸出，三塊板都一樣：

- **NuGestureAI-M55M1**：板子列舉出一個 USB 虛擬 COM port（PID 0x1105，不會開相機）。
- **NuMaker-M55M1**：板載 Nu-Link 的虛擬 COM port。
- **NuMaker-VoiceAI-M55M1**：板子列舉出一個 USB 虛擬 COM port（PID 0x1105，與 GestureAI
  共用同一份 cdc_only 韌體；這塊板本來就沒有相機）。

用 PuTTY／Tera Term 以 **115200 8N1** 連線，開機時每一類先印一行門檻，之後每 0.5 秒一行：

```text
INFO threshold[玻璃破裂] 0.50 -> output code >= 13
live rms  -42.1 dBFS | front   14 ms | npu    9 ms | ovr 0 | drop 0 | 槍聲=0.031  狗叫聲=0.008  玻璃破裂=0.742*
```

每一類是獨立的信心分數，**不會加總成 1**，和 PC 上的 Preview 語意完全相同；
分數後面的 `*` 表示這一類超過了它自己的門檻。`front`／`npu` 是前處理與推論耗時，
`ovr`（麥克風 DMA 溢位）與 `drop`（來不及送出的日誌行）正常都應該是 0。

> 尚未在真實硬體上跑過：GestureAI 的 DMIC1 channel 2 接線、USB CDC 列舉，以及 X 板的
> 除錯 UART 埠位。細節與其餘待實測項見 `docs/MCU_DEPLOY_zh-TW.md`。

## 匯出

- 融合成**單一** TFLite：`96×64 log-mel patch → n 個獨立分數`，附有意義的 `labels.txt`。
- FFT／STFT **不進 flatbuffer**；frontend 留在 host／MCU 端，與 `abnormal_sound` 同規則。
- 一律 `export_inference_saved_model()` + `from_saved_model`，禁止 `from_keras_model`。
- representative dataset **只取 train 側 sessions**，不得接觸 validation 音訊。

### 嚴格整數要求（硬性）

INT8／UINT8 必須 integer I/O、零 float tensor、零 Flex、零 custom op，
不成立即 `ExportError`，不降級交付。

### Classifier parity gate

`abnormal_sound` 比較的是 embedding **幾何**；分類器要看的是**答案有沒有變**。
主要門檻是 `detection_agreement`，分數誤差是次要防線。實測值（tone-vs-noise fixture、
44 個 representative patches、TF 2.21 Windows CPU、strict INT8）：

| | mean abs err | max abs err | detection agreement |
|---|---:|---:|---:|
| epochs=120（預設） | 0.0138 | 0.0528 | **1.000** |
| epochs=20（訓練不足） | 0.0349 | 0.1534 | 0.977 |

誤差幾乎全部集中在**中間分數**（0.2–0.8 區間平均誤差 0.055，飽和區只有 0.016），
因為 sigmoid 在那裡最陡。訓練充分的 head 會飽和，量化就很乾淨。
門檻設為 `detection_agreement ≥ 0.95`、`mean ≤ 0.08`、`max ≤ 0.25`，
兩種情況都能通過，但仍能擋下真正壞掉的轉換。

## 誠實邊界

1. **訓練資料是單標籤。** 每個 sample 只屬於一個類別；多標籤能力來自 sigmoid 與混音增強，
   不等同於在真實混合音訊上訓練。
2. **約 3.5 MiB。** 定位 Desktop／Advanced。strict INT8 通過 **不等於** 任意 MCU 可跑；
   未指定板子的 flash、SRAM arena、op resolver 與 latency 前，不使用「可上 MCU」字眼。
3. **每 patch 約 68.9 M MACs。**
4. **分數未經 calibration**，不是真實機率。
5. **驗證成績只代表「這支麥克風、這個房間、這種播放方式」下的成績。**

## 相關檔案

- 設計文件：`docs/superpowers/specs/2026-09-02-known-sound-project-kind-design.md`
- 部署到開發板：`docs/MCU_DEPLOY_zh-TW.md`
- 可執行的參考程式：`reference/known_sound/`（總覽 `reference/README_zh-TW.md`）
- 實作：`tm_local/known_sound_pipeline.py`
- 模型與 frontend：`tm_local/yamnet_model.py`
- 設定與 kind registry：`tm_local/config.py`
- 測試：`tests/test_known_sound.py`、`tests/test_distribution.py`
