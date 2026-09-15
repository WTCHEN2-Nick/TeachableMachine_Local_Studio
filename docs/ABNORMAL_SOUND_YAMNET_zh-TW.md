# Abnormal Sound Project：YAMNet 實作契約

## 產品語意

這個 project 偵測「聲音偏離使用者收集的正常基準」，不是把聲音分類成軸承、漏氣、
撞擊等已命名類別。YAMNet 官方 521 類分類 head 不參與推論；使用的是 frozen
1024-D embedding encoder。

```text
Normal recordings
  → frozen YAMNet embeddings
  → session-balanced robust Normal reference
  → held-out Normal threshold

new audio
  → embedding distance ratio ┐
                              ├→ max ratio → temporal vote
  → RMS level ratio ──────────┘
```

## 資料角色與 readiness

- `Normal baseline` 是唯一的 `normal_train` container，不能改名或刪除。
- `Anomaly Examples` 與使用者新增的群組都是 `anomaly_eval`，只能衡量
  `detected / total`，不會影響模型、reference、threshold 或量化代表資料。
- 每次按住錄音及每個獨立上傳檔會建立自己的 `recording_session_id`。同一 session
  產生的 clips 永遠在同一個 split。
- 最低 training gate：3 個獨立 Normal sessions、合計 60 秒，且 train／held-out
  兩側都要有資料。這時可以看 anomaly ratio，但 calibration confidence 可能仍是低。
- 二元 verdict gate：至少 6 個 sessions、120 秒、2 個 held-out sessions、40 個獨立
  audit clips、沒有尚未處理的 high-review clips，且 95% false-positive upper bound
  必須符合 sensitivity 要求。未達時 API 與 UI 保持 `Uncertain`。

正常資料應涵蓋實際會出現的正常轉速、負載、距離、時間、背景與麥克風位置；只延長
同一次錄音不會增加 session 數，也不能證明跨工況穩定。

## 固定 YAMNet frontend

| 項目 | 契約 |
|---|---|
| PCM | mono，16,000 Hz |
| STFT | 25 ms periodic Hann，10 ms hop，FFT 512，magnitude |
| Mel | 64 bands，125–7,500 Hz |
| Log | `log(mel + 0.001)` |
| Patch | 96 × 64，patch hop 48 frames |
| Model output | 1024-D embedding |
| Clip aggregation | 同一 clip 多 patch 時取 mean，之後 L2 normalize |
| Detection window | 固定 1.0 秒，固定 0.5 秒 hop |
| File tail | 只評估 0.5 秒網格上的完整 window，不追加重疊的 off-grid 尾窗 |

Local Studio 的產品路徑只在音訊短於第一個完整 patch 時補零；不為最後一個不完整 hop
額外製造大半靜音的 patch。Project settings 也會拒絕改動 1.0／0.5 秒幾何；完整契約
會寫入 `yamnet_frontend.json`，Preview、Export 與 runner 遇到缺檔或內容不符時會停止。

## Scorer 與 threshold

每個 recording session 先各自估 robust center／within-session scale，再以 session median
合併，避免一段特別長的錄音支配模型。1024 維 scale 使用 shrinkage diagonal estimate；
小資料不直接估 1024×1024 full covariance。最終比率為：

```text
embedding_ratio = standardized_embedding_distance / T_shape
level_ratio     = abs(RMS_dBFS - C_level) / T_level
anomaly_ratio   = max(embedding_ratio, level_ratio)
```

Live Preview 以最近 5 個 window 做 temporal vote；Balanced 與 Low false alarm 為 3-of-5，
Sensitive 為 2-of-5。前 4 個 window 是 warm-up，不會提前發出 alarm。

## 模型資產與完整性

官方 YAMNet HDF5 權重固定在：

```text
assets\yamnet\yamnet.h5
SHA-256 13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744
```

`01_INSTALL.bat` 會預先下載及核對；Train 不會臨時下載。完整 tagger 為 3,751,369
params，實際保存的 frozen embedding core 為 3,217,344 params。

每個 runtime scorer entry 綁定 model filename、model SHA-256、reference、threshold 與
representative digest，再建立 binding SHA-256。Preview 與匯出 runner 會驗證 model 與
binding；整數 conversion cache 也必須同時符合 source Keras SHA、TFLite SHA 與 exact
representative tensor digest。檔案被替換、資料改變、混用 runtime 或手動改 scorer 時，
系統會拒絕重用或推論。

## Train、Preview、Export

Train 只完成：

```text
abnormal_sound_yamnet.keras
yamnet_scorer.json          （Keras reference / threshold）
yamnet_frontend.json
training_report.json
contamination_report.json
```

Export 才在隔離 worker 產生選定 TFLite，並用該量化 runtime 的 embedding 重新擬合
Normal reference 與 threshold。ZIP 同時包含 scorer、frontend reference、runner、metadata
與 conversion audit；不會輸出把 `anomaly_eval` 群組冒充分類 label 的 `labels.txt`。

Strict INT8／UINT8 會強制整數 input/output、`TFLITE_BUILTINS_INT8`、零 Float tensor、
零 Flex/custom op，並要求 Keras／TFLite embedding cosine 與 relative-L2 門檻通過。
YAMNet INT8 embedding core 已在本機實測可轉；但約 3.3 MiB flatbuffer、
frontend FFT、activation arena、實板 latency 與連續 throughput 仍須依指定 MCU 驗證。
因此匯出成功代表「strict-integer artifact 可供評估」，不等於任意 MCU-ready。

## 已知限制與正確驗證

- 25 ms／64-mel YAMNet frontend 可能無法保留極窄機械 sideband；需要另外做
  richer／multi-resolution frontend A/B，不能只靠換 backbone 宣稱解決。
- AudioSet 分類準確率不是 anomaly 準確率。應以相同 GestureAI DMIC、session-disjoint
  資料，比較 Dense AE＋RMS 與 YAMNet＋RMS 的 Normal false alarms/hour、固定 FPR recall、
  各 anomaly group detection rate、跨日／轉速／負載／位置穩定性。
- YAMNet 是本版唯一開放的 Abnormal Sound backend，定位為 Desktop／高階裝置。
  小型 Dense AE 僅保留為後續 Compact MCU 的研究比較基線，未在本版 UI／API 開放；
  在指定板子的 flash、SRAM、op resolver 與 latency 通過前，不宣稱 YAMNet 可直接上板。
