# Known Sound Project 參考腳本

用 Studio 自己的程式，在命令列看 YAMNet 前處理、重跑訓練、用 INT8 模型對一段聲音打分。
邏輯全部在 `tm_local/known_sound_pipeline.py` 與 `tm_local/yamnet_model.py`。

**這是 multi-label（多標籤）模型**：每個類別各自一個 sigmoid，分數彼此獨立、
不會加總成 1，也不是校正過的統計量，所以一律稱「信心分數」。
兩個聲音可以同時發生（狗叫的同時玻璃破掉），這正是它與 Audio(KWS) 的差別。

## 資料流

```
WAV（16 kHz mono，1 秒片段，帶 recording_session_id）
      │  yamnet_model.waveform_to_log_mel_patches()
      │    25 ms 視窗／10 ms 位移、512 點 FFT、64 個 mel（125-7500 Hz）
      │    log(mel + 0.001) → 96 × 64 的 patch，每 0.48 秒一張
      ▼
  凍結的 YAMNet encoder（只保留前 encoder_depth 個 block）→ 嵌入向量
      │  （深度 14 = 官方完整 encoder，嵌入 1024 維）
      ▼
  Dropout(head_dropout) → Dense(n, sigmoid)   ← 只有這層在訓練
      ▼
  models/<專案>.keras      ← Train 到此為止
      ▼  （Export Model，是另一個 job）
  known_sound_yamnet_classifier[_d<深度>]_int8.tflite ＋ labels.txt
  ＋ yamnet_frontend.json ＋ training_report.json（含每類門檻）
      ▼  （「部署到開發板」頁籤）
  Vela → flash 預算檢查 → 組專案 → GCC → firmware.bin
```

**驗證一律 session-disjoint**（`split_sessions()`，`KNOWN_SOUND_VAL_SESSION_FRACTION = 0.25`）：
同一段錄音切出來的片段是近乎重複的，若讓它們橫跨訓練與驗證，報出來的準確率會系統性虛高。
每類至少要 2 個獨立 session，否則 Train 會直接拒絕並說明缺什麼。

## 可調參數

預設值出自 `tm_local/config.py` 的 `KNOWN_SOUND_DEFAULTS`，
合法範圍出自 `validate_known_sound_settings()`。

| 參數 | 預設 | 範圍 | 怎麼影響準確率 | 改了會牽動開發板嗎 |
|---|---|---|---|---|
| `encoder_depth` | `14` | 2 ~ 14 | 保留幾個 YAMNet block。淺一點省很多 flash（YAMNet 的參數集中在最後幾個 block），但表達力下降；tensor arena 幾乎不受影響 | **是**（`model-shape`。`KNOWN_SOUND_BOARD_MAX_DEPTH`：NuMaker-M55M1 最多 `12`、NuGestureAI-M55M1 最多 `11`、NuMaker-VoiceAI-M55M1 最多 `11`；X 板與 GestureAI 是實測 flash 上限，VoiceAI 是刻意保守的判斷值，這塊板從未實際燒錄驗證過） |
| `head_dropout` | `0.0` | 0 ~ 0.5 | Dense 之前加一層 Dropout（`name="head_dropout"`）。只影響訓練，可訓練參數量不變 | 否（`training`） |
| `waveform_augment_level` | `off` | `off`／`light`／`medium`／`strong` | 在**波形**上做增益 ±3／6／9 dB、循環時移 ±50／100／200 ms、加噪 SNR 30／20／10 dB，複本 1／2／3 份；只增強訓練 session，驗證永遠是真實錄音 | 否（`training`） |
| `mixup_ratio` | `0.5` | ≥ 0 | 把兩個**不同類別**的片段相加，做出「同時發生」的訓練樣本（單標籤收集做不出來的情況）。0 關閉 | 否（`training`） |
| `background_class_id` | `""` | 類別 id | 指定背景／沒有目標聲音的類別 | 否（但會寫進報告與韌體邏輯） |
| `background_weight` | `1.0` | 0.25 ~ 4 | 背景類在加權 BCE 內的倍率；誤報多就調高 | 否（`training`） |
| `class_thresholds` | `{}` | `{類別 id: 0~1}` | 每一類自己的判定門檻。罕見但重要的警報聲可以壓低門檻，常見背景音可以提高。沒設定的類別退回 `detection_threshold` | 是（`runtime`。匯出與部署時從 `training_report.json` 的 `class_thresholds_by_label` 取出，寫進韌體門檻陣列） |
| `detection_threshold` | `0.5` | 0 ~ 1（不含端點） | 共用門檻（舊版單一門檻的後備值） | 是（`runtime`） |
| `preview_peak_hold_seconds` | `1.5` | ≥ 0 | Preview 的峰值保持秒數；槍聲只有 100-200 ms，沒有保持就看不到 | 是（`runtime`，同時決定韌體的峰值保持時間） |
| `epochs` | `120` | ≥ 1 | head 只有 `1024 × 類別數 + 類別數` 個參數，而且 encoder 只前向一次，所以高回合數也只要幾秒 | 否 |
| `batch_size`／`learning_rate` | `32`／`0.001` | ≥ 1／> 0 | 常見訓練參數 | 否 |
| `early_stopping` | `true` | true／false | 驗證分數不再進步就停 | 否 |
| `minimum_sessions_per_class` | `2` | ≥ 2 | 低於 2 就無法 session-disjoint 驗證，Train 會拒絕 | 否 |
| `minimum_clips_per_class` | `20` | ≥ 1 | 每類最少片段數 | 否 |
| `hop_seconds`／`clip_seconds`／`sample_rate` | `0.5`／`1.0`／`16000` | 固定 | YAMNet 契約，收集、訓練、Preview、匯出必須一致，不可改 | **是**（`frontend-contract`） |
| `deployment_target` | `pc` | `pc`／`NuMaker-M55M1`／`NuGestureAI-M55M1`／`NuMaker-VoiceAI-M55M1` | 不影響學習，但會限制 `encoder_depth` | **是**（`lock`） |

> 專案設定裡還有一組 `mel_bins`、`window_ms`…（98 × 40 的那套）。那是**收集端**用來切片與畫縮圖的，
> 模型吃的是上面 96 × 64 的官方 YAMNet 前處理，兩者不要混在一起。

## 怎麼跑

```bat
rem 只看設定
.venv\Scripts\python.exe reference\known_sound\train.py --project "我的聲音專案" --dry-run

rem 訓練：淺一點的 encoder（NuGestureAI／VoiceAI 上限 11）＋ 波形增強
.venv\Scripts\python.exe reference\known_sound\train.py --project "我的聲音專案" ^
    --encoder-depth 11 --waveform-augment-level medium --head-dropout 0.2

rem 幫某一類單獨設門檻（可重複；會整批取代 class_thresholds，見下方說明）
.venv\Scripts\python.exe reference\known_sound\train.py --project "我的聲音專案" ^
    --class-threshold <class_id>=0.35 --class-threshold <class_id>=0.7

rem 看 YAMNet patch 長什麼樣
.venv\Scripts\python.exe reference\known_sound\preprocess.py --wav dog.wav --png dog.png

rem 用 INT8 模型打分（加上 --report 就用訓練當下的每類門檻判定）
.venv\Scripts\python.exe reference\known_sound\run_tflite.py ^
    --model known_sound_yamnet_classifier_d11_int8.tflite --labels labels.txt ^
    --input dog.wav --report training_report.json
```

`--class-threshold`是**整批取代**，不是合併：上面的例子只列出兩個類別，執行後
`class_thresholds` 就只剩這兩筆，專案裡任何其他類別原本設定的門檻都會遺失、
改退回 `detection_threshold`。要保留其他類別的門檻，得把所有要保留的類別都
一起列在同一次呼叫的 `--class-threshold` 裡。

`run_tflite.py` 會對整段音訊的每一張 patch 推論，然後取每類的**最大值**：
只響 150 ms 的敲擊聲若取平均就被稀釋掉了。

## 對應的程式

| 這裡做的事 | Studio 的程式 |
|---|---|
| YAMNet 前處理與契約 | `tm_local/yamnet_model.py`：`waveform_to_log_mel_patches()`、`frontend_contract()`、`encoder_layer_name()`、`encoder_output_dim()`、`build_yamnet_models()` |
| session 切分與資料門檻 | `tm_local/known_sound_pipeline.py`：`split_sessions()`、`check_data_gates()` |
| 增強與 mixup | `tm_local/known_sound_pipeline.py`：`build_waveform_augmentations()`、`build_mixup_examples()`、`background_sample_weights()` |
| 建模與訓練 | `tm_local/known_sound_pipeline.py`：`build_known_sound_model()`、`train_known_sound_project()` |
| 門檻解析 | `tm_local/known_sound_pipeline.py`：`resolve_class_thresholds()`、`thresholds_for_labels()` |
| 匯出的 PC 執行器 | `tm_local/known_sound_pipeline.py`：`write_known_sound_runner()`、`export_inference_saved_model()` |
| 參數預設與驗證 | `tm_local/config.py`：`KNOWN_SOUND_DEFAULTS`、`validate_known_sound_settings()`、`KNOWN_SOUND_BOARD_MAX_DEPTH` |
| 上開發板 | `tm_local/mcu/deploy_service.py`：`run_deploy()`、`tm_local/mcu/codegen.py` |

設計文件：`docs/KNOWN_SOUND_zh-TW.md`。
