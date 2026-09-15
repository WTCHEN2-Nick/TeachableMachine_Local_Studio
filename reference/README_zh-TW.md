# reference/ — 看得見、跑得動的教學腳本（Teachable Machine Local Studio v2.1.0）

這個資料夾裡的腳本**不是另一套實作**。每一支都只是把 Studio 自己用的函式叫出來，
讓你可以在瀏覽器之外，一步一步看清楚「收集 → 前處理 → 訓練 → 匯出 → 上開發板」到底發生什麼事。
真正的邏輯全部在 `tm_local/`，腳本只負責解析參數、印出結果。

所有指令都用專案內的 `.venv`，而且在 Studio 根目錄執行：

```bat
.venv\Scripts\python.exe reference\image\train.py --project "我的圖片專案" --dry-run
.venv\Scripts\python.exe reference\audio_kws\preprocess.py --wav yes.wav --png yes.png
.venv\Scripts\python.exe reference\known_sound\run_tflite.py --model model_int8.tflite --labels labels.txt --input dog.wav
.venv\Scripts\python.exe reference\audio_kws\mcu_tables.py --frontend audio_frontend.json --out kws_tables.h
```

`--project` 可以給專案名稱或 id；打錯時腳本會把現有專案列出來。
每支 `train.py` 都有 `--dry-run`：只印出「這次會用哪些設定」就結束，不會載入 TensorFlow，一秒就跑完。

> **不要對 Studio 網頁當下開著的專案跑這些腳本。** Studio 服務（`02_START.bat`）
> 把每個專案的 `project.json`、`training` 狀態與 `models\` 都當成只有自己會寫入；
> 這裡的腳本是另一個獨立行程，如果網頁還開著同一個專案，兩邊同時寫入可能互相
> 覆蓋，甚至讓 `models\` 卡在訓練或匯出到一半的狀態。跑腳本前，請先在網頁切到
> 別的專案、回首頁，或直接關掉 Studio 服務。

| 資料夾 | 專案種類（kind） | 腳本 |
|---|---|---|
| `image/` | `image` | `train.py`、`run_tflite.py` |
| `audio_kws/` | `audio`（關鍵字辨識） | `train.py`、`preprocess.py`、`run_tflite.py`、`mcu_tables.py` |
| `known_sound/` | `known_sound`（多標籤聲音辨識） | `train.py`、`preprocess.py`、`run_tflite.py` |
| `abnormal_sound/` | `abnormal_sound`（異常偵測，僅 PC） | `train.py`、`preprocess.py`、`run_tflite.py` |

`reference/_common.py` 不屬於任何一個 kind：四個資料夾底下的腳本都從它匯入
`open_store`／`find_project`／`require_kind`、`add_common_args`／`add_tunable_args`／
`collect_overrides`（把 `--dry-run` 表格與命令列參數接上 `tm_local/config.py` 的
tunables）、`print_settings`／`run_training`，以及 `run_tflite.py` 共用的
`load_interpreter`／`quantize_input`／`dequantize_output`／`print_scores` 等推論小工具。
import 它的當下就會設好 `TF_USE_LEGACY_KERAS` 等環境變數、UTF-8 主控台輸出，並把
Studio 根目錄加進 `sys.path`；每支 `train.py` 自己的 `bootstrap()` 只是在直接用檔案
路徑執行、還沒 `cd` 進資料夾時，先讓這次 import 找得到它。它本身不含任何一種
kind 的訓練邏輯。

---

## 共通的資料流

```
收集樣本                前處理                 模型                 匯出                開發板
workspace/projects/  →  tm_local/*_frontend  →  .keras（Train）  →  INT8 .tflite    →  Vela → 韌體
  samples/<class_id>/    image_pipeline          Preview 可用        （Export 是      （只有 image／
                         audio_frontend                               另一個 job）      audio／known_sound）
                         yamnet_model
```

兩件事情要先記住：

1. **Train 與 Export 是兩個獨立的工作。** 訓練只產生 `.keras`，Preview 立刻可用；
   INT8 `.tflite` 要按 Export Model 才會生成。`reference/*/train.py` 也一樣只做到 `.keras`。
2. **任何樣本或設定的變動都會作廢已訓練的模型**（`ProjectStore._invalidate_training()`）。
   用腳本改設定時也一樣：改完就要重訓。

---

## 參數的四種「效果類別」

每支 `train.py --dry-run` 印出來的表格，最後一欄就是這個分類：

| 效果類別 | 意思 | 改了之後開發板端會怎樣 |
|---|---|---|
| `training` | 只影響這次怎麼學（增強、Dropout、權重、提早停止…） | 模型形狀不變，重新 Export + 重新部署即可 |
| `model-shape` | 改變模型的輸入或大小（`image_size`、`encoder_depth`…） | 部署時會重新跑 Vela、重新檢查 flash，再重新編譯韌體 |
| `frontend-contract` | 改變「訊號怎麼變成數字」（log-mel 幾何） | PC 與 MCU 必須同時改；`mel_bins` 會讓 C 表整個換掉 |
| `runtime` | 只改判定門檻／平滑（不動模型權重） | 韌體只換掉幾個常數 |
| `lock` | `deployment_target`：選了開發板，其他設定的合法範圍就被鎖住 | 不合法時 Export／部署會直接擋下並說明要改哪一項 |

合法範圍與預設值只有一份來源：`tm_local/config.py`
（`IMAGE_DEFAULTS`、`AUDIO_DEFAULTS`、`KNOWN_SOUND_DEFAULTS`、`ABNORMAL_SOUND_DEFAULTS`，
以及 `validate_image_settings()`／`validate_audio_settings()`／`validate_known_sound_settings()`／
`validate_abnormal_sound_settings()`）。腳本與這份文件都不另外複製一份數字。

---

## 1. Image（`reference/image/`）

```
JPG/PNG → ImageOps.fit 到 image_size×image_size（RGB 0-255）
        → MobileNetV2（alpha=0.35，凍結）或 small_cnn
        → Dropout → Dense(n, softmax)
```

| 參數 | 預設 | 範圍 | 對準確率的影響 | 牽動開發板？ |
|---|---|---|---|---|
| `augmentation_level` | `medium` | `off`／`light`／`medium`／`strong` | 翻轉、亮度、對比（strong 另有縮放、旋轉、色相）。資料少、環境單一時調高比較不會過擬合 | 否（`training`） |
| `dropout` | `0.2` | 0 ~ 0.6 | 訓練分數遠高於驗證分數時調高 | 否（`training`） |
| `fine_tune_blocks` | `0` | 0 ~ 4 | 解凍 MobileNetV2 最後 N 個 block 再以 1/10 學習率微調；**只有 `mobilenet_v2` 有效**，`small_cnn` 會被忽略並在訓練訊息中說明 | 否（`training`） |
| `image_size` | `224` | 96-320（32 的倍數） | 小尺寸較快、較省 flash，細節多的題目會掉準 | **是**（`model-shape`；上板只能是 `MCU_IMAGE_SIZES` = 96/128/160/192/224） |
| `epochs` | `30` | 1 ~ 300 | 太少學不完，太多會過擬合 | 否 |
| `early_stopping` | `true` | true／false | 驗證分數不再進步就停，通常保持開啟 | 否 |
| `batch_size`／`learning_rate`／`validation_split` | `16`／`0.001`／`0.20` | 1-128／1e-6-0.1／0-0.45 | 常見的訓練三件套；不確定就別動 | 否 |
| `backbone`／`mobilenet_alpha` | `mobilenet_v2`／`0.35` | `mobilenet_v2`／`small_cnn`；0.35/0.5/0.75/1.0 | alpha 越大越準也越大 | **是**（`model-shape`） |

對應程式：`tm_local/image_pipeline.py`（`train_image_project()`、`_build_model()`、`_make_dataset()`、
`fine_tune_layer_names()`、`load_image_for_prediction()`）。

```bat
.venv\Scripts\python.exe reference\image\train.py --project <名稱> --augmentation-level strong --dropout 0.3
.venv\Scripts\python.exe reference\image\run_tflite.py --model image_classifier_int8.tflite --labels labels.txt --input test.jpg
```

---

## 2. Audio / KWS（`reference/audio_kws/`）

```
WAV 16 kHz mono 1 秒 → log-mel（98 × 40 × 1，值域 0-1）→ 小型 CNN → Dense(n, softmax)
```

| 參數 | 預設 | 範圍 | 對準確率的影響 | 牽動開發板？ |
|---|---|---|---|---|
| `augmentation_level` | `medium` | `off`/`light`/`medium`/`strong` | 在已算好的 log-mel 上做時移、增益、加噪（medium = 位移 1/12 frame、±10%、噪聲 0.012）；前處理本身不動 | 否（`training`） |
| `spec_augment` | `false` | true／false | 在頻譜上隨機遮掉約 1/10 的時間與頻帶，讓模型不要只背某一段 | 否（`training`） |
| `session_disjoint_validation` | `true` | true／false | **預設開啟**：驗證片段不會與訓練片段來自同一段錄音。關掉會讓報表上的準確率系統性虛高 | 否（但會改變你看到的分數） |
| `background_weight` | `1.0` | 0.25 ~ 4 | 背景類的權重；常常誤觸發就調高 | 否（`training`） |
| `background_class_id` | `""` | 類別 id | 指定哪一類是「沒有人說話」；沒指定時用名稱關鍵字猜（`background`、`silence`、`背景`…） | 是（韌體用它當抑制與 margin 基準） |
| `dropout` | `0.2` | 0 ~ 0.6 | 過擬合時調高 | 否 |
| `detection_threshold` | `0.5` | 0.05 ~ 0.99 | 觸發門檻；只改判定，不用重訓 | 是（`runtime`，燒進韌體常數） |
| `mel_bins` | `40` | 40 或 64 | 頻率解析度；64 比較細但模型與 C 表都變大 | **是**（`frontend-contract`） |
| `sample_rate`／`clip_seconds`／`window_ms`／`hop_ms`／`fft_size`／`fmin`／`fmax`／`db_floor` | 16000／1.0／25／10／512／20／8000／-80 | 上板時等於 `MCU_AUDIO_FRONTEND_LOCK` | 這是 PC 與 MCU 的共同契約，改了就對不起來 | **是**（`frontend-contract`，選了開發板就鎖死） |
| `epochs`／`batch_size`／`learning_rate`／`validation_split`／`early_stopping` | 40／16／0.001／0.20／true | 1-300／1-128／1e-6-0.1／0-0.45 | 常見訓練參數 | 否 |

韌體端另有一組執行參數（`tm_local/config.py` 的 `KWS_RUNTIME_DEFAULTS`，由
`tm_local/mcu/kws_codegen.py` 的 `kws_tokens()` 燒進韌體）：
`inference_hop_seconds` 0.25 秒、`smooth_windows` 4、`required_hits` 2、`trigger_margin` 0.20、
`rearm_score` 0.40、`rms_gate_dbfs` -55 dBFS、`log_epsilon` 1e-10（必須與訓練前處理一致，不可改）。

對應程式：`tm_local/audio_frontend.py`（`log_mel_spectrogram()`、`mel_filterbank()`、
`spectrogram_thumbnail()`）、`tm_local/audio_pipeline.py`（`train_audio_project()`、
`background_class_index()`）、`tm_local/mcu/kws_codegen.py`（`kws_tables()`、`kws_tokens()`）。

```bat
.venv\Scripts\python.exe reference\audio_kws\train.py --project <名稱> --mel-bins 64 --spec-augment true
.venv\Scripts\python.exe reference\audio_kws\preprocess.py --wav yes.wav --png yes.png
.venv\Scripts\python.exe reference\audio_kws\run_tflite.py --model audio_classifier_spectrogram_int8.tflite --labels labels.txt --input yes.wav
.venv\Scripts\python.exe reference\audio_kws\mcu_tables.py --frontend audio_frontend.json
```

---

## 3. Known Sound（`reference/known_sound/`）

```
WAV 16 kHz → YAMNet log-mel patch（96 × 64）→ 凍結的 YAMNet encoder（保留 encoder_depth 個 block）
          → Dropout(head_dropout) → Dense(n, sigmoid)
```

這是 **multi-label**：每個類別都有自己獨立的 sigmoid，分數不會加總成 1。
UI、匯出的 metadata 與這裡的腳本一律稱「信心分數」。

| 參數 | 預設 | 範圍 | 對準確率的影響 | 牽動開發板？ |
|---|---|---|---|---|
| `encoder_depth` | `14` | 2 ~ 14 | 保留幾個 YAMNet block。幾乎是唯一會改變模型大小的旋鈕（深度越淺 flash 越小，實測表在 `config.py` 的註解） | **是**（`model-shape`；`KNOWN_SOUND_BOARD_MAX_DEPTH`：NuMaker-M55M1 最多 12、NuGestureAI-M55M1 最多 11、NuMaker-VoiceAI-M55M1 最多 11——VoiceAI 是刻意保守的判斷值，尚未實測） |
| `head_dropout` | `0.0` | 0 ~ 0.5 | Dense 之前的 Dropout；只影響訓練，參數量不變 | 否（`training`） |
| `waveform_augment_level` | `off` | `off`/`light`/`medium`/`strong` | 在**波形**上做增益 ±3/6/9 dB、時移 ±50/100/200 ms、加噪 SNR 30/20/10 dB，複本 1/2/3 倍；只增強訓練 session | 否（`training`） |
| `class_thresholds` | `{}` | `{類別 id: 0~1}` | 每一類各自的判定門檻；沒設定的類別退回 `detection_threshold` | 是（`runtime`，寫進韌體的門檻陣列） |
| `detection_threshold` | `0.5` | 0 ~ 1（不含端點） | 共用門檻（`class_thresholds` 的後備值） | 是（`runtime`） |
| `background_class_id` + `background_weight` | `""` + `1.0` | 類別 id；0.25 ~ 4 | 背景類在加權 BCE 內的倍率；誤報多就調高 | 否（`training`） |
| `mixup_ratio` | `0.5` | ≥ 0 | 把兩個不同類別的片段相加，做出「同時發生」的訓練樣本 | 否（`training`） |
| `preview_peak_hold_seconds` | `1.5` | ≥ 0 | Preview 的峰值保持秒數，短促的聲音才看得見 | 是（`runtime`，同時決定韌體的峰值保持時間） |
| `epochs`／`batch_size`／`learning_rate`／`early_stopping` | 120／32／0.001／true | — | head 只有幾千個參數，訓練很快 | 否 |
| `minimum_sessions_per_class`／`minimum_clips_per_class` | 2／20 | ≥ 2／≥ 1 | 資料量門檻；不足時 Train 會直接拒絕並說明缺什麼 | 否 |

驗證一律 **session-disjoint**（`known_sound_pipeline.split_sessions()`）：同一段錄音切出來的
片段永遠不會同時進訓練與驗證，這是這個 kind 存在的主要理由之一。

對應程式：`tm_local/yamnet_model.py`（`waveform_to_log_mel_patches()`、`frontend_contract()`、
`encoder_layer_name()`）、`tm_local/known_sound_pipeline.py`（`train_known_sound_project()`、
`split_sessions()`、`resolve_class_thresholds()`、`thresholds_for_labels()`、
`write_known_sound_runner()`）。

```bat
.venv\Scripts\python.exe reference\known_sound\train.py --project <名稱> --encoder-depth 11 --waveform-augment-level medium
.venv\Scripts\python.exe reference\known_sound\preprocess.py --wav dog.wav
.venv\Scripts\python.exe reference\known_sound\run_tflite.py --model known_sound_yamnet_classifier_d11_int8.tflite --labels labels.txt --input dog.wav --report training_report.json
```

---

## 4. Abnormal Sound（`reference/abnormal_sound/`，只有 PC）

```
WAV 16 kHz → YAMNet log-mel patch（96 × 64）→ 凍結 encoder → 1024 維嵌入
          → 與「正常」的中心／尺度比較 → 偏離比值（形狀分支）＋ 音量分支
```

這是 **open-set 偏離偵測**，不是分類器：只收 Normal 音訊，輸出只有偏離程度，
**永遠不會說出那是什麼聲音**。`tm_local/config.py` 的 `MCU_APPLICATION_BY_KIND` 把它對應到
`None`，所以**沒有開發板版本**，部署面板也會直接說「此專案類型無法部署到開發板」。

| 參數 | 預設 | 範圍 | 對判定的影響 | 牽動開發板？ |
|---|---|---|---|---|
| `sensitivity` | `balanced` | `sensitive`／`balanced`／`low_false_alarm` | alpha 與投票規則：0.10（2/5）／0.05（3/5）／0.02（3/5） | 不適用（僅 PC） |
| `level_tolerance_floor_db` | `3.0` | > 0 | 音量分支的容忍下限；太安靜的房間才不會一點聲音就報警 | 不適用 |
| `yamnet_scale_shrinkage`／`yamnet_scale_floor` | `0.50`／`0.0001` | 0-1／> 0 | 正常統計量的收縮與標準差下限，樣本少時比較穩 | 不適用 |
| `top_k`／`n_contexts`／`context_frames` | 10／94／5 | 固定 | 分數聚合幾何，PC 與匯出必須一致，刻意不開放調整 | 不適用 |
| `minimum_train_sessions` 等資料量門檻 | 3／60 秒；校正 6／120 秒；held-out 2 session／40 片段 | — | 未達門檻時只給「分數」而不給判定，並會明講信心不足 | 不適用 |

`epochs`／`batch_size`／`learning_rate`／`denoise_sigma`／`bottleneck_dim` 屬於原始的
Dense 自編碼器基準線，YAMNet 流程不會用到（`config.py` 的註解有說明），所以腳本不開那些旗標。

對應程式：`tm_local/yamnet_anomaly_pipeline.py`（`train_yamnet_abnormal_sound_project()`、
`score_wav_bytes()`、`write_yamnet_runner()`）、`tm_local/anomaly_schema.py`。

```bat
.venv\Scripts\python.exe reference\abnormal_sound\train.py --project <名稱> --sensitivity low_false_alarm
.venv\Scripts\python.exe reference\abnormal_sound\preprocess.py --wav room.wav
.venv\Scripts\python.exe reference\abnormal_sound\run_tflite.py --model abnormal_sound_yamnet_embedding_int8.tflite --scorer yamnet_scorer.json --input room.wav
```

---

## 改了參數之後，開發板端怎麼跟著變

開發板端**不讀專案設定**，只讀匯出產生的契約檔。部署（Export modal 的「部署到開發板」頁籤）
會依序做：Vela 編譯 → flash 預算檢查 → 組出韌體專案 → GCC 編譯
（`tm_local/mcu/deploy_service.py` 的 `run_deploy()`）。

| 契約檔 | 內容 | 誰會因為它而改變 |
|---|---|---|
| `audio_frontend.json` | log-mel 幾何（`mel_bins`、window/hop/fft、`db_floor`…） | `mcu_tables.py` 印的那幾張 C 表、韌體的 `NUML_*` 巨集 |
| `yamnet_frontend.json` | YAMNet 前處理契約（96 × 64、log_offset…） | known_sound／abnormal_sound 的前處理程式碼 |
| `conversion_report.json` | 輸入輸出的 dtype、shape、scale／zero_point、運算子稽核 | 韌體的量化常數與張量檢查 |
| `labels.txt` | 類別順序（`0 Class 1`…） | 韌體的標籤陣列、背景類索引 |
| `training_report.json` | 實際用到的設定、每類門檻（`class_thresholds_by_label`）、評估結果 | 韌體的門檻陣列、判定邏輯 |

所以流程永遠是：**改設定 → 重新 Train → 重新 Export → 重新部署**。
只改 `runtime` 類的參數時，模型檔本身不會變，但韌體仍要重新編譯才能帶入新常數。

工具鏈（Arm GNU Toolchain、Nu-Link）不隨本工具打包，安裝與燒錄步驟見
`docs/MCU_DEPLOY_zh-TW.md`；MCU 端音訊前處理的逐步規格見 `docs/MCU_AUDIO_INT8_zh-TW.md`。
