# Abnormal Sound Project 參考腳本（只有 PC，沒有開發板版本）

用 Studio 自己的程式，在命令列看 YAMNet 前處理、重跑訓練（其實是「量出正常長什麼樣子」），
以及用匯出的 INT8 嵌入模型 + `yamnet_scorer.json` 算一段聲音的偏離比值。

**它不是分類器。** 只收 Normal 音訊，輸出只有「偏離正常多少」，永遠不會說出那是什麼聲音。
`tm_local/config.py` 的 `MCU_APPLICATION_BY_KIND` 把 `abnormal_sound` 對應到 `None`，
所以 Studio 的「部署到開發板」頁籤會直接說明此專案類型無法部署；要上板請改用
Audio(KWS) 或 Known Sound 專案。

## 資料流

```
只錄「正常」的聲音（每個 recording_session 是一次獨立的錄音）
      │  yamnet_model.waveform_to_log_mel_patches()：96 × 64 的官方 YAMNet patch
      ▼
  凍結的 YAMNet encoder → 1024 維嵌入 → 正規化成單位長度
      ▼
  以 session 切分（VAL_SESSION_FRACTION = 0.25，用 ceil 不用四捨五入）
      ├─ 訓練 session：算出正常的中心與尺度（robust_diagonal，shrinkage 0.5、floor 1e-4）
      └─ 保留 session：校正門檻 T_shape（每個 runtime 各算一次）與音量門檻 T_level
      ▼
  models/<專案>.keras ＋ 統計量        ← Train 到此為止
      ▼  （Export Model，是另一個 job）
  abnormal_sound_yamnet_embedding_int8.tflite ＋ yamnet_scorer.json ＋ yamnet_frontend.json
      ▼
  判定：形狀分支比值 = mean(((嵌入 - 中心)/尺度)^2) / T_shape
        音量分支比值 = |dBFS - C_level| / T_level
        兩者取大者 > 1 就是偏離；再經 sensitivity 的連續視窗投票
```

門檻對每個 runtime（`keras`／`float32`／`dynamic`／`int8`／`uint8`）各自校正：
量化會讓分數整體平移，共用一組門檻就會失準。

## 可調參數

預設值出自 `tm_local/config.py` 的 `ABNORMAL_SOUND_DEFAULTS`，
驗證規則出自 `validate_abnormal_sound_settings()`。

| 參數 | 預設 | 範圍 | 怎麼影響判定 | 牽動開發板？ |
|---|---|---|---|---|
| `sensitivity` | `balanced` | `sensitive`／`balanced`／`low_false_alarm` | `SENSITIVITY_PRESETS`：alpha 0.10（5 個視窗中 2 個）／0.05（3/5）／0.02（3/5）。越敏感越容易報警，也越容易誤報 | 不適用（僅 PC） |
| `level_tolerance_floor_db` | `3.0` | > 0 | 音量分支的容忍下限。非常安靜的房間若沒有下限，一點點音量起伏就會報警 | 不適用 |
| `yamnet_scale_shrinkage` | `0.50` | 0 ~ 1 | 正常統計量往整體平均收縮的比例；session 少時調高比較穩 | 不適用 |
| `yamnet_scale_floor` | `0.0001` | > 0 | 每一維標準差的下限，避免某維幾乎不變時除出巨大的比值 | 不適用 |
| `hop_seconds` | `0.5` | 固定 0.5 | 視窗位移；瀏覽器、匯出的執行器與投票規則共用同一套幾何，不可改 | 不適用 |
| `clip_seconds`／`sample_rate` | `1.0`／`16000` | 固定 | YAMNet 契約 | 不適用 |
| `context_frames`／`n_contexts`／`top_k` | `5`／`94`／`10` | 固定 | 分數聚合幾何，存成絕對整數是為了不讓 PC 與其他實作各自四捨五入 | 不適用 |
| `yamnet_embedding_dim`／`yamnet_scorer` | `1024`／`robust_diagonal` | 固定 | 這個版本只實作這一種 scorer | 不適用 |
| `minimum_train_sessions`／`minimum_train_seconds` | `3`／`60` | — | 未達標就不讓訓練，並說明還差多少 | 不適用 |
| `minimum_calibration_sessions`／`minimum_calibration_seconds` | `6`／`120` | — | 未達標時只給「分數」而不給判定，並明講信心不足 | 不適用 |
| `minimum_heldout_sessions`／`minimum_heldout_clips` | `2`／`40` | — | 校正門檻所需的保留資料量 | 不適用 |

`epochs`（60）、`batch_size`（64）、`learning_rate`（0.001）、`denoise_sigma`（0.01）、
`bottleneck_dim`（8）屬於原始的 Dense 自編碼器基準線。目前的產品流程固定走
`detector_backend=yamnet_embedding`，不會用到它們，所以 `train.py` 也不開那些旗標
（`config.py` 內有完整說明）。

## 怎麼跑

```bat
rem 只看設定
.venv\Scripts\python.exe reference\abnormal_sound\train.py --project "機台聲音" --dry-run

rem 訓練（會先把設定寫回專案：這個 pipeline 要求 options 與專案設定完全一致）
.venv\Scripts\python.exe reference\abnormal_sound\train.py --project "機台聲音" ^
    --sensitivity low_false_alarm --level-tolerance-floor-db 4.0

rem 看 YAMNet patch 與音量
.venv\Scripts\python.exe reference\abnormal_sound\preprocess.py --wav room.wav --png room.png

rem 只算嵌入向量
.venv\Scripts\python.exe reference\abnormal_sound\run_tflite.py ^
    --model abnormal_sound_yamnet_embedding_int8.tflite --input room.wav

rem 加上 scorer 才會有偏離比值（--runtime 要對應模型檔：int8／uint8／float32…）
.venv\Scripts\python.exe reference\abnormal_sound\run_tflite.py ^
    --model abnormal_sound_yamnet_embedding_int8.tflite ^
    --scorer yamnet_scorer.json --runtime int8 --input room.wav
```

腳本只算**一個** 1 秒視窗，不做多視窗投票；完整的投票與信心等級在匯出 ZIP 的
`run_model.py` 與 Studio 的 Preview 裡。

## 對應的程式

| 這裡做的事 | Studio 的程式 |
|---|---|
| YAMNet 前處理 | `tm_local/yamnet_model.py`：`waveform_to_log_mel_patches()`、`frontend_contract()` |
| 訓練與校正 | `tm_local/yamnet_anomaly_pipeline.py`：`train_yamnet_abnormal_sound_project()`、`score_wav_bytes()`、`write_yamnet_runner()` |
| 資料量與信心等級 | `tm_local/anomaly_schema.py`：`assess_readiness()` |
| 音量分支 | `tm_local/audio_frontend.py`：`rms_dbfs()`、`level_diagnostics()` |
| 參數預設與驗證 | `tm_local/config.py`：`ABNORMAL_SOUND_DEFAULTS`、`SENSITIVITY_PRESETS`、`VAL_SESSION_FRACTION`、`validate_abnormal_sound_settings()` |
| 匯出與嚴格 INT8 | `tm_local/export_service.py`、`tm_local/tflite_export.py` |

更多說明：`docs/ABNORMAL_SOUND_YAMNET_zh-TW.md`。
