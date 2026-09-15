# Image Project 參考腳本

用 Studio 自己的程式，在命令列重跑「訓練」與「用 INT8 模型辨識一張圖」。
邏輯全部在 `tm_local/image_pipeline.py`，這裡的兩支腳本只負責參數與列印。

## 資料流

```
workspace/projects/<id>/samples/<class_id>/*.jpg
      │  ImageOps.exif_transpose → convert("RGB") → ImageOps.fit(image_size, image_size)
      │  （float32 0-255，不做 /255；正規化在模型的第一層）
      ▼
  tf.data 資料集（augmentation_level 只在訓練分支生效）
      ▼
  MobileNetV2(alpha=0.35, 凍結) 或 small_cnn
      → GlobalAveragePooling → Dropout(dropout) → Dense(n, softmax)
      ▼
  models/<專案>.keras        ← Train 到此為止，Preview 立刻可用
      ▼  （按 Export Model 才會做，是另一個 job）
  SavedModel → TFLiteConverter.from_saved_model → image_classifier_int8.tflite
      ▼  （Export modal 的「部署到開發板」頁籤）
  Vela 編譯 → flash 預算檢查 → 組專案 → GCC → firmware.bin
```

## 可調參數

預設值出自 `tm_local/config.py` 的 `IMAGE_DEFAULTS`，合法範圍出自 `validate_image_settings()`。

| 參數 | 預設 | 範圍 | 怎麼影響準確率 | 改了會牽動開發板嗎 |
|---|---|---|---|---|
| `augmentation_level` | `medium` | `off`／`light`／`medium`／`strong` | 翻轉＋亮度＋對比（`strong` 再加縮放 0.15、旋轉 10°、色相 0.03、飽和度 0.8-1.2）。樣本少、背景單一時往上調 | 否（`training`） |
| `dropout` | `0.2` | 0 ~ 0.6 | 訓練準確率遠高於驗證時往上調 | 否（`training`） |
| `fine_tune_blocks` | `0` | 0 ~ 4 | 第二階段解凍 MobileNetV2 最後 N 個 block（BN 維持凍結、學習率 ×0.1、`max(1, epochs//2)` 回合）。**只有 `mobilenet_v2` 有效**；`small_cnn` 會忽略並在進度訊息中說明 | 否（`training`） |
| `image_size` | `224` | 96-320，32 的倍數 | 小尺寸快又省 flash，但細節題目會掉準 | **是**（`model-shape`；上板限 `MCU_IMAGE_SIZES` = 96/128/160/192/224） |
| `epochs` | `30` | 1 ~ 300 | 太少學不完，太多過擬合 | 否 |
| `early_stopping` | `true` | true／false | 驗證分數不再進步就停 | 否 |
| `backbone` | `mobilenet_v2` | `mobilenet_v2`／`small_cnn` | `small_cnn` 小很多但通常較不準 | **是**（`model-shape`） |
| `mobilenet_alpha` | `0.35` | 0.35／0.5／0.75／1.0 | 越大越準、也越大越慢 | **是**（`model-shape`） |
| `batch_size` | `16` | 1 ~ 128 | 影響收斂穩定度 | 否 |
| `learning_rate` | `0.001` | 1e-6 ~ 0.1 | 太大不收斂、太小學不動 | 否 |
| `validation_split` | `0.20` | 0 ~ 0.45 | 驗證集越大分數越可信、可訓練資料越少 | 否 |
| `minimum_samples_per_class` | `5` | 2 ~ 100 | 每類樣本不足時 Train 直接拒絕並說明 | 否 |
| `deployment_target` | `pc` | `pc`／`NuMaker-M55M1`／`NuGestureAI-M55M1`（**不含** `NuMaker-VoiceAI-M55M1`：這塊板沒有相機，不是 Image 專案的合法目標） | 不影響學習，但會鎖住 `image_size` 的合法值 | **是**（`lock`） |

## 怎麼跑

```bat
rem 只看設定（不載入 TensorFlow，一秒完成）
.venv\Scripts\python.exe reference\image\train.py --project "我的圖片專案" --dry-run

rem 真的訓練，順便改兩個參數（會先寫回專案設定，再走 Studio 同一條路徑）
.venv\Scripts\python.exe reference\image\train.py --project "我的圖片專案" ^
    --augmentation-level strong --dropout 0.3 --fine-tune-blocks 2

rem 用匯出 ZIP 內的 INT8 模型辨識一張圖
.venv\Scripts\python.exe reference\image\run_tflite.py ^
    --model image_classifier_int8.tflite --labels labels.txt --input test.jpg

rem 只想看模型的輸入輸出張量規格
.venv\Scripts\python.exe reference\image\run_tflite.py --model image_classifier_int8.tflite --dry-run
```

`train.py` 只做到 `.keras`；INT8 `.tflite` 請在 Studio 按 Export Model（那是另一個 job）。

## 對應的程式

| 這裡做的事 | Studio 的程式 |
|---|---|
| 開專案、讀寫設定 | `tm_local/project_store.py`：`ProjectStore`、`update_project()`、`set_training_started()`、`set_training_result()` |
| 分派訓練 | `tm_local/training_dispatch.py`：`train_project()` |
| 真正的訓練 | `tm_local/image_pipeline.py`：`train_image_project()`、`_build_model()`、`_make_dataset()`、`fine_tune_layer_names()` |
| 載入一張圖 | `tm_local/image_pipeline.py`：`load_image_for_prediction()`、`_load_image_array()` |
| 參數預設與驗證 | `tm_local/config.py`：`IMAGE_DEFAULTS`、`validate_image_settings()`、`MCU_IMAGE_SIZES` |
| 匯出（另一個 job） | `tm_local/export_service.py`、`tm_local/tflite_export.py`、`tm_local/conversion_worker.py` |
| 上開發板 | `tm_local/mcu/deploy_service.py`、`tm_local/mcu/codegen.py` |
