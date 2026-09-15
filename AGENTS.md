# Codex 專案指引

本專案是 **Teachable Machine Local Studio v2.1.0 — Windows Only**。所有 Codex 的工作紀錄與回覆使用繁體中文；程式中的既有命名、API 欄位與使用者可見英文文案則依專案現況維持一致。

## 開發與驗證

只使用專案內的 Windows Python 虛擬環境：

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe scripts\start_local.py --check
.venv\Scripts\python.exe scripts\verify_install.py
node --check web\app.js
```

學生使用的入口只能保留 `01_INSTALL.bat` 與 `02_START.bat`。不要刪除 `workspace/`，其中包含使用者的樣本與模型。

## 架構護欄

- 唯一 runtime 是 Windows CPU；不要加入 WSL、CUDA 或 Linux 專用流程。
- `TF_USE_LEGACY_KERAS=1` 必須在 import TensorFlow 前設定，pipeline 維持函式內延遲 import。
- Train 只產生 `.keras`；TFLite 必須等使用者按 Export 後，由獨立 conversion worker 透過 SavedModel signature 轉換。
- INT8 / UINT8 匯出必須維持 integer I/O、無 float tensor、無 Flex op 的嚴格驗證。
- Keras 模型統一用 `training_common.save_native_keras_model()` 儲存，不傳 `include_optimizer`。
- 樣本或訓練設定變動後，必須使舊模型與相關校正資訊失效。
- 音訊 frontend 的訓練實作、匯出參考程式與 MCU 文件是同一份契約，變更時必須同步。
- 開發板韌體只走 Arm GNU Toolchain（GCC），不使用 Keil、make 或 project-generator；MCU 端常數一律由匯出契約檔生成，不手動同步。`mcu_toolkit/` 是 vendored 內容，修改要透過 `scripts/vendor_mcu_toolkit.py` 並更新 `manifest.json`。
- 專案 metadata 以 `atomic_write_json()` 寫入；檔案路徑需經 traversal 檢查，ZIP 解壓使用 `safe_extract_zip()`。
- `tests/test_distribution.py` 是發行架構護欄；變更受保護設計前先確認意圖，再同步更新斷言。

## 新增 Project Kind

目前多處將 kind 寫成 `image` / `audio` 二選一。加入第三種 kind 時，必須完整檢查 `config.py`、`project_store.py`、`app.py`、`training_dispatch.py`、`export_service.py`、Preview runtime、`web/index.html`、`web/app.js`、匯入／匯出與測試，不能讓新 kind 落入既有 `else = audio` 的隱式分支。
