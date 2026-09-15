# 疑難排解

## 安裝失敗

查看：

```text
logs\LATEST_INSTALL.log
```

常見處理：

1. 關閉 Local Studio。
2. 刪除 `.venv`。
3. 保留 `workspace`。
4. 重新執行 `01_INSTALL.bat`。

本版只安裝 Windows Python 套件，不會要求 Linux／WSL／CUDA。

## 執行、訓練或匯出失敗

在網頁右上角按「下載 Log」，取得：

```text
TM_Local_Studio_Diagnostic.txt
```

或查看：

```text
logs\LATEST.log
```

## Port 8765

```bat
netstat -ano | findstr :8765
```

沒有輸出代表 Port 沒有被使用。若有 PID，先確認是否為另一個 Local Studio 視窗。

## 訓練較慢

本版固定使用 Windows CPU。較舊電腦訓練 MobileNetV2 可能需要數分鐘。建議：

- 先用較少 Epoch 測試流程，例如 10。
- 使用較小 Image Size。
- 改用 small CNN。
- 關閉其他大量使用 CPU／RAM 的程式。

每個 Epoch 的訊息與時間會寫入 `LATEST.log`。

## Export 百分比暫時不變

單一 TensorFlow Lite conversion 沒有細部 callback。只要 elapsed time 持續增加、Log 有 heartbeat，就可能仍在轉換。超過設定時間後 worker 會自動停止，不會刪除 `.keras`。

## 建置開發板韌體失敗

「部署到開發板」的錯誤訊息都是中文，並且會指出要調哪一個設定。常見情況：

- 顯示「還不能部署：缺少 Arm GNU Toolchain（arm-none-eabi-gcc）」→ 還沒安裝工具鏈。
- 顯示模型「超過 … 的內部 flash」→ Known Sound 降 `encoder_depth`，Image 降 `image_size`
  或 MobileNet alpha，改完要重新訓練與匯出。
- 顯示「Studio 資料夾路徑太長」→ 把整個資料夾搬到像 `C:\TM_Studio` 的短路徑。

完整安裝步驟、燒錄方式、錯誤訊息對照表與待實測項見 `MCU_DEPLOY_zh-TW.md`。
失敗時的完整編譯紀錄在 `workspace\projects\<專案 id>\models\mcu\<板子>\build.log`。

## Image／Audio Preview 百分比看不到

先按：

```text
Ctrl + F5
```

讓瀏覽器重新載入新版 CSS 與 JavaScript。
