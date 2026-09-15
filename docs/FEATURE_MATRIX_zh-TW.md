# 功能矩陣

| 功能 | Image | Audio | Known Sound | Abnormal Sound |
|---|---:|---:|---:|---:|
| Webcam／麥克風收集 | ✅ | ✅ | ✅ | ✅ |
| 批次上傳樣本 | ✅ | ✅ | ✅ | ✅ |
| 本機 TensorFlow 訓練 | ✅ | ✅ | ✅（frozen YAMNet＋sigmoid head） | ✅（Normal-only scorer） |
| Windows CPU runtime | ✅ | ✅ | ✅ | ✅ |
| PC 即時 Preview | ✅ | ✅ | ✅（每類獨立信心分數＋峰值保持） | ✅（分數＋temporal vote） |
| Session-disjoint 驗證 | 不適用 | ✅（可關閉，預設開啟） | ✅（一律，無開關） | ✅（一律） |
| 可調增強／微調 | ✅（增強強度、Dropout、MobileNetV2 微調層數） | ✅（增強強度、SpecAugment、Dropout） | ✅（波形增強、分類頭 Dropout、混音、每類門檻） | 不提供（改用 Sensitivity） |
| Float32 TFLite | ✅ | ✅ | ✅ | ✅ |
| Dynamic Range TFLite | ✅ | ✅ | ✅ | ✅ |
| Strict INT8 TFLite | ✅ | ✅ | ✅ | ✅（embedding core） |
| Strict UINT8 TFLite | ✅ | ✅ | ✅ | ✅（embedding core） |
| C array `model_data.h` | ✅ | ✅ | ✅（仍需 frontend） | ✅（仍需 frontend＋scorer） |
| 部署到 M55M1 韌體 | ✅（不含 VoiceAI，無相機） | ✅ | ✅（深度上限：X 12／GestureAI 11／VoiceAI 11） | 不支援 |
| 專案匯出／匯入 | ✅ | ✅ | ✅ | ✅（匯入後須重訓 threshold） |
| 單一診斷 Log | ✅ | ✅ | ✅ | ✅ |
| Linux／WSL 安裝 | 不使用 | 不使用 | 不使用 | 不使用 |
| Native Windows NVIDIA training | 本版不提供 | 本版不提供 | 本版不提供 | 本版不提供 |

「部署到 M55M1 韌體」支援 NuMaker-M55M1（X 板）、NuGestureAI-M55M1 與 NuMaker-VoiceAI-M55M1
三塊板；VoiceAI 沒有相機，只能給 Audio 與 Known Sound 兩種聲音專案使用，Image 仍只能選
X 板或 GestureAI。需要自行安裝 Arm GNU Toolchain；步驟、限制與待實測項見 `MCU_DEPLOY_zh-TW.md`。
