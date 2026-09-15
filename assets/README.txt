此資料夾保留給可選的本機模型資產。

Image Project 第一次使用 MobileNetV2 時，TensorFlow 會下載 ImageNet 權重到使用者快取。
01_INSTALL.bat 會嘗試預先下載，但下載失敗不會阻止安裝；訓練時會自動改用內建 small CNN。

Abnormal Sound Project 使用 TensorFlow Models 的 Apache-2.0 YAMNet 預訓練權重：
https://storage.googleapis.com/audioset/yamnet.h5

01_INSTALL.bat 會下載到 assets/yamnet/yamnet.h5，並強制核對 SHA-256：
13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744

YAMNet 不會在每次 Train 或 Preview 時連網。資產缺少或校驗失敗時，Abnormal Sound
會停止並提示重新執行安裝，不會靜默換成另一個模型。
