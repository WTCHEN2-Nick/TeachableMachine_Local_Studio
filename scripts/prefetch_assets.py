from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAMNET_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
YAMNET_SHA256 = "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744"
YAMNET_PATH = PROJECT_ROOT / "assets" / "yamnet" / "yamnet.h5"

# The 521 AudioSet display names that go with those weights. Separate asset because the
# weight file carries no vocabulary. Used by the recording sanity check; training and
# export work without it.
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/"
    "yamnet/yamnet_class_map.csv"
)
YAMNET_CLASS_MAP_SHA256 = "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"
YAMNET_CLASS_MAP_PATH = PROJECT_ROOT / "assets" / "yamnet" / "yamnet_class_map.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_pinned(url: str, destination: Path, expected_sha256: str, label: str) -> None:
    """Download to a temporary file, verify the digest, then atomically move into place."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        print(f"[OK] {label}已存在且 SHA-256 正確：", destination)
        return
    temporary = destination.with_name(destination.name + ".download")
    temporary.unlink(missing_ok=True)
    try:
        print(f"[OPTIONAL] 下載並驗證{label}…")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "TeachableMachine-Local-Studio/2.1.0"},
        )
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        actual = _sha256(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            )
        os.replace(temporary, destination)
        print(f"[OK] {label}已安裝。")
    finally:
        temporary.unlink(missing_ok=True)


def prefetch_yamnet() -> None:
    """Install the pinned Apache-2.0 YAMNet weight asset for offline training."""

    _download_pinned(YAMNET_URL, YAMNET_PATH, YAMNET_SHA256, "YAMNet 權重（約 15 MB）")


def prefetch_yamnet_class_map() -> None:
    """Install the AudioSet class map used by the optional recording sanity check."""

    _download_pinned(
        YAMNET_CLASS_MAP_URL,
        YAMNET_CLASS_MAP_PATH,
        YAMNET_CLASS_MAP_SHA256,
        "YAMNet 521 類標籤表（約 14 KB）",
    )


def main() -> int:
    print("[OPTIONAL] 嘗試預先下載 MobileNetV2 ImageNet 權重…")
    try:
        import tensorflow as tf

        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
        print("[OK] MobileNetV2 權重已可離線重用。參數數量：", model.count_params())
    except Exception as exc:
        print("[WARN] 無法預先下載 MobileNetV2：", type(exc).__name__, exc)
        print("       安裝仍可完成；Image 訓練會在必要時改用內建 small CNN。")
    try:
        prefetch_yamnet()
    except Exception as exc:
        print("[WARN] 無法安裝 YAMNet：", type(exc).__name__, exc)
        print("       Image/Audio 仍可使用；Abnormal Sound Train 會明確要求重新執行安裝。")
    try:
        prefetch_yamnet_class_map()
    except Exception as exc:
        print("[WARN] 無法安裝 YAMNet 標籤表：", type(exc).__name__, exc)
        print("       訓練與匯出不受影響；只有『錄音健檢』會顯示未安裝。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
