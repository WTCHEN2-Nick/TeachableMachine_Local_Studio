# mcu_toolkit

本資料夾由 `scripts/vendor_mcu_toolkit.py` 從 Nuvoton NuML_Toolkit（commit 3126344）、M55M1BSP V3.01.005（adb58c85）與 NuML App Builder v0.1.8 產生，只保留 GCC 建置需要的部分。

## 相對上游的修改

- `NuML_TFLM_Tool/*/*_codegen.py`：樣板路徑改為相對 `__file__`；檔頭加註 Studio 修改標記。
- `NuML_TFLM_Tool/*/main_cpp_codegen.py`：Vela summary CSV 改用標準函式庫 `csv` 依**欄名**（`sram_memory_used`、`off_chip_flash_memory_used`）讀取，取代 pandas 的位置索引 `df.iloc[0,0]`／`df.iloc[0,1]`；Vela 的欄序即為此兩欄，結果與上游相同。
- `templates/M55M1/*/*/NN_*/ffconf_M55M1.h`：重生為 FatFs R0.16（FFCONF_DEF 80386）。
- `templates/M55M1/*/*/link_script/gcc/gcc.ld`：新增 SRAM_NONCACHEABLE 區段與 `__sram_noncacheable_*` 符號、`KEEP(*(nn_model))`、`*(ITCM)`；X 板 SRAM01_HYPERRAM 縮為 1 MiB（internal-only）。
- `templates/M55M1/*/*/NN_*/GCC/gcc.ld`：**整份覆寫**為上面那份修補後的 `link_script/gcc/gcc.ld`。progen `toolchain_settings.yaml` 的 `linker_file` 指向的是這份專案內副本，實際連結用的也是它，所以兩者必須逐位元相同。相對上游的**專案內**副本，記憶體配置因此改變：ITCM 由 0x00020000（128 KiB）改為 0x00010000（64 KiB）、SRAM012（0x00150000）改為 SRAM2（0x00050000）視窗、stack/heap 由 0x8000（32 KiB）改為 0xa000（40 KiB）。拿本資料夾與全新的 NuML checkout 做 diff 時，這份檔案的差異是預期的。
- X 板 `BoardInit.cpp`：HyperRAM 初始化註解掉；`Device/HyperRAM` 移除；`Display.h` 的 PDMA 關閉。
- `templates/M55M1BSP`：只含建置需要的子集，已套用 `BSP_patch/Profiler.hpp`。
- `apps/`：Studio 自己的每 kind 韌體來源（部分改寫自 App Builder）。
- `boards/NuMaker-VoiceAI-M55M1/BoardInit_VoiceAI.cpp`：由 App Builder `known_sound/device/BoardInit_VoiceAI.cpp` 原封不動複製，未作任何修改。VoiceAI 板沒有自己的 NuML 樣板，專案借用 NuGestureAI-M55M1 樣板產生，該樣板的除錯主控台在 UART5；VoiceAI 板實際的主控台是 UART4（J3 排針），而 BSP 的 `SetDebugUartMFP()` 沒有 UART4 分支，所以這個板子必須自帶 `SetDebugUartMFP()`／`SetDebugUartCLK()`／`InitDebugUart()` 三個函式——這是整支檔案的差異，不是編譯期 `-D` 定義能表達的。Task 3 的 `boards.json` 會把它登記為該板的 `extra_sources`。
- `LICENSES/`：本資料夾額外補上的授權全文（`Apache-2.0.txt`、`FatFs.txt`、`GPL-3.0.txt`），上游沒有隨附授權檔的元件在 `NOTICE_third_party.md` 指向這裡。`GPL-3.0.txt` 對應的是 `omv/Lib/libomv.a` 內的 `agast.o`，其原始碼 `omv/imlib/agast.c` 也一併隨附（只散布、不編譯）。

重新產生：`.venv\Scripts\python.exe scripts\vendor_mcu_toolkit.py toolkit --numl-root <NuML_Toolkit>`，再執行 `cdc`、`image-templates`、`known-sound`、`kws-template`、`board-init`、`manifest` 子命令。
`licenses` 子命令只重寫授權全文與本文件，不會重跑 93 MB 的 `toolkit`。
