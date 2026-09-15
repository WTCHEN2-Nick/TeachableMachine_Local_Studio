'use strict';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

// --------------------------------------------------------------------------------------
// i18n -- Phase 1 of 2. This table and the helpers below are the whole toggle: they cover
// the strings in web/index.html today. web/app.js's own ~350 Chinese literals (toasts, job
// progress text, MCU copy, ...) are converted in a later phase; until then those call sites
// keep writing plain Traditional Chinese, same as before this change. New keys belong in
// these same two objects -- do not start a second table. Backend error messages are never
// translated (see CLAUDE.md); a student who hits a server error sees Chinese in both modes.
// Key naming: `<area>.<element>.<purpose>` (a bare `<area>.<purpose>` when there is no
// single owning element, e.g. `common.refresh`). `common.*` is for strings reused verbatim
// in more than one place in the UI (Refresh, Working...).
// --------------------------------------------------------------------------------------
const I18N = Object.freeze({
  zh: Object.freeze({
    'header.home.aria': '回首頁',
    'header.projectName.aria': '專案名稱',
    'header.saveState.saved': '已儲存於本機',
    'header.status.connecting': '連線中',
    'header.importProject': '匯入專案',
    'header.exportProject': '下載專案',
    'header.diagnostics.title': '發生問題時下載這個檔案',
    'header.diagnostics.button': '下載 Log',
    'header.deleteProject': '刪除專案',
    'home.title.line1': '教電腦辨認',
    'home.title.line2': '自己的影像與聲音',
    'home.hero.subtitle': '所有樣本、訓練與模型量化都在這台 PC 執行。建立資料、訓練、預覽，再下載可部署的 INT8 TensorFlow Lite 模型。',
    'home.newProject.heading': '建立新專案',
    'home.card.image.desc': '用 Webcam 或圖片訓練影像分類模型',
    'home.card.audio.desc': '用麥克風或音訊檔訓練聲音分類模型',
    'home.card.abnormalSound.desc': '收集正常聲音，以 frozen YAMNet embedding 建立基準並偵測未知異常',
    'home.card.knownSound.desc': '教它認得每一種聲音（例如槍聲／狗叫／玻璃破裂），顯示每類獨立的百分比',
    'home.recent.heading': '最近的本機專案',
    'common.refresh': '重新整理',
    'home.empty.title': '還沒有專案',
    'home.empty.desc': '選擇上方的 Image、Audio 或 Abnormal Sound Project 開始。',
    'home.footer.disclaimer': '本工具是獨立本機實作，不是 Google 官方產品；介面沿用熟悉的資料 → 訓練 → 預覽工作流程。',
    'workspace.microphone.label': '麥克風',
    'workspace.microphone.selectAria': '選擇麥克風',
    'workspace.microphone.systemDefault': '系統預設麥克風',
    'workspace.microphone.redetect': '重新偵測',
    'workspace.microphone.redetectTitle': '重新偵測麥克風',
    'workspace.microphone.notStarted': '尚未開始收音',
    'workspace.stopMedia': '停止相機／麥克風',
    'workspace.training.hint': '每個類別加入足夠樣本後即可訓練。',
    'workspace.training.progressNote': '此階段只做訓練與儲存；INT8 量化會在 Export Model 時執行。',
    'workspace.training.doneNote': '模型已可預覽與匯出',
    'preview.runtimeSelect.title': '預覽模型',
    'preview.locked': '先在左邊訓練一個模型，才能在這裡預覽。',
    'modal.close.aria': '關閉',
    'export.modal.lede': 'Train Model 只負責訓練與 Preview；TensorFlow Lite／INT8 會在這裡依你勾選的格式才開始轉換。',
    'export.mcu.tabLabel': '部署到開發板',
    'export.format.int8.desc': '完整整數 input/output，適合 MCU／NPU',
    'export.format.uint8.desc': '較舊 TensorFlow Lite runtime 相容格式',
    'export.format.float32.desc': '做為準確度比對與桌面測試基準',
    'export.format.dynamic.desc': '權重量化、Float32 I/O',
    'export.format.keras.desc': '保留可重新載入的訓練模型',
    'export.includeCHeader': '同時產生',
    'export.audioNotice.pre': 'Audio INT8 模型輸入是 log-mel spectrogram；ZIP 會附上',
    'export.audioNotice.post': '與完全一致的 Python 參考前處理。',
    'export.abnormalSoundNotice': 'Abnormal Sound 使用 frozen YAMNet 1024-D embedding 與 Normal-only scorer。匯出內容必須包含相符的 YAMNet frontend、scorer 與 threshold metadata；它不是聲音分類器。',
    'export.timeNote': '第一次產生 INT8 會在 CPU 上校正與轉換數分鐘；畫面會顯示實際進度與已經過時間。',
    'mcu.modal.lede': '把 Strict INT8 模型建成 Nuvoton M55M1 韌體。需要已安裝 Arm GNU Toolchain。',
    'mcu.board.label': '開發板',
    'mcu.deployButton': '建置韌體',
    'mcu.flashMethod.label': '燒錄方式',
    'mcu.downloadButton': '下載韌體 ZIP',
    'mcu.flashButton': '燒錄到板子',
    'confirm.title': '確認操作',
    'confirm.cancel': '取消',
    'confirm.accept': '確定',
    'common.busy': '處理中…',
    // ------------------------------------------------------------------------------------
    // Phase 2 additions -- web/app.js's own runtime strings (toasts, hints, MCU copy, ...).
    // ------------------------------------------------------------------------------------
    'options.deploymentTarget.pc': '電腦（不限制）',
    'options.deploymentTarget.numakerM55m1': 'NuMaker-M55M1（X 板）',
    'options.deploymentTarget.gestureai': 'NuMaker-GestureAI-M55M1',
    'options.deploymentTarget.voiceai': 'NuMaker-VoiceAI-M55M1（只有聲音）',
    'options.augmentation.light': '輕',
    'options.augmentation.medium': '中',
    'options.augmentation.strong': '強',
    'options.mixup.recommended': '0.5 × （建議）',
    'options.mixup.full': '1.0 ×',
    'options.knownSoundDepth.full': '完整 14 層 · 約 2.97 MB · PC／高階裝置',
    'options.knownSoundDepth.12': '12 層 · 約 1.56 MB',
    'options.knownSoundDepth.11': '11 層 · 約 1.31 MB',
    'options.knownSoundDepth.10': '10 層 · 約 1.05 MB · 建議給 2 MB flash MCU',
    'options.knownSoundDepth.9': '9 層 · 約 0.78 MB',
    'options.knownSoundDepth.8': '8 層 · 約 0.53 MB',
    'options.knownSoundDepth.7': '7 層 · 約 0.27 MB',
    'options.knownSoundDepth.6': '6 層 · 約 0.14 MB · 最小',
    'common.off': '關閉',
    'common.withDefaultSuffix': '{label}（預設）',
    'common.listSeparator': '、',
    'common.unknown': '未知',
    'common.unknownError': '未知錯誤',
    'common.toast.mediaStopped': '相機與麥克風已停止',
    'header.status.offline': '本機服務中斷',
    'header.saveState.saving': '儲存中…',
    'header.saveState.failed': '儲存失敗',
    'header.busy.creatingProject': '建立 {kind} Project…',
    'header.busy.packagingProject': '封裝完整專案與樣本…',
    'header.busy.importingProject': '匯入專案…',
    'header.busy.deletingProject': '刪除專案…',
    'header.confirm.deleteProjectMessage': '確定刪除「{name}」與所有本機樣本嗎？',
    'header.toast.diagnosticsDownloaded': '診斷 Log 已下載；發生問題時把這個檔案傳給負責人。',
    'header.toast.loadProjectsFailed': '無法讀取專案：{message}',
    'header.toast.createProjectFailed': '建立專案失敗：{message}',
    'header.toast.openProjectFailed': '無法開啟專案：{message}',
    'header.toast.projectDownloadFailed': '專案下載失敗：{message}',
    'header.toast.projectImported': '專案已匯入',
    'header.toast.projectImportFailed': '專案匯入失敗：{message}',
    'workspace.noun.class': '類別',
    'workspace.sample.deleteTitle': '刪除',
    'workspace.class.groupNameAria': '資料群組名稱',
    'workspace.class.nameAria': '類別名稱',
    'workspace.class.menuAria': '類別選單',
    'workspace.class.todoClips': '再 {count} 個片段',
    'workspace.class.todoSessions': '再 {count} 次獨立錄音',
    'workspace.class.sessionSummary': '{sessions} / {minimum} 次獨立錄音 · {todo}',
    'workspace.class.noSamples': '還沒有樣本',
    'workspace.class.ready': '可以開始訓練',
    'workspace.class.addMore': '還需要 {count} 個樣本',
    'workspace.class.normalSummary': '{seconds} 秒 / {minSeconds} 秒 · {sessions} / {minSessions} 次獨立錄音場次',
    'workspace.class.evalNote': '永遠不會用於訓練、校正或門檻選擇',
    'workspace.class.normalLockedTitle': 'Normal 基準容器的角色是鎖定的',
    'workspace.class.todoPrefix': '還需要{items}',
    'workspace.capture.imageHelp': '按住按鈕連續擷取；放開後會自動加入 {name}。',
    'workspace.capture.imageResultDefault': '把物件放在鏡頭前。建議改變角度、距離與背景。',
    'workspace.capture.audioHelp.normal': '每次錄音是一個獨立 session。請跨不同日期、距離與正常運轉條件收集；只錄 Normal。',
    'workspace.capture.audioHelp.evaluation': '此群組只衡量偵測效果，不會加入訓練、校正或 threshold。',
    'workspace.capture.audioHelp.default': '錄音會自動切成 1 秒樣本。Background Noise 建議錄製教室實際環境。',
    'workspace.capture.audioResultHelp.normal': '至少收集多個獨立 recording sessions；大量連續 clips 仍只算一個 session。',
    'workspace.capture.audioResultHelp.evaluation': '使用訓練時未見過的異常錄音，避免用 evaluation 資料調整模型。',
    'workspace.capture.audioResultHelp.default': '按下錄音，再按一次停止。至少錄製多種音量與距離。',
    'workspace.toast.renameClassFailed': '類別重新命名失敗：{message}',
    'workspace.confirm.clearTitle': '清除所有樣本',
    'workspace.confirm.clearNormalMessage': '這會清除全部 Normal baseline，已訓練模型與 threshold 也會失效。',
    'workspace.confirm.clearClassMessage': '確定要清除「{name}」的全部樣本嗎？',
    'workspace.toast.samplesCleared': '樣本已清除',
    'workspace.confirm.deleteNounTitle': '刪除 {noun}',
    'workspace.confirm.deleteNounMessage': '將刪除「{name}」以及其中所有樣本，無法復原。',
    'workspace.toast.nounDeleted': '{noun} 已刪除',
    'workspace.toast.deleteSampleFailed': '刪除樣本失敗：{message}',
    'workspace.toast.addClassFailed': '新增類別失敗：{message}',
    'workspace.busy.addingImages': '加入 {count} 張圖片…',
    'workspace.toast.imagesAdded': '圖片樣本已加入',
    'workspace.toast.imageUploadFailed': '圖片上傳失敗：{message}',
    'workspace.busy.decodingAudio': '解碼 {count} 個音訊檔…',
    'workspace.busy.decodingAudioProgress': '解碼音訊 {index}/{total}…',
    'workspace.busy.addingAudioSession': '加入音訊 session {index}/{total}…',
    'workspace.toast.audioSamplesAdded': '已加入 {count} 個音訊樣本',
    'workspace.toast.audioUploadFailed': '音訊上傳失敗：{message}',
    'workspace.error.noWebcam': '瀏覽器不支援 Webcam。',
    'workspace.toast.cameraStartFailed': '無法啟動相機：{message}',
    'workspace.microphone.selectedNotStarted': '{selected} · 尚未開始收音',
    'workspace.microphone.notStartedDetail': '尚未開始收音；按 Mic 或 Preview 後顯示實際設定',
    'workspace.microphone.unnamedDevice': 'Microphone {index}（允許權限後顯示名稱）',
    'workspace.toast.listMicrophonesFailed': '無法列出麥克風：{message}',
    'workspace.toast.microphoneFallback': '先前選取的麥克風無法使用，已改用系統預設麥克風。',
    'workspace.error.noMicrophone': '瀏覽器不支援麥克風。',
    'workspace.toast.stopBeforeSwitchMic': '請先停止並儲存目前錄音，再切換麥克風。',
    'workspace.busy.switchingMicrophone': '切換麥克風…',
    'workspace.toast.microphoneSwitched': '已切換至 {label}',
    'workspace.toast.switchMicrophoneFailed': '無法切換麥克風：{message}',
    'workspace.toast.microphoneStartFailed': '無法啟動麥克風：{message}',
    'workspace.busy.splittingAudio': '切割並加入音訊樣本…',
    'workspace.toast.recordingAddFailed': '錄音加入失敗：{message}',
    'workspace.toast.audioSanity': '錄音健檢：這段聽起來像 {summary}',
    'training.deploymentTarget.label': '目標裝置',
    'training.deploymentTarget.hint': '先選好要燒到哪一塊板子：下面只會列出該韌體支援的設定，Export 的部署頁籤也會自動選同一塊板。選「電腦」就不限制。',
    'training.backgroundClass.auto': '自動偵測（依類別名稱）',
    'training.backgroundClass.label': '背景／環境音類別',
    'training.backgroundClass.hint': '指定哪一類代表「什麼都沒發生」，板子會拿它當作不觸發的基準線。留「自動偵測」就找名稱含 background／noise／背景／環境音／安靜 的類別。',
    'training.image.sizeNotice': '開發板韌體只支援 {sizes}，原本的 {stored} 已改成 {size}。',
    'training.image.sizeHint.pc': '輸入解析度。224 是預設值；256、288、320 沒有自己的 ImageNet 預訓練權重，會沿用 224 的那一份，訓練時間與記憶體卻明顯增加，通常不會比較準。想再提高準確率，先加樣本或把「資料增強」調強。',
    'training.image.sizeHint.board': '輸入解析度。開發板韌體只編譯了這幾種；解析度越高越吃 SRAM 與推論時間，卡在記憶體時先往下調這一項。',
    'training.fineTune.choice0': '0 · 不微調（預設）',
    'training.fineTune.choice1': '1 個 block',
    'training.fineTune.choice2': '2 個 blocks',
    'training.fineTune.choice3': '3 個 blocks',
    'training.fineTune.choice4': '4 個 blocks',
    'training.mobilenetAlpha.default': '0.35 · Teachable Machine 預設值，速度快',
    'training.mobilenetAlpha.sramWarning': '{value} · 這個大小可能超出開發板的 SRAM，放不下時請調低解析度。',
    'training.fineTune.label': '微調層數',
    'training.fineTune.hint.locked': 'Small CNN 是從零訓練的，沒有預訓練權重可以微調；要用微調請把 Model 改回 MobileNetV2。',
    'training.fineTune.hint.unlocked': '解凍 MobileNetV2 最後幾個 block 跟著一起訓練，準確率通常再高一點，訓練時間大約多 50%。0 = 不微調。',
    'training.augmentation.label': '資料增強',
    'training.augmentation.hint.image': '中（預設）：亮度與左右翻轉。強：再加旋轉與縮放，樣本少時建議。關閉：只用原始照片。',
    'training.augmentation.hint.audio': '中（預設）：時間位移與音量變化。強：再加雜訊混入，錄音環境單一時建議。關閉：只用原始錄音。',
    'training.dropout.hint.image': 'Dropout 越高越不容易死背訓練照片，但也學得慢。0 = 關閉，預設 0.2。',
    'training.dropout.hint.audio': 'Dropout 越高越不容易死背訓練錄音，但也學得慢。0 = 關閉，預設 0.2。',
    'training.audio.rateNotice': '原本的 {rate} kHz 已改成 16 kHz，FFT 長度與最高頻率也會跟著改。頻譜規格一換，之前訓練好的模型就不能用了，要重新訓練。',
    'training.sampleRate.hint.locked': '開發板韌體的前處理是照 16 kHz 編譯的，所以部署到板子時只能用 16 kHz。',
    'training.sampleRate.hint.free': '16 kHz 就足夠辨識人聲與大部分環境音，而且是開發板唯一支援的取樣率。',
    'training.melBins.hint': '頻譜的頻帶數。40 是預設也最省記憶體；64 比較細緻，模型會稍大。',
    'training.specAugment.label': 'SpecAugment（頻譜遮罩）',
    'training.specAugment.hint': '在頻譜上隨機遮住幾條時間／頻率帶，逼模型不要只靠某一個頻段判斷。樣本少時特別有用。',
    'training.sessionDisjoint.label': 'Session-disjoint 驗證',
    'training.sessionDisjoint.hint': '同一段錄音切出來的片段不會同時進訓練與驗證，accuracy 比較誠實。只有一個錄音場次的類別會自動退回片段切分並提醒你。',
    'training.backgroundWeight.label': '背景類別權重',
    'training.backgroundWeight.hint.keyword': '調高會讓模型更不敢把背景音當成關鍵字（誤觸變少、漏聽變多），調低則相反。1 = 不特別加權。',
    'training.backgroundWeight.hint.knownSound': '調高會讓模型更不敢把背景音當成目標聲音（誤觸變少、漏聽變多），調低則相反。1 = 不特別加權。',
    'training.detectionThreshold.label': '偵測門檻',
    'training.detectionThreshold.hint.audio': '分數要超過這個值才算「聽到了」。它不參與訓練，上面的 Preview 也不會套用（Preview 一律顯示原始分數）；真正讀它的是匯出 ZIP 裡的 run_model.py 與開發板韌體。調高＝誤觸少但容易漏聽，調低＝反過來。{floorHint}',
    'training.audio.thresholdFloorHint': '這個專案只有兩個類別：板子上兩個分數加起來一定是 1，而韌體還要求贏家至少比另一個高 0.2，所以實際門檻不會低於 0.6。要讓門檻真的降下來，先多收一個類別（例如背景音）。',
    'training.knownSound.noteTitle': 'YAMNet · Frozen embedding + 分類頭',
    'training.knownSound.noteBody': '16 kHz mono → frozen embedding → 每類一個獨立 sigmoid。只訓練最後一層小分類頭，預訓練權重不會被微調。驗證一律以「整段錄音」為單位切分，同一段錄音不會同時出現在訓練與驗證。',
    'training.knownSound.encoderDepth.label': 'Encoder 深度',
    'training.knownSound.encoderDepth.hint': '保留 YAMNet 的前幾個 block。層數越少模型越小、燒得進 flash，但比較難分辨相似的聲音。',
    'training.knownSound.depthNotice': '{target} 的 flash 只放得下 {cap} 層，原本的 {stored} 層會被拒絕，已改成 {depth} 層。',
    'training.knownSound.detectionThreshold.label': '偵測門檻（預設值）',
    'training.knownSound.detectionThreshold.hint': '沒有另外設定的類別就用這個門檻。分數是各類獨立的信心分數，不會加總成 100%。',
    'training.knownSound.perClassThresholdHint': '每一類可以有自己的門檻：常常誤觸的聲音調高一點，很少出現但不能漏的警報聲調低一點。沒有動過的類別不會另外存，會自動跟著上面的「偵測門檻（預設值）」走。',
    'training.knownSound.classThresholdLabel': '{name} 的門檻',
    'training.waveformAugment.label': '波形增強',
    'training.waveformAugment.hint': '在送進 YAMNet 之前對錄音做時間位移／音量／雜訊擾動。關閉（預設）＝只用原始錄音；樣本少或錄音環境單一時再往上調。',
    'training.headDropout.label': '分類頭 Dropout',
    'training.headDropout.hint': '只加在分類頭前面，不會動到 frozen encoder。類別樣本少、驗證分數明顯低於訓練分數時調高。0 = 關閉。',
    'training.mixup.label': '混音增強',
    'training.mixup.hint': '把兩個不同類別的錄音疊在一起當新樣本，讓模型看得到「同時發生」的情況。',
    'training.mixup.notice': '混音增強只有 {choices} 三種，原本的 {stored} 已改成 {mixup}。',
    'training.hint.abnormal.calibrationCandidate': 'Normal baseline 已有 {seconds} 秒、{sessions} 個獨立 sessions。可建立 YAMNet scorer；最終 confidence 仍由 held-out audit 決定。',
    'training.hint.abnormal.lowConfidence': '已達低信心訓練門檻；可先看 anomaly ratio，但正式 Normal／Abnormal verdict 建議至少 {seconds} 秒、{sessions} 個 sessions，且須通過 held-out audit。',
    'training.hint.abnormal.collectNormal': '先在 Normal baseline 收集正常聲音；anomaly evaluation 不會拿來訓練。',
    'training.hint.abnormal.measured': 'Normal baseline 目前 {seconds} / {minSeconds} 秒、{sessions} / {minSessions} 個獨立 sessions。',
    'training.hint.abnormal.recordAcrossSessions': '請跨時段重新錄音，不要只延長同一次錄音。',
    'training.hint.ready': '資料已準備完成。按下 Train Model 開始。',
    'training.hint.needMoreSamples': '每個類別至少需要 {minimum} 個樣本。',
    'training.summary.epochs': '訓練 {epochs} epochs',
    'training.summary.fineTuneEpochs': '微調 {epochs} epochs',
    'training.summary.backboneUsed': '實際使用 {backbone}',
    'training.summary.sessions': '錄音場次 訓練 {train} / 驗證 {validation}',
    'training.splitWarning': '提醒：{names} 只有一個錄音場次，驗證改用片段切分，這幾類的 accuracy 可能偏高。換個時間、換個位置再錄一次會更準。',
    'training.toast.needNormalBaseline': 'Normal baseline 需要至少 {seconds} 秒、{sessions} 個獨立 recording sessions；evaluation groups 不列入要求。',
    'training.toast.startFailed': '無法開始訓練：{message}',
    'training.toast.abnormalTrained': 'YAMNet Normal-only detector 已建立，可立即 Preview 異常分數',
    'training.toast.trained': '模型訓練完成，可立即 Preview；INT8 會在 Export Model 時產生',
    'training.toast.failed': '訓練失敗：{detail}（請按右上角「下載 Log」）',
    'training.toast.statusFailed': '訓練狀態讀取失敗：{message}',
    'preview.classThresholdTitle': '{name} 的門檻 {threshold}',
    'preview.thresholdText.shared': '超過 {value} 的視為偵測到',
    'preview.thresholdText.perClass': '每一類有自己的門檻（長條上的刻度就是該類的門檻），超過自己的門檻才算偵測到',
    'preview.knownSound.noteMain': '每一類都是獨立分數，<strong>不會加總成 100%</strong>；{thresholdText}。這是模型在已知類別上的信心，不是真實機率。',
    'preview.knownSound.noteBackground': '<br>背景類別：<strong>{name}</strong>（代表「什麼都沒發生」，板子不會拿它觸發動作）',
    'preview.knownSound.noteDetected': '<br>目前偵測到：<strong>{names}</strong>',
    'preview.toast.cameraStartFailed': '無法啟動預覽相機：{message}',
    'preview.toast.microphoneStartFailed': '無法啟動預覽麥克風：{message}',
    'preview.toast.predictFailed': '預覽失敗：{message}',
    'preview.busy.decodingAudio': '解碼預覽音訊…',
    'preview.toast.audioPreviewFailed': '音訊預覽失敗：{message}',
    'export.toast.selectFormat': '請至少選擇一種模型格式。',
    'export.busy.starting': '準備模型匯出… 0%',
    'export.toast.requestFailed': '模型匯出失敗：{message}',
    'export.busy.progress': '模型匯出 {percent}% · {message} · 已經 {elapsed}',
    'export.toast.completed': '模型轉換完成，ZIP 已下載；INT8／UINT8 現在也可在 Preview 選擇',
    'export.toast.failed': '模型匯出失敗：{detail}（請按右上角「下載 Log」）',
    'export.toast.statusFailed': '模型匯出狀態讀取失敗：{message}',
    'export.busy.resuming': '恢復模型匯出進度…',
    'mcu.flashSteps.msc': '1. 按住板上 User 按鈕（PA.4），同時按一下 Reset，再放開 User。\n2. 電腦會出現名稱為 M55M1 的隨身碟。\n3. 把 firmware.bin 拖放到該隨身碟（或直接按「燒錄到板子」）。\n4. 拖放完成後按一下 Reset，新韌體就會啟動。',
    'mcu.flashSteps.nulink': '1. 用 USB 線接板上的 Nu-Link 埠。\n2. 按「燒錄到板子」（需先安裝 Nu-Link Command Tool），或用 Keil 的 Download。\n3. 燒錄完成後板子會自動重置。',
    'mcu.serialHint': '結果：這個專案沒有畫面，只有文字。請在 Windows「裝置管理員 → 連接埠 (COM 和 LPT)」找出板子的 COM port，用 PuTTY／Tera Term 以 115200 8N1 連線。\n',
    'mcu.flashResult.image.msc': '結果：Windows 相機 app 會看到影像與左上角的辨識標籤；COM port（115200 8N1）會印出文字。',
    'mcu.flashResult.image.nulink': '結果：LCD 會顯示辨識標籤；Nu-Link 的虛擬 COM port（115200 8N1）會印出文字。',
    'mcu.flashResult.knownSound': '開機時每一類印一行 INFO threshold[類別名] …，之後每 0.5 秒一行 live rms … dBFS | 類別名=0.123 …（每類一個獨立信心分數）。',
    'mcu.flashResult.audio': '之後每 0.25 秒一行 KWS hop=… rms=… top=標籤 0.876 … drop=0 | 標籤=0.876* …，分數後的 * 標出這個 hop 判給哪一類，drop 不是 0 就表示日誌有缺字；判定成立時多印一行 KWS DETECTED: label=…。',
    'mcu.kindBlocked': '此專案類型無法部署到開發板。',
    'mcu.staleBuild': '模型已變更，請重新建置。',
    'mcu.flashMethodLabel.msc': 'USB 隨身碟 bootloader',
    'mcu.flashMethodLabel.nulink': 'Nu-Link',
    'mcu.flashMethodLabel.pyocd': 'pyocd（外接 Nu-Link2）',
    'mcu.missing.toolkit': '開發板工具組 mcu_toolkit（韌體樣板）',
    'mcu.missing.vela': 'Vela 編譯器',
    'mcu.missing.toolchain': 'Arm GNU Toolchain（arm-none-eabi-gcc）',
    'mcu.missing.boardList': '開發板清單 mcu_toolkit\\boards.json',
    'mcu.missing.generic': '開發板工具鏈',
    'mcu.toolchain.probing': '正在偵測開發板工具鏈…',
    'mcu.toolchain.notReady': '還不能部署：缺少 {missing}。請先安裝 Arm GNU Toolchain（arm-none-eabi-gcc），再重新執行 01_INSTALL.bat。',
    'mcu.building': '建置中…',
    'mcu.toast.noBoard': '目前沒有可用的開發板設定。',
    'mcu.busy.creatingJob': '建立韌體工作…',
    'mcu.toast.startFailed': '無法開始建置韌體：{message}',
    'mcu.toast.buildComplete': '韌體建置完成，可以下載 ZIP 或直接燒錄',
    'mcu.log.buildFailed': '建置失敗：{detail}',
    'mcu.toast.buildFailed': '韌體建置失敗：{detail}（請按右上角「下載 Log」）',
    'mcu.log.statusFailed': '讀取建置狀態失敗：{message}',
    'mcu.toast.statusFailed': '韌體建置狀態讀取失敗：{message}',
    'mcu.busy.preparingZip': '準備韌體 ZIP…',
    'mcu.toast.zipDownloaded': '韌體 ZIP 已下載',
    'mcu.toast.downloadFailed': '韌體下載失敗：{message}',
    'mcu.confirm.flashMessage': '即將把 {name}（{size} bytes）燒錄到 {label}，確定要繼續？',
    'mcu.confirm.flashTitle': '燒錄到開發板',
    'mcu.flashing': '燒錄中…',
    'mcu.flashDoneDefault': '燒錄完成，請按一下板子上的 Reset。',
    'mcu.toast.flashComplete': '燒錄完成',
    'mcu.toast.buildContinuesInBackground': '韌體建置仍在背景進行；重新開啟 Export Model 就能看到進度。',
  }),
  en: Object.freeze({
    'header.home.aria': 'Back to home',
    'header.projectName.aria': 'Project name',
    'header.saveState.saved': 'Saved locally',
    'header.status.connecting': 'Connecting…',
    'header.importProject': 'Import Project',
    'header.exportProject': 'Download Project',
    'header.diagnostics.title': 'Download this file when something goes wrong',
    'header.diagnostics.button': 'Download Log',
    'header.deleteProject': 'Delete Project',
    'home.title.line1': 'Teach a computer to recognize',
    'home.title.line2': 'your own images and sounds',
    'home.hero.subtitle': 'All your samples, training, and model quantization run on this PC. Build your dataset, train, preview, then download a deployable INT8 TensorFlow Lite model.',
    'home.newProject.heading': 'Create a new project',
    'home.card.image.desc': 'Train an image classifier with your webcam or photos',
    'home.card.audio.desc': 'Train a sound classifier with your microphone or audio files',
    'home.card.abnormalSound.desc': 'Collect normal sounds to build a baseline with a frozen YAMNet embedding, then detect unknown anomalies',
    'home.card.knownSound.desc': 'Teach it to recognize specific sounds (a gunshot, a dog bark, breaking glass) and show an independent confidence score for each class',
    'home.recent.heading': 'Recent local projects',
    'common.refresh': 'Refresh',
    'home.empty.title': 'No projects yet',
    'home.empty.desc': 'Pick Image, Audio, or Abnormal Sound Project above to get started.',
    'home.footer.disclaimer': 'This is an independent local tool, not an official Google product; the interface follows the familiar Collect → Train → Preview workflow.',
    'workspace.microphone.label': 'Microphone',
    'workspace.microphone.selectAria': 'Select microphone',
    'workspace.microphone.systemDefault': 'System default microphone',
    'workspace.microphone.redetect': 'Re-detect',
    'workspace.microphone.redetectTitle': 'Re-detect microphones',
    'workspace.microphone.notStarted': "Recording hasn't started yet",
    'workspace.stopMedia': 'Stop camera / microphone',
    'workspace.training.hint': 'Once every class has enough samples, you can train.',
    'workspace.training.progressNote': 'This step only trains and saves the model — INT8 quantization happens when you Export Model.',
    'workspace.training.doneNote': 'Model ready to preview and export',
    'preview.runtimeSelect.title': 'Preview model',
    'preview.locked': 'You must train a model on the left before you can preview it here.',
    'modal.close.aria': 'Close',
    'export.modal.lede': 'Train Model only trains and previews the model — TensorFlow Lite/INT8 conversion starts here, only for the formats you check.',
    'export.mcu.tabLabel': 'Deploy to a Dev Board',
    'export.format.int8.desc': 'Fully integer input/output — ideal for MCU/NPU',
    'export.format.uint8.desc': 'Compatible format for older TensorFlow Lite runtimes',
    'export.format.float32.desc': 'A baseline for accuracy comparison and desktop testing',
    'export.format.dynamic.desc': 'Weights quantized, Float32 I/O',
    'export.format.keras.desc': 'Keeps a reloadable copy of the trained model',
    'export.includeCHeader': 'Also generate',
    'export.audioNotice.pre': "The Audio INT8 model's input is a log-mel spectrogram — the ZIP includes",
    'export.audioNotice.post': 'along with a bit-identical Python reference for the preprocessing.',
    'export.abnormalSoundNotice': 'Abnormal Sound uses a frozen YAMNet 1024-D embedding with a Normal-only scorer. Exported files must include matching YAMNet frontend, scorer, and threshold metadata — it is not a sound classifier.',
    'export.timeNote': 'The first INT8 build calibrates and converts on the CPU for a few minutes; the screen shows real progress and elapsed time.',
    'mcu.modal.lede': 'Build your Strict INT8 model into Nuvoton M55M1 firmware. Requires the Arm GNU Toolchain to be installed.',
    'mcu.board.label': 'Dev board',
    'mcu.deployButton': 'Build Firmware',
    'mcu.flashMethod.label': 'Flash method',
    'mcu.downloadButton': 'Download Firmware ZIP',
    'mcu.flashButton': 'Flash to Board',
    'confirm.title': 'Confirm',
    'confirm.cancel': 'Cancel',
    'confirm.accept': 'OK',
    'common.busy': 'Working…',
    // ------------------------------------------------------------------------------------
    // Phase 2 additions -- web/app.js's own runtime strings (toasts, hints, MCU copy, ...).
    // ------------------------------------------------------------------------------------
    'options.deploymentTarget.pc': 'Computer (no limit)',
    'options.deploymentTarget.numakerM55m1': 'NuMaker-M55M1 (Board X)',
    'options.deploymentTarget.gestureai': 'NuMaker-GestureAI-M55M1',
    'options.deploymentTarget.voiceai': 'NuMaker-VoiceAI-M55M1 (audio only)',
    'options.augmentation.light': 'Light',
    'options.augmentation.medium': 'Medium',
    'options.augmentation.strong': 'Strong',
    'options.mixup.recommended': '0.5× (recommended)',
    'options.mixup.full': '1.0×',
    'options.knownSoundDepth.full': 'Full 14 layers · ~2.97 MB · PC/high-end device',
    'options.knownSoundDepth.12': '12 layers · ~1.56 MB',
    'options.knownSoundDepth.11': '11 layers · ~1.31 MB',
    'options.knownSoundDepth.10': '10 layers · ~1.05 MB · recommended for 2 MB flash MCUs',
    'options.knownSoundDepth.9': '9 layers · ~0.78 MB',
    'options.knownSoundDepth.8': '8 layers · ~0.53 MB',
    'options.knownSoundDepth.7': '7 layers · ~0.27 MB',
    'options.knownSoundDepth.6': '6 layers · ~0.14 MB · smallest',
    'common.off': 'Off',
    'common.withDefaultSuffix': '{label} (default)',
    'common.listSeparator': ', ',
    'common.unknown': 'Unknown',
    'common.unknownError': 'Unknown error',
    'common.toast.mediaStopped': 'Camera and microphone stopped',
    'header.status.offline': 'Local service disconnected',
    'header.saveState.saving': 'Saving…',
    'header.saveState.failed': 'Save failed',
    'header.busy.creatingProject': 'Creating {kind} Project…',
    'header.busy.packagingProject': 'Packaging the full project and samples…',
    'header.busy.importingProject': 'Importing project…',
    'header.busy.deletingProject': 'Deleting project…',
    'header.confirm.deleteProjectMessage': 'Delete "{name}" and all its local samples?',
    'header.toast.diagnosticsDownloaded': 'Diagnostic log downloaded — send this file to whoever is helping you when something goes wrong.',
    'header.toast.loadProjectsFailed': 'Could not load projects: {message}',
    'header.toast.createProjectFailed': 'Failed to create project: {message}',
    'header.toast.openProjectFailed': 'Could not open project: {message}',
    'header.toast.projectDownloadFailed': 'Project download failed: {message}',
    'header.toast.projectImported': 'Project imported',
    'header.toast.projectImportFailed': 'Project import failed: {message}',
    'workspace.noun.class': 'class',
    'workspace.sample.deleteTitle': 'Delete',
    'workspace.class.groupNameAria': 'Group name',
    'workspace.class.nameAria': 'Class name',
    'workspace.class.menuAria': 'Class menu',
    'workspace.class.todoClips': '{count} more clip(s)',
    'workspace.class.todoSessions': '{count} more independent recording(s)',
    'workspace.class.sessionSummary': '{sessions} / {minimum} independent recordings · {todo}',
    'workspace.class.noSamples': 'No samples yet',
    'workspace.class.ready': 'Ready for training',
    'workspace.class.addMore': 'Add {count} more sample(s)',
    'workspace.class.normalSummary': '{seconds} s / {minSeconds} s · {sessions} / {minSessions} independent sessions',
    'workspace.class.evalNote': 'Never used for training, calibration, or threshold selection',
    'workspace.class.normalLockedTitle': 'Normal baseline role is locked',
    'workspace.class.todoPrefix': 'Still need {items}',
    'workspace.capture.imageHelp': 'Hold the button to capture continuously; release it to add the images to {name}.',
    'workspace.capture.imageResultDefault': 'Hold the object in front of the camera. Try different angles, distances, and backgrounds.',
    'workspace.capture.audioHelp.normal': 'Each recording is one independent session. Collect across different days, distances, and normal operating conditions — Normal only.',
    'workspace.capture.audioHelp.evaluation': 'This group only measures detection performance — it never enters training, calibration, or threshold selection.',
    'workspace.capture.audioHelp.default': 'Recordings are automatically sliced into 1-second samples. For Background Noise, record the actual classroom environment.',
    'workspace.capture.audioResultHelp.normal': 'Collect at least several independent recording sessions — many consecutive clips still count as just one session.',
    'workspace.capture.audioResultHelp.evaluation': 'Use anomaly recordings the model has never seen during training — avoid tuning the model on evaluation data.',
    'workspace.capture.audioResultHelp.default': 'Press to record, then press again to stop. Record a variety of volumes and distances.',
    'workspace.toast.renameClassFailed': 'Failed to rename class: {message}',
    'workspace.confirm.clearTitle': 'Clear All Samples',
    'workspace.confirm.clearNormalMessage': 'This clears the entire Normal baseline; the trained model and threshold will be invalidated too.',
    'workspace.confirm.clearClassMessage': 'Clear all samples in "{name}"?',
    'workspace.toast.samplesCleared': 'Samples cleared',
    'workspace.confirm.deleteNounTitle': 'Delete {noun}',
    'workspace.confirm.deleteNounMessage': 'This deletes "{name}" and every sample in it. This cannot be undone.',
    'workspace.toast.nounDeleted': '{noun} deleted',
    'workspace.toast.deleteSampleFailed': 'Failed to delete sample: {message}',
    'workspace.toast.addClassFailed': 'Failed to add class: {message}',
    'workspace.busy.addingImages': 'Adding {count} image(s)…',
    'workspace.toast.imagesAdded': 'Image samples added',
    'workspace.toast.imageUploadFailed': 'Image upload failed: {message}',
    'workspace.busy.decodingAudio': 'Decoding {count} audio file(s)…',
    'workspace.busy.decodingAudioProgress': 'Decoding audio {index}/{total}…',
    'workspace.busy.addingAudioSession': 'Adding audio session {index}/{total}…',
    'workspace.toast.audioSamplesAdded': 'Added {count} audio sample(s)',
    'workspace.toast.audioUploadFailed': 'Audio upload failed: {message}',
    'workspace.error.noWebcam': 'This browser does not support the webcam.',
    'workspace.toast.cameraStartFailed': 'Could not start the camera: {message}',
    'workspace.microphone.selectedNotStarted': "{selected} · recording hasn't started yet",
    'workspace.microphone.notStartedDetail': "Recording hasn't started yet; press Mic or Preview to see the actual settings",
    'workspace.microphone.unnamedDevice': 'Microphone {index} (name shown after permission is granted)',
    'workspace.toast.listMicrophonesFailed': 'Could not list microphones: {message}',
    'workspace.toast.microphoneFallback': 'The previously selected microphone is unavailable; switched to the system default microphone.',
    'workspace.error.noMicrophone': 'This browser does not support the microphone.',
    'workspace.toast.stopBeforeSwitchMic': 'Stop and save the current recording before switching microphones.',
    'workspace.busy.switchingMicrophone': 'Switching microphone…',
    'workspace.toast.microphoneSwitched': 'Switched to {label}',
    'workspace.toast.switchMicrophoneFailed': 'Could not switch microphone: {message}',
    'workspace.toast.microphoneStartFailed': 'Could not start the microphone: {message}',
    'workspace.busy.splittingAudio': 'Splitting and adding audio samples…',
    'workspace.toast.recordingAddFailed': 'Failed to add the recording: {message}',
    'workspace.toast.audioSanity': 'Audio sanity check: this sounds like {summary}',
    'training.deploymentTarget.label': 'Target device',
    'training.deploymentTarget.hint': 'Pick which board you plan to flash first: the settings below only show what that firmware supports, and Export\'s deploy tab will pick the same board automatically. Choose "Computer" for no limit.',
    'training.backgroundClass.auto': 'Auto-detect (by class name)',
    'training.backgroundClass.label': 'Background/ambient class',
    'training.backgroundClass.hint': 'Pick which class means "nothing happened" — the board uses it as the do-not-trigger baseline. Leave "Auto-detect" to match a class named background/noise/ambient/quiet (or the Chinese equivalents).',
    'training.image.sizeNotice': 'The board firmware only supports {sizes} — {stored} has been changed to {size}.',
    'training.image.sizeHint.pc': 'Input resolution. 224 is the default; 256, 288, and 320 have no ImageNet pretrained weights of their own, so they reuse the 224 weights while training time and memory increase noticeably, usually without improving accuracy. To improve accuracy, add more samples or turn up augmentation instead.',
    'training.image.sizeHint.board': 'Input resolution. The board firmware is only compiled for these sizes; a higher resolution costs more SRAM and inference time, so lower this first if you run out of memory.',
    'training.fineTune.choice0': '0 · No fine-tuning (default)',
    'training.fineTune.choice1': '1 block',
    'training.fineTune.choice2': '2 blocks',
    'training.fineTune.choice3': '3 blocks',
    'training.fineTune.choice4': '4 blocks',
    'training.mobilenetAlpha.default': '0.35 · Teachable Machine default / fast',
    'training.mobilenetAlpha.sramWarning': "{value} · This size may exceed the board's SRAM — lower the resolution if it doesn't fit.",
    'training.fineTune.label': 'Fine-tune layers',
    'training.fineTune.hint.locked': 'Small CNN trains from scratch and has no pretrained weights to fine-tune; switch Model back to MobileNetV2 to use fine-tuning.',
    'training.fineTune.hint.unlocked': 'Unfreezes the last few MobileNetV2 blocks to train along with the head — usually a bit more accurate, at roughly 50% more training time. 0 = no fine-tuning.',
    'training.augmentation.label': 'Data augmentation',
    'training.augmentation.hint.image': 'Medium (default): brightness and horizontal flip. Strong: adds rotation and zoom — recommended with few samples. Off: uses the original photos only.',
    'training.augmentation.hint.audio': 'Medium (default): time shift and volume changes. Strong: adds noise mixing — recommended when recordings come from one consistent environment. Off: uses the original recordings only.',
    'training.dropout.hint.image': 'Higher dropout makes the model less likely to memorize the training photos, but it also learns more slowly. 0 = off; the default is 0.2.',
    'training.dropout.hint.audio': 'Higher dropout makes the model less likely to memorize the training recordings, but it also learns more slowly. 0 = off; the default is 0.2.',
    'training.audio.rateNotice': 'The previous {rate} kHz has been changed to 16 kHz, along with the FFT length and max frequency. Changing the spectrogram spec invalidates any previously trained model — retrain after this.',
    'training.sampleRate.hint.locked': "The board firmware's preprocessing is compiled for 16 kHz, so deploying to a board requires 16 kHz.",
    'training.sampleRate.hint.free': '16 kHz is enough to recognize speech and most ambient sounds, and it is the only sample rate the board supports.',
    'training.melBins.hint': 'Number of mel frequency bands. 40 is the default and uses the least memory; 64 is more detailed and makes the model slightly larger.',
    'training.specAugment.label': 'SpecAugment (spectrogram masking)',
    'training.specAugment.hint': 'Randomly masks a few time/frequency bands in the spectrogram, forcing the model not to rely on just one band. Especially useful with few samples.',
    'training.sessionDisjoint.label': 'Session-disjoint validation',
    'training.sessionDisjoint.hint': 'Clips sliced from the same recording never land in both training and validation, so accuracy is more honest. A class with only one recording session automatically falls back to clip-level splitting and warns you.',
    'training.backgroundWeight.label': 'Background class weight',
    'training.backgroundWeight.hint.keyword': 'Raising this makes the model more reluctant to mistake background noise for a keyword (fewer false triggers, more misses); lowering it does the opposite. 1 = no extra weighting.',
    'training.backgroundWeight.hint.knownSound': 'Raising this makes the model more reluctant to mistake background noise for a target sound (fewer false triggers, more misses); lowering it does the opposite. 1 = no extra weighting.',
    'training.detectionThreshold.label': 'Detection threshold',
    'training.detectionThreshold.hint.audio': 'The score must exceed this value to count as "heard." It plays no part in training, and Preview above ignores it too (Preview always shows the raw score) — it is only read by run_model.py in the exported ZIP and by the board firmware. Higher = fewer false triggers but more misses, and vice versa. {floorHint}',
    'training.audio.thresholdFloorHint': 'This project only has two classes: on the board the two scores always add up to 1, and the firmware also requires the winner to beat the other by at least 0.2 — so the effective floor never drops below 0.6. To actually lower the threshold, collect one more class first (a background class, for example).',
    'training.knownSound.noteTitle': 'YAMNet · Frozen embedding + classification head',
    'training.knownSound.noteBody': '16 kHz mono → frozen embedding → one independent sigmoid per class. Only the small classification head is trained; the pretrained weights are never fine-tuned. Validation always splits by whole recording, so the same recording never appears in both training and validation.',
    'training.knownSound.encoderDepth.label': 'Encoder depth',
    'training.knownSound.encoderDepth.hint': 'Keeps the first several YAMNet blocks. Fewer layers make a smaller model that fits into flash more easily, but make similar sounds harder to tell apart.',
    'training.knownSound.depthNotice': "{target}'s flash only fits {cap} layers; the previous {stored} layers would be rejected, so it has been changed to {depth} layers.",
    'training.knownSound.detectionThreshold.label': 'Detection threshold (default)',
    'training.knownSound.detectionThreshold.hint': 'Classes without their own override use this threshold. Scores are independent per-class confidence scores; they do not add up to 100%.',
    'training.knownSound.perClassThresholdHint': 'Each class can have its own threshold: raise it for a sound that triggers too often, lower it for a rare alarm you cannot afford to miss. Classes you never touch are not stored separately — they automatically track the "Detection threshold (default)" above.',
    'training.knownSound.classThresholdLabel': '{name} threshold',
    'training.waveformAugment.label': 'Waveform augmentation',
    'training.waveformAugment.hint': 'Applies time shift, volume, and noise perturbation to the recording before it reaches YAMNet. Off (default) = uses the original recording only; turn it up with few samples or a single recording environment.',
    'training.headDropout.label': 'Head dropout',
    'training.headDropout.hint': 'Applied only before the classification head; it never touches the frozen encoder. Raise it when a class has few samples and its validation score trails its training score noticeably. 0 = off.',
    'training.mixup.label': 'Mixup augmentation',
    'training.mixup.hint': 'Overlays recordings from two different classes as a new sample, so the model sees what "happening at the same time" looks like.',
    'training.mixup.notice': 'Mixup augmentation only supports {choices} — the previous {stored} has been changed to {mixup}.',
    'training.hint.abnormal.calibrationCandidate': 'The Normal baseline already has {seconds} s across {sessions} independent sessions. A YAMNet scorer can be built; the final confidence still comes from the held-out audit.',
    'training.hint.abnormal.lowConfidence': 'The low-confidence training floor has been reached; you can look at the anomaly ratio now, but an official Normal/Abnormal verdict should have at least {seconds} s across {sessions} sessions and pass the held-out audit.',
    'training.hint.abnormal.collectNormal': 'First collect normal sounds in the Normal baseline; anomaly evaluation clips are never used for training.',
    'training.hint.abnormal.measured': 'The Normal baseline currently has {seconds} / {minSeconds} s and {sessions} / {minSessions} independent sessions.',
    'training.hint.abnormal.recordAcrossSessions': 'Record across different sessions rather than just extending a single recording.',
    'training.hint.ready': 'Your data is ready. Press Train Model to start.',
    'training.hint.needMoreSamples': 'Each class needs at least {minimum} sample(s).',
    'training.summary.epochs': 'Trained {epochs} epochs',
    'training.summary.fineTuneEpochs': 'Fine-tuned {epochs} epochs',
    'training.summary.backboneUsed': 'Actually used {backbone}',
    'training.summary.sessions': 'Recording sessions: {train} train / {validation} validation',
    'training.splitWarning': 'Note: {names} only have one recording session, so validation fell back to clip-level splitting — their accuracy may read optimistically high. Recording again at a different time and location will make it more reliable.',
    'training.toast.needNormalBaseline': 'The Normal baseline needs at least {seconds} s across {sessions} independent recording sessions; evaluation groups do not count toward this.',
    'training.toast.startFailed': 'Could not start training: {message}',
    'training.toast.abnormalTrained': 'YAMNet Normal-only detector built — you can Preview the anomaly score right away',
    'training.toast.trained': 'Model training complete — Preview it right away; INT8 is produced when you Export Model',
    'training.toast.failed': 'Training failed: {detail} (click "Download Log" in the top right)',
    'training.toast.statusFailed': 'Failed to read training status: {message}',
    'preview.classThresholdTitle': '{name} threshold {threshold}',
    'preview.thresholdText.shared': 'Anything above {value} counts as detected',
    'preview.thresholdText.perClass': 'Each class has its own threshold (the tick mark on its bar is that threshold) — a class counts as detected only above its own threshold',
    'preview.knownSound.noteMain': "Each class is an independent score — <strong>they do not add up to 100%</strong>; {thresholdText}. This is the model's confidence among the known classes, not a true probability.",
    'preview.knownSound.noteBackground': '<br>Background class: <strong>{name}</strong> (means "nothing happened" — the board never triggers on it)',
    'preview.knownSound.noteDetected': '<br>Currently detected: <strong>{names}</strong>',
    'preview.toast.cameraStartFailed': 'Could not start the preview camera: {message}',
    'preview.toast.microphoneStartFailed': 'Could not start the preview microphone: {message}',
    'preview.toast.predictFailed': 'Preview failed: {message}',
    'preview.busy.decodingAudio': 'Decoding preview audio…',
    'preview.toast.audioPreviewFailed': 'Audio preview failed: {message}',
    'export.toast.selectFormat': 'Select at least one model format.',
    'export.busy.starting': 'Preparing model export… 0%',
    'export.toast.requestFailed': 'Model export failed: {message}',
    'export.busy.progress': 'Exporting model {percent}% · {message} · {elapsed} elapsed',
    'export.toast.completed': 'Model conversion complete — the ZIP has been downloaded; INT8/UINT8 can now be selected in Preview too',
    'export.toast.failed': 'Model export failed: {detail} (click "Download Log" in the top right)',
    'export.toast.statusFailed': 'Failed to read model export status: {message}',
    'export.busy.resuming': 'Resuming model export progress…',
    'mcu.flashSteps.msc': "1. Hold the board's User button (PA.4), press Reset once, then release User.\n2. A removable drive named M55M1 appears on the computer.\n3. Drag firmware.bin onto that drive (or just click \"Flash to Board\").\n4. After the drag finishes, press Reset once and the new firmware starts.",
    'mcu.flashSteps.nulink': '1. Connect a USB cable to the board\'s Nu-Link port.\n2. Click "Flash to Board" (requires the Nu-Link Command Tool installed first), or use Download in Keil.\n3. The board resets automatically once flashing finishes.',
    'mcu.serialHint': 'Result: this project kind has no display, only text. In Windows, find the board\'s COM port under "Device Manager → Ports (COM & LPT)," then connect with PuTTY/Tera Term at 115200 8N1.\n',
    'mcu.flashResult.image.msc': 'Result: the Windows Camera app shows the video with the recognized label in the top-left corner; the COM port (115200 8N1) prints text too.',
    'mcu.flashResult.image.nulink': 'Result: the LCD shows the recognized label; the Nu-Link virtual COM port (115200 8N1) prints text too.',
    'mcu.flashResult.knownSound': 'At boot, one INFO threshold[class name] … line prints per class; after that, one live rms … dBFS | class_name=0.123 … line prints every 0.5 s (one independent confidence score per class).',
    'mcu.flashResult.audio': 'After that, one KWS hop=… rms=… top=label 0.876 … drop=0 | label=0.876* … line prints every 0.25 s; the * after a score marks which class this hop was assigned to, and drop being nonzero means the log dropped characters; an extra KWS DETECTED: label=… line prints when a detection fires.',
    'mcu.kindBlocked': 'This project kind cannot be deployed to a dev board.',
    'mcu.staleBuild': 'The model has changed — please rebuild.',
    'mcu.flashMethodLabel.msc': 'USB drive bootloader',
    'mcu.flashMethodLabel.nulink': 'Nu-Link',
    'mcu.flashMethodLabel.pyocd': 'pyocd (external Nu-Link2)',
    'mcu.missing.toolkit': 'The mcu_toolkit dev-board toolkit (firmware templates)',
    'mcu.missing.vela': 'The Vela compiler',
    'mcu.missing.toolchain': 'The Arm GNU Toolchain (arm-none-eabi-gcc)',
    'mcu.missing.boardList': 'The dev board list mcu_toolkit\\boards.json',
    'mcu.missing.generic': 'The dev board toolchain',
    'mcu.toolchain.probing': 'Detecting the dev board toolchain…',
    'mcu.toolchain.notReady': 'Not ready to deploy yet: missing {missing}. Install the Arm GNU Toolchain (arm-none-eabi-gcc) first, then rerun 01_INSTALL.bat.',
    'mcu.building': 'Building…',
    'mcu.toast.noBoard': 'No dev board configuration is available right now.',
    'mcu.busy.creatingJob': 'Creating the firmware build job…',
    'mcu.toast.startFailed': 'Could not start the firmware build: {message}',
    'mcu.toast.buildComplete': 'Firmware build complete — download the ZIP or flash it directly',
    'mcu.log.buildFailed': 'Build failed: {detail}',
    'mcu.toast.buildFailed': 'Firmware build failed: {detail} (click "Download Log" in the top right)',
    'mcu.log.statusFailed': 'Failed to read build status: {message}',
    'mcu.toast.statusFailed': 'Failed to read the firmware build status: {message}',
    'mcu.busy.preparingZip': 'Preparing the firmware ZIP…',
    'mcu.toast.zipDownloaded': 'Firmware ZIP downloaded',
    'mcu.toast.downloadFailed': 'Firmware download failed: {message}',
    'mcu.confirm.flashMessage': 'About to flash {name} ({size} bytes) to {label} — continue?',
    'mcu.confirm.flashTitle': 'Flash to Dev Board',
    'mcu.flashing': 'Flashing…',
    'mcu.flashDoneDefault': 'Flashing complete — press Reset on the board once.',
    'mcu.toast.flashComplete': 'Flashing complete',
    'mcu.toast.buildContinuesInBackground': 'The firmware build keeps running in the background; reopen Export Model to see its progress.',
  }),
});

const LANG_STORAGE_KEY = 'tm-local-lang';

// A private window, or a browser with site data blocked, makes localStorage throw on
// access (not just on write) -- both directions must be wrapped, or the page never boots.
function readStoredLang() {
  try {
    const stored = localStorage.getItem(LANG_STORAGE_KEY);
    return stored === 'en' || stored === 'zh' ? stored : null;
  } catch {
    return null;
  }
}

function writeStoredLang(lang) {
  try { localStorage.setItem(LANG_STORAGE_KEY, lang); } catch { /* non-fatal: see above. */ }
}

// Every key above is defined in `zh`; `en` only needs the ones that have been translated
// so far. Falling back to the zh entry (never to the bare key) means a missing en string
// reads as Chinese, not as a literal like "btn.train" on screen.
// `params` (phase 2): an optional {name: value} map substituted into `{name}` placeholders
// in the resolved string -- used instead of splitting a sentence into concatenated
// fragments, because a sentence cut at a variable cannot be reworded for a language with
// different word order.
function t(key, params) {
  const lang = state.lang === 'en' ? 'en' : 'zh';
  let text = I18N[lang][key] ?? I18N.zh[key] ?? key;
  if (params) {
    for (const name of Object.keys(params)) {
      text = text.replaceAll(`{${name}}`, params[name]);
    }
  }
  return text;
}

// `、` in zh, `, ` in en -- every place that joins a list of names/values for display goes
// through this instead of a hardcoded join('、'), so the join reads naturally in English.
function listJoin(items) {
  return items.join(t('common.listSeparator'));
}

// Several module-level option tables below (DEPLOYMENT_TARGET_OPTIONS, AUGMENTATION_LEVEL_
// OPTIONS, MIXUP_RATIO_CHOICES, KNOWN_SOUND_DEPTH_CHOICES) are frozen at load time -- before
// a student can ever toggle language -- so they store an i18n *key* as the label, not text.
// This resolves the key to the current language's text at the point of use (inside a render
// function, called after boot), which is the only time state.lang is meaningful.
function localizeOptions(pairs) {
  return pairs.map(([value, key]) => [value, t(key)]);
}

// Walks every element carrying an i18n data attribute and (re)applies the active language.
// Called once at boot and again on every toggle click. Elements whose text app.js also sets
// dynamically at runtime (save state, server status, training hint/accuracy, busy overlay,
// confirm dialog...) get blindly reset to their static placeholder by the loop above, so the
// re-render calls below repaint each one from current app state -- that is what makes a
// mid-session toggle show the *current* dynamic text in the new language instead of reverting.
function applyI18n() {
  document.documentElement.lang = state.lang === 'en' ? 'en' : 'zh-Hant';
  $$('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  $$('[data-i18n-title]').forEach(el => { el.title = t(el.dataset.i18nTitle); });
  $$('[data-i18n-aria-label]').forEach(el => { el.setAttribute('aria-label', t(el.dataset.i18nAriaLabel)); });
  $$('[data-i18n-placeholder]').forEach(el => { el.placeholder = t(el.dataset.i18nPlaceholder); });
  $$('#langToggle .lang-option').forEach(button => {
    button.classList.toggle('active', button.dataset.lang === state.lang);
  });
  renderServerStatus();
  renderSaveState();
  updateMicrophoneSettingsDisplay();
  if (state.project) renderTrainingPanel();
  renderConfirmDialog();
  renderBusyText();
  renderMcuTab();
}

function setLang(lang) {
  state.lang = lang === 'en' ? 'en' : 'zh';
  writeStoredLang(state.lang);
  applyI18n();
}

const PROJECT_KIND_META = Object.freeze({
  image: { label: 'Image', icon: '▧' },
  audio: { label: 'Audio', icon: '≋' },
  abnormal_sound: { label: 'Abnormal Sound', icon: '⌁!' },
  known_sound: { label: 'Known Sound', icon: '♪?' },
});

const NORMAL_TRAIN_ROLE = 'normal_train';
const ANOMALY_EVAL_ROLE = 'anomaly_eval';
const SYSTEM_DEFAULT_MICROPHONE = '__system_default__';

// --------------------------------------------------------------------------------------
// Advanced training panel. Every entry below mirrors a constant in tm_local/config.py; the
// server is still the authority (it re-validates every key and answers 400 in Traditional
// Chinese), these copies only stop the panel from OFFERING a value the server will reject.
// --------------------------------------------------------------------------------------

// config.DEPLOYMENT_TARGETS -- second element of each pair is an i18n key, not text: this
// array is frozen at load time, before a student can ever toggle language, so the actual
// label is only resolved at render time via localizeOptions(). See localizeOptions() above.
const DEPLOYMENT_TARGET_OPTIONS = Object.freeze([
  ['pc', 'options.deploymentTarget.pc'],
  ['NuMaker-M55M1', 'options.deploymentTarget.numakerM55m1'],
  ['NuGestureAI-M55M1', 'options.deploymentTarget.gestureai'],
  ['NuMaker-VoiceAI-M55M1', 'options.deploymentTarget.voiceai'],
]);
// config.MCU_BOARD_KINDS。沒有相機的板子不會出現在 image 專案的下拉裡。這只是不去「提供」
// 一個伺服器會拒絕的值；真正的把關在 config.validate_*_settings() 與 contract.validate()。
const MCU_BOARD_KINDS = Object.freeze({
  'NuMaker-M55M1': ['image', 'audio', 'known_sound'],
  'NuGestureAI-M55M1': ['image', 'audio', 'known_sound'],
  'NuMaker-VoiceAI-M55M1': ['audio', 'known_sound'],
});
// config.AUGMENTATION_LEVELS. 「（預設）」刻意不寫死在這裡：同一組層級被兩個設定共用，而它們的
// 預設值不同（image/audio 的 augmentation_level 是 medium，known_sound 的
// waveform_augment_level 是 off），標在清單裡就一定會對其中一個說謊。由 augmentationSelect()
// 依呼叫端傳進來的預設值即時標記。
const AUGMENTATION_LEVEL_OPTIONS = Object.freeze([
  ['off', 'common.off'], ['light', 'options.augmentation.light'],
  ['medium', 'options.augmentation.medium'], ['strong', 'options.augmentation.strong'],
]);
// config.MCU_IMAGE_SIZES — only these compile into the camera firmware.
const MCU_IMAGE_SIZES = Object.freeze([224, 192, 160, 128, 96]);
// PC-only projects may use any multiple of 32 between 96 and 320 (validate_image_settings).
const PC_IMAGE_SIZES = Object.freeze([320, 288, 256, 224, 192, 160, 128, 96]);
// config.MCU_MEL_BINS — the filterbank tables only exist at these two widths, board or not.
const MEL_BIN_CHOICES = Object.freeze([40, 64]);
// config.KNOWN_SOUND_DEFAULTS.mixup_ratio 的三個選項。validate_known_sound_settings 只擋負數，
// 所以 API 或匯入的專案可能帶著 0.75 這種不在清單上的值進來；走 optionMarkup() 才能和其他下拉
// 一樣「選不到就退回預設值並說一聲」，而不是被瀏覽器默默選成第一個（關閉）。
const MIXUP_RATIO_CHOICES = Object.freeze([
  [0, 'common.off'], [0.5, 'options.mixup.recommended'], [1, 'options.mixup.full'],
]);
// config.MCU_AUDIO_FRONTEND_LOCK.sample_rate — a board build is compiled against 16 kHz.
const MCU_SAMPLE_RATE = 16000;
// Fallback copy of config.KNOWN_SOUND_BOARD_MAX_DEPTH (measured post-Vela flash budget).
// /api/mcu/status reports the same numbers per board and wins once it has loaded; this
// literal is what a student sees if they open Advanced before that request lands.
// tests/test_distribution.py parses it and compares it with config.py so it cannot drift.
const KNOWN_SOUND_BOARD_MAX_DEPTH = Object.freeze({ 'NuMaker-M55M1': 12, 'NuGestureAI-M55M1': 11, 'NuMaker-VoiceAI-M55M1': 11 });
// Post-Vela flash on NuMaker-GestureAI-M55M1 (ethos-u55-256 / Shared_Sram), measured.
// Tensor-arena SRAM is ~151 KB at every depth, so this trades flash only.
const KNOWN_SOUND_DEPTH_CHOICES = Object.freeze([
  [14, 'options.knownSoundDepth.full'],
  [12, 'options.knownSoundDepth.12'],
  [11, 'options.knownSoundDepth.11'],
  [10, 'options.knownSoundDepth.10'],
  [9, 'options.knownSoundDepth.9'],
  [8, 'options.knownSoundDepth.8'],
  [7, 'options.knownSoundDepth.7'],
  [6, 'options.knownSoundDepth.6'],
]);

const state = {
  lang: readStoredLang() || 'zh',
  // Cached results of the last runtime render for elements app.js sets dynamically, so a
  // language toggle can repaint them in the new language instead of losing the live value.
  health: null,
  saveState: 'saved',
  busy: null,
  confirm: null,
  projects: [],
  project: null,
  activePanel: null,
  jobTimer: null,
  activeJobId: null,
  exportJobTimer: null,
  activeExportJobId: null,
  exportStartedAt: 0,
  cameraStream: null,
  imageCaptureTimer: null,
  imageCaptureBlobs: [],
  imageCaptureStartedAt: 0,
  imageCaptureClassId: null,
  audioStream: null,
  audioContext: null,
  audioSource: null,
  audioProcessor: null,
  audioAnalyser: null,
  audioSilentGain: null,
  audioRecording: false,
  audioRecordedChunks: [],
  audioRecordStartedAt: 0,
  audioRecordTimer: null,
  audioRecordingClassId: null,
  audioRecordingSessionId: null,
  audioDeviceId: '',
  audioDevices: [],
  audioTrackSettings: null,
  microphoneRefreshBusy: false,
  audioRollingChunks: [],
  audioRollingSamples: 0,
  audioAnimation: null,
  previewTimer: null,
  previewBusy: false,
  previewSessionId: null,
  abnormalPreviewVotes: [],
  previewImageFile: null,
  // 目前 #trainingOptions 面板畫的是哪一個專案。null = 面板還沒畫過（或畫的是別的專案），
  // 這時候不可以去讀面板上的值，見 trainingOptionsDraft()。
  trainingOptionsProjectId: null,
  confirmResolver: null,
  mcuStatus: null,
  mcuStatusPending: null,
  mcuProjectId: null,
  mcuError: '',
  mcuResult: null,
  mcuDownloadUrl: '',
  mcuDownloadName: '',
  deployJobTimer: null,
  activeDeployJobId: null,
  pendingDeployJobId: null,
};

const elements = {
  homeView: $('#homeView'),
  projectView: $('#projectView'),
  projectHeaderTools: $('#projectHeaderTools'),
  projectNameInput: $('#projectNameInput'),
  saveState: $('#saveState'),
  recentProjects: $('#recentProjects'),
  emptyProjects: $('#emptyProjects'),
  classStack: $('#classStack'),
  projectKindBadge: $('#projectKindBadge'),
  projectSampleSummary: $('#projectSampleSummary'),
  trainingIdle: $('#trainingIdle'),
  trainingProgress: $('#trainingProgress'),
  trainingDone: $('#trainingDone'),
  trainingHint: $('#trainingHint'),
  trainingMessage: $('#trainingMessage'),
  trainingPercent: $('#trainingPercent'),
  trainingProgressBar: $('#trainingProgressBar'),
  trainingAccuracy: $('#trainingAccuracy'),
  trainingOptions: $('#trainingOptions'),
  trainButton: $('#trainButton'),
  openExportButton: $('#openExportButton'),
  previewLocked: $('#previewLocked'),
  previewReady: $('#previewReady'),
  imagePreviewPane: $('#imagePreviewPane'),
  audioPreviewPane: $('#audioPreviewPane'),
  predictionBars: $('#predictionBars'),
  predictionNote: $('#predictionNote'),
  previewInputToggle: $('#previewInputToggle'),
  previewToggleText: $('#previewToggleText'),
  previewRuntime: $('#previewRuntime'),
  previewVideo: $('#previewVideo'),
  previewCanvas: $('#previewCanvas'),
  liveSpectrogramCanvas: $('#liveSpectrogramCanvas'),
  audioLiveText: $('#audioLiveText'),
  microphoneDeviceControls: $('#microphoneDeviceControls'),
  microphoneDeviceSelect: $('#microphoneDeviceSelect'),
  microphoneTrackSettings: $('#microphoneTrackSettings'),
  connectorSvg: $('#connectorSvg'),
  busyOverlay: $('#busyOverlay'),
  busyText: $('#busyText'),
  toastRegion: $('#toastRegion'),
  exportModal: $('#exportModal'),
  mcuBoardSelect: $('#mcuBoardSelect'),
  mcuToolchainNotice: $('#mcuToolchainNotice'),
  mcuKindNotice: $('#mcuKindNotice'),
  mcuErrorNotice: $('#mcuErrorNotice'),
  mcuLastBuild: $('#mcuLastBuild'),
  deployMcuButton: $('#deployMcuButton'),
  mcuProgress: $('#mcuProgress'),
  mcuProgressBar: $('#mcuProgressBar'),
  mcuProgressMessage: $('#mcuProgressMessage'),
  deployLog: $('#deployLog'),
  mcuResult: $('#mcuResult'),
  mcuResultSummary: $('#mcuResultSummary'),
  mcuFlashMethodRow: $('#mcuFlashMethodRow'),
  mcuFlashMethodSelect: $('#mcuFlashMethodSelect'),
  mcuFlashInstructions: $('#mcuFlashInstructions'),
  mcuDownloadButton: $('#mcuDownloadButton'),
  mcuFlashButton: $('#mcuFlashButton'),
  mcuFlashStatus: $('#mcuFlashStatus'),
};

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatDate(value) {
  try {
    const locale = state.lang === 'en' ? 'en-US' : 'zh-TW';
    return new Intl.DateTimeFormat(locale, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value));
  } catch (_) {
    return value || '';
  }
}

function formatPercent(value) {
  const percent = Math.max(0, Math.min(1, Number(value) || 0)) * 100;
  if (percent >= 99.95) return '100%';
  if (percent <= 0) return '0%';
  if (percent < 0.05) return '<0.1%';
  return `${percent.toFixed(1)}%`;
}

function projectKindMeta(kind) {
  return PROJECT_KIND_META[kind] || { label: String(kind || 'Project'), icon: '?' };
}

function isAbnormalSoundProject(project = state.project) {
  return project?.kind === 'abnormal_sound';
}

function isKnownSoundProject(project = state.project) {
  return project?.kind === 'known_sound';
}

function isAudioLikeProject(project = state.project) {
  return project?.kind === 'audio' || project?.kind === 'abnormal_sound' || project?.kind === 'known_sound';
}

// Known Sound scores are independent sigmoids, so several classes can be "detected" at
// once and the bars deliberately do not add up to 100%.
function knownSoundThreshold(project = state.project) {
  const fromReport = Number(project?.training?.report?.settings?.detection_threshold);
  if (Number.isFinite(fromReport) && fromReport > 0 && fromReport < 1) return fromReport;
  const fromSettings = Number(project?.settings?.detection_threshold);
  return Number.isFinite(fromSettings) && fromSettings > 0 && fromSettings < 1 ? fromSettings : 0.5;
}

function usableThreshold(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 && number < 1 ? number : fallback;
}

// One threshold per class, in project class order. A rare alarm and a constant background
// hum do not deserve the same decision point, so `detection_threshold` is only the default
// that per-class overrides fall back to. The trained report wins over project settings:
// Preview must decide with exactly the thresholds the model was evaluated with.
function knownSoundThresholds(project = state.project) {
  const classes = project?.classes || [];
  const base = knownSoundThreshold(project);
  const reportSettings = project?.training?.report?.settings || {};
  const byLabel = reportSettings.class_thresholds_by_label;
  if (Array.isArray(byLabel) && byLabel.length === classes.length) {
    return byLabel.map(value => usableThreshold(value, base));
  }
  const stored = reportSettings.class_thresholds || project?.settings?.class_thresholds || {};
  const overrides = stored && typeof stored === 'object' ? stored : {};
  return classes.map(item => usableThreshold(overrides[item.id], base));
}

// Index of the class the model treats as "nothing was said", or -1. Written by training;
// `Number(null)` is 0, so a missing value must never be coerced into class 0.
function knownSoundBackgroundIndex(project = state.project) {
  const raw = project?.training?.report?.settings?.background_class_index;
  if (raw === null || raw === undefined || raw === '') return -1;
  const index = Number(raw);
  return Number.isInteger(index) && index >= 0 ? index : -1;
}

function knownSoundPeakHoldMs(project = state.project) {
  const seconds = Number(project?.settings?.preview_peak_hold_seconds);
  return (Number.isFinite(seconds) && seconds >= 0 ? seconds : 1.5) * 1000;
}

// A gunshot lasts 100-200 ms. Without a hold the bar spikes between animation frames and
// the user never sees it, so each class keeps its highest recent score for a short while.
function applyKnownSoundPeakHold(predictions) {
  const now = Date.now();
  const holdMs = knownSoundPeakHoldMs();
  const held = state.knownSoundPeaks || (state.knownSoundPeaks = new Map());
  const output = [];
  for (const item of predictions || []) {
    const name = item.class_name;
    const score = Math.max(0, Math.min(1, Number(item.score) || 0));
    const previous = held.get(name);
    if (previous && previous.score > score && now - previous.at < holdMs) {
      output.push({ ...item, score: previous.score, held: true });
      continue;
    }
    held.set(name, { score, at: now });
    output.push({ ...item, score, held: false });
  }
  return output;
}

function projectClassRole(classItem, index = 0, project = state.project) {
  if (!isAbnormalSoundProject(project)) return '';
  const role = String(classItem?.role || '').trim().toLowerCase();
  if (role === NORMAL_TRAIN_ROLE || role === ANOMALY_EVAL_ROLE) return role;
  return index === 0 ? NORMAL_TRAIN_ROLE : ANOMALY_EVAL_ROLE;
}

function normalTrainingClass(project = state.project) {
  if (!isAbnormalSoundProject(project)) return null;
  return project.classes?.find((item, index) => projectClassRole(item, index, project) === NORMAL_TRAIN_ROLE) || null;
}

function classSessionCount(classItem) {
  const explicit = Number(classItem?.recording_session_count ?? classItem?.session_count);
  if (Number.isFinite(explicit) && explicit >= 0) return explicit;
  const keys = new Set();
  for (const sample of classItem?.samples || []) {
    const key = sample.recording_session_id || sample.session_id || sample.source_name;
    if (key) keys.add(String(key));
  }
  return keys.size;
}

function abnormalTrainingReadiness(project = state.project) {
  const normal = normalTrainingClass(project);
  const settings = project?.settings || {};
  const server = project?.abnormal_readiness || project?.data_readiness || {};
  const clips = Number(server.normal_clips ?? normal?.sample_count ?? 0);
  const clipSeconds = Math.max(0.1, Number(settings.clip_seconds || 1));
  const seconds = Number(server.normal_seconds ?? normal?.duration_seconds ?? normal?.total_duration_seconds ?? clips * clipSeconds);
  const sessions = Number(server.normal_sessions ?? classSessionCount(normal));
  const minimumSeconds = Math.max(clipSeconds, Number(server.minimum_train_seconds ?? settings.minimum_train_seconds ?? clipSeconds));
  const minimumSessions = Math.max(1, Number(server.minimum_train_sessions ?? settings.minimum_train_sessions ?? 1));
  const calibrationSeconds = Math.max(minimumSeconds, Number(
    server.minimum_calibration_seconds ?? settings.minimum_calibration_seconds ?? minimumSeconds,
  ));
  const calibrationSessions = Math.max(minimumSessions, Number(
    server.minimum_calibration_sessions ?? settings.minimum_calibration_sessions ?? minimumSessions,
  ));
  const serverReady = server.can_train;
  const ready = typeof serverReady === 'boolean'
    ? serverReady
    : Boolean(normal && clips > 0 && seconds >= minimumSeconds && sessions >= minimumSessions);
  const calibrationCandidate = typeof server.calibration_candidate === 'boolean'
    ? server.calibration_candidate
    : seconds >= calibrationSeconds && sessions >= calibrationSessions;
  return {
    normal, clips, seconds, sessions, minimumSeconds, minimumSessions, ready,
    calibrationSeconds, calibrationSessions, calibrationCandidate, note: String(server.note || ''),
  };
}

function newRecordingSessionId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function readSavedMicrophoneId() {
  try {
    const value = localStorage.getItem('tm-local-microphone-device-id');
    return value === SYSTEM_DEFAULT_MICROPHONE ? '' : (value || '');
  }
  catch (_) { return ''; }
}

function hasSavedMicrophonePreference() {
  try { return localStorage.getItem('tm-local-microphone-device-id') !== null; }
  catch (_) { return false; }
}

function saveMicrophoneId(deviceId) {
  try {
    localStorage.setItem('tm-local-microphone-device-id', deviceId || SYSTEM_DEFAULT_MICROPHONE);
  } catch (_) {}
}

function toast(message, type = '') {
  const node = document.createElement('div');
  node.className = `toast ${type}`.trim();
  node.textContent = message;
  elements.toastRegion.appendChild(node);
  setTimeout(() => node.remove(), 5200);
}

// `key`/`params` (not raw text) so a language toggle mid-busy can re-render the same message
// in the new language -- see state.busy and renderBusyText() below.
function setBusy(active, key = 'common.busy', params = null) {
  state.busy = active ? { key, params } : null;
  elements.busyText.textContent = t(key, params);
  elements.busyOverlay.classList.toggle('hidden', !active);
}

// For a busy message that updates several times while the overlay stays open (a progress
// counter), without toggling the overlay itself on/off.
function updateBusyText(key, params) {
  state.busy = { key, params };
  elements.busyText.textContent = t(key, params);
}

function renderBusyText() {
  if (state.busy) elements.busyText.textContent = t(state.busy.key, state.busy.params);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch (_) {
      detail = (await response.text()) || detail;
    }
    throw new Error(detail);
  }
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) return response.json();
  return response;
}

function jsonOptions(method, payload) {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  };
}

async function checkHealth() {
  try {
    const data = await api('/api/health');
    const runtimeLabel = data.runtime?.backend_label || 'CPU';
    const gpuName = Array.isArray(data.runtime?.gpu_names) ? data.runtime.gpu_names[0] : '';
    state.health = { online: true, version: data.version, runtimeLabel, gpuName, reason: data.runtime?.reason };
  } catch (error) {
    state.health = { online: false };
  }
  renderServerStatus();
}

// Repaints #serverStatus from the last checkHealth() result cached in state.health, instead
// of re-fetching -- called both by checkHealth() itself and by applyI18n() on every language
// toggle, so a toggle mid-session shows the current connection state in the new language.
function renderServerStatus() {
  const health = state.health;
  if (!health) return;
  const statusEl = $('#serverStatus span');
  if (health.online) {
    $('#serverStatus').classList.add('online');
    statusEl.textContent = `Local v${health.version} · ${health.runtimeLabel}`;
    const pill = $('#runtimePill');
    if (pill) {
      const detail = health.gpuName ? `${health.runtimeLabel} · ${health.gpuName}` : health.runtimeLabel;
      pill.textContent = `LOCAL · PRIVATE · PYTHON 3.13 · ${detail}`;
      pill.title = health.reason || detail;
    }
  } else {
    $('#serverStatus').classList.remove('online');
    statusEl.textContent = t('header.status.offline');
  }
}

function routeProjectId() {
  const match = location.hash.match(/^#\/project\/([0-9a-f-]+)$/i);
  return match ? match[1] : null;
}

function goHome() {
  location.hash = '#/';
}

function goProject(projectId) {
  location.hash = `#/project/${projectId}`;
}

async function handleRoute() {
  stopPreview();
  const projectId = routeProjectId();
  if (!projectId) {
    state.project = null;
    state.activePanel = null;
    discardTrainingOptionsDraft();
    resetMcuStateForProject(null);
    elements.projectView.classList.add('hidden');
    elements.homeView.classList.remove('hidden');
    elements.projectHeaderTools.classList.add('hidden');
    $('#exportProjectButton').classList.add('hidden');
    $('#deleteProjectButton').classList.add('hidden');
    await loadProjects();
    return;
  }
  elements.homeView.classList.add('hidden');
  elements.projectView.classList.remove('hidden');
  elements.projectHeaderTools.classList.remove('hidden');
  $('#exportProjectButton').classList.remove('hidden');
  $('#deleteProjectButton').classList.remove('hidden');
  await loadProject(projectId);
}

async function loadProjects() {
  try {
    const payload = await api('/api/projects');
    state.projects = payload.projects || [];
    renderRecentProjects();
  } catch (error) {
    toast(t('header.toast.loadProjectsFailed', { message: error.message }), 'error');
  }
}

function renderRecentProjects() {
  elements.recentProjects.innerHTML = '';
  elements.emptyProjects.classList.toggle('hidden', state.projects.length > 0);
  for (const project of state.projects) {
    const kindMeta = projectKindMeta(project.kind);
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'recent-project-card';
    card.innerHTML = `
      <span class="recent-top">
        <span class="recent-kind-icon ${escapeHtml(project.kind)}" title="${escapeHtml(kindMeta.label)} Project">${kindMeta.icon}</span>
        <small>${project.training?.state === 'trained' ? 'Trained' : 'Untrained'}</small>
      </span>
      <strong>${escapeHtml(project.name)}</strong>
      <span class="recent-stats"><span>${project.total_samples || 0} samples</span><span>${escapeHtml(formatDate(project.updated_at))}</span></span>`;
    card.addEventListener('click', () => goProject(project.id));
    elements.recentProjects.appendChild(card);
  }
}

async function createProject(kind) {
  setBusy(true, 'header.busy.creatingProject', { kind: projectKindMeta(kind).label });
  try {
    const project = await api('/api/projects', jsonOptions('POST', { kind }));
    goProject(project.id);
  } catch (error) {
    toast(t('header.toast.createProjectFailed', { message: error.message }), 'error');
  } finally {
    setBusy(false);
  }
}

async function loadProject(projectId, { quiet = false } = {}) {
  try {
    const project = await api(`/api/projects/${projectId}`);
    state.project = project;
    // 伺服器的副本剛換掉，面板上那份暫存值（可能還是上一個專案的）就此作廢。
    discardTrainingOptionsDraft();
    resetMcuStateForProject(project.id);
    renderProject();
    if (project.active_job && ['queued', 'running'].includes(project.active_job.state)) {
      if (project.active_job.job_type === 'export') {
        state.activeExportJobId = project.active_job.id;
        state.exportStartedAt = Date.now();
        setBusy(true, 'export.busy.resuming');
        pollExportJob(project.active_job.id);
      } else if (project.active_job.job_type === 'deploy') {
        showExportModal();
        switchExportTab('mcu');
        pollDeployJob(project.active_job.id);
      } else {
        pollJob(project.active_job.id);
      }
    }
  } catch (error) {
    if (!quiet) toast(t('header.toast.openProjectFailed', { message: error.message }), 'error');
    goHome();
  }
}

function trainingState() {
  return state.project?.training?.state || 'untrained';
}

function classMinimum(project = state.project) {
  if (isAbnormalSoundProject(project)) {
    const settings = project?.settings || {};
    const clipSeconds = Math.max(0.1, Number(settings.clip_seconds || 1));
    return Math.max(1, Math.ceil(Number(settings.minimum_train_seconds || clipSeconds) / clipSeconds));
  }
  if (isKnownSoundProject(project)) {
    return Number(project?.settings?.minimum_clips_per_class || 20);
  }
  return Number(project?.minimum_samples_per_class || project?.settings?.minimum_samples_per_class || (project?.kind === 'audio' ? 8 : 5));
}

// Known Sound also needs several INDEPENDENT recording sessions per class, because a
// single 20 s take sliced into 20 clips is one example, not twenty, and cannot be split
// into train/validation without leaking.
function knownSoundSessionMinimum(project = state.project) {
  return Number(project?.settings?.minimum_sessions_per_class || 2);
}

function canTrain() {
  if (!state.project) return false;
  if (isAbnormalSoundProject()) return abnormalTrainingReadiness().ready;
  if (state.project.classes.length < 2) return false;
  const minimum = classMinimum();
  if (isKnownSoundProject()) {
    const sessionMinimum = knownSoundSessionMinimum();
    return state.project.classes.every(item => (
      Number(item.sample_count || 0) >= minimum && classSessionCount(item) >= sessionMinimum
    ));
  }
  return state.project.classes.every(item => Number(item.sample_count || 0) >= minimum);
}

// image/audio only: INT8 vs Float top-1 agreement warning(s) written by
// export_service.ensure_export_artifacts() into training.report.conversion.warnings.
// known_sound (multi-label) and abnormal_sound (no classifier comparison) never carry this.
function exportAgreementWarnings(project = state.project) {
  const warnings = project?.training?.report?.conversion?.warnings;
  return Array.isArray(warnings) ? warnings : [];
}

function renderExportWarningNotice(project = state.project) {
  const notice = $('#exportWarningNotice');
  const warnings = exportAgreementWarnings(project);
  notice.textContent = warnings.join(' ');
  notice.classList.toggle('hidden', warnings.length === 0);
}

function renderProject() {
  const project = state.project;
  if (!project) return;
  const kindMeta = projectKindMeta(project.kind);
  const abnormal = isAbnormalSoundProject(project);
  elements.projectNameInput.value = project.name;
  elements.projectKindBadge.textContent = `${kindMeta.label.toUpperCase()} PROJECT`;
  if (abnormal) {
    const readiness = abnormalTrainingReadiness(project);
    const evaluationSamples = (project.classes || []).reduce((total, item, index) => (
      projectClassRole(item, index, project) === ANOMALY_EVAL_ROLE
        ? total + Number(item.sample_count || 0)
        : total
    ), 0);
    elements.projectSampleSummary.textContent = `${readiness.clips} Normal samples · ${readiness.sessions} sessions · ${evaluationSamples} evaluation samples`;
  } else {
    elements.projectSampleSummary.textContent = `${project.total_samples || 0} samples · ${project.classes.length} classes`;
  }
  $('#addClassButton').innerHTML = abnormal
    ? '<span>⊞</span> Add anomaly evaluation group'
    : '<span>⊞</span> Add a class';
  renderClasses();
  renderTrainingOptions(trainingOptionsDraft());
  renderTrainingPanel();
  renderPreviewPanel();
  $('#audioExportNotice').classList.toggle('hidden', project.kind !== 'audio');
  $('#abnormalSoundExportNotice').classList.toggle('hidden', !abnormal);
  renderExportWarningNotice(project);
  elements.microphoneDeviceControls.classList.toggle('hidden', !isAudioLikeProject(project));
  setTimeout(() => {
    if (isAudioLikeProject(project)) refreshMicrophoneDevices({ silent: true });
    restoreActivePanelMedia();
    drawConnectors();
  }, 20);
}

function renderClasses() {
  const project = state.project;
  const abnormal = isAbnormalSoundProject(project);
  const readiness = abnormal ? abnormalTrainingReadiness(project) : null;
  elements.classStack.innerHTML = '';
  for (const [index, classItem] of project.classes.entries()) {
    const role = projectClassRole(classItem, index, project);
    const isNormal = role === NORMAL_TRAIN_ROLE;
    const card = document.createElement('article');
    card.className = `class-card${abnormal ? ` abnormal-role-${role}` : ''}`;
    card.dataset.classId = classItem.id;
    if (role) card.dataset.role = role;
    card.style.setProperty('--class-color', classItem.color);
    const minimum = classMinimum(project);
    const samples = classItem.samples || [];
    const sampleHtml = samples.length
      ? samples.map(sample => `
          <span class="sample-thumb" data-sample-id="${escapeHtml(sample.id)}">
            <img src="${escapeHtml(sample.thumbnail_url)}" alt="sample" loading="lazy">
            <button type="button" data-delete-sample="${escapeHtml(sample.id)}" title="${t('workspace.sample.deleteTitle')}">×</button>
          </span>`).join('')
      : `<span class="empty-sample-strip">${t('workspace.class.noSamples')}</span>`;
    const sourceLabel = project.kind === 'image' ? ['▣', 'Webcam'] : ['♩', 'Mic'];
    const active = state.activePanel?.classId === classItem.id;
    let roleBadge = '';
    let sampleSummary = `
      <strong>${classItem.sample_count || 0} ${project.kind === 'image' ? 'Image' : 'Audio'} Samples <span>/ ${minimum} minimum</span></strong>
      <small>${classItem.sample_count >= minimum
        ? t('workspace.class.ready')
        : t('workspace.class.addMore', { count: Math.max(0, minimum - classItem.sample_count) })}</small>`;
    if (abnormal && isNormal) {
      const seconds = Number(classItem.duration_seconds ?? classItem.total_duration_seconds ?? Number(classItem.sample_count || 0) * Number(project.settings?.clip_seconds || 1));
      const sessions = classSessionCount(classItem);
      roleBadge = '<span class="class-role-badge normal">NORMAL BASELINE · TRAINING</span>';
      sampleSummary = `
        <strong>${classItem.sample_count || 0} Normal Audio Samples</strong>
        <small>${t('workspace.class.normalSummary', {
          seconds: seconds.toFixed(0), minSeconds: readiness.minimumSeconds.toFixed(0),
          sessions, minSessions: readiness.minimumSessions,
        })}</small>`;
    } else if (abnormal) {
      roleBadge = '<span class="class-role-badge evaluation">ANOMALY EVALUATION ONLY</span>';
      sampleSummary = `
        <strong>${classItem.sample_count || 0} Evaluation Audio Samples</strong>
        <small>${t('workspace.class.evalNote')}</small>`;
    } else if (isKnownSoundProject(project)) {
      // Sessions matter as much as clip count here, and the difference is invisible
      // unless the card says so: 20 clips sliced from one take is ONE example.
      const sessions = classSessionCount(classItem);
      const sessionMinimum = knownSoundSessionMinimum(project);
      const clipsShort = Math.max(0, minimum - Number(classItem.sample_count || 0));
      const sessionsShort = Math.max(0, sessionMinimum - sessions);
      const todo = [];
      if (clipsShort) todo.push(t('workspace.class.todoClips', { count: clipsShort }));
      if (sessionsShort) todo.push(t('workspace.class.todoSessions', { count: sessionsShort }));
      sampleSummary = `
        <strong>${classItem.sample_count || 0} Audio Samples <span>/ ${minimum} minimum</span></strong>
        <small>${t('workspace.class.sessionSummary', {
          sessions, minimum: sessionMinimum,
          todo: todo.length
            ? t('workspace.class.todoPrefix', { items: listJoin(todo) })
            : t('workspace.class.ready'),
        })}</small>`;
    }
    card.innerHTML = `
      <div class="class-card-header">
        <input class="class-name-input" value="${escapeHtml(classItem.name)}" maxlength="80" aria-label="${abnormal ? t('workspace.class.groupNameAria') : t('workspace.class.nameAria')}" ${isNormal ? `readonly title="${t('workspace.class.normalLockedTitle')}"` : ''}>
        ${roleBadge}
        <button class="class-menu-button" type="button" aria-label="${t('workspace.class.menuAria')}">⋮</button>
        <div class="class-action-menu hidden">
          <button type="button" data-action="clear">Clear samples</button>
          ${isNormal ? '' : `<button type="button" data-action="delete" class="danger">${abnormal ? 'Delete evaluation group' : 'Delete class'}</button>`}
        </div>
      </div>
      <div class="class-card-body">
        <div class="class-summary-row">
          <div class="sample-count-copy">
            ${sampleSummary}
          </div>
          <div class="source-buttons">
            <button class="source-button ${active ? 'active' : ''}" type="button" data-open-capture="${classItem.id}"><b>${sourceLabel[0]}</b>${sourceLabel[1]}</button>
            <button class="source-button" type="button" data-open-upload="${classItem.id}"><b>↥</b>Upload</button>
          </div>
          <div class="sample-strip">${sampleHtml}</div>
        </div>
        ${active ? renderCapturePanel(classItem) : ''}
      </div>`;
    wireClassCard(card, classItem);
    elements.classStack.appendChild(card);
  }
}

function renderCapturePanel(classItem) {
  if (state.project.kind === 'image') {
    return `
      <div class="class-input-panel">
        <div class="capture-pane">
          <button class="class-panel-close" type="button" data-close-panel>×</button>
          <video id="classCaptureVideo" autoplay playsinline muted></video>
          <button id="holdCaptureButton" class="record-button" type="button">Hold to Record</button>
          <small class="capture-help">${t('workspace.capture.imageHelp', { name: escapeHtml(classItem.name) })}</small>
        </div>
        <div class="capture-results">
          <h4>Add Image Samples:</h4>
          <p id="captureResultText">${t('workspace.capture.imageResultDefault')}</p>
          <div class="capture-progress"><i id="captureProgressBar"></i></div>
        </div>
      </div>`;
  }
  const index = state.project.classes.findIndex(item => item.id === classItem.id);
  const role = projectClassRole(classItem, Math.max(0, index));
  const abnormal = isAbnormalSoundProject();
  const isNormal = role === NORMAL_TRAIN_ROLE;
  const heading = abnormal
    ? (isNormal ? 'Add Normal Baseline Audio:' : 'Add Evaluation-only Anomaly Audio:')
    : 'Add Audio Samples:';
  const help = abnormal
    ? (isNormal
      ? t('workspace.capture.audioHelp.normal')
      : t('workspace.capture.audioHelp.evaluation'))
    : t('workspace.capture.audioHelp.default');
  const resultHelp = abnormal
    ? (isNormal
      ? t('workspace.capture.audioResultHelp.normal')
      : t('workspace.capture.audioResultHelp.evaluation'))
    : t('workspace.capture.audioResultHelp.default');
  return `
    <div class="class-input-panel">
      <div class="capture-pane audio">
        <button class="class-panel-close" type="button" data-close-panel>×</button>
        <div class="audio-meter"><canvas id="classAudioMeter" width="360" height="150"></canvas></div>
        <button id="audioRecordButton" class="record-button" type="button">Record 20 Seconds</button>
        <small id="classMicrophoneSettings" class="capture-device-settings">${escapeHtml(formatMicrophoneSettings())}</small>
        <small class="capture-help">${help}</small>
      </div>
      <div class="capture-results">
        <h4>${heading}</h4>
        <p id="captureResultText">${resultHelp}</p>
        <div class="capture-progress"><i id="captureProgressBar"></i></div>
      </div>
    </div>`;
}

function wireClassCard(card, classItem) {
  const nameInput = card.querySelector('.class-name-input');
  const classIndex = state.project.classes.findIndex(item => item.id === classItem.id);
  const role = projectClassRole(classItem, Math.max(0, classIndex));
  const isNormal = role === NORMAL_TRAIN_ROLE;
  let originalName = classItem.name;
  async function saveClassName() {
    if (nameInput.readOnly) return;
    const value = nameInput.value.trim() || originalName;
    if (value === originalName) return;
    try {
      state.project = await api(`/api/projects/${state.project.id}/classes/${classItem.id}`, jsonOptions('PATCH', { name: value }));
      originalName = value;
      renderProject();
    } catch (error) {
      nameInput.value = originalName;
      toast(t('workspace.toast.renameClassFailed', { message: error.message }), 'error');
    }
  }
  if (!nameInput.readOnly) {
    nameInput.addEventListener('change', saveClassName);
    nameInput.addEventListener('keydown', event => { if (event.key === 'Enter') nameInput.blur(); });
  }

  const menu = card.querySelector('.class-action-menu');
  card.querySelector('.class-menu-button').addEventListener('click', event => {
    event.stopPropagation();
    $$('.class-action-menu').forEach(node => { if (node !== menu) node.classList.add('hidden'); });
    menu.classList.toggle('hidden');
  });
  menu.querySelector('[data-action="clear"]').addEventListener('click', async () => {
    menu.classList.add('hidden');
    const messageKey = isNormal ? 'workspace.confirm.clearNormalMessage' : 'workspace.confirm.clearClassMessage';
    const ok = await confirmDialog('workspace.confirm.clearTitle', null, messageKey, { name: classItem.name });
    if (!ok) return;
    await mutateProject(`/api/projects/${state.project.id}/classes/${classItem.id}/samples`, { method: 'DELETE' }, t('workspace.toast.samplesCleared'));
  });
  const deleteButton = menu.querySelector('[data-action="delete"]');
  if (deleteButton) {
    deleteButton.addEventListener('click', async () => {
      menu.classList.add('hidden');
      const noun = isAbnormalSoundProject() ? 'evaluation group' : t('workspace.noun.class');
      const ok = await confirmDialog('workspace.confirm.deleteNounTitle', { noun }, 'workspace.confirm.deleteNounMessage', { name: classItem.name });
      if (!ok) return;
      await mutateProject(`/api/projects/${state.project.id}/classes/${classItem.id}`, { method: 'DELETE' }, t('workspace.toast.nounDeleted', { noun }));
    });
  }

  card.querySelector('[data-open-capture]').addEventListener('click', () => {
    state.activePanel = state.activePanel?.classId === classItem.id ? null : { classId: classItem.id };
    renderProject();
  });
  card.querySelector('[data-open-upload]').addEventListener('click', () => openSampleUpload(classItem));
  card.querySelectorAll('[data-delete-sample]').forEach(button => {
    button.addEventListener('click', async event => {
      event.stopPropagation();
      const sampleId = button.dataset.deleteSample;
      try {
        state.project = await api(`/api/projects/${state.project.id}/samples/${sampleId}`, { method: 'DELETE' });
        renderProject();
      } catch (error) {
        toast(t('workspace.toast.deleteSampleFailed', { message: error.message }), 'error');
      }
    });
  });
  const close = card.querySelector('[data-close-panel]');
  if (close) close.addEventListener('click', () => { state.activePanel = null; renderProject(); });
}

async function mutateProject(path, options, successMessage = '') {
  setBusy(true);
  try {
    state.project = await api(path, options);
    renderProject();
    if (successMessage) toast(successMessage, 'success');
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    setBusy(false);
  }
}

function restoreActivePanelMedia() {
  if (!state.activePanel || !state.project) return;
  if (state.project.kind === 'image') setupImageCapturePanel();
  else setupAudioCapturePanel();
}

async function openSampleUpload(classItem) {
  const input = document.createElement('input');
  input.type = 'file';
  input.multiple = true;
  input.accept = state.project.kind === 'image' ? 'image/*' : 'audio/*,.wav,.mp3,.m4a,.ogg,.flac';
  input.addEventListener('change', async () => {
    if (!input.files?.length) return;
    if (state.project.kind === 'image') await uploadImageFiles(classItem.id, [...input.files]);
    else await uploadAudioFiles(classItem.id, [...input.files]);
  });
  input.click();
}

async function uploadImageFiles(classId, files) {
  setBusy(true, 'workspace.busy.addingImages', { count: files.length });
  try {
    for (let start = 0; start < files.length; start += 40) {
      const data = new FormData();
      for (const file of files.slice(start, start + 40)) data.append('files', file, file.name);
      const result = await api(`/api/projects/${state.project.id}/classes/${classId}/images`, { method: 'POST', body: data });
      state.project = result.project;
      if (result.errors?.length) toast(result.errors.slice(0, 3).join('\n'), 'error');
    }
    renderProject();
    toast(t('workspace.toast.imagesAdded'), 'success');
  } catch (error) {
    toast(t('workspace.toast.imageUploadFailed', { message: error.message }), 'error');
  } finally {
    setBusy(false);
  }
}

async function uploadAudioFiles(classId, files) {
  setBusy(true, 'workspace.busy.decodingAudio', { count: files.length });
  try {
    const wavEntries = [];
    for (let index = 0; index < files.length; index += 1) {
      updateBusyText('workspace.busy.decodingAudioProgress', { index: index + 1, total: files.length });
      const arrayBuffer = await files[index].arrayBuffer();
      const context = new (window.AudioContext || window.webkitAudioContext)();
      let decoded;
      try {
        decoded = await context.decodeAudioData(arrayBuffer.slice(0));
      } finally {
        await context.close();
      }
      const mono = mixAudioBufferToMono(decoded);
      const wavBlob = encodeWavBlob(mono, decoded.sampleRate);
      wavEntries.push({
        file: new File([wavBlob], `${files[index].name.replace(/\.[^.]+$/, '')}.wav`, { type: 'audio/wav' }),
        originalName: files[index].name,
        decodedSampleRate: decoded.sampleRate,
      });
    }
    let added = 0;
    const errors = [];
    if (isAbnormalSoundProject()) {
      // Each source file is one independent recording session. Sending one per request keeps
      // the session boundary unambiguous even when the server cuts it into many 1 s clips.
      for (let index = 0; index < wavEntries.length; index += 1) {
        updateBusyText('workspace.busy.addingAudioSession', { index: index + 1, total: wavEntries.length });
        const entry = wavEntries[index];
        const metadata = {
          recording_session_id: newRecordingSessionId(),
          source: 'upload',
          source_name: entry.originalName,
          device_label: 'Uploaded audio file',
          device_id: '',
          device_settings: { decoded_sample_rate: entry.decodedSampleRate },
        };
        const result = await postAudioWavFiles(classId, [entry.file], metadata);
        added += Number(result.added || 0);
        errors.push(...(result.errors || []));
      }
    } else {
      const wavFiles = wavEntries.map(entry => entry.file);
      for (let start = 0; start < wavFiles.length; start += 20) {
        const result = await postAudioWavFiles(classId, wavFiles.slice(start, start + 20));
        added += Number(result.added || 0);
        errors.push(...(result.errors || []));
      }
    }
    renderProject();
    if (errors.length) toast(errors.slice(0, 3).join('\n'), 'error');
    else toast(t('workspace.toast.audioSamplesAdded', { count: added }), 'success');
  } catch (error) {
    toast(t('workspace.toast.audioUploadFailed', { message: error.message }), 'error');
  } finally {
    setBusy(false);
  }
}

function currentAudioSessionMetadata(sessionId, source = 'microphone') {
  const track = state.audioStream?.getAudioTracks?.()[0];
  const settings = track?.getSettings?.() || state.audioTrackSettings || {};
  return {
    recording_session_id: sessionId || newRecordingSessionId(),
    source,
    device_label: track?.label || selectedMicrophoneLabel() || 'System default microphone',
    device_id: String(settings.deviceId || state.audioDeviceId || ''),
    device_settings: settings,
  };
}

function appendAudioSessionMetadata(data, metadata) {
  if (!metadata) return;
  data.append('recording_session_id', String(metadata.recording_session_id || ''));
  data.append('device_label', String(metadata.device_label || ''));
  data.append('device_id', String(metadata.device_id || ''));
  data.append('device_settings', JSON.stringify(metadata.device_settings || {}));
  data.append('device_metadata', JSON.stringify({
    label: metadata.device_label || '',
    device_id: metadata.device_id || '',
    source: metadata.source || '',
    settings: metadata.device_settings || {},
  }));
  data.append('session_metadata', JSON.stringify(metadata));
}

async function postAudioWavFiles(classId, wavFiles, metadata = null) {
  const data = new FormData();
  wavFiles.forEach(file => data.append('files', file, file.name));
  data.append('overlap', '0');
  appendAudioSessionMetadata(data, metadata);
  const result = await api(`/api/projects/${state.project.id}/classes/${classId}/audio`, { method: 'POST', body: data });
  state.project = result.project;
  return result;
}

async function ensureCamera() {
  if (state.cameraStream?.active) return state.cameraStream;
  if (!navigator.mediaDevices?.getUserMedia) throw new Error(t('workspace.error.noWebcam'));
  state.cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 640 }, facingMode: 'user' },
    audio: false,
  });
  $('#stopMediaButton').classList.remove('hidden');
  return state.cameraStream;
}

async function setupImageCapturePanel() {
  const video = $('#classCaptureVideo');
  if (!video) return;
  try {
    video.srcObject = await ensureCamera();
    await video.play().catch(() => {});
  } catch (error) {
    toast(t('workspace.toast.cameraStartFailed', { message: error.message }), 'error');
    return;
  }
  const button = $('#holdCaptureButton');
  if (!button) return;
  const start = event => {
    event.preventDefault();
    if (state.imageCaptureTimer) return;
    state.imageCaptureBlobs = [];
    state.imageCaptureClassId = state.activePanel?.classId || null;
    state.imageCaptureStartedAt = performance.now();
    button.textContent = 'Recording… release to stop';
    button.classList.add('recording');
    captureImageFrame(video);
    state.imageCaptureTimer = setInterval(() => captureImageFrame(video), 260);
  };
  const stop = event => {
    if (event) event.preventDefault();
    stopImageCapture(button);
  };
  button.addEventListener('pointerdown', start);
  button.addEventListener('pointerup', stop);
  button.addEventListener('pointercancel', stop);
  button.addEventListener('pointerleave', event => { if (event.buttons === 1) stop(event); });
  window.addEventListener('pointerup', stop, { once: true });
}

function captureImageFrame(video) {
  if (!video.videoWidth || !video.videoHeight) return;
  const canvas = document.createElement('canvas');
  const side = Math.min(video.videoWidth, video.videoHeight);
  canvas.width = 320;
  canvas.height = 320;
  const x = (video.videoWidth - side) / 2;
  const y = (video.videoHeight - side) / 2;
  canvas.getContext('2d').drawImage(video, x, y, side, side, 0, 0, 320, 320);
  canvas.toBlob(blob => {
    if (blob) state.imageCaptureBlobs.push(blob);
    const text = $('#captureResultText');
    const bar = $('#captureProgressBar');
    if (text) text.textContent = `${state.imageCaptureBlobs.length} images captured…`;
    if (bar) bar.style.width = `${Math.min(100, state.imageCaptureBlobs.length * 4)}%`;
  }, 'image/jpeg', 0.90);
}

async function stopImageCapture(button) {
  if (!state.imageCaptureTimer) return;
  clearInterval(state.imageCaptureTimer);
  state.imageCaptureTimer = null;
  button?.classList.remove('recording');
  if (button) button.textContent = 'Hold to Record';
  const blobs = state.imageCaptureBlobs.splice(0);
  const classId = state.imageCaptureClassId;
  state.imageCaptureClassId = null;
  if (!blobs.length || !classId) return;
  const files = blobs.map((blob, index) => new File([blob], `webcam_${Date.now()}_${index}.jpg`, { type: 'image/jpeg' }));
  await uploadImageFiles(classId, files);
}

function microphoneLooksLikeGestureAI(device) {
  return /gesture\s*ai|dmic|0416/i.test(String(device?.label || ''));
}

function selectedMicrophoneLabel() {
  const option = elements.microphoneDeviceSelect?.selectedOptions?.[0];
  return option?.dataset?.deviceLabel || option?.textContent || '';
}

function formatMicrophoneSettings() {
  const track = state.audioStream?.getAudioTracks?.()[0];
  const settings = track?.getSettings?.() || state.audioTrackSettings || {};
  if (!track && !Object.keys(settings).length) {
    const selected = selectedMicrophoneLabel();
    return selected ? t('workspace.microphone.selectedNotStarted', { selected }) : t('workspace.microphone.notStartedDetail');
  }
  const parts = [track?.label || selectedMicrophoneLabel() || 'Microphone'];
  if (settings.sampleRate) parts.push(`${settings.sampleRate} Hz`);
  if (settings.channelCount) parts.push(`${settings.channelCount} ch`);
  for (const [key, label] of [
    ['echoCancellation', 'EC'],
    ['noiseSuppression', 'NS'],
    ['autoGainControl', 'AGC'],
  ]) {
    if (typeof settings[key] === 'boolean') parts.push(`${label} ${settings[key] ? 'on' : 'off'}`);
  }
  return parts.join(' · ');
}

function updateMicrophoneSettingsDisplay() {
  const text = formatMicrophoneSettings();
  if (elements.microphoneTrackSettings) {
    elements.microphoneTrackSettings.textContent = text;
    elements.microphoneTrackSettings.title = text;
  }
  const captureSettings = $('#classMicrophoneSettings');
  if (captureSettings) {
    captureSettings.textContent = text;
    captureSettings.title = text;
  }
}

async function refreshMicrophoneDevices({ silent = false } = {}) {
  if (state.microphoneRefreshBusy || !navigator.mediaDevices?.enumerateDevices) return;
  state.microphoneRefreshBusy = true;
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === 'audioinput');
    state.audioDevices = devices;
    const saved = readSavedMicrophoneId();
    const hasPreference = hasSavedMicrophonePreference();
    const current = state.audioDeviceId || saved;
    const currentExists = devices.some(device => device.deviceId === current);
    const gestureAI = devices.find(microphoneLooksLikeGestureAI);
    const nextId = hasPreference
      ? (current && currentExists ? current : '')
      : (currentExists ? current : (gestureAI?.deviceId || ''));
    state.audioDeviceId = nextId;
    const options = [`<option value="">${t('workspace.microphone.systemDefault')}</option>`];
    devices.forEach((device, index) => {
      const label = device.label || t('workspace.microphone.unnamedDevice', { index: index + 1 });
      const gestureLabel = microphoneLooksLikeGestureAI(device) ? ' · GestureAI candidate' : '';
      options.push(`<option value="${escapeHtml(device.deviceId)}" data-device-label="${escapeHtml(label)}">${escapeHtml(label + gestureLabel)}</option>`);
    });
    elements.microphoneDeviceSelect.innerHTML = options.join('');
    elements.microphoneDeviceSelect.value = nextId;
    if (nextId) saveMicrophoneId(nextId);
    updateMicrophoneSettingsDisplay();
  } catch (error) {
    if (!silent) toast(t('workspace.toast.listMicrophonesFailed', { message: error.message }), 'error');
  } finally {
    state.microphoneRefreshBusy = false;
  }
}

async function openSelectedAudioStream() {
  const requestedId = state.audioDeviceId || elements.microphoneDeviceSelect?.value || readSavedMicrophoneId();
  const audio = {
    channelCount: 1,
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  };
  if (requestedId) audio.deviceId = { exact: requestedId };
  try {
    return await navigator.mediaDevices.getUserMedia({ audio, video: false });
  } catch (error) {
    if (!requestedId) throw error;
    state.audioDeviceId = '';
    saveMicrophoneId('');
    if (elements.microphoneDeviceSelect) elements.microphoneDeviceSelect.value = '';
    toast(t('workspace.toast.microphoneFallback'), 'error');
    const fallback = { ...audio };
    delete fallback.deviceId;
    return navigator.mediaDevices.getUserMedia({ audio: fallback, video: false });
  }
}

async function ensureAudio() {
  if (state.audioContext && state.audioContext.state !== 'closed') {
    if (state.audioContext.state === 'suspended') await state.audioContext.resume();
    return state.audioContext;
  }
  if (!navigator.mediaDevices?.getUserMedia) throw new Error(t('workspace.error.noMicrophone'));
  await refreshMicrophoneDevices({ silent: true });
  state.audioStream = await openSelectedAudioStream();

  // Device labels are often hidden until the first permission grant. Refresh once permission
  // exists, then automatically prefer GestureAI/DMIC/0416 only when no device was selected.
  const originallyRequested = hasSavedMicrophonePreference() || Boolean(state.audioDeviceId || readSavedMicrophoneId());
  const activeSettings = state.audioStream.getAudioTracks()[0]?.getSettings?.() || {};
  if (!originallyRequested) {
    await refreshMicrophoneDevices({ silent: true });
    if (state.audioDeviceId && activeSettings.deviceId && state.audioDeviceId !== activeSettings.deviceId) {
      state.audioStream.getTracks().forEach(track => track.stop());
      state.audioStream = await openSelectedAudioStream();
    }
  }
  const activeTrack = state.audioStream.getAudioTracks()[0];
  state.audioTrackSettings = activeTrack?.getSettings?.() || {};
  const Context = window.AudioContext || window.webkitAudioContext;
  state.audioContext = new Context();
  state.audioSource = state.audioContext.createMediaStreamSource(state.audioStream);
  state.audioAnalyser = state.audioContext.createAnalyser();
  state.audioAnalyser.fftSize = 512;
  state.audioAnalyser.smoothingTimeConstant = 0.55;
  state.audioProcessor = state.audioContext.createScriptProcessor(4096, 1, 1);
  state.audioSilentGain = state.audioContext.createGain();
  state.audioSilentGain.gain.value = 0;
  state.audioSource.connect(state.audioAnalyser);
  state.audioSource.connect(state.audioProcessor);
  state.audioProcessor.connect(state.audioSilentGain);
  state.audioSilentGain.connect(state.audioContext.destination);
  state.audioProcessor.onaudioprocess = event => {
    const data = new Float32Array(event.inputBuffer.getChannelData(0));
    appendRollingAudio(data);
    if (state.audioRecording) state.audioRecordedChunks.push(data);
  };
  $('#stopMediaButton').classList.remove('hidden');
  await refreshMicrophoneDevices({ silent: true });
  updateMicrophoneSettingsDisplay();
  startAudioVisualization();
  return state.audioContext;
}

async function releaseAudioInput() {
  if (state.audioProcessor) state.audioProcessor.onaudioprocess = null;
  state.audioStream?.getTracks().forEach(track => track.stop());
  try { await state.audioContext?.close(); } catch (_) {}
  if (state.audioAnimation) cancelAnimationFrame(state.audioAnimation);
  state.audioStream = null;
  state.audioContext = null;
  state.audioSource = null;
  state.audioProcessor = null;
  state.audioAnalyser = null;
  state.audioSilentGain = null;
  state.audioTrackSettings = null;
  state.audioRollingChunks = [];
  state.audioRollingSamples = 0;
  state.audioAnimation = null;
  updateMicrophoneSettingsDisplay();
}

async function selectMicrophoneDevice() {
  const nextId = elements.microphoneDeviceSelect.value;
  const hadActiveAudio = Boolean(state.audioStream || state.audioContext);
  if (state.audioRecording) {
    toast(t('workspace.toast.stopBeforeSwitchMic'), 'error');
    elements.microphoneDeviceSelect.value = state.audioDeviceId || '';
    return;
  }
  state.audioDeviceId = nextId;
  saveMicrophoneId(nextId);
  if (!hadActiveAudio) {
    updateMicrophoneSettingsDisplay();
    return;
  }
  setBusy(true, 'workspace.busy.switchingMicrophone');
  try {
    await releaseAudioInput();
    await ensureAudio();
    toast(t('workspace.toast.microphoneSwitched', { label: selectedMicrophoneLabel() || t('workspace.microphone.systemDefault') }), 'success');
  } catch (error) {
    toast(t('workspace.toast.switchMicrophoneFailed', { message: error.message }), 'error');
  } finally {
    setBusy(false);
  }
}

function appendRollingAudio(chunk) {
  state.audioRollingChunks.push(chunk);
  state.audioRollingSamples += chunk.length;
  const maxSamples = Math.ceil((state.audioContext?.sampleRate || 48000) * 4);
  while (state.audioRollingSamples > maxSamples && state.audioRollingChunks.length > 1) {
    const removed = state.audioRollingChunks.shift();
    state.audioRollingSamples -= removed.length;
  }
}

function recentAudio(seconds = 1) {
  const sampleRate = state.audioContext?.sampleRate || 48000;
  const needed = Math.round(sampleRate * seconds);
  const output = new Float32Array(needed);
  let write = needed;
  for (let index = state.audioRollingChunks.length - 1; index >= 0 && write > 0; index -= 1) {
    const chunk = state.audioRollingChunks[index];
    const count = Math.min(write, chunk.length);
    write -= count;
    output.set(chunk.subarray(chunk.length - count), write);
  }
  return output;
}

async function setupAudioCapturePanel() {
  try {
    await ensureAudio();
  } catch (error) {
    toast(t('workspace.toast.microphoneStartFailed', { message: error.message }), 'error');
    return;
  }
  const button = $('#audioRecordButton');
  if (!button) return;
  button.addEventListener('click', () => {
    if (state.audioRecording) stopAudioRecording();
    else startAudioRecording();
  });
}

function startAudioRecording() {
  state.audioRecordedChunks = [];
  state.audioRecording = true;
  state.audioRecordingClassId = state.activePanel?.classId || null;
  state.audioRecordingSessionId = newRecordingSessionId();
  state.audioRecordStartedAt = performance.now();
  const button = $('#audioRecordButton');
  button?.classList.add('recording');
  if (button) button.textContent = 'Stop Recording';
  const text = $('#captureResultText');
  const bar = $('#captureProgressBar');
  state.audioRecordTimer = setInterval(() => {
    const seconds = (performance.now() - state.audioRecordStartedAt) / 1000;
    if (text) text.textContent = `Recording ${seconds.toFixed(1)} / 20.0 seconds…`;
    if (bar) bar.style.width = `${Math.min(100, seconds / 20 * 100)}%`;
    if (seconds >= 20) stopAudioRecording();
  }, 100);
}

async function stopAudioRecording() {
  if (!state.audioRecording) return;
  state.audioRecording = false;
  if (state.audioRecordTimer) clearInterval(state.audioRecordTimer);
  state.audioRecordTimer = null;
  const button = $('#audioRecordButton');
  button?.classList.remove('recording');
  if (button) button.textContent = 'Record 20 Seconds';
  const chunks = state.audioRecordedChunks.splice(0);
  const classId = state.audioRecordingClassId;
  const sessionId = state.audioRecordingSessionId;
  state.audioRecordingClassId = null;
  state.audioRecordingSessionId = null;
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  if (!length || !classId) return;
  const signal = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { signal.set(chunk, offset); offset += chunk.length; }
  const blob = encodeWavBlob(signal, state.audioContext.sampleRate);
  const file = new File([blob], `microphone_${Date.now()}.wav`, { type: 'audio/wav' });
  await uploadAudioWavFiles(classId, [file], currentAudioSessionMetadata(sessionId));
}

async function uploadAudioWavFiles(classId, wavFiles, metadata = null) {
  setBusy(true, 'workspace.busy.splittingAudio');
  try {
    const result = await postAudioWavFiles(classId, wavFiles, metadata);
    renderProject();
    if (result.errors?.length) toast(result.errors.join('\n'), 'error');
    else toast(t('workspace.toast.audioSamplesAdded', { count: result.added }), 'success');
    await runAudioSanityCheck(wavFiles?.[0]);
  } catch (error) {
    toast(t('workspace.toast.recordingAddFailed', { message: error.message }), 'error');
  } finally {
    setBusy(false);
  }
}

// Ask the official YAMNet tagger what it hears in the clip that was just added. This is
// purely informational -- it never touches training, thresholds or stored metadata -- but
// it is the only thing in the product that can tell a user "you recorded speech, not a
// gunshot" BEFORE they spend an afternoon collecting the wrong audio.
async function runAudioSanityCheck(wavFile) {
  if (!wavFile || !state.project || !isAudioLikeProject()) return;
  try {
    const data = new FormData();
    data.append('file', wavFile, wavFile.name || 'check.wav');
    const result = await api(`/api/projects/${state.project.id}/audio-sanity`, { method: 'POST', body: data });
    if (!result?.available || !result.top?.length) return;
    const summary = result.top
      .map(item => `${item.name} ${formatPercent(Number(item.score) || 0)}`)
      .join(' · ');
    toast(t('workspace.toast.audioSanity', { summary }), 'info');
  } catch (error) {
    // A failed sanity check must never block or undo a successful recording.
    console.warn('audio sanity check failed', error);
  }
}

function startAudioVisualization() {
  if (state.audioAnimation) cancelAnimationFrame(state.audioAnimation);
  const draw = () => {
    drawWaveform($('#classAudioMeter'));
    drawLiveSpectrogram();
    state.audioAnimation = requestAnimationFrame(draw);
  };
  draw();
}

function drawWaveform(canvas) {
  if (!canvas || !state.audioAnalyser) return;
  const context = canvas.getContext('2d');
  const data = new Uint8Array(state.audioAnalyser.fftSize);
  state.audioAnalyser.getByteTimeDomainData(data);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#d7e8ff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = '#1967d2';
  context.lineWidth = 3;
  context.beginPath();
  for (let index = 0; index < data.length; index += 1) {
    const x = index / (data.length - 1) * canvas.width;
    const y = data[index] / 255 * canvas.height;
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  }
  context.stroke();
}

function drawLiveSpectrogram() {
  const canvas = elements.liveSpectrogramCanvas;
  if (!canvas || canvas.closest('.hidden') || !state.audioAnalyser) return;
  const context = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const previous = context.getImageData(1, 0, width - 1, height);
  context.putImageData(previous, 0, 0);
  const bins = new Uint8Array(state.audioAnalyser.frequencyBinCount);
  state.audioAnalyser.getByteFrequencyData(bins);
  for (let y = 0; y < height; y += 1) {
    const index = Math.floor((1 - y / height) * (bins.length - 1));
    const value = bins[index] / 255;
    const [r, g, b] = spectrogramColor(value);
    context.fillStyle = `rgb(${r},${g},${b})`;
    context.fillRect(width - 1, y, 1, 1);
  }
}

function spectrogramColor(value) {
  const stops = [
    [4, 16, 38], [10, 46, 91], [52, 45, 134], [117, 64, 145], [224, 97, 70], [255, 190, 80],
  ];
  const scaled = Math.max(0, Math.min(1, value)) * (stops.length - 1);
  const low = Math.floor(scaled);
  const high = Math.min(stops.length - 1, low + 1);
  const fraction = scaled - low;
  return stops[low].map((entry, index) => Math.round(entry * (1 - fraction) + stops[high][index] * fraction));
}

function mixAudioBufferToMono(buffer) {
  const output = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const input = buffer.getChannelData(channel);
    for (let index = 0; index < input.length; index += 1) output[index] += input[index] / buffer.numberOfChannels;
  }
  return output;
}

function encodeWavBlob(signal, sampleRate) {
  const buffer = new ArrayBuffer(44 + signal.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, value) => { for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i)); };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + signal.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, signal.length * 2, true);
  for (let index = 0; index < signal.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, signal[index]));
    view.setInt16(44 + index * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

// --------------------------------------------------------------------------------------
// Advanced training panel. Each control id maps to exactly one project settings key and is
// read straight back by readTrainingOptions(), so what the student sees is what Train
// persists. Keep the two functions in step: a control with no reader silently does nothing.
// --------------------------------------------------------------------------------------

function optionMarkup(pairs, current) {
  return pairs.map(([value, label]) => (
    `<option value="${escapeHtml(value)}"${String(current) === String(value) ? ' selected' : ''}>${escapeHtml(label)}</option>`
  )).join('');
}

function hintMarkup(text) {
  return `<small>${escapeHtml(text)}</small>`;
}

// 只留下 'pc' 與支援這個 kind 的板子。沒有相機的板子不該出現在 image 專案的下拉裡，但這只是
// 面板不去「提供」一個伺服器會拒絕的值，不是第三個把關點——伺服器仍會在存設定與部署時各自
// 再檢查一次（見 config.validate_*_settings()、tm_local/mcu/contract.py）。
function deploymentTargetOptionsForKind(kind) {
  return DEPLOYMENT_TARGET_OPTIONS.filter(([name]) => name === 'pc' || (MCU_BOARD_KINDS[name] || []).includes(kind));
}

function deploymentTargetValue(settings, kind) {
  const value = String(settings?.deployment_target || 'pc');
  return deploymentTargetOptionsForKind(kind).some(([name]) => name === value) ? value : 'pc';
}

function deploymentTargetSelect(current, kind) {
  return `<label>${t('training.deploymentTarget.label')}<select id="optDeploymentTarget">${optionMarkup(localizeOptions(deploymentTargetOptionsForKind(kind)), current)}</select>
      ${hintMarkup(t('training.deploymentTarget.hint'))}</label>`;
}

// `fallback` 同時是「沒有存過值時要選哪一個」和「哪一個要標（預設）」——兩者永遠是同一個值，
// 所以只能有一個參數。不要把預設值寫死在函式裡：波形增強的預設是 off，資料增強是 medium。
function augmentationSelect(id, current, label, hint, fallback) {
  const pairs = AUGMENTATION_LEVEL_OPTIONS.map(([value, key]) => {
    const text = t(key);
    return [value, value === fallback ? t('common.withDefaultSuffix', { label: text }) : text];
  });
  return `<label>${escapeHtml(label)}<select id="${id}">${optionMarkup(pairs, current || fallback)}</select>
      ${hintMarkup(hint)}</label>`;
}

// "" = 自動偵測；其餘是目前專案的 class id。class id 會在匯入時重新配發，所以每次都從
// state.project.classes 重新產生，不要記住上一個專案的 id。
function backgroundClassSelect(current) {
  const pairs = [['', t('training.backgroundClass.auto')], ...(state.project?.classes || []).map(item => [item.id, item.name])];
  return `<label>${t('training.backgroundClass.label')}<select id="optBackgroundClass">${optionMarkup(pairs, current ?? '')}</select>
      ${hintMarkup(t('training.backgroundClass.hint'))}</label>`;
}

function numberOption(id, label, value, min, max, step, hint) {
  return `<label>${escapeHtml(label)}<input id="${id}" type="number" min="${min}" max="${max}" step="${step}" value="${value}">
      ${hintMarkup(hint)}</label>`;
}

// 板子的 known_sound 深度上限：/api/mcu/status 回報的值優先，還沒載入就用前端備援常數。
function knownSoundDepthCap(target) {
  const live = Number(mcuBoardInfo(target)?.known_sound_max_depth);
  if (Number.isFinite(live) && live > 0) return live;
  const fallback = Number(KNOWN_SOUND_BOARD_MAX_DEPTH[target]);
  return Number.isFinite(fallback) && fallback > 0 ? fallback : null;
}

// 學生填在 Advanced 面板、還沒按 Train 的值只存在於 DOM 裡。renderProject() 在收樣本、刪類別、
// 改名等等之後都會重畫整個面板，所以重畫前要先把面板上的值讀回來當 override，否則「調好參數再去
// 補幾張照片」就會把參數默默打回伺服器上的舊值——而且那份舊值正是下一次 Train 真的會用的值。
// 回傳 null 代表「沒有可用的暫存值，照伺服器的 settings 畫」：
//   * 面板還沒畫過（state.trainingOptionsProjectId 是 null）——readTrainingOptions() 會去 $() 抓
//     控制項，面板還沒畫就會拿到 null 而丟例外。
//   * 面板畫的是別的專案——換專案時把上一個專案的 epochs 帶過來是最糟的結果。
function trainingOptionsDraft() {
  const project = state.project;
  if (!project || state.trainingOptionsProjectId !== project.id) return null;
  return { ...project.settings, ...readTrainingOptions() };
}

// 換專案、重新載入、或 Train 已經把設定 PATCH 進伺服器之後呼叫：下一次 render 一律以伺服器的
// settings 為準，不要再把過期的面板值撈回來。
function discardTrainingOptionsDraft() {
  state.trainingOptionsProjectId = null;
}

// `override` 是「還沒存到伺服器」的暫存設定：改目標裝置時要用學生當下填的值重畫面板，但那份
// 值只存在於這次 render，不可以寫回 state.project.settings（那是伺服器回來的副本）。
function renderTrainingOptions(override = null) {
  const settings = override || state.project.settings || {};
  const target = deploymentTargetValue(settings, state.project.kind);
  const common = `
    <label>Epochs<input id="optEpochs" type="number" min="1" max="300" value="${Number(settings.epochs || 30)}"></label>
    <label>Batch Size<input id="optBatchSize" type="number" min="1" max="128" value="${Number(settings.batch_size || 16)}"></label>
    <label>Learning Rate<input id="optLearningRate" type="number" min="0.000001" max="0.1" step="0.0001" value="${Number(settings.learning_rate || 0.001)}"></label>
    <label>Validation Split<input id="optValidationSplit" type="number" min="0" max="0.45" step="0.05" value="${Number(settings.validation_split || 0.2)}"></label>
    <label class="inline-check"><input id="optEarlyStopping" type="checkbox" ${settings.early_stopping !== false ? 'checked' : ''}> Early stopping</label>`;
  if (state.project.kind === 'image') {
    const backbone = settings.backbone === 'small_cnn' ? 'small_cnn' : 'mobilenet_v2';
    const sizes = target === 'pc' ? PC_IMAGE_SIZES : MCU_IMAGE_SIZES;
    const storedSize = Number(settings.image_size ?? 224);
    const imageSize = sizes.includes(storedSize) ? storedSize : (sizes.includes(224) ? 224 : sizes[0]);
    const sizeNotice = imageSize === storedSize
      ? ''
      : hintMarkup(t('training.image.sizeNotice', {
          sizes: listJoin([...MCU_IMAGE_SIZES].reverse()), stored: storedSize, size: imageSize,
        }));
    // 256／288／320 伺服器收（validate_image_settings 允許 96–320、32 的倍數），但 Keras 只發佈
    // 到 224 的 ImageNet 權重，更大的輸入會沿用 224 那一份。這件事對學生原本是完全無聲的，
    // 所以寫在選單下面而不是把選項拿掉。
    const sizeHint = hintMarkup(target === 'pc'
      ? t('training.image.sizeHint.pc')
      : t('training.image.sizeHint.board'));
    // fine_tune_blocks 只對 MobileNetV2 有意義：Small CNN 是從零訓練的，沒有預訓練權重
    // 可以解凍。image_pipeline 會照實記錄「跳過微調」，但先在這裡講清楚比較好教。
    const fineTuneLocked = backbone === 'small_cnn';
    // 鎖住時連顯示的值也一起歸零：畫面上留著「3 個 blocks」但存進去的是 0（見
    // readTrainingOptions()）只會讓學生更困惑，而且改回 MobileNetV2 時那個 3 又會冒出來。
    const fineTuneBlocks = fineTuneLocked ? 0 : Number(settings.fine_tune_blocks ?? 0);
    const fineTuneChoices = [
      [0, t('training.fineTune.choice0')], [1, t('training.fineTune.choice1')],
      [2, t('training.fineTune.choice2')], [3, t('training.fineTune.choice3')], [4, t('training.fineTune.choice4')],
    ];
    elements.trainingOptions.innerHTML = `
      ${deploymentTargetSelect(target, state.project.kind)}
      <label>Model<select id="optBackbone"><option value="mobilenet_v2" ${backbone !== 'small_cnn' ? 'selected' : ''}>MobileNetV2 transfer learning</option><option value="small_cnn" ${backbone === 'small_cnn' ? 'selected' : ''}>Small CNN (offline)</option></select></label>
      <label>MobileNet Size<select id="optMobileNetAlpha"><option value="0.35" ${Number(settings.mobilenet_alpha || 0.35) === 0.35 ? 'selected' : ''}>${t('training.mobilenetAlpha.default')}</option><option value="0.5" ${Number(settings.mobilenet_alpha) === 0.5 ? 'selected' : ''}>0.50</option><option value="0.75" ${Number(settings.mobilenet_alpha) === 0.75 ? 'selected' : ''}>${t('training.mobilenetAlpha.sramWarning', { value: '0.75' })}</option><option value="1" ${Number(settings.mobilenet_alpha) === 1 ? 'selected' : ''}>${t('training.mobilenetAlpha.sramWarning', { value: '1.00' })}</option></select></label>
      <label>Image Size<select id="optImageSize">${optionMarkup(sizes.map(value => [value, `${value} × ${value}`]), imageSize)}</select>${sizeNotice}${sizeHint}</label>
      ${augmentationSelect(
        'optAugmentationLevel', settings.augmentation_level || 'medium', t('training.augmentation.label'),
        t('training.augmentation.hint.image'),
        'medium',
      )}
      ${numberOption(
        'optDropout', 'Dropout', Number(settings.dropout ?? 0.2), 0, 0.6, 0.05,
        t('training.dropout.hint.image'),
      )}
      <label>${t('training.fineTune.label')}<select id="optFineTuneBlocks"${fineTuneLocked ? ' disabled' : ''}>${optionMarkup(fineTuneChoices, fineTuneBlocks)}</select>
        ${hintMarkup(fineTuneLocked
          ? t('training.fineTune.hint.locked')
          : t('training.fineTune.hint.unlocked'))}</label>${common}`;
  } else if (state.project.kind === 'audio') {
    const boardLocked = target !== 'pc';
    const storedRate = Number(settings.sample_rate || MCU_SAMPLE_RATE);
    const sampleRate = boardLocked ? MCU_SAMPLE_RATE : storedRate;
    const sampleRatePairs = boardLocked
      ? [[MCU_SAMPLE_RATE, '16 kHz']]
      : [[16000, '16 kHz'], [44100, '44.1 kHz']];
    // 和 image 的 sizeNotice、known_sound 的 depthNotice 一樣要講「原本的值被改掉了」。這一項
    // 尤其要講：sample_rate 一變，readTrainingOptions() 會連 fft_size 與 fmax 一起改掉，等於換了
    // 一套頻譜規格，已經訓練好的模型會被作廢（存設定就會清掉 models/）。
    const rateNotice = boardLocked && storedRate !== MCU_SAMPLE_RATE
      ? hintMarkup(t('training.audio.rateNotice', { rate: (storedRate / 1000).toFixed(1) }))
      : '';
    const melBins = MEL_BIN_CHOICES.includes(Number(settings.mel_bins)) ? Number(settings.mel_bins) : 40;
    // 韌體除了比門檻，還要求贏家至少比「基準線」高 trigger_margin（KWS_RUNTIME_DEFAULTS，0.20，
    // 沒有 UI 可調）。兩個類別時分數是同一組 softmax，加起來必定是 1，於是 best >= (1-best)+0.2
    // 就等於 best >= 0.6——門檻設到 0.3 在板子上也還是 0.6。三類以上就不會被這條卡死，所以只有
    // 兩類的專案要講。
    const audioThresholdFloorHint = (state.project.classes || []).length === 2
      ? t('training.audio.thresholdFloorHint')
      : '';
    elements.trainingOptions.innerHTML = `
      ${deploymentTargetSelect(target, state.project.kind)}
      <label>Sample Rate<select id="optSampleRate"${boardLocked ? ' disabled' : ''}>${optionMarkup(sampleRatePairs, sampleRate)}</select>${rateNotice}
        ${boardLocked ? hintMarkup(t('training.sampleRate.hint.locked')) : hintMarkup(t('training.sampleRate.hint.free'))}</label>
      <label>Mel Bins<select id="optMelBins">${optionMarkup(MEL_BIN_CHOICES.map(value => [value, String(value)]), melBins)}</select>
        ${hintMarkup(t('training.melBins.hint'))}</label>
      ${augmentationSelect(
        'optAugmentationLevel', settings.augmentation_level || 'medium', t('training.augmentation.label'),
        t('training.augmentation.hint.audio'),
        'medium',
      )}
      <label class="inline-check"><input id="optSpecAugment" type="checkbox" ${settings.spec_augment ? 'checked' : ''}> ${t('training.specAugment.label')}</label>
      ${hintMarkup(t('training.specAugment.hint'))}
      <label class="inline-check"><input id="optSessionDisjoint" type="checkbox" ${settings.session_disjoint_validation !== false ? 'checked' : ''}> ${t('training.sessionDisjoint.label')}</label>
      ${hintMarkup(t('training.sessionDisjoint.hint'))}
      ${numberOption(
        'optBackgroundWeight', t('training.backgroundWeight.label'), Number(settings.background_weight ?? 1), 0.25, 4, 0.25,
        t('training.backgroundWeight.hint.keyword'),
      )}
      ${backgroundClassSelect(settings.background_class_id ?? '')}
      ${numberOption(
        'optDropout', 'Dropout', Number(settings.dropout ?? 0.2), 0, 0.6, 0.05,
        t('training.dropout.hint.audio'),
      )}
      ${numberOption(
        'optDetectionThreshold', t('training.detectionThreshold.label'), Number(settings.detection_threshold ?? 0.5), 0.05, 0.99, 0.05,
        t('training.detectionThreshold.hint.audio', { floorHint: audioThresholdFloorHint }),
      )}${common}`;
  } else if (state.project.kind === 'known_sound') {
    const threshold = Number(settings.detection_threshold ?? 0.5);
    const storedMixup = Number(settings.mixup_ratio ?? 0.5);
    // 清單上沒有的值（API 或匯入的專案可能存 0.75）以前會被瀏覽器默默選成第一個「關閉」，下一次
    // Train 就把混音增強關掉了。改成和 image_size／encoder_depth 一樣：退回預設值並說一聲。
    const mixup = MIXUP_RATIO_CHOICES.some(([value]) => value === storedMixup) ? storedMixup : 0.5;
    const mixupNotice = mixup === storedMixup
      ? ''
      : hintMarkup(t('training.mixup.notice', {
          choices: listJoin(MIXUP_RATIO_CHOICES.map(([value]) => value)), stored: storedMixup, mixup,
        }));
    const storedDepth = Number(settings.encoder_depth ?? 14);
    const cap = knownSoundDepthCap(target);
    const allowed = KNOWN_SOUND_DEPTH_CHOICES.filter(([value]) => cap === null || value <= cap);
    // 板子回報的上限低到連最小的選項都不剩時，寧可留最小的那一個，也不要畫出空的 select。
    const depthChoices = allowed.length ? allowed : [KNOWN_SOUND_DEPTH_CHOICES[KNOWN_SOUND_DEPTH_CHOICES.length - 1]];
    const depth = depthChoices.some(([value]) => value === storedDepth) ? storedDepth : depthChoices[0][0];
    const depthNotice = cap !== null && storedDepth > cap
      ? hintMarkup(t('training.knownSound.depthNotice', { target, cap, stored: storedDepth, depth }))
      : '';
    const storedThresholds = settings.class_thresholds && typeof settings.class_thresholds === 'object'
      ? settings.class_thresholds
      : {};
    // data-touched 是「這一類的門檻是學生自己設定的」旗標，readTrainingOptions() 只送標記過的
    // 類別。沒有它的話，畫出來的值一律等於當下的預設門檻，學生一改上面的預設值，每一類就會以
    // 「和新預設值不同」的身分被當成覆寫送出去，把舊預設值釘死在每一類身上（改了等於沒改）。
    // 已經存過覆寫的類別一開始就算「設定過」，重新訓練才不會把它悄悄清掉。
    const thresholdRows = (state.project.classes || []).map(item => {
      const stored = usableThreshold(storedThresholds[item.id], null);
      const value = stored === null ? threshold : stored;
      return `<label>${t('training.knownSound.classThresholdLabel', { name: escapeHtml(item.name) })}<input class="opt-class-threshold" data-class-id="${escapeHtml(item.id)}" data-touched="${stored === null ? '0' : '1'}" type="number" min="0.05" max="0.95" step="0.05" value="${value}"></label>`;
    }).join('');
    elements.trainingOptions.innerHTML = `
      <div class="yamnet-training-note">
        <strong>${t('training.knownSound.noteTitle')}</strong>
        <small>${t('training.knownSound.noteBody')}</small>
      </div>
      <label>Model<input type="text" value="YAMNet embedding (frozen) + sigmoid head" readonly></label>
      <label>Frontend<input type="text" value="16 kHz · 64 mel · 0.96 s patch" readonly></label>
      ${deploymentTargetSelect(target, state.project.kind)}
      <label>${t('training.knownSound.encoderDepth.label')}<select id="optEncoderDepth">${optionMarkup(localizeOptions(depthChoices), depth)}</select>${depthNotice}
        ${hintMarkup(t('training.knownSound.encoderDepth.hint'))}</label>
      ${numberOption(
        'optDetectionThreshold', t('training.knownSound.detectionThreshold.label'), threshold, 0.05, 0.99, 0.05,
        t('training.knownSound.detectionThreshold.hint'),
      )}
      ${thresholdRows}
      ${hintMarkup(t('training.knownSound.perClassThresholdHint'))}
      ${augmentationSelect(
        'optWaveformAugment', settings.waveform_augment_level || 'off', t('training.waveformAugment.label'),
        t('training.waveformAugment.hint'),
        'off',
      )}
      ${numberOption(
        'optHeadDropout', t('training.headDropout.label'), Number(settings.head_dropout ?? 0), 0, 0.5, 0.05,
        t('training.headDropout.hint'),
      )}
      ${numberOption(
        'optBackgroundWeight', t('training.backgroundWeight.label'), Number(settings.background_weight ?? 1), 0.25, 4, 0.25,
        t('training.backgroundWeight.hint.knownSound'),
      )}
      ${backgroundClassSelect(settings.background_class_id ?? '')}
      <label>${t('training.mixup.label')}<select id="optMixupRatio">${optionMarkup(localizeOptions(MIXUP_RATIO_CHOICES), mixup)}</select>${mixupNotice}
        ${hintMarkup(t('training.mixup.hint'))}</label>
      <label>Epochs<input id="optEpochs" type="number" min="1" max="500" value="${Number(settings.epochs || 120)}"></label>
      <label>Batch Size<input id="optBatchSize" type="number" min="1" max="128" value="${Number(settings.batch_size || 32)}"></label>
      <label>Learning Rate<input id="optLearningRate" type="number" min="0.000001" max="0.1" step="0.0001" value="${Number(settings.learning_rate || 0.001)}"></label>
      <label class="inline-check"><input id="optEarlyStopping" type="checkbox" ${settings.early_stopping !== false ? 'checked' : ''}> Early stopping</label>`;
  } else {
    const sensitivity = String(settings.sensitivity || 'balanced');
    elements.trainingOptions.innerHTML = `
      <div class="yamnet-training-note">
        <strong>YAMNet · Frozen embedding</strong>
        <small>16 kHz mono → frozen 1024-D embedding → Normal-only scorer. Encoder weights are not fine-tuned, and anomaly evaluation groups never enter training or threshold calibration.</small>
      </div>
      <label>Model<input type="text" value="YAMNet embedding (frozen)" readonly></label>
      <label>Frontend<input type="text" value="16 kHz · 64 mel · 0.96 s patch" readonly></label>
      <label>Sensitivity<select id="optSensitivity">
        <option value="sensitive" ${sensitivity === 'sensitive' ? 'selected' : ''}>Sensitive · more detections</option>
        <option value="balanced" ${sensitivity === 'balanced' ? 'selected' : ''}>Balanced</option>
        <option value="low_false_alarm" ${sensitivity === 'low_false_alarm' ? 'selected' : ''}>Low false alarm</option>
      </select></label>`;
  }
  // 面板現在確實畫著這個專案的控制項，接下來的 render 才可以放心把上面填的值讀回來。
  state.trainingOptionsProjectId = state.project?.id || null;
}

// 讀回目前面板上的每一個控制項。回傳的物件會直接 PATCH 到 project settings，所以布林值一定
// 要送真正的 JSON boolean（checkbox.checked），不要送 "true"／"false" 字串。
function readTrainingOptions() {
  if (isAbnormalSoundProject()) {
    return {
      detector_backend: 'yamnet_embedding',
      sample_rate: 16000,
      sensitivity: $('#optSensitivity').value,
    };
  }
  if (isKnownSoundProject()) {
    const threshold = Number($('#optDetectionThreshold').value);
    // 只送學生自己設定過的類別（data-touched='1'，含開啟面板前就已經存過的覆寫）。沒動過的
    // 類別完全不送，才能跟著上面的預設門檻一起變——以前是比「值和預設門檻不同」，於是學生一把
    // 預設值從 0.5 調到 0.7，三個還停在 0.5 的欄位就全部變成覆寫，等於把舊值釘死。順帶也不再
    // 需要拿兩個浮點數做 === 比較。
    const classThresholds = {};
    for (const input of $$('.opt-class-threshold')) {
      const classId = input.dataset.classId;
      const value = Number(input.value);
      if (input.dataset.touched !== '1' || !classId || !Number.isFinite(value)) continue;
      classThresholds[classId] = value;
    }
    // No validation_split: the split is by recording session, not by a clip fraction.
    return {
      encoder_backend: 'yamnet_embedding',
      sample_rate: 16000,
      epochs: Number($('#optEpochs').value),
      batch_size: Number($('#optBatchSize').value),
      learning_rate: Number($('#optLearningRate').value),
      early_stopping: $('#optEarlyStopping').checked,
      detection_threshold: threshold,
      class_thresholds: classThresholds,
      mixup_ratio: Number($('#optMixupRatio').value),
      encoder_depth: Number($('#optEncoderDepth').value),
      head_dropout: Number($('#optHeadDropout').value),
      waveform_augment_level: $('#optWaveformAugment').value,
      background_weight: Number($('#optBackgroundWeight').value),
      background_class_id: $('#optBackgroundClass').value,
      deployment_target: $('#optDeploymentTarget').value,
    };
  }
  const options = {
    epochs: Number($('#optEpochs').value),
    batch_size: Number($('#optBatchSize').value),
    learning_rate: Number($('#optLearningRate').value),
    validation_split: Number($('#optValidationSplit').value),
    early_stopping: $('#optEarlyStopping').checked,
    augmentation_level: $('#optAugmentationLevel').value,
    dropout: Number($('#optDropout').value),
    deployment_target: $('#optDeploymentTarget').value,
  };
  if (state.project.kind === 'image') {
    options.backbone = $('#optBackbone').value;
    options.mobilenet_alpha = Number($('#optMobileNetAlpha').value);
    options.image_size = Number($('#optImageSize').value);
    // `disabled` 只擋表單送出，不擋 .value。Small CNN 是從零訓練的，沒有預訓練權重可以解凍，
    // 照抄面板上的數字會存成 {backbone:'small_cnn', fine_tune_blocks:3} 這種沒有意義的組合，
    // 而且之後改回 MobileNetV2 時那個 3 又會無聲無息地復活。
    const fineTuneBlocks = $('#optFineTuneBlocks');
    options.fine_tune_blocks = fineTuneBlocks.disabled ? 0 : Number(fineTuneBlocks.value);
  } else {
    options.sample_rate = Number($('#optSampleRate').value);
    options.mel_bins = Number($('#optMelBins').value);
    options.fft_size = options.sample_rate >= 32000 ? 2048 : 512;
    options.fmax = options.sample_rate / 2;
    options.spec_augment = $('#optSpecAugment').checked;
    options.session_disjoint_validation = $('#optSessionDisjoint').checked;
    options.background_weight = Number($('#optBackgroundWeight').value);
    options.background_class_id = $('#optBackgroundClass').value;
    options.detection_threshold = Number($('#optDetectionThreshold').value);
  }
  return options;
}

function renderTrainingPanel() {
  const status = trainingState();
  const isTraining = status === 'training' || Boolean(state.activeJobId);
  elements.trainingIdle.classList.toggle('hidden', status === 'trained' || isTraining);
  elements.trainingProgress.classList.toggle('hidden', !isTraining);
  elements.trainingDone.classList.toggle('hidden', status !== 'trained' || isTraining);
  elements.trainButton.disabled = !canTrain();
  const minimum = classMinimum();
  if (isAbnormalSoundProject()) {
    const readiness = abnormalTrainingReadiness();
    if (readiness.ready) {
      elements.trainingHint.textContent = readiness.calibrationCandidate
        ? t('training.hint.abnormal.calibrationCandidate', { seconds: readiness.seconds.toFixed(0), sessions: readiness.sessions })
        : t('training.hint.abnormal.lowConfidence', { seconds: readiness.calibrationSeconds.toFixed(0), sessions: readiness.calibrationSessions });
    } else if (!readiness.clips) {
      elements.trainingHint.textContent = readiness.note || t('training.hint.abnormal.collectNormal');
    } else {
      const measured = t('training.hint.abnormal.measured', {
        seconds: readiness.seconds.toFixed(0), minSeconds: readiness.minimumSeconds.toFixed(0),
        sessions: readiness.sessions, minSessions: readiness.minimumSessions,
      });
      elements.trainingHint.textContent = `${readiness.note ? `${readiness.note} ` : ''}${measured} ${t('training.hint.abnormal.recordAcrossSessions')}`;
    }
  } else {
    elements.trainingHint.textContent = canTrain()
      ? t('training.hint.ready')
      : t('training.hint.needMoreSamples', { minimum });
  }
  if (status === 'trained') {
    const report = state.project.training?.report || {};
    const evaluation = report.evaluation || {};
    const accuracy = evaluation.accuracy;
    const runtime = report.training_runtime || {};
    const runtimeLabel = runtime.label || 'Windows CPU';
    let accuracyText = accuracy == null ? t('workspace.training.doneNote') : `Evaluation accuracy: ${formatPercent(accuracy)}`;
    if (isAbnormalSoundProject()) {
      const confidence = report.readiness?.confidence || report.calibration?.confidence || 'unknown confidence';
      const reason = Array.isArray(report.readiness?.reasons) ? report.readiness.reasons[0] : '';
      accuracyText = `YAMNet anomaly detector ready · calibration ${confidence}${reason ? ` · ${reason}` : ''}`;
    }
    elements.trainingAccuracy.innerHTML = trainingSummaryHtml(report, `${accuracyText} · ${runtimeLabel}`);
  }
}

// 訓練完的一行摘要。除了 accuracy 之外，還要說出學生剛剛調的東西實際做了什麼：微調真的跑了
// 幾個 epoch、驗證是用幾個錄音場次切的，以及哪些類別的 accuracy 不能全信。
function trainingSummaryHtml(report, headline) {
  const parts = [headline];
  const settings = report?.settings || {};
  const epochs = Number(settings.epochs_completed);
  if (Number.isFinite(epochs) && epochs > 0) parts.push(t('training.summary.epochs', { epochs }));
  // 微調是第二段訓練，epochs_completed 不含它（image_pipeline 分開記）。
  const fineTuneEpochs = Number(settings.fine_tune_epochs_completed);
  if (Number.isFinite(fineTuneEpochs) && fineTuneEpochs > 0) parts.push(t('training.summary.fineTuneEpochs', { epochs: fineTuneEpochs }));
  if (settings.backbone_used && settings.backbone_requested && settings.backbone_used !== settings.backbone_requested) {
    parts.push(t('training.summary.backboneUsed', { backbone: settings.backbone_used }));
  }
  const dataset = report?.dataset || {};
  const trainSessions = Array.isArray(dataset.train_sessions) ? dataset.train_sessions.length : null;
  const validationSessions = Array.isArray(dataset.validation_sessions) ? dataset.validation_sessions.length : null;
  if (trainSessions !== null && validationSessions !== null && trainSessions + validationSessions > 0) {
    parts.push(t('training.summary.sessions', { train: trainSessions, validation: validationSessions }));
  }
  let html = parts.map(escapeHtml).join(' · ');
  const warning = splitWarningText(dataset);
  if (warning) html += `<br>${escapeHtml(warning)}`;
  return html;
}

// split_warning 帶的是類別「索引」（韌體與報告都用索引），對學生沒有意義，換成類別名稱。
// 這是提醒不是錯誤：訓練有跑完，只是那幾類的 accuracy 偏樂觀。
function splitWarningText(dataset) {
  if (!String(dataset?.split_warning || '').trim()) return '';
  const classes = state.project?.classes || [];
  const names = (Array.isArray(dataset.fallback_classes) ? dataset.fallback_classes : [])
    .map(index => classes[Number(index)]?.name)
    .filter(Boolean);
  if (!names.length) return String(dataset.split_warning).trim();
  return t('training.splitWarning', { names: listJoin(names) });
}

async function startTraining() {
  if (!canTrain()) {
    if (isAbnormalSoundProject()) {
      const readiness = abnormalTrainingReadiness();
      toast(t('training.toast.needNormalBaseline', { seconds: readiness.minimumSeconds.toFixed(0), sessions: readiness.minimumSessions }), 'error');
    } else {
      toast(t('training.hint.needMoreSamples', { minimum: classMinimum() }), 'error');
    }
    return;
  }
  stopPreview();
  const options = readTrainingOptions();
  try {
    state.project = await api(`/api/projects/${state.project.id}`, jsonOptions('PATCH', { settings: options }));
    // 設定已經進伺服器了，面板上那份暫存值不再是「還沒存的修改」。留著它的話，訓練完成後的
    // render 會把這一輪的值又當成草稿疊回去，蓋掉伺服器實際存下來（正規化過）的設定。
    discardTrainingOptionsDraft();
    const job = await api(`/api/projects/${state.project.id}/train`, jsonOptions('POST', { options }));
    state.activeJobId = job.id;
    setTrainingProgress(job.progress || 0, job.message || 'Queued');
    renderTrainingPanel();
    pollJob(job.id);
  } catch (error) {
    toast(t('training.toast.startFailed', { message: error.message }), 'error');
  }
}

function setTrainingProgress(progress, message) {
  const percent = Math.round((Number(progress) || 0) * 100);
  elements.trainingPercent.textContent = `${percent}%`;
  elements.trainingMessage.textContent = message || 'Training…';
  elements.trainingProgressBar.style.width = `${percent}%`;
  const ring = elements.trainingPercent.closest('.progress-ring');
  if (ring) ring.style.setProperty('--progress', `${percent}%`);
}

function pollJob(jobId) {
  if (state.jobTimer) clearInterval(state.jobTimer);
  state.activeJobId = jobId;
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      setTrainingProgress(job.progress, job.message);
      if (job.state === 'completed') {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        state.activeJobId = null;
        state.project = job.result?.project || await api(`/api/projects/${state.project.id}`);
        renderProject();
        toast(
          isAbnormalSoundProject()
            ? t('training.toast.abnormalTrained')
            : t('training.toast.trained'),
          'success',
        );
      } else if (job.state === 'failed') {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        state.activeJobId = null;
        await loadProject(state.project.id, { quiet: true });
        toast(t('training.toast.failed', { detail: job.error || job.message }), 'error');
      }
    } catch (error) {
      clearInterval(state.jobTimer);
      state.jobTimer = null;
      state.activeJobId = null;
      toast(t('training.toast.statusFailed', { message: error.message }), 'error');
    }
  };
  tick();
  state.jobTimer = setInterval(tick, 900);
}

function renderPreviewPanel() {
  const trained = trainingState() === 'trained';
  elements.openExportButton.disabled = !trained;
  elements.previewLocked.classList.toggle('hidden', trained);
  elements.previewReady.classList.toggle('hidden', !trained);
  if (!trained) return;
  const artifacts = state.project.training?.artifacts || {};
  for (const option of [...elements.previewRuntime.options]) {
    if (option.value === 'keras') option.disabled = !artifacts.keras;
    else option.disabled = !artifacts[option.value];
  }
  const selectedOption = elements.previewRuntime.selectedOptions[0];
  if (!selectedOption || selectedOption.disabled) elements.previewRuntime.value = 'keras';
  elements.imagePreviewPane.classList.toggle('hidden', state.project.kind !== 'image');
  elements.audioPreviewPane.classList.toggle('hidden', !isAudioLikeProject());
  clearPreviewOutput();
}


function renderPredictionBars(predictions, result = null) {
  if (!state.project) return;
  const safePredictions = predictions || [];
  const multiLabel = isKnownSoundProject();
  // 每一類有自己的門檻。優先用這次預測回傳的 threshold（伺服器用的就是這個值），沒有才回到
  // 訓練報告／專案設定推出來的清單。
  const thresholdByName = new Map();
  if (multiLabel) {
    const resolved = knownSoundThresholds();
    (state.project.classes || []).forEach((item, index) => thresholdByName.set(item.name, resolved[index]));
    for (const item of safePredictions) {
      const value = Number(item.threshold);
      if (Number.isFinite(value) && value > 0 && value < 1) thresholdByName.set(item.class_name, value);
    }
  }
  const byName = new Map(safePredictions.map(item => [item.class_name, item.score]));
  const topName = safePredictions.length
    ? [...safePredictions].sort((a, b) => Number(b.score || 0) - Number(a.score || 0))[0].class_name
    : null;
  elements.predictionBars.innerHTML = state.project.classes.map(classItem => {
    const score = Math.max(0, Math.min(1, Number(byName.get(classItem.name) || 0)));
    const width = (score * 100).toFixed(2);
    const threshold = multiLabel ? (thresholdByName.get(classItem.name) ?? knownSoundThreshold()) : null;
    // For a mutually exclusive softmax the winner is meaningful; for independent
    // sigmoids "above its own threshold" is the meaningful state, and several classes
    // can be in it at once.
    const highlight = multiLabel ? score >= threshold : classItem.name === topName;
    const marker = multiLabel
      ? `<i class="prediction-threshold" style="--prediction-threshold:${(threshold * 100).toFixed(2)}%" title="${escapeHtml(t('preview.classThresholdTitle', { name: classItem.name, threshold: formatPercent(threshold) }))}"></i>`
      : '';
    return `<div class="prediction-row ${highlight ? 'prediction-top' : ''}" style="--prediction-color:${classItem.color}">
      <div class="prediction-meta">
        <span class="prediction-label"><i class="prediction-dot"></i>${escapeHtml(classItem.name)}</span>
        <strong class="prediction-score">${formatPercent(score)}</strong>
      </div>
      <span class="prediction-track" role="progressbar" aria-label="${escapeHtml(classItem.name)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(score * 100)}" aria-valuetext="${formatPercent(score)}">
        <i class="prediction-fill ${score > 0 ? 'has-value' : ''}" style="--prediction-width:${width}%"></i>${marker}
      </span>
    </div>`;
  }).join('');
  if (elements.predictionNote) {
    if (multiLabel) {
      const detected = (result?.detected || []).filter(Boolean);
      const values = state.project.classes.map(item => thresholdByName.get(item.name) ?? knownSoundThreshold());
      const shared = values.length && values.every(value => value === values[0]);
      const thresholdText = shared
        ? t('preview.thresholdText.shared', { value: formatPercent(values[0]) })
        : t('preview.thresholdText.perClass');
      const backgroundIndex = knownSoundBackgroundIndex();
      const backgroundName = state.project.classes[backgroundIndex]?.name;
      elements.predictionNote.innerHTML = t('preview.knownSound.noteMain', { thresholdText: escapeHtml(thresholdText) })
        + (backgroundName ? t('preview.knownSound.noteBackground', { name: escapeHtml(backgroundName) }) : '')
        + (detected.length ? t('preview.knownSound.noteDetected', { names: detected.map(escapeHtml).join(t('common.listSeparator')) }) : '');
      elements.predictionNote.classList.remove('hidden');
    } else {
      elements.predictionNote.classList.add('hidden');
    }
  }
}

function clearPreviewOutput() {
  state.knownSoundPeaks = new Map();
  if (isAbnormalSoundProject()) renderAbnormalPrediction(null);
  else renderPredictionBars([]);
}

function finiteNumber(...values) {
  for (const value of values) {
    if (value == null || value === '') continue;
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }
  return null;
}

function componentNumber(value) {
  if (value && typeof value === 'object') {
    return finiteNumber(value.ratio, value.value, value.score, value.distance);
  }
  return finiteNumber(value);
}

function normalizedVerdict(result) {
  const raw = String(result?.verdict || result?.decision || result?.state || '').trim().toLowerCase();
  if (['abnormal', 'anomaly', 'alarm', 'detected'].includes(raw) || result?.abnormal === true) return 'abnormal';
  if (['normal', 'ok', 'healthy'].includes(raw) || result?.abnormal === false) return 'normal';
  return 'uncertain';
}

function anomalyComponentEntries(result) {
  const components = result?.components && typeof result.components === 'object' ? result.components : {};
  const preferred = [
    ['embedding', 'YAMNet embedding'],
    ['embedding_ratio', 'YAMNet embedding'],
    ['yamnet_ratio', 'YAMNet embedding'],
    ['distance_ratio', 'Embedding distance'],
    ['level', 'RMS level'],
    ['level_ratio', 'RMS level'],
    ['rms_ratio', 'RMS level'],
  ];
  const entries = [];
  const seenLabels = new Set();
  for (const [key, label] of preferred) {
    const value = componentNumber(components[key] ?? result?.[key]);
    if (value == null || seenLabels.has(label)) continue;
    entries.push([label, value]);
    seenLabels.add(label);
  }
  for (const [key, raw] of Object.entries(components)) {
    const value = componentNumber(raw);
    if (value == null || preferred.some(([preferredKey]) => preferredKey === key)) continue;
    const label = key.replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());
    entries.push([label, value]);
  }
  return entries;
}

function smoothingText(result) {
  const smoothing = result?.smoothing;
  if (!smoothing || typeof smoothing !== 'object') return '';
  if (smoothing.warming_up || smoothing.state === 'warming_up') {
    const observed = finiteNumber(smoothing.observed_windows, smoothing.windows_seen, 0);
    const windowCount = finiteNumber(smoothing.window_count, smoothing.required_windows);
    return windowCount == null ? 'Temporal vote warming up' : `Temporal vote warming up · ${observed}/${windowCount} windows`;
  }
  const votes = finiteNumber(smoothing.votes, smoothing.abnormal_votes, smoothing.exceedances);
  const required = finiteNumber(smoothing.votes_required, smoothing.required_votes);
  const windowCount = finiteNumber(smoothing.window_count, smoothing.windows);
  if (votes == null) {
    return required != null && windowCount != null ? `Temporal policy ${required}-of-${windowCount}` : '';
  }
  return `Temporal vote ${votes}${required == null ? '' : `/${required}`}${windowCount == null ? '' : ` in ${windowCount} windows`}`;
}

function resetAbnormalTemporalSmoothing() {
  state.abnormalPreviewVotes = [];
}

function temporalPolicy(result) {
  const smoothing = result?.smoothing && typeof result.smoothing === 'object' ? result.smoothing : {};
  const presetName = String(state.project?.settings?.sensitivity || 'balanced');
  const presetFallback = presetName === 'sensitive'
    ? { votesRequired: 2, windowCount: 5 }
    : { votesRequired: 3, windowCount: 5 };
  const windowCount = Math.max(1, Math.round(finiteNumber(smoothing.window_count, smoothing.windows, presetFallback.windowCount)));
  const votesRequired = Math.max(1, Math.min(
    windowCount,
    Math.round(finiteNumber(smoothing.votes_required, smoothing.required_votes, presetFallback.votesRequired)),
  ));
  return { smoothing, windowCount, votesRequired };
}

function applyAbnormalTemporalSmoothing(result) {
  const ratio = finiteNumber(result?.anomaly_ratio, result?.composite_ratio, result?.ratio, result?.score);
  const threshold = finiteNumber(result?.threshold, result?.threshold_ratio, 1) || 1;
  const { smoothing, windowCount, votesRequired } = temporalPolicy(result);
  const confidence = String(
    result?.confidence || result?.calibration_confidence || result?.readiness?.confidence || '',
  ).trim().toLowerCase();
  const calibrated = confidence === 'calibrated';
  if (ratio == null) {
    return {
      ...result,
      verdict: 'uncertain',
      smoothing: {
        ...smoothing,
        source: 'browser_sliding_vote',
        warming_up: true,
        observed_windows: state.abnormalPreviewVotes.length,
        abnormal_votes: state.abnormalPreviewVotes.filter(Boolean).length,
        votes_required: votesRequired,
        window_count: windowCount,
      },
    };
  }
  state.abnormalPreviewVotes.push(ratio > threshold);
  if (state.abnormalPreviewVotes.length > windowCount) state.abnormalPreviewVotes.shift();
  const abnormalVotes = state.abnormalPreviewVotes.filter(Boolean).length;
  const warmedUp = state.abnormalPreviewVotes.length >= windowCount;
  return {
    ...result,
    verdict: calibrated && warmedUp
      ? (abnormalVotes >= votesRequired ? 'abnormal' : 'normal')
      : 'uncertain',
    smoothing: {
      ...smoothing,
      source: 'browser_sliding_vote',
      state: warmedUp ? 'ready' : 'warming_up',
      warming_up: !warmedUp,
      observed_windows: state.abnormalPreviewVotes.length,
      abnormal_votes: abnormalVotes,
      votes: abnormalVotes,
      votes_required: votesRequired,
      window_count: windowCount,
      calibration_ready: calibrated,
    },
  };
}

function renderAbnormalPrediction(result) {
  if (!state.project) return;
  if (!result) {
    elements.predictionBars.innerHTML = `
      <div class="anomaly-result verdict-uncertain empty">
        <div class="anomaly-verdict"><i></i><span><strong>Waiting for audio</strong><small>YAMNet embedding distance and RMS level will appear here.</small></span></div>
      </div>`;
    return;
  }
  const verdict = normalizedVerdict(result);
  const verdictLabels = {
    normal: ['Normal', 'Within the collected Normal baseline'],
    abnormal: ['Abnormal', 'Outside the calibrated Normal baseline'],
    uncertain: ['Uncertain', 'Warming up or calibration evidence is insufficient'],
  };
  const verdictDetail = result.message || verdictLabels[verdict][1];
  const ratio = finiteNumber(result.anomaly_ratio, result.composite_ratio, result.ratio, result.score);
  const threshold = finiteNumber(result.threshold, result.threshold_ratio, 1) || 1;
  const ratioText = ratio == null ? '—' : `${ratio.toFixed(2)}×`;
  const gaugeWidth = ratio == null ? 0 : Math.max(0, Math.min(100, ratio / Math.max(threshold * 2, 0.000001) * 100));
  const components = anomalyComponentEntries(result);
  const componentHtml = components.length
    ? `<div class="anomaly-components">${components.map(([label, value]) => `
        <div><span>${escapeHtml(label)}</span><strong>${value.toFixed(2)}×</strong></div>`).join('')}</div>`
    : '';
  const confidence = result.confidence || result.calibration_confidence || result.readiness?.confidence;
  const smoothing = smoothingText(result);
  elements.predictionBars.innerHTML = `
    <div class="anomaly-result verdict-${verdict}">
      <div class="anomaly-verdict"><i></i><span><strong>${verdictLabels[verdict][0]}</strong><small>${escapeHtml(verdictDetail)}</small></span></div>
      <div class="anomaly-ratio-row"><span>Anomaly ratio</span><strong>${ratioText}</strong></div>
      <div class="anomaly-gauge" role="meter" aria-label="Anomaly ratio" aria-valuemin="0" aria-valuemax="${threshold * 2}" aria-valuenow="${ratio ?? 0}">
        <i style="--anomaly-width:${gaugeWidth.toFixed(2)}%"></i><b title="Threshold ${threshold.toFixed(2)}×"></b>
      </div>
      <small class="anomaly-threshold-copy">Threshold ${threshold.toFixed(2)}× · values above the marker are anomalous</small>
      ${componentHtml}
      ${(confidence || smoothing) ? `<div class="anomaly-footnote">${confidence ? `Calibration: ${escapeHtml(confidence)}` : ''}${confidence && smoothing ? ' · ' : ''}${escapeHtml(smoothing)}</div>` : ''}
    </div>`;
}

async function togglePreview() {
  if (elements.previewInputToggle.checked) await startPreview();
  else stopPreview();
}

async function startPreview() {
  if (!state.project || trainingState() !== 'trained') return;
  if (isAbnormalSoundProject()) {
    state.previewSessionId = newRecordingSessionId();
    resetAbnormalTemporalSmoothing();
  }
  elements.previewToggleText.textContent = 'ON';
  if (state.project.kind === 'image') {
    try {
      elements.previewVideo.srcObject = await ensureCamera();
      await elements.previewVideo.play().catch(() => {});
      state.previewTimer = setInterval(predictCameraFrame, 450);
      predictCameraFrame();
    } catch (error) {
      elements.previewInputToggle.checked = false;
      toast(t('preview.toast.cameraStartFailed', { message: error.message }), 'error');
    }
  } else {
    try {
      await ensureAudio();
      $('.audio-live-status').classList.add('active');
      elements.audioLiveText.textContent = 'Listening…';
      state.previewTimer = setInterval(predictRecentAudio, 500);
      setTimeout(predictRecentAudio, 800);
    } catch (error) {
      elements.previewInputToggle.checked = false;
      toast(t('preview.toast.microphoneStartFailed', { message: error.message }), 'error');
    }
  }
}

function stopPreview() {
  if (state.previewTimer) clearInterval(state.previewTimer);
  state.previewTimer = null;
  state.previewBusy = false;
  state.previewSessionId = null;
  resetAbnormalTemporalSmoothing();
  if (elements.previewInputToggle) elements.previewInputToggle.checked = false;
  if (elements.previewToggleText) elements.previewToggleText.textContent = 'OFF';
  const status = $('.audio-live-status');
  status?.classList.remove('active');
  if (elements.audioLiveText) elements.audioLiveText.textContent = 'Microphone is off';
}

async function predictCameraFrame() {
  const video = elements.previewVideo;
  if (!video.videoWidth || state.previewBusy) return;
  const canvas = elements.previewCanvas;
  const context = canvas.getContext('2d');
  const side = Math.min(video.videoWidth, video.videoHeight);
  const x = (video.videoWidth - side) / 2;
  const y = (video.videoHeight - side) / 2;
  context.drawImage(video, x, y, side, side, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.88));
  if (blob) await predictBlob(blob, 'image/jpeg', 'preview.jpg', 'image');
}

async function predictRecentAudio() {
  const minimumSeconds = isAbnormalSoundProject() ? 1.0 : 0.5;
  if (!state.audioContext || state.previewBusy || state.audioRollingSamples < state.audioContext.sampleRate * minimumSeconds) return;
  const signal = recentAudio(1.0);
  const blob = encodeWavBlob(signal, state.audioContext.sampleRate);
  await predictBlob(blob, 'audio/wav', 'preview.wav', 'audio');
}

async function predictBlob(blob, type, filename, kind) {
  state.previewBusy = true;
  try {
    const data = new FormData();
    data.append('file', new File([blob], filename, { type }), filename);
    data.append('runtime_name', elements.previewRuntime.value);
    if (isAbnormalSoundProject()) {
      state.previewSessionId ||= newRecordingSessionId();
      data.append('preview_session_id', state.previewSessionId);
    }
    let endpointKind = kind;
    if (kind === 'audio' && isAbnormalSoundProject()) endpointKind = 'abnormal-sound';
    else if (kind === 'audio' && isKnownSoundProject()) endpointKind = 'known-sound';
    const result = await api(`/api/projects/${state.project.id}/predict/${endpointKind}`, { method: 'POST', body: data });
    if (isAbnormalSoundProject()) renderAbnormalPrediction(applyAbnormalTemporalSmoothing(result));
    else if (isKnownSoundProject()) renderPredictionBars(applyKnownSoundPeakHold(result.predictions), result);
    else renderPredictionBars(result.predictions);
  } catch (error) {
    stopPreview();
    toast(t('preview.toast.predictFailed', { message: error.message }), 'error');
  } finally {
    state.previewBusy = false;
  }
}

async function previewImageFile(file) {
  if (!file) return;
  const image = new Image();
  image.onload = () => {
    const canvas = elements.previewCanvas;
    const context = canvas.getContext('2d');
    const side = Math.min(image.naturalWidth, image.naturalHeight);
    context.drawImage(image, (image.naturalWidth - side) / 2, (image.naturalHeight - side) / 2, side, side, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(image.src);
  };
  image.src = URL.createObjectURL(file);
  await predictBlob(file, file.type || 'image/jpeg', file.name || 'preview.jpg', 'image');
}

async function previewAudioFile(file) {
  if (!file) return;
  setBusy(true, 'preview.busy.decodingAudio');
  try {
    const Context = window.AudioContext || window.webkitAudioContext;
    const context = new Context();
    const decoded = await context.decodeAudioData((await file.arrayBuffer()).slice(0));
    await context.close();
    const mono = mixAudioBufferToMono(decoded);
    const blob = encodeWavBlob(mono, decoded.sampleRate);
    await predictBlob(blob, 'audio/wav', 'preview.wav', 'audio');
  } catch (error) {
    toast(t('preview.toast.audioPreviewFailed', { message: error.message }), 'error');
  } finally {
    setBusy(false);
  }
}

async function stopAllMedia() {
  stopPreview();
  if (state.imageCaptureTimer) clearInterval(state.imageCaptureTimer);
  state.imageCaptureTimer = null;
  if (state.audioRecordTimer) clearInterval(state.audioRecordTimer);
  state.audioRecordTimer = null;
  state.audioRecording = false;
  state.audioRecordingClassId = null;
  state.audioRecordingSessionId = null;
  state.imageCaptureClassId = null;
  state.cameraStream?.getTracks().forEach(track => track.stop());
  state.cameraStream = null;
  await releaseAudioInput();
  $('#stopMediaButton').classList.add('hidden');
  toast(t('common.toast.mediaStopped'));
}

function openExportModal() {
  if (trainingState() !== 'trained') return;
  showExportModal();
}

// 部署工作在背景跑完之後也要能把 modal 叫回來，所以「開啟」與「必須先訓練」分成兩個函式。
function showExportModal() {
  elements.exportModal.classList.remove('hidden');
  state.mcuError = '';
  renderMcuTab();
  refreshMcuStatus().then(() => renderMcuTab());
  if (state.pendingDeployJobId && !state.deployJobTimer) {
    const jobId = state.pendingDeployJobId;
    state.pendingDeployJobId = null;
    switchExportTab('mcu');
    pollDeployJob(jobId);
  }
}

async function downloadModel() {
  const selected = $$('input[name="exportFormat"]:checked').map(input => input.value);
  if (!selected.length) {
    toast(t('export.toast.selectFormat'), 'error');
    return;
  }
  closeExportModal();
  try {
    const job = await api(
      `/api/projects/${state.project.id}/export-model`,
      jsonOptions('POST', {
        formats: selected,
        include_c_header: $('#includeCHeader').checked,
      }),
    );
    state.activeExportJobId = job.id;
    state.exportStartedAt = Date.now();
    setBusy(true, 'export.busy.starting');
    pollExportJob(job.id);
  } catch (error) {
    setBusy(false);
    toast(t('export.toast.requestFailed', { message: error.message }), 'error');
  }
}

function exportElapsedText() {
  const elapsed = Math.max(0, Math.floor((Date.now() - state.exportStartedAt) / 1000));
  const minutes = Math.floor(elapsed / 60);
  const seconds = String(elapsed % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function pollExportJob(jobId) {
  if (state.exportJobTimer) clearInterval(state.exportJobTimer);
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      const percent = Math.round((Number(job.progress) || 0) * 100);
      setBusy(true, 'export.busy.progress', { percent, message: job.message || t('common.busy'), elapsed: exportElapsedText() });
      if (job.state === 'completed') {
        clearInterval(state.exportJobTimer);
        state.exportJobTimer = null;
        state.activeExportJobId = null;
        const result = job.result || {};
        if (result.project) {
          state.project = result.project;
          renderProject();
        } else {
          await loadProject(state.project.id, { quiet: true });
        }
        for (const warning of exportAgreementWarnings(state.project)) {
          toast(warning, 'info');
        }
        const response = await api(result.download_url);
        await downloadResponse(response, result.filename || `${state.project.name}_TFLite.zip`);
        setBusy(false);
        toast(t('export.toast.completed'), 'success');
      } else if (job.state === 'failed') {
        clearInterval(state.exportJobTimer);
        state.exportJobTimer = null;
        state.activeExportJobId = null;
        setBusy(false);
        toast(t('export.toast.failed', { detail: job.error || job.message }), 'error');
      }
    } catch (error) {
      clearInterval(state.exportJobTimer);
      state.exportJobTimer = null;
      state.activeExportJobId = null;
      setBusy(false);
      toast(t('export.toast.statusFailed', { message: error.message }), 'error');
    }
  };
  tick();
  state.exportJobTimer = setInterval(tick, 900);
}


// ---------------------------------------------------------------------------
// 部署到開發板（Export modal 的第二個頁籤）
// 一次建置就是一個 job_type === 'deploy' 的 job，輪詢方式與 train／export 完全相同。
// ---------------------------------------------------------------------------

// 與 deploy_service.flash_instructions() 等價的指引。deploy_report.json 自 2.1.0 起一定
// 會帶 flash_instructions（伺服器是唯一真相），下面只是給更早版本建置的報告用的後備。
// 燒錄步驟看板子，「會看到什麼」看專案類型：只有 image 有畫面，audio／known_sound 完全沒有
// 畫面，只在 COM port 印文字（GestureAI 是 USB 虛擬 COM port，X 板是 Nu-Link 的）。
// Values are i18n *keys*, not text -- these tables are frozen at load time (before a student
// can ever toggle language), so the actual copy is only resolved at render time by the two
// functions below, the same way DEPLOYMENT_TARGET_OPTIONS etc. are resolved via t()/localizeOptions().
const MCU_FLASH_STEPS = Object.freeze({
  msc: 'mcu.flashSteps.msc',
  nulink: 'mcu.flashSteps.nulink',
});

const MCU_FLASH_RESULT = Object.freeze({
  image: Object.freeze({
    msc: 'mcu.flashResult.image.msc',
    nulink: 'mcu.flashResult.image.nulink',
  }),
  known_sound: 'mcu.flashResult.knownSound',
  audio: 'mcu.flashResult.audio',
});

function mcuFlashStepsText(method) {
  return t(MCU_FLASH_STEPS[method] || MCU_FLASH_STEPS.msc);
}

function mcuFlashResultText(kind, hasLcd) {
  const result = MCU_FLASH_RESULT[kind] || MCU_FLASH_RESULT.image;
  // image 的「會看到什麼」由板子有沒有 LCD 決定，不是由燒錄方式決定：GestureAI 選 nulink 燒錄
  // 時看到的仍然是相機視窗，不會憑空冒出一片 LCD。
  if (typeof result === 'string') return t('mcu.serialHint') + t(result);
  return t(hasLcd ? result.nulink : result.msc);
}

// 板子註冊表裡的三個燒錄方式，給學生看的名稱。板子清單只列出 boards.json 真的支援的那幾個
// （見 mcu_toolkit/boards.json 的 flash_methods），這裡只負責翻譯，不負責過濾。
function mcuFlashMethodLabel(value) {
  const key = { msc: 'mcu.flashMethodLabel.msc', nulink: 'mcu.flashMethodLabel.nulink', pyocd: 'mcu.flashMethodLabel.pyocd' }[value];
  return key ? t(key) : value;
}

function mcuBoards() {
  const boards = state.mcuStatus?.boards;
  return Array.isArray(boards) ? boards : [];
}

function mcuBoardInfo(name) {
  return mcuBoards().find(board => board.name === name) || null;
}

// 一塊板子真正支援的燒錄方式：優先看 /api/mcu/status 回報的 flash_methods 陣列（新欄位），
// 沒有的話（很舊的伺服器）退回單一 flash_method，最後才是寫死的 'msc'。
function mcuFlashMethodsFor(boardName) {
  const info = mcuBoardInfo(boardName);
  if (Array.isArray(info?.flash_methods) && info.flash_methods.length) return info.flash_methods;
  return [info?.flash_method || 'msc'];
}

function selectedMcuBoard() {
  return elements.mcuBoardSelect?.value || '';
}

// 上一次建置結果只對「當初建它的那個專案」有效；換專案之後同名的板子不算數。
function activeMcuResult() {
  const result = state.mcuResult;
  return result && result.projectId === (state.project?.id || null) ? result : null;
}

// 換專案（或回首頁）時把部署頁籤的狀態全部丟掉：輪詢、log、下載 token 都不能跨專案沿用。
function resetMcuStateForProject(projectId) {
  if (state.mcuProjectId === projectId) return;
  state.mcuProjectId = projectId;
  stopDeployPolling();
  state.pendingDeployJobId = null;
  state.mcuError = '';
  state.mcuResult = null;
  state.mcuDownloadUrl = '';
  state.mcuDownloadName = '';
  elements.deployLog.textContent = '';
  // 清空選項並回到預設板子，A 專案選的板子才不會跟著跑到 B 專案。
  elements.mcuBoardSelect.innerHTML = '';
  elements.mcuBoardSelect.value = preferredMcuBoard();
  elements.mcuFlashMethodSelect.innerHTML = '';
  elements.mcuFlashMethodSelect.dataset.board = '';
  elements.mcuFlashMethodRow.classList.add('hidden');
  elements.mcuProgress.classList.add('hidden');
  elements.mcuResult.classList.add('hidden');
  elements.mcuFlashStatus.textContent = '';
  elements.mcuErrorNotice.textContent = '';
  elements.mcuErrorNotice.classList.add('hidden');
}

function preferredMcuBoard() {
  const target = String(state.project?.settings?.deployment_target || '').trim();
  return target && target !== 'pc' ? target : '';
}

function formatKb(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return '—';
  return value < 1048576 ? `${(value / 1024).toFixed(1)} KB` : `${(value / 1048576).toFixed(2)} MB`;
}

function formatMcuUsage(used, limit) {
  const usedText = formatKb(used);
  const limitValue = Number(limit);
  if (!Number.isFinite(limitValue) || limitValue <= 0) return usedText;
  const percent = Math.round(((Number(used) || 0) / limitValue) * 100);
  return `${usedText} / ${formatKb(limitValue)}（${percent}%）`;
}

// /api/mcu/status 每次都會真的跑一次 arm-none-eabi-gcc --version，所以只在開啟 Export modal
// 或切到這個頁籤時查詢，絕對不要放進 setInterval。同時只允許一個查詢在路上。
function refreshMcuStatus() {
  if (state.mcuStatusPending) return state.mcuStatusPending;
  state.mcuStatusPending = (async () => {
    try {
      state.mcuStatus = await api('/api/mcu/status');
    } catch (_) {
      state.mcuStatus = {
        available: false, boards: [], deployable_kinds: [],
        toolchain: {}, toolkit: {}, vela: {}, nulink: {},
      };
    } finally {
      state.mcuStatusPending = null;
    }
    return state.mcuStatus;
  })();
  return state.mcuStatusPending;
}

function switchExportTab(name) {
  $$('.export-tabs button').forEach(button => button.classList.toggle('active', button.dataset.exportTab === name));
  $('#exportTabTflite').classList.toggle('hidden', name !== 'tflite');
  $('#exportTabMcu').classList.toggle('hidden', name !== 'mcu');
  if (name !== 'mcu') return;
  renderMcuTab();
  refreshMcuStatus().then(() => renderMcuTab());
}

function mcuMissingParts(status) {
  const missing = [];
  if (!status?.toolkit?.available) missing.push(t('mcu.missing.toolkit'));
  if (!status?.vela?.available) missing.push(t('mcu.missing.vela'));
  if (!status?.toolchain?.available) missing.push(t('mcu.missing.toolchain'));
  if (!missing.length && !mcuBoards().length) missing.push(t('mcu.missing.boardList'));
  return missing.length ? missing : [t('mcu.missing.generic')];
}

function renderMcuTab() {
  const select = elements.mcuBoardSelect;
  if (!select) return;
  const status = state.mcuStatus;
  // 只列出支援這個 kind 的板子（同 deploymentTargetOptionsForKind()，共用 MCU_BOARD_KINDS）：
  // 這只是面板不去「提供」一個伺服器會拒絕的值，不是第二個把關點——contract.validate() 在
  // 部署時仍會再檢查一次，不因為這裡加了過濾就變得多餘或該被搬到別處。
  const kind = state.project?.kind || '';
  const boards = mcuBoards().filter(board => (MCU_BOARD_KINDS[board.name] || []).includes(kind));
  const wanted = select.value || preferredMcuBoard();
  select.innerHTML = '';
  for (const board of boards) {
    const option = document.createElement('option');
    option.value = board.name;
    option.textContent = board.label || board.name;
    select.appendChild(option);
  }
  if (boards.some(board => board.name === wanted)) select.value = wanted;
  else if (boards.length) select.value = boards[0].name;

  const busy = Boolean(state.deployJobTimer);
  select.disabled = busy || !boards.length;

  // Kind gating：只相信伺服器回報的 deployable_kinds，不在前端寫死清單。
  const kinds = Array.isArray(status?.deployable_kinds) ? status.deployable_kinds : [];
  const kindBlocked = Boolean(status) && kinds.length > 0 && !kinds.includes(state.project?.kind || '');
  const kindText = kindBlocked ? t('mcu.kindBlocked') : '';
  elements.mcuKindNotice.textContent = kindText;
  elements.mcuKindNotice.classList.toggle('hidden', !kindText);
  // 伺服器回的 400／409 是暫時性的，放自己的格子，不要蓋掉 kind gating 的說明。
  elements.mcuErrorNotice.textContent = state.mcuError;
  elements.mcuErrorNotice.classList.toggle('hidden', !state.mcuError);

  const probing = !status;
  const ready = Boolean(status?.available) && boards.length > 0;
  let toolchainText = '';
  if (probing) toolchainText = t('mcu.toolchain.probing');
  else if (!ready) {
    toolchainText = t('mcu.toolchain.notReady', { missing: listJoin(mcuMissingParts(status)) });
  }
  elements.mcuToolchainNotice.textContent = toolchainText;
  elements.mcuToolchainNotice.classList.toggle('hidden', !toolchainText);

  elements.deployMcuButton.disabled = busy || probing || !ready || kindBlocked || !select.value;
  elements.deployMcuButton.textContent = busy ? t('mcu.building') : t('mcu.deployButton');

  renderMcuBuildPanels(select.value);
}

// 只有真的支援兩種以上燒錄方式的板子才畫這個下拉（目前只有 NuGestureAI-M55M1：msc + nulink）；
// 只有一種就不畫，跟原本「沒有選擇」的畫面一樣。換板子時以該板的第一個（預設）方式重新起始，
// 同一塊板重畫（例如伺服器狀態刷新）則保留學生已經選的值。
function renderMcuFlashMethodSelect(boardName) {
  const row = elements.mcuFlashMethodRow;
  const select = elements.mcuFlashMethodSelect;
  if (!row || !select) return;
  const methods = mcuFlashMethodsFor(boardName);
  if (methods.length <= 1) {
    row.classList.add('hidden');
    select.innerHTML = '';
    select.dataset.board = '';
    return;
  }
  const keepPrevious = select.dataset.board === boardName && methods.includes(select.value);
  const wanted = keepPrevious ? select.value : methods[0];
  select.innerHTML = optionMarkup(
    methods.map(value => [value, mcuFlashMethodLabel(value)]),
    wanted,
  );
  select.value = wanted;
  select.dataset.board = boardName;
  row.classList.remove('hidden');
}

// 目前選到的燒錄方式：板子只有一種就直接回那一種（下拉根本沒畫出來，不看它的值）。
function selectedFlashMethod(boardName) {
  const methods = mcuFlashMethodsFor(boardName);
  if (methods.length <= 1) return methods[0];
  const select = elements.mcuFlashMethodSelect;
  const value = select && select.dataset.board === boardName ? select.value : '';
  return methods.includes(value) ? value : methods[0];
}

function mcuResultSummary(boardName, report) {
  const parts = [mcuBoardInfo(boardName)?.label || boardName];
  if (report.built_at) parts.push(formatDate(report.built_at));
  parts.push(`firmware.bin ${formatKb(report.bin?.bytes)}`);
  parts.push(`Flash ${formatMcuUsage(report.image?.flash_used, report.image?.flash_limit)}`);
  parts.push(`SRAM01 ${formatMcuUsage(report.image?.sram01_used, report.image?.sram01_limit)}`);
  if (report.vela?.arena_bytes) parts.push(`Vela arena ${formatKb(report.vela.arena_bytes)}`);
  if (report.vela?.optimise) parts.push(`Vela optimise=${report.vela.optimise}`);
  return parts.join(' · ');
}

// 結果區塊只認「這塊板子現在磁碟上有什麼韌體」：這次 page session 剛建好的優先（它帶著
// 一次性的下載 token），否則就是 project.deploy 裡上次成功建置留下來的 deploy_report。
// 重新整理頁面之後也因此仍然可以下載與燒錄上一次的建置，不必重建。
function renderMcuBuildPanels(boardName) {
  const stale = elements.mcuLastBuild;
  const session = activeMcuResult();
  const stored = state.project?.deploy?.[boardName] || null;
  // 任何樣本／設定變動都會刪掉整個 models/，deploy 區塊跟著清空。這時要講清楚是模型變了，
  // 而不是讓上一次的建置結果無聲消失、或讓學生下載到對不上模型的韌體。
  if (!stored && session?.board === boardName) {
    state.mcuResult = null;
    state.mcuDownloadUrl = '';
    state.mcuDownloadName = '';
    stale.textContent = t('mcu.staleBuild');
    stale.classList.remove('hidden');
    elements.mcuResult.classList.add('hidden');
    elements.mcuFlashMethodRow.classList.add('hidden');
    elements.mcuFlashStatus.textContent = '';
    return;
  }
  stale.textContent = '';
  stale.classList.add('hidden');
  const report = (session?.board === boardName ? session.report : null) || stored;
  if (!report) {
    elements.mcuResult.classList.add('hidden');
    elements.mcuFlashMethodRow.classList.add('hidden');
    elements.mcuFlashStatus.textContent = '';
    return;
  }
  renderMcuFlashMethodSelect(boardName);
  elements.mcuResultSummary.textContent = mcuResultSummary(boardName, report);
  elements.mcuFlashInstructions.textContent =
    mcuFlashText(report, boardName, selectedFlashMethod(boardName));
  elements.mcuResult.classList.remove('hidden');
}

function setMcuProgress(progress, message) {
  const percent = Math.round(Math.max(0, Math.min(1, Number(progress) || 0)) * 100);
  elements.mcuProgressBar.style.width = `${percent}%`;
  elements.mcuProgressMessage.textContent = `${percent}% · ${message || t('common.busy')}`;
}

// 只有原本就停在最底下才跟著捲動：學生往上捲去看某一行編譯訊息時不要把他拉回去。
function writeDeployLog(text) {
  const node = elements.deployLog;
  if (node.textContent === text) return;
  const atBottom = node.scrollHeight - node.scrollTop - node.clientHeight <= 24;
  node.textContent = text;
  if (atBottom) node.scrollTop = node.scrollHeight;
}

function appendDeployText(text) {
  writeDeployLog(elements.deployLog.textContent ? `${elements.deployLog.textContent}\n${text}` : text);
}

// 每次輪詢都把伺服器目前的視窗整段畫上去，不要自己記游標：jobs.py 只回最後 100 則，
// 而 GCC 階段每個 .c 檔就是一則訊息（幾百則），視窗一定會滑動，游標會卡在 100 而讓
// log 從編譯階段開始整個空白。伺服器已經濾掉連續重複的訊息，直接 join 不會有重複。
function renderDeployLog(lines) {
  writeDeployLog((Array.isArray(lines) ? lines : []).map(item => String(item)).join('\n'));
}

function mcuFlashText(report, boardName, selectedMethod) {
  const buildMethod = report?.flash_method || mcuBoardInfo(boardName)?.flash_method || 'msc';
  const method = selectedMethod || buildMethod;
  const provided = String(report?.flash_instructions || report?.instructions || '').trim();
  // 伺服器帶的文字是照建置當下那個（預設）方式寫的；學生若在下拉改選別的方式，這段文字就不再
  // 對得上，退回下面自己的對照表，照「選到的」方式給步驟。
  if (provided && method === buildMethod) return provided;
  const steps = mcuFlashStepsText(method);
  // 報告帶的 kind 才是這份韌體的類型；報告沒有 kind（很舊的版本）時退回目前專案的類型。
  const kind = report?.kind || state.project?.kind || 'image';
  // image 的「會看到什麼」由板子有沒有 LCD 決定，不是由燒錄方式決定：GestureAI 選 nulink 燒錄
  // 時看到的仍然是相機視窗，不會憑空冒出一片 LCD。
  const hasLcd = Boolean(mcuBoardInfo(boardName)?.has_lcd);
  return `${steps}\n${mcuFlashResultText(kind, hasLcd)}`;
}

function showMcuResult(result) {
  const report = result?.report || {};
  const boardName = result?.board || selectedMcuBoard();
  state.mcuResult = {
    projectId: state.project?.id || null,
    board: boardName,
    filename: result?.filename || '',
    report,
  };
  state.mcuDownloadUrl = result?.download_url || '';
  state.mcuDownloadName = result?.filename || '';
  elements.mcuFlashStatus.textContent = '';
  elements.mcuFlashStatus.className = 'mcu-flash-status';
  renderMcuBuildPanels(boardName);
}

async function startMcuDeploy() {
  if (!state.project) return;
  const board = selectedMcuBoard();
  if (!board) {
    toast(t('mcu.toast.noBoard'), 'error');
    return;
  }
  state.mcuError = '';
  state.mcuResult = null;
  state.mcuDownloadUrl = '';
  state.mcuDownloadName = '';
  elements.deployLog.textContent = '';
  elements.mcuErrorNotice.classList.add('hidden');
  elements.mcuResult.classList.add('hidden');
  elements.mcuFlashStatus.textContent = '';
  elements.mcuProgress.classList.remove('hidden');
  setMcuProgress(0, t('mcu.busy.creatingJob'));
  elements.deployMcuButton.disabled = true;
  elements.deployMcuButton.textContent = t('mcu.building');
  try {
    const job = await api(
      `/api/projects/${state.project.id}/deploy-mcu`,
      jsonOptions('POST', { board }),
    );
    pollDeployJob(job.id);
  } catch (error) {
    elements.mcuProgress.classList.add('hidden');
    // 伺服器的 detail 已經是給學生看的中文（尚未支援的 kind、工具鏈缺件、專案忙碌），
    // 原樣顯示，不要在前端重寫或猜測。
    state.mcuError = error.message;
    renderMcuTab();
    toast(t('mcu.toast.startFailed', { message: error.message }), 'error');
  }
}

function stopDeployPolling() {
  if (state.deployJobTimer) clearInterval(state.deployJobTimer);
  state.deployJobTimer = null;
  state.activeDeployJobId = null;
}

// jobs.py 存的 job.error 是 `類別名稱: 訊息`（例如 `DeployError: 韌體連結失敗…`）。
// 伺服器保留原字串，但學生只需要訊息本身，看到 Python 類別名稱只會更困惑。
const ERROR_CLASS_PREFIX = /^[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception):\s*/;

function studentErrorText(raw) {
  return String(raw || '').replace(ERROR_CLASS_PREFIX, '').trim();
}

function pollDeployJob(jobId) {
  stopDeployPolling();
  state.activeDeployJobId = jobId;
  state.pendingDeployJobId = null;
  elements.mcuProgress.classList.remove('hidden');
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      setMcuProgress(job.progress, job.message);
      renderDeployLog(job.log);
      if (job.state === 'completed') {
        stopDeployPolling();
        const result = job.result || {};
        if (result.project) {
          state.project = result.project;
          renderProject();
        } else if (state.project) {
          await loadProject(state.project.id, { quiet: true });
        }
        // showMcuResult() 要先跑：renderMcuTab() 會依「這塊板子有沒有結果」決定
        // 要不要同時顯示「上次建置」，先後顛倒就會兩塊同時出現。
        showMcuResult(result);
        renderMcuTab();
        toast(t('mcu.toast.buildComplete'), 'success');
      } else if (job.state === 'failed') {
        stopDeployPolling();
        renderMcuTab();
        const detail = studentErrorText(job.error) || job.message || t('common.unknownError');
        if (!elements.deployLog.textContent.includes(detail)) appendDeployText(t('mcu.log.buildFailed', { detail }));
        toast(t('mcu.toast.buildFailed', { detail }), 'error');
      }
    } catch (error) {
      stopDeployPolling();
      renderMcuTab();
      appendDeployText(t('mcu.log.statusFailed', { message: error.message }));
      toast(t('mcu.toast.statusFailed', { message: error.message }), 'error');
    }
  };
  tick();
  state.deployJobTimer = setInterval(tick, 900);
}

async function downloadMcuFirmware(boardName) {
  const board = boardName || selectedMcuBoard();
  if (!state.project || !board) return;
  // 只有這次 page session 建的那塊板子才會有 token；磁碟上的舊建置一律重新跟伺服器要。
  const canReuse = Boolean(state.mcuDownloadUrl) && activeMcuResult()?.board === board;
  setBusy(true, 'mcu.busy.preparingZip');
  try {
    let url = state.mcuDownloadUrl;
    let filename = state.mcuDownloadName;
    if (!canReuse) {
      const payload = await api(
        `/api/projects/${state.project.id}/deploy-mcu/download`,
        jsonOptions('POST', { board }),
      );
      url = payload.download_url;
      filename = payload.filename;
    }
    // /api/exports/<token> 下載一次就把暫存檔刪掉，所以用過的 URL 一律作廢，
    // 下次再按就重新跟伺服器要一個新的 token。
    state.mcuDownloadUrl = '';
    const response = await api(url);
    await downloadResponse(response, filename || `${state.project.name}_${board}_firmware.zip`);
    toast(t('mcu.toast.zipDownloaded'), 'success');
  } catch (error) {
    toast(t('mcu.toast.downloadFailed', { message: error.message }), 'error');
  } finally {
    setBusy(false);
  }
}

// 燒錄會直接覆蓋板子上現有的韌體，所以先把「要燒哪個檔、多大、燒到哪一塊板」講清楚再問一次。
// 取消就什麼請求都不送。
function flashConfirmParams(board, report) {
  const label = mcuBoardInfo(board)?.label || board;
  const name = String(report?.bin?.file || 'firmware.bin');
  const bytes = Number(report?.bin?.bytes);
  const size = Number.isFinite(bytes) && bytes > 0 ? bytes.toLocaleString('en-US') : t('common.unknown');
  return { name, size, label };
}

async function flashMcu() {
  const board = selectedMcuBoard();
  const info = mcuBoardInfo(board);
  if (!state.project || !info) return;
  const report = (activeMcuResult()?.board === board ? activeMcuResult().report : null)
    || state.project?.deploy?.[board] || {};
  const accepted = await confirmDialog('mcu.confirm.flashTitle', null, 'mcu.confirm.flashMessage', flashConfirmParams(board, report));
  if (!accepted) return;
  const node = elements.mcuFlashStatus;
  node.className = 'mcu-flash-status';
  node.textContent = t('mcu.flashing');
  state.mcuError = '';
  elements.mcuErrorNotice.classList.add('hidden');
  elements.mcuFlashButton.disabled = true;
  try {
    const payload = await api(
      `/api/projects/${state.project.id}/deploy-mcu/flash`,
      jsonOptions('POST', { board, method: selectedFlashMethod(board) }),
    );
    node.textContent = payload?.message || t('mcu.flashDoneDefault');
    node.classList.add('ok');
    toast(t('mcu.toast.flashComplete'), 'success');
  } catch (error) {
    // 400 會帶 bootloader／Nu-Link 的中文指引，409 表示還有工作在跑：原樣顯示。
    node.textContent = '';
    state.mcuError = error.message;
    elements.mcuErrorNotice.textContent = error.message;
    elements.mcuErrorNotice.classList.remove('hidden');
  } finally {
    elements.mcuFlashButton.disabled = false;
  }
}

function closeExportModal() {
  elements.exportModal.classList.add('hidden');
  onExportModalClosed();
}

// 關掉 modal 時停掉輪詢（伺服器上的 job 會自己跑完），重新開啟時再接回去繼續看進度。
function onExportModalClosed() {
  if (!state.deployJobTimer) return;
  state.pendingDeployJobId = state.activeDeployJobId;
  stopDeployPolling();
  renderMcuTab();
  toast(t('mcu.toast.buildContinuesInBackground'), 'info');
}


function downloadDiagnostics() {
  const link = document.createElement('a');
  link.href = `/api/diagnostics/download?ts=${Date.now()}`;
  link.download = 'TM_Local_Studio_Diagnostic.txt';
  document.body.appendChild(link);
  link.click();
  link.remove();
  toast(t('header.toast.diagnosticsDownloaded'), 'success');
}

async function downloadProject() {
  if (!state.project) return;
  setBusy(true, 'header.busy.packagingProject');
  try {
    const response = await api(`/api/projects/${state.project.id}/export-project`);
    await downloadResponse(response, `${state.project.name}_Project.zip`);
  } catch (error) {
    toast(t('header.toast.projectDownloadFailed', { message: error.message }), 'error');
  } finally {
    setBusy(false);
  }
}

async function downloadResponse(response, fallbackName) {
  const blob = await response.blob();
  const disposition = response.headers.get('content-disposition') || '';
  const match = disposition.match(/filename\*?=(?:UTF-8''|"?)([^";]+)/i);
  const filename = match ? decodeURIComponent(match[1].replaceAll('"', '')) : fallbackName;
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function importProject(file) {
  if (!file) return;
  setBusy(true, 'header.busy.importingProject');
  try {
    const data = new FormData();
    data.append('file', file, file.name);
    const project = await api('/api/projects/import', { method: 'POST', body: data });
    toast(t('header.toast.projectImported'), 'success');
    goProject(project.id);
  } catch (error) {
    toast(t('header.toast.projectImportFailed', { message: error.message }), 'error');
  } finally {
    setBusy(false);
    $('#importProjectInput').value = '';
  }
}

// title/message are (key, params) pairs, not resolved text, so the dialog can be repainted
// in a newly selected language while it is still open -- see renderConfirmDialog() below.
function confirmDialog(titleKey, titleParams, messageKey, messageParams) {
  state.confirm = { titleKey, titleParams, messageKey, messageParams };
  renderConfirmDialog();
  $('#confirmModal').classList.remove('hidden');
  return new Promise(resolve => { state.confirmResolver = resolve; });
}

function renderConfirmDialog() {
  if (!state.confirm) return;
  $('#confirmTitle').textContent = t(state.confirm.titleKey, state.confirm.titleParams);
  $('#confirmMessage').textContent = t(state.confirm.messageKey, state.confirm.messageParams);
}

function resolveConfirm(value) {
  $('#confirmModal').classList.add('hidden');
  state.confirmResolver?.(value);
  state.confirmResolver = null;
  state.confirm = null;
}

function drawConnectors() {
  if (!state.project || elements.projectView.classList.contains('hidden')) return;
  const canvas = $('#workspaceCanvas');
  const canvasRect = canvas.getBoundingClientRect();
  const width = canvas.scrollWidth;
  const height = Math.max(canvas.scrollHeight, canvas.clientHeight);
  elements.connectorSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  elements.connectorSvg.setAttribute('width', width);
  elements.connectorSvg.setAttribute('height', height);
  elements.connectorSvg.innerHTML = '';
  const training = $('#trainingCard').getBoundingClientRect();
  const preview = $('#previewCard').getBoundingClientRect();
  const scrollX = canvas.scrollLeft;
  const scrollY = canvas.scrollTop;
  const point = rect => ({
    left: rect.left - canvasRect.left + scrollX,
    right: rect.right - canvasRect.left + scrollX,
    top: rect.top - canvasRect.top + scrollY,
    bottom: rect.bottom - canvasRect.top + scrollY,
    midY: (rect.top + rect.bottom) / 2 - canvasRect.top + scrollY,
  });
  const trainPoint = point(training);
  for (const card of $$('.class-card')) {
    const from = point(card.getBoundingClientRect());
    addConnector(from.right, from.midY, trainPoint.left, trainPoint.midY);
  }
  const previewPoint = point(preview);
  addConnector(trainPoint.right, trainPoint.midY, previewPoint.left, previewPoint.midY);
}

function addConnector(x1, y1, x2, y2) {
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  const distance = Math.max(25, (x2 - x1) * 0.48);
  path.setAttribute('d', `M ${x1} ${y1} C ${x1 + distance} ${y1}, ${x2 - distance} ${y2}, ${x2} ${y2}`);
  elements.connectorSvg.appendChild(path);
}

// Repaints #saveState from state.saveState ('saved' | 'saving' | 'failed') so a language
// toggle mid-edit shows the current save state in the new language instead of reverting to
// the static "Saved locally" placeholder that data-i18n would otherwise blindly restore.
function renderSaveState() {
  const key = state.saveState === 'saving' ? 'header.saveState.saving'
    : state.saveState === 'failed' ? 'header.saveState.failed'
    : 'header.saveState.saved';
  elements.saveState.textContent = t(key);
}

function wireGlobalEvents() {
  $$('#langToggle .lang-option').forEach(button => button.addEventListener('click', () => setLang(button.dataset.lang)));
  $$('[data-create-kind]').forEach(button => button.addEventListener('click', () => createProject(button.dataset.createKind)));
  $('#homeButton').addEventListener('click', goHome);
  $('#refreshProjectsButton').addEventListener('click', loadProjects);
  $('#reloadProjectButton').addEventListener('click', () => state.project && loadProject(state.project.id));
  $('#addClassButton').addEventListener('click', async () => {
    try {
      state.project = await api(`/api/projects/${state.project.id}/classes`, jsonOptions('POST', {}));
      renderProject();
    } catch (error) { toast(t('workspace.toast.addClassFailed', { message: error.message }), 'error'); }
  });
  elements.projectNameInput.addEventListener('change', async () => {
    if (!state.project) return;
    const name = elements.projectNameInput.value.trim() || state.project.name;
    state.saveState = 'saving';
    renderSaveState();
    try {
      state.project = await api(`/api/projects/${state.project.id}`, jsonOptions('PATCH', { name }));
      state.saveState = 'saved';
      renderSaveState();
      document.title = `${state.project.name} · Teachable Machine Local`;
    } catch (error) {
      elements.projectNameInput.value = state.project.name;
      state.saveState = 'failed';
      renderSaveState();
      toast(error.message, 'error');
    }
  });
  elements.projectNameInput.addEventListener('keydown', event => { if (event.key === 'Enter') elements.projectNameInput.blur(); });
  elements.trainButton.addEventListener('click', startTraining);
  $('#retrainButton').addEventListener('click', startTraining);
  // 每一類的門檻只有學生自己動過才算覆寫，見 readTrainingOptions() 的 class_thresholds。
  // 綁 input 而不是只綁 change：數字欄位的上下箭頭、鍵盤輸入都會發 input，change 只在離開欄位
  // 時才發，先按 Train 的話 change 根本還沒來得及觸發。
  const markClassThresholdEdited = event => {
    const input = event.target;
    if (input?.classList?.contains('opt-class-threshold')) input.dataset.touched = '1';
  };
  elements.trainingOptions.addEventListener('input', markClassThresholdEdited);
  // 目標裝置／backbone 會改變其他控制項的選項清單，所以要重畫面板。重畫前先把目前填的值讀回
  // settings 的暫存副本，否則學生已經改好的 epochs、門檻等等會被預設值蓋掉。那份副本只給這次
  // render 用，不寫回 state.project.settings（那是伺服器回來的專案，沒 PATCH 就不該被改掉）。
  // 只綁一次。
  elements.trainingOptions.addEventListener('change', event => {
    markClassThresholdEdited(event);
    const id = event.target?.id;
    if (!state.project || (id !== 'optDeploymentTarget' && id !== 'optBackbone')) return;
    const pending = { ...state.project.settings, ...readTrainingOptions() };
    renderTrainingOptions(pending);
  });
  elements.previewInputToggle.addEventListener('change', togglePreview);
  elements.previewRuntime.addEventListener('change', () => {
    clearPreviewOutput();
    if (elements.previewInputToggle.checked) {
      stopPreview();
      elements.previewInputToggle.checked = true;
      startPreview();
    }
  });
  $('#previewImageInput').addEventListener('change', event => previewImageFile(event.target.files?.[0]));
  $('#previewAudioInput').addEventListener('change', event => previewAudioFile(event.target.files?.[0]));
  elements.microphoneDeviceSelect.addEventListener('change', selectMicrophoneDevice);
  $('#refreshMicrophonesButton').addEventListener('click', () => refreshMicrophoneDevices());
  elements.openExportButton.addEventListener('click', openExportModal);
  $('#downloadModelButton').addEventListener('click', downloadModel);
  $$('[data-export-tab]').forEach(button => button.addEventListener('click', () => switchExportTab(button.dataset.exportTab)));
  elements.mcuBoardSelect.addEventListener('change', () => renderMcuTab());
  elements.mcuFlashMethodSelect.addEventListener('change', () => renderMcuBuildPanels(selectedMcuBoard()));
  elements.deployMcuButton.addEventListener('click', startMcuDeploy);
  elements.mcuDownloadButton.addEventListener('click', () => downloadMcuFirmware());
  elements.mcuFlashButton.addEventListener('click', flashMcu);
  $$('[data-close-modal]').forEach(button => button.addEventListener('click', () => {
    $(`#${button.dataset.closeModal}`).classList.add('hidden');
    if (button.dataset.closeModal === 'exportModal') onExportModalClosed();
  }));
  $('#confirmCancel').addEventListener('click', () => resolveConfirm(false));
  $('#confirmAccept').addEventListener('click', () => resolveConfirm(true));
  $('#importProjectButton').addEventListener('click', () => $('#importProjectInput').click());
  $('#importProjectInput').addEventListener('change', event => importProject(event.target.files?.[0]));
  $('#exportProjectButton').addEventListener('click', downloadProject);
  $('#downloadDiagnosticsButton').addEventListener('click', downloadDiagnostics);
  $('#deleteProjectButton').addEventListener('click', async () => {
    if (!state.project) return;
    const ok = await confirmDialog('header.deleteProject', null, 'header.confirm.deleteProjectMessage', { name: state.project.name });
    if (!ok) return;
    setBusy(true, 'header.busy.deletingProject');
    try {
      await api(`/api/projects/${state.project.id}`, { method: 'DELETE' });
      await stopAllMedia();
      goHome();
    } catch (error) { toast(error.message, 'error'); }
    finally { setBusy(false); }
  });
  $('#stopMediaButton').addEventListener('click', stopAllMedia);
  window.addEventListener('hashchange', handleRoute);
  window.addEventListener('resize', drawConnectors);
  navigator.mediaDevices?.addEventListener?.('devicechange', () => refreshMicrophoneDevices({ silent: true }));
  document.addEventListener('click', event => {
    if (!event.target.closest('.class-menu-button') && !event.target.closest('.class-action-menu')) {
      $$('.class-action-menu').forEach(menu => menu.classList.add('hidden'));
    }
  });
}

async function boot() {
  applyI18n();
  state.audioDeviceId = readSavedMicrophoneId();
  wireGlobalEvents();
  await checkHealth();
  // 工具鏈探測會實際執行 arm-none-eabi-gcc，所以只在開機與開啟 Export modal 時做一次。
  refreshMcuStatus();
  await handleRoute();
  setInterval(checkHealth, 15000);
}

boot();
