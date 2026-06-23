# RoachBlocker 訓練腳本

訓練 RoachBlocker Chrome 擴充功能所用的蟑螂偵測模型。
目前使用 **YOLO11n** 架構，並加入「負樣本」來降低誤判（false positive）。

---

## 目錄
1. [這個腳本會做什麼](#這個腳本會做什麼)
2. [關於模型架構](#關於模型架構)
3. [電腦需求](#電腦需求)
4. [安裝步驟](#安裝步驟)
5. [執行訓練](#執行訓練)
6. [替換模型到擴充功能](#替換模型到擴充功能)
7. [調整與自訂](#調整與自訂)
8. [常見問題排除](#常見問題排除)

---

## 這個腳本會做什麼

依序自動完成：

1. **下載蟑螂正樣本**：多個 Roboflow 公開資料集
2. **處理昆蟲綜合資料集**：逐張檢查標註並分流
   - 含蟑螂的圖 → 移入正樣本，只保留蟑螂標籤
   - 不含蟑螂的圖 → 當純背景負樣本
3. **下載 COCO 一般物體**（約 1200 張）→ 當背景負樣本
4. **合併、重新切分** 成 80% 訓練 / 20% 驗證
5. **訓練** YOLO11n
6. **匯出** 成 `roach.onnx`

### 為什麼要負樣本？

模型只看過「有蟑螂」的圖，沒看過「長得有點像但不是蟑螂」的東西，
遇到其他昆蟲、深色物體、雜亂背景時容易誤判。YOLO 對「沒有標註框的圖」
會當成純背景學習，所以把容易誤判的圖當背景放進訓練集，能壓低誤判率
（hard negative mining）。

### 三類負樣本（對付不同誤判來源）

腳本用三種負樣本壓低誤判，各針對不同問題：

1. **其他昆蟲**（蜘蛛/蠍子/害蟲）→ 對付「把別種蟲認成蟑螂」
2. **COCO 一般物體** → 對付「把日常雜物認成蟑螂」
3. **商標 + 複雜紋理**（新增）→ 對付「把木紋/石頭/地毯/商標認成蟑螂」

第三類由 `HARD_BG_DATASETS`（Roboflow 的商標、車標、地毯資料集）提供，整批當背景。
上限為 `MAX_NEG_HARDBG`（預設 2000）。

**最對症的做法——自備紋理資料夾**：在腳本同層建立 `my_textures/` 資料夾，
把你「實際被誤判」的那類圖（木紋、石頭、地毯、布料截圖等）放進去，
腳本會自動當背景負樣本。少量「對症」的圖，效果遠勝大量隨機背景。

### ⚠️ 關鍵：負樣本不能含有蟑螂！

這是本專案踩過的最大坑。昆蟲綜合資料集（蜘蛛/蠍子）本身**也包含蟑螂**。
如果整批當背景圖、把標註全部抹除，等於告訴模型「這隻蟑螂不是蟑螂」——
標籤自相矛盾，會同時造成**漏抓**（模型混淆而不敢報）與準確率停滯。

本腳本的解法：對昆蟲資料集**逐張檢查標註**，含蟑螂的圖救回來當正樣本
（只留蟑螂標籤），不含的才當背景。修正後 mAP50 從卡關的 0.948 提升到 0.968，
證實了標籤污染才是先前的真正瓶頸。

### 類別名稱辨識（三分類，避免污染）

不同資料集對「蟑螂」的命名不一致：cockroach、Cockroach、nymph（蟑螂若蟲）、
蟑螂、學名 Periplaneta 等。腳本把每個類別分成三種處理：

1. **明確是蟑螂**（`ROACH_KEYWORDS`）→ 該圖當正樣本
2. **模糊類別**（`AMBIGUOUS_KEYWORDS`，如 object、insect、pest、bug、unknown、
   數字編號等無法判斷是不是蟑螂的名稱）→ **含此類的圖整張丟棄**
3. **明確的其他生物**（spider、rat、caterpillar 等）→ 當背景負樣本

**為什麼模糊類別要丟棄？** 假設某資料集有個類別叫 `object`，而它標的其實是蟑螂，
若把這張圖當背景，等於告訴模型「這隻蟑螂不是蟑螂」，造成漏抓。由於無法從名稱
確定 `object` 是不是蟑螂，最安全的做法是整張丟棄——不當正樣本也不當背景。

執行時 log 會印出每個資料集的「判定為蟑螂的類別」「模糊類別」「丟棄張數」，方便檢查。
若發現某資料集的蟑螂類別沒被認出，把名稱加進 `ROACH_KEYWORDS`；
若有正常類別被誤判為模糊，調整 `AMBIGUOUS_KEYWORDS`。
---

## 關於模型架構

腳本預設使用 **YOLO11n**（nano）。可在腳本最上方的 `MODEL_ARCH` 切換：

| 架構 | 參數量 | ONNX 大小 | 準確率 | 特點 |
|------|--------|-----------|--------|------|
| `yolo11n.pt`（預設） | ~2.6M | ~10 MB | 普通 | **小、快、載入迅速，準確率優於 YOLOv8n** |
| `yolo11s.pt` | ~9.4M | ~38 MB | 較好 | 準確率較高，但檔案大、載入慢 |
| `yolov8n.pt` | ~3.0M | ~12 MB | 普通 | 舊版相容 |

> 💡 **為什麼預設用 11n？**
> 實測發現本專案的瓶頸在「訓練資料的多樣性」，而非模型容量
> （nano 與 small 實際表現差異不大）。因此選擇輕量的 11n：
> 檔案小、瀏覽器載入快、訓練快，把資源花在補充資料而非加大模型。
> YOLO11n 又比 YOLOv8n 架構更新、準確率更好，是輕量首選。

> ⚠️ 所有 YOLO 架構匯出的 ONNX 輸出格式相同（`[1, 5, 3549]`），
> 因此**換架構不需要改擴充功能的程式碼**，只要替換 `roach.onnx` 即可。

---

## 電腦需求

| 項目 | 需求 |
|------|------|
| 作業系統 | Windows 10/11、macOS、Linux 皆可 |
| Python | 3.9 ～ 3.12（建議 3.11） |
| 顯示卡 | 有 NVIDIA GPU 最佳（11n 訓練約 1~2 小時）；無 GPU 用 CPU 會慢很多 |
| 硬碟空間 | 約 7 GB（含 COCO 與其他資料集） |
| 記憶體 | 建議 8 GB 以上 |

> 💡 沒有 NVIDIA 顯卡？可以把腳本內容貼到免費的
> [Google Colab](https://colab.research.google.com)（選 T4 GPU）執行。

---

## 安裝步驟

### 1. 確認 Python 已安裝

開啟終端機（Windows 用 PowerShell；Mac/Linux 用 Terminal），輸入：

```
python --version
```

若顯示版本號（如 3.11.x）即可。沒有的話到 [python.org](https://www.python.org/downloads/) 下載安裝，
**Windows 安裝時務必勾選「Add Python to PATH」**。

### 2. 進入這個資料夾並建立虛擬環境

**Windows（PowerShell）**
```powershell
cd 這個資料夾的路徑

python -m venv venv
.\venv\Scripts\Activate.ps1
```
> 若出現「無法載入...因為執行原則」錯誤，先執行下面這行再重試啟用：
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

**macOS / Linux（Terminal）**
```bash
cd 這個資料夾的路徑

python3 -m venv venv
source venv/bin/activate
```

> ✅ 啟用成功後，命令列開頭會出現 `(venv)`。之後每次訓練都要先啟用 venv。

### 3. 安裝 PyTorch

⚠️ **這是最容易出錯的一步，請依你的情況選擇正確指令：**

**有 NVIDIA 顯卡（Windows / Linux）** — 裝 CUDA 版才能用 GPU 加速：
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
> torch 與 torchvision **務必一起安裝**，確保版本配對
> （版本不配對會導致訓練時 `torchvision::nms` 錯誤）。

**macOS，或沒有 NVIDIA 顯卡** — 裝一般版即可：
```
pip install torch torchvision
```

### 4. 安裝其餘套件

```
pip install -r requirements.txt
```

### 5. 確認安裝成功（重要）

```
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

- torch 與 torchvision 都應顯示相同的 CUDA 後綴（如都是 `+cu124`）
- 有 NVIDIA 顯卡時，`CUDA可用` 應為 `True`
- 若 `CUDA可用: False`，訓練會改用 CPU（能跑但很慢）

---

## 執行訓練

### 1. 填入你的 Roboflow API Key

1. 到 [roboflow.com](https://roboflow.com) 註冊免費帳號
2. 登入後左側 **Settings → API Keys** → 複製 **Private API Key**（不是 `rf_` 開頭的 Publishable Key）
3. 用文字編輯器打開 `train_roach.py`，把最上方這行引號內換成你的 key：

```python
ROBOFLOW_API_KEY = "在這裡貼上你的_ROBOFLOW_API_KEY"
```

> ⚠️ API Key 等同密碼，請勿把填好 key 的腳本上傳到公開的 GitHub 或分享給他人。

### 2.（建議）先用探測模式確認資料集

正式訓練前，先跑一次「探測模式」——它只下載資料集、印出每個的類別與正樣本張數，
**不訓練**，幾分鐘就跑完。這樣能先確認哪些資料集能用、含不含蟑螂，
避免花好幾小時訓練後才發現某些資料集沒貢獻或下載失敗。

```
python train_roach.py --probe      # Windows
python3 train_roach.py --probe     # macOS / Linux
```

看 log 裡每個資料集的：
- `判定為蟑螂的類別` → 確認有沒有認出蟑螂
- `此資料集 → 正樣本 X 張` → 確認有沒有貢獻正樣本

把「正樣本 0 張」或「下載失敗（跳過）」的資料集從 `MIXED_INSECT_DATASETS` 移除，
若某資料集明明有蟑螂卻沒被認出，把它的類別名關鍵字加進 `ROACH_KEYWORDS`。
調整好之後再執行下面的正式訓練。

### 3. 正式執行

確認 venv 已啟用（開頭有 `(venv)`），然後：

```
python train_roach.py      # Windows
python3 train_roach.py     # macOS / Linux
```

腳本會自動下載資料、合併、訓練、匯出。第一次會下載約 1GB 的 COCO，請耐心等候。
訓練結束後，會在這個資料夾產生 **`roach.onnx`**，並印出準確率（mAP50）。

> ✅ **驗收重點**：mAP50 數字僅供參考。真正要看的是**裝進擴充功能後，
> 實際網頁測試時漏抓與誤判有沒有改善**。

---

## 替換模型到擴充功能

⚠️ 從 Chrome 線上商店安裝的版本檔案唯讀、會被自動更新覆蓋，**無法直接替換**。
要使用自訓練模型，需改用「開發人員模式」載入擴充功能原始碼：

1. 取得擴充功能原始碼資料夾
2. 把訓練產生的 `roach.onnx` 複製到原始碼的 `assets/` 內，覆蓋原檔
3. Chrome 開 `chrome://extensions` → 右上角開啟「開發人員模式」
4. 點「載入未封裝項目」→ 選擇剛才放好模型的資料夾
5. 開 F12 → Console，看到 `✅ 本地模型載入完成` 即成功

---

## 調整與自訂

腳本最上方可調整這些參數：

| 參數 | 說明 |
|------|------|
| `MODEL_ARCH` | 模型架構（預設 `yolo11n.pt`，見上方架構表） |
| `BATCH` | 批次大小（預設 16；顯存不足時改 8 或 4） |
| `MAX_NEG_INSECTS` | 其他昆蟲負樣本上限（預設 1200） |
| `MAX_NEG_COCO` | COCO 背景負樣本上限（預設 1200） |
| `EPOCHS` | 訓練輪數（預設 100） |
| `IMG_SIZE` | 輸入尺寸（預設 416，**勿改**，除非同步改擴充功能的 `roach-inference.mjs`） |

**負樣本怎麼調：**
- 誤判仍多 → 調高 `MAX_NEG_*`（讓模型更保守）
- 開始漏抓 → 調低 `MAX_NEG_*`（負樣本太多會讓模型太保守）

**加入更多資料集：**
編輯 `POSITIVE_DATASETS`（純蟑螂）與 `MIXED_INSECT_DATASETS`（綜合害蟲，會自動分流）清單。
腳本已內建多個查證存在的害蟲資料集（IP102、pest-detection 等），會自動萃取其中的蟑螂當正樣本、其餘當背景。
到 [Roboflow Universe](https://universe.roboflow.com/search?q=class%3Acockroach) 搜尋資料集，
點進去從網址列取得 `workspace` 與 `project` 名稱，照格式新增一行。

> 💡 **提升真實場景準確率最有效的方法**：蒐集你實際會遇到、會被漏抓的蟑螂圖
> （各種角度、背景、距離），在 Roboflow 標註後加入正樣本清單。
> 這種「貼近真實使用」的資料，少量也勝過大量的標準照——這是突破準確率瓶頸的關鍵。

---

## 常見問題排除

### ❌ `ModuleNotFoundError: No module named 'torch'`
torch 沒有裝進目前的環境。確認 venv 已啟用（開頭有 `(venv)`），重新執行安裝步驟 3、4。

### ❌ `NotImplementedError: Could not run 'torchvision::nms' with arguments from the 'CUDA' backend`
torch 與 torchvision 版本不配對。修法：
```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
務必兩個一起裝。

### ❌ pip 安裝 requirements.txt 出現 `UnicodeDecodeError: 'cp950' codec...`
Windows 中文環境的編碼問題。本檔案的 requirements.txt 已改為純英文可避免。
若仍遇到，可直接手動安裝：
```
pip install "ultralytics>=8.3.0" roboflow onnx onnxruntime onnxslim pyyaml
```

### ❌ `CUDA out of memory`
顯存不足。打開 `train_roach.py`，把 `BATCH = 16` 改成 8 或 4 再重跑。

### ❌ 啟用 venv 時出現「無法載入...因為執行原則」（Windows）
PowerShell 預設禁止執行腳本。先跑：
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
再重新啟用 venv。

### ❌ 找不到 `yolo11n.pt` 或下載失敗
需要 `ultralytics >= 8.3.0` 才支援 YOLO11。先更新：
```
pip install -U ultralytics
```

### ❌ 用 `yolo predict` 測試 ONNX 時出現 onnxruntime-gpu / CUDA 錯誤
`yolo predict` 會自動裝 onnxruntime-gpu，可能與你的 CUDA 版本衝突。
測試時加上 `device=cpu` 強制用 CPU（測單張圖很快）：
```
yolo predict model=roach.onnx source="圖片.jpg" imgsz=416 device=cpu
```

### ❌ 訓練卡在下載權重或 AMP 檢查報錯
在 `train()` 的 `model.train(...)` 參數中加上 `amp=False,` 即可跳過。

### ⚠️ 某個資料集下載失敗
腳本會自動跳過下載失敗的資料集，用其他成功的繼續，不會中斷。

### ⚠️ 想重新訓練但不想重新下載資料
已下載的資料在 `datasets/`、`negatives/`、`merged_roach/` 資料夾中，腳本會沿用。
若想完全重來，刪掉這幾個資料夾再執行。

---

## 相關連結
- [Roboflow Universe（找資料集）](https://universe.roboflow.com/search?q=class%3Acockroach)
- [Ultralytics YOLO11 文件](https://docs.ultralytics.com)
- [Google Colab（線上 GPU）](https://colab.research.google.com)
- [下載 Python](https://www.python.org/downloads/)
