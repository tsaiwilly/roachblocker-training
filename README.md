# RoachBlocker 訓練腳本

訓練 RoachBlocker Chrome 擴充功能所用的蟑螂偵測模型。
目前使用 **YOLO11s** 架構，並加入「負樣本」來降低誤判（false positive）。

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
2. **下載其他昆蟲**（蜘蛛 / 蠍子等）→ 移除標註，當背景圖
3. **下載 COCO 一般物體**（約 1500 張日常物體）→ 當背景圖
4. **合併、重新切分** 成 80% 訓練 / 20% 驗證
5. **訓練** YOLO11s
6. **匯出** 成 `roach.onnx`

### 為什麼要負樣本？

原本的模型只看過「有蟑螂」的圖，沒看過「長得有點像但不是蟑螂」的東西，
所以遇到其他昆蟲、深色物體、雜亂背景時容易誤判。

YOLO 對「沒有標註框的圖」會當成純背景學習。因此把容易被誤判的圖放進訓練集
（但不標註），能有效教模型「這些不是蟑螂」，壓低誤判率。這個技巧叫
hard negative mining，比單純增加蟑螂圖有效得多。

---

## 關於模型架構

腳本預設使用 **YOLO11s**（small）。可在腳本最上方的 `MODEL_ARCH` 切換：

| 架構 | 參數量 | ONNX 大小 | 準確率 | 適用情境 |
|------|--------|-----------|--------|----------|
| `yolo11n.pt` | ~2.6M | ~10 MB | 普通 | 追求最小檔案、最快載入 |
| `yolo11s.pt`（預設） | ~9.4M | ~38 MB | 較好 | **準確率與大小的平衡** |
| `yolov8n.pt` | ~3.0M | ~12 MB | 普通 | 舊版相容 |
| `yolov8s.pt` | ~11.2M | ~43 MB | 較好 | 舊版相容 |

> 💡 **為什麼用 YOLO11s 而非 nano？**
> nano 模型容量小，面對「種類繁多」的蟑螂時容易學不全、導致漏抓。
> small 版容量足夠，準確率明顯更好。YOLO11s 又比 YOLOv8s 更準、參數更少，是目前的最佳選擇。
> 代價是 ONNX 檔較大（約 38MB），擴充功能首次載入會稍慢，但會被快取。

> ⚠️ 所有 YOLO 架構匯出的 ONNX 輸出格式相同（`[1, 5, 3549]`），
> 因此**換架構不需要改擴充功能的程式碼**，只要替換 `roach.onnx` 即可。

---

## 電腦需求

| 項目 | 需求 |
|------|------|
| 作業系統 | Windows 10/11、macOS、Linux 皆可 |
| Python | 3.9 ～ 3.12（建議 3.11） |
| 顯示卡 | 有 NVIDIA GPU 最佳（訓練約 2.5~3.5 小時）；無 GPU 用 CPU 會慢很多 |
| 顯存 | 建議 6GB 以上跑 YOLO11s；4GB 需把 `BATCH` 設為 8 或 4 |
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

> 虛擬環境（venv）能把套件裝在獨立空間，避免汙染系統 Python。

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

> ✅ 啟用成功後，命令列開頭會出現 `(venv)`。**之後每次要訓練，都要先啟用 venv。**

### 3. 安裝 PyTorch

⚠️ **這是最容易出錯的一步，請依你的情況選擇正確指令：**

**有 NVIDIA 顯卡（Windows / Linux）** — 裝 CUDA 版才能用 GPU 加速：
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
> `cu124` 對應 CUDA 12.4，適用多數近年的顯卡。torch 與 torchvision **務必一起安裝**，
> 確保兩者版本配對（版本不配對會導致訓練時 `torchvision::nms` 錯誤）。

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

訓練資料來自 [Roboflow](https://roboflow.com)，需要免費 API Key 才能下載：

1. 到 [roboflow.com](https://roboflow.com) 註冊免費帳號
2. 登入後左側 **Settings → API Keys** → 複製 **Private API Key**（不是 `rf_` 開頭的 Publishable Key）
3. 用文字編輯器打開 `train_roach.py`，找到最上方這行，把引號內換成你的 key：

```python
ROBOFLOW_API_KEY = "在這裡貼上你的_ROBOFLOW_API_KEY"
```

> ⚠️ API Key 等同密碼，請勿把填好 key 的腳本上傳到公開的 GitHub 或分享給他人。

### 2. 執行

確認 venv 已啟用（開頭有 `(venv)`），然後：

```
python train_roach.py      # Windows
python3 train_roach.py     # macOS / Linux
```

腳本會自動下載資料、合併、訓練、匯出。第一次會下載約 1GB 的 COCO 與 YOLO11s 權重，請耐心等候。
訓練結束後，會在這個資料夾產生 **`roach.onnx`**，並印出準確率（mAP50）。

> ✅ **驗收重點**：mAP50 數字僅供參考。真正要看的是**實際網頁測試時，漏抓與誤判有沒有改善**。
> 把新模型裝進擴充功能後，去之前會出問題的網站實測，才是最終驗收。

---

## 替換模型到擴充功能

⚠️ 從 Chrome 線上商店安裝的版本檔案唯讀、會被自動更新覆蓋，**無法直接替換**。
要使用自訓練模型，需改用「開發人員模式」載入擴充功能原始碼：

1. 取得擴充功能原始碼資料夾
2. 把訓練產生的 `roach.onnx` 複製到原始碼的 `assets/` 內，覆蓋原檔
3. Chrome 開 `chrome://extensions` → 右上角開啟「開發人員模式」
4. 點「載入未封裝項目」→ 選擇剛才放好模型的資料夾
5. 開 F12 → Console，看到 `✅ 本地模型載入完成` 即成功

> 💡 YOLO11s 的 ONNX 約 38MB，首次載入會比舊版（12MB）慢數秒，屬正常現象，之後會快取。

---

## 調整與自訂

腳本最上方可調整這些參數：

| 參數 | 說明 |
|------|------|
| `MODEL_ARCH` | 模型架構（預設 `yolo11s.pt`，見上方架構表） |
| `BATCH` | 批次大小（預設 8；顯存不足時改 4，充足時可調高加速） |
| `MAX_NEG_INSECTS` | 其他昆蟲負樣本上限（預設 1500） |
| `MAX_NEG_COCO` | COCO 背景負樣本上限（預設 1500） |
| `EPOCHS` | 訓練輪數（預設 100） |
| `IMG_SIZE` | 輸入尺寸（預設 416，**勿改**，除非同步改擴充功能的 `roach-inference.mjs`） |

**負樣本怎麼調：**
- 誤判仍多 → 調高 `MAX_NEG_*`（讓模型更保守）
- 開始漏抓 → 調低 `MAX_NEG_*`（負樣本太多會讓模型太保守）

**加入更多資料集：**
編輯 `POSITIVE_DATASETS`（蟑螂）與 `NEGATIVE_INSECT_DATASETS`（其他昆蟲）清單。
到 [Roboflow Universe](https://universe.roboflow.com/search?q=class%3Acockroach) 搜尋資料集，
點進去從網址列取得 `workspace` 與 `project` 名稱，照格式新增一行。

> 💡 **提升真實場景準確率最有效的方法**：蒐集你實際會遇到的蟑螂圖
> （卡通、迷因、廣告、各種角度），在 Roboflow 標註後加入正樣本清單。
> 這種「貼近真實使用」的資料，少量也勝過大量的標準照。

---

## 常見問題排除

### ❌ `ModuleNotFoundError: No module named 'torch'`
torch 沒有裝進目前的環境。確認：
1. venv 已啟用（命令列開頭要有 `(venv)`）
2. 重新執行安裝步驟 3、4
3. 用安裝步驟 5 的指令確認 torch 真的裝好

### ❌ `NotImplementedError: Could not run 'torchvision::nms' with arguments from the 'CUDA' backend`
torch 與 torchvision 版本不配對（一個是 CUDA 版、另一個是 CPU 版）。修法：
```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
**務必兩個一起裝**，確保版本後綴一致。

### ❌ `CUDA out of memory`
顯存不足（YOLO11s 比 nano 吃更多）。打開 `train_roach.py`，把 `BATCH = 8` 改成 `4` 再重跑。
4GB 顯卡跑 small 模型較吃緊，這很常見。

### ❌ 啟用 venv 時出現「無法載入...因為執行原則」（Windows）
PowerShell 預設禁止執行腳本。先跑：
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
再重新啟用 venv。

### ❌ 找不到 `yolo11s.pt` 或下載失敗
需要 `ultralytics >= 8.3.0` 才支援 YOLO11。先更新：
```
pip install -U ultralytics
```

### ❌ 訓練卡在下載權重或 AMP 檢查報錯
某些環境的自動混合精度（AMP）檢查會出問題。在 `train()` 的
`model.train(...)` 參數中加上 `amp=False,` 即可跳過：
```python
results = model.train(
    data=data_yaml, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
    patience=30, project="roach_blocker", name="yolo11s_v3", exist_ok=True,
    amp=False,   # 加這行
)
```

### ❌ `CUDA可用: False`（明明有 NVIDIA 顯卡）
可能裝到 CPU 版 torch，或顯卡驅動太舊。先更新顯卡驅動（GeForce Experience），
再用安裝步驟 3 的 CUDA 版指令重裝 torch。

### ⚠️ 某個資料集下載失敗
腳本會自動跳過下載失敗的資料集，用其他成功的繼續，不會中斷。

### ⚠️ 想重新訓練但不想重新下載資料
已下載的資料在 `datasets/`、`negatives/`、`merged_roach/` 資料夾中，
腳本會沿用既有檔案。若想完全重來，刪掉這幾個資料夾再執行。

### ⚠️ 換了模型架構後，擴充功能要改嗎？
不用。所有 YOLO 架構的 ONNX 輸出格式相同，只要替換 `roach.onnx` 即可，
擴充功能的程式碼不需任何修改。

---

## 相關連結
- [Roboflow Universe（找資料集）](https://universe.roboflow.com/search?q=class%3Acockroach)
- [Ultralytics YOLO11 文件](https://docs.ultralytics.com)
- [Google Colab（線上 GPU）](https://colab.research.google.com)
- [下載 Python](https://www.python.org/downloads/)
