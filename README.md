# RoachBlocker 訓練腳本（YOLO26n 版）

訓練 RoachBlocker Chrome 擴充功能所用的蟑螂偵測模型。
此版本使用 **YOLO26n**，採 **NMS-free end2end** 架構，輸出格式為 `[1, 300, 6]`。

> ✅ 已實測驗證：此版本訓練出的 26n 模型能在瀏覽器（onnxruntime-web）正常載入與推理，
> end2end 輸出 `[1, 300, 6]` 運作正常，準確率接近 YOLO11s 但速度與檔案大小同 nano 等級。

---

## 與 YOLO11n 版的差異

| 項目 | YOLO11n 版 | YOLO26n 版（本檔） |
|------|-----------|-------------------|
| 模型架構 | yolo11n.pt | yolo26n.pt |
| 輸出格式 | `[1, 5, 3549]`（需 NMS） | `[1, 300, 6]`（end2end，免 NMS） |
| 座標格式 | xywh（中心+寬高） | xyxy（左上+右下角） |
| 後處理 | 需自寫 NMS | 只需信心門檻過濾 |
| ultralytics | >= 8.3.0 | **>= 8.4.0** |
| CPU 推理 | 基準 | 官方稱比 11n 快約 43% |

> 擴充功能的 `roach-inference.mjs` 已改為**自動偵測輸出格式**，
> 兩種模型都能載入，換模型時不需改程式碼。

### 實測數據對照（同樣資料下訓練）

| 指標 | YOLO11n | YOLO11s | **YOLO26n** |
|------|---------|---------|-------------|
| mAP50 | 0.942 | 0.960 | **0.954** |
| mAP50-95 | 0.644 | 0.696 | **0.662** |
| 推理速度 | 2.5ms | 6.3ms | **2.4ms** |
| ONNX 大小 | 10 MB | 36 MB | **9.3 MB** |
| 參數量 | 2.58M | 9.41M | **2.38M** |

結論：YOLO26n 用 nano 的速度與檔案大小，達到接近 YOLO11s 的準確率——
是「準」與「快」的最佳平衡，特別適合需在低階機器（如入門筆電）上跑的瀏覽器擴充功能。

---

## 目錄
1. [這個腳本會做什麼](#這個腳本會做什麼)
2. [電腦需求](#電腦需求)
3. [安裝步驟](#安裝步驟)
4. [執行訓練](#執行訓練)
5. [確認 end2end 匯出是否成功](#確認-end2end-匯出是否成功)
6. [替換模型到擴充功能](#替換模型到擴充功能)
7. [調整與自訂](#調整與自訂)
8. [常見問題排除](#常見問題排除)

---

## 這個腳本會做什麼

依序自動完成：

1. **下載蟑螂正樣本**：多個 Roboflow 公開資料集
2. **處理昆蟲綜合資料集**：逐張分流（含蟑螂→正樣本、不含→背景）
3. **下載 COCO 一般物體**：當背景負樣本
4. **下載商標/紋理硬背景**：對付「把木紋、地毯、商標誤判成蟑螂」
5. **合併、切分、訓練 YOLO26n、以 end2end 匯出** 成 `roach.onnx`

### 三類負樣本（對付不同誤判來源）

1. **其他昆蟲**（蜘蛛/蠍子/害蟲）→ 對付「把別種蟲認成蟑螂」
2. **COCO 一般物體** → 對付「把日常雜物認成蟑螂」
3. **商標 + 複雜紋理** → 對付「把木紋/石頭/地毯/商標認成蟑螂」

第三類由 `HARD_BG_DATASETS` 提供。**最對症的做法**：在腳本同層建立 `my_textures/`
資料夾，把你「實際被誤判」的紋理圖放進去，會自動當背景負樣本。

### ⚠️ 關鍵：負樣本不能含有蟑螂！

昆蟲綜合資料集本身可能含蟑螂。若整批當背景、抹除標註，等於教模型
「這隻蟑螂不是蟑螂」，會同時造成漏抓與準確率停滯。本腳本對昆蟲資料集
**逐張檢查標註**：含蟑螂的救回當正樣本，不含的才當背景。

### 類別名稱辨識（三分類）

類別分三種處理：明確是蟑螂（`ROACH_KEYWORDS`）→ 正樣本；
模糊類別（`AMBIGUOUS_KEYWORDS`，如 object、insect、pest 等無法判斷的名稱）
→ 整張丟棄；明確的其他生物 → 背景負樣本。
執行時 log 會印出每個資料集的判定結果，方便檢查。

---

## 電腦需求

| 項目 | 需求 |
|------|------|
| 作業系統 | Windows 10/11、macOS、Linux 皆可 |
| Python | 3.9 ～ 3.12（建議 3.11） |
| 顯示卡 | 有 NVIDIA GPU 最佳；無 GPU 用 CPU 會慢很多 |
| 硬碟空間 | 約 8 GB（含多個資料集 + COCO） |
| 記憶體 | 建議 8 GB 以上 |

> 💡 沒有 NVIDIA 顯卡？可把腳本貼到免費的
> [Google Colab](https://colab.research.google.com)（選 T4 GPU）執行。

---

## 安裝步驟

### 1. 確認 Python 已安裝
```
python --version
```
沒有的話到 [python.org](https://www.python.org/downloads/) 下載，
Windows 安裝時務必勾選「Add Python to PATH」。

### 2. 建立並啟用虛擬環境

**Windows（PowerShell）**
```powershell
cd 這個資料夾的路徑
python -m venv venv
.\venv\Scripts\Activate.ps1
```
> 若出現執行原則錯誤，先跑 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 再重試。

**macOS / Linux**
```bash
cd 這個資料夾的路徑
python3 -m venv venv
source venv/bin/activate
```

### 3. 安裝 PyTorch

**有 NVIDIA 顯卡（Windows / Linux）**
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
**macOS / 無顯卡**
```
pip install torch torchvision
```

### 4. 安裝其餘套件（含支援 YOLO26 的 ultralytics）
```
pip install -r requirements.txt
```
> ⚠️ YOLO26 需要較新的 ultralytics。若出現「找不到 yolo26n」，執行：
> ```
> pip install -U ultralytics
> ```

### 5. 確認安裝成功
```
python -c "import torch, torchvision; print('torch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

---

## 執行訓練

### 1. 填入 Roboflow API Key
到 [roboflow.com](https://roboflow.com) 註冊，複製 **Private API Key**，
打開 `train_roach.py` 把最上方這行引號內換成你的 key：
```python
ROBOFLOW_API_KEY = "在這裡貼上你的_ROBOFLOW_API_KEY"
```
> ⚠️ 勿把填好 key 的腳本上傳公開 GitHub。

### 2.（建議）先用探測模式確認資料集
```
python train_roach.py --probe
```
只下載並印出每個資料集的類別與正樣本數，不訓練。把「正樣本 0 張」或
「下載失敗」的資料集移除後，再正式訓練。

### 3. 正式訓練
```
python train_roach.py      # Windows
python3 train_roach.py     # macOS / Linux
```
訓練結束會在資料夾產生 `roach.onnx` 並印出 mAP50。

---

## 確認 end2end 匯出是否成功

**這是 YOLO26 版最重要的檢查。** 匯出後 log 會印出：
```
[檢查] 輸出 'xxx' 形狀 = [1, 300, 6]
```

- **`[1, 300, 6]`** → end2end 成功，NMS-free 生效 ✅
- **`[1, 5, xxxx]`** → end2end 未生效（退回傳統格式）。仍可使用，
  因為擴充功能會自動偵測格式，只是少了 NMS-free 的好處。

無論哪種格式，擴充功能的 `roach-inference.mjs` 都能自動處理，不需改程式碼。

---

## 替換模型到擴充功能

⚠️ Chrome 線上商店版檔案唯讀、會被自動更新覆蓋，無法直接替換。
需用「開發人員模式」載入：

1. 取得擴充功能原始碼資料夾
2. 把訓練產生的 `roach.onnx` 複製到 `assets/`，覆蓋原檔
3. `chrome://extensions` → 開啟「開發人員模式」
4. 「載入未封裝項目」→ 選擇資料夾
5. 開 F12 → Console，看到 `✅ 本地模型載入完成` 即成功

> ⚠️ **YOLO26 特別注意**：若 Console 出現載入錯誤或推理報錯，
> 很可能是 onnxruntime-web 尚未支援 YOLO26 的某些算子。
> 此時建議退回 YOLO11n 模型（它在瀏覽器上已驗證穩定）。

---

## 調整與自訂

| 參數 | 說明 |
|------|------|
| `MODEL_ARCH` | 模型架構（預設 `yolo26n.pt`） |
| `BATCH` | 批次大小（預設 16；顯存不足改 8 或 4） |
| `MAX_NEG_INSECTS` | 昆蟲負樣本上限（預設 3000） |
| `MAX_NEG_COCO` | COCO 背景負樣本上限（預設 3000） |
| `MAX_NEG_HARDBG` | 商標/紋理負樣本上限（預設 2000） |
| `EPOCHS` | 訓練輪數（預設 100） |
| `IMG_SIZE` | 輸入尺寸（預設 416，**勿改**，否則需同步改 `roach-inference.mjs` 的 `INPUT_SIZE`） |

**負樣本怎麼調**：誤判多→調高對應上限；開始漏抓→調低（負樣本過多會讓模型太保守）。

**正樣本**：一律全收不設限。注意總負樣本量別超過正樣本太多。

---

## 常見問題排除

### 找不到 `yolo26n` / 不支援 yolo26
ultralytics 版本太舊。執行 `pip install -U ultralytics`。

### `model.export()` 不接受 `end2end` 參數
ultralytics 版本太舊或該版本參數名不同。先 `pip install -U ultralytics`；
若仍不支援，移除 `end2end=True`（YOLO26 預設就是 end2end）。

### 匯出的 ONNX 在瀏覽器載入失敗
onnxruntime-web 可能尚未支援 YOLO26 的某些算子。
建議退回 YOLO11n 模型（瀏覽器上已驗證穩定）。

### `CUDA out of memory`
把 `BATCH` 改小（8 或 4）。

### `torchvision::nms` CUDA 錯誤
torch 與 torchvision 版本不配對。重裝：
```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### pip 讀 requirements.txt 出現 cp950 編碼錯誤
本檔的 requirements.txt 已為純英文。若仍遇到，手動安裝：
```
pip install "ultralytics>=8.4.0" roboflow onnx onnxruntime onnxslim pyyaml
```

### 想重訓但不想重新下載資料
已下載的資料在 `datasets/`、`negatives/`、`merged_roach/`，腳本會沿用。
想完全重來就刪掉這些資料夾。

---

## 相關連結
- [YOLO26 官方文件](https://docs.ultralytics.com/models/yolo26)
- [end2end 偵測說明](https://docs.ultralytics.com/guides/end2end-detection)
- [Roboflow Universe（找資料集）](https://universe.roboflow.com/search?q=class%3Acockroach)
- [Google Colab（線上 GPU）](https://colab.research.google.com)
