# RoachBlocker 訓練腳本 v2（含誤判抑制）

訓練 RoachBlocker Chrome 擴充功能所用的 YOLOv8n 蟑螂偵測模型。
這版加入「負樣本」來降低誤判（false positive）。

## 為什麼要負樣本？

原本的模型只看過「有蟑螂」的圖，沒看過「長得有點像但不是蟑螂」的東西，
所以遇到其他昆蟲、深色物體、雜亂背景就容易誤判。

這版額外加入三種資料：
1. **多個蟑螂正樣本**資料集
2. **其他昆蟲**（蜘蛛/蠍子等）→ 不標註，當背景圖，教模型「這些不是蟑螂」
3. **COCO 一般物體**（約 1500 張日常物體）→ 當背景圖，降低對雜亂背景的過度敏感

原理：YOLO 對「沒有標註框的圖」會當成純背景學習，因此放入容易被誤判的圖（但不標註），
能有效壓低誤判率。

## 快速開始

1. 安裝 Python 3.9~3.12（建議 3.11）
2. 安裝套件：
   ```
   pip install -r requirements.txt
   ```
   有 NVIDIA 顯卡：先裝 CUDA 版 PyTorch：
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. 打開 `train_roach.py`，把 `ROBOFLOW_API_KEY` 換成你的 key（https://roboflow.com）
4. 執行：
   ```
   python train_roach.py     # Windows
   python3 train_roach.py    # macOS / Linux
   ```

完成後會在同資料夾產生 `roach.onnx`。

## 注意

- 這版會多下載約 1GB 的 COCO 圖片與其他昆蟲圖，**訓練時間比基本版長**（GPU 約 2~3 小時）。
- 可在腳本中調整 `MAX_NEG_INSECTS` 與 `MAX_NEG_COCO` 控制負樣本數量。
- 若誤判仍多，可調高負樣本上限；若開始漏抓，則調低（負樣本過多會讓模型太保守）。

## 電腦需求

- 作業系統：Windows / macOS / Linux
- Python 3.9~3.12
- 顯示卡：NVIDIA GPU 最佳；無 GPU 可用 CPU（較慢）
- 硬碟：約 7 GB（含 COCO）
- 記憶體：建議 8 GB 以上

## 自訂訓練資料

編輯 `train_roach.py` 中的 `POSITIVE_DATASETS`（蟑螂）與 `NEGATIVE_INSECT_DATASETS`（其他昆蟲）清單，
到 [Roboflow Universe](https://universe.roboflow.com/search?q=class%3Acockroach) 搜尋更多資料集加入。
