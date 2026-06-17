# RoachBlocker 訓練腳本

訓練 RoachBlocker Chrome 擴充功能所用的 YOLOv8n 蟑螂偵測模型。

## 快速開始

1. 安裝 Python 3.9~3.12（建議 3.11）
2. 安裝套件：
   ```
   pip install -r requirements.txt
   ```
   有 NVIDIA 顯卡的話，先裝 CUDA 版 PyTorch：
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. 打開 `train_roach.py`，把 `ROBOFLOW_API_KEY` 換成你自己的 key（免費申請：https://roboflow.com）
4. 執行：
   ```
   python train_roach.py     # Windows
   python3 train_roach.py    # macOS / Linux
   ```

完成後會在同資料夾產生 `roach.onnx`。

## 電腦需求

- 作業系統：Windows / macOS / Linux
- Python 3.9~3.12
- 顯示卡：NVIDIA GPU 最佳；無 GPU 可用 CPU（較慢）
- 硬碟：約 5 GB
- 記憶體：建議 8 GB 以上

沒有 GPU 也可以把腳本內容貼到免費的 [Google Colab](https://colab.research.google.com) 執行。

## 自訂訓練資料

編輯 `train_roach.py` 中的 `DATASETS` 清單，到 [Roboflow Universe](https://universe.roboflow.com/search?q=class%3Acockroach) 搜尋更多蟑螂資料集加入。
