"""
RoachBlocker 蟑螂偵測模型 - 訓練腳本
===========================================
這個腳本會自動：下載公開蟑螂資料集 → 合併並重新切分 → 訓練 YOLOv8n → 匯出成 ONNX

使用前：
  1. 安裝相依套件（見 README 或使用說明分頁）
  2. 把下方的 ROBOFLOW_API_KEY 換成你自己的 key（免費申請：https://roboflow.com）
  3. 執行：  python train_roach.py

完成後會在同資料夾產生 roach.onnx，
把它複製到擴充功能的 assets/ 資料夾、覆蓋原本的 roach.onnx 即可。

支援：Windows / macOS / Linux
"""
import os
import glob
import shutil
import random
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# ============================================================
# 設定區：把這裡換成你自己的 Roboflow API Key
# ============================================================
ROBOFLOW_API_KEY = "在這裡貼上你的_ROBOFLOW_API_KEY"

# 訓練參數（一般不用改）
VAL_RATIO = 0.2          # 驗證集比例 (20%)
EPOCHS = 100             # 訓練輪數
IMG_SIZE = 416           # 輸入尺寸，必須與擴充功能一致（勿改，除非同步改 roach-inference.mjs）
BATCH = 16               # 批次大小，GPU 記憶體不足時改小（如 8 或 4）

# 公開蟑螂資料集 (workspace, project, version)
# 想加更多資料集，到 https://universe.roboflow.com 搜尋 cockroach，
# 點進資料集 → Download → 看網址列的 workspace/project，照格式加進來即可。
DATASETS = [
    ("adriann", "cockroach-gkzut", 1),
    ("roach", "roach_detection", 1),
    ("cc-bhzoo", "cockroach-u5xi2", 1),
]

random.seed(42)
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(WORK_DIR, "datasets")
MERGED = os.path.join(WORK_DIR, "merged_roach")


def check_api_key():
    if "在這裡" in ROBOFLOW_API_KEY or not ROBOFLOW_API_KEY.strip():
        print("=" * 56)
        print("  錯誤：尚未設定 Roboflow API Key")
        print("  請打開 train_roach.py，找到 ROBOFLOW_API_KEY")
        print("  把它換成你自己的 key（免費申請：https://roboflow.com）")
        print("=" * 56)
        raise SystemExit(1)


def download_datasets():
    print("=" * 56, "\n下載資料集\n", "=" * 56)
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    downloaded = []
    for ws, proj, ver in DATASETS:
        try:
            print(f"下載 {ws}/{proj} v{ver} ...")
            d = rf.workspace(ws).project(proj).version(ver).download(
                "yolov8", location=os.path.join(RAW_DIR, proj)
            )
            downloaded.append(d.location)
            print(f"  OK: {d.location}")
        except Exception as e:
            print(f"  跳過 ({proj}): {str(e)[:120]}")
    assert downloaded, "沒有任何資料集下載成功，請確認 API Key 是否正確"
    return downloaded


def remap_label(src, dst):
    """所有類別統一成 0（單一 cockroach 類別）"""
    out = []
    with open(src) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                p[0] = "0"
                out.append(" ".join(p))
    with open(dst, "w") as f:
        f.write("\n".join(out))


def collect_pairs(downloaded):
    """蒐集所有 (圖片, 標註) 配對，不分原本的 split"""
    pairs = []
    for loc in downloaded:
        tag = os.path.basename(loc).replace(" ", "_")
        for img in glob.glob(f"{loc}/*/images/*"):
            split_dir = os.path.dirname(os.path.dirname(img))
            base = os.path.basename(img)
            stem = os.path.splitext(base)[0]
            lbl = os.path.join(split_dir, "labels", stem + ".txt")
            if os.path.exists(lbl):
                pairs.append((img, lbl, f"{tag}_{base}", f"{tag}_{stem}.txt"))
    return pairs


def resplit_and_merge(downloaded):
    print("=" * 56, "\n打散重新切分 80/20\n", "=" * 56)
    for split in ["train", "valid"]:
        os.makedirs(f"{MERGED}/{split}/images", exist_ok=True)
        os.makedirs(f"{MERGED}/{split}/labels", exist_ok=True)

    pairs = collect_pairs(downloaded)
    random.shuffle(pairs)
    n_val = int(len(pairs) * VAL_RATIO)
    val_set, train_set = pairs[:n_val], pairs[n_val:]

    for split, dataset in [("train", train_set), ("valid", val_set)]:
        for img, lbl, img_name, lbl_name in dataset:
            shutil.copy(img, f"{MERGED}/{split}/images/{img_name}")
            remap_label(lbl, f"{MERGED}/{split}/labels/{lbl_name}")

    with open(f"{MERGED}/data.yaml", "w") as f:
        yaml.dump({
            "train": f"{MERGED}/train/images",
            "val":   f"{MERGED}/valid/images",
            "nc": 1,
            "names": ["cockroach"],
        }, f)

    print(f"總計 {len(pairs)} 張 → 訓練 {len(train_set)} / 驗證 {len(val_set)}")
    return f"{MERGED}/data.yaml"


def train(data_yaml):
    print("=" * 56, "\n訓練 YOLOv8n\n", "=" * 56)
    model = YOLO("yolov8n.pt")
    # device 留空 → ultralytics 自動選擇（有 NVIDIA GPU 用 GPU，否則用 CPU）
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=30,
        project="roach_blocker",
        name="yolov8n",
        exist_ok=True,
    )
    return os.path.join(str(results.save_dir), "weights", "best.pt")


def export_onnx(best_pt):
    print("=" * 56, "\n匯出 ONNX\n", "=" * 56)
    model = YOLO(best_pt)
    val = model.val()
    print(f"\n>>> mAP50: {val.box.map50:.3f}   mAP50-95: {val.box.map:.3f} <<<\n")
    model.export(format="onnx", imgsz=IMG_SIZE, opset=12, simplify=True, dynamic=False)
    onnx_src = best_pt.replace("best.pt", "best.onnx")
    onnx_dst = os.path.join(WORK_DIR, "roach.onnx")
    shutil.copy(onnx_src, onnx_dst)
    size_mb = os.path.getsize(onnx_dst) / 1024 / 1024
    print(f"完成！ONNX 已存到：{onnx_dst}  ({size_mb:.1f} MB)")
    print("把這個 roach.onnx 複製到擴充功能的 assets/ 資料夾覆蓋原檔即可。")


if __name__ == "__main__":
    check_api_key()
    dl = download_datasets()
    yml = resplit_and_merge(dl)
    best = train(yml)
    export_onnx(best)
    print("\n全部完成！")
