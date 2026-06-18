"""
RoachBlocker 蟑螂偵測模型 - 訓練腳本 v2（含 hard negative 強化）
======================================================================
相比 v1，這版額外加入「負樣本」來降低誤判：
  策略一：其他昆蟲（蜘蛛/蠍子/甲蟲等）→ 移除標註，當背景圖，教模型「這些不是蟑螂」
  策略二：COCO 一般日常物體 → 當背景圖，降低對雜亂背景/深色物體的過度敏感
  策略三：多個蟑螂正樣本資料集合併

原理：YOLO 看到「沒有標註框的圖片」會把整張當成背景學習，
      因此放入會被誤判的圖片（但不標註），能有效壓低 false positive。

使用前：
  1. pip install -r requirements.txt
  2. 把 ROBOFLOW_API_KEY 換成你自己的 key（https://roboflow.com）
  3. python train_roach.py      （Mac/Linux 用 python3）

注意：這版會多下載數千張負樣本，訓練時間比 v1 長（GPU 約 2~3 小時）。
"""
import os
import glob
import shutil
import random
import urllib.request
import zipfile
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# ============================================================
# 設定區
# ============================================================
ROBOFLOW_API_KEY = "在這裡貼上你的_ROBOFLOW_API_KEY"

VAL_RATIO = 0.2
EPOCHS = 100
IMG_SIZE = 416
BATCH = 8                  # 4GB 顯卡建議 8（OOM 再降到 4）
MODEL_ARCH = "yolo11s.pt"  # YOLO11 small：比 YOLOv8s 更準、參數更少（ONNX 約 38MB）

# 負樣本上限：避免負樣本壓過正樣本，建議負樣本總數 ≈ 正樣本數量的 0.5~1 倍
MAX_NEG_INSECTS = 1500     # 其他昆蟲負樣本上限
MAX_NEG_COCO = 1500        # COCO 背景負樣本上限

# --- 正樣本：蟑螂資料集 (workspace, project, version) ---
POSITIVE_DATASETS = [
    ("adriann", "cockroach-gkzut", 1),
    ("roach", "roach_detection", 1),
    ("cc-bhzoo", "cockroach-u5xi2", 1),
]

# --- 負樣本：其他昆蟲資料集（會移除所有標註，當背景圖）---
# 這些是「容易跟蟑螂搞混」的蟲，教模型分辨
NEGATIVE_INSECT_DATASETS = [
    ("new-workspace-v84mt", "cockroach-spider-scorpion-detection", 1),  # 含蜘蛛/蠍子
]

random.seed(42)
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(WORK_DIR, "datasets")
NEG_DIR = os.path.join(WORK_DIR, "negatives")
MERGED = os.path.join(WORK_DIR, "merged_roach")


def check_api_key():
    if "在這裡" in ROBOFLOW_API_KEY or not ROBOFLOW_API_KEY.strip():
        print("=" * 56)
        print("  錯誤：尚未設定 Roboflow API Key")
        print("  打開 train_roach.py，把 ROBOFLOW_API_KEY 換成你的 key")
        print("  免費申請：https://roboflow.com")
        print("=" * 56)
        raise SystemExit(1)


# ============================================================
# 1. 下載正樣本（蟑螂）
# ============================================================
def download_positives(rf):
    print("=" * 56, "\n[1/4] 下載蟑螂正樣本\n", "=" * 56)
    locs = []
    for ws, proj, ver in POSITIVE_DATASETS:
        try:
            print(f"下載 {ws}/{proj} v{ver} ...")
            d = rf.workspace(ws).project(proj).version(ver).download(
                "yolov8", location=os.path.join(RAW_DIR, proj)
            )
            locs.append(d.location)
            print(f"  OK: {d.location}")
        except Exception as e:
            print(f"  跳過 ({proj}): {str(e)[:100]}")
    assert locs, "沒有任何正樣本下載成功，請確認 API Key"
    return locs


# ============================================================
# 2. 下載負樣本：其他昆蟲（移除標註）
# ============================================================
def download_negative_insects(rf):
    print("=" * 56, "\n[2/4] 下載其他昆蟲負樣本\n", "=" * 56)
    imgs = []
    for ws, proj, ver in NEGATIVE_INSECT_DATASETS:
        try:
            print(f"下載 {ws}/{proj} v{ver} ...")
            d = rf.workspace(ws).project(proj).version(ver).download(
                "yolov8", location=os.path.join(NEG_DIR, proj)
            )
            for img in glob.glob(f"{d.location}/*/images/*"):
                imgs.append(img)
            print(f"  OK: 取得 {len(imgs)} 張")
        except Exception as e:
            print(f"  跳過 ({proj}): {str(e)[:100]}")
    random.shuffle(imgs)
    return imgs[:MAX_NEG_INSECTS]


# ============================================================
# 3. 下載負樣本：COCO 一般物體
# ============================================================
def download_negative_coco():
    print("=" * 56, "\n[3/4] 下載 COCO 一般物體負樣本\n", "=" * 56)
    coco_dir = os.path.join(NEG_DIR, "coco")
    os.makedirs(coco_dir, exist_ok=True)
    zip_path = os.path.join(coco_dir, "val2017.zip")

    # COCO val2017（約 1GB，5000 張各類日常物體，無蟑螂）
    url = "http://images.cocodataset.org/zips/val2017.zip"
    if not glob.glob(f"{coco_dir}/val2017/*.jpg"):
        try:
            print("下載 COCO val2017（約 1GB，請耐心等候）...")

            def _progress(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100, block_num * block_size * 100 // total_size)
                    print(f"\r  下載進度 {pct}%", end="", flush=True)

            urllib.request.urlretrieve(url, zip_path, _progress)
            print("\n解壓縮中 ...")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(coco_dir)
            os.remove(zip_path)
        except Exception as e:
            print(f"\n  COCO 下載失敗，略過此來源: {str(e)[:100]}")
            return []
    imgs = glob.glob(f"{coco_dir}/val2017/*.jpg")
    random.shuffle(imgs)
    print(f"  取得 {min(len(imgs), MAX_NEG_COCO)} 張 COCO 背景圖")
    return imgs[:MAX_NEG_COCO]


# ============================================================
# 4. 合併 + 重新切分
# ============================================================
def remap_label(src, dst):
    out = []
    with open(src) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                p[0] = "0"
                out.append(" ".join(p))
    with open(dst, "w") as f:
        f.write("\n".join(out))


def collect_positive_pairs(locs):
    pairs = []
    for loc in locs:
        tag = os.path.basename(loc).replace(" ", "_")
        for img in glob.glob(f"{loc}/*/images/*"):
            split_dir = os.path.dirname(os.path.dirname(img))
            base = os.path.basename(img)
            stem = os.path.splitext(base)[0]
            lbl = os.path.join(split_dir, "labels", stem + ".txt")
            if os.path.exists(lbl):
                pairs.append((img, lbl, f"pos_{tag}_{base}"))
    return pairs


def build_dataset(pos_pairs, neg_imgs):
    print("=" * 56, "\n[4/4] 合併資料集並切分\n", "=" * 56)
    for split in ["train", "valid"]:
        os.makedirs(f"{MERGED}/{split}/images", exist_ok=True)
        os.makedirs(f"{MERGED}/{split}/labels", exist_ok=True)

    # 正樣本：複製圖 + 標註（class 統一為 0）
    # 負樣本：只複製圖，標註檔留空（YOLO 視為純背景）
    items = []  # (img_path, label_src_or_None, out_name)
    for img, lbl, name in pos_pairs:
        items.append((img, lbl, name))
    for i, img in enumerate(neg_imgs):
        ext = os.path.splitext(img)[1] or ".jpg"
        items.append((img, None, f"neg_{i:05d}{ext}"))

    random.shuffle(items)
    n_val = int(len(items) * VAL_RATIO)
    val_items, train_items = items[:n_val], items[n_val:]

    n_pos, n_neg = 0, 0
    for split, group in [("train", train_items), ("valid", val_items)]:
        for img, lbl, name in group:
            stem = os.path.splitext(name)[0]
            try:
                shutil.copy(img, f"{MERGED}/{split}/images/{name}")
            except Exception:
                continue
            if lbl:  # 正樣本
                remap_label(lbl, f"{MERGED}/{split}/labels/{stem}.txt")
                n_pos += 1
            else:    # 負樣本 → 空標註檔
                open(f"{MERGED}/{split}/labels/{stem}.txt", "w").close()
                n_neg += 1

    with open(f"{MERGED}/data.yaml", "w") as f:
        yaml.dump({
            "train": f"{MERGED}/train/images",
            "val":   f"{MERGED}/valid/images",
            "nc": 1,
            "names": ["cockroach"],
        }, f)

    print(f"正樣本 {n_pos} 張 / 負樣本 {n_neg} 張 / 共 {n_pos + n_neg} 張")
    print(f"  訓練 {len(train_items)} / 驗證 {len(val_items)}")
    return f"{MERGED}/data.yaml"


# ============================================================
# 5. 訓練 + 匯出
# ============================================================
def train(data_yaml):
    print("=" * 56, "\n訓練", MODEL_ARCH, "\n", "=" * 56)
    model = YOLO(MODEL_ARCH)
    results = model.train(
        data=data_yaml, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
        patience=30, project="roach_blocker", name="yolo11s_v3", exist_ok=True,
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
    print(f"完成！ONNX：{onnx_dst}  ({size_mb:.1f} MB)")
    print("把 roach.onnx 放進擴充功能的 assets/ 覆蓋原檔即可。")


if __name__ == "__main__":
    check_api_key()
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    pos = download_positives(rf)
    neg_insects = download_negative_insects(rf)
    neg_coco = download_negative_coco()

    pos_pairs = collect_positive_pairs(pos)
    neg_all = neg_insects + neg_coco

    print(f"\n資料統計：正樣本 {len(pos_pairs)} 張，負樣本 {len(neg_all)} 張 "
          f"(昆蟲 {len(neg_insects)} + COCO {len(neg_coco)})")

    data_yaml = build_dataset(pos_pairs, neg_all)
    best = train(data_yaml)
    export_onnx(best)
    print("\n全部完成！")
