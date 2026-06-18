"""
RoachBlocker 蟑螂偵測模型 - 訓練腳本（YOLO11n 版，含標籤清洗與正樣本萃取）
======================================================================
架構：YOLO11n（nano）- 小、快、載入迅速

資料優化策略（重點：避免「負樣本內含蟑螂」污染訓練）：
  策略一：下載標準蟑螂正樣本資料集。
  策略二：下載昆蟲綜合資料集，逐張檢查標註：
          - 含蟑螂的圖 → 移入「正樣本」，只保留蟑螂標籤、抹除其他昆蟲標籤。
          - 不含蟑螂的圖 → 抹除標籤，當「純背景負樣本」。
          （這一步是關鍵：避免把含蟑螂的圖誤當背景，造成模型學到矛盾標籤而漏抓。）
  策略三：COCO 一般物體 → 當背景負樣本。

使用前：
  1. pip install -r requirements.txt
  2. 把 ROBOFLOW_API_KEY 換成你自己的 key（https://roboflow.com）
  3. python train_roach.py      （Mac/Linux 用 python3）
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
BATCH = 16                 # 11n 較輕量；OOM 再降到 8
MODEL_ARCH = "yolo11n.pt"  # YOLO11 nano（ONNX 約 10MB）

# 負樣本上限
MAX_NEG_INSECTS = 1200
MAX_NEG_COCO = 1200

# --- 正樣本：標準蟑螂資料集 (workspace, project, version) ---
POSITIVE_DATASETS = [
    ("adriann", "cockroach-gkzut", 1),
    ("roach", "roach_detection", 1),
    ("cc-bhzoo", "cockroach-u5xi2", 1),
    ("qewr65qwe4r12s1af56-wyjxp", "cockroach", 1),
]

# --- 綜合昆蟲資料集（會逐張萃取正樣本、篩選純背景）---
MIXED_INSECT_DATASETS = [
    ("new-workspace-v84mt", "cockroach-spider-scorpion-detection", 1),
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


def download_positives(rf):
    print("=" * 56, "\n[1/4] 下載標準蟑螂正樣本\n", "=" * 56)
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


def process_mixed_insects(rf):
    """下載昆蟲綜合資料集，逐張分流：含蟑螂→正樣本(清洗標籤)，不含→純背景負樣本"""
    print("=" * 56, "\n[2/4] 處理昆蟲綜合資料集（萃取正樣本 + 篩選背景）\n", "=" * 56)
    mixed_pos_pairs = []
    pure_neg_imgs = []

    for ws, proj, ver in MIXED_INSECT_DATASETS:
        try:
            print(f"下載 {ws}/{proj} v{ver} ...")
            d = rf.workspace(ws).project(proj).version(ver).download(
                "yolov8", location=os.path.join(NEG_DIR, proj)
            )
            tag = proj.replace(" ", "_")

            # 1. 從 data.yaml 找出哪些 class id 屬於蟑螂
            roach_ids = set()
            yaml_path = os.path.join(d.location, "data.yaml")
            if os.path.exists(yaml_path):
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    names = data.get("names", [])
                    if isinstance(names, list):
                        for idx, name in enumerate(names):
                            if "roach" in str(name).lower():
                                roach_ids.add(str(idx))
                    elif isinstance(names, dict):
                        for idx, name in names.items():
                            if "roach" in str(name).lower():
                                roach_ids.add(str(idx))
            print(f"  [分析] 蟑螂的 Class ID: {sorted(roach_ids) or '（無，全部當背景）'}")

            # 2. 逐張分流
            for img in glob.glob(f"{d.location}/*/images/*"):
                split_dir = os.path.dirname(os.path.dirname(img))
                base = os.path.basename(img)
                stem = os.path.splitext(base)[0]
                lbl = os.path.join(split_dir, "labels", stem + ".txt")

                has_roach = False
                if os.path.exists(lbl) and roach_ids:
                    with open(lbl, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and parts[0] in roach_ids:
                                has_roach = True
                                break

                if has_roach:
                    # 含蟑螂 → 正樣本（需清洗：只留蟑螂標籤）
                    mixed_pos_pairs.append((img, lbl, f"pos_extracted_{tag}_{base}", True, roach_ids))
                else:
                    # 不含蟑螂 → 純背景負樣本
                    pure_neg_imgs.append(img)

            print(f"  [結果] 萃取正樣本 {len(mixed_pos_pairs)} 張，純背景 {len(pure_neg_imgs)} 張")
        except Exception as e:
            print(f"  跳過 ({proj}): {str(e)[:100]}")

    random.shuffle(pure_neg_imgs)
    return mixed_pos_pairs, pure_neg_imgs[:MAX_NEG_INSECTS]


def download_negative_coco():
    print("=" * 56, "\n[3/4] 下載 COCO 一般物體負樣本\n", "=" * 56)
    coco_dir = os.path.join(NEG_DIR, "coco")
    os.makedirs(coco_dir, exist_ok=True)
    zip_path = os.path.join(coco_dir, "val2017.zip")
    url = "http://images.cocodataset.org/zips/val2017.zip"
    if not glob.glob(f"{coco_dir}/val2017/*.jpg"):
        try:
            print("下載 COCO val2017（約 1GB）...")

            def _progress(bn, bs, total):
                if total > 0:
                    pct = min(100, bn * bs * 100 // total)
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


def remap_label(src, dst):
    """標準正樣本：所有類別統一轉為 0"""
    out = []
    with open(src) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                p[0] = "0"
                out.append(" ".join(p))
    with open(dst, "w") as f:
        f.write("\n".join(out))


def remap_mixed_label(src, dst, roach_ids):
    """綜合資料集：只保留蟑螂標籤並轉為 0，其餘昆蟲標籤抹除（變相成為背景區域）"""
    out = []
    with open(src) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5 and p[0] in roach_ids:
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
                # 格式: (圖, 標籤, 新檔名, 是否混合集, 蟑螂ID集合)
                pairs.append((img, lbl, f"pos_{tag}_{base}", False, None))
    return pairs


def build_dataset(pos_pairs, neg_imgs):
    print("=" * 56, "\n[4/4] 合併資料集並切分\n", "=" * 56)
    for split in ["train", "valid"]:
        os.makedirs(f"{MERGED}/{split}/images", exist_ok=True)
        os.makedirs(f"{MERGED}/{split}/labels", exist_ok=True)

    items = []
    for img, lbl, name, is_mixed, roach_ids in pos_pairs:
        items.append((img, lbl, name, is_mixed, roach_ids))
    for i, img in enumerate(neg_imgs):
        ext = os.path.splitext(img)[1] or ".jpg"
        items.append((img, None, f"neg_{i:05d}{ext}", False, None))

    random.shuffle(items)
    n_val = int(len(items) * VAL_RATIO)
    val_items, train_items = items[:n_val], items[n_val:]

    n_pos, n_neg = 0, 0
    for split, group in [("train", train_items), ("valid", val_items)]:
        for img, lbl, name, is_mixed, roach_ids in group:
            stem = os.path.splitext(name)[0]
            try:
                shutil.copy(img, f"{MERGED}/{split}/images/{name}")
            except Exception:
                continue
            if lbl:
                if is_mixed:
                    remap_mixed_label(lbl, f"{MERGED}/{split}/labels/{stem}.txt", roach_ids)
                else:
                    remap_label(lbl, f"{MERGED}/{split}/labels/{stem}.txt")
                n_pos += 1
            else:
                open(f"{MERGED}/{split}/labels/{stem}.txt", "w").close()
                n_neg += 1

    with open(f"{MERGED}/data.yaml", "w") as f:
        yaml.dump({
            "train": f"{MERGED}/train/images",
            "val":   f"{MERGED}/valid/images",
            "nc": 1,
            "names": ["cockroach"],
        }, f)

    print(f"最終統計：正樣本 {n_pos} 張 / 負樣本 {n_neg} 張 / 共 {n_pos + n_neg} 張")
    print(f"  訓練 {len(train_items)} / 驗證 {len(val_items)}")
    return f"{MERGED}/data.yaml"


def train(data_yaml):
    print("=" * 56, "\n訓練", MODEL_ARCH, "\n", "=" * 56)
    model = YOLO(MODEL_ARCH)
    results = model.train(
        data=data_yaml, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
        patience=30, project="roach_blocker", name="yolo11n_v4", exist_ok=True,
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
    mixed_pos_pairs, neg_insects = process_mixed_insects(rf)
    neg_coco = download_negative_coco()

    pos_pairs = collect_positive_pairs(pos) + mixed_pos_pairs
    neg_all = neg_insects + neg_coco

    print("\n資料流統計：")
    print(f"  - 標準集正樣本: {len(pos_pairs) - len(mixed_pos_pairs)} 張")
    print(f"  - 昆蟲集萃取正樣本: {len(mixed_pos_pairs)} 張")
    print(f"  - 總正樣本: {len(pos_pairs)} 張")
    print(f"  - 總背景負樣本: {len(neg_all)} 張 (昆蟲背景 {len(neg_insects)} + COCO {len(neg_coco)})")

    data_yaml = build_dataset(pos_pairs, neg_all)
    best = train(data_yaml)
    export_onnx(best)
    print("\n全部完成！")
