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

# 負樣本上限（只限制負樣本；正樣本一律全收，不設上限）
MAX_NEG_INSECTS = 3000
MAX_NEG_COCO = 3000
MAX_NEG_HARDBG = 2000      # 第三類負樣本（商標 + 紋理）上限

# --- 正樣本：標準蟑螂資料集 (workspace, project, version) ---
# 註：已移除 qewr65qwe4r12s1af56-wyjxp/cockroach —— 它的類別其實是
#     Tribolium castaneum（赤擬穀盜，一種甲蟲，不是蟑螂），會污染模型。
# 注意：download_positives 會把這些資料集的「所有類別」一律當成蟑螂（class 0）。
#       因此這裡只能放「人工確認過、整個資料集都是蟑螂」的來源。
#       下面 esp3902 / -0ujgl / cockroach3 標籤命名雖不清楚，但已人工確認是蟑螂。
POSITIVE_DATASETS = [
    ("adriann", "cockroach-gkzut", 1),
    # 以下標籤命名不清楚，但已人工確認內容是蟑螂：
    ("school-project-cwbwv", "esp3902", 1),
    ("guet-wnqb9", "-0ujgl", 3),
    ("yun-oykhq", "cockroach3-iwlo0", 1),
]

# --- 綜合昆蟲/害蟲資料集（會逐張萃取正樣本、篩選純背景）---
# 這些含多種害蟲，腳本會自動分流：含蟑螂→正樣本、不含→背景負樣本
MIXED_INSECT_DATASETS = [
    ("new-workspace-v84mt", "cockroach-spider-scorpion-detection", 1),
    ("tini", "pest-detection-0sv8g", 5),
    ("air-uni-i206m", "pest-detection-yu4hv", 7),
    ("pestmodel", "pest-detector-dataset", 11),
    ("purizumo", "pestguard", 2),                          # ~20 類含 cockroach，Public Domain
    ("jose-rizal-university", "pest-detection-2", 6),      # 703 張，含 Cockroach
    # 以下為使用者提供，跑時看 log 的「判定為蟑螂的類別」確認是否含蟑螂：
    ("insects-gt20t", "dataset-zxann", 1),
    ("ai-camp-project", "dynamite-duelers-project", 42),
    ("mindcue", "combo-dataset", 3),
    ("sams-sgift", "pest-detection-qbalv", 5),
    ("lab-889z6", "1600_1200", 3),
    ("patronusmobiles-workspace", "pest-detection-vuziq", 5),
    ("s-workspace-ddomg", "my-first-project_mix_mechanism", 2),
    ("mariam-eq11t", "insectdetectionn", 1),
    ("126-thanakool-wongsutthikul", "kusk-ai-pest-detect-2n8qu", 3),
    ("tiger-emltm", "insects-9yf6s", 2),
    ("cc-bhzoo", "cockroach-u5xi2", 1),
    ("roach", "roach_detection", 5),
    ("mayurworkspace", "forest_animal_identification", 1),
    ("mushroomcare-research", "pestdetect", 1),
    ("object-detection-cafff", "object_detection1-9xmkt", 3),
]

# --- 第三類負樣本：商標 / 複雜紋理（整批當背景，不分流）---
# 用途：壓低「把木紋、石頭、地毯、商標等誤判成蟑螂」的情況。
# 這些資料集裡不會有蟑螂，所以整批當純背景負樣本即可。
HARD_BG_DATASETS = [
    ("data6000", "brand-logo-recognition-yolov8", 1),   # 503 張品牌商標
    ("fyp1-aidez", "logo-juzxl", 1),                    # 5440 張各種logo
    ("ai-dataset-8dqwo", "carpet-rjju3", 2),            # 201 張地毯紋理
]
# 本地紋理資料夾：把你自己蒐集的木紋/石頭/大理石/布料等圖片
# 放進腳本同層的 my_textures/ 資料夾，會自動當背景負樣本（最對症）。
LOCAL_TEXTURE_DIR = "my_textures"

random.seed(42)
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(WORK_DIR, "datasets")
NEG_DIR = os.path.join(WORK_DIR, "negatives")
MERGED = os.path.join(WORK_DIR, "merged_roach")

# 判斷某類別名是否屬於蟑螂（全部轉小寫比對）
# 涵蓋：cockroach、roach、nymph（蟑螂若蟲）、蟑螂、學名 periplaneta/blattella 等
# 排除：tribolium（赤擬穀盜，非蟑螂）等易誤判詞
ROACH_KEYWORDS = ["cockroach", "roach", "nymph", "蟑螂", "periplaneta", "blattodea", "blattella", "blatta"]
ROACH_EXCLUDE = ["tribolium", "approach", "broach"]

# 模糊類別：名稱無法判斷裡面是不是蟑螂。
# 只要一張圖含有任一模糊類別，整張圖「直接丟棄」——
# 不進正樣本（不確定是蟑螂）也不進負樣本（可能其實有蟑螂，當背景會污染訓練）。
AMBIGUOUS_KEYWORDS = [
    # 泛稱（可能是任何東西）
    "object", "objects", "object detection", "thing", "things", "item", "items",
    "target", "targets", "detection", "detect", "detected", "roi",
    # 昆蟲/害蟲泛稱（可能包含蟑螂但沒明說）
    "insect", "insects", "pest", "pests", "bug", "bugs", "creature", "creatures",
    "animal", "animals", "vermin", "critter",
    # 占位／未命名
    "unknown", "unlabeled", "unlabelled", "label", "labels", "none", "other",
    "others", "misc", "miscellaneous", "mode", "model", "class", "default", "test",
]
# 這些「短的純占位詞」用完全相符比對，避免誤砍正常字（例如 na 不該砍到 banana）
AMBIGUOUS_EXACT = {"na", "n/a", "nan", "0", "1", "2", "3", "obj", "cls", "id"}


def is_roach_class(name):
    n = str(name).lower().strip()
    if any(ex in n for ex in ROACH_EXCLUDE):
        return False
    return any(kw in n for kw in ROACH_KEYWORDS)


def is_ambiguous_class(name):
    """名稱模糊、無法判斷是否含蟑螂 → 該圖整張丟棄較安全"""
    n = str(name).lower().strip()
    # 已明確是蟑螂的，不算模糊
    if is_roach_class(n):
        return False
    if n in AMBIGUOUS_EXACT:
        return True
    return any(kw in n for kw in AMBIGUOUS_KEYWORDS)


def check_api_key():
    if "在這裡" in ROBOFLOW_API_KEY or not ROBOFLOW_API_KEY.strip():
        print("=" * 56)
        print("  錯誤：尚未設定 Roboflow API Key")
        print("  打開 train_roach.py，把 ROBOFLOW_API_KEY 換成你的 key")
        print("  免費申請：https://roboflow.com")
        print("=" * 56)
        raise SystemExit(1)


def download_positives(rf):
    print("=" * 56, "\n[1/5] 下載標準蟑螂正樣本\n", "=" * 56)
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
    print("=" * 56, "\n[2/5] 處理昆蟲綜合資料集（萃取正樣本 + 篩選背景）\n", "=" * 56)
    mixed_pos_pairs = []
    pure_neg_imgs = []

    for ws, proj, ver in MIXED_INSECT_DATASETS:
        try:
            print(f"下載 {ws}/{proj} v{ver} ...")
            d = rf.workspace(ws).project(proj).version(ver).download(
                "yolov8", location=os.path.join(NEG_DIR, proj)
            )
            tag = proj.replace(" ", "_")

            # 1. 從 data.yaml 找出哪些 class id 屬於蟑螂 / 屬於模糊類別
            roach_ids = set()
            ambiguous_ids = set()
            yaml_path = os.path.join(d.location, "data.yaml")
            roach_names = []
            ambiguous_names = []
            if os.path.exists(yaml_path):
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    names = data.get("names", [])
                    pairs = enumerate(names) if isinstance(names, list) else (names.items() if isinstance(names, dict) else [])
                    for idx, name in pairs:
                        if is_roach_class(name):
                            roach_ids.add(str(idx))
                            roach_names.append(name)
                        elif is_ambiguous_class(name):
                            ambiguous_ids.add(str(idx))
                            ambiguous_names.append(name)
            print(f"  [分析] 此資料集所有類別: {names if names else '（讀不到）'}")
            print(f"  [分析] 判定為蟑螂的類別: {roach_names or '（無）'} → ID {sorted(roach_ids) or '無'}")
            if ambiguous_names:
                print(f"  [分析] 模糊類別（含此類的圖將整張丟棄）: {ambiguous_names} → ID {sorted(ambiguous_ids)}")

            # 2. 逐張分流（三選一）：
            #    - 含蟑螂 → 正樣本（清洗標籤，只留蟑螂）
            #    - 含模糊類別 → 整張丟棄（不確定有沒有蟑螂，當背景會污染）
            #    - 其餘（明確的其他生物）→ 純背景負樣本
            ds_pos, ds_neg, ds_skip = 0, 0, 0
            for img in glob.glob(f"{d.location}/*/images/*"):
                split_dir = os.path.dirname(os.path.dirname(img))
                base = os.path.basename(img)
                stem = os.path.splitext(base)[0]
                lbl = os.path.join(split_dir, "labels", stem + ".txt")

                has_roach = False
                has_ambiguous = False
                if os.path.exists(lbl):
                    with open(lbl, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            cid = parts[0]
                            if cid in roach_ids:
                                has_roach = True
                            elif cid in ambiguous_ids:
                                has_ambiguous = True

                if has_roach:
                    # 含蟑螂 → 正樣本（清洗：只留蟑螂標籤），全收不設限
                    mixed_pos_pairs.append((img, lbl, f"pos_extracted_{tag}_{base}", True, roach_ids))
                    ds_pos += 1
                elif has_ambiguous:
                    # 含模糊類別、又無明確蟑螂 → 安全起見整張丟棄
                    ds_skip += 1
                else:
                    # 只有明確的其他生物 → 純背景負樣本
                    pure_neg_imgs.append(img)
                    ds_neg += 1

            print(f"  [結果] 此資料集 → 正樣本 {ds_pos} 張，背景負樣本 {ds_neg} 張，丟棄(模糊) {ds_skip} 張")
        except Exception as e:
            print(f"  跳過 ({proj}): {str(e)[:100]}")

    random.shuffle(pure_neg_imgs)
    return mixed_pos_pairs, pure_neg_imgs[:MAX_NEG_INSECTS]


def download_negative_coco():
    print("=" * 56, "\n[3/5] 下載 COCO 一般物體負樣本\n", "=" * 56)
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


def download_hard_backgrounds(rf):
    """第三類負樣本：商標 + 紋理（整批當背景，不分流）+ 本地紋理資料夾"""
    print("=" * 56, "\n[4/5] 下載商標/紋理硬背景負樣本\n", "=" * 56)
    imgs = []
    # 1) Roboflow 上的商標 / 紋理資料集
    for ws, proj, ver in HARD_BG_DATASETS:
        try:
            print(f"下載 {ws}/{proj} v{ver} ...")
            d = rf.workspace(ws).project(proj).version(ver).download(
                "yolov8", location=os.path.join(NEG_DIR, "hardbg_" + proj)
            )
            got = glob.glob(f"{d.location}/*/images/*")
            imgs.extend(got)
            print(f"  OK: +{len(got)} 張")
        except Exception as e:
            print(f"  跳過 ({proj}): {str(e)[:100]}")

    # 2) 本地自備紋理資料夾（最對症：放你實際被誤判的木紋/石頭/地毯等）
    local_dir = os.path.join(WORK_DIR, LOCAL_TEXTURE_DIR)
    if os.path.isdir(local_dir):
        local = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            local.extend(glob.glob(os.path.join(local_dir, "**", ext), recursive=True))
        if local:
            print(f"  本地 {LOCAL_TEXTURE_DIR}/：+{len(local)} 張")
            imgs.extend(local)
    else:
        print(f"  （提示：可建立 {LOCAL_TEXTURE_DIR}/ 資料夾放自備紋理圖，效果最好）")

    random.shuffle(imgs)
    print(f"  硬背景負樣本合計取用 {min(len(imgs), MAX_NEG_HARDBG)} 張")
    return imgs[:MAX_NEG_HARDBG]


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
    print("=" * 56, "\n[5/5] 合併資料集並切分\n", "=" * 56)
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
    import sys
    # 探測模式：只下載資料集並印出類別/正樣本統計，不訓練。
    # 用法：python train_roach.py --probe
    # 目的：先確認哪些資料集能用、含不含蟑螂，再決定要不要花時間訓練。
    PROBE_ONLY = "--probe" in sys.argv

    check_api_key()
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    pos = download_positives(rf)
    mixed_pos_pairs, neg_insects = process_mixed_insects(rf)

    print("\n資料流統計（正樣本部分）：")
    print(f"  - 標準集正樣本: {len(collect_positive_pairs(pos))} 張")
    print(f"  - 昆蟲/害蟲集萃取正樣本: {len(mixed_pos_pairs)} 張")

    if PROBE_ONLY:
        print("\n[探測模式] 只檢查資料集，不訓練。")
        print("請往上檢視每個資料集的『判定為蟑螂的類別』與『正樣本張數』，")
        print("把沒貢獻正樣本或下載失敗的資料集從清單移除後，再正式執行訓練。")
        raise SystemExit(0)

    neg_coco = download_negative_coco()
    neg_hardbg = download_hard_backgrounds(rf)
    pos_pairs = collect_positive_pairs(pos) + mixed_pos_pairs
    neg_all = neg_insects + neg_coco + neg_hardbg

    print("\n資料流統計：")
    print(f"  - 標準集正樣本: {len(pos_pairs) - len(mixed_pos_pairs)} 張")
    print(f"  - 昆蟲集萃取正樣本: {len(mixed_pos_pairs)} 張")
    print(f"  - 總正樣本: {len(pos_pairs)} 張")
    print(f"  - 總背景負樣本: {len(neg_all)} 張 "
          f"(昆蟲 {len(neg_insects)} + COCO {len(neg_coco)} + 商標/紋理 {len(neg_hardbg)})")

    data_yaml = build_dataset(pos_pairs, neg_all)
    best = train(data_yaml)
    export_onnx(best)
    print("\n全部完成！")
