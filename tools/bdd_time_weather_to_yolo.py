# tools/bdd_time_weather_to_yolo.py
import json, os, shutil
from pathlib import Path
from PIL import Image

# ===== 사용자 설정 =====
IMG_ROOT = Path(r"C:\Users\korea\Downloads\bdd100k_images_100k\100k")  # train/val/test 디렉토리 존재
LABEL_ROOT = Path(r"C:\Users\korea\Downloads\bdd100k_labels\100k")     # train.json, val.json (test는 라벨X)
OUT_ROOT = Path(r".\datasets\yolo_by_time_weather")                    # 결과 루트 (프로젝트 내부)
COPY_MODE = "link"  # "link"=하드링크(빠름, NTFS 필요), "copy"=실제 복사

# 사용할 클래스 (YOLO names 순서)
NAMES = [
    "person","rider","car","bus","truck","bike","motor",
    "traffic light","traffic sign","train"
]
CLS2ID = {c:i for i,c in enumerate(NAMES)}

# 버킷 규칙
CLEAR_SET = {"clear","partly cloudy","overcast"}
ADVERSE_SET = {"rainy","snowy","foggy"}
TIME_MAP = {
    "day": "day",
    "night": "night",
    "dawn/dusk": "dawn_dusk",
    "dawn_dusk": "dawn_dusk",
}

# time x weather → 버킷명
def bucket_name(attr):
    w = (attr.get("weather") or "").lower()
    t = (attr.get("timeofday") or "").lower()
    t = TIME_MAP.get(t, "undefined")
    if w in CLEAR_SET and t in ("day","night"):
        return f"clear_{t}"
    if w in ADVERSE_SET and t in ("day","night"):
        return f"adverse_{t}"
    return None  # 나머지는 제외

def load_bdd_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    for item in data:
        name = item.get("name")
        attrs = item.get("attributes", {})
        # frames 형식/labels 형식 모두 대응
        if "labels" in item:
            objs = item["labels"]
        elif "frames" in item and item["frames"]:
            objs = item["frames"][0].get("objects", [])
        else:
            objs = []
        idx[name] = (attrs, objs)
    return idx

def ensure_dirs(base, split):
    (base / "images" / split).mkdir(parents=True, exist_ok=True)
    (base / "labels" / split).mkdir(parents=True, exist_ok=True)

def yolo_line(box, W, H, cls_id):
    x1,y1,x2,y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    # clamp
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1+x2)/2) / W
    cy = ((y1+y2)/2) / H
    bw = (x2-x1)/W
    bh = (y2-y1)/H
    if bw <= 0 or bh <= 0: 
        return None
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if COPY_MODE == "link":
        if dst.exists(): return
        os.link(src, dst)  # hardlink (NTFS)
    else:
        if not dst.exists():
            shutil.copy2(src, dst)

def process_split(split):
    print(f"\n=== Split: {split} ===")
    # 경로
    img_dir = IMG_ROOT / split
    json_path = LABEL_ROOT / f"{split}.json"
    if not json_path.exists():
        print(f"[WARN] {json_path} not found (test에는 라벨이 없을 수 있음). skip.")
        return

    labels_idx = load_bdd_labels(json_path)
    total, kept = 0, 0
    bucket_stats = {}

    for name, (attrs, objs) in labels_idx.items():
        total += 1
        bname = bucket_name(attrs)
        if not bname:
            continue  # 제외 (undefined 등)

        out_base = OUT_ROOT / bname
        ensure_dirs(out_base, split)

        img_src = img_dir / name
        if not img_src.exists():
            # 이미지가 val 폴더에 있는데 train.json에 있다? 등 경로오류 예방
            continue

        # 이미지 크기
        try:
            with Image.open(img_src) as im:
                W, H = im.size
        except Exception as e:
            print(f"[ERR] open image failed: {img_src} ({e})")
            continue

        # YOLO 라벨 저장
        lines = []
        for o in objs:
            cat = o.get("category","")
            if cat not in CLS2ID: 
                continue
            box = o.get("box2d")
            if not box: 
                continue
            L = yolo_line(box, W, H, CLS2ID[cat])
            if L: lines.append(L)

        img_dst = out_base / "images" / split / name
        lbl_dst = out_base / "labels" / split / (Path(name).stem + ".txt")

        link_or_copy(img_src, img_dst)
        lbl_dst.write_text("\n".join(lines), encoding="utf-8")

        kept += 1
        bucket_stats[bname] = bucket_stats.get(bname, 0) + 1

    print(f"[DONE] {split}: kept {kept}/{total}")
    for k,v in sorted(bucket_stats.items()):
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # json 파일이 실제로 있을 때만 처리
    for sp in ("train", "val", "test"):
        json_path = LABEL_ROOT / f"{sp}.json"
        if json_path.exists():
            process_split(sp)
        else:
            print(f"[SKIP] {json_path} 없음")

