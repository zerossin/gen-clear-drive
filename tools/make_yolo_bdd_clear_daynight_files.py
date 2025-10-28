# tools/make_yolo_bdd_clear_daytimenight_files.py
import json, os, shutil, sys, argparse, time
from pathlib import Path
from PIL import Image

# ===== 입력 경로 (네가 가진 실제 경로) =====
LABEL_ROOT = Path(r"C:\Users\korea\Downloads\bdd100k_labels\100k")         # train/val/test 폴더, 그 안에 *.json
IMG_ROOT   = Path(r"C:\Users\korea\Downloads\bdd100k_images_100k\100k")    # train/val/test 폴더, 그 안에 *.jpg

# ===== 출력 경로 =====
OUT_ROOT = Path(r".\datasets\yolo_bdd100k")  # gen-clear-drive 기준(현재 작업 디렉토리)

# 링크/복사 정책: 기본 하드링크, 실패 시 복사로 폴백
PREFER_LINK = True

# COCO 80 클래스 이름 (YOLO COCO 사전학습과 동일 순서)
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# BDD → COCO id 매핑 (traffic sign은 기본 드롭)
BDD2COCO = {
    "person": 0,
    "rider": 0,           # COCO에 rider 없음 → person으로 합침
    "car": 2,
    "motor": 3,           # BDD 'motor' == COCO 'motorcycle'
    "bus": 5,
    "train": 6,
    "truck": 7,
    "bike": 1,            # BDD 'bike' == COCO 'bicycle'
    "traffic light": 9,
    # "traffic sign": 11, # 필요 시 stop sign(11)로 강매핑 가능. 권장은 드롭.
}

# 필터: weather=clear만 사용
CLEAR_ONLY = {"clear"}

# time bucket: daytime, night만 사용
def bucket_from_attr(attr: dict):
    w = (attr.get("weather") or "").lower()
    t = (attr.get("timeofday") or "").lower()
    if w in CLEAR_ONLY:
        if t == "daytime":
            return "clear_daytime"
        if t == "night":
            return "clear_night"
    return None  # 그 외는 제외

def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if PREFER_LINK:
        try:
            if not dst.exists():
                os.link(src, dst)
            return
        except Exception:
            pass
    if not dst.exists():
        shutil.copy2(src, dst)

def iter_jsons(folder: Path):
    # 폴더 아래의 *.json 모두
    for p in folder.glob("*.json"):
        yield p

def load_bdd_per_image(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    name = d.get("name")              # 예: cabc9045-d91ecb66.jpg
    attrs = d.get("attributes", {})   # weather, timeofday 등
    # frames[0].objects 형태
    objs = []
    frames = d.get("frames") or []
    if frames:
        objs = frames[0].get("objects", []) or []
    return name, attrs, objs

def yolo_line(box, W, H, cls_id):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    # clamp
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    if bw <= 0 or bh <= 0:
        return None
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def process_split(split: str, copy_images: bool = True, write_labels: bool = True, log_every: int = 500):
    print(f"\n=== Split: {split} ===", flush=True)
    json_dir = LABEL_ROOT / split
    img_dir  = IMG_ROOT / split
    if not json_dir.exists():
        print(f"[SKIP] {json_dir} 없음")
        return

    # 전체 파일 수 파악 (진행률 표시용)
    try:
        total_jsons = sum(1 for _ in iter_jsons(json_dir))
    except Exception:
        total_jsons = 0
    print(f"[INFO] json dir: {json_dir} | images dir: {img_dir} | files: {total_jsons} | copy_images={copy_images} | write_labels={write_labels}", flush=True)

    kept = {"clear_daytime": 0, "clear_night": 0}
    processed = 0
    t0 = time.time()

    for jf in iter_jsons(json_dir):
        processed += 1
        name, attrs, objs = load_bdd_per_image(jf)
        if not name:
            continue
        # 일부 JSON은 name에 확장자 없이 들어올 수 있으니 보정
        name_jpg = name if name.lower().endswith(".jpg") else name + ".jpg"
        img_src = img_dir / name_jpg
        if not img_src.exists():
            # 드물게 png일 수 있으면 필요 시 확장자 탐색
            alt = img_src.with_suffix(".png")
            if alt.exists():
                img_src = alt
            else:
                continue

        bname = bucket_from_attr(attrs)
        if bname is None:
            continue  # clear/daytime or clear/night가 아닌 경우 제외

        # YOLO 라벨 생성 (BDD → COCO id 매핑 적용)
        lines = []
        if write_labels:
            # 이미지 크기 (라벨 생성 시에만 필요)
            try:
                with Image.open(img_src) as im:
                    W, H = im.size
            except Exception as e:
                print(f"[ERR] open failed: {img_src} ({e})")
                continue

            for o in objs:
                cat = o.get("category", "")
                box = o.get("box2d")
                if not box:
                    continue
                cid = BDD2COCO.get(cat)
                if cid is None:
                    continue  # 매핑 불가(예: traffic sign 드롭)
                L = yolo_line(box, W, H, cid)
                if L:
                    lines.append(L)

        base = OUT_ROOT / bname
        img_dst = base / "images" / split / img_src.name
        lbl_dst = base / "labels" / split / (img_src.stem + ".txt")
        if copy_images:
            link_or_copy(img_src, img_dst)
        if write_labels:
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.write_text("\n".join(lines), encoding="utf-8")

        kept[bname] += 1

        # 주기적 진행 로그
        if log_every and (processed % log_every == 0):
            dt = time.time() - t0
            rate = processed / dt if dt > 0 else 0
            print(f"[PROGRESS] {split}: {processed}/{total_jsons if total_jsons else '?'} files | kept day={kept['clear_daytime']} night={kept['clear_night']} | {rate:.1f} files/s", flush=True)

    dt = time.time() - t0
    print(f"[DONE] {split}: processed={processed} kept day={kept['clear_daytime']} night={kept['clear_night']} | elapsed={dt:.1f}s", flush=True)

def write_yaml(bname: str, include_test=True):
    (OUT_ROOT / bname).mkdir(parents=True, exist_ok=True)
    y = []
    y.append(f"path: datasets/yolo_bdd100k/{bname}")
    y.append("train: images/train")
    y.append("val: images/val")
    if include_test:
        y.append("test: images/test")
    y.append("")
    y.append("names:")
    for n in COCO_NAMES:
        y.append(f"  - {n}")
    (OUT_ROOT / bname / "data.yaml").write_text("\n".join(y), encoding="utf-8")

# synth도 공용 YAML 작성 함수 사용으로 일원화

def copy_test_labels_with_suffix(src_base: Path, dst_base: Path, suffix: str = "_fake") -> int:
    """
    src_base/labels/test 의 모든 .txt를 dst_base/labels/test로 복사하되,
    파일명 끝에 suffix를 붙여서 저장한다.
    예: name.txt -> name_fake.txt

    반환값은 복사된 파일 수.
    """
    src_dir = src_base / "labels" / "test"
    dst_dir = dst_base / "labels" / "test"
    if not src_dir.exists():
        print(f"[WARN] source test labels not found: {src_dir}")
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for p in src_dir.glob("*.txt"):
        new_name = p.stem + suffix + p.suffix
        dst = dst_dir / new_name
        try:
            shutil.copy2(p, dst)
            count += 1
        except Exception as e:
            print(f"[ERR] label copy failed: {p} -> {dst} ({e})")
    print(f"[INFO] copied {count} test labels from {src_dir} to {dst_dir} with suffix '{suffix}'")
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLO labels and/or organize images for BDD100K (clear daytime/night) with COCO mapping.")
    parser.add_argument("--labels-only", action="store_true", help="Generate labels only (do not copy/link images)")
    parser.add_argument("--images-only", action="store_true", help="Copy/link images only (do not write labels)")
    parser.add_argument("--log-every", type=int, default=500, help="Progress log interval in files (0 to disable)")
    args = parser.parse_args()

    if args.labels_only and args.images_only:
        print("[ERR] --labels-only 와 --images-only 는 동시에 사용할 수 없습니다.")
        sys.exit(2)

    copy_images = not args.labels_only
    write_labels = not args.images_only

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # clear_daytime / clear_night 생성 (train/val/test 모두)
    for sp in ("train","val","test"):
        process_split(sp, copy_images=copy_images, write_labels=write_labels, log_every=args.log_every)

    # YAML 작성 (둘 다 test 포함)
    write_yaml("clear_daytime", include_test=True)
    write_yaml("clear_night", include_test=True)

    # 인조 밤/낮용(train/val/test) 빈 구조 + YAML (일부 러너 호환성 위해 모두 생성)
    for bname in ("clear_synth_night", "clear_synth_daytime"):
        for sp in ("train", "val", "test"):
            (OUT_ROOT / bname / "images" / sp).mkdir(parents=True, exist_ok=True)
            (OUT_ROOT / bname / "labels" / sp).mkdir(parents=True, exist_ok=True)
        write_yaml(bname, include_test=True)

    # 라벨 자동 복사: test 세트에만, 파일명에 _fake suffix 추가
    # clear_synth_daytime은 night를 바꾼 이미지이므로, clear_night의 test 라벨을 붙여넣기
    # clear_synth_night는 daytime을 바꾼 이미지이므로, clear_daytime의 test 라벨을 붙여넣기
    if write_labels:
        copy_test_labels_with_suffix(OUT_ROOT / "clear_night", OUT_ROOT / "clear_synth_daytime", suffix="_fake")
        copy_test_labels_with_suffix(OUT_ROOT / "clear_daytime", OUT_ROOT / "clear_synth_night", suffix="_fake")

    print(f"\n[OK] Wrote datasets under: {OUT_ROOT.resolve()}")
