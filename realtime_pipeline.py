# realtime_pipeline.py
# Project root: C:\Users\korea\Documents\GitHub\gen-clear-drive
import sys, argparse
from pathlib import Path
import torch
import cv2
import numpy as np

# --- paths ---
PROJ = Path(r"C:\Users\korea\Documents\GitHub\gen-clear-drive")
CYCLEGAN = PROJ / r"pytorch-CycleGAN-and-pix2pix"
CKPT_NAME = "clear_d2n_256_e200_k10k"  # checkpoints/<name>/latest_net_G_B.pth

# --- night gate (same as run.py heuristic) ---
def night_score(img_bgr: np.ndarray):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2]
    mean_v = float(V.mean())
    dark_ratio = float((V < 40).sum() / V.size)
    return mean_v, dark_ratio

def is_night(img_bgr, v_thresh=55.0, dark_ratio_thresh=0.35):
    mv, dr = night_score(img_bgr)
    return (mv < v_thresh) and (dr > dark_ratio_thresh), mv, dr

# --- CycleGAN G_B (B->A) loader: instance norm, no_dropout, resnet_9blocks ---
def load_gb(device="cuda:0"):
    sys.path.insert(0, str(CYCLEGAN))
    from models.networks import ResnetGenerator, get_norm_layer
    # netG config must match train: norm=instance, no_dropout=True, 9blocks
    netG = ResnetGenerator(
        input_nc=3, output_nc=3, ngf=64, norm_layer=get_norm_layer('instance'),
        use_dropout=False, n_blocks=9
    )
    ckpt = CYCLEGAN / "checkpoints" / CKPT_NAME / "latest_net_G_B.pth"
    state = torch.load(str(ckpt), map_location="cpu")
    netG.load_state_dict(state)
    netG = netG.to(device).eval()
    return netG

# preprocess: train-like(256 crop) for consistency, then back to original size
def gan_forward_b2a(netG, img_bgr, device="cuda:0", train_like=True):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if train_like:
        # resize 286, center-crop 256
        im = cv2.resize(img_rgb, (286, 286), interpolation=cv2.INTER_AREA)
        s = 15  # (286-256)//2
        im = im[s: s+256, s: s+256]
    else:
        # direct resize to 256 short-side then center crop
        scale = 256.0 / min(h, w)
        im = cv2.resize(img_rgb, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_AREA)
        ch, cw = im.shape[:2]
        y0 = max(0, (ch-256)//2); x0 = max(0, (cw-256)//2)
        im = im[y0:y0+256, x0:x0+256]

    # to [-1,1] tensor
    t = torch.from_numpy(im).float().permute(2,0,1) / 255.0
    t = (t - 0.5) / 0.5
    t = t.unsqueeze(0).to(device)

    with torch.no_grad():
        out = netG(t)

    # back to uint8
    out = out.squeeze(0).cpu()
    out = (out * 0.5 + 0.5).clamp(0,1)
    out = (out.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    # upsample back to original size for YOLO
    out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    return out_bgr

# --- YOLO ---
def load_yolo(model_path="yolo11s.pt", device="0", imgsz=1280):
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    # device: "0"/"cpu"; imgsz는 predict 시 지정 가능
    return model

def yolo_predict(model, img_bgr, imgsz=1280, device="0", conf=0.25):
    # Ultralytics는 ndarray 입력 가능
    res = model.predict(
        source=img_bgr, imgsz=imgsz, device=device, conf=conf, verbose=False
    )[0]
    dets = []
    if res.boxes is not None:
        names = model.names
        for b in res.boxes:
            cls = int(b.cls.item())
            dets.append({
                "cls": cls,
                "name": names.get(cls, str(cls)),
                "conf": float(b.conf.item()),
                "xyxy": [float(x) for x in b.xyxy.squeeze(0).tolist()],
            })
    return dets

# --- single image pipeline ---
class RealtimePipeline:
    def __init__(self, yolo_model="yolo11s.pt", device_yolo="0", imgsz=1280,
                 device_gan="cuda:0", train_like=True,
                 night_v_thresh=55.0, night_dark_ratio=0.35):
        self.yolo = load_yolo(yolo_model, device=device_yolo, imgsz=imgsz)
        self.device_yolo = device_yolo
        self.imgsz = imgsz
        self.gan = load_gb(device=device_gan)
        self.device_gan = device_gan
        self.train_like = train_like
        self.v_thresh = night_v_thresh
        self.dark_thresh = night_dark_ratio

    def process(self, image_path: Path, conf=0.25):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        isN, mv, dr = is_night(img, self.v_thresh, self.dark_thresh)
        used = "original"
        if isN:
            img = gan_forward_b2a(self.gan, img, device=self.device_gan, train_like=self.train_like)
            used = "fake_day(B->A)"

        dets = yolo_predict(self.yolo, img, imgsz=self.imgsz, device=self.device_yolo, conf=conf)
        return {
            "image": str(image_path),
            "night_gate": {"is_night": bool(isN), "mean_v": mv, "dark_ratio": dr, "used": used},
            "detections": dets
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="단일 이미지 경로")
    ap.add_argument("--yolo_model", type=str, default="yolo11s.pt")
    ap.add_argument("--device_yolo", type=str, default="0")
    ap.add_argument("--device_gan", type=str, default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--night_v_thresh", type=float, default=55.0)
    ap.add_argument("--night_dark_ratio", type=float, default=0.35)
    ap.add_argument("--train_like", action="store_true", help="CycleGAN 추론을 train과 동일(286->256 crop)로 수행")
    args = ap.parse_args()

    pipe = RealtimePipeline(
        yolo_model=args.yolo_model, device_yolo=args.device_yolo, imgsz=args.imgsz,
        device_gan=args.device_gan, train_like=args.train_like,
        night_v_thresh=args.night_v_thresh, night_dark_ratio=args.night_dark_ratio
    )
    out = pipe.process(Path(args.image), conf=args.conf)
    import json
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
