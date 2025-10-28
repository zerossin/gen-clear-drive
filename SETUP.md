# Setup Guide for gen-clear-drive

## 📦 Initial Setup Steps

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/gen-clear-drive.git
cd gen-clear-drive

# Clone CycleGAN repository (NOT a submodule, clone separately)
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Datasets

**⚠️ Important: Datasets are NOT included in this repository due to size (14+ GB)**

#### Option A: Download BDD100K (Original)

1. Visit [BDD100K Official Website](https://bdd-data.berkeley.edu/)
2. Register and download `bdd100k_images_100k` and `bdd100k_labels`
3. Convert to YOLO format:

```bash
python tools/bdd_time_weather_to_yolo.py \
  --bdd_images /path/to/bdd100k/images/100k \
  --bdd_labels /path/to/bdd100k/labels/bdd100k_labels_images_train.json \
  --output datasets/yolo_bdd100k
```

#### Option B: Use Shared Dataset (if available)

Contact the project maintainer for access to pre-processed datasets.

Expected structure after setup:
```
datasets/yolo_bdd100k/
  ├── clear_daytime/
  │   ├── images/test/*.jpg
  │   ├── labels/test/*.txt
  │   └── data.yaml
  └── clear_night/
      ├── images/test/*.jpg
      ├── labels/test/*.txt
      └── data.yaml
```

### 5. Download Pre-trained Models

#### YOLO11 Models (Auto-download)

YOLO models will be automatically downloaded on first use. Or download manually:

```bash
# yolo11n.pt (5.35 MB) - Nano version
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# yolo11s.pt (18.42 MB) - Small version (recommended)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

#### CycleGAN Checkpoints

**Option A: Train from scratch**

```bash
cd pytorch-CycleGAN-and-pix2pix

# Prepare training data
python build_clear_day2night.py

# Train (takes hours/days on GPU)
python train.py \
  --dataroot ./datasets/clear_day2night \
  --name clear_d2n_256_e200_k10k \
  --model cycle_gan \
  --netG resnet_9blocks \
  --norm instance \
  --no_dropout \
  --n_epochs 100 \
  --n_epochs_decay 100
```

**Option B: Use pre-trained checkpoint (if shared)**

Place checkpoint files in:
```
pytorch-CycleGAN-and-pix2pix/checkpoints/clear_d2n_256_e200_k10k/
  ├── latest_net_G_B.pth
  └── latest_net_D_B.pth
```

### 6. Verify Setup

```bash
# Test with small sample
python run.py --n_day 10 --n_night 10 --yolo_model yolo11n.pt --device cpu

# Check if experiment folder is created
ls experiments/
```

## 🔧 Configuration

### For Different Environments

If you have datasets in a different location, you can:

1. **Symlink** (Linux/Mac):
```bash
ln -s /path/to/your/datasets datasets
```

2. **Junction** (Windows):
```powershell
New-Item -ItemType Junction -Path "datasets" -Target "D:\path\to\datasets"
```

3. **Edit paths** in `run.py` (lines 13-23) - but keep `PROJ` as relative path!

## 📊 Storage Requirements

| Component | Size | Can Delete After Setup? |
|-----------|------|------------------------|
| Datasets | 14.47 GB | ❌ No (required for experiments) |
| CycleGAN checkpoints | 3.24 GB | ❌ No (required for inference) |
| YOLO models | 23.77 MB | ❌ No (required for detection) |
| Experiment results | 0.73 GB | ✅ Yes (regenerable) |
| Virtual environment | 7.66 GB | ⚠️ Optional (recreate with pip) |

**Total minimum required: ~18 GB**

## 🐛 Common Issues

### Issue: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: CUDA out of memory

**Solution:**
```bash
# Use smaller model
python run.py --yolo_model yolo11n.pt --imgsz 640

# Or use CPU
python run.py --device cpu
```

### Issue: `FileNotFoundError: datasets/yolo_bdd100k/...`

**Solution:** Complete Step 4 (Download Datasets) above.

## 🚀 Next Steps

Once setup is complete, proceed to:
- [Main README](README.md) for usage examples
- [.github/copilot-instructions.md](.github/copilot-instructions.md) for detailed architecture

## 💡 Tips

1. **Use Git LFS for large files** (if you have checkpoints to share):
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
```

2. **Create placeholder folders** for clean structure:
```bash
mkdir -p datasets/yolo_bdd100k
mkdir -p experiments
mkdir -p pytorch-CycleGAN-and-pix2pix/checkpoints
```

3. **Document your environment** for reproducibility:
```bash
pip freeze > requirements-frozen.txt
```
