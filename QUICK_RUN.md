# 🚀 빠른 실행 가이드 (Quick Start)

## 📁 정리된 파일 구조

```
pytorch-CycleGAN-and-pix2pix/
├── TRAIN.bat                              ← CycleGAN+YOLO 학습 (V2.1)
├── train_clear_day2night_10k_ask_continue.bat  ← CycleGAN만 학습 (YOLO 없음)
├── models/
│   └── yolo_loss_v2.py                   ← YOLO Loss V2.1 (최신)
└── YOLO_LOSS_V2_FIX.md                   ← V2.1 수정 내역

루트/
├── run.py                                 ← 전체 평가 파이프라인
└── realtime_pipeline.py                   ← 실시간 단일 이미지 처리
```

## 🎯 사용 목적별 실행 방법

### 1️⃣ CycleGAN+YOLO 학습 (추천)

**목적**: 야간→주간 변환 + YOLO 탐지 성능 개선

```bash
# pytorch-CycleGAN-and-pix2pix 폴더에서 실행
TRAIN.bat
```

**설정 변경**:
- `TRAIN.bat` 파일 열어서 수정
```bat
REM 학습량 조절
set N_EPOCHS=100          # 초기 학습 에폭 (기본: 50)
set N_EPOCHS_DECAY=100    # Learning rate decay 에폭 (기본: 50)

REM YOLO Loss 가중치
set LAMBDA_TASK=1.0       # 1.0 = CycleGAN:YOLO = 1:1
                          # 0.5 = YOLO 약하게
                          # 2.0 = YOLO 강하게

REM 데이터셋 크기
set MAX_DATASET_SIZE=500  # 500 = 빠른 테스트
                          # inf = 전체 데이터
```

**모니터링**:
```bash
# Tensorboard 실행 (별도 터미널)
cd pytorch-CycleGAN-and-pix2pix
tensorboard --logdir runs --port 6006

# 브라우저에서 확인
# http://localhost:6006
```

**체크포인트 위치**:
```
pytorch-CycleGAN-and-pix2pix/checkpoints/clear_d2n_yolo_v2_lambda1/
├── latest_net_G_A.pth
├── latest_net_G_B.pth  ← 이게 야간→주간 생성기
├── latest_net_D_A.pth
└── latest_net_D_B.pth
```

---

### 2️⃣ 학습된 모델로 전체 평가

**목적**: 샘플링 → 변환 → YOLO 평가 → 보고서 생성

```bash
# 루트 폴더에서 실행
cd C:\Users\korea\Documents\GitHub\gen-clear-drive
..\venv\Scripts\activate
python run.py --n_day 100 --n_night 100 --device 0 --imgsz 1280 --yolo_model yolo11s.pt
```

**주요 옵션**:
```bash
--n_day 100           # 주간 이미지 샘플 개수
--n_night 100         # 야간 이미지 샘플 개수
--device 0            # GPU 번호 (0, 1, ...)
--imgsz 1280          # YOLO 이미지 크기
--yolo_model yolo11s.pt  # YOLO 모델 (yolo11n.pt, yolo11s.pt, yolo11m.pt)
```

**결과 확인**:
```
experiments/run_YYYYMMDD_HHMMSS/
├── report/
│   └── summary.json          ← 핵심 결과 (mAP, Precision, Recall)
├── yolo_results/
│   ├── subset_day/           ← 원본 주간 평가
│   ├── subset_night/         ← 원본 야간 평가
│   └── subset_fake_day_from_night/  ← 변환 후 평가 ⭐
└── outputs/
    └── fake_day_from_night/  ← 변환된 이미지들
```

---

### 3️⃣ 실시간 단일 이미지 처리

**목적**: 한 장의 이미지만 빠르게 변환+탐지

```bash
# 루트 폴더에서 실행
cd C:\Users\korea\Documents\GitHub\gen-clear-drive
..\venv\Scripts\activate
python realtime_pipeline.py --image path/to/your_night_image.jpg --yolo yolo11s.pt
```

**출력**:
- `{원본파일명}_processed.jpg`: 변환 + YOLO 박스 표시
- 콘솔에 탐지된 객체 정보 출력

---

## ⚙️ 학습 중 확인사항

### 로그 모니터링
학습 중 이런 로그가 나오는지 확인:

```
[YOLO Loss V2 Debug]
  Predictions: 500 total, 120 TP, 380 FP  ← TP 점차 증가
  GT: 150 objects, 95 matched (recall=0.633)  ← Recall 상승
  TP confidences: mean=0.45 ↑
  FP confidences: mean=0.20 ↓
  Loss Components:
    TP:   0.5500 (α=0.40 → weighted: 0.2200)
    FP:   0.0400 (β=0.10 → weighted: 0.0040)  ← 낮게 유지
    Rec:  0.3667 (γ=0.40 → weighted: 0.1467)
    Loc:  0.4800 (δ=0.10 → weighted: 0.0480)
  Total Loss: 0.4187
```

**좋은 신호** ✅:
- TP 개수 증가 추세
- TP confidence 상승 (0.3 → 0.6+)
- FP confidence 하락 (0.3 → 0.15-)
- Recall 상승 (0.2 → 0.5+)

**나쁜 신호** ❌:
- TP 개수 0에 가까움 (모델이 탐지 안 함)
- FP confidence > TP confidence (FP가 더 확신있음)
- Recall 정체 (0.1 이하)

### Tensorboard 확인
```
Scalars 탭:
- loss_G: CycleGAN loss (천천히 감소)
- loss_task: YOLO loss (초기 높다가 감소)
- loss_D_A, loss_D_B: Discriminator loss (0.5 근처 유지)

Images 탭:
- real_A / fake_B: 야간 원본 / 변환된 주간
- 변환 품질이 점차 개선되는지 확인
```

---

## 🐛 문제 해결

### 학습이 안 시작됨
```bash
# venv 확인
cd C:\Users\korea\Documents\GitHub\gen-clear-drive
..\venv\Scripts\activate  # 성공해야 함

# Python 버전 확인
python --version  # Python 3.8+

# 패키지 확인
pip list | findstr "torch ultralytics"
```

### "CUDA out of memory"
```bash
# TRAIN.bat에서 배치 크기 줄이기
set BATCH_SIZE=1  # 이미 1이면 이미지 크기 줄이기
set CROP_SIZE=128  # 256 → 128
```

### YOLO Loss가 너무 높음 (>0.8)
```bash
# LAMBDA_TASK 줄이기
set LAMBDA_TASK=0.5  # 1.0 → 0.5
```

### YOLO Loss가 너무 낮음 (<0.1)
```bash
# LAMBDA_TASK 늘리기
set LAMBDA_TASK=2.0  # 1.0 → 2.0
```

### 변환 품질은 좋은데 YOLO 성능이 안 오름
- 10 epoch 이후부터 효과 나타남
- 50+ epoch에서 평가 시작
- YOLO Loss 가중치 확인 (alpha, beta, gamma, delta)

---

## 📊 성공 기준

### 학습 완료 후 (Epoch 100+)
| Metric | Before (원본 야간) | Target (변환 후) | 비고 |
|--------|-------------------|-----------------|------|
| mAP50 | 0.307 | **0.25+** | 80% 유지 |
| Recall | 0.40~0.50 | **0.50+** | 탐지율 개선 |
| Precision | 0.30~0.40 | **0.40+** | 정밀도 개선 |

### run.py 평가 결과 확인
```bash
# experiments/run_*/report/summary.json 열기
{
  "subset_night": {
    "map50": 0.307,
    "recall": 0.45
  },
  "subset_fake_day_from_night": {
    "map50": 0.280,  ← 목표: 0.25 이상
    "recall": 0.52   ← 목표: 0.50 이상
  }
}
```

---

## 📝 파일 정리 완료

### 삭제된 파일 (구버전)
- ❌ `models/yolo_loss.py` (V1)
- ❌ `train_cyclegan_yolo.bat` (V1)
- ❌ `train_improved.bat` (YOLO 없음)
- ❌ `train_baseline_no_yolo.bat`
- ❌ `test_baseline_no_yolo.bat`
- ❌ `test_cyclegan_yolo.bat`
- ❌ `horse2zebra_test.bat`
- ❌ `evaluate_cyclegan_performance.bat`
- ❌ `test_loss_v2_integration.py`
- ❌ `YOLO_LOSS_V2_INTEGRATION.md`

### 유지된 파일 (최신)
- ✅ `TRAIN.bat` (V2.1, 메인 학습 스크립트)
- ✅ `train_clear_day2night_10k_ask_continue.bat` (YOLO 없는 순수 CycleGAN)
- ✅ `models/yolo_loss_v2.py` (V2.1, 수정된 loss)
- ✅ `YOLO_LOSS_V2_FIX.md` (수정 내역)
- ✅ `run.py` (평가 스크립트)
- ✅ `realtime_pipeline.py` (실시간 처리)

---

## 🔥 TL;DR (초간단)

```bash
# 1. 학습
cd pytorch-CycleGAN-and-pix2pix
TRAIN.bat

# 2. 평가 (학습 완료 후)
cd ..
python run.py --n_day 100 --n_night 100 --device 0

# 3. 결과 확인
notepad experiments\run_*\report\summary.json
```

끝! 🎉
