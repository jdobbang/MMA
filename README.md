# MMA Player Detection and Tracking

YOLO 기반 MMA 선수 검출 및 ByteTrack 추적 시스템

## 프로젝트 구조

```
MMA/
├── preprocessor/          # 데이터 전처리 스크립트
│   ├── generate_yolo_dataset.py      # YOLO 데이터셋 생성 (player + crowd)
│   ├── split_yolo_dataset.py         # Train/Val 분할
│   └── cleanup_folders.py
├── script/               # 학습 및 추론 스크립트
│   ├── train_yolo.sh                 # YOLO 학습 스크립트
│   ├── visualize_training.py         # 학습 진행 시각화
│   ├── inference_base.py             # 검출 추론 (confidence 기반)
│   ├── inference_base_bytetrack.py   # 추적 추론 (ByteTrack)
│   ├── detection.py                  # 이미지/비디오 검출 스크립트
│   └── concat_results.py             # 결과 비교 시각화
├── analyzer/             # 분석 도구
└── model/               # 모델 파일 (gitignore)
```

## 주요 기능

### 1. 데이터 전처리
- Harmony4D 데이터셋에서 YOLO 형식 어노테이션 생성
- Player 검출 + Crowd 검출 (Dice coefficient 기반 필터링)
- 80/20 Train/Val 분할

### 2. 모델 학습
- YOLOv11x 기반 2-class 검출 (player, crowd)
- 150 epochs, AdamW optimizer
- Data augmentation: mosaic, mixup, copy-paste
- TensorBoard 시각화 지원

### 3. 추론 및 추적
- **Confidence 기반 검출**: Low/Mid/High 3단계 색상 구분
- **ByteTrack 추적**: 프레임 간 ID 일관성 유지
- 최대 2명 선수 추적 지원

## 사용법

### 데이터셋 준비
```bash
# YOLO 어노테이션 생성
python preprocessor/generate_yolo_dataset.py --root_dir /path/to/dataset

# Train/Val 분할
python preprocessor/split_yolo_dataset.py --source_dir /path/to/dataset
```

### 학습
```bash
cd script
./train_yolo.sh

# 학습 진행 시각화 (백그라운드)
nohup python visualize_training.py > /dev/null 2>&1 &
```

### 추론
```bash
# 검출만
python script/inference_base.py \
  --model log/mma_training/weights/best.pt \
  --source /path/to/images \
  --conf 0.25

# 추적 포함
python script/inference_base_bytetrack.py \
  --model log/mma_training/weights/best.pt \
  --source /path/to/images \
  --conf 0.1 \
  --track-thresh 0.5 \
  --max-players 2

# 결과 비교
python script/concat_results.py --base-dir log/mma_training
```

### 이미지/비디오 검출
```bash
python script/detection.py log/mma_training/weights/best.pt \
  --mode images \
  --input /path/to/images \
  --output detection/results.csv

python script/detection.py log/mma_training/weights/best.pt \
  --mode video \
  --input /path/to/video.mp4 \
  --interval 5
```

## 환경 설정

```bash
# Conda 환경
conda create -n mma python=3.10
conda activate mma

# 패키지 설치
pip install ultralytics opencv-python numpy tqdm pandas matplotlib
```

## 결과

- **mAP@0.5**: ~0.94
- **mAP@0.5:0.95**: ~0.73
- **Precision**: ~0.96
- **Recall**: ~0.87

## 참고

- Dataset: Harmony4D
- Model: YOLOv11x
- Tracker: ByteTrack

---

## script/detection.py

YOLO 기반 이미지/비디오 선수 검출 스크립트

- **기능:**
    - 이미지 폴더 또는 비디오 파일에서 선수(클래스 0) 객체 검출
    - 결과를 CSV 파일로 저장
    - 이미지/비디오 모두 지원 (mode 선택)
    - tqdm 진행바 및 오류/재시작 지원

- **입력:**
    - 모델 파일명 (예: yolo11n.pt, yolo11x.pt)
    - --mode: 'images' 또는 'video' (기본값: images)
    - --input: 입력 이미지 폴더 또는 비디오 파일 경로
    - --output: 결과 CSV 파일 경로 (images 모드)
    - --interval: 비디오 모드에서 N프레임마다 검출 (기본 1)

- **출력:**
    - 이미지 모드: [output]에 각 이미지별 검출 결과 CSV 저장
    - 비디오 모드: [results/모델명/비디오명.csv]로 프레임별 검출 결과 저장

- **실행 예시:**

이미지 폴더 검출:
```bash
python script/detection.py log/mma_training_v1_202512052/weights/best.pt \
  --mode images \
  --input dataset/yolodataset/images/val \
  --output detection/test.csv
```

비디오 파일 검출:
```bash
python script/detection.py log/mma_training_v1_202512052/weights/best.pt \
  --mode video \
  --input /path/to/video.mp4 \
  --interval 5
```

- **출력 CSV 포맷:**
    - 이미지: image_name, object_id, x1, y1, x2, y2, confidence, width, height
    - 비디오: frame, object_id, x1, y1, x2, y2, confidence, width, height

---
