#!/bin/bash

# YOLO Training Script with TensorBoard Logging
# Created: December 5, 2025

echo "Starting YOLO training..."
echo "================================"

# Training parameters
DATA_YAML="/workspace/dataset/yolodataset/data.yaml"
MODEL="yolo11x.pt"
PROJECT_DIR="/workspace/log"
EXPERIMENT_NAME="mma_training_v1_20251205"

# Check if data.yaml exists
if [ ! -f "$DATA_YAML" ]; then
    echo "Error: data.yaml not found at $DATA_YAML"
    echo "Please run split_yolo_dataset.py first."
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$PROJECT_DIR"

# Start training
yolo detect train \
  data="$DATA_YAML" \
  model="$MODEL" \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  project="$PROJECT_DIR" \
  name="$EXPERIMENT_NAME" \
  patience=30 \
  save_period=10 \
  optimizer=AdamW \
  lr0=0.001 \
  lrf=0.01 \
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=5.0 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.1 \
  box=7.5 \
  cls=1.0 \
  dfl=1.5 \
  hsv_h=0.02 \
  hsv_s=0.8 \
  hsv_v=0.5 \
  degrees=5.0 \
  translate=0.15 \
  scale=0.7 \
  shear=2.0 \
  perspective=0.0001 \
  flipud=0.0 \
  fliplr=0.5 \
  mosaic=1.0 \
  mixup=0.1 \
  copy_paste=0.1 \
  device=0 \
  workers=8 \
  seed=0 \
  deterministic=True \
  single_cls=False \
  rect=False \
  cos_lr=False \
  close_mosaic=10 \
  amp=True \
  fraction=1.0 \
  profile=False \
  overlap_mask=True \
  mask_ratio=4 \
  dropout=0.0 \
  val=True \
  plots=True \
  save=True \
  save_json=False \
  verbose=True

echo "================================"
echo "Training completed!"
echo "Results saved to: $PROJECT_DIR/$EXPERIMENT_NAME"
echo ""
echo "To view TensorBoard logs, run:"
echo "tensorboard --logdir=$PROJECT_DIR --host=0.0.0.0 --port=6006"
