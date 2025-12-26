#!/bin/bash

# YOLO Pose Dataset Preparation Pipeline
# =======================================
# 1. Generate YOLO pose annotations from poses2d
# 2. Split dataset into train/val

set -e  # Exit on error

echo "=========================================="
echo "YOLO Pose Dataset Preparation Pipeline"
echo "=========================================="

# Configuration
DATASET_DIR="/workspace/MMA/dataset"
OUTPUT_DIR="/workspace/dataset/yolo_pose_dataset"
CAMERA="cam01"
TRAIN_RATIO=0.8

# Datasets to process
DATASETS=("03_grappling2" "13_mma2")

# Step 1: Generate YOLO pose annotations
echo ""
echo "Step 1: Generating YOLO pose annotations from poses2d..."
echo "----------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Processing dataset: $dataset"
    python /workspace/MMA/preprocessor/generate_yolo_pose_dataset.py \
        --root_dir "$DATASET_DIR/$dataset" \
        --camera "$CAMERA"
done

# Step 2: Split dataset
echo ""
echo "Step 2: Splitting dataset into train/val..."
echo "----------------------------------------------------------"

python /workspace/MMA/preprocessor/split_yolo_pose_dataset.py \
    --source_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio "$TRAIN_RATIO" \
    --cameras "$CAMERA" \
    --datasets "${DATASETS[@]}"

echo ""
echo "=========================================="
echo "Dataset preparation complete!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Configuration file: $OUTPUT_DIR/data.yaml"
echo ""
echo "Next steps:"
echo "1. Verify dataset: ls $OUTPUT_DIR"
echo "2. Train YOLO pose model:"
echo "   yolo pose train data=$OUTPUT_DIR/data.yaml model=yolo11x-pose.pt epochs=100 imgsz=640"
echo "=========================================="
