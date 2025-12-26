#!/usr/bin/env python3
"""
Split YOLO Pose Dataset into Train/Val
=======================================

YOLO pose 데이터셋을 train/val로 분할
- 시퀀스 기반 분할 (data leakage 방지)
- 각 데이터셋별로 독립적으로 8:2 분할
"""

import os
import shutil
import glob
import random
from pathlib import Path
from tqdm import tqdm
import argparse


def split_by_sequences(source_dir='/workspace/MMA/dataset',
                       output_dir='/workspace/dataset/yolo_pose_dataset',
                       train_ratio=0.8, seed=42, use_symlink=False,
                       cameras=None, datasets=None):
    """
    Split YOLO pose dataset by sequences

    Args:
        source_dir: root directory containing dataset folders
        output_dir: output directory for YOLO dataset
        train_ratio: train/val split ratio
        seed: random seed
        use_symlink: use symbolic links instead of copying
        cameras: list of cameras to process (default: ['cam01'])
        datasets: list of datasets to process (e.g., ['03_grappling2', '13_mma2'])
    """
    random.seed(seed)

    # Default cameras if not specified
    if cameras is None:
        cameras = ['cam01']

    # Default datasets if not specified
    if datasets is None:
        datasets = ['03_grappling2', '13_mma2']

    print(f"Processing cameras: {cameras}")
    print(f"Processing datasets: {datasets}")

    # Create directories
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # Collect and split sequences per dataset
    train_sequences = []
    val_sequences = []

    for dataset in datasets:
        dataset_path = os.path.join(source_dir, dataset)

        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset not found - {dataset_path}")
            continue

        sequences = sorted([d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d))])

        # Filter sequences that have pose data
        valid_sequences = []
        for seq in sequences:
            pose_dir = os.path.join(dataset_path, seq, 'processed_data', 'bbox_pose')
            if os.path.exists(pose_dir):
                valid_sequences.append(seq)

        if not valid_sequences:
            print(f"Warning: No valid sequences found in {dataset}")
            continue

        # Shuffle and split within each dataset
        random.shuffle(valid_sequences)
        split_idx = int(len(valid_sequences) * train_ratio)

        dataset_train = [(dataset, seq) for seq in valid_sequences[:split_idx]]
        dataset_val = [(dataset, seq) for seq in valid_sequences[split_idx:]]

        train_sequences.extend(dataset_train)
        val_sequences.extend(dataset_val)

        print(f"{dataset}: {len(valid_sequences)} sequences (train: {len(dataset_train)}, val: {len(dataset_val)})")

    print(f"\nTotal sequences: {len(train_sequences) + len(val_sequences)}")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    print(f"\nMode: {'Symbolic links' if use_symlink else 'Copy files'}")

    # Process train sequences
    print("\nProcessing train sequences...")
    train_count = 0
    for dataset, seq in tqdm(train_sequences, desc="Train"):
        count = copy_sequence_data(source_dir, dataset, seq, output_dir, 'train', use_symlink, cameras)
        train_count += count

    # Process val sequences
    print("Processing val sequences...")
    val_count = 0
    for dataset, seq in tqdm(val_sequences, desc="Val"):
        count = copy_sequence_data(source_dir, dataset, seq, output_dir, 'val', use_symlink, cameras)
        val_count += count

    # Create data.yaml
    yaml_content = f"""# YOLO Pose Dataset Configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Keypoints
kpt_shape: [17, 3]  # 17 keypoints, 3 values per keypoint (x, y, visibility)

# Classes
names:
  0: person

# Number of classes
nc: 1

# Keypoint names (COCO format)
keypoint_names:
  0: nose
  1: left_eye
  2: right_eye
  3: left_ear
  4: right_ear
  5: left_shoulder
  6: right_shoulder
  7: left_elbow
  8: right_elbow
  9: left_wrist
  10: right_wrist
  11: left_hip
  12: right_hip
  13: left_knee
  14: right_knee
  15: left_ankle
  16: right_ankle

# Keypoint edges for visualization (skeleton)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
"""

    with open(f"{output_dir}/data.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"\n{'='*80}")
    print(f"Dataset created!")
    print(f"Train sequences: {len(train_sequences)} ({train_count} files)")
    print(f"Val sequences: {len(val_sequences)} ({val_count} files)")
    print(f"Output: {output_dir}")
    print(f"Config: {output_dir}/data.yaml")
    print(f"{'='*80}")


def copy_sequence_data(source_dir, dataset, seq, output_dir, split, use_symlink=False, cameras=None):
    """
    Copy or symlink all images and labels from one sequence to train or val

    Args:
        source_dir: root directory containing dataset folders
        dataset: dataset name (e.g., '03_grappling2')
        seq: sequence name (e.g., '001_grappling2')
        output_dir: output directory
        split: 'train' or 'val'
        use_symlink: use symbolic links instead of copying
        cameras: list of cameras to process

    Returns:
        int: number of files processed
    """
    # Default cameras if not specified
    if cameras is None:
        cameras = ['cam01']

    seq_path = os.path.join(source_dir, dataset, seq)
    pose_dir = os.path.join(seq_path, 'processed_data', 'bbox_pose')
    exo_dir = os.path.join(seq_path, 'exo')

    if not os.path.exists(pose_dir):
        return 0

    file_count = 0

    # Process only specified cameras
    for cam in cameras:
        label_dir = os.path.join(pose_dir, cam)
        img_dir = os.path.join(exo_dir, cam, 'images')

        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            continue

        # Process all files
        for label_file in glob.glob(f"{label_dir}/*.txt"):
            frame = os.path.basename(label_file).replace('.txt', '')
            img_file = f"{img_dir}/{frame}.jpg"

            if os.path.exists(img_file):
                unique_name = f"{dataset}_{seq}_{cam}_{frame}"
                img_dst = f"{output_dir}/images/{split}/{unique_name}.jpg"
                label_dst = f"{output_dir}/labels/{split}/{unique_name}.txt"

                # Use symlink or copy
                if use_symlink:
                    # Create absolute path symlinks
                    os.symlink(os.path.abspath(img_file), img_dst)
                    os.symlink(os.path.abspath(label_file), label_dst)
                else:
                    shutil.copy(img_file, img_dst)
                    shutil.copy(label_file, label_dst)

                file_count += 1

    return file_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split YOLO pose dataset by sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split with default settings
  python split_yolo_pose_dataset.py --source_dir /workspace/MMA/dataset \\
                                    --output_dir /workspace/dataset/yolo_pose_dataset

  # Use symbolic links (faster, saves disk space)
  python split_yolo_pose_dataset.py --source_dir /workspace/MMA/dataset \\
                                    --output_dir /workspace/dataset/yolo_pose_dataset \\
                                    --use_symlink

  # Custom split ratio and specific datasets
  python split_yolo_pose_dataset.py --source_dir /workspace/MMA/dataset \\
                                    --output_dir /workspace/dataset/yolo_pose_dataset \\
                                    --train_ratio 0.9 \\
                                    --datasets 03_grappling2
        """
    )

    parser.add_argument('--source_dir', default='/workspace/MMA/dataset',
                        help='Source directory containing dataset folders (default: /workspace/MMA/dataset)')
    parser.add_argument('--output_dir', default='/workspace/dataset/yolo_pose_dataset',
                        help='Output directory for YOLO pose dataset (default: /workspace/dataset/yolo_pose_dataset)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--use_symlink', action='store_true',
                        help='Use symbolic links instead of copying files')
    parser.add_argument('--cameras', nargs='+', default=['cam01'],
                        help='Cameras to process (default: cam01)')
    parser.add_argument('--datasets', nargs='+', default=['03_grappling2', '13_mma2'],
                        help='Datasets to process (default: 03_grappling2 13_mma2)')

    args = parser.parse_args()

    split_by_sequences(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        use_symlink=args.use_symlink,
        cameras=args.cameras,
        datasets=args.datasets
    )
