"""
Dataset splitting utilities for train/val/test partitioning

Provides:
- YOLO dataset splitting by files
- Detection + pose dataset co-splitting
- Stratified splitting by class or sequence
- Statistical validation

Replaces:
- split_yolo_dataset.py: Dataset splitting
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def split_yolo_dataset(
    dataset_dir: str,
    ratios: Dict[str, float] = None,
    seed: int = 42,
    stratify_by: Optional[str] = None
) -> None:
    """
    Split YOLO dataset into train/val/test

    Args:
        dataset_dir: YOLO dataset root directory
        ratios: {'train': 0.7, 'val': 0.15, 'test': 0.15}
        seed: Random seed for reproducibility
        stratify_by: Split strategy ('sequence', 'class', None for random)

    Example:
        >>> split_yolo_dataset('yolo_dataset/',
        ...                   ratios={'train': 0.8, 'val': 0.2})
    """
    if ratios is None:
        ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

    random.seed(seed)

    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Dataset structure invalid: {dataset_dir}")

    # Get all image files
    image_files = sorted([f for f in images_dir.glob('*')
                         if f.suffix in ['.jpg', '.jpeg', '.png']])

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    # Create split directories
    for split in ratios.keys():
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Stratify if needed
    if stratify_by == 'sequence':
        _split_by_sequence(image_files, dataset_path, labels_dir, ratios)
    elif stratify_by == 'class':
        _split_by_class(image_files, dataset_path, labels_dir, ratios)
    else:
        _random_split(image_files, dataset_path, labels_dir, ratios)

    # Print statistics
    _print_split_stats(dataset_path, ratios)


def split_detection_pose_dataset(
    detection_dir: str,
    pose_dir: str,
    output_dir: str,
    ratios: Dict[str, float] = None,
    seed: int = 42
) -> None:
    """
    Split detection and pose datasets together (co-split)

    Ensures same images appear in same split for both datasets

    Args:
        detection_dir: Detection dataset directory
        pose_dir: Pose dataset directory
        output_dir: Output directory with splits
        ratios: Split ratios
        seed: Random seed

    Example:
        >>> split_detection_pose_dataset(
        ...     'detection_dataset/',
        ...     'pose_dataset/',
        ...     'combined_dataset/',
        ...     ratios={'train': 0.8, 'val': 0.2}
        ... )
    """
    if ratios is None:
        ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

    random.seed(seed)

    det_path = Path(detection_dir)
    pose_path = Path(pose_dir)
    out_path = Path(output_dir)

    # Get common image files
    det_images = set(f.stem for f in (det_path / 'images').glob('*'))
    pose_images = set(f.stem for f in (pose_path / 'images').glob('*'))
    common_images = sorted(det_images & pose_images)

    if not common_images:
        raise ValueError("No common images between detection and pose datasets")

    # Create output structure
    for split in ratios.keys():
        (out_path / split / 'detection' / 'images').mkdir(parents=True, exist_ok=True)
        (out_path / split / 'detection' / 'labels').mkdir(parents=True, exist_ok=True)
        (out_path / split / 'pose' / 'images').mkdir(parents=True, exist_ok=True)
        (out_path / split / 'pose' / 'labels').mkdir(parents=True, exist_ok=True)

    # Random split
    num_total = len(common_images)
    num_train = int(num_total * ratios.get('train', 0.7))
    num_val = int(num_total * ratios.get('val', 0.15))

    indices = list(range(num_total))
    random.shuffle(indices)

    train_indices = set(indices[:num_train])
    val_indices = set(indices[num_train:num_train + num_val])
    test_indices = set(indices[num_train + num_val:])

    # Copy files with split assignment
    for idx, image_stem in enumerate(common_images):
        if idx in train_indices:
            split = 'train'
        elif idx in val_indices:
            split = 'val'
        else:
            split = 'test'

        # Copy detection files
        _copy_dataset_files(
            det_path, out_path / split / 'detection',
            image_stem
        )

        # Copy pose files
        _copy_dataset_files(
            pose_path, out_path / split / 'pose',
            image_stem
        )

    print(f"✓ Co-split dataset created: {out_path}")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")


def _random_split(
    image_files: List[Path],
    dataset_path: Path,
    labels_dir: Path,
    ratios: Dict[str, float]
) -> None:
    """Random split of images"""
    random.shuffle(image_files)

    num_total = len(image_files)
    split_keys = list(ratios.keys())

    for idx, img_path in enumerate(image_files):
        # Determine split
        cumsum = 0
        for split_key in split_keys:
            cumsum += ratios[split_key]
            if idx / num_total < cumsum:
                split = split_key
                break

        # Copy image
        src_img = img_path
        dst_img = dataset_path / split / 'images' / img_path.name
        shutil.copy2(src_img, dst_img)

        # Copy corresponding label
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            dst_label = dataset_path / split / 'labels' / label_path.name
            shutil.copy2(label_path, dst_label)


def _split_by_sequence(
    image_files: List[Path],
    dataset_path: Path,
    labels_dir: Path,
    ratios: Dict[str, float]
) -> None:
    """Split by sequence (group consecutive frames)"""
    # Group images by sequence name (e.g., seq001_frame_001.jpg)
    sequences = defaultdict(list)
    for img_path in image_files:
        seq_name = img_path.stem.rsplit('_', 2)[0]  # Remove _frame_xxx
        sequences[seq_name].append(img_path)

    # Split sequences
    seq_names = sorted(sequences.keys())
    random.shuffle(seq_names)

    num_total = len(seq_names)
    split_keys = list(ratios.keys())

    for idx, seq_name in enumerate(seq_names):
        # Determine split
        cumsum = 0
        for split_key in split_keys:
            cumsum += ratios[split_key]
            if idx / num_total < cumsum:
                split = split_key
                break

        # Copy all frames of this sequence to split
        for img_path in sequences[seq_name]:
            dst_img = dataset_path / split / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)

            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                dst_label = dataset_path / split / 'labels' / label_path.name
                shutil.copy2(label_path, dst_label)


def _split_by_class(
    image_files: List[Path],
    dataset_path: Path,
    labels_dir: Path,
    ratios: Dict[str, float]
) -> None:
    """Stratified split by class distribution"""
    # Group by dominant class in label
    class_groups = defaultdict(list)

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            with open(label_path) as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = first_line.split()[0]
                    class_groups[class_id].append(img_path)
        else:
            class_groups['unknown'].append(img_path)

    # Split each class proportionally
    split_keys = list(ratios.keys())

    for class_id, class_images in class_groups.items():
        random.shuffle(class_images)
        num_total = len(class_images)

        for idx, img_path in enumerate(class_images):
            cumsum = 0
            for split_key in split_keys:
                cumsum += ratios[split_key]
                if idx / num_total < cumsum:
                    split = split_key
                    break

            dst_img = dataset_path / split / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)

            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                dst_label = dataset_path / split / 'labels' / label_path.name
                shutil.copy2(label_path, dst_label)


def _copy_dataset_files(src_path: Path, dst_path: Path, image_stem: str) -> None:
    """Copy image and label files"""
    # Find and copy image
    for img_ext in ['.jpg', '.jpeg', '.png']:
        src_img = src_path / 'images' / (image_stem + img_ext)
        if src_img.exists():
            dst_img = dst_path / 'images' / src_img.name
            shutil.copy2(src_img, dst_img)
            break

    # Copy label
    src_label = src_path / 'labels' / (image_stem + '.txt')
    if src_label.exists():
        dst_label = dst_path / 'labels' / src_label.name
        shutil.copy2(src_label, dst_label)


def _print_split_stats(dataset_path: Path, ratios: Dict[str, float]) -> None:
    """Print split statistics"""
    print(f"✓ Dataset split complete")

    total_images = 0
    for split in ratios.keys():
        split_path = dataset_path / split / 'images'
        num_images = len(list(split_path.glob('*')))
        total_images += num_images
        print(f"  {split}: {num_images} images")

    print(f"  Total: {total_images} images")
