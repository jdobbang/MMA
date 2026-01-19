"""
Dataset generation utilities for YOLO models

Provides:
- YOLO detection dataset generation from annotations
- YOLO pose dataset generation from keypoint data
- Image organization and label conversion
- Batch processing

Replaces:
- generate_yolo_dataset.py: YOLO detection dataset creation
- generate_yolo_pose_dataset.py: YOLO pose dataset creation
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.constants import COCO_KEYPOINT_NAMES
from ..core.exceptions import DataLoadError, PathError


def create_yolo_detection_dataset(
    source_images_dir: str,
    annotations: List[Dict],
    output_dir: str,
    split_ratios: Dict[str, float] = None,
    symlink: bool = False
) -> None:
    """
    Create YOLO detection dataset from annotations

    Args:
        source_images_dir: Directory containing source images
        annotations: List of annotation dicts with keys:
            - image_path: Path to image
            - objects: List of {class, x, y, w, h} normalized coordinates
        output_dir: Output dataset root directory
        split_ratios: {'train': 0.7, 'val': 0.15, 'test': 0.15}
        symlink: Use symlinks instead of copying images

    Example:
        >>> annotations = [
        ...     {
        ...         'image_path': 'frame_001.jpg',
        ...         'objects': [{'class': 'player', 'x': 0.5, 'y': 0.5, 'w': 0.2, 'h': 0.4}]
        ...     }
        ... ]
        >>> create_yolo_detection_dataset('images/', annotations, 'yolo_dataset/')
    """
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

    # Create directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in split_ratios.keys():
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Create data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images' if 'test' in split_ratios else None,
        'nc': 1,
        'names': ['player']
    }

    yaml_path = output_path / 'data.yaml'
    _write_yaml(yaml_path, data_yaml)

    # Distribute annotations to splits
    num_samples = len(annotations)
    split_idx = 0
    split_keys = list(split_ratios.keys())

    for idx, ann in enumerate(annotations):
        # Determine split
        cumsum = 0
        for split_key in split_keys:
            cumsum += split_ratios[split_key]
            if idx / num_samples < cumsum:
                split = split_key
                break

        # Copy/link image
        src_img = Path(source_images_dir) / ann['image_path']
        dst_img = output_path / split / 'images' / src_img.name

        if src_img.exists():
            if symlink:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(src_img.absolute())
            else:
                shutil.copy2(src_img, dst_img)

        # Write label file
        label_path = output_path / split / 'labels' / src_img.stem + '.txt'
        _write_yolo_labels(label_path, ann.get('objects', []))

    print(f"✓ YOLO detection dataset created: {output_path}")


def create_yolo_pose_dataset(
    source_images_dir: str,
    annotations: List[Dict],
    output_dir: str,
    split_ratios: Dict[str, float] = None,
    symlink: bool = False
) -> None:
    """
    Create YOLO pose dataset from keypoint annotations

    Args:
        source_images_dir: Directory containing source images
        annotations: List of annotation dicts with keys:
            - image_path: Path to image
            - keypoints: List of {class, keypoints: [(x,y,conf), ...]}
        output_dir: Output dataset root directory
        split_ratios: Split ratios for train/val/test
        symlink: Use symlinks instead of copying

    Example:
        >>> annotations = [
        ...     {
        ...         'image_path': 'frame_001.jpg',
        ...         'keypoints': [
        ...             {'x': 0.5, 'y': 0.5, 'conf': 0.9, ...}  # 17 COCO keypoints
        ...         ]
        ...     }
        ... ]
        >>> create_yolo_pose_dataset('images/', annotations, 'yolo_pose_dataset/')
    """
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

    # Create directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in split_ratios.keys():
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Create data.yaml for pose
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images' if 'test' in split_ratios else None,
        'nc': 1,
        'names': ['person'],
        'kpt_shape': [len(COCO_KEYPOINT_NAMES), 3],  # 17 keypoints, (x, y, conf)
    }

    yaml_path = output_path / 'data.yaml'
    _write_yaml(yaml_path, data_yaml)

    # Distribute annotations
    num_samples = len(annotations)
    split_keys = list(split_ratios.keys())

    for idx, ann in enumerate(annotations):
        # Determine split
        cumsum = 0
        for split_key in split_keys:
            cumsum += split_ratios[split_key]
            if idx / num_samples < cumsum:
                split = split_key
                break

        # Copy/link image
        src_img = Path(source_images_dir) / ann['image_path']
        dst_img = output_path / split / 'images' / src_img.name

        if src_img.exists():
            if symlink:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(src_img.absolute())
            else:
                shutil.copy2(src_img, dst_img)

        # Write pose labels (YOLO OBB format)
        label_path = output_path / split / 'labels' / src_img.stem + '.txt'
        _write_yolo_pose_labels(label_path, ann.get('objects', []))

    print(f"✓ YOLO pose dataset created: {output_path}")


def _write_yolo_labels(label_path: Path, objects: List[Dict]) -> None:
    """
    Write YOLO detection labels

    Format: <class_id> <x_center> <y_center> <width> <height>
    (normalized coordinates)
    """
    with open(label_path, 'w') as f:
        for obj in objects:
            class_id = obj.get('class_id', 0)
            x = obj.get('x', 0)
            y = obj.get('y', 0)
            w = obj.get('w', 0)
            h = obj.get('h', 0)
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def _write_yolo_pose_labels(label_path: Path, objects: List[Dict]) -> None:
    """
    Write YOLO pose labels

    Format: <class_id> <x_center> <y_center> <width> <height> <kpt_x1> <kpt_y1> <kpt_conf1> ... <kpt_x17> <kpt_y17> <kpt_conf17>
    (normalized coordinates, 17 COCO keypoints)
    """
    with open(label_path, 'w') as f:
        for obj in objects:
            class_id = obj.get('class_id', 0)
            x = obj.get('x', 0)
            y = obj.get('y', 0)
            w = obj.get('w', 0)
            h = obj.get('h', 0)

            # Keypoints (17 COCO points)
            keypoints = obj.get('keypoints', [])
            kpt_str = ' '.join(
                [f"{kpt.get('x', 0):.6f} {kpt.get('y', 0):.6f} {kpt.get('conf', 0):.6f}"
                 for kpt in keypoints[:17]]
            )

            # Pad missing keypoints with 0 0 0
            num_kpts = len(keypoints)
            if num_kpts < 17:
                kpt_str += ' ' + ' '.join(['0 0 0'] * (17 - num_kpts))

            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {kpt_str}\n")


def _write_yaml(yaml_path: Path, data: Dict) -> None:
    """Write YAML file"""
    try:
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    except ImportError:
        # Fallback: write as simple key: value format
        with open(yaml_path, 'w') as f:
            for key, value in data.items():
                if value is not None:
                    f.write(f"{key}: {value}\n")


def validate_dataset_structure(dataset_dir: str) -> bool:
    """
    Validate YOLO dataset directory structure

    Args:
        dataset_dir: Dataset root directory

    Returns:
        True if valid structure

    Example:
        >>> is_valid = validate_dataset_structure('yolo_dataset/')
    """
    dataset_path = Path(dataset_dir)

    # Check required files
    required_files = [
        dataset_path / 'data.yaml',
        dataset_path / 'train' / 'images',
        dataset_path / 'train' / 'labels',
        dataset_path / 'val' / 'images',
        dataset_path / 'val' / 'labels',
    ]

    for file_path in required_files:
        if not file_path.exists():
            print(f"✗ Missing: {file_path}")
            return False

    # Count samples
    train_images = len(list((dataset_path / 'train' / 'images').glob('*')))
    val_images = len(list((dataset_path / 'val' / 'images').glob('*')))
    train_labels = len(list((dataset_path / 'train' / 'labels').glob('*.txt')))
    val_labels = len(list((dataset_path / 'val' / 'labels').glob('*.txt')))

    print(f"✓ Dataset structure valid")
    print(f"  Train: {train_images} images, {train_labels} labels")
    print(f"  Val: {val_images} images, {val_labels} labels")

    return True
