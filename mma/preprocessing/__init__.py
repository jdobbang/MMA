"""
Preprocessing module - Dataset generation and preparation

Provides:
- YOLO dataset generation from detection and pose data
- Train/val/test set splitting
- Data validation and augmentation
- Dataset utilities
"""

from .dataset_generator import (
    create_yolo_detection_dataset,
    create_yolo_pose_dataset,
)
from .splitter import (
    split_yolo_dataset,
    split_detection_pose_dataset,
)

__all__ = [
    # Dataset generation
    "create_yolo_detection_dataset",
    "create_yolo_pose_dataset",
    # Splitting
    "split_yolo_dataset",
    "split_detection_pose_dataset",
]
