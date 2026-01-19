"""
Pose estimation module - 2D keypoint detection

Provides:
- Pose estimator with YOLO models
- Keypoint utilities and transformations
- Keypoint filtering and validation
"""

from .estimator import PoseEstimator, get_pose_estimator
from .keypoint_utils import (
    crop_person,
    transform_keypoints_to_original,
    filter_keypoints,
    compute_keypoint_stats,
    compute_pose_center,
    validate_keypoints,
    keypoints_to_array,
    array_to_keypoints,
)

__all__ = [
    # Estimator
    "PoseEstimator",
    "get_pose_estimator",
    # Keypoint utilities
    "crop_person",
    "transform_keypoints_to_original",
    "filter_keypoints",
    "compute_keypoint_stats",
    "compute_pose_center",
    "validate_keypoints",
    "keypoints_to_array",
    "array_to_keypoints",
]
