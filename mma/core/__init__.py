"""
Core module - Configuration, constants, and exceptions for MMA pipeline
"""

from .config import (
    MMAConfig,
    DetectionConfig,
    TrackingConfig,
    PoseConfig,
    PathConfig,
)
from .constants import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON_CONNECTIONS,
    SMPL_NUM_VERTICES,
    SMPL_NUM_FACES,
    SMPL_NUM_JOINTS,
    POSES2D_TO_COCO_MAPPING,
    DEFAULT_PLAYER_COLORS,
)
from .exceptions import (
    MMAException,
    ImageLoadError,
    DataLoadError,
    ModelLoadError,
    ConfigError,
)

__all__ = [
    "MMAConfig",
    "DetectionConfig",
    "TrackingConfig",
    "PoseConfig",
    "PathConfig",
    "COCO_KEYPOINT_NAMES",
    "COCO_SKELETON_CONNECTIONS",
    "SMPL_NUM_VERTICES",
    "SMPL_NUM_FACES",
    "SMPL_NUM_JOINTS",
    "POSES2D_TO_COCO_MAPPING",
    "DEFAULT_PLAYER_COLORS",
    "MMAException",
    "ImageLoadError",
    "DataLoadError",
    "ModelLoadError",
    "ConfigError",
]
