"""
MMA Pipeline - Modular Multi-task Architecture for MMA Athlete Tracking and Pose Estimation

A comprehensive Python package for:
- YOLO-based object detection
- Multi-object tracking with Re-ID
- 2D/3D pose estimation
- SMPL rendering and visualization
"""

__version__ = "0.1.0"
__author__ = "MMA Project Team"

# Core imports (no external dependencies)
from .core.config import MMAConfig, DetectionConfig, TrackingConfig, PoseConfig, PathConfig
from .core.constants import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON_CONNECTIONS,
    SMPL_NUM_VERTICES,
    SMPL_NUM_FACES,
    SMPL_NUM_JOINTS,
    POSES2D_TO_COCO_MAPPING,
    DEFAULT_PLAYER_COLORS,
)
from .core.exceptions import (
    MMAException,
    ImageLoadError,
    DataLoadError,
    ModelLoadError,
    ConfigError,
)

# Lazy imports for modules with external dependencies
def __getattr__(name):
    """Lazy loading for modules with external dependencies"""
    if name == "ImageLoader":
        from .io.data_loader import ImageLoader
        return ImageLoader
    elif name == "NPYLoader":
        from .io.data_loader import NPYLoader
        return NPYLoader
    elif name == "CSVWriter":
        from .io.csv_handler import CSVWriter
        return CSVWriter
    elif name == "CSVReader":
        from .io.csv_handler import CSVReader
        return CSVReader
    elif name == "DetectionRow":
        from .io.csv_handler import DetectionRow
        return DetectionRow
    elif name == "TrackingRow":
        from .io.csv_handler import TrackingRow
        return TrackingRow
    elif name in ("iou", "iou_batch", "bbox_to_yolo", "yolo_to_bbox",
                  "bbox_from_keypoints", "convert_bbox_to_z", "convert_x_to_bbox"):
        from .detection import bbox_utils
        return getattr(bbox_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Config
    "MMAConfig",
    "DetectionConfig",
    "TrackingConfig",
    "PoseConfig",
    "PathConfig",
    # Constants
    "COCO_KEYPOINT_NAMES",
    "COCO_SKELETON_CONNECTIONS",
    "SMPL_NUM_VERTICES",
    "SMPL_NUM_FACES",
    "SMPL_NUM_JOINTS",
    "POSES2D_TO_COCO_MAPPING",
    "DEFAULT_PLAYER_COLORS",
    # Exceptions
    "MMAException",
    "ImageLoadError",
    "DataLoadError",
    "ModelLoadError",
    "ConfigError",
    # IO
    "ImageLoader",
    "NPYLoader",
    "CSVWriter",
    "CSVReader",
    "DetectionRow",
    "TrackingRow",
    # Detection
    "iou",
    "iou_batch",
    "bbox_to_yolo",
    "yolo_to_bbox",
    "bbox_from_keypoints",
    "convert_bbox_to_z",
    "convert_x_to_bbox",
]
