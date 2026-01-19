"""
Detection module - YOLO-based object detection

Provides:
- YOLO detector wrapper with singleton caching
- Bounding box utilities (IoU, conversions, transformations)
- Batch detection support
- Bbox validation and manipulation
"""

from .bbox_utils import (
    iou,
    iou_batch,
    bbox_to_yolo,
    yolo_to_bbox,
    bbox_from_keypoints,
    convert_bbox_to_z,
    convert_x_to_bbox,
    clip_bbox,
    get_bbox_area,
    get_bbox_center,
    get_bbox_size,
    validate_bbox,
)

from .detector import YOLODetector, get_detector

__all__ = [
    # Bbox utilities
    "iou",
    "iou_batch",
    "bbox_to_yolo",
    "yolo_to_bbox",
    "bbox_from_keypoints",
    "convert_bbox_to_z",
    "convert_x_to_bbox",
    "clip_bbox",
    "get_bbox_area",
    "get_bbox_center",
    "get_bbox_size",
    "validate_bbox",
    # Detector
    "YOLODetector",
    "get_detector",
]
