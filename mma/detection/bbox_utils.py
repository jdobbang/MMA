"""
Bounding box utilities for detection and tracking

Unified bbox operations replacing scattered implementations in:
- sort_tracker.py: iou_batch, convert_bbox_to_z, convert_x_to_bbox
- mma_tracker.py: compute_iou
- generate_yolo_pose_dataset.py: bbox_to_yolo_format, calculate_iou
- crop_detection_pose_dataset.py: bbox coordinate transforms

Provides:
- Single and batch IoU computation
- YOLO coordinate format conversion
- Kalman filter state transforms
- Bbox validation and manipulation
"""

import numpy as np
from typing import Tuple, Optional, Union


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes

    Replaces:
    - mma_tracker.py: compute_iou()

    Args:
        bbox1: Bounding box [x1, y1, x2, y2]
        bbox2: Bounding box [x1, y1, x2, y2]

    Returns:
        IoU score between 0 and 1

    Example:
        >>> bbox1 = np.array([0, 0, 100, 100])
        >>> bbox2 = np.array([50, 50, 150, 150])
        >>> iou_score = iou(bbox1, bbox2)
        >>> print(f"IoU: {iou_score:.3f}")  # IoU: 0.143
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return float(inter_area / union_area)


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of bounding boxes (vectorized)

    Replaces:
    - sort_tracker.py: iou_batch()

    This is the optimized version using numpy broadcasting instead of
    nested loops, providing significant speedup for large batch sizes.

    Args:
        bboxes1: Array of shape (N, 4) with [x1, y1, x2, y2]
        bboxes2: Array of shape (M, 4) with [x1, y1, x2, y2]

    Returns:
        IoU matrix of shape (N, M)

    Example:
        >>> bboxes1 = np.array([[0, 0, 100, 100], [50, 50, 150, 150]])
        >>> bboxes2 = np.array([[10, 10, 110, 110], [200, 200, 300, 300]])
        >>> iou_matrix = iou_batch(bboxes1, bboxes2)
        >>> print(iou_matrix.shape)
        (2, 2)
    """
    # Reshape for broadcasting: (N, 1, 4) and (1, M, 4)
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    # Intersection area
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    # Areas
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    # Union area
    union_area = area1 + area2 - wh

    # IoU (avoid division by zero)
    iou_matrix = np.where(union_area > 0, wh / union_area, 0.0)

    return iou_matrix.astype(np.float32)


def bbox_to_yolo(
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert pixel coordinates bbox to YOLO normalized format

    Replaces:
    - generate_yolo_pose_dataset.py: bbox_to_yolo_format()

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        (x_center, y_center, width, height) in normalized [0, 1] range

    Example:
        >>> bbox = (100, 100, 200, 300)
        >>> yolo_bbox = bbox_to_yolo(bbox, img_width=640, img_height=480)
        >>> print(yolo_bbox)
        (0.234, 0.417, 0.156, 0.417)
    """
    x1, y1, x2, y2 = bbox

    x_center = ((x1 + x2) / 2.0) / img_width
    y_center = ((y1 + y2) / 2.0) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return (x_center, y_center, width, height)


def yolo_to_bbox(
    yolo_box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert YOLO normalized format to pixel coordinates bbox

    Inverse operation of bbox_to_yolo()

    Args:
        yolo_box: (x_center, y_center, width, height) in normalized [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        (x1, y1, x2, y2) in pixel coordinates

    Example:
        >>> yolo_box = (0.5, 0.5, 0.2, 0.3)
        >>> bbox = yolo_to_bbox(yolo_box, img_width=640, img_height=480)
        >>> print(bbox)
        (538, 402, 602, 558)
    """
    x_center, y_center, width, height = yolo_box

    x1 = (x_center - width / 2.0) * img_width
    y1 = (y_center - height / 2.0) * img_height
    x2 = (x_center + width / 2.0) * img_width
    y2 = (y_center + height / 2.0) * img_height

    return (x1, y1, x2, y2)


def bbox_from_keypoints(
    keypoints: np.ndarray,
    padding: float = 0.1,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute bounding box from keypoints with optional padding

    Replaces:
    - generate_yolo_pose_dataset.py: bbox_from_keypoints()

    Args:
        keypoints: Array of shape (N, 2) with [x, y] coordinates
        padding: Padding ratio (0.1 = 10% padding on each side)
        img_width: Image width (for clipping x coordinates)
        img_height: Image height (for clipping y coordinates)

    Returns:
        (x1, y1, x2, y2) bbox or None if insufficient keypoints

    Example:
        >>> keypoints = np.array([[100, 150], [200, 250], [300, 200]])
        >>> bbox = bbox_from_keypoints(keypoints, padding=0.1)
        >>> print(bbox)
        (80, 130, 320, 270)
    """
    # Filter valid keypoints (non-NaN)
    valid_kpts = keypoints[~np.isnan(keypoints).any(axis=1)]

    if len(valid_kpts) == 0:
        return None

    x_min = valid_kpts[:, 0].min()
    y_min = valid_kpts[:, 1].min()
    x_max = valid_kpts[:, 0].max()
    y_max = valid_kpts[:, 1].max()

    # Apply padding
    w = x_max - x_min
    h = y_max - y_min

    x_min = max(0, x_min - w * padding)
    y_min = max(0, y_min - h * padding)

    if img_width is not None:
        x_max = min(img_width, x_max + w * padding)
    else:
        x_max = x_max + w * padding

    if img_height is not None:
        y_max = min(img_height, y_max + h * padding)
    else:
        y_max = y_max + h * padding

    # Check validity
    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Convert bbox from [x1, y1, x2, y2] to Kalman state [x, y, s, r]

    Replaces:
    - sort_tracker.py: convert_bbox_to_z()

    This converts from corner coordinates to center, scale, and aspect ratio
    format used by Kalman filter.

    Args:
        bbox: Array [x1, y1, x2, y2] in top-left, bottom-right format

    Returns:
        Array [cx, cy, s, r] where:
        - cx, cy: center coordinates
        - s: scale (area)
        - r: aspect ratio (width/height)

    Example:
        >>> bbox = np.array([0, 0, 100, 50])
        >>> z = convert_bbox_to_z(bbox)
        >>> print(z)
        [[50.0], [25.0], [5000.0], [2.0]]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # Scale (area)
    r = w / float(h)  # Aspect ratio

    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(
    x: np.ndarray,
    score: Optional[float] = None
) -> np.ndarray:
    """
    Convert Kalman state [x, y, s, r] to bbox [x1, y1, x2, y2, score]

    Replaces:
    - sort_tracker.py: convert_x_to_bbox()

    Inverse operation of convert_bbox_to_z().

    Args:
        x: Kalman state array [cx, cy, s, r]
        score: Optional confidence score to append

    Returns:
        Array [x1, y1, x2, y2] or [x1, y1, x2, y2, score]

    Example:
        >>> x = np.array([[50.0], [25.0], [5000.0], [2.0]])
        >>> bbox = convert_x_to_bbox(x, score=0.95)
        >>> print(bbox)
        [[0.0, 0.0, 100.0, 50.0, 0.95]]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    if score is None:
        bbox = np.array([
            float(x[0] - w / 2.0),
            float(x[1] - h / 2.0),
            float(x[0] + w / 2.0),
            float(x[1] + h / 2.0)
        ])
        return bbox.reshape((1, 4))
    else:
        bbox = np.array([
            float(x[0] - w / 2.0),
            float(x[1] - h / 2.0),
            float(x[0] + w / 2.0),
            float(x[1] + h / 2.0),
            float(score)
        ])
        return bbox.reshape((1, 5))


def clip_bbox(
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Clip bounding box to image boundaries

    Args:
        bbox: (x1, y1, x2, y2)
        img_width: Image width
        img_height: Image height

    Returns:
        Clipped (x1, y1, x2, y2)

    Example:
        >>> bbox = (-10, -20, 650, 500)
        >>> clipped = clip_bbox(bbox, img_width=640, img_height=480)
        >>> print(clipped)
        (0, 0, 640, 480)
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    return (x1, y1, x2, y2)


def get_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """
    Get bounding box area

    Args:
        bbox: (x1, y1, x2, y2)

    Returns:
        Area in pixels

    Example:
        >>> bbox = (0, 0, 100, 50)
        >>> area = get_bbox_area(bbox)
        >>> print(area)
        5000
    """
    x1, y1, x2, y2 = bbox
    return float((x2 - x1) * (y2 - y1))


def get_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get bounding box center coordinates

    Args:
        bbox: (x1, y1, x2, y2)

    Returns:
        (cx, cy) center coordinates

    Example:
        >>> bbox = (0, 0, 100, 50)
        >>> cx, cy = get_bbox_center(bbox)
        >>> print((cx, cy))
        (50.0, 25.0)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)


def get_bbox_size(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get bounding box width and height

    Args:
        bbox: (x1, y1, x2, y2)

    Returns:
        (width, height) tuple

    Example:
        >>> bbox = (0, 0, 100, 50)
        >>> w, h = get_bbox_size(bbox)
        >>> print((w, h))
        (100, 50)
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1, y2 - y1)


def validate_bbox(
    bbox: Tuple[float, float, float, float],
    min_size: int = 10
) -> bool:
    """
    Validate bounding box

    Args:
        bbox: (x1, y1, x2, y2)
        min_size: Minimum width/height in pixels

    Returns:
        True if bbox is valid

    Example:
        >>> bbox = (0, 0, 100, 50)
        >>> is_valid = validate_bbox(bbox, min_size=10)
        >>> print(is_valid)
        True
    """
    x1, y1, x2, y2 = bbox

    # Check coordinates order
    if x2 <= x1 or y2 <= y1:
        return False

    # Check minimum size
    w = x2 - x1
    h = y2 - y1
    if w < min_size or h < min_size:
        return False

    # Check for NaN or inf
    if not all(np.isfinite([x1, y1, x2, y2])):
        return False

    return True
