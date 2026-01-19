"""
Keypoint utilities for pose estimation

Provides:
- Coordinate transformations (crop ↔ original)
- Keypoint filtering and validation
- Skeleton connectivity
- Pose metrics computation

Replaces:
- pose_estimation.py: transform_keypoints_to_original()
- visualize_tracking.py: Skeleton constants
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

from ..core.constants import COCO_KEYPOINT_NAMES, COCO_SKELETON_CONNECTIONS


def crop_person(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding: float = 0.0
) -> Tuple[np.ndarray, Dict]:
    """
    Extract person crop from image with optional padding

    Args:
        image: Input image (H, W, 3) BGR
        bbox: (x1, y1, x2, y2) in pixel coordinates
        padding: Padding ratio (0.1 = 10% on each side)

    Returns:
        (crop_image, crop_info) where crop_info contains offset and dimensions

    Example:
        >>> image = cv2.imread('frame.jpg')
        >>> bbox = (100, 150, 300, 400)
        >>> crop, info = crop_person(image, bbox, padding=0.1)
        >>> print(info['offset_x'], info['offset_y'])
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Apply padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding
    pad_y = box_h * padding

    # Expand bbox (clipped to image boundaries)
    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(w, int(x2 + pad_x))
    crop_y2 = min(h, int(y2 + pad_y))

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_info = {
        "offset_x": crop_x1,
        "offset_y": crop_y1,
        "crop_w": crop_x2 - crop_x1,
        "crop_h": crop_y2 - crop_y1,
    }

    return cropped, crop_info


def transform_keypoints_to_original(
    keypoints: Dict[str, Tuple[float, float, float]],
    crop_info: Dict
) -> Dict[str, Tuple[float, float, float]]:
    """
    Transform keypoints from crop coordinates to original image coordinates

    Args:
        keypoints: Dict mapping keypoint name to (x, y, conf) in crop coords
        crop_info: Crop information with offset_x, offset_y

    Returns:
        Keypoints dict in original image coordinates

    Example:
        >>> keypoints = {'nose': (50, 30, 0.9), 'left_shoulder': (40, 60, 0.85)}
        >>> crop_info = {'offset_x': 100, 'offset_y': 150}
        >>> transformed = transform_keypoints_to_original(keypoints, crop_info)
        >>> print(transformed['nose'])  # (150, 180, 0.9)
    """
    transformed = {}
    for name, (x, y, conf) in keypoints.items():
        orig_x = x + crop_info["offset_x"]
        orig_y = y + crop_info["offset_y"]
        transformed[name] = (orig_x, orig_y, conf)
    return transformed


def filter_keypoints(
    keypoints: Dict[str, Tuple[float, float, float]],
    conf_threshold: float = 0.3
) -> Dict[str, Tuple[float, float, float]]:
    """
    Filter keypoints below confidence threshold

    Args:
        keypoints: Keypoint dict
        conf_threshold: Minimum confidence

    Returns:
        Filtered keypoint dict (missing kpts have conf=-1 as flag)

    Example:
        >>> keypoints = {'nose': (100, 100, 0.9), 'left_eye': (110, 95, 0.2)}
        >>> filtered = filter_keypoints(keypoints, conf_threshold=0.3)
        >>> print('left_eye' in filtered)  # False
    """
    filtered = {}
    for name, (x, y, conf) in keypoints.items():
        if conf >= conf_threshold:
            filtered[name] = (x, y, conf)
    return filtered


def get_skeleton_connections() -> List[Tuple[int, int]]:
    """
    Get COCO skeleton connections as keypoint index pairs

    Returns:
        List of (keypoint_idx1, keypoint_idx2) tuples

    Example:
        >>> connections = get_skeleton_connections()
        >>> for idx1, idx2 in connections:
        ...     name1 = COCO_KEYPOINT_NAMES[idx1]
        ...     name2 = COCO_KEYPOINT_NAMES[idx2]
        ...     print(f"{name1} -- {name2}")
    """
    return COCO_SKELETON_CONNECTIONS


def compute_keypoint_stats(
    keypoints: Dict[str, Tuple[float, float, float]]
) -> Dict:
    """
    Compute statistics about keypoints

    Args:
        keypoints: Keypoint dict

    Returns:
        Stats dict with num_valid, mean_confidence, bounds

    Example:
        >>> keypoints = {'nose': (100, 100, 0.9), 'left_shoulder': (80, 150, 0.85)}
        >>> stats = compute_keypoint_stats(keypoints)
        >>> print(stats['num_valid'])  # 2
        >>> print(stats['mean_confidence'])  # 0.875
    """
    if not keypoints:
        return {
            "num_valid": 0,
            "mean_confidence": 0.0,
            "bounds": None,
            "center": None,
        }

    coords = np.array([(x, y) for x, y, _ in keypoints.values()])
    confs = np.array([conf for _, _, conf in keypoints.values()])

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    return {
        "num_valid": len(keypoints),
        "mean_confidence": float(np.mean(confs)),
        "bounds": (x_min, y_min, x_max, y_max),
        "center": (center_x, center_y),
    }


def compute_pose_center(
    keypoints: Dict[str, Tuple[float, float, float]]
) -> Optional[Tuple[float, float]]:
    """
    Compute center of mass of keypoints

    Args:
        keypoints: Keypoint dict with confidence

    Returns:
        (center_x, center_y) or None if no valid keypoints

    Example:
        >>> keypoints = {'nose': (100, 100, 0.9), 'left_hip': (80, 200, 0.8)}
        >>> center = compute_pose_center(keypoints)
        >>> print(center)  # (90, 150)
    """
    if not keypoints:
        return None

    coords = np.array([(x, y) for x, y, _ in keypoints.values()])
    center = coords.mean(axis=0)
    return tuple(center)


def validate_keypoints(
    keypoints: Dict[str, Tuple[float, float, float]],
    img_width: int,
    img_height: int
) -> bool:
    """
    Validate keypoints are within image bounds

    Args:
        keypoints: Keypoint dict
        img_width: Image width
        img_height: Image height

    Returns:
        True if all keypoints within bounds

    Example:
        >>> keypoints = {'nose': (100, 100, 0.9)}
        >>> is_valid = validate_keypoints(keypoints, 640, 480)
        >>> print(is_valid)  # True if (100, 100) is within bounds
    """
    for x, y, conf in keypoints.values():
        if x < 0 or x >= img_width or y < 0 or y >= img_height:
            return False
    return True


def keypoints_to_array(
    keypoints: Dict[str, Tuple[float, float, float]],
    keypoint_order: List[str] = None
) -> np.ndarray:
    """
    Convert keypoint dict to array format

    Args:
        keypoints: Keypoint dict
        keypoint_order: Order of keypoints (default: COCO order)

    Returns:
        Array of shape (N, 3) with [x, y, conf] rows

    Example:
        >>> keypoints = {'nose': (100, 100, 0.9), 'left_eye': (110, 95, 0.85)}
        >>> arr = keypoints_to_array(keypoints)
        >>> print(arr.shape)  # (2, 3)
    """
    if keypoint_order is None:
        keypoint_order = COCO_KEYPOINT_NAMES

    result = []
    for name in keypoint_order:
        if name in keypoints:
            x, y, conf = keypoints[name]
            result.append([x, y, conf])
        else:
            result.append([0.0, 0.0, 0.0])

    return np.array(result, dtype=np.float32)


def array_to_keypoints(
    arr: np.ndarray,
    keypoint_order: List[str] = None
) -> Dict[str, Tuple[float, float, float]]:
    """
    Convert array format back to keypoint dict

    Args:
        arr: Array of shape (N, 3) with [x, y, conf]
        keypoint_order: Order of keypoints (default: COCO order)

    Returns:
        Keypoint dict

    Example:
        >>> arr = np.array([[100, 100, 0.9], [110, 95, 0.85]])
        >>> keypoints = array_to_keypoints(arr)
        >>> print(keypoints['nose'])  # (100, 100, 0.9)
    """
    if keypoint_order is None:
        keypoint_order = COCO_KEYPOINT_NAMES

    keypoints = {}
    for i, name in enumerate(keypoint_order):
        if i < len(arr):
            x, y, conf = arr[i]
            keypoints[name] = (float(x), float(y), float(conf))

    return keypoints
