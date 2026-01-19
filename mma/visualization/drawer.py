"""
Drawing utilities for visualization of detection, tracking, and pose results

Provides:
- Draw bounding boxes with track IDs
- Draw pose skeleton
- Draw keypoints with confidence
- Color management

Replaces:
- visualize_tracking.py: draw_skeleton(), generate_color()
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

from ..core.constants import COCO_KEYPOINT_NAMES, COCO_SKELETON_CONNECTIONS, DEFAULT_PLAYER_COLORS


def generate_track_color(track_id: int) -> Tuple[int, int, int]:
    """
    Generate consistent color for track ID

    Args:
        track_id: Track ID (int)

    Returns:
        (B, G, R) color tuple

    Example:
        >>> color = generate_track_color(1)
        >>> print(color)  # (0, 0, 255) for player 1 (red)
    """
    # Special colors for MMA players
    if track_id in DEFAULT_PLAYER_COLORS:
        return DEFAULT_PLAYER_COLORS[track_id]
    else:
        # Consistent random color for other IDs
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    track_id: int,
    confidence: float = 1.0,
    thickness: int = 2,
    text_scale: float = 0.6
) -> np.ndarray:
    """
    Draw bounding box with track ID on image

    Args:
        image: Input image (H, W, 3) BGR
        bbox: (x1, y1, x2, y2) in pixel coordinates
        track_id: Track ID to display
        confidence: Detection confidence (optional)
        thickness: Box line thickness
        text_scale: Text font scale

    Returns:
        Modified image with bbox drawn

    Example:
        >>> image = cv2.imread('frame.jpg')
        >>> bbox = (100, 150, 300, 400)
        >>> image = draw_bbox(image, bbox, track_id=1, confidence=0.95)
    """
    import cv2

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    color = generate_track_color(track_id)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Draw label
    label = f"ID:{track_id}"
    if confidence < 1.0:
        label += f" ({confidence:.2f})"

    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, text_scale, 1)

    # Draw background rectangle for text
    cv2.rectangle(
        image,
        (x1, y1 - text_h - baseline - 4),
        (x1 + text_w + 4, y1),
        color,
        -1
    )

    # Draw text
    cv2.putText(
        image,
        label,
        (x1 + 2, y1 - baseline - 2),
        font,
        text_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return image


def draw_skeleton(
    image: np.ndarray,
    keypoints: Dict[str, Tuple[float, float, float]],
    track_id: int,
    conf_threshold: float = 0.3,
    line_thickness: int = 2,
    point_radius: int = 4
) -> np.ndarray:
    """
    Draw pose skeleton on image

    Args:
        image: Input image (H, W, 3) BGR
        keypoints: Dict mapping keypoint names to (x, y, conf)
        track_id: Track ID for color selection
        conf_threshold: Minimum confidence for visualization
        line_thickness: Skeleton line thickness
        point_radius: Keypoint circle radius

    Returns:
        Modified image with skeleton drawn

    Example:
        >>> keypoints = {'nose': (100, 100, 0.9), 'left_shoulder': (80, 150, 0.85)}
        >>> image = draw_skeleton(image, keypoints, track_id=1)
    """
    import cv2

    point_color = generate_track_color(track_id)
    # Brighter line color
    line_color = tuple(min(255, int(c * 1.3)) for c in point_color)

    # Convert keypoints to array with indices
    kp_coords = []
    for name in COCO_KEYPOINT_NAMES:
        if name in keypoints:
            x, y, conf = keypoints[name]
            if conf >= conf_threshold and x > 0 and y > 0:
                kp_coords.append((int(x), int(y), conf))
            else:
                kp_coords.append(None)
        else:
            kp_coords.append(None)

    # Draw skeleton connections
    for idx1, idx2 in COCO_SKELETON_CONNECTIONS:
        if kp_coords[idx1] is not None and kp_coords[idx2] is not None:
            pt1 = (kp_coords[idx1][0], kp_coords[idx1][1])
            pt2 = (kp_coords[idx2][0], kp_coords[idx2][1])
            cv2.line(image, pt1, pt2, line_color, line_thickness)

    # Draw keypoint circles
    for kp in kp_coords:
        if kp is not None:
            x, y, conf = kp
            # Radius scales with confidence
            radius = int(point_radius * (0.5 + conf * 0.5))
            cv2.circle(image, (x, y), radius, point_color, -1)
            cv2.circle(image, (x, y), radius, (255, 255, 255), 1)  # White border

    return image


def draw_keypoints(
    image: np.ndarray,
    keypoints: Dict[str, Tuple[float, float, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    conf_threshold: float = 0.3,
    radius: int = 3
) -> np.ndarray:
    """
    Draw keypoint circles only (no skeleton)

    Args:
        image: Input image
        keypoints: Keypoint dict
        color: (B, G, R) color
        conf_threshold: Minimum confidence
        radius: Circle radius

    Returns:
        Modified image with keypoints drawn

    Example:
        >>> image = draw_keypoints(image, keypoints, color=(0, 255, 0))
    """
    import cv2

    for x, y, conf in keypoints.values():
        if conf >= conf_threshold and x > 0 and y > 0:
            cv2.circle(image, (int(x), int(y)), radius, color, -1)
            cv2.circle(image, (int(x), int(y)), radius, (255, 255, 255), 1)

    return image


def draw_multiple_bboxes(
    image: np.ndarray,
    detections: List[Dict],
    thickness: int = 2
) -> np.ndarray:
    """
    Draw multiple bounding boxes on image

    Args:
        image: Input image
        detections: List of dicts with 'bbox', 'track_id', 'confidence'
        thickness: Box line thickness

    Returns:
        Modified image

    Example:
        >>> detections = [
        ...     {'bbox': (100, 150, 300, 400), 'track_id': 1, 'confidence': 0.95},
        ...     {'bbox': (350, 100, 500, 350), 'track_id': 2, 'confidence': 0.92}
        ... ]
        >>> image = draw_multiple_bboxes(image, detections)
    """
    for det in detections:
        bbox = det["bbox"]
        track_id = det["track_id"]
        confidence = det.get("confidence", 1.0)
        image = draw_bbox(image, bbox, track_id, confidence, thickness)

    return image


def draw_multiple_skeletons(
    image: np.ndarray,
    pose_data: List[Dict],
    conf_threshold: float = 0.3,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw multiple pose skeletons on image

    Args:
        image: Input image
        pose_data: List of dicts with 'track_id', 'keypoints'
        conf_threshold: Keypoint confidence threshold
        line_thickness: Skeleton line thickness

    Returns:
        Modified image

    Example:
        >>> pose_data = [
        ...     {'track_id': 1, 'keypoints': {'nose': (100, 100, 0.9), ...}},
        ...     {'track_id': 2, 'keypoints': {'nose': (400, 100, 0.88), ...}}
        ... ]
        >>> image = draw_multiple_skeletons(image, pose_data)
    """
    for pose in pose_data:
        track_id = pose["track_id"]
        keypoints = pose["keypoints"]
        image = draw_skeleton(image, keypoints, track_id, conf_threshold, line_thickness)

    return image


def add_text_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    thickness: int = 1,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0)
) -> np.ndarray:
    """
    Add text label to image

    Args:
        image: Input image
        text: Text to display
        position: (x, y) position
        font_scale: Font size
        thickness: Text thickness
        color: (B, G, R) text color
        bg_color: Background color (None for no background)

    Returns:
        Modified image

    Example:
        >>> image = add_text_label(image, "Frame 1/100", position=(10, 30))
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position

    if bg_color is not None:
        # Draw background
        cv2.rectangle(
            image,
            (x - 2, y - text_h - baseline - 2),
            (x + text_w + 2, y + baseline + 2),
            bg_color,
            -1
        )

    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return image


def overlay_alpha(
    image: np.ndarray,
    overlay: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Overlay transparent image on top of base image

    Args:
        image: Base image (H, W, 3)
        overlay: Overlay image (H, W, 3) or (H, W, 4) with alpha channel
        alpha: Transparency (0.0 = fully transparent, 1.0 = fully opaque)

    Returns:
        Blended image

    Example:
        >>> heatmap = np.zeros_like(image)
        >>> # ... populate heatmap ...
        >>> result = overlay_alpha(image, heatmap, alpha=0.5)
    """
    if overlay.shape[2] == 4:
        # Has alpha channel
        overlay_rgb = overlay[:, :, :3]
        overlay_alpha = overlay[:, :, 3:4] / 255.0
        blended = image.astype(np.float32) * (1 - alpha) + overlay_rgb.astype(np.float32) * alpha * overlay_alpha
    else:
        # No alpha channel
        blended = image.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha

    return np.clip(blended, 0, 255).astype(np.uint8)
