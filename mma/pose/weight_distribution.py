"""
Weight distribution analysis from 2D pose estimation

Provides:
- Front/rear weight distribution
- Left/right weight distribution
- Balance stability metrics
- Body lean analysis
"""

import numpy as np
from typing import Dict, Tuple, Optional


def calculate_weight_distribution(
    left_ankle: Tuple[float, float],
    right_ankle: Tuple[float, float],
    left_hip: Tuple[float, float],
    right_hip: Tuple[float, float],
    left_knee: Tuple[float, float],
    right_knee: Tuple[float, float],
    conf_threshold: float = 0.3
) -> Dict:
    """
    Calculate front/rear and left/right weight distribution from 2D pose keypoints

    Args:
        left_ankle: (x, y) or (x, y, conf)
        right_ankle: (x, y) or (x, y, conf)
        left_hip: (x, y) or (x, y, conf)
        right_hip: (x, y) or (x, y, conf)
        left_knee: (x, y) or (x, y, conf)
        right_knee: (x, y) or (x, y, conf)
        conf_threshold: Minimum confidence to use keypoint

    Returns:
        Dict with:
        - front_weight: 0~1, 1 = all weight on front
        - rear_weight: 0~1, 1 = all weight on rear
        - left_weight: 0~1, 1 = all weight on left
        - right_weight: 0~1, 1 = all weight on right
        - forward_lean: 0~1, amount of forward body lean
        - stability_score: 0~1, higher = more stable
        - is_valid: bool, all required keypoints available

    Example:
        >>> result = calculate_weight_distribution(
        ...     left_ankle=(100, 200), right_ankle=(150, 205),
        ...     left_hip=(110, 100), right_hip=(140, 100),
        ...     left_knee=(105, 150), right_knee=(145, 150)
        ... )
        >>> print(f"Front: {result['front_weight']:.2f}, Rear: {result['rear_weight']:.2f}")
    """
    # Extract coordinates and confidence
    def extract_coords(point):
        if len(point) == 2:
            return point[0], point[1], 1.0
        else:
            return point[0], point[1], point[2]

    la_x, la_y, la_conf = extract_coords(left_ankle)
    ra_x, ra_y, ra_conf = extract_coords(right_ankle)
    lh_x, lh_y, lh_conf = extract_coords(left_hip)
    rh_x, rh_y, rh_conf = extract_coords(right_hip)
    lk_x, lk_y, lk_conf = extract_coords(left_knee)
    rk_x, rk_y, rk_conf = extract_coords(right_knee)

    # Check if all required keypoints are valid
    is_valid = all([
        la_conf >= conf_threshold,
        ra_conf >= conf_threshold,
        lh_conf >= conf_threshold,
        rh_conf >= conf_threshold,
        lk_conf >= conf_threshold,
        rk_conf >= conf_threshold
    ])

    if not is_valid:
        return {
            'front_weight': 0.5,
            'rear_weight': 0.5,
            'left_weight': 0.5,
            'right_weight': 0.5,
            'forward_lean': 0.5,
            'stability_score': 0.0,
            'is_valid': False
        }

    # ============================================
    # 1. FRONT/REAR WEIGHT DISTRIBUTION (Y-axis)
    # ============================================
    # Principle: If body is tilted forward, weight shifts to front (lower ankle Y)
    # If body is upright, weight is balanced on rear

    ankle_center_y = (la_y + ra_y) / 2
    hip_center_y = (lh_y + rh_y) / 2

    # Leg length (hip to ankle vertical distance)
    leg_length = hip_center_y - ankle_center_y

    if leg_length < 1:  # Avoid division by zero
        leg_length = 1

    # Forward lean ratio (0 = fully forward, 1 = fully backward)
    # When ankle is lower (smaller Y), forward_lean is closer to 0
    forward_lean = (ankle_center_y - hip_center_y) / leg_length
    forward_lean = np.clip(forward_lean, 0, 1)

    # Weight distribution (inverse of forward_lean)
    front_weight = 1.0 - forward_lean
    rear_weight = forward_lean

    # ============================================
    # 2. LEFT/RIGHT WEIGHT DISTRIBUTION (X-axis)
    # ============================================
    # Principle: The leg that is lower (higher Y) carries more weight

    left_leg_height = la_y  # Lower Y = higher position = less weight
    right_leg_height = ra_y

    # Normalize: higher Y value = more weight on that leg
    total_leg_activity = left_leg_height + right_leg_height
    if total_leg_activity < 1:
        total_leg_activity = 1

    # Inverse: if left ankle is lower (higher Y), then right has more weight
    right_weight = left_leg_height / total_leg_activity
    left_weight = right_leg_height / total_leg_activity

    # ============================================
    # 3. STABILITY SCORE
    # ============================================
    # Factors: hip-ankle alignment, leg symmetry

    # Hip-ankle horizontal alignment (left)
    left_hip_ankle_dist = abs(lh_x - la_x)

    # Hip-ankle horizontal alignment (right)
    right_hip_ankle_dist = abs(rh_x - ra_x)

    # Average alignment distance (smaller is better)
    avg_alignment = (left_hip_ankle_dist + right_hip_ankle_dist) / 2

    # Normalize alignment (assume 50 pixels = poor, 0 = perfect)
    alignment_score = max(0, 1 - avg_alignment / 100)

    # Leg symmetry (how balanced left/right are)
    weight_asymmetry = abs(left_weight - right_weight)
    symmetry_score = 1 - weight_asymmetry

    # Stability = weighted average of alignment and symmetry
    stability_score = (alignment_score * 0.5 + symmetry_score * 0.5)
    stability_score = np.clip(stability_score, 0, 1)

    return {
        'front_weight': float(front_weight),
        'rear_weight': float(rear_weight),
        'left_weight': float(left_weight),
        'right_weight': float(right_weight),
        'forward_lean': float(forward_lean),
        'stability_score': float(stability_score),
        'is_valid': True
    }


def calculate_weight_from_keypoints_dict(
    keypoints: Dict[str, Tuple[float, float, float]],
    conf_threshold: float = 0.3
) -> Dict:
    """
    Calculate weight distribution from keypoints dictionary (from CSV)

    Args:
        keypoints: Dict with keys like 'left_ankle_x', 'left_ankle_y', 'left_ankle_conf'
        conf_threshold: Minimum confidence threshold

    Returns:
        Weight distribution dict

    Example:
        >>> keypoints = {
        ...     'left_ankle_x': 100, 'left_ankle_y': 200, 'left_ankle_conf': 0.9,
        ...     'right_ankle_x': 150, 'right_ankle_y': 205, 'right_ankle_conf': 0.9,
        ...     'left_hip_x': 110, 'left_hip_y': 100, 'left_hip_conf': 0.95,
        ...     'right_hip_x': 140, 'right_hip_y': 100, 'right_hip_conf': 0.95,
        ...     'left_knee_x': 105, 'left_knee_y': 150, 'left_knee_conf': 0.9,
        ...     'right_knee_x': 145, 'right_knee_y': 150, 'right_knee_conf': 0.9,
        ... }
        >>> result = calculate_weight_from_keypoints_dict(keypoints)
    """
    try:
        left_ankle = (keypoints['left_ankle_x'], keypoints['left_ankle_y'], keypoints['left_ankle_conf'])
        right_ankle = (keypoints['right_ankle_x'], keypoints['right_ankle_y'], keypoints['right_ankle_conf'])
        left_hip = (keypoints['left_hip_x'], keypoints['left_hip_y'], keypoints['left_hip_conf'])
        right_hip = (keypoints['right_hip_x'], keypoints['right_hip_y'], keypoints['right_hip_conf'])
        left_knee = (keypoints['left_knee_x'], keypoints['left_knee_y'], keypoints['left_knee_conf'])
        right_knee = (keypoints['right_knee_x'], keypoints['right_knee_y'], keypoints['right_knee_conf'])
    except KeyError as e:
        return {
            'front_weight': 0.5,
            'rear_weight': 0.5,
            'left_weight': 0.5,
            'right_weight': 0.5,
            'forward_lean': 0.5,
            'stability_score': 0.0,
            'is_valid': False,
            'error': f"Missing keypoint: {e}"
        }

    return calculate_weight_distribution(
        left_ankle=left_ankle,
        right_ankle=right_ankle,
        left_hip=left_hip,
        right_hip=right_hip,
        left_knee=left_knee,
        right_knee=right_knee,
        conf_threshold=conf_threshold
    )


def analyze_stance(weight_dist: Dict) -> str:
    """
    Classify stance type based on weight distribution

    Args:
        weight_dist: Weight distribution dict from calculate_weight_distribution()

    Returns:
        Stance description string

    Example:
        >>> stance = analyze_stance(weight_dist)
        >>> print(stance)  # "Front-heavy aggressive stance"
    """
    if not weight_dist['is_valid']:
        return "Invalid pose data"

    front = weight_dist['front_weight']
    left = weight_dist['left_weight']

    front_desc = ""
    if front > 0.6:
        front_desc = "Front-heavy (aggressive/attacking)"
    elif front < 0.4:
        front_desc = "Rear-heavy (defensive/backing up)"
    else:
        front_desc = "Balanced stance"

    left_desc = ""
    if abs(left - 0.5) > 0.2:
        if left > 0.6:
            left_desc = " - Left leg weighted"
        else:
            left_desc = " - Right leg weighted"

    return front_desc + left_desc


def get_weight_direction(weight_dist: Dict) -> Tuple[float, float]:
    """
    Get 2D direction of weight distribution (like a vector)

    Args:
        weight_dist: Weight distribution dict

    Returns:
        (x_direction, y_direction) where:
        - x: -1 (left) to 1 (right)
        - y: -1 (forward) to 1 (backward)

    Example:
        >>> direction = get_weight_direction(weight_dist)
        >>> print(f"Direction: {direction}")  # (-0.2, -0.6) = forward-left lean
    """
    x_dir = (weight_dist['right_weight'] - weight_dist['left_weight'])  # -1 to 1
    y_dir = (weight_dist['rear_weight'] - weight_dist['front_weight'])  # -1 to 1

    return (float(x_dir), float(y_dir))
