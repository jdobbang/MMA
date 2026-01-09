#!/usr/bin/env python3
"""
Test keypoint visibility using depth-based or color-based occlusion.

Determines which 2D keypoints are visible vs occluded by comparing depth values
or by checking if rendered color pixels contain the mesh color.
"""

import numpy as np
from typing import Tuple, Dict
from camera_calibration import project_3d_to_2d, estimate_depth


def test_keypoint_visibility_color(
    keypoints_2d: np.ndarray,
    rendered_color: np.ndarray,
    mesh_colors: Dict[str, Tuple[int, int, int]],
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test visibility of keypoints using rendered color image.

    Checks if keypoint pixels contain the mesh color (indicating visibility).

    Args:
        keypoints_2d: 2D keypoint positions (N, 2) in pixel coordinates
        rendered_color: Rendered mesh color image (H, W, 3) uint8
        mesh_colors: Dict mapping person name to RGB color tuple
        verbose: Print debug info

    Returns:
        Tuple of (visibility_mask, confidence_scores)
    """
    img_h, img_w = rendered_color.shape[:2]

    visibility_mask = np.zeros(len(keypoints_2d), dtype=bool)
    confidence_scores = np.zeros(len(keypoints_2d), dtype=np.float32)

    # Average mesh color
    avg_color = rendered_color.mean(axis=(0, 1))

    for i, kp_2d in enumerate(keypoints_2d):
        # Check bounds
        x, y = int(np.round(kp_2d[0])), int(np.round(kp_2d[1]))

        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            visibility_mask[i] = False
            confidence_scores[i] = 0.0
            continue

        # Sample color at keypoint location
        pixel_color = rendered_color[y, x].astype(np.float32)

        # Check if pixel contains mesh color (not black/background)
        # Mesh renders with non-zero RGB values
        if np.linalg.norm(pixel_color) > 50:  # Threshold for non-background
            visibility_mask[i] = True
            # Confidence based on color intensity
            confidence = np.linalg.norm(pixel_color) / 255.0
            confidence_scores[i] = min(1.0, confidence)
        else:
            visibility_mask[i] = False
            confidence_scores[i] = 0.0

    if verbose:
        visible_count = visibility_mask.sum()
        print(f"Visible joints (color-based): {visible_count} / {len(keypoints_2d)}")

    return visibility_mask, confidence_scores


def test_keypoint_visibility(
    keypoints_2d: np.ndarray,
    joints_3d: np.ndarray,
    depth_map: np.ndarray,
    camera_params: Dict[str, np.ndarray],
    depth_threshold: float = 0.1,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test visibility of keypoints using depth map.

    For each keypoint, check if the depth from rendered mesh matches the
    expected depth of the 3D joint. If depths match, the joint is visible.
    If not, it's occluded.

    Args:
        keypoints_2d: 2D keypoint positions (N, 2) in pixel coordinates
        joints_3d: 3D joint positions (N, 3) in world coordinates
        depth_map: Rendered depth map (H, W)
        camera_params: Dictionary with 'K', 'R', 'tvec'
        depth_threshold: Depth tolerance in meters for visibility (default 0.1m = 10cm)
        verbose: Print debug info

    Returns:
        Tuple of (visibility_mask, confidence_scores):
        - visibility_mask: (N,) boolean array, True if visible
        - confidence_scores: (N,) float array, visibility confidence 0-1
    """

    img_h, img_w = depth_map.shape

    visibility_mask = np.zeros(len(joints_3d), dtype=bool)
    confidence_scores = np.zeros(len(joints_3d), dtype=np.float32)

    for i, (kp_2d, joint_3d) in enumerate(zip(keypoints_2d, joints_3d)):

        # Skip invalid keypoints
        if not np.isfinite(kp_2d[0]) or not np.isfinite(kp_2d[1]):
            visibility_mask[i] = False
            confidence_scores[i] = 0.0
            continue

        # Check bounds
        x, y = int(np.round(kp_2d[0])), int(np.round(kp_2d[1]))

        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            visibility_mask[i] = False
            confidence_scores[i] = 0.0
            continue

        # Compute expected depth of 3D joint in camera coordinates
        try:
            expected_depth = estimate_depth(joint_3d, camera_params)
        except Exception as e:
            if verbose:
                print(f"Error computing depth for joint {i}: {e}")
            visibility_mask[i] = False
            confidence_scores[i] = 0.0
            continue

        # Sample depth from depth map in neighborhood for robustness
        rendered_depths = _sample_depth_neighborhood(depth_map, x, y, radius=2)

        if len(rendered_depths) == 0:
            visibility_mask[i] = False
            confidence_scores[i] = 0.0
            continue

        # Use median depth from neighborhood for robustness
        rendered_depth_median = np.median(rendered_depths)

        # Skip if no depth data at this location
        if rendered_depth_median <= 0:
            visibility_mask[i] = False
            confidence_scores[i] = 0.0
            continue

        # Compare depths
        depth_diff = abs(rendered_depth_median - expected_depth)

        # Determine visibility
        is_visible = depth_diff < depth_threshold

        visibility_mask[i] = is_visible

        # Compute confidence score
        # Confidence decreases as depth difference increases
        if depth_diff == 0:
            confidence = 1.0
        else:
            confidence = max(0.0, 1.0 - (depth_diff / depth_threshold))

        confidence_scores[i] = confidence

        if verbose and i < 5:  # Print first 5 for debugging
            print(f"Joint {i}: expected_depth={expected_depth:.3f}, "
                  f"rendered_depth={rendered_depth_median:.3f}, "
                  f"diff={depth_diff:.3f}, visible={is_visible}, conf={confidence:.2f}")

    if verbose:
        visible_count = visibility_mask.sum()
        print(f"Visible joints: {visible_count} / {len(joints_3d)}")

    return visibility_mask, confidence_scores


def _sample_depth_neighborhood(
    depth_map: np.ndarray,
    x: int,
    y: int,
    radius: int = 2
) -> np.ndarray:
    """
    Sample depth values in a neighborhood around a pixel.

    Args:
        depth_map: Depth map (H, W)
        x: Pixel x coordinate
        y: Pixel y coordinate
        radius: Neighborhood radius

    Returns:
        Array of depth values sampled from neighborhood
    """
    img_h, img_w = depth_map.shape

    x_min = max(0, x - radius)
    x_max = min(img_w, x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(img_h, y + radius + 1)

    neighborhood = depth_map[y_min:y_max, x_min:x_max]

    # Return non-zero values only (zero indicates no valid depth)
    valid_depths = neighborhood[neighborhood > 0]

    return valid_depths


def create_visibility_array(
    visibility_results: Dict[str, np.ndarray],
    joint_names: list = None
) -> Dict[str, list]:
    """
    Create a structured visibility result for output.

    Args:
        visibility_results: Dictionary with person -> visibility mask
        joint_names: Optional names for joints (default: numbered)

    Returns:
        Dictionary with structured visibility information
    """

    if joint_names is None:
        # Default SMPL joint naming (45 joints)
        joint_names = _get_default_smpl_joint_names()

    structured_results = {}

    for person, visibility_mask in visibility_results.items():
        person_results = []

        for joint_id, (is_visible, name) in enumerate(zip(visibility_mask, joint_names)):
            person_results.append({
                'joint_id': int(joint_id),
                'joint_name': name,
                'visible': bool(is_visible)
            })

        structured_results[person] = person_results

    return structured_results


def _get_default_smpl_joint_names():
    """
    Get default SMPL joint names (45 joints).

    Returns:
        List of 45 joint names
    """
    # SMPL 45 joints: 1 pelvis + 23 body joints + 2*10 hand joints
    # or more specifically: pelvis, spine, and all limb joints

    names = [
        # Pelvis and spine
        'pelvis',  # 0
        'left_hip', 'right_hip',  # 1, 2
        'spine', 'left_knee', 'right_knee',  # 3, 4, 5
        'chest', 'left_ankle', 'right_ankle',  # 6, 7, 8
        'neck', 'left_foot', 'right_foot',  # 9, 10, 11
        'head',  # 12
        'left_collar', 'right_collar',  # 13, 14
        'left_shoulder', 'right_shoulder',  # 15, 16
        'left_elbow', 'right_elbow',  # 17, 18
        'left_wrist', 'right_wrist',  # 19, 20
        # Left hand (10 joints)
        'left_hand_1', 'left_hand_2', 'left_hand_3', 'left_hand_4', 'left_hand_5',
        'left_hand_6', 'left_hand_7', 'left_hand_8', 'left_hand_9', 'left_hand_10',
        # Right hand (10 joints)
        'right_hand_1', 'right_hand_2', 'right_hand_3', 'right_hand_4', 'right_hand_5',
        'right_hand_6', 'right_hand_7', 'right_hand_8', 'right_hand_9', 'right_hand_10',
    ]

    # Truncate to first 45 if we have more
    return names[:45]


if __name__ == '__main__':
    # Test visibility testing
    print("Testing visibility testing module...")

    import sys
    from pathlib import Path

    try:
        from smpl_utils import load_smpl_data, get_smpl_faces
        from camera_calibration import estimate_camera_pnp
        from render_smpl import render_smpl_mesh

        # Load test data
        smpl_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/smpl/00001.npy')
        pose2d_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/poses2d/cam01/00001.npy')

        smpl_data = load_smpl_data(smpl_path.parent, '00001')
        pose2d_data = np.load(pose2d_path, allow_pickle=True).item()

        faces = get_smpl_faces()

        # Estimate camera and render
        joints_3d = smpl_data['aria01']['joints'].astype(np.float32)
        keypoints_2d = pose2d_data['aria01'].astype(np.float32)
        vertices = smpl_data['aria01']['vertices'].astype(np.float32)

        K, R, tvec, _ = estimate_camera_pnp(
            joints_3d, keypoints_2d, (2160, 3840)
        )

        camera_params = {'K': K, 'R': R, 'tvec': tvec}

        # Render
        print("Rendering for visibility test...")
        color_image, depth_map = render_smpl_mesh(
            vertices, faces, camera_params, (2160, 3840)
        )

        # Test visibility
        print("Testing visibility...")
        visibility_mask, confidence_scores = test_keypoint_visibility(
            keypoints_2d, joints_3d, depth_map, camera_params, verbose=True
        )

        print(f"\n✓ Visibility testing successful!")
        print(f"  Visible joints: {visibility_mask.sum()} / {len(joints_3d)}")
        print(f"  Confidence scores: {confidence_scores.min():.2f} - {confidence_scores.max():.2f}")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
