#!/usr/bin/env python3
"""
Camera calibration from 3D-2D point correspondences using PnP (Perspective-n-Point).

This module estimates camera intrinsic and extrinsic parameters from 3D joints
(from SMPL) and their 2D projections (from pose2d data) using RANSAC-based PnP.
"""

import numpy as np
import cv2
from typing import Tuple, Dict


def estimate_camera_pnp(
    joints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
    img_shape: Tuple[int, int],
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Estimate camera parameters using Perspective-n-Point (PnP) with RANSAC.

    Args:
        joints_3d: 3D joint positions from SMPL, shape (N, 3)
        keypoints_2d: 2D keypoint positions, shape (N, 2) in pixel coordinates
        img_shape: Image shape (height, width)
        verbose: Print debug information

    Returns:
        Tuple containing:
        - K: Camera intrinsic matrix (3, 3)
        - R: Rotation matrix (3, 3)
        - t: Translation vector (3,)
        - reprojection_error: RMSE of reprojection error in pixels
    """

    img_h, img_w = img_shape

    # 1. Filter valid correspondences
    valid_mask = _filter_valid_correspondences(keypoints_2d, img_shape)

    if valid_mask.sum() < 6:
        raise ValueError(f"Not enough valid correspondences: {valid_mask.sum()} < 6")

    joints_3d_valid = joints_3d[valid_mask].astype(np.float32)
    keypoints_2d_valid = keypoints_2d[valid_mask].astype(np.float32)

    if verbose:
        print(f"Valid correspondences: {len(joints_3d_valid)} / {len(joints_3d)}")

    # 2. Initialize intrinsic matrix
    K = _initialize_intrinsics(img_h, img_w)

    if verbose:
        print(f"Initial intrinsics: fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}, "
              f"cx={K[0, 2]:.1f}, cy={K[1, 2]:.1f}")

    # 3. Solve PnP with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=joints_3d_valid,
        imagePoints=keypoints_2d_valid,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=10.0,  # 10 pixel threshold for inlier
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success or rvec is None:
        raise RuntimeError("PnP RANSAC failed to find a solution")

    # 4. Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # 5. Refine calibration using inliers
    if inliers is not None and len(inliers) > 6:
        inlier_indices = inliers.flatten()
        joints_inliers = joints_3d_valid[inlier_indices]
        keypoints_inliers = keypoints_2d_valid[inlier_indices]

        # Refine using only inliers with Levenberg-Marquardt
        success_ref, rvec_ref, tvec_ref = cv2.solvePnP(
            objectPoints=joints_inliers,
            imagePoints=keypoints_inliers,
            cameraMatrix=K,
            distCoeffs=None,
            useExtrinsicGuess=True,
            rvec=rvec,
            tvec=tvec,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success_ref:
            rvec = rvec_ref
            tvec = tvec_ref
            R, _ = cv2.Rodrigues(rvec)

    # 6. Compute reprojection error
    reprojection_error = _compute_reprojection_error(
        joints_3d_valid, keypoints_2d_valid, K, R, tvec
    )

    if verbose:
        print(f"Reprojection RMSE: {reprojection_error:.2f} pixels")
        print(f"Camera translation: {tvec.flatten()}")

    return K, R, tvec, reprojection_error


def _filter_valid_correspondences(
    keypoints_2d: np.ndarray,
    img_shape: Tuple[int, int],
    margin: int = 10
) -> np.ndarray:
    """
    Filter keypoints that are within image bounds.

    Args:
        keypoints_2d: 2D keypoint positions (N, 2)
        img_shape: Image shape (height, width)
        margin: Margin from image edges

    Returns:
        Boolean mask of valid keypoints
    """
    img_h, img_w = img_shape

    # Check bounds
    valid_x = (keypoints_2d[:, 0] >= margin) & (keypoints_2d[:, 0] < img_w - margin)
    valid_y = (keypoints_2d[:, 1] >= margin) & (keypoints_2d[:, 1] < img_h - margin)

    # Check for NaN or infinity
    valid_finite = np.isfinite(keypoints_2d[:, 0]) & np.isfinite(keypoints_2d[:, 1])

    return valid_x & valid_y & valid_finite


def _initialize_intrinsics(
    img_h: int,
    img_w: int,
    focal_length_scale: float = 1.0
) -> np.ndarray:
    """
    Initialize camera intrinsic matrix.

    Args:
        img_h: Image height
        img_w: Image width
        focal_length_scale: Scale factor for focal length (default: 1.0)

    Returns:
        Camera intrinsic matrix K (3, 3)
    """
    # Assume focal length proportional to image size
    # Reasonable initial value: max(H, W) or similar
    focal_length = max(img_h, img_w) * focal_length_scale

    # Principal point at image center
    cx = img_w / 2.0
    cy = img_h / 2.0

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


def _compute_reprojection_error(
    joints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray
) -> float:
    """
    Compute reprojection error as RMSE.

    Args:
        joints_3d: 3D points (N, 3)
        keypoints_2d: 2D points (N, 2)
        K: Intrinsic matrix (3, 3)
        R: Rotation matrix (3, 3)
        tvec: Translation vector (3,) or (3, 1)

    Returns:
        RMSE reprojection error in pixels
    """
    if tvec.shape == (3, 1):
        tvec = tvec.flatten()

    # Project 3D points to 2D
    P = K @ np.hstack([R, tvec.reshape(3, 1)])
    joints_3d_h = np.hstack([joints_3d, np.ones((len(joints_3d), 1))])
    projected_2d = (P @ joints_3d_h.T).T
    projected_2d = projected_2d[:, :2] / projected_2d[:, 2:3]

    # Compute error
    errors = np.linalg.norm(projected_2d - keypoints_2d, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    return rmse


def create_camera_pose(
    K: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Create a camera pose dictionary.

    Args:
        K: Intrinsic matrix (3, 3)
        R: Rotation matrix (3, 3)
        tvec: Translation vector (3,)

    Returns:
        Dictionary with 'K', 'R', 'tvec' and additional derived parameters
    """
    if tvec.shape == (3, 1):
        tvec = tvec.flatten()

    # Compute extrinsic matrix [R | t]
    extrinsic = np.hstack([R, tvec.reshape(3, 1)])

    # Compute camera center in world coordinates
    camera_center = -R.T @ tvec

    return {
        'K': K,
        'R': R,
        'tvec': tvec,
        'extrinsic': extrinsic,
        'camera_center': camera_center,
        'P': K @ extrinsic  # Projection matrix
    }


def project_3d_to_2d(
    points_3d: np.ndarray,
    camera_params: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Project 3D points to 2D image plane.

    Args:
        points_3d: 3D points (N, 3)
        camera_params: Dictionary with 'K', 'R', 'tvec'

    Returns:
        2D projected points (N, 2) in pixel coordinates
    """
    K = camera_params['K']
    R = camera_params['R']
    tvec = camera_params['tvec']

    if tvec.shape == (3, 1):
        tvec = tvec.flatten()

    # Project
    P = K @ np.hstack([R, tvec.reshape(3, 1)])
    points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    projected = (P @ points_3d_h.T).T
    projected_2d = projected[:, :2] / projected[:, 2:3]

    return projected_2d


def estimate_depth(
    point_3d: np.ndarray,
    camera_params: Dict[str, np.ndarray]
) -> float:
    """
    Compute depth of a 3D point in camera coordinates.

    Args:
        point_3d: 3D point in world coordinates (3,)
        camera_params: Dictionary with 'R', 'tvec'

    Returns:
        Depth (z-coordinate in camera frame)
    """
    R = camera_params['R']
    tvec = camera_params['tvec']

    if tvec.shape == (3, 1):
        tvec = tvec.flatten()

    # Transform to camera coordinates
    point_cam = R @ point_3d + tvec

    return point_cam[2]


if __name__ == '__main__':
    # Test camera calibration
    import sys
    from pathlib import Path

    print("Testing camera calibration...")

    # Load test data
    test_smpl_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/smpl/00001.npy')
    test_pose2d_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/poses2d/cam01/00001.npy')

    if not test_smpl_path.exists() or not test_pose2d_path.exists():
        print("Test data not found")
        sys.exit(1)

    # Load SMPL joints and 2D keypoints
    smpl_data = np.load(test_smpl_path, allow_pickle=True).item()
    pose2d_data = np.load(test_pose2d_path, allow_pickle=True).item()

    joints_3d = smpl_data['aria01']['joints'].astype(np.float32)
    keypoints_2d = pose2d_data['aria01'].astype(np.float32)

    print(f"3D joints shape: {joints_3d.shape}")
    print(f"2D keypoints shape: {keypoints_2d.shape}")

    # Estimate camera
    try:
        K, R, tvec, error = estimate_camera_pnp(
            joints_3d, keypoints_2d, (2160, 3840), verbose=True
        )
        print(f"\n✓ Successfully estimated camera")
        print(f"  Intrinsic K:\n{K}")
        print(f"  Translation: {tvec.flatten()}")
        print(f"  Reprojection error: {error:.2f} px")
    except Exception as e:
        print(f"✗ Camera calibration failed: {e}")
        sys.exit(1)
