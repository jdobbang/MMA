#!/usr/bin/env python3
"""
Visualization of SMPL rendering and visibility results.

Creates multi-panel visualizations showing original image, rendered mesh,
and visibility-overlaid results.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from pathlib import Path


def create_visualization(
    original_image: np.ndarray,
    rendered_color_aria01: np.ndarray,
    rendered_color_aria02: Optional[np.ndarray],
    depth_map: np.ndarray,
    pose2d_data: Dict[str, np.ndarray],
    visibility_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    joints_3d: Dict[str, np.ndarray] = None,
    camera_params: Dict = None,
    layout: str = 'horizontal'
) -> np.ndarray:
    """
    Create multi-panel visualization of SMPL rendering and visibility.

    Args:
        original_image: Original image (H, W, 3) uint8
        rendered_color_aria01: Rendered mesh for person 1 (H, W, 3)
        rendered_color_aria02: Rendered mesh for person 2 (H, W, 3) or None
        depth_map: Combined depth map (H, W)
        pose2d_data: 2D pose data {'aria01': (45, 2), 'aria02': (45, 2)}
        visibility_results: {'aria01': (visibility_mask, confidence_scores), ...}
        joints_3d: Optional 3D joints for visualization
        camera_params: Optional camera parameters
        layout: 'horizontal' or 'vertical' for panel arrangement

    Returns:
        Combined visualization image
    """

    # Panel 1: Original image
    panel1 = original_image.copy()

    # Panel 2: Rendered mesh
    panel2 = _create_rendered_panel(
        rendered_color_aria01, rendered_color_aria02, panel1.shape[:2]
    )

    # Panel 3: Overlay with visibility
    panel3 = _create_overlay_panel(
        original_image,
        rendered_color_aria01,
        rendered_color_aria02,
        pose2d_data,
        visibility_results
    )

    # Combine panels
    if layout == 'horizontal':
        vis_image = np.concatenate([panel1, panel2, panel3], axis=1)
    else:  # vertical
        vis_image = np.concatenate([panel1, panel2, panel3], axis=0)

    return vis_image


def _create_rendered_panel(
    rendered_color_aria01: np.ndarray,
    rendered_color_aria02: Optional[np.ndarray],
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Create panel showing pure rendered mesh.

    Args:
        rendered_color_aria01: Rendered mesh for person 1
        rendered_color_aria02: Rendered mesh for person 2 or None
        target_shape: Target output shape (H, W)

    Returns:
        Panel image (H, W, 3)
    """
    h, w = target_shape

    # Create black background
    panel = np.zeros((h, w, 3), dtype=np.uint8)

    # Alpha blend both rendered meshes
    if rendered_color_aria02 is not None:
        # Blend both meshes - simple alpha composite
        # Where both have color, use blend; otherwise use whichever is present

        mask1 = (rendered_color_aria01.sum(axis=2) > 0).astype(np.float32)
        mask2 = (rendered_color_aria02.sum(axis=2) > 0).astype(np.float32)

        # Normalize masks
        mask_sum = mask1 + mask2
        mask_sum = np.clip(mask_sum, 1e-6, 1)  # Avoid division by zero

        # Blend
        panel = (
            rendered_color_aria01 * (mask1 / mask_sum)[:, :, None] +
            rendered_color_aria02 * (mask2 / mask_sum)[:, :, None]
        ).astype(np.uint8)
    else:
        # Only person 1
        panel = rendered_color_aria01.copy()

    return panel


def _create_overlay_panel(
    original_image: np.ndarray,
    rendered_color_aria01: np.ndarray,
    rendered_color_aria02: Optional[np.ndarray],
    pose2d_data: Dict[str, np.ndarray],
    visibility_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    mesh_alpha: float = 0.3,
    keypoint_radius: int = 5
) -> np.ndarray:
    """
    Create overlay panel with mesh and visibility-coded keypoints.

    Args:
        original_image: Original image (H, W, 3)
        rendered_color_aria01: Rendered mesh for person 1
        rendered_color_aria02: Rendered mesh for person 2 or None
        pose2d_data: 2D pose data
        visibility_results: Visibility masks and confidences
        mesh_alpha: Alpha blending for mesh (0-1)
        keypoint_radius: Radius of keypoint circles

    Returns:
        Overlay panel (H, W, 3)
    """
    panel = original_image.copy().astype(np.float32)

    # Alpha blend rendered meshes
    for person, rendered_color in [
        ('aria01', rendered_color_aria01),
        ('aria02', rendered_color_aria02)
    ]:
        if rendered_color is None:
            continue

        mask = (rendered_color.sum(axis=2) > 0).astype(np.float32)
        rendered_color_f = rendered_color.astype(np.float32)

        panel = (
            panel * (1 - mesh_alpha * mask[:, :, None]) +
            rendered_color_f * (mesh_alpha * mask[:, :, None])
        )

    panel = np.clip(panel, 0, 255).astype(np.uint8)

    # Draw keypoints and skeleton for each person
    for person in ['aria01', 'aria02']:
        if person not in pose2d_data:
            continue

        keypoints_2d = pose2d_data[person]
        visibility_mask, confidence_scores = visibility_results[person]

        # Color for this person
        if person == 'aria01':
            color_visible = (0, 255, 0)  # Green for visible
            color_occluded = (0, 0, 255)  # Red for occluded
        else:
            color_visible = (200, 255, 0)  # Cyan-ish for visible
            color_occluded = (0, 165, 255)  # Orange for occluded

        # Draw skeleton first (only visible joints)
        panel = _draw_skeleton(panel, keypoints_2d, visibility_mask)

        # Draw keypoints
        for joint_id, (kp_2d, is_visible, conf) in enumerate(
            zip(keypoints_2d, visibility_mask, confidence_scores)
        ):
            x, y = int(np.round(kp_2d[0])), int(np.round(kp_2d[1]))

            # Check bounds
            if x < 0 or x >= panel.shape[1] or y < 0 or y >= panel.shape[0]:
                continue

            if is_visible:
                # Green circle for visible
                cv2.circle(panel, (x, y), keypoint_radius, color_visible, -1)
                cv2.circle(panel, (x, y), keypoint_radius, (255, 255, 255), 1)
            else:
                # Red X for occluded
                size = keypoint_radius
                cv2.line(panel, (x - size, y - size), (x + size, y + size),
                        color_occluded, 2)
                cv2.line(panel, (x - size, y + size), (x + size, y - size),
                        color_occluded, 2)

    return panel


def _draw_skeleton(
    image: np.ndarray,
    keypoints_2d: np.ndarray,
    visibility_mask: np.ndarray,
    skeleton_pairs: Optional[list] = None,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw skeleton connections on image.

    Args:
        image: Input image (H, W, 3)
        keypoints_2d: 2D keypoint positions (N, 2)
        visibility_mask: Visibility mask (N,) boolean
        skeleton_pairs: List of (joint_i, joint_j) pairs to connect
        color: Line color BGR
        thickness: Line thickness

    Returns:
        Image with drawn skeleton
    """
    if skeleton_pairs is None:
        # Default SMPL skeleton connections (common major joints)
        skeleton_pairs = _get_default_skeleton_pairs()

    for joint_i, joint_j in skeleton_pairs:
        # Only draw if both joints are visible
        if joint_i >= len(visibility_mask) or joint_j >= len(visibility_mask):
            continue

        if not (visibility_mask[joint_i] and visibility_mask[joint_j]):
            continue

        x1, y1 = keypoints_2d[joint_i]
        x2, y2 = keypoints_2d[joint_j]

        x1, y1 = int(np.round(x1)), int(np.round(y1))
        x2, y2 = int(np.round(x2)), int(np.round(y2))

        # Check bounds
        if (0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
            0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]):
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    return image


def _get_default_skeleton_pairs() -> list:
    """
    Get default SMPL skeleton connections.

    Returns:
        List of (joint_i, joint_j) pairs
    """
    # Major body skeleton for SMPL (simplified)
    # Pelvis, spine, limbs
    pairs = [
        # Spine
        (0, 3), (3, 6), (6, 9), (9, 12),  # pelvis -> neck -> head
        # Left arm
        (13, 15), (15, 17), (17, 19),  # left collar -> shoulder -> elbow -> wrist
        # Right arm
        (14, 16), (16, 18), (18, 20),  # right collar -> shoulder -> elbow -> wrist
        # Left leg
        (0, 1), (1, 4), (4, 7), (7, 10),  # pelvis -> hip -> knee -> ankle -> foot
        # Right leg
        (0, 2), (2, 5), (5, 8), (8, 11),  # pelvis -> hip -> knee -> ankle -> foot
    ]
    return pairs


def save_visualization(
    vis_image: np.ndarray,
    output_path: str,
    quality: int = 95
) -> None:
    """
    Save visualization image.

    Args:
        vis_image: Image to save (H, W, 3) in RGB
        output_path: Output file path
        quality: JPEG quality (0-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    # Save with specified quality
    cv2.imwrite(str(output_path), vis_image_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, quality])

    print(f"Saved visualization to {output_path}")


if __name__ == '__main__':
    # Test visualization module
    print("Testing visualization module...")

    import sys

    try:
        from pathlib import Path
        from smpl_utils import load_smpl_data, get_smpl_faces
        from camera_calibration import estimate_camera_pnp
        from render_smpl import render_smpl_mesh
        from visibility_test import test_keypoint_visibility
        import cv2

        # Load test data
        smpl_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/smpl/00001.npy')
        pose2d_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/poses2d/cam01/00001.npy')
        img_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/exo/cam01/images/00001.jpg')

        print("Loading data...")
        smpl_data = load_smpl_data(smpl_path.parent, '00001')
        pose2d_data = np.load(pose2d_path, allow_pickle=True).item()
        original_image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        print("Processing person 1 (aria01)...")
        faces = get_smpl_faces()
        joints_3d = smpl_data['aria01']['joints'].astype(np.float32)
        keypoints_2d = pose2d_data['aria01'].astype(np.float32)
        vertices = smpl_data['aria01']['vertices'].astype(np.float32)

        K, R, tvec, _ = estimate_camera_pnp(
            joints_3d, keypoints_2d, original_image.shape[:2]
        )
        camera_params = {'K': K, 'R': R, 'tvec': tvec}

        color1, depth1 = render_smpl_mesh(
            vertices, faces, camera_params, original_image.shape[:2],
            color=(1.0, 0.0, 0.0)  # Red for person 1
        )

        visibility1, conf1 = test_keypoint_visibility(
            keypoints_2d, joints_3d, depth1, camera_params
        )

        # Create visualization
        print("Creating visualization...")
        vis_image = create_visualization(
            original_image, color1, None,
            depth1,
            {'aria01': keypoints_2d},
            {'aria01': (visibility1, conf1)}
        )

        print(f"✓ Visualization created: {vis_image.shape}")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
