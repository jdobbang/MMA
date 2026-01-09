#!/usr/bin/env python3
"""
Enhanced SMPL mesh rendering with dense face rasterization.

This module provides improved rendering over the basic vertex projection approach
by rasterizing triangular faces to create dense mesh coverage and accurate depth maps.
"""

import numpy as np
import cv2
from typing import Dict, Tuple


def point_in_triangle(p: Tuple[float, float], triangle: np.ndarray) -> bool:
    """
    Check if point p is inside triangle using cross products.

    Args:
        p: Point (x, y)
        triangle: Triangle vertices (3, 2) array

    Returns:
        True if point is inside triangle
    """
    x, y = p
    x0, y0 = triangle[0]
    x1, y1 = triangle[1]
    x2, y2 = triangle[2]

    # Compute barycentric coordinates using cross products
    denom = ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
    if abs(denom) < 1e-10:
        return False

    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    c = 1 - a - b

    return a >= 0 and b >= 0 and c >= 0


def barycentric_coords(p: Tuple[float, float], triangle: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates of point p in triangle.

    Args:
        p: Point (x, y)
        triangle: Triangle vertices (3, 2) array

    Returns:
        Barycentric coordinates (a, b, c) where a + b + c = 1
    """
    x, y = p
    x0, y0 = triangle[0]
    x1, y1 = triangle[1]
    x2, y2 = triangle[2]

    denom = ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
    if abs(denom) < 1e-10:
        return np.array([1.0, 0.0, 0.0])

    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    c = 1 - a - b

    return np.array([a, b, c])


def render_smpl_mesh_enhanced(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_params: Dict[str, np.ndarray],
    img_shape: Tuple[int, int],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 1.0,
    use_face_rendering: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render SMPL mesh with dense triangular face rasterization.

    Args:
        vertices: SMPL mesh vertices (N, 3) in world coordinates
        faces: Triangle face indices (M, 3)
        camera_params: Dictionary with 'K', 'R', 'tvec'
        img_shape: Output image shape (height, width)
        color: RGB color tuple (0-1 range)
        alpha: Opacity (0-1)
        use_face_rendering: If True, render full triangular faces; if False, use vertex-only
        verbose: Print debug info

    Returns:
        color_image: (H, W, 3) uint8 RGB image with dense mesh rendering
        depth_map: (H, W) float32 depth values in meters
    """
    img_h, img_w = img_shape
    color_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    depth_map = np.full((img_h, img_w), np.inf, dtype=np.float32)

    K = camera_params['K']
    R = camera_params['R']
    tvec = camera_params['tvec'].flatten()

    # Convert color to BGR for OpenCV
    color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

    if verbose:
        print(f"  Rendering {len(vertices)} vertices, {len(faces)} faces")
        print(f"  Image shape: {img_shape}")

    # 1. Project all vertices to 2D
    P_cam = (R @ vertices.T).T + tvec
    proj_2d = []
    depths = []

    for p in P_cam:
        if p[2] > 0.01:  # Valid depth (in front of camera)
            px = K[0, 0] * p[0] / p[2] + K[0, 2]
            py = K[1, 1] * p[1] / p[2] + K[1, 2]
            proj_2d.append([px, py])
            depths.append(p[2])
        else:
            proj_2d.append([-1, -1])
            depths.append(-1)

    proj_2d = np.array(proj_2d)
    depths = np.array(depths)

    if use_face_rendering:
        # 2. Rasterize each triangular face
        face_count = 0

        for face in faces:
            v0, v1, v2 = face

            # Skip if any vertex behind camera
            if depths[v0] < 0 or depths[v1] < 0 or depths[v2] < 0:
                continue

            # Get 2D triangle vertices
            pts = np.array([proj_2d[v0], proj_2d[v1], proj_2d[v2]], dtype=np.float32)

            # Check if triangle is within image bounds (with margin)
            if (pts[:, 0].max() < 0 or pts[:, 0].min() >= img_w or
                pts[:, 1].max() < 0 or pts[:, 1].min() >= img_h):
                continue

            # Bounding box
            min_x = max(0, int(np.floor(pts[:, 0].min())))
            max_x = min(img_w - 1, int(np.ceil(pts[:, 0].max())))
            min_y = max(0, int(np.floor(pts[:, 1].min())))
            max_y = min(img_h - 1, int(np.ceil(pts[:, 1].max())))

            # Rasterize pixels in bounding box
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if point_in_triangle((x, y), pts):
                        # Interpolate depth using barycentric coordinates
                        bc = barycentric_coords((x, y), pts)
                        depth = bc[0] * depths[v0] + bc[1] * depths[v1] + bc[2] * depths[v2]

                        # Z-buffer check (only draw if closer)
                        if depth < depth_map[y, x]:
                            depth_map[y, x] = depth
                            color_image[y, x] = color_bgr

            face_count += 1

        if verbose:
            print(f"  Rendered {face_count} faces")

    else:
        # Fallback: render vertices only (like original method)
        vertex_count = 0

        for i, (pt_2d, depth) in enumerate(zip(proj_2d, depths)):
            if depth < 0:
                continue

            px_int = int(np.round(pt_2d[0]))
            py_int = int(np.round(pt_2d[1]))

            if 0 <= px_int < img_w and 0 <= py_int < img_h:
                # Draw larger circles for better coverage
                cv2.circle(color_image, (px_int, py_int), 4, color_bgr, -1)

                # Update depth map in neighborhood
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        nx, ny = px_int + dx, py_int + dy
                        if 0 <= nx < img_w and 0 <= ny < img_h:
                            if depth < depth_map[ny, nx]:
                                depth_map[ny, nx] = depth

                vertex_count += 1

        if verbose:
            print(f"  Rendered {vertex_count} vertices")

    # Replace inf with 0 in depth map
    depth_map[depth_map == np.inf] = 0.0

    if verbose:
        valid_depth_pixels = (depth_map > 0).sum()
        print(f"  Valid depth pixels: {valid_depth_pixels} / {img_h * img_w}")
        if valid_depth_pixels > 0:
            print(f"  Depth range: {depth_map[depth_map > 0].min():.3f} - {depth_map[depth_map > 0].max():.3f} m")

    return color_image, depth_map


if __name__ == '__main__':
    print("Testing enhanced SMPL rendering...")

    import sys
    from pathlib import Path

    try:
        from smpl_utils import load_smpl_data, get_smpl_faces
        from camera_calibration import estimate_camera_pnp, create_camera_pose

        # Load test data
        smpl_path = Path('/workspace/MMA/dataset/13_mma2/001_mma2/processed_data/smpl')
        pose2d_path = Path('/workspace/MMA/dataset/13_mma2/001_mma2/processed_data/poses2d/cam01')

        smpl_data = load_smpl_data(smpl_path, '00001')
        pose2d_data = np.load(pose2d_path / '00001.npy', allow_pickle=True).item()

        faces = get_smpl_faces()

        # Camera calibration
        joints_3d = smpl_data['aria01']['joints'].astype(np.float32)
        keypoints_2d = pose2d_data['aria01'].astype(np.float32)
        vertices = smpl_data['aria01']['vertices'].astype(np.float32)

        K, R, tvec, error = estimate_camera_pnp(joints_3d, keypoints_2d, (2160, 3840))
        camera_params = create_camera_pose(K, R, tvec)

        print(f"\nCalibration RMSE: {error:.2f} px")

        # Test face rendering
        print("\nTesting face rendering...")
        color_img, depth_map = render_smpl_mesh_enhanced(
            vertices, faces, camera_params, (2160, 3840),
            color=(1.0, 0.0, 0.0), use_face_rendering=True, verbose=True
        )

        # Test vertex-only rendering
        print("\nTesting vertex-only rendering...")
        color_img_v, depth_map_v = render_smpl_mesh_enhanced(
            vertices, faces, camera_params, (2160, 3840),
            color=(0.0, 0.0, 1.0), use_face_rendering=False, verbose=True
        )

        print("\n✓ Enhanced rendering test successful!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
