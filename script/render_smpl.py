#!/usr/bin/env python3
"""
SMPL mesh rendering using pyrender and trimesh.

Renders SMPL body mesh onto 2D images with proper 3D projection and depth.
"""

import os
import numpy as np
import cv2
from typing import Tuple, Dict
import warnings

# Configure headless rendering BEFORE importing pyrender
# Try EGL first, fall back to osmesa
try:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
except:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import trimesh
import pyrender

# Suppress pyrender/OpenGL warnings
warnings.filterwarnings('ignore')


def render_smpl_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_params: Dict[str, np.ndarray],
    img_shape: Tuple[int, int],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.8,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render SMPL mesh to 2D image using OpenCV vertex projection.

    Args:
        vertices: SMPL vertices (6890, 3) in world coordinates
        faces: SMPL face topology (13776, 3)
        camera_params: Dictionary with 'K', 'R', 'tvec'
        img_shape: Image shape (height, width)
        color: RGB color for mesh (range 0-1)
        alpha: Transparency (ignored in OpenCV rendering)
        verbose: Print debug info

    Returns:
        Tuple of (color_image, depth_map):
        - color_image: (H, W, 3) uint8 RGB image with projected vertices
        - depth_map: (H, W) float32 depth values from projected vertices
    """

    img_h, img_w = img_shape

    try:
        # Create blank output image
        color_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        depth_map = np.zeros((img_h, img_w), dtype=np.float32)

        # Extract camera parameters
        K = camera_params['K']
        R = camera_params['R']
        tvec = camera_params['tvec']
        if tvec.shape == (3, 1):
            tvec = tvec.flatten()

        # Project vertices to 2D
        P_cam = (R @ vertices.T).T + tvec

        # Convert color from 0-1 to 0-255 BGR
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))  # RGB to BGR

        # Draw projected vertices
        for i, p in enumerate(P_cam):
            if p[2] > 0:  # In front of camera
                px = K[0, 0] * p[0] / p[2] + K[0, 2]
                py = K[1, 1] * p[1] / p[2] + K[1, 2]

                # Check bounds
                px_int, py_int = int(np.round(px)), int(np.round(py))
                if 0 <= px_int < img_w and 0 <= py_int < img_h:
                    # Draw vertex as small circle
                    cv2.circle(color_image, (px_int, py_int), 2, color_bgr, -1)
                    # Store depth
                    if depth_map[py_int, px_int] == 0 or p[2] < depth_map[py_int, px_int]:
                        depth_map[py_int, px_int] = p[2]

        if verbose:
            print(f"Rendering complete: {img_h}x{img_w}")
            print(f"  Color range: {color_image.min()}-{color_image.max()}")
            print(f"  Depth range: {depth_map[depth_map > 0].min():.3f}-{depth_map.max():.3f}" if depth_map.max() > 0 else "  Depth: all zeros")

        return color_image, depth_map

    except Exception as e:
        print(f"Error in render_smpl_mesh: {e}")
        raise


def _create_trimesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    color: Tuple[float, float, float],
    alpha: float
) -> trimesh.Trimesh:
    """
    Create trimesh from SMPL vertices and faces.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        color: RGB color tuple (range 0-1)
        alpha: Transparency (range 0-1)

    Returns:
        trimesh.Trimesh object
    """
    # Ensure correct data types
    vertices = vertices.astype(np.float64)
    faces = faces.astype(np.uint32)

    # Create mesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,  # Don't process (slow, not needed for visualization)
        validate=False   # Skip validation
    )

    # Set vertex colors (convert from 0-1 to 0-255)
    color_8bit = np.array(color[:3]) * 255
    color_rgba = np.array([color_8bit[0], color_8bit[1], color_8bit[2], int(alpha * 255)], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=color_rgba)

    return mesh


def _setup_scene(
    mesh: trimesh.Trimesh,
    camera_params: Dict[str, np.ndarray],
    img_shape: Tuple[int, int],
    verbose: bool = False
) -> pyrender.Scene:
    """
    Setup pyrender scene with mesh and camera.

    Args:
        mesh: trimesh.Trimesh object
        camera_params: Dictionary with camera parameters
        img_shape: Image shape (height, width)
        verbose: Print debug info

    Returns:
        pyrender.Scene configured for rendering
    """

    # Create scene with minimal ambient light to show mesh color
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])

    # Add mesh to scene
    # Convert to pyrender mesh with material
    # Extract color from trimesh visual
    if hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = mesh.visual.vertex_colors[0]  # Get first vertex color
    else:
        # Default to white if no colors set
        vertex_colors = np.array([255, 255, 255, 200], dtype=np.uint8)

    # Create material from vertex colors
    # baseColorFactor must be 1D array
    base_color = (vertex_colors.astype(np.float32) / 255.0).flatten()
    if len(base_color) < 4:
        base_color = np.append(base_color, 1.0)  # Add alpha if missing

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=base_color[:4],
        metallicFactor=0.2,
        roughnessFactor=0.8
    )
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(pyrender_mesh)

    # Add camera
    camera = _create_pyrender_camera(camera_params, img_shape)
    camera_pose = _compute_camera_pose(camera_params)
    scene.add(camera, pose=camera_pose)

    # Add directional light from camera direction
    light = pyrender.DirectionalLight(
        color=np.array([1.0, 1.0, 1.0]),
        intensity=2.0
    )
    scene.add(light, pose=camera_pose)

    # Add ambient light for better visibility
    light2 = pyrender.DirectionalLight(
        color=np.array([0.5, 0.5, 0.5]),
        intensity=0.5
    )
    # Light from back
    back_pose = np.eye(4)
    back_pose[:3, 3] = [0, 0, -5]
    scene.add(light2, pose=back_pose)

    return scene


def _create_pyrender_camera(
    camera_params: Dict[str, np.ndarray],
    img_shape: Tuple[int, int]
) -> pyrender.Camera:
    """
    Create pyrender camera from estimated parameters.

    Args:
        camera_params: Dictionary with 'K', 'R', 'tvec'
        img_shape: Image shape (height, width)

    Returns:
        pyrender.IntrinsicsCamera
    """
    K = camera_params['K']

    img_h, img_w = img_shape

    # Extract intrinsics from K matrix
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Create camera
    camera = pyrender.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        znear=0.01,
        zfar=1000.0
    )

    return camera


def _compute_camera_pose(camera_params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute camera pose matrix from rotation and translation.

    Args:
        camera_params: Dictionary with 'R', 'tvec'

    Returns:
        4x4 pose matrix (world-to-camera transformation)
    """
    R = camera_params['R']
    tvec = camera_params['tvec']

    if tvec.shape == (3, 1):
        tvec = tvec.flatten()

    # PnP returns R and t that transform world points to camera coordinates
    # For pyrender, we need the camera pose (camera in world coordinates)
    # This is the inverse of the PnP transformation

    # Create 4x4 pose matrix for world-to-camera
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = tvec

    # Inverse to get camera position in world coordinates
    pose_inv = np.linalg.inv(pose)

    return pose_inv


def _render_scene(
    scene: pyrender.Scene,
    img_shape: Tuple[int, int],
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render scene and get color and depth.

    Args:
        scene: pyrender.Scene
        img_shape: Image shape (height, width)
        verbose: Print debug info

    Returns:
        Tuple of (color_image, depth_map):
        - color_image: (H, W, 3) uint8 RGB image
        - depth_map: (H, W) float32 depth values
    """

    img_h, img_w = img_shape

    # Create renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=img_w,
        viewport_height=img_h
    )

    try:
        # Render
        color, depth = renderer.render(scene)

        # Convert color from RGBA to RGB and to uint8
        if color.shape[2] == 4:
            color = color[:, :, :3]

        color = (color * 255).astype(np.uint8)

        # Depth is already in the correct format
        # Values of 0 indicate "no depth" (outside frustum)
        # Other values are actual depth in camera coordinates

        if verbose:
            print(f"Renderer output: color {color.shape}, depth {depth.shape}")

        return color, depth

    finally:
        renderer.delete()


def composite_depth_maps(
    depth_map1: np.ndarray,
    depth_map2: np.ndarray,
    color_map1: np.ndarray = None,
    color_map2: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Composite two depth maps, keeping the closer (smaller) depth at each pixel.

    Args:
        depth_map1: First depth map (H, W)
        depth_map2: Second depth map (H, W)
        color_map1: Optional color map for person 1 (H, W, 3)
        color_map2: Optional color map for person 2 (H, W, 3)

    Returns:
        Tuple of (composite_depth, composite_color):
        - composite_depth: Combined depth map (H, W)
        - composite_color: Combined color (H, W, 3) if colors provided, else None
    """

    # Create mask of which depth is closer at each pixel
    # Handle zero values (no depth) - treat as very far away
    depth1_valid = depth_map1 > 0
    depth2_valid = depth_map2 > 0

    composite_depth = np.zeros_like(depth_map1)
    composite_color = None

    if color_map1 is not None and color_map2 is not None:
        composite_color = np.zeros_like(color_map1)

    # Pixels with only one valid depth
    only1_valid = depth1_valid & ~depth2_valid
    only2_valid = ~depth1_valid & depth2_valid
    both_valid = depth1_valid & depth2_valid

    # One valid
    composite_depth[only1_valid] = depth_map1[only1_valid]
    if composite_color is not None:
        composite_color[only1_valid] = color_map1[only1_valid]

    composite_depth[only2_valid] = depth_map2[only2_valid]
    if composite_color is not None:
        composite_color[only2_valid] = color_map2[only2_valid]

    # Both valid - keep closer
    closer_mask = depth_map1[both_valid] < depth_map2[both_valid]
    composite_depth[both_valid] = np.where(
        closer_mask,
        depth_map1[both_valid],
        depth_map2[both_valid]
    )

    if composite_color is not None:
        composite_color[both_valid] = np.where(
            closer_mask[:, None],
            color_map1[both_valid],
            color_map2[both_valid]
        )

    return composite_depth, composite_color


if __name__ == '__main__':
    # Test rendering
    print("Testing SMPL rendering...")
    print("Note: This requires SMPL face topology to be available")

    from pathlib import Path
    import sys

    try:
        from smpl_utils import load_smpl_data, get_smpl_faces
        from camera_calibration import estimate_camera_pnp
        import cv2

        # Load test data
        smpl_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/smpl/00001.npy')
        pose2d_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/poses2d/cam01/00001.npy')

        print("Loading test data...")
        smpl_data = load_smpl_data(smpl_path.parent, '00001')
        pose2d_data = np.load(pose2d_path, allow_pickle=True).item()

        # Get SMPL faces
        print("Loading SMPL faces...")
        faces = get_smpl_faces()

        # Estimate camera
        print("Estimating camera...")
        joints_3d = smpl_data['aria01']['joints'].astype(np.float32)
        keypoints_2d = pose2d_data['aria01'].astype(np.float32)

        K, R, tvec, error = estimate_camera_pnp(
            joints_3d, keypoints_2d, (2160, 3840), verbose=True
        )

        camera_params = {
            'K': K, 'R': R, 'tvec': tvec,
            'extrinsic': np.hstack([R, tvec.reshape(3, 1)])
        }

        # Render mesh
        print("Rendering SMPL mesh...")
        vertices = smpl_data['aria01']['vertices'].astype(np.float32)

        color_image, depth_map = render_smpl_mesh(
            vertices, faces, camera_params, (2160, 3840),
            color=(1.0, 0.0, 0.0), verbose=True
        )

        print(f"✓ Rendering successful!")
        print(f"  Color image: {color_image.shape}, dtype {color_image.dtype}")
        print(f"  Depth map: {depth_map.shape}, dtype {depth_map.dtype}")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
