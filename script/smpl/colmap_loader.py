#!/usr/bin/env python3
"""
COLMAP camera parameters loader.

Loads camera intrinsics and extrinsics from COLMAP reconstruction output.
COLMAP (https://colmap.github.io/) provides accurate camera calibration
through structure-from-motion.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class COLMAPLoader:
    """Load COLMAP camera parameters and poses."""

    def __init__(self, colmap_dir: Path):
        """
        Initialize COLMAP loader.

        Args:
            colmap_dir: Path to COLMAP output directory containing cameras.txt and images.txt
        """
        self.colmap_dir = Path(colmap_dir)
        self.cameras_file = self.colmap_dir / 'cameras.txt'
        self.images_file = self.colmap_dir / 'images.txt'

        if not self.cameras_file.exists() or not self.images_file.exists():
            raise FileNotFoundError(
                f"COLMAP files not found in {self.colmap_dir}. "
                "Need cameras.txt and images.txt"
            )

        self.cameras = self._load_cameras()
        self.images = self._load_images()

    def _load_cameras(self) -> Dict[int, Dict]:
        """
        Load camera parameters from cameras.txt.

        Returns:
            Dictionary mapping camera_id to camera parameters
        """
        cameras = {}

        with open(self.cameras_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]

                cameras[camera_id] = {
                    'id': camera_id,
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': np.array(params, dtype=np.float64)
                }

        logger.info(f"Loaded {len(cameras)} camera(s) from COLMAP")
        return cameras

    def _load_images(self) -> Dict[str, Dict]:
        """
        Load image poses from images.txt.

        Returns:
            Dictionary mapping image filename to pose parameters
        """
        images = {}
        i = 0

        with open(self.images_file, 'r') as f:
            lines = f.readlines()

        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            image_id = int(parts[0])
            qw, qx, qy, qz = [float(p) for p in parts[1:5]]
            tx, ty, tz = [float(p) for p in parts[5:8]]
            camera_id = int(parts[8])
            image_name = parts[9]

            # Skip the points2D line
            if i < len(lines):
                i += 1

            images[image_name] = {
                'id': image_id,
                'quat': np.array([qw, qx, qy, qz], dtype=np.float64),
                'tvec': np.array([tx, ty, tz], dtype=np.float64),
                'camera_id': camera_id
            }

        logger.info(f"Loaded {len(images)} image pose(s) from COLMAP")
        return images

    def get_camera_intrinsics(self, camera_id: int) -> np.ndarray:
        """
        Get camera intrinsic matrix for given camera.

        Args:
            camera_id: Camera ID from cameras.txt

        Returns:
            3x3 intrinsic matrix K
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not found")

        camera = self.cameras[camera_id]
        model = camera['model']
        params = camera['params']

        if model == 'OPENCV_FISHEYE':
            # OPENCV_FISHEYE format: fx, fy, cx, cy, k1, k2, k3, k4
            fx, fy, cx, cy = params[:4]

        elif model == 'PINHOLE':
            # PINHOLE format: fx, fy, cx, cy
            fx, fy, cx, cy = params[:4]

        elif model == 'OPENCV':
            # OPENCV format: fx, fy, cx, cy, k1, k2, p1, p2
            fx, fy, cx, cy = params[:4]

        else:
            raise ValueError(f"Unsupported camera model: {model}")

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        return K

    def get_distortion_coeffs(self, camera_id: int) -> Optional[np.ndarray]:
        """
        Get distortion coefficients for given camera.

        Args:
            camera_id: Camera ID from cameras.txt

        Returns:
            Distortion coefficients array or None if not available
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not found")

        camera = self.cameras[camera_id]
        model = camera['model']
        params = camera['params']

        if model == 'OPENCV_FISHEYE':
            # OPENCV_FISHEYE: k1, k2, k3, k4
            return params[4:8]

        elif model == 'OPENCV':
            # OPENCV: k1, k2, p1, p2
            return params[4:8]

        elif model == 'PINHOLE':
            return None

        return None

    def get_image_pose(self, image_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera pose (R, t) for given image.

        COLMAP stores quaternion and translation of camera in world coordinates.
        We convert to camera-to-world transformation.

        Args:
            image_name: Image filename (e.g., 'cam01/00001.jpg')

        Returns:
            Tuple of (R, t):
            - R: 3x3 rotation matrix (world to camera)
            - t: 3x1 translation vector
        """
        if image_name not in self.images:
            raise ValueError(f"Image {image_name} not found")

        image_data = self.images[image_name]
        quat = image_data['quat']
        tvec = image_data['tvec']

        # Convert quaternion to rotation matrix
        # COLMAP uses qw, qx, qy, qz format
        qw, qx, qy, qz = quat

        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ], dtype=np.float64)

        return R, tvec

    def get_camera_params(self, image_name: str) -> Dict[str, np.ndarray]:
        """
        Get complete camera parameters for given image.

        Args:
            image_name: Image filename

        Returns:
            Dictionary with:
            - 'K': 3x3 intrinsic matrix
            - 'R': 3x3 rotation matrix (world to camera)
            - 'tvec': 3x1 translation vector
            - 'dist': distortion coefficients (if available)
            - 'width': image width
            - 'height': image height
        """
        if image_name not in self.images:
            raise ValueError(f"Image {image_name} not found")

        image_data = self.images[image_name]
        camera_id = image_data['camera_id']

        R, tvec = self.get_image_pose(image_name)
        K = self.get_camera_intrinsics(camera_id)
        dist = self.get_distortion_coeffs(camera_id)
        camera = self.cameras[camera_id]

        params = {
            'K': K,
            'R': R,
            'tvec': tvec,
            'width': camera['width'],
            'height': camera['height'],
            'model': camera['model']
        }

        if dist is not None:
            params['dist'] = dist

        return params

    def get_camera_matrix(self, image_name: str) -> np.ndarray:
        """
        Get projection matrix P = K[R | t].

        Args:
            image_name: Image filename

        Returns:
            3x4 projection matrix
        """
        params = self.get_camera_params(image_name)
        K = params['K']
        R = params['R']
        t = params['tvec']

        P = K @ np.hstack([R, t.reshape(3, 1)])
        return P

    def get_camera_center(self, image_name: str) -> np.ndarray:
        """
        Get camera center in world coordinates.

        Args:
            image_name: Image filename

        Returns:
            3x1 camera center position
        """
        R, t = self.get_image_pose(image_name)

        # Camera center: C = -R^T @ t
        C = -R.T @ t

        return C


if __name__ == '__main__':
    # Test COLMAP loader
    import sys

    colmap_path = Path('/workspace/MMA/dataset/13_mma2/001_mma2/processed_data/colmap/workplace')

    if not colmap_path.exists():
        print(f"COLMAP directory not found: {colmap_path}")
        sys.exit(1)

    print("Loading COLMAP data...")
    loader = COLMAPLoader(colmap_path)

    print("\n=== COLMAP Camera Information ===\n")

    # Show camera info
    for camera_id, camera in loader.cameras.items():
        print(f"Camera {camera_id}:")
        print(f"  Model: {camera['model']}")
        print(f"  Resolution: {camera['width']}x{camera['height']}")
        print(f"  Params: {camera['params']}")

        K = loader.get_camera_intrinsics(camera_id)
        print(f"  K matrix:\n{K}")

    print("\n=== Sample Images ===\n")

    # Show first few images
    for i, (image_name, image_data) in enumerate(list(loader.images.items())[:3]):
        print(f"Image {i+1}: {image_name}")
        print(f"  Image ID: {image_data['id']}")
        print(f"  Camera ID: {image_data['camera_id']}")

        R, t = loader.get_image_pose(image_name)
        print(f"  Translation: {t}")
        print(f"  Rotation matrix:\n{R}")

        C = loader.get_camera_center(image_name)
        print(f"  Camera center: {C}")

    print("\n✓ COLMAP loader working correctly!")
