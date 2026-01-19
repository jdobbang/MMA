"""
Data loading utilities for MMA pipeline

Unified interfaces for loading:
- Images (with OpenCV, PIL)
- NPY files (numpy arrays, SMPL data, poses2d)
- Batch operations
- Error handling and validation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from tqdm import tqdm

from ..core.exceptions import ImageLoadError, DataLoadError


class ImageLoader:
    """
    Unified image loading with comprehensive error handling

    Supports:
    - Single image loading
    - Batch loading
    - Automatic color space conversion
    - Size queries
    - Error handling and logging
    """

    @staticmethod
    def load(image_path: str, color_space: str = 'bgr') -> np.ndarray:
        """
        Load a single image with error handling

        Replaces repeated pattern in:
        - run_mma_pipeline.py
        - pose_estimation.py
        - detection.py

        Args:
            image_path: Path to image file
            color_space: 'bgr' (default, OpenCV) or 'rgb'

        Returns:
            Image array (H, W, 3)

        Raises:
            ImageLoadError: If image cannot be loaded

        Example:
            >>> from mma.io import ImageLoader
            >>> img = ImageLoader.load("frame_001.jpg")
            >>> print(img.shape)
            (480, 640, 3)
        """
        path = Path(image_path)

        # Check existence
        if not path.exists():
            raise ImageLoadError(f"Image file not found: {image_path}")

        # Load image
        image = cv2.imread(str(path))

        if image is None:
            raise ImageLoadError(
                f"Failed to read image (corrupted or unsupported format): {image_path}"
            )

        # Color space conversion
        if color_space.lower() == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    @staticmethod
    def load_batch(
        image_paths: List[str],
        color_space: str = 'bgr',
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Load multiple images with progress tracking

        Args:
            image_paths: List of image file paths
            color_space: 'bgr' or 'rgb'
            show_progress: Show progress bar

        Returns:
            List of image arrays, skipping failed images

        Example:
            >>> paths = [f"frame_{i:04d}.jpg" for i in range(100)]
            >>> images = ImageLoader.load_batch(paths)
            >>> print(len(images))
            100
        """
        images = []
        failed_count = 0

        iterator = tqdm(image_paths) if show_progress else image_paths

        for path in iterator:
            try:
                img = ImageLoader.load(path, color_space)
                images.append(img)
            except ImageLoadError:
                failed_count += 1
                continue

        if failed_count > 0:
            print(f"Warning: Failed to load {failed_count} images")

        return images

    @staticmethod
    def get_size(image_path: str) -> Tuple[int, int]:
        """
        Get image dimensions without loading full image

        Args:
            image_path: Path to image file

        Returns:
            (height, width) tuple

        Raises:
            ImageLoadError: If image cannot be read
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ImageLoadError(f"Cannot read image: {image_path}")
        return img.shape[:2]

    @staticmethod
    def validate_format(image_path: str) -> bool:
        """
        Check if file is a valid image format

        Args:
            image_path: Path to check

        Returns:
            True if valid image format
        """
        path = Path(image_path)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return path.suffix.lower() in valid_extensions


class NPYLoader:
    """
    Unified NPY file loading for SMPL and poses data

    Replaces repeated pattern in:
    - generate_yolo_pose_dataset.py
    - crop_detection_pose_dataset.py
    - render_smpl.py

    Handles:
    - SMPL parameter files
    - Poses2D keypoint data
    - Bounding box data
    - Error handling and validation
    """

    @staticmethod
    def load_npy(npy_path: str, allow_pickle: bool = True) -> Dict:
        """
        Load NPY file as dictionary

        Args:
            npy_path: Path to NPY file
            allow_pickle: Allow pickle format (required for dict files)

        Returns:
            Loaded dictionary

        Raises:
            DataLoadError: If NPY file cannot be loaded
        """
        npy_path = Path(npy_path)

        if not npy_path.exists():
            raise DataLoadError(f"NPY file not found: {npy_path}")

        try:
            data = np.load(npy_path, allow_pickle=allow_pickle)

            # If it's an npz file (dict of arrays)
            if isinstance(data, np.lib.npyio.NpzFile):
                data = {key: data[key] for key in data.files}
            # If it's a dict in pickle format
            elif isinstance(data, np.ndarray):
                data = data.item() if data.dtype == object else data

            if not isinstance(data, dict):
                raise DataLoadError(
                    f"Expected dict format, got {type(data).__name__}: {npy_path}"
                )

            return data

        except Exception as e:
            raise DataLoadError(f"Failed to load NPY file {npy_path}: {e}")

    @staticmethod
    def load_smpl(smpl_path: str, frame_id: str) -> Dict:
        """
        Load SMPL parameters for a specific frame

        Replaces pattern in:
        - render_smpl.py
        - smpl_utils.py

        Args:
            smpl_path: Path to SMPL directory (contains frame_*.npy files)
            frame_id: Frame identifier (e.g., "00001")

        Returns:
            Dictionary with keys like 'aria01', 'aria02' containing:
            - 'vertices': (6890, 3) array
            - 'betas': (10,) array
            - 'pose': (72,) array
            - 'trans': (3,) array

        Raises:
            DataLoadError: If SMPL data cannot be loaded
        """
        smpl_path = Path(smpl_path)
        frame_file = smpl_path / f"{frame_id}.npy"

        if not frame_file.exists():
            raise DataLoadError(f"SMPL frame data not found: {frame_file}")

        try:
            data = np.load(frame_file, allow_pickle=True).item()
            return data
        except Exception as e:
            raise DataLoadError(f"Failed to load SMPL data from {frame_file}: {e}")

    @staticmethod
    def load_poses2d(
        pose2d_path: str,
        frame_id: str,
        camera: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Load Poses2D keypoint data for a specific frame

        Replaces pattern in:
        - generate_yolo_pose_dataset.py
        - crop_detection_pose_dataset.py

        Args:
            pose2d_path: Path to Poses2D directory
            frame_id: Frame identifier
            camera: Specific camera name (e.g., 'aria01'). If None, load all.

        Returns:
            Dictionary with camera names as keys and (45, 3) keypoint arrays
            Returns None if frame_id file not found

        Raises:
            DataLoadError: If data is corrupted
        """
        pose2d_path = Path(pose2d_path)
        frame_file = pose2d_path / f"{frame_id}.npy"

        if not frame_file.exists():
            return None

        try:
            data = np.load(frame_file, allow_pickle=True).item()

            if camera is not None:
                if camera not in data:
                    return None
                return {0: data[camera]}  # Return as {0: keypoints}
            else:
                # All cameras
                result = {}
                for player_id, cam_name in enumerate(sorted(data.keys())):
                    result[player_id] = data[cam_name]
                return result

        except Exception as e:
            raise DataLoadError(f"Failed to load poses2d data: {e}")

    @staticmethod
    def load_bbox(bbox_file: str) -> Optional[Dict]:
        """
        Load bounding box data

        Replaces pattern in:
        - generate_yolo_pose_dataset.py
        - crop_detection_pose_dataset.py

        Args:
            bbox_file: Path to bbox NPY file

        Returns:
            Dictionary with player_id as keys and [x1, y1, x2, y2] bbox arrays
            Returns None if file not found

        Raises:
            DataLoadError: If bbox data is corrupted
        """
        bbox_file = Path(bbox_file)

        if not bbox_file.exists():
            return None

        try:
            bbox_dict = np.load(bbox_file, allow_pickle=True).item()
            if not isinstance(bbox_dict, dict):
                raise DataLoadError("Bbox file format is not a dictionary")
            return bbox_dict
        except Exception as e:
            raise DataLoadError(f"Failed to load bbox data: {e}")

    @staticmethod
    def load_batch_smpl(
        smpl_path: str,
        frame_ids: List[str],
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Load multiple SMPL frames

        Args:
            smpl_path: Path to SMPL directory
            frame_ids: List of frame identifiers
            show_progress: Show progress bar

        Returns:
            Dictionary mapping frame_id to SMPL data
        """
        smpl_data = {}
        iterator = tqdm(frame_ids) if show_progress else frame_ids

        for frame_id in iterator:
            try:
                smpl_data[frame_id] = NPYLoader.load_smpl(smpl_path, frame_id)
            except DataLoadError:
                continue

        return smpl_data

    @staticmethod
    def validate_smpl_data(data: Dict) -> bool:
        """
        Validate SMPL data structure

        Args:
            data: SMPL data dictionary

        Returns:
            True if data has required keys and correct shapes
        """
        required_keys = ['vertices', 'betas', 'pose', 'trans']

        # Check required keys exist
        if not all(key in data for key in required_keys):
            return False

        # Check shapes
        if data['vertices'].shape != (6890, 3):
            return False
        if data['betas'].shape != (10,):
            return False
        if data['pose'].shape != (72,):
            return False
        if data['trans'].shape != (3,):
            return False

        return True
