"""
Pose Estimation module for 2D keypoint detection

Provides:
- PoseEstimator: YOLO pose model wrapper (singleton)
- Batch processing with progress tracking
- Coordinate transformations (crop ↔ original)
- Integration with detection tracking

Replaces:
- pose_estimation.py: Pose estimation logic
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..core.config import PoseConfig
from ..core.exceptions import ModelLoadError, InferenceError
from ..core.constants import COCO_KEYPOINT_NAMES


def _get_tqdm():
    """Lazy load tqdm"""
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        return lambda x, *args, **kwargs: x


class PoseEstimator:
    """
    YOLO Pose model wrapper with singleton pattern

    Estimates 2D keypoints for persons from image crops.
    Supports YOLO11-pose, RTMPose, ViTPose models.

    Example:
        >>> from mma.pose.estimator import PoseEstimator
        >>> from mma.core.config import PoseConfig
        >>> config = PoseConfig(model_path="yolo11x-pose.pt")
        >>> estimator = PoseEstimator(config)
        >>> keypoints = estimator.estimate_pose(crop_image, original_bbox)
    """

    _instance: Optional["PoseEstimator"] = None
    _model: Optional[object] = None

    def __new__(cls, config: PoseConfig):
        """Singleton factory"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: PoseConfig):
        """
        Initialize pose estimator

        Args:
            config: PoseConfig with model parameters

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        if self._initialized:
            return

        self.config = config
        self._model = None
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """
        Load YOLO pose model

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ModelLoadError(
                "ultralytics package not installed. Install with: pip install ultralytics"
            )

        try:
            model_path = self.config.model_path

            # Try to load from local path first
            if Path(model_path).exists():
                self._model = YOLO(str(model_path))
            else:
                # Try to download from ultralytics
                self._model = YOLO(model_path)

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load pose model '{self.config.model_path}': {e}"
            )

    def estimate_pose(
        self,
        crop_image: np.ndarray,
        original_bbox: Tuple[float, float, float, float],
        crop_info: Dict,
        conf_threshold: float = 0.3
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Estimate pose on cropped image

        Args:
            crop_image: Cropped person region (H, W, 3) BGR
            original_bbox: (x1, y1, x2, y2) in crop coordinates
            crop_info: {'offset_x', 'offset_y', 'crop_w', 'crop_h'}
            conf_threshold: Keypoint confidence threshold

        Returns:
            Dict mapping keypoint names to (x, y, confidence) in crop coordinates

        Example:
            >>> crop_info = {'offset_x': 10, 'offset_y': 20, 'crop_w': 200, 'crop_h': 300}
            >>> keypoints = estimator.estimate_pose(crop, (10, 20, 100, 150), crop_info)
            >>> print(keypoints['nose'])  # (x, y, conf)
        """
        if self._model is None:
            raise ModelLoadError("Pose model not initialized")

        if crop_image.size == 0:
            return {}

        try:
            results = self._model.predict(
                crop_image,
                conf=0.1,
                verbose=False,
                device=self.config.device
            )

            if len(results) == 0 or results[0].keypoints is None:
                return {}

            keypoints_data = results[0].keypoints

            if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
                return {}

            # Convert original_bbox to crop coordinates
            orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox
            orig_bbox_center_x = (orig_x1 + orig_x2) / 2
            orig_bbox_center_y = (orig_y1 + orig_y2) / 2

            # Find detection closest to original bbox
            best_idx = 0
            best_distance = float("inf")

            if hasattr(results[0], "boxes") and results[0].boxes is not None:
                boxes = results[0].boxes
                for idx in range(len(boxes)):
                    box = boxes[idx].xyxy[0].cpu().numpy()
                    box_center_x = (box[0] + box[2]) / 2
                    box_center_y = (box[1] + box[3]) / 2
                    distance = np.sqrt(
                        (box_center_x - orig_bbox_center_x) ** 2
                        + (box_center_y - orig_bbox_center_y) ** 2
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_idx = idx

            # Extract keypoints
            kpts_xy = keypoints_data.xy[best_idx].cpu().numpy()  # (17, 2)
            kpts_conf = (
                keypoints_data.conf[best_idx].cpu().numpy()
                if keypoints_data.conf is not None
                else np.ones(17)
            )

            # Build keypoint dict
            keypoints = {}
            for i, name in enumerate(COCO_KEYPOINT_NAMES):
                x, y = kpts_xy[i]
                conf = float(kpts_conf[i])
                keypoints[name] = (float(x), float(y), conf)

            return keypoints

        except Exception as e:
            raise InferenceError(f"Pose estimation failed: {e}")

    def estimate_batch(
        self,
        crop_images: List[np.ndarray],
        original_bboxes: List[Tuple[float, float, float, float]],
        crop_infos: List[Dict],
        conf_threshold: float = 0.3,
        show_progress: bool = True
    ) -> List[Dict[str, Tuple[float, float, float]]]:
        """
        Estimate poses for multiple crops

        Args:
            crop_images: List of cropped images
            original_bboxes: List of original bboxes
            crop_infos: List of crop info dicts
            conf_threshold: Confidence threshold
            show_progress: Show progress bar

        Returns:
            List of keypoint dictionaries
        """
        results = []
        tqdm_fn = _get_tqdm()
        iterator = (
            tqdm_fn(zip(crop_images, original_bboxes, crop_infos),
                   total=len(crop_images), desc="Estimating poses")
            if show_progress
            else zip(crop_images, original_bboxes, crop_infos)
        )

        for crop_img, orig_bbox, crop_info in iterator:
            try:
                kpts = self.estimate_pose(crop_img, orig_bbox, crop_info, conf_threshold)
                results.append(kpts)
            except InferenceError:
                results.append({})

        return results

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)"""
        cls._instance = None
        cls._model = None


# Global instance
_global_estimator: Optional[PoseEstimator] = None


def get_pose_estimator(config: PoseConfig) -> PoseEstimator:
    """
    Get or create global pose estimator instance

    Args:
        config: PoseConfig

    Returns:
        PoseEstimator singleton instance
    """
    global _global_estimator
    if _global_estimator is None:
        _global_estimator = PoseEstimator(config)
    return _global_estimator
