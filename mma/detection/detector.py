"""
YOLO object detector wrapper with singleton caching

Replaces:
- detection.py: repeated YOLO model loading and inference

Features:
- Singleton pattern for single model instance (memory efficient)
- Batch detection support
- Automatic device management (CPU/GPU)
- Progress tracking
- Error handling and validation
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np

from ..core.config import DetectionConfig
from ..core.exceptions import ModelLoadError, InferenceError
from .bbox_utils import validate_bbox


def _get_tqdm():
    """Lazy load tqdm"""
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        # Return identity function if tqdm not available
        return lambda x, *args, **kwargs: x


class YOLODetector:
    """
    YOLO object detector with singleton pattern

    Ensures only one model instance exists in memory, reducing memory usage
    and avoiding redundant model loading.

    Example:
        >>> from mma.detection import YOLODetector
        >>> from mma.core.config import DetectionConfig
        >>> config = DetectionConfig(model_path="yolo11x.pt", conf_threshold=0.1)
        >>> detector = YOLODetector(config)
        >>> detections = detector.detect(image)
    """

    _instance: Optional["YOLODetector"] = None
    _model: Optional[object] = None

    def __new__(cls, config: DetectionConfig):
        """
        Singleton factory method

        Creates single instance, subsequent calls return same instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: DetectionConfig):
        """
        Initialize detector

        Args:
            config: DetectionConfig instance with model parameters

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        # Skip reinitialize if already done
        if self._initialized:
            return

        self.config = config
        self._model = None
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """
        Load YOLO model

        Raises:
            ModelLoadError: If model file not found or corrupt
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
                f"Failed to load YOLO model '{self.config.model_path}': {e}"
            )

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect objects in a single image

        Args:
            image: Input image (H, W, 3) in BGR format
            conf_threshold: Confidence threshold (uses config if None)
            classes: Classes to detect (uses config if None)

        Returns:
            List of detections: [(x1, y1, x2, y2, conf), ...]

        Raises:
            InferenceError: If detection fails
        """
        conf_threshold = conf_threshold or self.config.conf_threshold
        classes = classes or self.config.classes

        try:
            results = self._model.predict(
                image,
                conf=conf_threshold,
                classes=classes,
                verbose=False,
                device=self.config.device
            )

            detections = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                # Extract detection results
                boxes = result.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
                confs = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    detections.append((float(x1), float(y1), float(x2), float(y2), float(conf)))

            return detections

        except Exception as e:
            raise InferenceError(f"YOLO detection failed: {e}")

    def detect_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
        show_progress: bool = True
    ) -> List[List[Tuple[float, float, float, float, float]]]:
        """
        Detect objects in multiple images

        Args:
            images: List of images
            conf_threshold: Confidence threshold
            classes: Classes to detect
            show_progress: Show progress bar

        Returns:
            List of detection lists

        Example:
            >>> images = [img1, img2, img3]
            >>> batch_detections = detector.detect_batch(images)
            >>> for detections in batch_detections:
            ...     print(f"Found {len(detections)} objects")
        """
        conf_threshold = conf_threshold or self.config.conf_threshold
        classes = classes or self.config.classes

        batch_detections = []
        tqdm_fn = _get_tqdm()
        iterator = tqdm_fn(images, desc="Detecting") if show_progress else images

        for image in iterator:
            try:
                detections = self.detect(image, conf_threshold, classes)
                batch_detections.append(detections)
            except InferenceError as e:
                print(f"Warning: {e}, skipping image")
                batch_detections.append([])

        return batch_detections

    def detect_from_paths(
        self,
        image_paths: List[str],
        conf_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
        show_progress: bool = True
    ) -> List[List[Tuple[float, float, float, float, float]]]:
        """
        Detect objects from list of image file paths

        Args:
            image_paths: List of image file paths
            conf_threshold: Confidence threshold
            classes: Classes to detect
            show_progress: Show progress bar

        Returns:
            List of detection lists

        Raises:
            FileNotFoundError: If image file not found
        """
        from ..io.data_loader import ImageLoader

        images = []
        valid_paths = []

        # Load images
        tqdm_fn = _get_tqdm()
        iterator = tqdm_fn(image_paths, desc="Loading images") if show_progress else image_paths
        for path in iterator:
            try:
                img = ImageLoader.load(path)
                images.append(img)
                valid_paths.append(path)
            except Exception:
                continue

        # Batch detect
        batch_detections = self.detect_batch(
            images,
            conf_threshold=conf_threshold,
            classes=classes,
            show_progress=False
        )

        return batch_detections

    def get_model_info(self) -> Dict:
        """
        Get YOLO model information

        Returns:
            Dictionary with model metadata
        """
        return {
            'model_path': self.config.model_path,
            'device': self.config.device,
            'conf_threshold': self.config.conf_threshold,
            'classes': self.config.classes,
            'imgsz': self.config.imgsz,
        }

    def update_config(self, config: DetectionConfig) -> None:
        """
        Update detector configuration

        Args:
            config: New DetectionConfig
        """
        self.config = config

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing)

        This unloads the model and allows creating a new instance.
        """
        cls._instance = None
        cls._model = None


# Global detector instance
_global_detector: Optional[YOLODetector] = None


def get_detector(config: DetectionConfig) -> YOLODetector:
    """
    Get or create global detector instance

    Convenience function for accessing singleton detector.

    Args:
        config: DetectionConfig

    Returns:
        YOLODetector instance

    Example:
        >>> from mma.detection.detector import get_detector
        >>> from mma.core.config import DetectionConfig
        >>> config = DetectionConfig()
        >>> detector = get_detector(config)
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = YOLODetector(config)
    return _global_detector
