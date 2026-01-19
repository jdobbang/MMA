"""
Re-ID feature extraction for person re-identification

Replaces:
- mma_tracker.py: Re-ID model loading and feature extraction

Provides:
- ReIDModel singleton for efficient feature extraction
- Batch feature extraction support
- Cosine similarity computation
"""

import numpy as np
from typing import List, Tuple, Optional

from ..core.config import TrackingConfig
from ..core.exceptions import ModelLoadError


class ReIDModel:
    """
    Person Re-ID model wrapper with singleton pattern

    Ensures only one Re-ID model instance exists in memory.
    Currently uses OSNet from torchreid.

    Replaces mma_tracker.py Re-ID loading logic

    Example:
        >>> from mma.tracking.reid import get_reid_model
        >>> from mma.core.config import TrackingConfig
        >>> config = TrackingConfig()
        >>> reid = get_reid_model(config)
        >>> features = reid.extract_features(image, detections)
    """

    _instance: Optional["ReIDModel"] = None
    _model: Optional[object] = None
    _transform: Optional[object] = None

    def __new__(cls, config: TrackingConfig):
        """Singleton factory"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: TrackingConfig):
        """
        Initialize Re-ID model

        Args:
            config: TrackingConfig with Re-ID parameters

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        if self._initialized:
            return

        self.config = config
        self._model = None
        self._transform = None
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """
        Load OSNet Re-ID model from torchreid

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            import torch
            import torchvision.transforms as T
            import torchreid
        except ImportError:
            raise ModelLoadError(
                "torch, torchvision, or torchreid not installed. "
                "Install with: pip install torch torchvision torchreid"
            )

        try:
            # Load pre-trained OSNet model
            self._model = torchreid.models.build_model(
                name=self.config.reid_model_name,
                num_classes=1000,
                loss='softmax',
                pretrained=True,
                device='cuda' if self.config.device == 'cuda' else 'cpu'
            )
            self._model.eval()

            # Create transform pipeline
            self._transform = T.Compose([
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        except Exception as e:
            raise ModelLoadError(f"Failed to load Re-ID model: {e}")

    def extract_features(
        self,
        image: np.ndarray,
        detections: List[Tuple[float, float, float, float]]
    ) -> List[np.ndarray]:
        """
        Extract Re-ID features for detections in image

        Args:
            image: Image array (H, W, 3) BGR format
            detections: List of detections [(x1, y1, x2, y2), ...]

        Returns:
            List of feature vectors (512-dim for OSNet)

        Example:
            >>> image = cv2.imread('frame.jpg')
            >>> detections = [(100, 100, 200, 200), (300, 300, 400, 400)]
            >>> features = reid.extract_features(image, detections)
            >>> print(len(features))  # 2
            >>> print(features[0].shape)  # (512,)
        """
        if self._model is None:
            raise ModelLoadError("Re-ID model not initialized")

        import torch
        import cv2

        features_list = []

        for detection in detections:
            x1, y1, x2, y2 = detection

            # Crop person region
            try:
                crop = image[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    # Invalid crop, use zero feature
                    features_list.append(np.zeros(512, dtype=np.float32))
                    continue
            except:
                features_list.append(np.zeros(512, dtype=np.float32))
                continue

            # Resize to model input size
            crop_resized = cv2.resize(crop, self.config.reid_resize[::-1])

            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

            # Apply transform
            try:
                crop_tensor = self._transform(crop_rgb)
                crop_tensor = crop_tensor.unsqueeze(0).to(
                    'cuda' if self.config.device == 'cuda' else 'cpu'
                )

                # Extract features
                with torch.no_grad():
                    feature = self._model(crop_tensor)

                # Convert to numpy
                feature_np = feature.cpu().numpy().flatten().astype(np.float32)
                features_list.append(feature_np)

            except Exception as e:
                print(f"Warning: Failed to extract feature: {e}")
                features_list.append(np.zeros(512, dtype=np.float32))

        return features_list

    def extract_features_batch(
        self,
        crops: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Extract Re-ID features for batch of person crops

        Args:
            crops: List of person crop images

        Returns:
            List of feature vectors
        """
        if self._model is None:
            raise ModelLoadError("Re-ID model not initialized")

        import torch

        features_list = []

        for crop in crops:
            # Resize
            crop_resized = cv2.resize(crop, self.config.reid_resize[::-1])

            # BGR to RGB
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

            # Transform
            try:
                crop_tensor = self._transform(crop_rgb)
                crop_tensor = crop_tensor.unsqueeze(0).to(
                    'cuda' if self.config.device == 'cuda' else 'cpu'
                )

                with torch.no_grad():
                    feature = self._model(crop_tensor)

                feature_np = feature.cpu().numpy().flatten().astype(np.float32)
                features_list.append(feature_np)

            except Exception:
                features_list.append(np.zeros(512, dtype=np.float32))

        return features_list

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance"""
        cls._instance = None
        cls._model = None
        cls._transform = None


# Global instance
_global_reid: Optional[ReIDModel] = None


def get_reid_model(config: TrackingConfig) -> ReIDModel:
    """
    Get or create global Re-ID model

    Args:
        config: TrackingConfig

    Returns:
        ReIDModel singleton instance
    """
    global _global_reid
    if _global_reid is None:
        _global_reid = ReIDModel(config)
    return _global_reid


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors

    Args:
        vec1: Feature vector
        vec2: Feature vector

    Returns:
        Cosine similarity score in [-1, 1]

    Example:
        >>> feat1 = np.random.rand(512)
        >>> feat2 = np.random.rand(512)
        >>> sim = cosine_similarity(feat1, feat2)
        >>> print(f"Similarity: {sim:.3f}")
    """
    # Normalize
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

    # Cosine similarity
    return float(np.dot(vec1_norm, vec2_norm))


def cosine_similarity_batch(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two feature sets

    Args:
        features1: (N, 512) feature matrix
        features2: (M, 512) feature matrix

    Returns:
        (N, M) similarity matrix
    """
    # Normalize
    features1_norm = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    features2_norm = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)

    # Pairwise similarity
    similarity_matrix = np.dot(features1_norm, features2_norm.T)

    return similarity_matrix
