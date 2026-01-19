"""
Kalman filter implementation for bounding box tracking

Replaces:
- sort_tracker.py: KalmanBoxTracker class

Provides:
- Kalman filter for 1D state [center_x, center_y, scale, aspect_ratio]
- Velocity estimation
- Prediction and update operations
"""

import numpy as np
from typing import Tuple, Optional

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


class KalmanBoxTracker:
    """
    Kalman filter tracking for bounding boxes

    Tracks bbox state using Kalman filter with 4D state:
    - x, y: bbox center
    - s: scale (area)
    - r: aspect ratio

    Replaces sort_tracker.py: KalmanBoxTracker

    Example:
        >>> tracker = KalmanBoxTracker(np.array([100, 100, 200, 200]))
        >>> # Predict next state
        >>> pred = tracker.predict()
        >>> # Update with new detection
        >>> tracker.update(np.array([105, 105, 205, 205]))
    """

    count = 0  # Class variable for bbox ID assignment

    def __init__(self, bbox: np.ndarray):
        """
        Initialize Kalman filter for bbox

        Args:
            bbox: Initial bbox [x1, y1, x2, y2]
        """
        if KalmanFilter is None:
            raise ImportError("filterpy package required. Install with: pip install filterpy")

        # Kalman filter setup
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (position + velocity)
        # [x, y, s, r, vx, vy, vs]
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement matrix (we observe position and scale only)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise (covariance of measurements)
        self.kf.R[2:, 2:] *= 5.0  # Scale and aspect ratio have higher noise
        self.kf.R[:2, :2] *= 0.5  # Position measurement noise

        # Process noise (how much we trust the model vs measurements)
        self.kf.P *= 5.0  # Initial state covariance
        self.kf.P[4:, 4:] *= 500.0  # High velocity uncertainty initially

        # Process noise covariance
        self.kf.Q[-1, -1] *= 0.005  # Aspect ratio process noise
        self.kf.Q[4:, 4:] *= 0.005  # Velocity process noise

        # Initialize state from bbox
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Confidence score (updated on each measurement)
        self.confidence = bbox[4] if len(bbox) > 4 else 1.0

        # Convert bbox to Kalman state and initialize
        self.kf.x[:4, 0] = self._bbox_to_z(bbox)

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """
        Convert bbox [x1, y1, x2, y2] to Kalman state [x, y, s, r]

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            [x, y, s, r] state
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h  # Scale (area)
        r = w / float(h)  # Aspect ratio
        return np.array([x, y, s, r])

    @staticmethod
    def _z_to_bbox(x: np.ndarray, score: Optional[float] = None) -> np.ndarray:
        """
        Convert Kalman state [x, y, s, r] to bbox [x1, y1, x2, y2]

        Args:
            x: Kalman state [x, y, s, r, ...]
            score: Optional confidence score

        Returns:
            [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([
                x[0] - w / 2.0,
                x[1] - h / 2.0,
                x[0] + w / 2.0,
                x[1] + h / 2.0
            ])
        else:
            return np.array([
                x[0] - w / 2.0,
                x[1] - h / 2.0,
                x[0] + w / 2.0,
                x[1] + h / 2.0,
                score
            ])

    def predict(self) -> np.ndarray:
        """
        Predict next bbox based on Kalman filter

        Returns:
            Predicted bbox [x1, y1, x2, y2]
        """
        # Ensure state uncertainty is large if not recently updated
        if (self.kf.x[6] + self.time_since_update) > 0:
            self.kf.x[6] += self.kf.Q[6, 6] * self.time_since_update

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self._z_to_bbox(self.kf.x[:4, 0])

    def update(self, bbox: np.ndarray) -> None:
        """
        Update Kalman filter with new bbox measurement

        Args:
            bbox: Measured bbox [x1, y1, x2, y2, conf] or [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        # Update confidence if provided
        if len(bbox) > 4:
            self.confidence = bbox[4]

        # Measurement
        z = self._bbox_to_z(bbox)
        self.kf.update(z)

    def get_state(self) -> np.ndarray:
        """
        Get current bbox state

        Returns:
            Current bbox [x1, y1, x2, y2]
        """
        return self._z_to_bbox(self.kf.x[:4, 0])

    def get_state_with_score(self, score: float) -> np.ndarray:
        """
        Get current bbox state with confidence score

        Args:
            score: Confidence score

        Returns:
            [x1, y1, x2, y2, score]
        """
        return self._z_to_bbox(self.kf.x[:4, 0], score=score)


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two bboxes

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    from ..detection.bbox_utils import iou as compute_iou
    return compute_iou(bbox1, bbox2)


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two bbox sets

    Args:
        bboxes1: (N, 4) array
        bboxes2: (M, 4) array

    Returns:
        (N, M) IoU matrix
    """
    from ..detection.bbox_utils import iou_batch as compute_iou_batch
    return compute_iou_batch(bboxes1, bboxes2)
