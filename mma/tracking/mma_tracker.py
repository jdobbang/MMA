"""
MMA Template Tracker

Hybrid tracking for MMA 2-player scenarios using:
- Kalman Filter for position prediction (from SORT)
- Re-ID features for appearance-based identification
- Dynamic weighting strategy based on IoU

Replaces:
- mma_tracker.py: MMA-specific tracking logic
- Separates Re-ID model loading (moved to reid.py)
- Separates track interpolation (moved to interpolation.py)

Provides:
- MMATracker: 2-player tracker for MMA matches
- PlayerTemplate: Per-player state management
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from scipy.optimize import linear_sum_assignment

from ..core.config import TrackingConfig
from ..core.exceptions import TrackingError
from .kalman import KalmanBoxTracker
from .reid import get_reid_model
from ..detection.bbox_utils import iou


@dataclass
class PlayerTemplate:
    """
    Per-player state template for MMA tracking

    Stores both fixed and adaptive Re-ID templates:
    - initial_reid_feature: Fixed template for ID recovery after swaps
    - adaptive_reid_feature: EMA-updated template for continuous tracking
    """

    id: int  # Player 1 or 2

    # Fixed template (ID recovery)
    initial_reid_feature: np.ndarray  # 512-dim, averaged from separated frames

    # Adaptive template (continuous tracking)
    adaptive_reid_feature: np.ndarray  # 512-dim, EMA-updated

    # Initial template collection
    initial_feature_sum: np.ndarray = None  # Accumulated features
    initial_feature_count: int = 0  # Number of frames collected
    initial_collection_done: bool = False  # Collection complete flag

    # Kalman state (updated per frame)
    kalman_tracker: KalmanBoxTracker = None

    # State tracking
    last_bbox: List[float] = field(default_factory=list)  # Last detection bbox
    last_seen_frame: int = 0  # Last detection frame
    hit_streak: int = 0  # Consecutive detections
    time_since_update: int = 0  # Frames since last update


class MMATracker:
    """
    Hybrid 2-player tracker for MMA matches

    Core strategy:
    - Exactly 2 fixed players (no new IDs created)
    - Kalman Filter for position prediction
    - Re-ID for appearance-based identification
    - Dynamic weighting based on IoU (separation state)

    Example:
        >>> from mma.tracking.mma_tracker import MMATracker
        >>> from mma.core.config import TrackingConfig
        >>> config = TrackingConfig(max_age=30, reid_ema_alpha=0.1)
        >>> tracker = MMATracker(config)
        >>> for frame_num, (image, detections) in enumerate(video_data):
        ...     # detections: [(x1, y1, x2, y2, conf), ...]
        ...     tracked = tracker.update(image, detections, frame_num)
        ...     # tracked: [[x1, y1, x2, y2, track_id, conf], ...]
    """

    def __init__(self, config: TrackingConfig):
        """
        Initialize MMA tracker

        Args:
            config: TrackingConfig with tracking parameters
        """
        self.config = config

        # Player templates (1, 2)
        self.templates: List[Optional[PlayerTemplate]] = [None, None]
        self.initialized = False
        self.frame_count = 0

        # Re-ID model (lazy loaded)
        self.reid_model = None

    def update(
        self,
        frame_img: np.ndarray,
        detections: List[tuple],
        frame_num: int
    ) -> np.ndarray:
        """
        Update tracker with new frame and detections

        Args:
            frame_img: BGR image (H, W, 3)
            detections: List of (x1, y1, x2, y2, conf) tuples
            frame_num: Current frame number

        Returns:
            Tracked objects: [[x1, y1, x2, y2, track_id, conf], ...] (max 2)

        Algorithm:
        1. Initialize on first 2 separated detections
        2. Kalman predict for each player
        3. Extract Re-ID features from detections
        4. Compute hybrid cost matrix (IoU + Re-ID similarity)
        5. Hungarian assignment
        6. Update Kalman and Re-ID templates
        """
        self.frame_count = frame_num

        # Convert to numpy array
        if isinstance(detections, list):
            detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # Step 1: Initialize if needed
        if not self.initialized:
            if self._initialize(frame_img, detections, frame_num):
                return self._get_output()
            else:
                return np.empty((0, 6))

        # Step 2: Kalman predict
        predicted_bboxes = []
        for template in self.templates:
            if template is not None:
                pred = template.kalman_tracker.predict()
                predicted_bboxes.append(pred)
            else:
                predicted_bboxes.append(np.zeros((1, 4)))

        # Step 3: Handle empty detections
        if len(detections) == 0:
            for template in self.templates:
                if template is not None:
                    template.time_since_update += 1
                    template.hit_streak = 0
            return self._get_output()

        # Step 4: Extract Re-ID features
        det_features = self._extract_reid_features(frame_img, detections)
        if det_features is None:
            det_features = [np.zeros(512) for _ in range(len(detections))]

        # Step 5: Compute cost matrix
        cost_matrix = self._compute_cost_matrix(
            detections, det_features, predicted_bboxes
        )

        # Step 6: Hungarian assignment
        assignments = self._assign(cost_matrix, detections)

        # Step 7: Check player separation (IoU)
        players_separated = False
        players_overlapping = False
        if assignments[0] is not None and assignments[1] is not None:
            det0_bbox = detections[assignments[0], :4]
            det1_bbox = detections[assignments[1], :4]
            players_iou = iou(
                np.array(det0_bbox, dtype=np.float32),
                np.array(det1_bbox, dtype=np.float32)
            )
            players_separated = players_iou == 0
            players_overlapping = players_iou > 0

        # Step 8: Mark initial collection done on first overlap
        if players_overlapping:
            for template in self.templates:
                if template is not None and not template.initial_collection_done:
                    template.initial_collection_done = True

        # Step 9: Update templates
        for player_idx, det_idx in assignments.items():
            template = self.templates[player_idx]
            if template is None:
                continue

            if det_idx is not None:
                det = detections[det_idx]
                feat = det_features[det_idx]

                # Kalman update
                template.kalman_tracker.update(det)

                # EMA update of adaptive template (always)
                template.adaptive_reid_feature = (
                    (1 - self.config.reid_ema_alpha)
                    * template.adaptive_reid_feature
                    + self.config.reid_ema_alpha * feat
                )

                # Collect initial template (separated state only)
                if players_separated and not template.initial_collection_done:
                    template.initial_feature_sum += feat
                    template.initial_feature_count += 1
                    template.initial_reid_feature = (
                        template.initial_feature_sum
                        / template.initial_feature_count
                    )

                # Update state
                template.last_bbox = det[:4].tolist()
                template.last_seen_frame = frame_num
                template.hit_streak += 1
                template.time_since_update = 0
            else:
                # No detection
                template.time_since_update += 1
                template.hit_streak = 0

        return self._get_output()

    def _initialize(
        self,
        frame_img: np.ndarray,
        detections: np.ndarray,
        frame_num: int
    ) -> bool:
        """
        Initialize tracker with 2 separated detections

        Args:
            frame_img: BGR image
            detections: Detection array
            frame_num: Frame number

        Returns:
            True if initialization successful
        """
        if len(detections) < 2:
            return False

        # Select top 2 by confidence
        if len(detections) > 2:
            indices = np.argsort(detections[:, 4])[::-1][:2]
            detections = detections[indices]

        # Check IoU (separation state)
        det1_bbox = np.array(detections[0, :4], dtype=np.float32)
        det2_bbox = np.array(detections[1, :4], dtype=np.float32)
        det_iou = iou(det1_bbox, det2_bbox)

        # Require separation (IoU == 0) for initialization
        if det_iou > 0:
            return False

        # Extract Re-ID features
        features = self._extract_reid_features(frame_img, detections)
        if features is None:
            features = [np.zeros(512), np.zeros(512)]

        # Assign left/right to Player 1/2
        centers = [(det[0] + det[2]) / 2 for det in detections]
        if centers[0] < centers[1]:
            order = [0, 1]  # First is left
        else:
            order = [1, 0]  # Second is left

        # Create templates
        for i, idx in enumerate(order):
            det = detections[idx]
            feat = features[idx]

            kalman = KalmanBoxTracker(det)

            self.templates[i] = PlayerTemplate(
                id=i + 1,
                initial_reid_feature=feat.copy(),
                adaptive_reid_feature=feat.copy(),
                initial_feature_sum=feat.copy(),
                initial_feature_count=1,
                kalman_tracker=kalman,
                last_bbox=det[:4].tolist(),
                last_seen_frame=frame_num,
                hit_streak=1,
                time_since_update=0,
            )

        self.initialized = True
        return True

    def _extract_reid_features(
        self,
        frame_img: np.ndarray,
        detections: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """
        Extract Re-ID features for detections

        Args:
            frame_img: BGR image
            detections: Detection array

        Returns:
            List of 512-dim feature vectors, or None if extraction fails
        """
        if len(detections) == 0:
            return []

        try:
            if self.reid_model is None:
                self.reid_model = get_reid_model(self.config)

            # Convert detections to list of tuples for reid model
            det_list = [
                (float(det[0]), float(det[1]), float(det[2]), float(det[3]))
                for det in detections
            ]

            features = self.reid_model.extract_features(frame_img, det_list)
            return features

        except Exception as e:
            # Fall back to zero features if extraction fails
            return [np.zeros(512) for _ in range(len(detections))]

    def _compute_cost_matrix(
        self,
        detections: np.ndarray,
        det_features: List[np.ndarray],
        predicted_bboxes: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute hybrid cost matrix (IoU + Re-ID similarity)

        Args:
            detections: (N, 5) detection array
            det_features: N Re-ID feature vectors
            predicted_bboxes: 2 predicted bboxes from Kalman

        Returns:
            (2, N) cost matrix (negative scores for Hungarian minimization)
        """
        n_dets = len(detections)
        cost = np.zeros((2, n_dets))

        for i, template in enumerate(self.templates):
            if template is None:
                cost[i, :] = 1e6
                continue

            pred_bbox = predicted_bboxes[i].flatten()[:4]

            for j in range(n_dets):
                det_bbox = detections[j, :4]

                # Compute IoU
                iou_val = iou(
                    np.array(pred_bbox, dtype=np.float32),
                    np.array(det_bbox, dtype=np.float32)
                )

                # Dynamic weighting based on separation state
                weights = self._compute_dynamic_weights(iou_val)

                # Re-ID similarities
                initial_sim = self._cosine_similarity(
                    template.initial_reid_feature, det_features[j]
                )
                adaptive_sim = self._cosine_similarity(
                    template.adaptive_reid_feature, det_features[j]
                )

                # Hybrid score
                score = (
                    iou_val * weights["iou"]
                    + initial_sim * weights["reid_initial"]
                    + adaptive_sim * weights["reid_adaptive"]
                )

                cost[i, j] = -score  # Hungarian minimizes

        return cost

    def _compute_dynamic_weights(self, iou_value: float) -> Dict[str, float]:
        """
        Compute dynamic weights based on IoU

        Args:
            iou_value: IoU between two players

        Returns:
            Weight dictionary for [iou, reid_initial, reid_adaptive]
        """
        if iou_value == 0:
            # Separated: use Re-ID initial template
            return {"iou": 0.0, "reid_initial": 1.0, "reid_adaptive": 0.0}
        elif iou_value < 0.5:
            # Slight overlap: still mostly Re-ID initial
            return {"iou": 0.0, "reid_initial": 1.0, "reid_adaptive": 0.0}
        else:
            # Significant overlap (clinch/ground): balance IoU and initial Re-ID
            return {"iou": 0.5, "reid_initial": 0.5, "reid_adaptive": 0.0}

    def _assign(
        self,
        cost_matrix: np.ndarray,
        detections: np.ndarray
    ) -> Dict[int, Optional[int]]:
        """
        Assign detections to players using Hungarian algorithm

        Args:
            cost_matrix: (2, N) cost matrix
            detections: Detection array

        Returns:
            {player_idx: detection_idx or None}
        """
        n_dets = len(detections)
        assignments = {0: None, 1: None}

        if n_dets == 0:
            return assignments

        # Single detection
        if n_dets == 1:
            if cost_matrix[0, 0] < cost_matrix[1, 0]:
                if -cost_matrix[0, 0] > self.config.min_score_threshold:
                    assignments[0] = 0
            else:
                if -cost_matrix[1, 0] > self.config.min_score_threshold:
                    assignments[1] = 0
            return assignments

        # Multiple detections: Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if -cost_matrix[r, c] > self.config.min_score_threshold:
                assignments[r] = c

        return assignments

    def _get_output(self) -> np.ndarray:
        """
        Get current tracking output

        Returns:
            Tracked objects: [[x1, y1, x2, y2, track_id, conf], ...]
        """
        results = []

        for template in self.templates:
            if template is None:
                continue

            # Check max_age
            if template.time_since_update > self.config.max_age:
                continue

            # Get current state
            bbox = template.kalman_tracker.get_state()
            if bbox.ndim == 2:
                bbox = bbox[0]  # Extract first element if 2D
            bbox = bbox.flatten()[:4]
            conf = template.kalman_tracker.confidence

            results.append(
                [bbox[0], bbox[1], bbox[2], bbox[3], template.id, conf]
            )

        if len(results) == 0:
            return np.empty((0, 6))

        return np.array(results)

    def reset(self) -> None:
        """Reset tracker state for new sequence"""
        self.templates = [None, None]
        self.initialized = False
        self.frame_count = 0
        KalmanBoxTracker.reset_instance()

    def is_initialized(self) -> bool:
        """Check if tracker is initialized"""
        return self.initialized

    def get_player_templates(self) -> List[Optional[PlayerTemplate]]:
        """Get current player templates"""
        return self.templates

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors

        Args:
            vec1: Feature vector
            vec2: Feature vector

        Returns:
            Similarity in [-1, 1]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        dot_product = np.dot(vec1, vec2)
        return float(dot_product / (norm1 * norm2))
