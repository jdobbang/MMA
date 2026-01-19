"""
SORT: Simple Online and Realtime Tracking

A simple and effective tracking algorithm using Kalman Filter and Hungarian Algorithm.
Replaces the original script/sort_tracker.py with modular structure.

Provides:
- Sort: Main tracking algorithm
- Data association via IoU and Hungarian matching
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Optional

from .kalman import KalmanBoxTracker
from ..detection.bbox_utils import iou_batch


class Sort:
    """
    SORT: Simple Online and Realtime Tracking

    Implements the SORT tracking algorithm using Kalman filtering for state prediction
    and Hungarian algorithm for data association.

    Example:
        >>> from mma.tracking.sort_tracker import Sort
        >>> tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        >>> for frame_num, detections in enumerate(video_detections):
        ...     # detections shape: (N, 5) = [x1, y1, x2, y2, conf]
        ...     tracked = tracker.update(detections)
        ...     # tracked shape: (M, 6) = [x1, y1, x2, y2, track_id, conf]
    """

    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize SORT tracker

        Args:
            max_age: Maximum frames to keep track alive without detections
            min_hits: Minimum associated detections before confirming track
            iou_threshold: Minimum IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(
        self,
        detections: np.ndarray
    ) -> np.ndarray:
        """
        Update tracker with new detections

        Args:
            detections: Array of shape (N, 5) with [x1, y1, x2, y2, conf]

        Returns:
            Tracked objects array of shape (M, 6) with [x1, y1, x2, y2, track_id, conf]

        Example:
            >>> detections = np.array([[10, 20, 100, 150, 0.9], [200, 50, 300, 200, 0.8]])
            >>> tracked = tracker.update(detections)
            >>> print(tracked.shape)  # (2, 6)
            >>> print(tracked[:, 4])  # Track IDs
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, tracker in enumerate(self.trackers):
            pos = tracker.predict()
            if pos.ndim == 2:
                pos = pos[0]  # Extract first element if 2D
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks
        )

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)

        # Return tracked objects
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            state = trk.get_state()
            if state.ndim == 2:
                d = state[0]  # Get first element if 2D
            else:
                d = state  # Use directly if 1D

            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                tracked = np.array([
                    d[0], d[1], d[2], d[3],
                    float(trk.id + 1),
                    float(trk.confidence)
                ], dtype=np.float32)
                ret.append(tracked.reshape(1, -1))
            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

    def _associate_detections_to_trackers(
        self,
        detections: np.ndarray,
        trackers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assign detections to tracked objects using Hungarian algorithm

        Args:
            detections: (N, 5) array of detections [x1, y1, x2, y2, conf]
            trackers: (M, 5) array of tracker predictions [x1, y1, x2, y2, 0]

        Returns:
            matched_indices: (K, 2) array of [detection_idx, tracker_idx]
            unmatched_detections: array of unmatched detection indices
            unmatched_trackers: array of unmatched tracker indices

        Algorithm:
        1. Compute pairwise IoU between detections and predicted tracks
        2. Use Hungarian algorithm to find optimal assignment
        3. Filter matches below IoU threshold
        """
        if len(trackers) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0, 5), dtype=int),
            )

        # Compute IoU matrix
        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                # Greedy matching if all assignments are unique
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Use Hungarian algorithm for optimal assignment
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack([row_ind, col_ind], axis=1)
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out matched with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def reset(self) -> None:
        """Reset tracker state (for testing or new sequence)"""
        self.trackers = []
        self.frame_count = 0

    def get_trackers(self):
        """Get current tracker list"""
        return self.trackers

    def get_num_active_trackers(self) -> int:
        """Get number of active (recently updated) trackers"""
        return sum(1 for t in self.trackers if t.time_since_update < 1)
