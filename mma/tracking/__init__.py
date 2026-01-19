"""
Tracking module - Multi-object tracking with Re-ID

Provides:
- Kalman filter for bbox tracking
- Re-ID feature extraction (OSNet)
- Track interpolation for gap filling
- SORT algorithm
- MMA-specific tracker
"""

from .kalman import KalmanBoxTracker
from .reid import ReIDModel, get_reid_model, cosine_similarity, cosine_similarity_batch
from .interpolation import (
    interpolate_track,
    interpolate_tracks_batch,
    fill_frame_gaps,
    validate_interpolation,
    get_interpolation_stats,
)
from .sort_tracker import Sort
from .mma_tracker import MMATracker, PlayerTemplate

__all__ = [
    # Kalman
    "KalmanBoxTracker",
    # Re-ID
    "ReIDModel",
    "get_reid_model",
    "cosine_similarity",
    "cosine_similarity_batch",
    # Interpolation
    "interpolate_track",
    "interpolate_tracks_batch",
    "fill_frame_gaps",
    "validate_interpolation",
    "get_interpolation_stats",
    # SORT tracking
    "Sort",
    # MMA tracking
    "MMATracker",
    "PlayerTemplate",
]
