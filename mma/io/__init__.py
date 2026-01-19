"""
IO module - Data loading and saving utilities

Provides unified interfaces for:
- Image loading with error handling
- NPY file loading (SMPL, poses2d, etc.)
- CSV reading/writing with dataclasses
- Video frame extraction
"""

from .data_loader import ImageLoader, NPYLoader
from .csv_handler import (
    CSVWriter,
    CSVReader,
    DetectionRow,
    TrackingRow,
    PoseRow,
)

__all__ = [
    "ImageLoader",
    "NPYLoader",
    "CSVWriter",
    "CSVReader",
    "DetectionRow",
    "TrackingRow",
    "PoseRow",
]
