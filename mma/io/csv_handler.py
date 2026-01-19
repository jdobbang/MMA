"""
CSV handling utilities for MMA pipeline

Unified dataclass-based CSV I/O replacing repeated patterns in:
- run_mma_pipeline.py
- pose_estimation.py
- detection.py

Provides:
- Type-safe CSV operations with dataclasses
- Automatic CSV header handling
- Batch read/write operations
- Data validation
"""

import csv
from pathlib import Path
from dataclasses import dataclass, asdict, fields, astuple
from typing import List, Dict, Optional
from collections import defaultdict

from ..core.exceptions import DataLoadError
from ..core.constants import (
    CSV_DETECTION_COLUMNS,
    CSV_TRACKING_COLUMNS,
    COCO_KEYPOINT_NAMES,
)


@dataclass
class DetectionRow:
    """Dataclass for detection result rows"""
    image_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @classmethod
    def from_dict(cls, d: Dict) -> "DetectionRow":
        """Create instance from dictionary"""
        return cls(
            image_name=d['image_name'],
            x1=float(d['x1']),
            y1=float(d['y1']),
            x2=float(d['x2']),
            y2=float(d['y2']),
            confidence=float(d['confidence']),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_tuple(self) -> tuple:
        """Convert to tuple for CSV writing"""
        return astuple(self)


@dataclass
class TrackingRow:
    """Dataclass for tracking result rows"""
    image_name: str
    frame: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @classmethod
    def from_dict(cls, d: Dict) -> "TrackingRow":
        """Create instance from dictionary"""
        return cls(
            image_name=d['image_name'],
            frame=int(d['frame']),
            track_id=int(d['track_id']),
            x1=float(d['x1']),
            y1=float(d['y1']),
            x2=float(d['x2']),
            y2=float(d['y2']),
            confidence=float(d['confidence']),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_tuple(self) -> tuple:
        """Convert to tuple for CSV writing"""
        return astuple(self)


@dataclass
class PoseRow:
    """Dataclass for pose estimation result rows"""
    image_name: str
    frame: int
    track_id: int
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    bbox_conf: float
    # Dynamically add keypoint columns (17 keypoints * 3 values)

    def __post_init__(self):
        """Add keypoint columns dynamically"""
        # Initialize keypoint data
        self.keypoints = {}  # Dict to store per-keypoint data

    @classmethod
    def from_dict(cls, d: Dict) -> "PoseRow":
        """Create instance from dictionary"""
        row = cls(
            image_name=d['image_name'],
            frame=int(d['frame']),
            track_id=int(d['track_id']),
            bbox_x1=float(d['bbox_x1']),
            bbox_y1=float(d['bbox_y1']),
            bbox_x2=float(d['bbox_x2']),
            bbox_y2=float(d['bbox_y2']),
            bbox_conf=float(d['bbox_conf']),
        )

        # Load keypoints
        for kpt_name in COCO_KEYPOINT_NAMES:
            row.keypoints[kpt_name] = {
                'x': float(d.get(f'{kpt_name}_x', 0)),
                'y': float(d.get(f'{kpt_name}_y', 0)),
                'conf': float(d.get(f'{kpt_name}_conf', 0)),
            }

        return row

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = {
            'image_name': self.image_name,
            'frame': self.frame,
            'track_id': self.track_id,
            'bbox_x1': self.bbox_x1,
            'bbox_y1': self.bbox_y1,
            'bbox_x2': self.bbox_x2,
            'bbox_y2': self.bbox_y2,
            'bbox_conf': self.bbox_conf,
        }

        # Add keypoints
        for kpt_name in COCO_KEYPOINT_NAMES:
            kpt = self.keypoints.get(kpt_name, {'x': 0, 'y': 0, 'conf': 0})
            d[f'{kpt_name}_x'] = kpt['x']
            d[f'{kpt_name}_y'] = kpt['y']
            d[f'{kpt_name}_conf'] = kpt['conf']

        return d


class CSVWriter:
    """
    Unified CSV writing for detection, tracking, and pose results

    Replaces repeated patterns in:
    - run_mma_pipeline.py
    - detection.py
    - pose_estimation.py
    """

    @staticmethod
    def write_detections(output_path: str, detections: List[DetectionRow]) -> None:
        """
        Write detection results to CSV

        Args:
            output_path: Path to output CSV file
            detections: List of DetectionRow instances

        Example:
            >>> from mma.io import CSVWriter, DetectionRow
            >>> rows = [
            ...     DetectionRow('img1.jpg', 10, 20, 100, 110, 0.95),
            ...     DetectionRow('img2.jpg', 15, 25, 105, 115, 0.92),
            ... ]
            >>> CSVWriter.write_detections('detections.csv', rows)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_DETECTION_COLUMNS)

            for det in detections:
                writer.writerow(det.to_tuple())

    @staticmethod
    def write_tracking(output_path: str, tracks: List[TrackingRow]) -> None:
        """
        Write tracking results to CSV

        Args:
            output_path: Path to output CSV file
            tracks: List of TrackingRow instances

        Example:
            >>> from mma.io import CSVWriter, TrackingRow
            >>> rows = [
            ...     TrackingRow('img1.jpg', 1, 1, 10, 20, 100, 110, 0.95),
            ...     TrackingRow('img2.jpg', 2, 1, 15, 25, 105, 115, 0.94),
            ... ]
            >>> CSVWriter.write_tracking('tracks.csv', rows)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_TRACKING_COLUMNS)

            for track in tracks:
                writer.writerow(track.to_tuple())

    @staticmethod
    def write_pose(output_path: str, poses: List[PoseRow]) -> None:
        """
        Write pose estimation results to CSV

        Args:
            output_path: Path to output CSV file
            poses: List of PoseRow instances
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not poses:
            return

        # Get all columns from first row
        columns = list(poses[0].to_dict().keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for pose in poses:
                writer.writerow(pose.to_dict())


class CSVReader:
    """
    Unified CSV reading for detection, tracking, and pose results

    Replaces repeated patterns in:
    - run_mma_pipeline.py
    - pose_estimation.py
    """

    @staticmethod
    def read_detections(
        csv_path: str,
        conf_threshold: float = 0.0
    ) -> Dict[str, List[DetectionRow]]:
        """
        Read detection results from CSV, grouped by image name

        Replaces pattern in run_mma_pipeline.py

        Args:
            csv_path: Path to detection CSV file
            conf_threshold: Minimum confidence threshold

        Returns:
            Dictionary mapping image_name to list of DetectionRow

        Raises:
            DataLoadError: If CSV cannot be read

        Example:
            >>> from mma.io import CSVReader
            >>> detections = CSVReader.read_detections('detections.csv', conf_threshold=0.1)
            >>> for img_name, det_list in detections.items():
            ...     print(f"{img_name}: {len(det_list)} detections")
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise DataLoadError(f"CSV file not found: {csv_path}")

        detections_by_image = defaultdict(list)

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    conf = float(row['confidence'])
                    if conf < conf_threshold:
                        continue

                    det = DetectionRow.from_dict(row)
                    detections_by_image[det.image_name].append(det)

            return dict(detections_by_image)

        except Exception as e:
            raise DataLoadError(f"Failed to read detection CSV: {e}")

    @staticmethod
    def read_tracking(csv_path: str) -> Dict[int, List[TrackingRow]]:
        """
        Read tracking results from CSV, grouped by frame number

        Args:
            csv_path: Path to tracking CSV file

        Returns:
            Dictionary mapping frame_num to list of TrackingRow

        Raises:
            DataLoadError: If CSV cannot be read

        Example:
            >>> from mma.io import CSVReader
            >>> tracks = CSVReader.read_tracking('tracks.csv')
            >>> for frame_num, track_list in tracks.items():
            ...     print(f"Frame {frame_num}: {len(track_list)} tracks")
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise DataLoadError(f"CSV file not found: {csv_path}")

        tracks_by_frame = defaultdict(list)

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    track = TrackingRow.from_dict(row)
                    tracks_by_frame[track.frame].append(track)

            return dict(tracks_by_frame)

        except Exception as e:
            raise DataLoadError(f"Failed to read tracking CSV: {e}")

    @staticmethod
    def read_pose(csv_path: str) -> Dict[int, List[PoseRow]]:
        """
        Read pose estimation results from CSV, grouped by frame number

        Args:
            csv_path: Path to pose estimation CSV file

        Returns:
            Dictionary mapping frame_num to list of PoseRow

        Raises:
            DataLoadError: If CSV cannot be read
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise DataLoadError(f"CSV file not found: {csv_path}")

        poses_by_frame = defaultdict(list)

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    pose = PoseRow.from_dict(row)
                    poses_by_frame[pose.frame].append(pose)

            return dict(poses_by_frame)

        except Exception as e:
            raise DataLoadError(f"Failed to read pose CSV: {e}")
