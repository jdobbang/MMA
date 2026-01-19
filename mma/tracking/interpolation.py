"""
Track interpolation for filling gaps in tracking results

Replaces:
- mma_tracker.py: interpolate_track() function
- run_mma_pipeline.py: run_interpolation() logic

Provides:
- Linear interpolation for bbox sequences
- Batch interpolation for multiple tracks
- Gap detection and filling
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


def interpolate_bbox(
    bbox1: Tuple[float, float, float, float, float],
    bbox2: Tuple[float, float, float, float, float],
    alpha: float
) -> Tuple[float, float, float, float, float]:
    """
    Linear interpolation between two bboxes

    Args:
        bbox1: Source bbox (x1, y1, x2, y2, conf)
        bbox2: Target bbox (x1, y1, x2, y2, conf)
        alpha: Interpolation factor in [0, 1]
               0 = bbox1, 1 = bbox2

    Returns:
        Interpolated bbox
    """
    x1 = bbox1[0] * (1 - alpha) + bbox2[0] * alpha
    y1 = bbox1[1] * (1 - alpha) + bbox2[1] * alpha
    x2 = bbox1[2] * (1 - alpha) + bbox2[2] * alpha
    y2 = bbox1[3] * (1 - alpha) + bbox2[3] * alpha
    conf = bbox1[4] * (1 - alpha) + bbox2[4] * alpha

    return (x1, y1, x2, y2, conf)


def interpolate_track(
    track_data: List[Tuple[int, Tuple[float, float, float, float, float]]],
    max_gap: int = 30
) -> List[Tuple[int, Tuple[float, float, float, float, float]]]:
    """
    Interpolate missing frames in track data

    Replaces:
    - mma_tracker.py: interpolate_track()

    Fills gaps in tracking sequences using linear interpolation between
    the first and last visible detection in a gap.

    Args:
        track_data: List of (frame_num, bbox_tuple) sorted by frame
        max_gap: Maximum frame gap to interpolate (frames > max_gap are skipped)

    Returns:
        Interpolated track data with all frames filled

    Example:
        >>> track = [
        ...     (0, (100, 100, 200, 200, 0.9)),
        ...     (3, (120, 120, 220, 220, 0.85))
        ... ]
        >>> interpolated = interpolate_track(track, max_gap=5)
        >>> print(len(interpolated))  # 4 frames
        >>> for frame, bbox in interpolated:
        ...     print(f"Frame {frame}: {bbox}")
    """
    if len(track_data) < 2:
        return track_data

    # Sort by frame number
    track_data = sorted(track_data, key=lambda x: x[0])

    interpolated = []
    current_idx = 0

    for i in range(len(track_data) - 1):
        frame_curr, bbox_curr = track_data[i]
        frame_next, bbox_next = track_data[i + 1]

        # Add current frame
        interpolated.append((frame_curr, bbox_curr))

        # Calculate gap
        gap = frame_next - frame_curr - 1

        # Fill gap if within max_gap
        if gap > 0 and gap <= max_gap:
            for gap_idx in range(1, gap + 1):
                frame_interp = frame_curr + gap_idx
                # Linear interpolation factor
                alpha = gap_idx / (gap + 1)
                bbox_interp = interpolate_bbox(bbox_curr, bbox_next, alpha)
                interpolated.append((frame_interp, bbox_interp))

    # Add last frame
    if len(track_data) > 0:
        interpolated.append(track_data[-1])

    return sorted(interpolated, key=lambda x: x[0])


def interpolate_tracks_batch(
    tracks_dict: Dict[int, List[Tuple[int, Tuple[float, float, float, float, float]]]],
    max_gap: int = 30
) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float, float]]]]:
    """
    Interpolate multiple tracks

    Args:
        tracks_dict: Dictionary mapping track_id to track_data
        max_gap: Maximum gap to interpolate

    Returns:
        Dictionary with interpolated tracks

    Example:
        >>> tracks = {
        ...     1: [(0, bbox0), (3, bbox3)],
        ...     2: [(1, bbox1), (2, bbox2)],
        ... }
        >>> interpolated = interpolate_tracks_batch(tracks, max_gap=5)
        >>> for track_id, track in interpolated.items():
        ...     print(f"Track {track_id}: {len(track)} frames")
    """
    interpolated_tracks = {}

    for track_id, track_data in tracks_dict.items():
        interpolated_tracks[track_id] = interpolate_track(track_data, max_gap)

    return interpolated_tracks


def fill_frame_gaps(
    detections_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float, float]]]],
    track_id: int,
    max_gap: int = 30
) -> List[Tuple[int, Tuple[float, float, float, float, float]]]:
    """
    Extract and interpolate specific track from frame-indexed detections

    Args:
        detections_by_frame: Dictionary mapping frame -> list of detections
        track_id: Track ID to extract and interpolate
        max_gap: Maximum gap for interpolation

    Returns:
        Interpolated track data
    """
    # Extract track data for this track_id
    track_data = []
    for frame_num, detections in sorted(detections_by_frame.items()):
        for det in detections:
            if det[0] == track_id:  # det[0] is track_id
                # det format: (track_id, (x1, y1, x2, y2, conf))
                bbox = det[1]
                track_data.append((frame_num, bbox))

    # Sort by frame
    track_data = sorted(track_data, key=lambda x: x[0])

    # Interpolate
    return interpolate_track(track_data, max_gap)


def validate_interpolation(
    original: List[Tuple[int, Tuple[float, float, float, float, float]]],
    interpolated: List[Tuple[int, Tuple[float, float, float, float, float]]]
) -> bool:
    """
    Validate interpolation results

    Args:
        original: Original track data
        interpolated: Interpolated track data

    Returns:
        True if interpolation is valid
    """
    # Check frame continuity
    frames_interp = [f for f, _ in interpolated]
    frames_orig = [f for f, _ in original]

    # All original frames must be present
    if not all(f in frames_interp for f in frames_orig):
        return False

    # Check monotonic increase
    if frames_interp != sorted(frames_interp):
        return False

    # Check no duplicates
    if len(frames_interp) != len(set(frames_interp)):
        return False

    return True


def get_interpolation_stats(
    original: List[Tuple[int, Tuple[float, float, float, float, float]]],
    interpolated: List[Tuple[int, Tuple[float, float, float, float, float]]]
) -> Dict:
    """
    Compute statistics about interpolation

    Args:
        original: Original track data
        interpolated: Interpolated track data

    Returns:
        Statistics dictionary
    """
    frames_orig = set(f for f, _ in original)
    frames_interp = set(f for f, _ in interpolated)

    frames_added = frames_interp - frames_orig

    # Compute mean displacement from interpolated frames
    if len(frames_added) > 0:
        displacements = []
        bboxes_dict = {f: b for f, b in interpolated}

        for frame in frames_added:
            bbox = bboxes_dict[frame]
            # Use center for displacement
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            displacements.append(np.sqrt(cx**2 + cy**2))

        mean_disp = np.mean(displacements) if displacements else 0
    else:
        mean_disp = 0

    return {
        'num_original': len(original),
        'num_interpolated': len(interpolated),
        'num_frames_added': len(frames_added),
        'mean_displacement': mean_disp,
        'valid': validate_interpolation(original, interpolated),
    }
