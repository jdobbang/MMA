#!/usr/bin/env python3
"""
4-Stage Soccer Tracking Pipeline with Re-ID
============================================

Pipeline:
1. SORT Tracking (short tracklets)
2. Tracklet Interpolation (fill gaps within same ID)
3. Re-ID based Merging (merge different IDs)
4. Post-Merge Interpolation (fill gaps after merging)

Outputs intermediate CSV results at each stage.
"""

import csv
import os
import argparse
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np
import cv2

# Import SORT tracker from existing implementation
from sort_tracker import Sort


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Tracklet:
    """Represents a single tracklet (sequence of detections with same ID)"""
    id: int
    frames: List[int] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)  # [[x1,y1,x2,y2], ...]
    confidences: List[float] = field(default_factory=list)
    reid_features: List[np.ndarray] = field(default_factory=list)  # 512-dim vectors

    def __len__(self):
        return len(self.frames)

    def add_detection(self, frame: int, bbox: List[float], confidence: float):
        """Add a detection to this tracklet"""
        self.frames.append(frame)
        self.bboxes.append(bbox)
        self.confidences.append(confidence)


# ============================================================================
# Stage 0: Helper Functions
# ============================================================================

def parse_image_name(image_name: str) -> Tuple[str, int]:
    """
    Parse image name to extract sequence name and frame number.

    Args:
        image_name: Image filename like '03_grappling2_001_grappling2_cam01_00001.jpg'

    Returns:
        Tuple of (sequence_name, frame_number)
        Example: ('03_grappling2_001_grappling2_cam01', 1)
    """
    # Remove extension
    base_name = os.path.splitext(image_name)[0]

    # Split by underscore and extract last part as frame number
    parts = base_name.rsplit('_', 1)
    if len(parts) == 2:
        sequence_name = parts[0]
        try:
            frame_number = int(parts[1])
            return sequence_name, frame_number
        except ValueError:
            pass

    # Fallback: return whole name as sequence, frame 0
    return base_name, 0


def load_detections_by_sequence(
    csv_path: str,
    images_dir: str = None,
    conf_threshold: float = 0.1
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, str]]]:
    """
    Load detection results from CSV file, grouped by sequence.

    Args:
        csv_path: Path to detection CSV file with image_name column
        images_dir: Directory containing images (for large bbox filtering)
        conf_threshold: Minimum confidence threshold for detections

    Returns:
        Tuple of:
        - detections_by_sequence: {sequence_name: {frame_num: detections_array}}
        - image_name_map: {sequence_name: {frame_num: image_name}}
    """
    print(f"Loading detections from {csv_path}...")
    if conf_threshold > 0.0:
        print(f"Applying confidence threshold: {conf_threshold}")

    # Get frame dimensions for large bbox filtering (using first image)
    frame_width, frame_height = None, None
    max_bbox_area = None

    detections_by_sequence = defaultdict(lambda: defaultdict(list))
    image_name_map = defaultdict(dict)
    total_loaded = 0
    filtered_by_conf = 0
    filtered_by_large_low_conf = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row['image_name']
            sequence_name, frame_num = parse_image_name(image_name)

            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])
            confidence = float(row['confidence'])

            total_loaded += 1

            # Filter by confidence threshold
            if confidence < conf_threshold:
                filtered_by_conf += 1
                continue

            # Lazy load frame dimensions from first image
            if images_dir is not None and frame_width is None:
                first_image_path = os.path.join(images_dir, image_name)
                if os.path.exists(first_image_path):
                    img = cv2.imread(first_image_path)
                    if img is not None:
                        frame_height, frame_width = img.shape[:2]
                        frame_area = frame_width * frame_height
                        max_bbox_area = frame_area * 0.1  # 10% of frame
                        print(f"Detected frame dimensions: {frame_width}x{frame_height}")
                        print(f"Large bbox filter: removing detections > {max_bbox_area:.0f} pixels")

            # Filter: Large bbox AND low confidence
            if max_bbox_area is not None:
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area > max_bbox_area and confidence < 0.5:
                    filtered_by_large_low_conf += 1
                    continue

            # Store detection
            detection = [x1, y1, x2, y2, confidence]
            detections_by_sequence[sequence_name][frame_num].append(detection)
            image_name_map[sequence_name][frame_num] = image_name

    # Convert lists to numpy arrays
    for seq_name in detections_by_sequence:
        for frame_num in detections_by_sequence[seq_name]:
            detections_by_sequence[seq_name][frame_num] = np.array(
                detections_by_sequence[seq_name][frame_num]
            )

    # Convert defaultdicts to regular dicts
    detections_by_sequence = {k: dict(v) for k, v in detections_by_sequence.items()}
    image_name_map = {k: dict(v) for k, v in image_name_map.items()}

    # Print statistics
    total_sequences = len(detections_by_sequence)
    total_detections = sum(
        sum(len(dets) for dets in seq.values())
        for seq in detections_by_sequence.values()
    )
    print(f"Loaded {total_detections} detections across {total_sequences} sequences")
    for seq_name in sorted(detections_by_sequence.keys()):
        seq_frames = len(detections_by_sequence[seq_name])
        seq_dets = sum(len(dets) for dets in detections_by_sequence[seq_name].values())
        print(f"  - {seq_name}: {seq_frames} frames, {seq_dets} detections")

    if filtered_by_conf > 0:
        print(f"Filtered out {filtered_by_conf} detections below confidence threshold")
    if filtered_by_large_low_conf > 0:
        print(f"Filtered out {filtered_by_large_low_conf} large+low-confidence detections")

    return detections_by_sequence, image_name_map


def get_frame_dimensions(frames_dir, start_frame=0):
    """
    Get frame dimensions by reading the first available frame image.

    Args:
        frames_dir: Directory containing frame images
        start_frame: Starting frame number to search from

    Returns:
        (width, height) tuple
    """
    # Try multiple frame naming patterns
    for frame_num in range(start_frame, start_frame + 10):
        frame_path = os.path.join(frames_dir, f"frame_{frame_num:06d}.jpg")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(frames_dir, f"frame_{frame_num}.jpg")

        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            if img is not None:
                height, width = img.shape[:2]
                print(f"Detected frame dimensions: {width}x{height} from {frame_path}")
                return width, height

    # Fallback default (HD resolution)
    print("Warning: Could not detect frame size, using default 1920x1080")
    return 1920, 1080


# ============================================================================
# Stage 1: SORT Tracking
# ============================================================================

def load_detections_from_csv(csv_path, frames_dir=None, start_frame=None, end_frame=None, conf_threshold=0.1):
    """
    Load detection results from CSV file.

    Args:
        csv_path: Path to detection CSV file
        frames_dir: Directory containing frame images (for frame size detection)
        start_frame: Optional start frame (inclusive)
        end_frame: Optional end frame (inclusive)
        conf_threshold: Minimum confidence threshold for detections

    Returns:
        detections_by_frame: Dictionary mapping frame numbers to detection arrays
    """
    print(f"Loading detections from {csv_path}...")
    if start_frame is not None or end_frame is not None:
        frame_range = f"frames {start_frame or 0} to {end_frame or 'end'}"
        print(f"Filtering {frame_range}")
    if conf_threshold > 0.0:
        print(f"Applying confidence threshold: {conf_threshold}")

    # Get frame dimensions for large bbox filtering
    frame_width, frame_height = None, None
    if frames_dir is not None:
        frame_width, frame_height = get_frame_dimensions(frames_dir, start_frame or 0)
        frame_area = frame_width * frame_height
        max_bbox_area = frame_area * 0.1  # 10% of frame
        print(f"Large bbox filter: removing detections > {max_bbox_area:.0f} pixels ({frame_width}x{frame_height} * 0.1)")
        print(f"Low confidence filter: removing large detections with confidence < 0.5")

    detections_by_frame = defaultdict(list)
    total_loaded = 0
    filtered_by_conf = 0
    filtered_by_large_low_conf = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row['frame'])

            # Filter by frame range if specified
            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num > end_frame:
                continue

            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])
            confidence = float(row['confidence'])

            total_loaded += 1

            # Filter by confidence threshold
            if confidence < conf_threshold:
                filtered_by_conf += 1
                continue

            # Filter: Large bbox AND low confidence
            if frame_width is not None and frame_height is not None:
                bbox_area = (x2 - x1) * (y2 - y1)
                # Remove if bbox is large (>10% of frame) AND confidence is low (<0.5)
                if bbox_area > max_bbox_area and confidence < 0.5:
                    filtered_by_large_low_conf += 1
                    continue

            # Format: [x1, y1, x2, y2, confidence]
            detection = [x1, y1, x2, y2, confidence]
            detections_by_frame[frame_num].append(detection)

    # Convert lists to numpy arrays
    for frame_num in detections_by_frame:
        detections_by_frame[frame_num] = np.array(detections_by_frame[frame_num])

    total_detections = sum(len(dets) for dets in detections_by_frame.values())
    print(f"Loaded {total_detections} detections across {len(detections_by_frame)} frames")
    if filtered_by_conf > 0:
        print(f"Filtered out {filtered_by_conf} detections below confidence threshold ({conf_threshold})")
    if filtered_by_large_low_conf > 0:
        print(f"Filtered out {filtered_by_large_low_conf} large+low-confidence detections (>10% frame area AND confidence<0.5)")

    return detections_by_frame


def run_sort_tracking(detections_by_frame, max_age=30, min_hits=3, iou_threshold=0.3):
    """
    Run SORT tracking on detections.

    Args:
        detections_by_frame: Dictionary mapping frame numbers to detection arrays
        max_age: Maximum number of frames to keep alive a track without detections
        min_hits: Minimum number of associated detections before track is confirmed
        iou_threshold: Minimum IoU for matching detections to tracks

    Returns:
        tracks_by_frame: Dictionary mapping frame numbers to tracking results
    """
    print(f"\n=== Stage 1: SORT Tracking ===")
    print(f"Parameters: max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")

    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    tracks_by_frame = {}

    # Get all frame numbers and sort them
    frame_numbers = sorted(detections_by_frame.keys())

    if not frame_numbers:
        print("No detections found!")
        return tracks_by_frame

    # Process each frame
    for frame_num in tqdm(frame_numbers, desc="SORT Tracking"):
        # Get detections for this frame
        detections = detections_by_frame.get(frame_num, np.empty((0, 5)))

        # Update tracker
        tracks = tracker.update(detections)

        # Store tracking results
        # tracks format: [[x1, y1, x2, y2, track_id, confidence], ...]
        tracks_by_frame[frame_num] = tracks

    total_tracks = sum(len(tracks) for tracks in tracks_by_frame.values())
    unique_track_ids = set()
    for tracks in tracks_by_frame.values():
        for track in tracks:
            unique_track_ids.add(int(track[4]))

    print(f"Tracking complete: {total_tracks} total tracks, {len(unique_track_ids)} unique IDs")

    return tracks_by_frame


def save_tracking_to_csv(tracks_by_frame, output_csv):
    """
    Save tracking results to CSV file.

    Args:
        tracks_by_frame: Dictionary mapping frame numbers to tracking results
        output_csv: Path to output CSV file
    """
    print(f"Saving to {output_csv}...")

    # Create output directory if needed
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])

        # Write tracking results
        frame_numbers = sorted(tracks_by_frame.keys())
        for frame_num in frame_numbers:
            tracks = tracks_by_frame[frame_num]
            for track in tracks:
                x1, y1, x2, y2, track_id, confidence = track
                writer.writerow([
                    int(frame_num),
                    int(track_id),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(confidence)
                ])

    print(f"Saved: {output_csv}")


# ============================================================================
# Stage 2: Tracklet Interpolation
# ============================================================================

def csv_to_tracklets(csv_path) -> Dict[int, Tracklet]:
    """
    Convert frame-by-frame CSV to tracklet dictionary.

    Args:
        csv_path: Path to tracking CSV file

    Returns:
        Dictionary mapping track_id to Tracklet object
    """
    tracklets_dict = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            track_id = int(row['track_id'])
            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])
            confidence = float(row['confidence'])

            if track_id not in tracklets_dict:
                tracklets_dict[track_id] = Tracklet(id=track_id)

            tracklets_dict[track_id].add_detection(frame, [x1, y1, x2, y2], confidence)

    return tracklets_dict


def tracklets_to_csv(tracklets: Dict[int, Tracklet], output_path: str):
    """
    Convert tracklets back to frame-by-frame CSV format.

    Args:
        tracklets: Dictionary of tracklets
        output_path: Output CSV file path
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])

        # Collect all (frame, track_id) pairs and sort
        all_entries = []
        for track_id, tracklet in tracklets.items():
            for i, frame in enumerate(tracklet.frames):
                bbox = tracklet.bboxes[i]
                conf = tracklet.confidences[i]
                all_entries.append((frame, track_id, bbox, conf))

        # Sort by frame, then track_id
        all_entries.sort(key=lambda x: (x[0], x[1]))

        # Write sorted entries
        for frame, track_id, bbox, conf in all_entries:
            writer.writerow([frame, track_id, bbox[0], bbox[1], bbox[2], bbox[3], conf])


def interpolate_tracklet(tracklet: Tracklet, max_gap: int = 30) -> Tracklet:
    """
    Interpolate gaps within a single tracklet using linear interpolation.

    Args:
        tracklet: Input tracklet
        max_gap: Maximum gap size to interpolate (frames)

    Returns:
        Tracklet with interpolated frames
    """
    if len(tracklet) <= 1:
        return tracklet

    frames = list(tracklet.frames)
    bboxes = list(tracklet.bboxes)
    confidences = list(tracklet.confidences)

    i = 0
    while i < len(frames) - 1:
        current_frame = frames[i]
        next_frame = frames[i + 1]
        gap_size = next_frame - current_frame - 1

        # Check if gap needs interpolation
        if gap_size > 0 and gap_size <= max_gap:
            # Linear interpolation
            current_bbox = np.array(bboxes[i])
            next_bbox = np.array(bboxes[i + 1])
            current_conf = confidences[i]
            next_conf = confidences[i + 1]

            # Insert interpolated frames
            for t in range(1, gap_size + 1):
                alpha = t / (gap_size + 1)
                interpolated_bbox = (1 - alpha) * current_bbox + alpha * next_bbox
                interpolated_conf = (1 - alpha) * current_conf + alpha * next_conf

                # Insert at position i + t
                frames.insert(i + t, current_frame + t)
                bboxes.insert(i + t, interpolated_bbox.tolist())
                confidences.insert(i + t, interpolated_conf)

            i += gap_size + 1
        else:
            i += 1

    # Create new tracklet with interpolated data
    interpolated = Tracklet(id=tracklet.id)
    interpolated.frames = frames
    interpolated.bboxes = bboxes
    interpolated.confidences = confidences

    return interpolated


def run_interpolation(input_csv: str, output_csv: str, max_gap: int = 30):
    """
    Run tracklet interpolation on tracking results.

    Args:
        input_csv: Input tracking CSV
        output_csv: Output CSV with interpolated tracklets
        max_gap: Maximum gap to interpolate (frames)
    """
    print(f"\n=== Stage 2: Tracklet Interpolation ===")
    print(f"Parameters: max_gap={max_gap}")

    # Load tracklets
    print(f"Loading tracklets from {input_csv}...")
    tracklets = csv_to_tracklets(input_csv)

    original_count = sum(len(t) for t in tracklets.values())
    print(f"Loaded {len(tracklets)} tracklets with {original_count} total detections")

    # Interpolate each tracklet
    interpolated_tracklets = {}
    gaps_filled = 0

    for track_id, tracklet in tqdm(tracklets.items(), desc="Interpolating"):
        original_len = len(tracklet)
        interpolated = interpolate_tracklet(tracklet, max_gap)
        interpolated_tracklets[track_id] = interpolated

        if len(interpolated) > original_len:
            gaps_filled += len(interpolated) - original_len

    new_count = sum(len(t) for t in interpolated_tracklets.values())
    print(f"Interpolation complete: {new_count} total detections (+{new_count - original_count}, {gaps_filled} gaps filled)")

    # Save results
    tracklets_to_csv(interpolated_tracklets, output_csv)
    print(f"Saved: {output_csv}")

    return interpolated_tracklets


# ============================================================================
# Stage 3: Re-ID Feature Extraction
# ============================================================================

def load_reid_model(device='cuda'):
    """
    Load OSNet Re-ID model using torchreid.

    Args:
        device: 'cuda' or 'cpu'

    Returns:
        FeatureExtractor object
    """
    try:
        import torch
        import torchreid

        print(f"Loading OSNet model (device={device})...")

        # Build OSNet model
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True,
            use_gpu=(device == 'cuda')
        )

        model.eval()

        if device == 'cuda':
            model = model.cuda()

        print("OSNet model loaded successfully")
        return model

    except ImportError as e:
        print(f"Error: torchreid not installed. Please run: pip install Cython && pip install git+https://github.com/KaiyangZhou/deep-person-reid.git")
        raise e


def extract_reid_features(
    tracking_csv: str,
    frames_dir: str,
    output_pkl: str,
    device='cuda',
    batch_size=32,
    image_name_map: Dict[int, str] = None
):
    """
    Extract Re-ID features for all tracklets.

    Args:
        tracking_csv: Path to tracking CSV
        frames_dir: Directory containing frame images
        output_pkl: Output pickle file path
        device: 'cuda' or 'cpu'
        batch_size: Batch size for feature extraction
        image_name_map: Optional mapping from frame number to image filename
                        If provided, uses this to locate images instead of frame_{frame:06d}.jpg
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image

    print(f"\n=== Stage 3a: Re-ID Feature Extraction ===")
    print(f"Parameters: device={device}, batch_size={batch_size}")
    if image_name_map:
        print(f"Using image_name_map with {len(image_name_map)} frame mappings")

    # Load Re-ID model
    model = load_reid_model(device)

    # Preprocessing transform
    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load tracking data
    print(f"Loading tracking data from {tracking_csv}...")
    tracklets = csv_to_tracklets(tracking_csv)

    # Count total images to process
    total_images = sum(len(t) for t in tracklets.values())
    print(f"Extracting features from {total_images} images...")

    # Extract features for each tracklet
    reid_features = {}

    # Process in batches for efficiency
    batch_images = []
    batch_keys = []

    with torch.no_grad():
        with tqdm(total=total_images, desc="Extracting Re-ID features") as pbar:
            for track_id, tracklet in tracklets.items():
                for i, frame in enumerate(tracklet.frames):
                    bbox = tracklet.bboxes[i]

                    # Determine frame path
                    if image_name_map and frame in image_name_map:
                        # Use image_name_map for sequence-based images
                        frame_path = os.path.join(frames_dir, image_name_map[frame])
                    else:
                        # Fallback to default naming convention
                        frame_path = os.path.join(frames_dir, f"frame_{frame:06d}.jpg")
                        if not os.path.exists(frame_path):
                            frame_path = os.path.join(frames_dir, f"frame_{frame}.jpg")

                    if not os.path.exists(frame_path):
                        print(f"Warning: Frame {frame_path} not found, skipping")
                        pbar.update(1)
                        continue

                    # Read and crop image
                    img = cv2.imread(frame_path)
                    if img is None:
                        print(f"Warning: Could not read {frame_path}, skipping")
                        pbar.update(1)
                        continue

                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                    if x2 <= x1 or y2 <= y1:
                        print(f"Warning: Invalid bbox {bbox}, skipping")
                        pbar.update(1)
                        continue

                    crop = img[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_pil = Image.fromarray(crop_rgb)

                    # Transform and add to batch
                    crop_tensor = transform(crop_pil)
                    batch_images.append(crop_tensor)
                    batch_keys.append((frame, track_id))

                    # Process batch if full
                    if len(batch_images) >= batch_size:
                        batch_tensor = torch.stack(batch_images)
                        if device == 'cuda':
                            batch_tensor = batch_tensor.cuda()

                        features = model(batch_tensor)

                        for j, key in enumerate(batch_keys):
                            reid_features[key] = features[j].cpu().numpy()

                        batch_images = []
                        batch_keys = []
                        pbar.update(batch_size)

            # Process remaining batch
            if batch_images:
                batch_tensor = torch.stack(batch_images)
                if device == 'cuda':
                    batch_tensor = batch_tensor.cuda()

                features = model(batch_tensor)

                for j, key in enumerate(batch_keys):
                    reid_features[key] = features[j].cpu().numpy()
                pbar.update(len(batch_images))

    # Save features
    print(f"Saving Re-ID features to {output_pkl}...")
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(reid_features, f)

    print(f"Extracted {len(reid_features)} Re-ID features")
    print(f"Saved: {output_pkl}")

    return reid_features


def _worker_extract_features(
    worker_id: int,
    tracklet_items: List[Tuple[int, 'Tracklet']],
    frames_dir: str,
    batch_size: int,
    result_queue,
    total_images: int,
    image_name_map: Dict[int, str] = None
):
    """
    Worker function for parallel Re-ID feature extraction.
    Each worker loads its own model and processes a subset of tracklets.
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image

    # Load model for this worker
    model = load_reid_model('cuda')

    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    reid_features = {}
    batch_images = []
    batch_keys = []
    processed = 0

    with torch.no_grad():
        for track_id, tracklet in tracklet_items:
            for i, frame in enumerate(tracklet.frames):
                bbox = tracklet.bboxes[i]

                # Determine frame path
                if image_name_map and frame in image_name_map:
                    frame_path = os.path.join(frames_dir, image_name_map[frame])
                else:
                    frame_path = os.path.join(frames_dir, f"frame_{frame:06d}.jpg")
                    if not os.path.exists(frame_path):
                        frame_path = os.path.join(frames_dir, f"frame_{frame}.jpg")

                if not os.path.exists(frame_path):
                    processed += 1
                    continue

                # Read and crop image
                img = cv2.imread(frame_path)
                if img is None:
                    processed += 1
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    processed += 1
                    continue

                crop = img[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)

                crop_tensor = transform(crop_pil)
                batch_images.append(crop_tensor)
                batch_keys.append((frame, track_id))

                # Process batch if full
                if len(batch_images) >= batch_size:
                    batch_tensor = torch.stack(batch_images).cuda()
                    features = model(batch_tensor)

                    for j, key in enumerate(batch_keys):
                        reid_features[key] = features[j].cpu().numpy()

                    processed += len(batch_images)
                    batch_images = []
                    batch_keys = []

        # Process remaining batch
        if batch_images:
            batch_tensor = torch.stack(batch_images).cuda()
            features = model(batch_tensor)

            for j, key in enumerate(batch_keys):
                reid_features[key] = features[j].cpu().numpy()
            processed += len(batch_images)

    # Put results in queue
    result_queue.put((worker_id, reid_features))
    print(f"Worker {worker_id}: Completed {len(reid_features)} features")


def extract_reid_features_parallel(
    tracking_csv: str,
    frames_dir: str,
    output_pkl: str,
    device='cuda',
    batch_size=32,
    num_workers=8,
    image_name_map: Dict[int, str] = None
):
    """
    Extract Re-ID features using multiple parallel workers.

    Each worker loads its own model instance (~800MiB GPU memory each).
    With 25GB VRAM, can safely run 8-10 workers in parallel.

    Args:
        tracking_csv: Path to tracking CSV
        frames_dir: Directory containing frame images
        output_pkl: Output pickle file path
        device: 'cuda' or 'cpu' (parallel mode only supports cuda)
        batch_size: Batch size per worker
        num_workers: Number of parallel workers
        image_name_map: Optional mapping from frame number to image filename
    """
    import torch.multiprocessing as mp

    print(f"\n=== Stage 3a: Re-ID Feature Extraction (Parallel) ===")
    print(f"Parameters: num_workers={num_workers}, batch_size={batch_size}")

    # Fall back to single-threaded if CPU or single worker
    if device != 'cuda' or num_workers <= 1:
        print("Falling back to single-threaded extraction...")
        return extract_reid_features(tracking_csv, frames_dir, output_pkl, device, batch_size, image_name_map)

    # Load tracking data
    print(f"Loading tracking data from {tracking_csv}...")
    tracklets = csv_to_tracklets(tracking_csv)

    total_images = sum(len(t) for t in tracklets.values())
    print(f"Total images to process: {total_images}")

    # Split tracklets among workers
    tracklet_items = list(tracklets.items())
    chunk_size = (len(tracklet_items) + num_workers - 1) // num_workers

    chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(tracklet_items))
        if start_idx < len(tracklet_items):
            chunks.append(tracklet_items[start_idx:end_idx])

    actual_workers = len(chunks)
    print(f"Splitting {len(tracklet_items)} tracklets across {actual_workers} workers")

    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Create result queue and launch workers
    result_queue = mp.Queue()
    processes = []

    for i, chunk in enumerate(chunks):
        chunk_images = sum(len(t) for _, t in chunk)
        p = mp.Process(
            target=_worker_extract_features,
            args=(i, chunk, frames_dir, batch_size, result_queue, chunk_images, image_name_map)
        )
        p.start()
        processes.append(p)
        print(f"Started worker {i} with {len(chunk)} tracklets ({chunk_images} images)")

    # Collect results
    all_features = {}
    for _ in range(actual_workers):
        worker_id, features = result_queue.get()
        all_features.update(features)
        print(f"Received results from worker {worker_id}: {len(features)} features")

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Save features
    print(f"Saving Re-ID features to {output_pkl}...")
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_features, f)

    print(f"Extracted {len(all_features)} Re-ID features (parallel)")
    print(f"Saved: {output_pkl}")

    return all_features


# ============================================================================
# Stage 4: Re-ID Based Merging
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1, vec2: Feature vectors

    Returns:
        Similarity score (0-1, higher is more similar)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def compute_tracklet_avg_feature(tracklet: Tracklet, reid_features: Dict) -> Optional[np.ndarray]:
    """
    Compute average Re-ID feature for a tracklet.

    Args:
        tracklet: Tracklet object
        reid_features: Dictionary of (frame, track_id) -> feature vector

    Returns:
        Average feature vector, or None if no features found
    """
    features = []

    for frame in tracklet.frames:
        key = (frame, tracklet.id)
        if key in reid_features:
            features.append(reid_features[key])

    if not features:
        return None

    return np.mean(features, axis=0)


def merge_tracklets_by_reid(
    tracklets: Dict[int, Tracklet],
    reid_features: Dict,
    similarity_threshold: float = 0.7,
    max_time_gap: int = 150
) -> Dict[int, Tracklet]:
    """
    Merge tracklets based on Re-ID similarity.

    Args:
        tracklets: Dictionary of tracklets
        reid_features: Re-ID features dictionary
        similarity_threshold: Minimum cosine similarity to merge
        max_time_gap: Maximum time gap between tracklets (frames)

    Returns:
        Merged tracklets with reassigned IDs
    """
    print(f"\n=== Stage 3b: Re-ID Based Merging ===")
    print(f"Parameters: similarity_threshold={similarity_threshold}, max_time_gap={max_time_gap}")

    # Sort tracklets by start frame
    sorted_tracklets = sorted(tracklets.items(), key=lambda x: x[1].frames[0])

    # Compute average features for all tracklets
    print("Computing average Re-ID features for tracklets...")
    tracklet_features = {}
    for track_id, tracklet in tqdm(sorted_tracklets, desc="Computing features"):
        avg_feat = compute_tracklet_avg_feature(tracklet, reid_features)
        if avg_feat is not None:
            tracklet_features[track_id] = avg_feat

    print(f"Computed features for {len(tracklet_features)}/{len(tracklets)} tracklets")

    # Merge tracklets
    merged_tracklets = {}
    used = set()
    merge_count = 0

    print("Merging similar tracklets...")
    for i, (track_id1, tracklet1) in enumerate(tqdm(sorted_tracklets, desc="Merging")):
        if track_id1 in used:
            continue

        # Get feature for tracklet1
        if track_id1 not in tracklet_features:
            merged_tracklets[track_id1] = tracklet1
            continue

        feat1 = tracklet_features[track_id1]
        track1_end = tracklet1.frames[-1]

        # Try to merge with later tracklets
        merged = False
        for track_id2, tracklet2 in sorted_tracklets[i+1:]:
            if track_id2 in used:
                continue

            # Get feature for tracklet2
            if track_id2 not in tracklet_features:
                continue

            feat2 = tracklet_features[track_id2]
            track2_start = tracklet2.frames[0]

            # Check temporal conditions
            # Track1 must end before track2 starts (no overlap)
            if track1_end >= track2_start:
                continue

            # Time gap must be within limit
            time_gap = track2_start - track1_end
            if time_gap > max_time_gap:
                continue

            # Check spatial distance (position constraint)
            # Compare last bbox of track1 with first bbox of track2
            last_bbox1 = tracklet1.bboxes[-1]  # [x1, y1, x2, y2]
            first_bbox2 = tracklet2.bboxes[0]

            # Calculate center points
            center1 = [(last_bbox1[0] + last_bbox1[2]) / 2, (last_bbox1[1] + last_bbox1[3]) / 2]
            center2 = [(first_bbox2[0] + first_bbox2[2]) / 2, (first_bbox2[1] + first_bbox2[3]) / 2]

            # Euclidean distance between centers
            spatial_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

            # Maximum allowed distance (pixels)
            # Adaptive threshold: distance increases with time gap
            # Base: 200 pixels, +2 pixels per frame gap
            max_spatial_distance = 200 + (time_gap * 2)

            if spatial_distance > max_spatial_distance:
                continue  # Too far apart spatially

            # Check Re-ID similarity
            similarity = cosine_similarity(feat1, feat2)

            if similarity > similarity_threshold:
                # Merge tracklet2 into tracklet1
                tracklet1.frames.extend(tracklet2.frames)
                tracklet1.bboxes.extend(tracklet2.bboxes)
                tracklet1.confidences.extend(tracklet2.confidences)

                # Update feature (weighted average)
                feat1 = (feat1 + feat2) / 2
                tracklet_features[track_id1] = feat1
                track1_end = tracklet1.frames[-1]

                used.add(track_id2)
                merged = True
                merge_count += 1

        merged_tracklets[track_id1] = tracklet1

    print(f"Merged {merge_count} tracklets")
    print(f"Result: {len(merged_tracklets)} tracklets (from {len(tracklets)})")

    # Reassign IDs (1, 2, 3, ...)
    print("Reassigning track IDs...")
    final_tracklets = {}
    for new_id, (old_id, tracklet) in enumerate(sorted(merged_tracklets.items()), start=1):
        tracklet.id = new_id
        final_tracklets[new_id] = tracklet

    return final_tracklets


def run_reid_merging(tracking_csv: str, reid_pkl: str, output_csv: str,
                     similarity_threshold: float = 0.7, max_time_gap: int = 150):
    """
    Run Re-ID based tracklet merging.

    Args:
        tracking_csv: Input tracking CSV
        reid_pkl: Re-ID features pickle file
        output_csv: Output merged tracking CSV
        similarity_threshold: Minimum similarity to merge
        max_time_gap: Maximum time gap (frames)
    """
    # Load tracklets
    print(f"Loading tracklets from {tracking_csv}...")
    tracklets = csv_to_tracklets(tracking_csv)

    # Load Re-ID features
    print(f"Loading Re-ID features from {reid_pkl}...")
    with open(reid_pkl, 'rb') as f:
        reid_features = pickle.load(f)

    # Merge tracklets
    merged_tracklets = merge_tracklets_by_reid(
        tracklets, reid_features, similarity_threshold, max_time_gap
    )

    # Save results
    tracklets_to_csv(merged_tracklets, output_csv)
    print(f"Saved: {output_csv}")

    return merged_tracklets


def run_post_merge_interpolation(tracking_csv: str, output_csv: str, max_gap: int = 30):
    """
    Run interpolation on merged tracklets (Stage 4).

    Args:
        tracking_csv: Input tracking CSV (step3_reid_merged.csv)
        output_csv: Output interpolated CSV (step4_post_interpolated.csv)
        max_gap: Maximum gap to interpolate (frames)

    Returns:
        interpolated_tracklets: Dictionary of interpolated tracklets
    """
    print(f"\n=== Stage 4: Post-Merge Interpolation ===")
    print(f"Loading tracklets from {tracking_csv}...")

    # Load tracklets from step3
    tracklets = csv_to_tracklets(tracking_csv)

    print(f"Interpolating {len(tracklets)} tracklets (max_gap={max_gap})...")

    # Interpolate each tracklet
    interpolated_tracklets = {}
    total_added = 0

    for track_id, tracklet in tqdm(tracklets.items(), desc="Post-Merge Interpolation"):
        original_len = len(tracklet)
        interpolated = interpolate_tracklet(tracklet, max_gap)
        added = len(interpolated) - original_len
        total_added += added
        interpolated_tracklets[track_id] = interpolated

    print(f"Added {total_added} interpolated detections")

    # Calculate statistics
    total_detections = sum(len(t) for t in interpolated_tracklets.values())
    print(f"Result: {total_detections} total detections, {len(interpolated_tracklets)} unique IDs")

    # Save results
    tracklets_to_csv(interpolated_tracklets, output_csv)
    print(f"Saved: {output_csv}")

    return interpolated_tracklets


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline_for_sequence(
    sequence_name: str,
    detections_by_frame: Dict[int, np.ndarray],
    images_dir: str,
    output_dir: str,
    image_name_map: Dict[int, str],
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.1,
    max_gap: int = 30,
    similarity_threshold: float = 0.7,
    max_time_gap: int = 150,
    device: str = 'cuda',
    batch_size: int = 32,
    num_reid_workers: int = 8
) -> Dict[int, Tracklet]:
    """
    Run the full tracking pipeline for a single sequence.

    Args:
        sequence_name: Name of the sequence
        detections_by_frame: Detection dictionary {frame_num: detections_array}
        images_dir: Directory containing images
        output_dir: Output directory for this sequence
        image_name_map: Mapping from frame number to image filename
        max_age: SORT max age parameter
        min_hits: SORT min hits parameter
        iou_threshold: SORT IoU threshold
        max_gap: Maximum gap for interpolation
        similarity_threshold: Re-ID similarity threshold
        max_time_gap: Re-ID maximum time gap
        device: Device for Re-ID model
        batch_size: Batch size for Re-ID extraction
        num_reid_workers: Number of parallel workers for Re-ID extraction

    Returns:
        Final tracklets dictionary
    """
    print(f"\n{'='*70}")
    print(f"Processing sequence: {sequence_name}")
    print(f"{'='*70}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define output paths
    step1_csv = os.path.join(output_dir, "step1_sort_raw.csv")
    step2_csv = os.path.join(output_dir, "step2_interpolated.csv")
    step3_csv = os.path.join(output_dir, "step3_reid_merged.csv")
    step4_csv = os.path.join(output_dir, "step4_post_interpolated.csv")
    reid_pkl = os.path.join(output_dir, "reid_features.pkl")

    # Stage 1: SORT Tracking
    tracks_by_frame = run_sort_tracking(
        detections_by_frame,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold
    )

    save_tracking_to_csv(tracks_by_frame, step1_csv)

    # Stage 2: Interpolation
    interpolated_tracklets = run_interpolation(
        step1_csv, step2_csv, max_gap=max_gap
    )

    # Stage 3: Re-ID Feature Extraction (parallel)
    try:
        reid_features = extract_reid_features_parallel(
            step2_csv, images_dir, reid_pkl,
            device=device, batch_size=batch_size,
            num_workers=num_reid_workers,
            image_name_map=image_name_map
        )

        # Stage 3b: Re-ID Based Merging
        merged_tracklets = run_reid_merging(
            step2_csv, reid_pkl, step3_csv,
            similarity_threshold=similarity_threshold,
            max_time_gap=max_time_gap
        )
    except (ImportError, ModuleNotFoundError) as e:
        print("\n" + "="*70)
        print("WARNING: Re-ID module not available, skipping stage 3")
        print("="*70)
        print("\nCopying step2 results to step3 (no Re-ID merging)...")
        import shutil
        shutil.copy(step2_csv, step3_csv)
        merged_tracklets = csv_to_tracklets(step2_csv)

    # Stage 4: Post-Merge Interpolation
    final_tracklets = run_post_merge_interpolation(
        step3_csv, step4_csv, max_gap=max_gap
    )

    # Print summary for this sequence
    print(f"\n--- Sequence {sequence_name} Summary ---")
    print(f"Final: {sum(len(t) for t in final_tracklets.values())} tracks, "
          f"{len(final_tracklets)} unique IDs")

    return final_tracklets


def main():
    parser = argparse.ArgumentParser(
        description="4-Stage Tracking Pipeline: SORT → Interpolation → Re-ID Merging → Post-Interpolation"
    )

    # Input/Output
    parser.add_argument("--detections", type=str, required=True,
                       help="Path to detection CSV file (with image_name column)")
    parser.add_argument("--images-dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="tracking_results",
                       help="Output directory for results")

    # Stage 1: SORT parameters
    parser.add_argument("--max-age", type=int, default=30,
                       help="SORT max age (default: 30)")
    parser.add_argument("--min-hits", type=int, default=3,
                       help="SORT min hits (default: 3)")
    parser.add_argument("--iou-threshold", type=float, default=0.1,
                       help="SORT IoU threshold (default: 0.1)")
    parser.add_argument("--confidence-threshold", type=float, default=0.1,
                       help="Detection confidence threshold (default: 0.1)")

    # Stage 2: Interpolation parameters
    parser.add_argument("--max-gap", type=int, default=30,
                       help="Maximum gap to interpolate (default: 30)")

    # Stage 3: Re-ID parameters
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                       help="Re-ID similarity threshold (default: 0.7)")
    parser.add_argument("--max-time-gap", type=int, default=150,
                       help="Re-ID maximum time gap (default: 150)")
    parser.add_argument("--device", type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help="Device for Re-ID model (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for Re-ID extraction (default: 32)")
    parser.add_argument("--num-reid-workers", type=int, default=8,
                       help="Number of parallel workers for Re-ID extraction (default: 8)")

    args = parser.parse_args()

    print("="*70)
    print("4-STAGE TRACKING PIPELINE (Images Mode)")
    print("="*70)
    print(f"Input detections: {args.detections}")
    print(f"Images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Re-ID workers: {args.num_reid_workers}")
    print("="*70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load detections grouped by sequence
    detections_by_sequence, image_name_map = load_detections_by_sequence(
        args.detections,
        images_dir=args.images_dir,
        conf_threshold=args.confidence_threshold
    )

    # Process each sequence
    all_results = {}
    for sequence_name in sorted(detections_by_sequence.keys()):
        detections_by_frame = detections_by_sequence[sequence_name]
        seq_image_name_map = image_name_map[sequence_name]
        output_subdir = os.path.join(args.output_dir, sequence_name)

        final_tracklets = run_pipeline_for_sequence(
            sequence_name=sequence_name,
            detections_by_frame=detections_by_frame,
            images_dir=args.images_dir,
            output_dir=output_subdir,
            image_name_map=seq_image_name_map,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
            max_gap=args.max_gap,
            similarity_threshold=args.similarity_threshold,
            max_time_gap=args.max_time_gap,
            device=args.device,
            batch_size=args.batch_size,
            num_reid_workers=args.num_reid_workers
        )
        all_results[sequence_name] = final_tracklets

    # Final summary
    print("\n" + "="*70)
    print("ALL SEQUENCES COMPLETE!")
    print("="*70)
    total_tracks = 0
    total_ids = 0
    for seq_name, tracklets in all_results.items():
        seq_tracks = sum(len(t) for t in tracklets.values())
        seq_ids = len(tracklets)
        total_tracks += seq_tracks
        total_ids += seq_ids
        print(f"  {seq_name}: {seq_tracks} tracks, {seq_ids} IDs")
    print(f"\nTotal: {total_tracks} tracks, {total_ids} unique IDs across {len(all_results)} sequences")
    print("="*70)


if __name__ == "__main__":
    main()
