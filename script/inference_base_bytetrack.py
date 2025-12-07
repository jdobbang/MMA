#!/usr/bin/env python3
"""
YOLO + ByteTrack Tracking Script for MMA Player Detection
- Tracks 2 players per camera sequence
- Groups images by camera name and processes separately
- Uses ByteTrack algorithm for robust tracking with low-confidence detections
"""

import argparse
import os
import re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import glob


class KalmanBoxTracker:
    """
    Kalman Filter for bounding box tracking
    State: [x_center, y_center, area, aspect_ratio, vx, vy, va]
    """

    def __init__(self, bbox):
        # State: [x, y, a, r, vx, vy, va] - center x, center y, area, aspect ratio, velocities
        self.kf = self._create_kalman_filter()

        # Initialize state from bbox [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect_ratio = w / max(h, 1e-6)

        self.kf.statePost = np.array([[cx], [cy], [area], [aspect_ratio], [0], [0], [0]], dtype=np.float32)
        self.time_since_update = 0
        self.hits = 1

    def _create_kalman_filter(self):
        kf = cv2.KalmanFilter(7, 4)  # 7 state variables, 4 measurement variables

        # Transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance
        kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.01
        kf.processNoiseCov[4:, 4:] *= 0.01  # Lower noise for velocities

        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0

        # Error covariance
        kf.errorCovPost = np.eye(7, dtype=np.float32) * 10.0

        return kf

    def predict(self):
        """Predict next state and return predicted bbox"""
        self.kf.predict()
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox):
        """Update state with new measurement"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect_ratio = w / max(h, 1e-6)

        measurement = np.array([[cx], [cy], [area], [aspect_ratio]], dtype=np.float32)
        self.kf.correct(measurement)
        self.time_since_update = 0
        self.hits += 1

    def get_state(self):
        """Get current state as bbox [x1, y1, x2, y2]"""
        state = self.kf.statePost
        cx, cy, area, aspect_ratio = state[0, 0], state[1, 0], state[2, 0], state[3, 0]

        area = max(area, 1.0)
        aspect_ratio = max(aspect_ratio, 0.1)

        h = np.sqrt(area / aspect_ratio)
        w = aspect_ratio * h

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return np.array([x1, y1, x2, y2])


class ByteTracker:
    """
    ByteTrack implementation with Kalman Filter for MMA tracking
    Based on ByteTrack paper: https://arxiv.org/abs/2110.06864
    """

    def __init__(self, track_thresh=0.7, track_buffer=30, match_thresh=0.8, max_players=2):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.max_players = max_players

        self.tracks = []
        self.lost_tracks = []
        self.frame_id = 0
        # ID pool: fixed IDs [1, 2] for MMA
        self.available_ids = list(range(1, max_players + 1))

    def reset(self):
        self.tracks = []
        self.lost_tracks = []
        self.frame_id = 0
        self.available_ids = list(range(1, self.max_players + 1))

    def update(self, detections, scores):
        self.frame_id += 1

        # Predict new locations for all tracks using Kalman Filter
        for track in self.tracks:
            track['kf'].predict()
            track['bbox'] = track['kf'].get_state()

        for track in self.lost_tracks:
            track['kf'].predict()
            track['bbox'] = track['kf'].get_state()

        if len(detections) == 0:
            for track in self.tracks:
                track['lost_frames'] += 1
                if track['lost_frames'] <= self.track_buffer:
                    self.lost_tracks.append(track)
            self.tracks = []
            return []

        # Separate high and low confidence detections (ByteTrack core idea)
        high_mask = scores >= self.track_thresh
        low_mask = (scores < self.track_thresh) & (scores >= 0.1)

        high_dets = detections[high_mask]
        high_scores = scores[high_mask]
        low_dets = detections[low_mask]
        low_scores = scores[low_mask]

        # First association: high confidence detections with existing tracks (using predicted bbox)
        matched_tracks = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets_high = list(range(len(high_dets)))

        if len(self.tracks) > 0 and len(high_dets) > 0:
            # Use predicted bboxes for matching
            predicted_bboxes = np.array([t['bbox'] for t in self.tracks])
            iou_matrix = self._compute_iou_matrix(predicted_bboxes, high_dets)
            matched_indices, unmatched_tracks, unmatched_dets_high = \
                self._linear_assignment(iou_matrix, self.match_thresh)

            for track_idx, det_idx in matched_indices:
                # Update Kalman Filter with detection
                self.tracks[track_idx]['kf'].update(high_dets[det_idx])
                self.tracks[track_idx]['bbox'] = high_dets[det_idx]
                self.tracks[track_idx]['score'] = high_scores[det_idx]
                self.tracks[track_idx]['lost_frames'] = 0
                matched_tracks.append(track_idx)

        # Second association: low confidence detections with remaining tracks
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]

        if len(remaining_tracks) > 0 and len(low_dets) > 0:
            predicted_bboxes = np.array([t['bbox'] for t in remaining_tracks])
            iou_matrix = self._compute_iou_matrix(predicted_bboxes, low_dets)
            matched_low, unmatched_idx, _ = self._linear_assignment(iou_matrix, self.match_thresh)

            for track_idx, det_idx in matched_low:
                original_idx = unmatched_tracks[track_idx]
                # Update Kalman Filter with detection
                self.tracks[original_idx]['kf'].update(low_dets[det_idx])
                self.tracks[original_idx]['bbox'] = low_dets[det_idx]
                self.tracks[original_idx]['score'] = low_scores[det_idx]
                self.tracks[original_idx]['lost_frames'] = 0
                matched_tracks.append(original_idx)

            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_idx]

        # Handle unmatched tracks (mark as lost or remove)
        active_tracks = []
        for i, track in enumerate(self.tracks):
            if i in matched_tracks:
                active_tracks.append(track)
            else:
                track['lost_frames'] += 1
                if track['lost_frames'] <= self.track_buffer:
                    self.lost_tracks.append(track)
                else:
                    # Return ID to pool when track is permanently lost
                    self.available_ids.append(track['id'])
                    self.available_ids.sort()

        # Recover lost tracks with unmatched high detections (no IoU matching)
        # Simply assign unmatched detections to lost tracks in order
        recovered = []
        remaining_unmatched = list(unmatched_dets_high)

        if len(self.lost_tracks) > 0 and len(remaining_unmatched) > 0:
            # Sort lost tracks by lost_frames (recover most recently lost first)
            self.lost_tracks.sort(key=lambda t: t['lost_frames'])

            num_to_recover = min(len(self.lost_tracks), len(remaining_unmatched))
            for i in range(num_to_recover):
                track = self.lost_tracks[i]
                det_idx = remaining_unmatched[i]
                # Update Kalman Filter with detection
                track['kf'].update(high_dets[det_idx])
                track['bbox'] = high_dets[det_idx]
                track['score'] = high_scores[det_idx]
                track['lost_frames'] = 0
                recovered.append(track)

            # Remove recovered tracks from lost_tracks and used detections from remaining
            self.lost_tracks = self.lost_tracks[num_to_recover:]
            remaining_unmatched = remaining_unmatched[num_to_recover:]

        # Clean up lost_tracks that exceeded buffer and return their IDs
        valid_lost_tracks = []
        for track in self.lost_tracks:
            if track['lost_frames'] <= self.track_buffer:
                valid_lost_tracks.append(track)
            else:
                # Return ID to pool
                if track['id'] not in self.available_ids:
                    self.available_ids.append(track['id'])
                    self.available_ids.sort()
        self.lost_tracks = valid_lost_tracks

        # Initialize new tracks (only up to max_players, using ID pool)
        new_tracks = []
        current_active_count = len(active_tracks) + len(recovered)

        for det_idx in remaining_unmatched:
            if current_active_count + len(new_tracks) >= self.max_players:
                break

            # If no available IDs but we have lost tracks, reclaim oldest lost track's ID
            if not self.available_ids and len(self.lost_tracks) > 0:
                # Find the oldest lost track (most frames lost)
                oldest_idx = max(range(len(self.lost_tracks)),
                               key=lambda i: self.lost_tracks[i]['lost_frames'])
                oldest_track = self.lost_tracks.pop(oldest_idx)
                reclaimed_id = oldest_track['id']
                self.available_ids.append(reclaimed_id)
                self.available_ids.sort()

            if not self.available_ids:
                break  # No available IDs

            new_id = self.available_ids.pop(0)  # Get smallest available ID
            new_track = {
                'id': new_id,
                'bbox': high_dets[det_idx],
                'score': high_scores[det_idx],
                'lost_frames': 0,
                'kf': KalmanBoxTracker(high_dets[det_idx])
            }
            new_tracks.append(new_track)

        self.tracks = active_tracks + recovered + new_tracks

        return [(t['id'], t['bbox'], t['score']) for t in self.tracks]

    def _compute_iou_matrix(self, boxes1, boxes2):
        n1, n2 = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._compute_ciou(boxes1[i], boxes2[j])
        return iou_matrix

    def _compute_ciou(self, box1, box2):
        """
        Complete IoU (CIoU) = IoU - distance_penalty - aspect_ratio_penalty
        Better than IoU for tracking as it considers center distance and shape
        """
        # Box coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2

        # Intersection
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Union
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = area1 + area2 - inter_area

        # IoU
        iou = inter_area / union_area if union_area > 0 else 0

        # Enclosing box (smallest box containing both boxes)
        enclose_x1 = min(b1_x1, b2_x1)
        enclose_y1 = min(b1_y1, b2_y1)
        enclose_x2 = max(b1_x2, b2_x2)
        enclose_y2 = max(b1_y2, b2_y2)

        # Diagonal length of enclosing box (c^2)
        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        if c2 == 0:
            return iou

        # Center points
        b1_cx = (b1_x1 + b1_x2) / 2
        b1_cy = (b1_y1 + b1_y2) / 2
        b2_cx = (b2_x1 + b2_x2) / 2
        b2_cy = (b2_y1 + b2_y2) / 2

        # Distance between centers (rho^2)
        rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

        # Aspect ratio consistency (v)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

        v = (4 / (np.pi ** 2)) * (np.arctan(w1 / max(h1, 1e-6)) - np.arctan(w2 / max(h2, 1e-6))) ** 2

        # Alpha (trade-off parameter)
        alpha = v / (1 - iou + v + 1e-6) if iou < 1 else 0

        # CIoU = IoU - (rho^2 / c^2) - alpha * v
        ciou = iou - (rho2 / c2) - alpha * v

        return ciou

    def _linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        matched = []
        unmatched_rows = set(range(cost_matrix.shape[0]))
        unmatched_cols = set(range(cost_matrix.shape[1]))

        indices = np.unravel_index(np.argsort(-cost_matrix.ravel()), cost_matrix.shape)
        for row, col in zip(indices[0], indices[1]):
            if row in unmatched_rows and col in unmatched_cols and cost_matrix[row, col] >= thresh:
                matched.append((row, col))
                unmatched_rows.discard(row)
                unmatched_cols.discard(col)

        return matched, list(unmatched_rows), list(unmatched_cols)


def extract_sequence_name(filename):
    """Extract sequence name (everything before _camXX_)
    Example: 03_grappling2_001_grappling2_cam01_00001.jpg -> 03_grappling2_001_grappling2
    """
    match = re.search(r'^(.+?)_cam\d+_', filename)
    return match.group(1) if match else 'unknown'


def extract_frame_number(filename):
    """Extract frame number (last number before extension)
    Example: 03_grappling2_001_grappling2_cam01_00001.jpg -> 1
    """
    match = re.search(r'_(\d+)\.[^.]+$', filename)
    return int(match.group(1)) if match else 0


def inference_with_tracking(model_path, source, output_dir, conf_threshold=0.1,
                            track_thresh=0.5, track_buffer=30, match_thresh=0.8,
                            save_txt=False, max_players=2):
    """
    Run YOLO inference with ByteTrack tracking
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    if save_txt:
        label_dir = output_dir / "labels"
        label_dir.mkdir(exist_ok=True)

    # Get image files
    source_path = Path(source)
    if not source_path.is_dir():
        raise ValueError(f"Source must be a directory: {source}")

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(str(source_path / ext)))
        image_files.extend(glob.glob(str(source_path / ext.upper())))

    if len(image_files) == 0:
        print(f"No images found in {source}")
        return

    # Group images by sequence name
    sequence_groups = defaultdict(list)
    for img_path in image_files:
        seq_name = extract_sequence_name(os.path.basename(img_path))
        sequence_groups[seq_name].append(img_path)

    # Sort each sequence group by frame number
    for seq_name in sequence_groups:
        sequence_groups[seq_name] = sorted(
            sequence_groups[seq_name],
            key=lambda x: extract_frame_number(os.path.basename(x))
        )

    print(f"Found {len(image_files)} images in {len(sequence_groups)} sequence groups")
    for seq_name, files in sorted(sequence_groups.items()):
        print(f"  - {seq_name}: {len(files)} frames")
    print(f"\nOutput directory: {output_dir}")
    print(f"Detection conf threshold: {conf_threshold}")
    print(f"Track threshold (high conf): {track_thresh}")
    print(f"Track buffer: {track_buffer} frames")
    print(f"Match threshold (IoU): {match_thresh}")
    print(f"Max players: {max_players}")
    print()

    # Track colors per ID
    track_colors = {
        1: (0, 255, 0),    # Green
        2: (255, 0, 0),    # Blue
        3: (0, 0, 255),    # Red
        4: (255, 255, 0),  # Cyan
    }

    def get_track_color(track_id):
        return track_colors.get(track_id, (128, 128, 128))

    stats = {
        'total_images': 0,
        'total_detections': 0,
        'total_tracked': 0,
        'sequence_stats': {}
    }

    # Process each sequence group separately
    for seq_name in sorted(sequence_groups.keys()):
        image_list = sequence_groups[seq_name]
        print(f"\n{'='*60}")
        print(f"Processing sequence: {seq_name} ({len(image_list)} frames)")
        print('='*60)

        tracker = ByteTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            max_players=max_players
        )

        seq_stats = {'frames': 0, 'detections': 0, 'tracked': 0, 'track_ids': set()}

        for img_path in tqdm(image_list, desc=f"[{seq_name}]"):
            img_path = Path(img_path)
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue

            results = model(image, conf=conf_threshold, verbose=False)

            # Extract detections for class 0 (player) only
            detections = []
            scores = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2])
                    scores.append(conf)

            detections = np.array(detections) if detections else np.array([]).reshape(0, 4)
            scores = np.array(scores)
            seq_stats['detections'] += len(detections)

            # Update tracker
            tracked_objects = tracker.update(detections, scores)
            seq_stats['tracked'] += len(tracked_objects)

            # Draw results
            vis_image = image.copy()
            tracking_results = []

            for track_id, bbox, score in tracked_objects:
                seq_stats['track_ids'].add(track_id)
                x1, y1, x2, y2 = bbox.astype(int)
                color = get_track_color(track_id)

                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                label = f"Player {track_id} ({score:.2f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if save_txt:
                    img_h, img_w = image.shape[:2]
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    tracking_results.append(
                        f"{track_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}"
                    )

            # Frame info overlay
            info_text = f"{seq_name} | Frame: {extract_frame_number(img_path.name)} | Tracked: {len(tracked_objects)}"
            cv2.putText(vis_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imwrite(str(vis_dir / img_path.name), vis_image)

            if save_txt and tracking_results:
                label_path = label_dir / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(tracking_results))

            seq_stats['frames'] += 1

        stats['sequence_stats'][seq_name] = seq_stats
        stats['total_images'] += seq_stats['frames']
        stats['total_detections'] += seq_stats['detections']
        stats['total_tracked'] += seq_stats['tracked']

        print(f"  Unique track IDs: {sorted(seq_stats['track_ids'])}")

    # Print summary
    print("\n" + "="*60)
    print("Tracking Summary")
    print("="*60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Total raw detections: {stats['total_detections']}")
    print(f"Total tracked objects: {stats['total_tracked']}")
    print(f"Average tracked per frame: {stats['total_tracked'] / max(stats['total_images'], 1):.2f}")
    print()
    print("Per-sequence statistics:")
    for seq_name, seq_stats in stats['sequence_stats'].items():
        print(f"  [{seq_name}]")
        print(f"    - Frames: {seq_stats['frames']}")
        print(f"    - Detections: {seq_stats['detections']}")
        print(f"    - Tracked: {seq_stats['tracked']}")
        print(f"    - Unique IDs: {sorted(seq_stats['track_ids'])}")
        print(f"    - Avg tracked/frame: {seq_stats['tracked'] / max(seq_stats['frames'], 1):.2f}")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"  - Visualizations: {vis_dir}")
    if save_txt:
        print(f"  - Labels: {label_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='YOLO + ByteTrack Tracking for MMA')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to best.pt model')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--output', type=str, default='/workspace/inference_results_tracking',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='Detection confidence threshold (default: 0.1)')
    parser.add_argument('--track-thresh', type=float, default=0.5,
                       help='ByteTrack high confidence threshold (default: 0.5)')
    parser.add_argument('--track-buffer', type=int, default=30,
                       help='Frames to keep lost tracks (default: 30)')
    parser.add_argument('--match-thresh', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.8)')
    parser.add_argument('--max-players', type=int, default=2,
                       help='Maximum players to track (default: 2 for MMA)')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save tracking results as txt files')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    inference_with_tracking(
        model_path=args.model,
        source=args.source,
        output_dir=args.output,
        conf_threshold=args.conf,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        save_txt=args.save_txt,
        max_players=args.max_players
    )


if __name__ == "__main__":
    main()

"""
Example usage:
python script/inference_base_bytetrack.py \
    --model log/mma_training_v1_202512052/weights/best.pt \
    --source dataset/yolodataset/images/val \
    --output log/mma_training_v1_202512052/inference_tracking \
    --conf 0.1 \
    --track-thresh 0.5 \
    --track-buffer 30 \
    --match-thresh 0.8 \
    --max-players 2 \
    --save-txt
"""