#!/usr/bin/env python3
"""
Visualize tracking results by drawing bounding boxes with track IDs on frames.
Reads tracking results from CSV and frames from disk, outputs visualized frames.

Usage:
    # 단일 시퀀스
    python visualize_tracking.py tracking.csv --frames-dir images --output-dir vis

    # 전체 시퀀스 배치 처리
    python visualize_tracking.py --batch --tracking-dir mma_tracking_results --frames-dir images
"""


import cv2
import csv
import os
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def generate_color(track_id):
    """Generate a consistent color for each track ID."""
    # MMA: ID 1은 빨강, ID 2는 파랑 (고정)
    if track_id == 1:
        return (0, 0, 255)  # Red (BGR)
    elif track_id == 2:
        return (255, 0, 0)  # Blue (BGR)
    else:
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())


def visualize_tracking(tracking_csv, frames_dir, output_dir):
    """
    Visualize tracking results on frames.

    Args:
        tracking_csv: Path to tracking results CSV file
        frames_dir: Directory containing frame images
        output_dir: Directory to save visualized frames
    """
    # Read tracking results grouped by frame/image_name
    print(f"Reading tracking results from {tracking_csv}...")
    tracks_by_frame = defaultdict(list)
    image_name_map = {}  # frame_num -> image_name

    with open(tracking_csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        has_image_name = 'image_name' in fieldnames

        for row in reader:
            frame_num = int(row['frame'])
            track_id = int(row['track_id'])
            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])
            confidence = float(row['confidence'])

            if has_image_name:
                image_name_map[frame_num] = row['image_name']

            tracks_by_frame[frame_num].append({
                'track_id': track_id,
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence
            })

    if not tracks_by_frame:
        print("No tracking data found!")
        return

    # Determine min/max frame number
    frame_numbers = sorted(tracks_by_frame.keys())
    min_frame = frame_numbers[0]
    max_frame = frame_numbers[-1]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get unique track IDs and assign colors
    all_track_ids = set()
    for tracks in tracks_by_frame.values():
        for track in tracks:
            all_track_ids.add(track['track_id'])

    track_colors = {track_id: generate_color(track_id) for track_id in all_track_ids}

    print(f"Found {len(tracks_by_frame)} frames with tracking data (frame {min_frame}-{max_frame})")
    print(f"Found {len(all_track_ids)} unique track IDs: {sorted(all_track_ids)}")

    # Process each frame
    print("Visualizing frames...")

    for frame_num in tqdm(frame_numbers):
        # Find corresponding frame file
        if frame_num in image_name_map:
            # Use image_name from CSV
            frame_path = os.path.join(frames_dir, image_name_map[frame_num])
            output_filename = image_name_map[frame_num]
        else:
            # Fallback to frame_{:06d}.jpg
            frame_path = os.path.join(frames_dir, f"frame_{frame_num:06d}.jpg")
            output_filename = f"frame_{frame_num:06d}.jpg"

        if not os.path.exists(frame_path):
            # Try without leading zeros
            alt_path = os.path.join(frames_dir, f"frame_{frame_num}.jpg")
            if os.path.exists(alt_path):
                frame_path = alt_path
            else:
                continue

        # Read frame
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Draw bounding boxes and track IDs
        for track in tracks_by_frame[frame_num]:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            confidence = track['confidence']
            color = track_colors[track_id]

            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box (thicker for visibility)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Draw track ID and confidence
            label = f"Player {track_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Draw background for text
            cv2.rectangle(frame,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 5, y1),
                         color, -1)

            # Draw text
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Save visualized frame
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, frame)

    print(f"Saved {len(frame_numbers)} frames to {output_dir}")


def visualize_all_sequences(tracking_dir, frames_dir, csv_filename="step2_interpolated.csv"):
    """
    모든 시퀀스에 대해 시각화 수행

    Args:
        tracking_dir: 추적 결과 디렉토리 (예: mma_tracking_results)
        frames_dir: 이미지 디렉토리
        csv_filename: 사용할 CSV 파일명 (default: step2_interpolated.csv)
    """
    print(f"="*70)
    print(f"Batch Visualization")
    print(f"="*70)
    print(f"Tracking dir: {tracking_dir}")
    print(f"Frames dir: {frames_dir}")
    print(f"CSV file: {csv_filename}")
    print(f"="*70)

    # 시퀀스 폴더 찾기
    sequences = []
    for item in os.listdir(tracking_dir):
        seq_path = os.path.join(tracking_dir, item)
        csv_path = os.path.join(seq_path, csv_filename)
        if os.path.isdir(seq_path) and os.path.exists(csv_path):
            sequences.append(item)

    if not sequences:
        print(f"No sequences found with {csv_filename}")
        return

    print(f"Found {len(sequences)} sequences: {sequences}")

    # 각 시퀀스 처리
    for seq_name in sorted(sequences):
        print(f"\n--- Processing: {seq_name} ---")

        csv_path = os.path.join(tracking_dir, seq_name, csv_filename)
        output_dir = os.path.join(tracking_dir, seq_name, "visualization")

        visualize_tracking(csv_path, frames_dir, output_dir)

    print(f"\n{'='*70}")
    print(f"All sequences visualized!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tracking results on frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 시퀀스 시각화
  python visualize_tracking.py tracking.csv --frames-dir images --output-dir vis

  # 전체 시퀀스 배치 처리
  python visualize_tracking.py --batch --tracking-dir mma_tracking_results --frames-dir images

  # step1 결과로 시각화
  python visualize_tracking.py --batch --tracking-dir mma_tracking_results --frames-dir images --csv-file step1_tracking.csv
        """
    )

    # 단일 모드
    parser.add_argument(
        "tracking_csv",
        type=str,
        nargs='?',
        default=None,
        help="Path to tracking results CSV file (단일 모드)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualized frames (단일 모드)"
    )

    # 배치 모드
    parser.add_argument(
        "--batch",
        action="store_true",
        help="배치 모드: 모든 시퀀스 폴더에 대해 시각화"
    )
    parser.add_argument(
        "--tracking-dir",
        type=str,
        default="mma_tracking_results",
        help="추적 결과 디렉토리 (배치 모드, default: mma_tracking_results)"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="step2_interpolated.csv",
        help="사용할 CSV 파일명 (배치 모드, default: step2_interpolated.csv)"
    )

    # 공통
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="images",
        help="Directory containing frame images (default: images)"
    )

    args = parser.parse_args()

    # 배치 모드
    if args.batch:
        if not os.path.exists(args.tracking_dir):
            print(f"Error: Tracking directory not found: {args.tracking_dir}")
            return
        if not os.path.exists(args.frames_dir):
            print(f"Error: Frames directory not found: {args.frames_dir}")
            return

        visualize_all_sequences(args.tracking_dir, args.frames_dir, args.csv_file)

    # 단일 모드
    else:
        if args.tracking_csv is None:
            parser.print_help()
            print("\nError: tracking_csv required for single mode. Use --batch for batch mode.")
            return

        if not os.path.exists(args.tracking_csv):
            print(f"Error: Tracking CSV file not found: {args.tracking_csv}")
            return

        if not os.path.exists(args.frames_dir):
            print(f"Error: Frames directory not found: {args.frames_dir}")
            return

        output_dir = args.output_dir or "visualization"
        visualize_tracking(args.tracking_csv, args.frames_dir, output_dir)


if __name__ == "__main__":
    main()
