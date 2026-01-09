#!/usr/bin/env python3
"""
Visualize tracking results by drawing bounding boxes with track IDs on frames.
Optionally includes pose estimation skeleton visualization.
Reads tracking results from CSV and frames from disk, outputs visualized frames.

Usage:
    # 단일 시퀀스 (tracking만)
    python visualize_tracking.py tracking.csv --frames-dir images --output-dir vis

    # 전체 시퀀스 배치 처리 (tracking만)
    python visualize_tracking.py --batch --tracking-dir mma_tracking_results --frames-dir images

    # pose 시각화 포함
    python visualize_tracking.py --batch --tracking-dir tracking_results --frames-dir images --pose
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


# ===== Pose Visualization 관련 상수 및 함수 =====

# COCO Keypoint 정의 (17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO Skeleton 연결 정의 (keypoint index pairs)
SKELETON_CONNECTIONS = [
    # 얼굴
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose - eyes - ears
    # 상체
    (5, 6),   # 어깨 연결
    (5, 7), (7, 9),    # 왼팔: 어깨 - 팔꿈치 - 손목
    (6, 8), (8, 10),   # 오른팔: 어깨 - 팔꿈치 - 손목
    # 몸통
    (5, 11), (6, 12),  # 어깨 - 엉덩이
    (11, 12),          # 엉덩이 연결
    # 하체
    (11, 13), (13, 15),  # 왼다리: 엉덩이 - 무릎 - 발목
    (12, 14), (14, 16),  # 오른다리: 엉덩이 - 무릎 - 발목
]


def load_pose_csv(csv_path: str) -> dict:
    """
    pose estimation CSV 로드하여 frame별로 그룹화

    Returns:
        dict: {frame_num: [{track_id, keypoints}, ...]}
    """
    frame_data = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame = int(row['frame'])
            track_id = int(row['track_id'])

            # keypoints 정보
            keypoints = {}
            for kp_name in KEYPOINT_NAMES:
                x = float(row[f'{kp_name}_x'])
                y = float(row[f'{kp_name}_y'])
                conf = float(row[f'{kp_name}_conf'])
                keypoints[kp_name] = (x, y, conf)

            frame_data[frame].append({
                'track_id': track_id,
                'keypoints': keypoints
            })

    return frame_data


def draw_skeleton(image, keypoints, track_id, conf_threshold=0.3, line_thickness=2, point_radius=4):
    """
    이미지에 skeleton 그리기

    Args:
        image: 원본 이미지
        keypoints: {keypoint_name: (x, y, conf)}
        track_id: 플레이어 ID
        conf_threshold: 시각화할 최소 confidence
        line_thickness: 선 두께
        point_radius: 점 반지름
    """
    point_color = generate_color(track_id)
    # 선 색상은 좀 더 밝게
    line_color = tuple(min(255, int(c * 1.3)) for c in point_color)

    # keypoints를 index로 변환
    kp_coords = []
    for name in KEYPOINT_NAMES:
        if name in keypoints:
            x, y, conf = keypoints[name]
            if conf >= conf_threshold and x > 0 and y > 0:
                kp_coords.append((int(x), int(y), conf))
            else:
                kp_coords.append(None)
        else:
            kp_coords.append(None)

    # Skeleton 연결선 그리기
    for (idx1, idx2) in SKELETON_CONNECTIONS:
        if kp_coords[idx1] is not None and kp_coords[idx2] is not None:
            pt1 = (kp_coords[idx1][0], kp_coords[idx1][1])
            pt2 = (kp_coords[idx2][0], kp_coords[idx2][1])
            cv2.line(image, pt1, pt2, line_color, line_thickness)

    # Keypoint 점 그리기
    for kp in kp_coords:
        if kp is not None:
            x, y, conf = kp
            radius = int(point_radius * (0.5 + conf * 0.5))
            cv2.circle(image, (x, y), radius, point_color, -1)
            cv2.circle(image, (x, y), radius, (255, 255, 255), 1)  # 테두리


def visualize_tracking(tracking_csv, frames_dir, output_dir, pose_csv=None, pose_conf_threshold=0.3):
    """
    Visualize tracking results on frames.

    Args:
        tracking_csv: Path to tracking results CSV file
        frames_dir: Directory containing frame images
        output_dir: Directory to save visualized frames
        pose_csv: Path to pose estimation CSV file (optional)
        pose_conf_threshold: Keypoint confidence threshold for pose visualization
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

    # Load pose data if provided
    pose_by_frame = {}
    if pose_csv and os.path.exists(pose_csv):
        print(f"Loading pose data from {pose_csv}...")
        pose_by_frame = load_pose_csv(pose_csv)
        print(f"Loaded pose data for {len(pose_by_frame)} frames")

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

        # Draw pose skeleton if available
        if frame_num in pose_by_frame:
            for pose_data in pose_by_frame[frame_num]:
                draw_skeleton(
                    frame,
                    pose_data['keypoints'],
                    pose_data['track_id'],
                    conf_threshold=pose_conf_threshold
                )

        # Save visualized frame
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, frame)

    print(f"Saved {len(frame_numbers)} frames to {output_dir}")


def visualize_all_sequences(tracking_dir, frames_dir, csv_filename="step2_interpolated.csv",
                            with_pose=False, pose_csv_filename="pose_estimation.csv",
                            pose_conf_threshold=0.3, output_subdir="visualization"):
    """
    모든 시퀀스에 대해 시각화 수행

    Args:
        tracking_dir: 추적 결과 디렉토리 (예: mma_tracking_results)
        frames_dir: 이미지 디렉토리
        csv_filename: 사용할 CSV 파일명 (default: step2_interpolated.csv)
        with_pose: pose 시각화 포함 여부
        pose_csv_filename: pose CSV 파일명 (default: pose_estimation.csv)
        pose_conf_threshold: keypoint confidence threshold
        output_subdir: 출력 폴더명 (default: visualization)
    """
    print(f"="*70)
    print(f"Batch Visualization")
    print(f"="*70)
    print(f"Tracking dir: {tracking_dir}")
    print(f"Frames dir: {frames_dir}")
    print(f"CSV file: {csv_filename}")
    if with_pose:
        print(f"Pose CSV: {pose_csv_filename}")
        print(f"Pose conf threshold: {pose_conf_threshold}")
    print(f"Output subdir: {output_subdir}")
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
        output_dir = os.path.join(tracking_dir, seq_name, output_subdir)

        # pose CSV 경로
        pose_csv = None
        if with_pose:
            pose_csv = os.path.join(tracking_dir, seq_name, pose_csv_filename)
            if not os.path.exists(pose_csv):
                print(f"Warning: Pose CSV not found: {pose_csv}")
                pose_csv = None

        visualize_tracking(csv_path, frames_dir, output_dir,
                          pose_csv=pose_csv, pose_conf_threshold=pose_conf_threshold)

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
  python visualize_tracking.py --batch --tracking-dir tracking_results --frames-dir images

  # pose 시각화 포함 (배치)
  python visualize_tracking.py --batch --tracking-dir tracking_results --frames-dir images --pose

  # pose 시각화 포함 (단일)
  python visualize_tracking.py tracking.csv --frames-dir images --pose --pose-csv pose_estimation.csv
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
        default="tracking_results",
        help="추적 결과 디렉토리 (배치 모드, default: tracking_results)"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="step2_interpolated.csv",
        help="사용할 CSV 파일명 (배치 모드, default: step2_interpolated.csv)"
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="visualization",
        help="출력 폴더명 (배치 모드, default: visualization)"
    )

    # Pose 옵션
    parser.add_argument(
        "--pose",
        action="store_true",
        help="pose estimation 시각화 포함"
    )
    parser.add_argument(
        "--pose-csv",
        type=str,
        default=None,
        help="pose estimation CSV 경로 (단일 모드)"
    )
    parser.add_argument(
        "--pose-csv-file",
        type=str,
        default="pose_estimation_finetuned.csv",
        help="pose CSV 파일명 (배치 모드, default: pose_estimation_finetuned.csv)"
    )
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=0.3,
        help="keypoint confidence threshold (default: 0.3)"
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

        # pose 옵션에 따라 출력 폴더명 변경
        output_subdir = args.output_subdir
        if args.pose and output_subdir == "visualization":
            output_subdir = "pose_visualization"

        visualize_all_sequences(
            args.tracking_dir, args.frames_dir, args.csv_file,
            with_pose=args.pose,
            pose_csv_filename=args.pose_csv_file,
            pose_conf_threshold=args.pose_conf,
            output_subdir=output_subdir
        )

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

        output_dir = args.output_dir or ("pose_visualization" if args.pose else "visualization")

        # pose CSV 경로 결정
        pose_csv = None
        if args.pose:
            if args.pose_csv:
                pose_csv = args.pose_csv
            else:
                # tracking_csv와 같은 폴더에서 찾기
                pose_csv = os.path.join(os.path.dirname(args.tracking_csv), "pose_estimation.csv")
            if not os.path.exists(pose_csv):
                print(f"Warning: Pose CSV not found: {pose_csv}")
                pose_csv = None

        visualize_tracking(args.tracking_csv, args.frames_dir, output_dir,
                          pose_csv=pose_csv, pose_conf_threshold=args.pose_conf)


if __name__ == "__main__":
    main()
