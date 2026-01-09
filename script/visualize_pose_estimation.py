#!/usr/bin/env python3
"""
Pose Estimation Results Visualizer
===================================

pose_estimation.csv의 결과를 시각화하여:
- 각 frame에서 track_id별로 다른 색상으로 bbox 표시
- 각 track_id의 keypoint를 연결선(skeleton)으로 표시
- 출력: 비디오 또는 이미지 시퀀스

Usage:
  # 단일 시퀀스 비디오 생성
  python visualize_pose_estimation.py \
    --pose_csv tracking_results/seq1/pose_estimation.csv \
    --image_folder dataset/images/val \
    --output output.mp4

  # 배치 모드 (모든 시퀀스)
  python visualize_pose_estimation.py \
    --batch \
    --tracking_dir tracking_results \
    --image_folder dataset/images/val
"""

import argparse
import os
import csv
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path


# COCO Keypoint 정의 (17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO 스켈레톤 (연결 관계)
SKELETON = [
    (0, 1), (0, 2),  # nose -> eyes
    (1, 3), (2, 4),  # eyes -> ears
    (5, 6),          # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12),# shoulders -> hips
    (11, 12),        # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Track ID별 색상 (BGR)
TRACK_COLORS = {
    1: (0, 0, 255),      # Red for track_id 1
    2: (255, 0, 0),      # Blue for track_id 2
    3: (0, 255, 0),      # Green for track_id 3
    4: (255, 255, 0),    # Cyan for track_id 4
    5: (255, 0, 255),    # Magenta for track_id 5
}

# Default colors for other track IDs
def get_track_color(track_id):
    """Get color for a given track_id"""
    if track_id in TRACK_COLORS:
        return TRACK_COLORS[track_id]
    # Generate color from track_id for unknown IDs
    np.random.seed(track_id)
    color = tuple(np.random.randint(0, 256, 3).tolist())
    return color


def load_pose_csv(csv_path: str) -> dict:
    """
    pose estimation CSV를 로드하여 frame별로 그룹화

    Returns:
        {frame_num: {track_id: {keypoint_name: (x, y, conf), 'bbox': (x1, y1, x2, y2, conf), 'image_name': str}}}
    """
    frame_data = defaultdict(lambda: defaultdict(dict))

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            track_id = int(row['track_id'])
            image_name = row['image_name']

            # bbox 저장
            frame_data[frame][track_id]['image_name'] = image_name
            frame_data[frame][track_id]['bbox'] = (
                float(row['bbox_x1']),
                float(row['bbox_y1']),
                float(row['bbox_x2']),
                float(row['bbox_y2']),
                float(row['bbox_conf'])
            )

            # keypoint 저장
            keypoints = {}
            for kp_name in KEYPOINT_NAMES:
                x = float(row[f'{kp_name}_x'])
                y = float(row[f'{kp_name}_y'])
                conf = float(row[f'{kp_name}_conf'])
                keypoints[kp_name] = (x, y, conf)

            frame_data[frame][track_id]['keypoints'] = keypoints

    return frame_data


def draw_bbox(image: np.ndarray, bbox: tuple, track_id: int, color: tuple = None) -> np.ndarray:
    """
    이미지에 bbox 그리기

    Args:
        image: 입력 이미지
        bbox: (x1, y1, x2, y2, conf)
        track_id: track ID
        color: BGR 색상 (None이면 자동)

    Returns:
        수정된 이미지
    """
    if color is None:
        color = get_track_color(track_id)

    x1, y1, x2, y2, conf = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # bbox 그리기
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # track_id와 confidence 텍스트
    label = f"ID:{track_id} Conf:{conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


def draw_keypoints(image: np.ndarray, keypoints: dict, track_id: int,
                   conf_threshold: float = 0.3, color: tuple = None) -> np.ndarray:
    """
    이미지에 keypoint와 skeleton 그리기

    Args:
        image: 입력 이미지
        keypoints: {keypoint_name: (x, y, conf)}
        track_id: track ID
        conf_threshold: 표시할 최소 confidence 값
        color: BGR 색상 (None이면 자동)

    Returns:
        수정된 이미지
    """
    if color is None:
        color = get_track_color(track_id)

    # skeleton 그리기 (먼저)
    for start_idx, end_idx in SKELETON:
        start_kp = KEYPOINT_NAMES[start_idx]
        end_kp = KEYPOINT_NAMES[end_idx]

        if start_kp not in keypoints or end_kp not in keypoints:
            continue

        start_x, start_y, start_conf = keypoints[start_kp]
        end_x, end_y, end_conf = keypoints[end_kp]

        # 둘 다 confidence 충분하면 선 그리기
        if start_conf >= conf_threshold and end_conf >= conf_threshold:
            cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)),
                    color, 2)

    # 개별 keypoint 그리기
    for kp_name, (x, y, conf) in keypoints.items():
        if conf >= conf_threshold:
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
            cv2.circle(image, (int(x), int(y)), 5, (255, 255, 255), 1)

    return image


def process_sequence(
    pose_csv: str,
    image_folder: str,
    output_dir: str = None,
    output_video: str = None,
    conf_threshold: float = 0.3,
    fps: int = 30
):
    """
    시퀀스 전체에 대해 pose visualization 수행

    Args:
        pose_csv: pose_estimation.csv 경로
        image_folder: 이미지 폴더 경로
        output_dir: 출력 이미지 폴더 (None이면 생성 안 함)
        output_video: 출력 비디오 경로 (None이면 생성 안 함)
        conf_threshold: keypoint confidence threshold
        fps: 출력 비디오 fps
    """
    # CSV 로드
    frame_data = load_pose_csv(pose_csv)

    if not frame_data:
        print(f"No pose data found in {pose_csv}")
        return

    print(f"Loaded {len(frame_data)} frames from pose CSV")

    # 출력 디렉토리 생성
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 비디오 writer 준비
    video_writer = None
    if output_video:
        os.makedirs(os.path.dirname(output_video) if os.path.dirname(output_video) else '.', exist_ok=True)

    # 프레임별 처리
    sorted_frames = sorted(frame_data.keys())

    for frame_num in tqdm(sorted_frames, desc="Processing frames"):
        frame_detections = frame_data[frame_num]

        # 첫 번째 detection에서 이미지 정보 추출
        first_det = next(iter(frame_detections.values()))
        image_name = first_det['image_name']
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Cannot read image - {image_path}")
            continue

        h, w = image.shape[:2]

        # 비디오 writer 초기화 (첫 프레임에서)
        if output_video and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

        # 각 track_id에 대해 그리기
        for track_id, det_data in sorted(frame_detections.items()):
            bbox = det_data['bbox']
            keypoints = det_data['keypoints']
            color = get_track_color(track_id)

            # bbox 그리기
            image = draw_bbox(image, bbox, track_id, color)

            # keypoint 그리기
            image = draw_keypoints(image, keypoints, track_id, conf_threshold, color)
            
        # 프레임 번호 표시
        cv2.putText(image, f"Frame: {frame_num}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 결과 저장
        if output_dir:
            output_path = os.path.join(output_dir, f"pose_{frame_num:06d}.jpg")
            cv2.imwrite(output_path, image)

        if video_writer:
            video_writer.write(image)

    # 비디오 finalize
    if video_writer:
        video_writer.release()
        print(f"Saved video: {output_video}")

    if output_dir:
        print(f"Saved {len(sorted_frames)} frames to {output_dir}")


def process_all_sequences(
    tracking_dir: str,
    image_folder: str,
    csv_filename: str = "pose_estimation.csv",
    output_dir_suffix: str = "_pose_visualization",
    conf_threshold: float = 0.3,
    fps: int = 30
):
    """
    모든 시퀀스에 대해 pose visualization 배치 처리

    Args:
        tracking_dir: tracking_results 디렉토리 경로
        image_folder: 이미지 폴더 경로
        csv_filename: 입력 CSV 파일명
        output_dir_suffix: 출력 이미지 폴더 suffix
        conf_threshold: keypoint confidence threshold
        fps: 출력 비디오 fps (미사용, 호환성 유지)
    """
    print("=" * 70)
    print("Batch Pose Visualization")
    print("=" * 70)
    print(f"Tracking dir: {tracking_dir}")
    print(f"Image folder: {image_folder}")
    print(f"Input CSV: {csv_filename}")
    print("=" * 70)

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

    print(f"Found {len(sequences)} sequences: {sequences}\n")

    # 각 시퀀스 처리
    for seq_name in sorted(sequences):
        print(f"\n{'='*50}")
        print(f"Processing: {seq_name}")
        print(f"{'='*50}")

        pose_csv = os.path.join(tracking_dir, seq_name, csv_filename)
        output_dir = os.path.join(tracking_dir, seq_name, f"{seq_name}{output_dir_suffix}")
        process_sequence(
            pose_csv=pose_csv,
            image_folder=image_folder,
            output_dir=output_dir,
            conf_threshold=conf_threshold,
            fps=fps
        )

    print(f"\n{'='*70}")
    print("All sequences visualized!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Pose Estimation Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 시퀀스 비디오 생성
  python visualize_pose_estimation.py \\
    --pose_csv tracking_results/seq1/pose_estimation.csv \\
    --image_folder dataset/images/val \\
    --output_video output.mp4

  # 단일 시퀀스 이미지 저장
  python visualize_pose_estimation.py \\
    --pose_csv tracking_results/seq1/pose_estimation.csv \\
    --image_folder dataset/images/val \\
    --output_dir output_frames

  # 배치 모드 (모든 시퀀스)
  python visualize_pose_estimation.py \\
    --batch \\
    --tracking_dir tracking_results \\
    --image_folder dataset/images/val
        """
    )

    # 배치 모드
    parser.add_argument('--batch', action='store_true',
                        help='배치 모드: 모든 시퀀스 처리')
    parser.add_argument('--tracking_dir', type=str, default='tracking_results',
                        help='Tracking 결과 디렉토리 (배치 모드)')
    parser.add_argument('--input_csv', type=str, default='pose_estimation_finetuned.csv',
                        help='입력 CSV 파일명 (배치 모드)')

    # 단일 모드
    parser.add_argument('--pose_csv', type=str, default=None,
                        help='Path to pose_estimation.csv (단일 모드)')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Output video path (단일 모드)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output image folder (단일 모드)')

    # 공통 옵션
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to image folder')
    parser.add_argument('--conf_threshold', type=float, default=0.0,
                        help='Keypoint confidence threshold (default: 0.0)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS (default: 30)')

    args = parser.parse_args()

    if args.batch:
        # 배치 모드
        process_all_sequences(
            tracking_dir=args.tracking_dir,
            image_folder=args.image_folder,
            csv_filename=args.input_csv,
            conf_threshold=args.conf_threshold,
            fps=args.fps
        )
    else:
        # 단일 모드
        if args.pose_csv is None:
            parser.error("단일 모드에서는 --pose_csv가 필요합니다. 또는 --batch 옵션을 사용하세요.")

        # output_dir을 기본값으로 설정 (pose_csv 폴더 기준)
        if args.output_dir is None and args.output_video is None:
            pose_dir = os.path.dirname(args.pose_csv)
            args.output_dir = os.path.join(pose_dir, "pose_visualization")

        process_sequence(
            pose_csv=args.pose_csv,
            image_folder=args.image_folder,
            output_dir=args.output_dir,
            output_video=args.output_video,
            conf_threshold=args.conf_threshold,
            fps=args.fps
        )


if __name__ == '__main__':
    main()
