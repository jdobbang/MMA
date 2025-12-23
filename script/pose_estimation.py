#!/usr/bin/env python3
"""
Pose Estimation for MMA Tracking Results
=========================================

Tracking 결과(step2_interpolated.csv)를 기반으로
single person에 대한 2D pose estimation을 수행하고 CSV로 저장

사용 모델: YOLO11-pose (Ultralytics)
"""

import argparse
import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict


# COCO Keypoint 정의 (17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def load_tracking_csv(csv_path: str) -> dict:
    """
    tracking CSV 로드하여 frame별로 그룹화

    Returns:
        dict: {frame_num: [(image_name, track_id, x1, y1, x2, y2, conf), ...]}
    """
    frame_data = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            frame_data[frame].append({
                'image_name': row['image_name'],
                'track_id': int(row['track_id']),
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
                'confidence': float(row['confidence'])
            })

    return frame_data


def crop_person(image: np.ndarray, bbox: dict, padding: float = 0.1) -> tuple:
    """
    bbox 기준으로 사람 영역 crop (padding 포함)

    Args:
        image: 원본 이미지
        bbox: dict with x1, y1, x2, y2
        padding: bbox 확장 비율 (0.1 = 10%)

    Returns:
        (cropped_image, crop_info) - crop_info는 좌표 변환용
    """
    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

    # padding 적용
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding
    pad_y = box_h * padding

    # 확장된 bbox (이미지 경계 내로 제한)
    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(w, int(x2 + pad_x))
    crop_y2 = min(h, int(y2 + pad_y))

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_info = {
        'offset_x': crop_x1,
        'offset_y': crop_y1,
        'crop_w': crop_x2 - crop_x1,
        'crop_h': crop_y2 - crop_y1
    }

    return cropped, crop_info


def run_pose_on_crop(model: YOLO, cropped_image: np.ndarray, conf_threshold: float = 0.3) -> dict:
    """
    crop된 이미지에서 pose estimation 수행

    Args:
        model: YOLO pose model
        cropped_image: crop된 사람 이미지
        conf_threshold: keypoint confidence threshold

    Returns:
        dict: {keypoint_name: (x, y, conf)} - crop 좌표계
    """
    if cropped_image.size == 0:
        return {}

    results = model.predict(cropped_image, conf=0.1, verbose=False)

    if len(results) == 0 or results[0].keypoints is None:
        return {}

    keypoints_data = results[0].keypoints

    # 가장 confidence 높은 detection 선택 (single person이므로)
    if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
        return {}

    # 첫 번째 detection 사용 (crop 내에 1명만 있다고 가정)
    kpts_xy = keypoints_data.xy[0].cpu().numpy()  # (17, 2)
    kpts_conf = keypoints_data.conf[0].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)

    keypoints = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = kpts_xy[i]
        conf = float(kpts_conf[i])
        keypoints[name] = (float(x), float(y), conf)

    return keypoints


def transform_keypoints_to_original(keypoints: dict, crop_info: dict) -> dict:
    """
    crop 좌표계 -> 원본 이미지 좌표계로 변환
    """
    transformed = {}
    for name, (x, y, conf) in keypoints.items():
        orig_x = x + crop_info['offset_x']
        orig_y = y + crop_info['offset_y']
        transformed[name] = (orig_x, orig_y, conf)
    return transformed


def process_sequence(
    tracking_csv: str,
    image_folder: str,
    output_csv: str,
    model: YOLO,
    padding: float = 0.1,
    keypoint_conf_threshold: float = 0.3
):
    """
    시퀀스 전체에 대해 pose estimation 수행

    Args:
        tracking_csv: step2_interpolated.csv 경로
        image_folder: 이미지 폴더 경로
        output_csv: 출력 CSV 경로
        model: YOLO pose model
        padding: bbox crop padding 비율
        keypoint_conf_threshold: keypoint confidence threshold
    """
    # tracking 데이터 로드
    frame_data = load_tracking_csv(tracking_csv)

    if not frame_data:
        print(f"No tracking data found in {tracking_csv}")
        return

    print(f"Loaded {len(frame_data)} frames from tracking CSV")

    # CSV 헤더 생성
    header = ['image_name', 'frame', 'track_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox_conf']
    for kp_name in KEYPOINT_NAMES:
        header.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_conf'])

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

    # 결과 저장
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # 프레임별 처리
        sorted_frames = sorted(frame_data.keys())

        for frame_num in tqdm(sorted_frames, desc="Processing frames"):
            detections = frame_data[frame_num]

            # 첫 번째 detection에서 이미지 이름 추출
            image_name = detections[0]['image_name']
            image_path = os.path.join(image_folder, image_name)

            if not os.path.exists(image_path):
                print(f"Warning: Image not found - {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Cannot read image - {image_path}")
                continue

            # 각 track_id에 대해 pose estimation
            for det in detections:
                track_id = det['track_id']

                # bbox crop
                cropped, crop_info = crop_person(image, det, padding=padding)

                # pose estimation
                keypoints = run_pose_on_crop(model, cropped, conf_threshold=keypoint_conf_threshold)

                # 원본 좌표계로 변환
                if keypoints:
                    keypoints = transform_keypoints_to_original(keypoints, crop_info)

                # CSV row 생성
                row = [
                    image_name,
                    frame_num,
                    track_id,
                    det['x1'], det['y1'], det['x2'], det['y2'],
                    det['confidence']
                ]

                # keypoint 데이터 추가
                for kp_name in KEYPOINT_NAMES:
                    if kp_name in keypoints:
                        x, y, conf = keypoints[kp_name]
                        row.extend([x, y, conf])
                    else:
                        row.extend([0.0, 0.0, 0.0])  # missing keypoint

                writer.writerow(row)

    print(f"Pose estimation results saved to: {output_csv}")


def process_all_sequences(
    tracking_dir: str,
    image_folder: str,
    model: YOLO,
    csv_filename: str = "step2_interpolated.csv",
    output_filename: str = "pose_estimation.csv",
    padding: float = 0.1,
    keypoint_conf_threshold: float = 0.3
):
    """
    모든 시퀀스에 대해 pose estimation 배치 처리

    Args:
        tracking_dir: tracking_results 디렉토리 경로
        image_folder: 이미지 폴더 경로
        model: YOLO pose model
        csv_filename: 입력 CSV 파일명 (default: step2_interpolated.csv)
        output_filename: 출력 CSV 파일명 (default: pose_estimation.csv)
        padding: bbox padding 비율
        keypoint_conf_threshold: keypoint confidence threshold
    """
    print("=" * 70)
    print("Batch Pose Estimation")
    print("=" * 70)
    print(f"Tracking dir: {tracking_dir}")
    print(f"Image folder: {image_folder}")
    print(f"Input CSV: {csv_filename}")
    print(f"Output CSV: {output_filename}")
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

        tracking_csv = os.path.join(tracking_dir, seq_name, csv_filename)
        output_csv = os.path.join(tracking_dir, seq_name, output_filename)

        process_sequence(
            tracking_csv=tracking_csv,
            image_folder=image_folder,
            output_csv=output_csv,
            model=model,
            padding=padding,
            keypoint_conf_threshold=keypoint_conf_threshold
        )

    print(f"\n{'='*70}")
    print("All sequences processed!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='2D Pose Estimation for MMA Tracking Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 시퀀스 처리
  python pose_estimation.py --tracking_csv tracking_results/seq1/step2_interpolated.csv \\
                            --image_folder dataset/images/val/

  # 전체 시퀀스 배치 처리
  python pose_estimation.py --batch \\
                            --tracking_dir tracking_results \\
                            --image_folder dataset/images/val/

  # 특정 모델 사용
  python pose_estimation.py --batch --tracking_dir tracking_results \\
                            --image_folder dataset/images/val/ \\
                            --model yolo11l-pose.pt
        """
    )

    # 배치 모드
    parser.add_argument('--batch', action='store_true',
                        help='배치 모드: 모든 시퀀스 폴더에 대해 처리')
    parser.add_argument('--tracking_dir', type=str, default='tracking_results',
                        help='Tracking 결과 디렉토리 (배치 모드)')
    parser.add_argument('--input_csv', type=str, default='step2_interpolated.csv',
                        help='입력 CSV 파일명 (배치 모드, default: step2_interpolated.csv)')
    parser.add_argument('--output_name', type=str, default='pose_estimation.csv',
                        help='출력 CSV 파일명 (배치 모드, default: pose_estimation.csv)')

    # 단일 모드
    parser.add_argument('--tracking_csv', type=str, default=None,
                        help='Path to step2_interpolated.csv (단일 모드)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV path (단일 모드, default: same folder as tracking_csv)')

    # 공통 옵션
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to image folder')
    parser.add_argument('--model', type=str, default='yolo11m-pose.pt',
                        help='YOLO pose model (default: yolo11m-pose.pt)')
    parser.add_argument('--padding', type=float, default=0.1,
                        help='Bbox padding ratio (default: 0.1)')
    parser.add_argument('--keypoint_conf', type=float, default=0.3,
                        help='Keypoint confidence threshold (default: 0.3)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # 모델 로드
    print(f"Loading pose model: {args.model}")
    model = YOLO(args.model)

    if args.batch:
        # 배치 모드
        process_all_sequences(
            tracking_dir=args.tracking_dir,
            image_folder=args.image_folder,
            model=model,
            csv_filename=args.input_csv,
            output_filename=args.output_name,
            padding=args.padding,
            keypoint_conf_threshold=args.keypoint_conf
        )
    else:
        # 단일 모드
        if args.tracking_csv is None:
            parser.error("단일 모드에서는 --tracking_csv가 필요합니다. 또는 --batch 옵션을 사용하세요.")

        if args.output_csv is None:
            output_dir = os.path.dirname(args.tracking_csv)
            args.output_csv = os.path.join(output_dir, 'pose_estimation.csv')

        process_sequence(
            tracking_csv=args.tracking_csv,
            image_folder=args.image_folder,
            output_csv=args.output_csv,
            model=model,
            padding=args.padding,
            keypoint_conf_threshold=args.keypoint_conf
        )


if __name__ == '__main__':
    main()
