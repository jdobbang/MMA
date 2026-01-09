#!/usr/bin/env python3
"""
ViTPose-based 2D Pose Estimation for MMA Tracking Results
===========================================================

Tracking 결과(step2_interpolated.csv)를 기반으로
single person에 대한 2D pose estimation을 수행하고 CSV로 저장

사용 모델: ViTPose-base (MMPose)
호환성: 기존 YOLO11-pose와 동일한 입출력 형식 유지
"""
import argparse
import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Tuple, Optional


# COCO Keypoint 정의 (17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# ViTPose Checkpoint URLs
VITPOSE_CHECKPOINT_URLS = {
    'small': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth',
    'base': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth',
    'large': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth',
    'huge': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
}


def get_vitpose_config(model_size: str, custom_config: Optional[str] = None) -> str:
    """
    Returns ViTPose config path.

    Priority:
    1. custom_config if provided
    2. mmpose built-in config based on model_size
    """
    if custom_config:
        return custom_config

    # Map model sizes to mmpose built-in configs
    config_map = {
        'small': 'td-hm_ViTPose-small_8xb64-210e_coco-256x192',
        'base': 'td-hm_ViTPose-base_8xb64-210e_coco-256x192',
        'large': 'td-hm_ViTPose-large_8xb64-210e_coco-256x192',
        'huge': 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192'
    }

    config_name = config_map.get(model_size.lower())
    if not config_name:
        raise ValueError(f"Invalid model_size: {model_size}. Choose from: {list(config_map.keys())}")

    # Return path to mmpose installation config
    try:
        import mmpose
        config_path = os.path.join(
            os.path.dirname(mmpose.__file__),
            '.mim/configs/body_2d_keypoint/topdown_heatmap/coco',
            f'{config_name}.py'
        )
        return config_path
    except ImportError:
        raise ImportError("MMPose not installed. Please run: pip install mmpose mmcv mmengine mmdet")


def get_default_checkpoint(model_size: str) -> str:
    """Returns default checkpoint URL for given model size."""
    url = VITPOSE_CHECKPOINT_URLS.get(model_size.lower())
    if not url:
        raise ValueError(f"Invalid model_size: {model_size}")
    return url


def init_vitpose_model(
    model_size: str = 'base',
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda'
):
    """
    ViTPose 모델 초기화

    Args:
        model_size: Model variant ('small', 'base', 'large', 'huge')
        config_path: Custom config path (overrides model_size)
        checkpoint_path: Checkpoint path or URL (defaults to pretrained COCO)
        device: 'cuda' 또는 'cpu'

    Returns:
        MMPose model instance
    """
    try:
        from mmpose.apis import init_model
        # Register mmpretrain models for ViTPose backbone
        import mmpretrain  # noqa: F401
    except ImportError:
        raise ImportError(
            "MMPose not installed. Please run:\n"
            "pip install mmpose mmcv mmengine mmdet mmpretrain"
        )

    # Get config path
    config = get_vitpose_config(model_size, config_path)

    # Get checkpoint (use default URL if not provided)
    checkpoint = checkpoint_path or get_default_checkpoint(model_size)

    print(f"Loading ViTPose-{model_size} model from: {checkpoint}")
    model = init_model(config, checkpoint, device=device)
    print(f"Model loaded successfully on device: {device}")

    return model


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


def crop_person(image: np.ndarray, bbox: dict, padding: float = 0.0) -> tuple:
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


def run_pose_on_crop_vitpose(
    model,
    cropped_image: np.ndarray,
    original_bbox: dict,
    crop_info: dict,
    conf_threshold: float = 0.3
) -> dict:
    """
    crop된 이미지에서 ViTPose inference 수행

    Args:
        model: ViTPose model
        cropped_image: crop된 이미지
        original_bbox: 원본 이미지의 bbox (x1, y1, x2, y2)
        crop_info: crop 정보 (offset_x, offset_y)
        conf_threshold: keypoint confidence threshold

    Returns:
        dict: {keypoint_name: (x, y, conf)} - 원본 이미지 좌표계
    """
    if cropped_image.size == 0:
        return {}

    try:
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples
    except ImportError:
        raise ImportError("MMPose not installed")

    try:
        # ViTPose inference (MMPose 통합 API)
        results = inference_topdown(model, cropped_image)

        if results is None or len(results) == 0:
            return {}

        # 첫 번째 결과에서 keypoint 추출
        result = results[0]

        # pred_instances에서 keypoints와 keypoint_scores 추출
        if not hasattr(result, 'pred_instances'):
            return {}

        pred_instances = result.pred_instances

        if not hasattr(pred_instances, 'keypoints') or len(pred_instances.keypoints) == 0:
            return {}

        # keypoints: (1, 17, 2), keypoint_scores: (1, 17)
        kpts_xy = pred_instances.keypoints[0]  # (17, 2)
        kpts_conf = pred_instances.keypoint_scores[0] if hasattr(pred_instances, 'keypoint_scores') else np.ones(17)

        # 원본 이미지 좌표로 변환
        keypoints = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            x = float(kpts_xy[i, 0]) + crop_info['offset_x']
            y = float(kpts_xy[i, 1]) + crop_info['offset_y']
            conf = float(kpts_conf[i]) if isinstance(kpts_conf, np.ndarray) else float(kpts_conf)
            keypoints[name] = (x, y, conf)

        return keypoints

    except Exception as e:
        print(f"Error during ViTPose inference: {e}")
        return {}


def process_sequence(
    tracking_csv: str,
    image_folder: str,
    output_csv: str,
    model,
    padding: float = 0.1,
    keypoint_conf_threshold: float = 0.3
):
    """
    시퀀스 전체에 대해 ViTPose 기반 pose estimation 수행

    Args:
        tracking_csv: step2_interpolated.csv 경로
        image_folder: 이미지 폴더 경로
        output_csv: 출력 CSV 경로
        model: ViTPose model
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

        for frame_num in tqdm(sorted_frames, desc="Processing frames (ViTPose)"):
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

                # ViTPose inference
                keypoints = run_pose_on_crop_vitpose(model, cropped, original_bbox=det, crop_info=crop_info, conf_threshold=keypoint_conf_threshold)

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
    model,
    csv_filename: str = "step2_interpolated.csv",
    output_filename: str = "pose_estimation_vitpose.csv",
    padding: float = 0.0,
    keypoint_conf_threshold: float = 0.3
):
    """
    모든 시퀀스에 대해 ViTPose 기반 pose estimation 배치 처리

    Args:
        tracking_dir: tracking_results 디렉토리 경로
        image_folder: 이미지 폴더 경로
        model: ViTPose model
        csv_filename: 입력 CSV 파일명
        output_filename: 출력 CSV 파일명
        padding: bbox padding 비율
        keypoint_conf_threshold: keypoint confidence threshold
    """
    print("=" * 70)
    print("Batch Pose Estimation (ViTPose)")
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
        description='ViTPose-based 2D Pose Estimation for MMA Tracking Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 시퀀스 처리 (ViTPose-base 사전학습 모델)
  python pose_estimation_vitpose.py \\
    --tracking_csv tracking_results/seq1/step2_interpolated.csv \\
    --image_folder dataset/images/val/ \\
    --model-size base

  # 전체 시퀀스 배치 처리 (ViTPose-base)
  python pose_estimation_vitpose.py --batch \\
    --tracking_dir tracking_results \\
    --image_folder dataset/images/val/ \\
    --model-size base

  # 큰 모델 사용 (더 높은 정확도)
  python pose_estimation_vitpose.py --batch \\
    --tracking_dir tracking_results \\
    --image_folder dataset/images/val/ \\
    --model-size large

  # 작은 모델 사용 (더 빠른 속도)
  python pose_estimation_vitpose.py --batch \\
    --tracking_dir tracking_results \\
    --image_folder dataset/images/val/ \\
    --model-size small

  # CPU 사용 (느림)
  python pose_estimation_vitpose.py --batch \\
    --tracking_dir tracking_results \\
    --image_folder dataset/images/val/ \\
    --model-size base \\
    --device cpu

  # Fine-tuned 모델 사용
  python pose_estimation_vitpose.py --batch \\
    --tracking_dir tracking_results \\
    --image_folder dataset/images/val/ \\
    --model-size base \\
    --checkpoint checkpoints/vitpose_base_mma_finetuned.pth

사전학습 모델 다운로드:
  ViTPose-small:  https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth
  ViTPose-base:   https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth
  ViTPose-large:  https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth
  ViTPose-huge:   https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth

모델 크기별 성능:
  - small: 가장 빠름 (AP 0.735)
  - base: 속도-정확도 균형 (AP 0.757) ← 권장
  - large: 높은 정확도 (AP 0.784)
  - huge: 최고 정확도 (AP 0.814)
        """
    )

    # 배치 모드
    parser.add_argument('--batch', action='store_true',
                        help='배치 모드: 모든 시퀀스 폴더에 대해 처리')
    parser.add_argument('--tracking_dir', type=str, default='tracking_results',
                        help='Tracking 결과 디렉토리 (배치 모드)')
    parser.add_argument('--input_csv', type=str, default='step2_interpolated.csv',
                        help='입력 CSV 파일명 (배치 모드)')
    parser.add_argument('--output_name', type=str, default='pose_estimation_vitpose.csv',
                        help='출력 CSV 파일명 (배치 모드)')

    # 단일 모드
    parser.add_argument('--tracking_csv', type=str, default=None,
                        help='Path to step2_interpolated.csv (단일 모드)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV path (단일 모드)')

    # 공통 옵션
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to image folder')

    # ViTPose 모델 옵션
    parser.add_argument('--model-size', type=str, default='base',
                        choices=['small', 'base', 'large', 'huge'],
                        help='ViTPose model size (default: base)')
    parser.add_argument('--config', type=str, default=None,
                        help='Custom ViTPose config file path (overrides --model-size)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='ViTPose checkpoint file path or URL (default: pretrained from OpenMMLab)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # 처리 옵션
    parser.add_argument('--padding', type=float, default=0.0,
                        help='Bbox padding ratio (default: 0.0)')
    parser.add_argument('--keypoint_conf', type=float, default=0.3,
                        help='Keypoint confidence threshold (default: 0.3)')

    args = parser.parse_args()

    # 모델 로드
    model = init_vitpose_model(
        model_size=args.model_size,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

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
            args.output_csv = os.path.join(output_dir, 'pose_estimation_vitpose.csv')

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
