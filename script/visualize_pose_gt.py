#!/usr/bin/env python3
"""
YOLO Pose Dataset Annotation Visualizer
========================================

images/val의 이미지와 labels/val의 annotation을 함께 시각화하여
annotation 품질을 확인합니다.

YOLO Pose Format:
- class_id center_x center_y width height kp1_x kp1_y kp1_vis ... kp17_x kp17_y kp17_vis
- 모든 좌표는 정규화됨 (0-1 범위)
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


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

# 색상 정의
COLORS = {
    'skeleton': (0, 255, 0),    # 초록색
    'keypoint': (0, 0, 255),    # 빨간색
    'bbox': (255, 0, 0),        # 파란색
    'low_conf': (0, 165, 255),  # 주황색
}

KEYPOINT_SIZE = 5
SKELETON_WIDTH = 2


def parse_yolo_pose_label(label_path: str, img_width: int, img_height: int) -> dict:
    """
    YOLO pose format 레이블 파일을 파싱하여 이미지 좌표로 변환

    Format: class_id center_x center_y width height kp1_x kp1_y kp1_vis ... kp17_x kp17_y kp17_vis
    (모든 좌표는 정규화됨)

    Returns:
        {
            'bbox': (x1, y1, x2, y2),
            'keypoints': [
                {'name': str, 'x': float, 'y': float, 'visible': bool, 'conf': float},
                ...
            ]
        }
    """
    result = {'bbox': None, 'keypoints': []}

    if not os.path.exists(label_path):
        return result

    with open(label_path, 'r') as f:
        line = f.readline().strip()
        if not line:
            return result

        values = list(map(float, line.split()))

        # bbox 처리 (정규화된 좌표 -> 이미지 좌표)
        class_id = int(values[0])
        center_x = values[1] * img_width
        center_y = values[2] * img_height
        width = values[3] * img_width
        height = values[4] * img_height

        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)

        result['bbox'] = (max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2))

        # 키포인트 처리
        idx = 5
        for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
            if idx + 2 < len(values):
                kp_x = values[idx] * img_width
                kp_y = values[idx + 1] * img_height
                kp_conf = values[idx + 2] if idx + 2 < len(values) else 0.0

                # visibility: conf > 0이면 visible
                visible = kp_conf > 0

                result['keypoints'].append({
                    'name': kp_name,
                    'x': int(kp_x),
                    'y': int(kp_y),
                    'visible': visible,
                    'conf': kp_conf
                })

                idx += 3

    return result


def draw_pose_annotation(image: np.ndarray, annotation: dict, conf_threshold: float = 0.3) -> np.ndarray:
    """
    annotation을 이미지에 그리기

    Args:
        image: 입력 이미지
        annotation: parse_yolo_pose_label의 결과
        conf_threshold: 표시할 최소 confidence 값

    Returns:
        annotation이 그려진 이미지
    """
    img = image.copy()
    h, w = img.shape[:2]

    # bbox 그리기
    if annotation['bbox']:
        x1, y1, x2, y2 = annotation['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS['bbox'], 2)

    # 키포인트 그리기
    keypoints = annotation['keypoints']

    # skeleton 그리기 (먼저 그려야 위에 표시됨)
    for start_idx, end_idx in SKELETON:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_kp = keypoints[start_idx]
            end_kp = keypoints[end_idx]

            # 둘 다 visible하고 confidence 충분하면 선 그리기
            if start_kp['visible'] and end_kp['visible'] and \
               start_kp['conf'] >= conf_threshold and end_kp['conf'] >= conf_threshold:
                cv2.line(img, (start_kp['x'], start_kp['y']),
                        (end_kp['x'], end_kp['y']), COLORS['skeleton'], SKELETON_WIDTH)

    # 개별 키포인트 그리기
    for kp in keypoints:
        if kp['visible']:
            color = COLORS['keypoint'] if kp['conf'] >= conf_threshold else COLORS['low_conf']
            cv2.circle(img, (kp['x'], kp['y']), KEYPOINT_SIZE, color, -1)
            cv2.circle(img, (kp['x'], kp['y']), KEYPOINT_SIZE, (255, 255, 255), 1)

    return img


def visualize_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str = None,
    conf_threshold: float = 0.3,
    max_images: int = 50,
    show_stats: bool = True
):
    """
    데이터셋의 annotation을 시각화

    Args:
        image_dir: images/val 디렉토리
        label_dir: labels/val 디렉토리
        output_dir: 결과 이미지 저장 디렉토리 (None이면 저장 안 함)
        conf_threshold: keypoint confidence threshold
        max_images: 최대 처리 이미지 수
        show_stats: 통계 정보 출력
    """
    image_files = sorted([f for f in os.listdir(image_dir)
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    image_files = image_files[:max_images]
    print(f"Found {len(image_files)} images to visualize")

    # 통계 수집
    stats = {
        'total_images': 0,
        'images_with_annotation': 0,
        'images_without_annotation': 0,
        'total_keypoints': 0,
        'visible_keypoints': 0,
        'keypoint_conf_dist': defaultdict(int),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("Visualizing Dataset Annotations")
    print("="*70)

    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_file)
        label_file = Path(img_file).stem + '.txt'
        label_path = os.path.join(label_dir, label_file)

        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            print(f"[{idx}/{len(image_files)}] SKIP: Cannot read {img_file}")
            continue

        h, w = image.shape[:2]

        # 레이블 파싱
        annotation = parse_yolo_pose_label(label_path, w, h)

        # 통계 업데이트
        stats['total_images'] += 1
        if annotation['keypoints']:
            stats['images_with_annotation'] += 1
            for kp in annotation['keypoints']:
                stats['total_keypoints'] += 1
                if kp['visible']:
                    stats['visible_keypoints'] += 1

                # confidence 분포
                conf_range = int(kp['conf'] * 10) / 10.0
                stats['keypoint_conf_dist'][f"{conf_range:.1f}"] += 1
        else:
            stats['images_without_annotation'] += 1

        # annotation 그리기
        img_annotated = draw_pose_annotation(image, annotation, conf_threshold)

        # 텍스트 정보 추가
        info_text = f"{img_file} | Keypoints: {sum(1 for kp in annotation['keypoints'] if kp['visible'])}/{len(annotation['keypoints'])}"
        cv2.putText(img_annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 결과 저장 또는 표시
        if output_dir:
            output_path = os.path.join(output_dir, f"vis_{img_file}")
            cv2.imwrite(output_path, img_annotated)
            status = "SAVED"
        else:
            status = "OK"

        has_anno = "✓" if annotation['keypoints'] else "✗"
        print(f"[{idx}/{len(image_files)}] {has_anno} {img_file} ... {status}")

    # 통계 출력
    if show_stats:
        print("\n" + "="*70)
        print("Dataset Statistics")
        print("="*70)
        print(f"Total images: {stats['total_images']}")
        print(f"Images with annotation: {stats['images_with_annotation']}")
        print(f"Images without annotation: {stats['images_without_annotation']}")
        print(f"Total keypoints: {stats['total_keypoints']}")
        print(f"Visible keypoints: {stats['visible_keypoints']}")
        if stats['total_keypoints'] > 0:
            print(f"Visibility rate: {stats['visible_keypoints'] / stats['total_keypoints'] * 100:.1f}%")

        print(f"\nKeypoint Confidence Distribution:")
        for conf_range in sorted(stats['keypoint_conf_dist'].keys()):
            count = stats['keypoint_conf_dist'][conf_range]
            print(f"  {conf_range}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize YOLO Pose Dataset Annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 시각화 (결과 저장)
  python script/visualize_pose_dataset.py \\
    --image_dir /workspace/MMA/dataset/yolo_pose_topdown_dataset/images/val \\
    --label_dir /workspace/MMA/dataset/yolo_pose_topdown_dataset/labels/val \\
    --output_dir ./pose_vis_results

  # 커스텀 confidence threshold
  python visualize_pose_dataset.py \\
    --image_dir images/val \\
    --label_dir labels/val \\
    --output_dir vis_results \\
    --conf_threshold 0.5

  # 최대 100개 이미지만 처리
  python visualize_pose_dataset.py \\
    --image_dir images/val \\
    --label_dir labels/val \\
    --output_dir vis_results \\
    --max_images 100
        """
    )

    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to images/val directory')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='Path to labels/val directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory to save visualizations (optional)')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                       help='Keypoint confidence threshold (default: 0.3)')
    parser.add_argument('--max_images', type=int, default=50,
                       help='Maximum number of images to process (default: 50)')
    parser.add_argument('--no_stats', action='store_true',
                       help='Don\'t print statistics')

    args = parser.parse_args()

    visualize_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        max_images=args.max_images,
        show_stats=not args.no_stats
    )

    if args.output_dir:
        print(f"\n✓ Visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
