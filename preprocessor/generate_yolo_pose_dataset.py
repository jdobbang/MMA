#!/usr/bin/env python3
"""
Generate YOLO Pose Dataset from poses2d annotations
====================================================

기존 poses2d 데이터를 YOLO11-pose 학습용 형식으로 변환
- Player만 사용 (고품질 annotation)
- YOLO keypoint format: class x_center y_center width height kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
- Visibility: 모두 2 (visible)로 설정
"""

import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse
import glob


# COCO Keypoint 정의 (17 keypoints) - poses2d와 YOLO 모두 COCO 형식 사용
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# poses2d의 keypoint 순서 (45개 keypoint 중 COCO 17개 매핑)
# poses2d는 SMPL-X 기반이므로 COCO keypoint만 추출
POSES2D_TO_COCO_MAPPING = {
    # COCO 인덱스: poses2d 인덱스
    0: 15,   # nose
    1: 26,   # left_eye
    2: 25,   # right_eye
    3: 27,   # left_ear
    4: 28,   # right_ear
    5: 16,   # left_shoulder
    6: 17,   # right_shoulder
    7: 18,   # left_elbow
    8: 19,   # right_elbow
    9: 20,   # left_wrist
    10: 21,  # right_wrist
    11: 1,   # left_hip
    12: 2,   # right_hip
    13: 4,   # left_knee
    14: 5,   # right_knee
    15: 7,   # left_ankle
    16: 8,   # right_ankle
}


def load_poses2d(poses2d_file, camera=None):
    """
    Load poses2d data from .npy file

    Args:
        poses2d_file: path to .npy file
        camera: camera name (default: None = load all cameras)

    Returns:
        dict: {player_id: keypoints_array (45, 2)} or None if file not exists
    """
    if not os.path.exists(poses2d_file):
        return None

    try:
        data = np.load(poses2d_file, allow_pickle=True).item()

        if camera is not None:
            # Single camera
            if camera not in data:
                return None
            keypoints = data[camera]
            return {0: keypoints}
        else:
            # All cameras - map to player IDs
            result = {}
            for player_id, cam_name in enumerate(sorted(data.keys())):
                result[player_id] = data[cam_name]
            return result
    except:
        return None


def load_bbox(bbox_file):
    """
    Load bbox data from .npy file

    Args:
        bbox_file: path to .npy file

    Returns:
        dict: {player_id: [x1, y1, x2, y2]} or None if file not exists
    """
    if not os.path.exists(bbox_file):
        return None

    try:
        bbox_dict = np.load(bbox_file, allow_pickle=True).item()
        # bbox_dict: {player_id: [x1, y1, x2, y2]}
        return bbox_dict
    except:
        return None


def convert_poses2d_to_coco(poses2d_keypoints):
    """
    Convert poses2d keypoints (45) to COCO keypoints (17)

    Args:
        poses2d_keypoints: numpy array (45, 2)

    Returns:
        numpy array (17, 2)
    """
    coco_keypoints = np.zeros((17, 2))

    for coco_idx, poses2d_idx in POSES2D_TO_COCO_MAPPING.items():
        if poses2d_idx < len(poses2d_keypoints):
            coco_keypoints[coco_idx] = poses2d_keypoints[poses2d_idx]

    return coco_keypoints


def bbox_from_keypoints(keypoints, img_width, img_height, padding=0.1):
    """
    Calculate bounding box from keypoints with padding

    Args:
        keypoints: numpy array (17, 2)
        img_width: image width
        img_height: image height
        padding: padding ratio

    Returns:
        [x1, y1, x2, y2] or None if invalid
    """
    valid_kpts = keypoints[~np.isnan(keypoints).any(axis=1)]

    if len(valid_kpts) == 0:
        return None

    x_min = valid_kpts[:, 0].min()
    y_min = valid_kpts[:, 1].min()
    x_max = valid_kpts[:, 0].max()
    y_max = valid_kpts[:, 1].max()

    # Apply padding
    w = x_max - x_min
    h = y_max - y_min
    x_min = max(0, x_min - w * padding)
    y_min = max(0, y_min - h * padding)
    x_max = min(img_width, x_max + w * padding)
    y_max = min(img_height, y_max + h * padding)

    # Validate bbox
    if x_max <= x_min or y_max <= y_min:
        return None

    return [x_min, y_min, x_max, y_max]


def keypoints_to_yolo_format(keypoints, bbox, img_width, img_height, class_id=0):
    """
    Convert keypoints and bbox to YOLO pose format

    YOLO pose format:
    class x_center y_center width height kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v

    Args:
        keypoints: numpy array (17, 2) - COCO format
        bbox: [x1, y1, x2, y2]
        img_width: image width
        img_height: image height
        class_id: class id (default: 0 for person)

    Returns:
        str: YOLO format line or None if invalid
    """
    x1, y1, x2, y2 = bbox

    # Validate bbox
    if x2 <= x1 or y2 <= y1:
        return None

    # Clip to image boundaries
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    if x2 <= x1 or y2 <= y1:
        return None

    # Normalize bbox
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # Validate normalized values
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None

    # Start with bbox
    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    # Add keypoints (17 keypoints × 3 values)
    for kpt in keypoints:
        x, y = kpt

        # Normalize keypoint coordinates
        x_norm = x / img_width
        y_norm = y / img_height

        # Clip to [0, 1]
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))

        # Visibility: 2 (visible) for all keypoints
        visibility = 2

        yolo_line += f" {x_norm:.6f} {y_norm:.6f} {visibility}"

    return yolo_line


def process_sequence(sequence_dir, camera='cam01', bbox_padding=0.1, use_bbox_from_file=True):
    """
    Process a single sequence and generate YOLO pose annotations

    Args:
        sequence_dir: path to sequence directory
        camera: camera name (default: cam01) - used for images and bbox directories only
        bbox_padding: padding ratio for bbox calculation from keypoints
        use_bbox_from_file: use bbox from bbox/ folder if True, otherwise calculate from keypoints
    """
    print(f"\n{'='*80}")
    print(f"Processing sequence: {os.path.basename(sequence_dir)}")
    print(f"{'='*80}")

    exo_dir = os.path.join(sequence_dir, 'exo')
    poses2d_dir = os.path.join(sequence_dir, 'processed_data', 'poses2d')
    bbox_dir = os.path.join(sequence_dir, 'processed_data', 'bbox')
    output_dir = os.path.join(sequence_dir, 'processed_data', 'bbox_pose')

    # Check if directories exist
    if not os.path.exists(exo_dir) or not os.path.exists(poses2d_dir):
        print(f"Skipping {sequence_dir}: missing exo or poses2d directory")
        return

    # Camera-specific paths for images and output
    img_dir = os.path.join(exo_dir, camera, 'images')
    poses2d_cam_dir = os.path.join(poses2d_dir, camera)
    bbox_cam_dir = os.path.join(bbox_dir, camera) if use_bbox_from_file else None
    output_cam_dir = os.path.join(output_dir, camera)

    if not os.path.exists(img_dir) or not os.path.exists(poses2d_cam_dir):
        print(f"Skipping {camera}: missing images or poses2d directory")
        return

    os.makedirs(output_cam_dir, exist_ok=True)

    # Get all pose files
    pose_files = sorted(glob.glob(os.path.join(poses2d_cam_dir, '*.npy')))

    print(f"Found {len(pose_files)} pose files")

    # Process each frame
    valid_count = 0
    skipped_count = 0

    for pose_file in tqdm(pose_files, desc=f"  {camera}"):
        frame_name = os.path.basename(pose_file).replace('.npy', '')
        img_path = os.path.join(img_dir, f'{frame_name}.jpg')
        output_txt_path = os.path.join(output_cam_dir, f'{frame_name}.txt')

        # Skip if output file already exists
        if os.path.exists(output_txt_path):
            valid_count += 1
            continue

        if not os.path.exists(img_path):
            skipped_count += 1
            continue

        # Load image to get dimensions
        image = cv2.imread(img_path)
        if image is None:
            skipped_count += 1
            continue

        img_height, img_width = image.shape[:2]

        # Load poses2d data (all players/cameras)
        poses2d_data = load_poses2d(pose_file)
        if poses2d_data is None:
            skipped_count += 1
            continue

        # Prepare YOLO annotations
        yolo_annotations = []

        # Process each player (usually just one)
        for player_id, poses2d_keypoints in poses2d_data.items():
            # Convert poses2d (45 keypoints) to COCO (17 keypoints)
            coco_keypoints = convert_poses2d_to_coco(poses2d_keypoints)

            # Get bbox
            if use_bbox_from_file and bbox_cam_dir is not None:
                bbox_file = os.path.join(bbox_cam_dir, f'{frame_name}.npy')
                bbox_data = load_bbox(bbox_file)
                if bbox_data is not None and player_id in bbox_data:
                    bbox = bbox_data[player_id]
                else:
                    # Fallback: calculate from keypoints
                    bbox = bbox_from_keypoints(coco_keypoints, img_width, img_height, bbox_padding)
            else:
                # Calculate bbox from keypoints
                bbox = bbox_from_keypoints(coco_keypoints, img_width, img_height, bbox_padding)

            if bbox is None:
                continue

            # Convert to YOLO format
            yolo_line = keypoints_to_yolo_format(coco_keypoints, bbox, img_width, img_height, class_id=0)

            if yolo_line is not None:
                yolo_annotations.append(yolo_line)

        # Save YOLO annotation file
        if yolo_annotations:
            with open(output_txt_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            valid_count += 1
        else:
            skipped_count += 1

    print(f"  Valid: {valid_count}, Skipped: {skipped_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate YOLO pose dataset from poses2d annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific sequences
  python generate_yolo_pose_dataset.py --root_dir /workspace/MMA/dataset/03_grappling2 \\
                                       --sequences 001_grappling2 002_grappling2

  # Process all sequences in a dataset
  python generate_yolo_pose_dataset.py --root_dir /workspace/MMA/dataset/13_mma2

  # Use keypoint-based bbox calculation instead of bbox files
  python generate_yolo_pose_dataset.py --root_dir /workspace/MMA/dataset/03_grappling2 \\
                                       --no-use-bbox-file
        """
    )

    parser.add_argument('--root_dir', required=True,
                        help='Root directory containing all sequences (e.g., /workspace/MMA/dataset/03_grappling2)')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='Specific sequences to process (e.g., 001_grappling2 002_grappling2)')
    parser.add_argument('--camera', default='cam01',
                        help='Camera to process (default: cam01)')
    parser.add_argument('--bbox_padding', type=float, default=0.1,
                        help='Bbox padding ratio when calculating from keypoints (default: 0.1)')
    parser.add_argument('--no-use-bbox-file', action='store_true',
                        help='Calculate bbox from keypoints instead of using bbox files')

    args = parser.parse_args()

    # Get all sequence directories
    if args.sequences:
        sequence_dirs = [os.path.join(args.root_dir, seq) for seq in args.sequences]
    else:
        mma_dirs = glob.glob(os.path.join(args.root_dir, '*_mma*'))
        grappling_dirs = glob.glob(os.path.join(args.root_dir, '*_grappling*'))
        sequence_dirs = sorted(mma_dirs + grappling_dirs)

    # Filter only existing directories
    sequence_dirs = [d for d in sequence_dirs if os.path.isdir(d)]

    print(f"\nFound {len(sequence_dirs)} sequences to process")
    print(f"Sequences: {[os.path.basename(d) for d in sequence_dirs]}")
    print(f"Camera: {args.camera}")
    print(f"Use bbox from file: {not args.no_use_bbox_file}")

    # Process each sequence
    for sequence_dir in sequence_dirs:
        process_sequence(
            sequence_dir,
            camera=args.camera,
            bbox_padding=args.bbox_padding,
            use_bbox_from_file=not args.no_use_bbox_file
        )

    print(f"\n{'='*80}")
    print("All sequences processed!")
    print(f"Annotations saved to: processed_data/bbox_pose/")
    print(f"Format: YOLO pose txt files (class 0=person, 17 keypoints)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
