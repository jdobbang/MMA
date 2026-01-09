#!/usr/bin/env python3

"""

Create Detection-based Top-Down Pose Dataset

=============================================



Detection bbox 기준으로 crop하여 RTMPose fine-tuning용 데이터셋 생성

- IoU > 0.1로 겹치는 2명의 케이스만 필터링

- Detection bbox와 pose-bbox 간 IoU 기반 최적 매칭

- 추론 환경과 동일한 crop 방식



Usage:

  # 기본 실행

  python crop_detection_pose_dataset.py



  # 커스텀 threshold

  python crop_detection_pose_dataset.py --iou-overlap 0.15 --padding 0.1



  # 출력 경로 지정

  python crop_detection_pose_dataset.py --output /path/to/output

"""



import cv2

import numpy as np

import os

import yaml

from tqdm import tqdm

import argparse

from typing import List, Tuple, Dict, Optional, Set, Any





# =============================================================================

# 기본 경로 및 상수

# =============================================================================

BASE_PATH = "/workspace/MMA/dataset"

DETECTION_ROOT = os.path.join(BASE_PATH, "yolodataset")

POSE_ROOT = os.path.join(BASE_PATH, "yolo_pose_dataset")

OUTPUT_ROOT = os.path.join(BASE_PATH, "yolo_pose_detection_crop_dataset")



SPLITS = ['train', 'val']

NUM_KEYPOINTS = 17

MIN_KEYPOINTS = 3

MIN_CROP_SIZE = 32

IOU_OVERLAP_THRESHOLD = 0.01  # 2명 겹침 판단 threshold

PADDING_RATIO = 0.0          # detection bbox에 추가 padding (0 = 원본 유지)





# =============================================================================

# NPY 파일 로드 함수

# =============================================================================

def load_npy_data(npy_path: str) -> Dict[str, Any]:

    """

    NPY 파일 로드 (aria01, aria02 키 포함)



    Args:

        npy_path: npy 파일 경로



    Returns:

        Dict: 'aria01', 'aria02' 등의 키를 포함하는 딕셔너리

    """

    if not os.path.exists(npy_path):

        return {}



    try:

        data = np.load(npy_path, allow_pickle=True).item()

        return data if isinstance(data, dict) else {}

    except Exception as e:

        print(f"Error loading {npy_path}: {e}")

        return {}




# =============================================================================

# IoU 계산 함수

# =============================================================================

def calculate_iou(box1: List[float], box2: List[float]) -> float:

    """

    두 bbox 간 IoU 계산



    Args:

        box1, box2: [x_center, y_center, width, height] 정규화 좌표



    Returns:

        float: IoU 값 (0.0 ~ 1.0)

    """

    # YOLO 형식 (center, size) -> corner 형식으로 변환

    b1_x1 = box1[0] - box1[2] / 2

    b1_y1 = box1[1] - box1[3] / 2

    b1_x2 = box1[0] + box1[2] / 2

    b1_y2 = box1[1] + box1[3] / 2



    b2_x1 = box2[0] - box2[2] / 2

    b2_y1 = box2[1] - box2[3] / 2

    b2_x2 = box2[0] + box2[2] / 2

    b2_y2 = box2[1] + box2[3] / 2



    # 교집합 영역

    inter_x1 = max(b1_x1, b2_x1)

    inter_y1 = max(b1_y1, b2_y1)

    inter_x2 = min(b1_x2, b2_x2)

    inter_y2 = min(b1_y2, b2_y2)



    # 교집합 면적

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:

        return 0.0



    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)



    # 각 bbox 면적

    b1_area = box1[2] * box1[3]

    b2_area = box2[2] * box2[3]



    # 합집합 면적

    union_area = b1_area + b2_area - inter_area



    if union_area <= 0:

        return 0.0



    return inter_area / union_area





# =============================================================================

# 겹침 필터링 함수

# =============================================================================

def filter_overlapping_detections(

    detection_bboxes: List[Dict],

    iou_threshold: float = 0.1

) -> List[int]:

    """

    IoU > threshold로 겹치는 detection bbox 인덱스들을 반환



    Args:

        detection_bboxes: [{'class_id': 0, 'bbox': [x, y, w, h]}, ...]

        iou_threshold: 겹침 판단 threshold



    Returns:

        List[int]: 겹치는 detection의 인덱스 목록

    """

    # Person class (class_id == 0)만 필터링

    person_indices = [i for i, d in enumerate(detection_bboxes)

                      if d['class_id'] == 0]



    if len(person_indices) < 2:

        return []



    overlapping_indices: Set[int] = set()



    for i in range(len(person_indices)):

        for j in range(i + 1, len(person_indices)):

            idx_i = person_indices[i]

            idx_j = person_indices[j]



            iou = calculate_iou(

                detection_bboxes[idx_i]['bbox'],

                detection_bboxes[idx_j]['bbox']

            )



            if iou > iou_threshold:

                overlapping_indices.add(idx_i)

                overlapping_indices.add(idx_j)



    return list(overlapping_indices)





# =============================================================================

# ARIA 키 기반 Detection-Pose 매칭 함수

# =============================================================================

def match_detection_to_pose_by_aria(

    detection_bboxes: List[Dict],

    pose_annotations: List[Dict],

    detection_indices: List[int],

    aria_keys: List[str]

) -> List[Tuple[int, int, float, str]]:

    """

    ARIA 키(aria01, aria02 등)를 이용한 Detection bbox와 pose annotation 매칭



    Args:

        detection_bboxes: detection 라벨 목록

        pose_annotations: pose 라벨 목록 (aria 키 포함)

        detection_indices: 매칭할 detection 인덱스들

        aria_keys: aria01, aria02 등의 키 목록



    Returns:

        List[Tuple[det_idx, pose_idx, iou, aria_key]]: 매칭 결과 (aria 키 포함)

    """

    if not detection_indices or not pose_annotations or not aria_keys:

        return []



    matches = []

    used_pose_indices: Set[int] = set()



    # Detection과 pose 간 IoU 행렬 계산

    iou_matrix = np.zeros((len(detection_indices), len(pose_annotations)))



    for i, det_idx in enumerate(detection_indices):

        det_bbox = detection_bboxes[det_idx]['bbox']

        for j, pose in enumerate(pose_annotations):

            pose_bbox = pose['bbox']

            iou_matrix[i, j] = calculate_iou(det_bbox, pose_bbox)



    # Greedy 매칭: IoU가 높은 순서로 매칭

    while True:

        # 최대 IoU 찾기

        max_iou = 0.0

        best_det_i, best_pose_j = -1, -1



        for i, det_idx in enumerate(detection_indices):

            if any(m[0] == det_idx for m in matches):

                continue  # 이미 매칭된 detection



            for j in range(len(pose_annotations)):

                if j in used_pose_indices:

                    continue  # 이미 매칭된 pose



                if iou_matrix[i, j] > max_iou:

                    max_iou = iou_matrix[i, j]

                    best_det_i = i

                    best_pose_j = j



        # 더 이상 매칭할 것이 없으면 종료

        if best_det_i == -1 or max_iou <= 0:

            break



        # 매칭 추가 (aria_key 포함)

        det_idx = detection_indices[best_det_i]

        aria_key = aria_keys[best_pose_j]

        matches.append((det_idx, best_pose_j, max_iou, aria_key))

        used_pose_indices.add(best_pose_j)



    return matches




# =============================================================================

# 기존 Detection-Pose 매칭 함수 (호환성 유지)

# =============================================================================

def match_detection_to_pose(

    detection_bboxes: List[Dict],

    pose_annotations: List[Dict],

    detection_indices: List[int]

) -> List[Tuple[int, int, float]]:

    """

    Detection bbox와 pose annotation을 IoU 기반으로 최적 매칭



    Args:

        detection_bboxes: detection 라벨 목록

        pose_annotations: pose 라벨 목록

        detection_indices: 매칭할 detection 인덱스들



    Returns:

        List[Tuple[det_idx, pose_idx, iou]]: 매칭 결과

    """

    if not detection_indices or not pose_annotations:

        return []



    matches = []

    used_pose_indices: Set[int] = set()



    # Detection과 pose 간 IoU 행렬 계산

    iou_matrix = np.zeros((len(detection_indices), len(pose_annotations)))



    for i, det_idx in enumerate(detection_indices):

        det_bbox = detection_bboxes[det_idx]['bbox']

        for j, pose in enumerate(pose_annotations):

            pose_bbox = pose['bbox']

            iou_matrix[i, j] = calculate_iou(det_bbox, pose_bbox)



    # Greedy 매칭: IoU가 높은 순서로 매칭

    while True:

        # 최대 IoU 찾기

        max_iou = 0.0

        best_det_i, best_pose_j = -1, -1



        for i, det_idx in enumerate(detection_indices):

            if any(m[0] == det_idx for m in matches):

                continue  # 이미 매칭된 detection



            for j in range(len(pose_annotations)):

                if j in used_pose_indices:

                    continue  # 이미 매칭된 pose



                if iou_matrix[i, j] > max_iou:

                    max_iou = iou_matrix[i, j]

                    best_det_i = i

                    best_pose_j = j



        # 더 이상 매칭할 것이 없으면 종료

        if best_det_i == -1 or max_iou <= 0:

            break



        # 매칭 추가

        det_idx = detection_indices[best_det_i]

        matches.append((det_idx, best_pose_j, max_iou))

        used_pose_indices.add(best_pose_j)



    return matches





# =============================================================================

# 라벨 파싱 함수들

# =============================================================================

def parse_detection_label(label_path: str) -> List[Dict]:

    """

    Detection 라벨 파일 파싱 (yolodataset)



    형식: class_id x_center y_center width height

    """

    detections = []



    if not os.path.exists(label_path):

        return detections



    with open(label_path, 'r') as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:

                continue



            class_id = int(parts[0])

            bbox = [float(parts[i]) for i in range(1, 5)]



            detections.append({

                'class_id': class_id,

                'bbox': bbox

            })



    return detections





def parse_pose_label(label_path: str) -> List[Dict]:

    """

    Pose 라벨 파일 파싱 (yolo_pose_dataset)



    형식: class_id x_center y_center width height [17 keypoints x 3]

    총 56개 필드

    """

    poses = []



    if not os.path.exists(label_path):

        return poses



    with open(label_path, 'r') as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 56:

                continue



            values = [float(p) for p in parts]



            class_id = int(values[0])

            bbox = values[1:5]



            keypoints = []

            for i in range(NUM_KEYPOINTS):

                idx = 5 + i * 3

                kpt = (values[idx], values[idx + 1], int(values[idx + 2]))

                keypoints.append(kpt)



            poses.append({

                'class_id': class_id,

                'bbox': bbox,

                'keypoints': keypoints

            })



    return poses




# =============================================================================

# NPY 기반 Pose 데이터 변환 함수

# =============================================================================

def convert_npy_poses_to_annotations(

    bbox_data: Dict[str, Any],

    img_w: int,

    img_h: int

) -> Tuple[List[Dict], List[str]]:

    """

    NPY bbox 데이터를 pose annotation 형식으로 변환



    Args:

        bbox_data: {'aria01': [x1, y1, x2, y2], 'aria02': [...], ...}

        img_w, img_h: 이미지 크기



    Returns:

        Tuple[pose_annotations, aria_keys]

        - pose_annotations: bbox 정보를 포함한 annotation 리스트

        - aria_keys: aria01, aria02 등의 키 리스트 (순서 보존)

    """

    poses = []

    aria_keys = []



    for aria_key in sorted(bbox_data.keys()):

        bbox_pixel = bbox_data[aria_key]



        # pixel 좌표 (x1, y1, x2, y2)를 정규화 좌표로 변환

        x1, y1, x2, y2 = bbox_pixel



        x_center = (x1 + x2) / (2 * img_w)

        y_center = (y1 + y2) / (2 * img_h)

        width = (x2 - x1) / img_w

        height = (y2 - y1) / img_h



        poses.append({

            'class_id': 0,

            'bbox': [x_center, y_center, width, height],

            'aria_key': aria_key

        })

        aria_keys.append(aria_key)



    return poses, aria_keys





# =============================================================================

# Keypoint 좌표 변환 함수

# =============================================================================

def transform_keypoints_to_crop(

    keypoints: List[Tuple[float, float, int]],

    crop_bbox: Tuple[int, int, int, int],

    img_w: int,

    img_h: int

) -> List[Tuple[float, float, int]]:

    """

    원본 이미지 기준 정규화 좌표를 crop 이미지 기준으로 변환



    Args:

        keypoints: [(x_norm, y_norm, vis), ...] 원본 이미지 기준

        crop_bbox: (x1, y1, x2, y2) pixel 좌표

        img_w, img_h: 원본 이미지 크기



    Returns:

        List[Tuple]: [(x_new, y_new, vis), ...] crop 기준 정규화 좌표

    """

    x1, y1, x2, y2 = crop_bbox

    crop_w = x2 - x1

    crop_h = y2 - y1



    if crop_w <= 0 or crop_h <= 0:

        return [(0.0, 0.0, 0) for _ in keypoints]



    transformed = []



    for kx_norm, ky_norm, vis in keypoints:

        if vis == 0:

            transformed.append((0.0, 0.0, 0))

            continue



        # Step 1: 원본 pixel 좌표로 변환

        kx_pixel = kx_norm * img_w

        ky_pixel = ky_norm * img_h



        # Step 2: crop 기준 pixel 좌표로 변환

        kx_crop = kx_pixel - x1

        ky_crop = ky_pixel - y1



        # Step 3: crop 기준 정규화 좌표로 변환

        kx_new = kx_crop / crop_w

        ky_new = ky_crop / crop_h



        # 범위 검증: crop 영역 내에 있는지 확인

        if 0 <= kx_new <= 1 and 0 <= ky_new <= 1:

            transformed.append((kx_new, ky_new, vis))

        else:

            # crop 영역 밖이면 visibility 0으로 설정

            transformed.append((0.0, 0.0, 0))



    return transformed





# =============================================================================

# Detection bbox를 pixel 좌표로 변환

# =============================================================================

def detection_bbox_to_pixel(

    bbox: List[float],

    img_w: int,

    img_h: int,

    padding: float = 0.0

) -> Tuple[int, int, int, int]:

    """

    Detection bbox (정규화 좌표)를 pixel crop 좌표로 변환



    Args:

        bbox: [x_center, y_center, width, height] 정규화 좌표

        img_w, img_h: 이미지 크기

        padding: bbox에 추가할 padding 비율



    Returns:

        (x1, y1, x2, y2): pixel 좌표, 이미지 경계로 clip됨

    """

    x_center, y_center, w, h = bbox



    # 정규화 좌표 -> pixel 좌표

    cx_pixel = x_center * img_w

    cy_pixel = y_center * img_h

    w_pixel = w * img_w

    h_pixel = h * img_h



    # Padding 적용

    if padding > 0:

        w_pixel *= (1 + padding)

        h_pixel *= (1 + padding)



    # Corner 좌표 계산

    x1 = cx_pixel - w_pixel / 2

    y1 = cy_pixel - h_pixel / 2

    x2 = cx_pixel + w_pixel / 2

    y2 = cy_pixel + h_pixel / 2



    # 이미지 경계로 clip

    x1 = max(0, int(x1))

    y1 = max(0, int(y1))

    x2 = min(img_w, int(x2))

    y2 = min(img_h, int(y2))



    return (x1, y1, x2, y2)





# =============================================================================

# YOLO Pose 라벨 생성 함수

# =============================================================================

def create_yolo_pose_line(

    class_id: int,

    keypoints: List[Tuple[float, float, int]]

) -> str:

    """

    Top-down 형식의 YOLO pose 라벨 라인 생성



    Top-down에서는 crop이 전체 이미지이므로:

    - bbox: center (0.5, 0.5), size (1.0, 1.0)

    - keypoints: crop 기준 정규화 좌표

    """

    # 형식: class_id 0.5 0.5 1.0 1.0 kp1_x kp1_y kp1_v ...

    parts = [str(class_id), '0.5', '0.5', '1.0', '1.0']



    for x, y, vis in keypoints:

        parts.extend([f'{x:.6f}', f'{y:.6f}', str(vis)])



    return ' '.join(parts)





# =============================================================================

# 단일 이미지 처리 함수

# =============================================================================

def process_image(

    img_path: str,

    detection_label_path: str,

    pose_label_path: str,

    output_img_dir: str,

    output_label_dir: str,

    basename: str,

    iou_overlap_threshold: float = 0.1,

    padding: float = 0.0,

    bbox_npy_path: Optional[str] = None

) -> int:

    """

    단일 이미지 처리: 겹치는 detection에 대해 crop 생성



    Args:

        img_path: 이미지 경로

        detection_label_path: detection 라벨 경로

        pose_label_path: pose 라벨 경로 (txt 형식 또는 None)

        output_img_dir: 출력 이미지 디렉토리

        output_label_dir: 출력 라벨 디렉토리

        basename: 파일 기본명

        iou_overlap_threshold: IoU overlap threshold

        padding: bbox padding

        bbox_npy_path: bbox NPY 파일 경로 (aria 키 사용)



    Returns:

        int: 성공적으로 생성된 crop 개수

    """

    # 이미지 로드

    img = cv2.imread(img_path)

    if img is None:

        return 0



    img_h, img_w = img.shape[:2]



    # 라벨 로드

    detections = parse_detection_label(detection_label_path)



    # NPY 파일이 제공되면 사용, 아니면 txt 라벨 파일 사용

    if bbox_npy_path and os.path.exists(bbox_npy_path):

        bbox_data = load_npy_data(bbox_npy_path)

        if bbox_data:

            poses, aria_keys = convert_npy_poses_to_annotations(bbox_data, img_w, img_h)

        else:

            # NPY 로드 실패시 txt 라벨 사용

            poses = parse_pose_label(pose_label_path)

            aria_keys = []

    else:

        # NPY 파일 없으면 txt 라벨 사용

        poses = parse_pose_label(pose_label_path)

        aria_keys = []



    if len(detections) < 2 or len(poses) < 2:

        return 0



    # Step 1: 겹치는 detection 필터링

    overlapping_indices = filter_overlapping_detections(

        detections,

        iou_threshold=iou_overlap_threshold

    )



    if not overlapping_indices:

        return 0



    # Step 2: Detection-Pose 매칭 (aria 키 사용 또는 기본 매칭)

    if aria_keys:

        # NPY 기반 aria 키 매칭

        matches = match_detection_to_pose_by_aria(

            detections,

            poses,

            overlapping_indices,

            aria_keys

        )

    else:

        # 기본 IoU 기반 매칭

        matches = match_detection_to_pose(

            detections,

            poses,

            overlapping_indices

        )



    if not matches:

        return 0



    success_count = 0



    # Step 3: 각 매칭에 대해 crop 생성

    for match_tuple in matches:

        # aria 키 포함 여부에 따라 처리

        if len(match_tuple) == 4:

            det_idx, pose_idx, match_iou, aria_key = match_tuple

        else:

            det_idx, pose_idx, match_iou = match_tuple

            aria_key = None



        detection = detections[det_idx]

        pose = poses[pose_idx]



        # Detection bbox를 pixel 좌표로 변환

        crop_bbox = detection_bbox_to_pixel(

            detection['bbox'],

            img_w, img_h,

            padding=padding

        )



        x1, y1, x2, y2 = crop_bbox

        crop_w = x2 - x1

        crop_h = y2 - y1



        # 최소 크기 검증

        if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:

            continue



        # 이미지 crop

        crop_img = img[y1:y2, x1:x2]



        if crop_img.size == 0:

            continue



        # Keypoint 좌표 변환

        transformed_kpts = transform_keypoints_to_crop(

            pose['keypoints'],

            crop_bbox,

            img_w, img_h

        )



        # Visible keypoint 개수 검증

        visible_count = sum(1 for _, _, vis in transformed_kpts if vis == 2)

        if visible_count < MIN_KEYPOINTS:

            continue



        # 출력 파일명 (aria 키가 있으면 포함)

        if aria_key:

            output_name = f"{basename}_det{det_idx}_{aria_key}"

        else:

            output_name = f"{basename}_det{det_idx}"



        # Crop 이미지 저장

        output_img_path = os.path.join(output_img_dir, f"{output_name}.jpg")

        cv2.imwrite(output_img_path, crop_img)



        # 라벨 저장

        yolo_line = create_yolo_pose_line(0, transformed_kpts)

        output_label_path = os.path.join(output_label_dir, f"{output_name}.txt")

        with open(output_label_path, 'w') as f:

            f.write(yolo_line + '\n')



        success_count += 1



    return success_count





# =============================================================================

# Split 처리 함수

# =============================================================================

def process_split(

    split: str,

    detection_root: str,

    pose_root: str,

    output_root: str,

    iou_overlap_threshold: float = 0.1,

    padding: float = 0.0,

    bbox_npy_root: Optional[str] = None

) -> Tuple[int, int]:

    """

    Train 또는 val split 전체 처리



    Args:

        split: train 또는 val

        detection_root: detection 데이터셋 root

        pose_root: pose 데이터셋 root

        output_root: 출력 root

        iou_overlap_threshold: IoU threshold

        padding: bbox padding

        bbox_npy_root: bbox NPY 파일 root (aria 키 사용시)



    Returns:

        (처리된 이미지 개수, 생성된 crop 개수)

    """

    img_dir = os.path.join(pose_root, "images", split)

    det_label_dir = os.path.join(detection_root, "labels", split)

    pose_label_dir = os.path.join(pose_root, "labels", split)

    bbox_npy_dir = os.path.join(bbox_npy_root, split) if bbox_npy_root else None



    output_img_dir = os.path.join(output_root, "images", split)

    output_label_dir = os.path.join(output_root, "labels", split)



    os.makedirs(output_img_dir, exist_ok=True)

    os.makedirs(output_label_dir, exist_ok=True)



    # 이미지 파일 목록

    if not os.path.exists(img_dir):

        print(f"Warning: Image directory not found: {img_dir}")

        return 0, 0



    image_files = sorted([

        f for f in os.listdir(img_dir)

        if f.lower().endswith(('.jpg', '.jpeg', '.png'))

    ])



    total_images = 0

    total_crops = 0



    pbar = tqdm(image_files, desc=f"Processing {split}")



    for img_name in pbar:

        basename = os.path.splitext(img_name)[0]



        img_path = os.path.join(img_dir, img_name)

        det_label_path = os.path.join(det_label_dir, f"{basename}.txt")

        pose_label_path = os.path.join(pose_label_dir, f"{basename}.txt")

        bbox_npy_path = os.path.join(bbox_npy_dir, f"{basename}.npy") if bbox_npy_dir else None



        crops = process_image(

            img_path,

            det_label_path,

            pose_label_path,

            output_img_dir,

            output_label_dir,

            basename,

            iou_overlap_threshold,

            padding,

            bbox_npy_path

        )



        if crops > 0:

            total_images += 1

            total_crops += crops

            pbar.set_postfix(images=total_images, crops=total_crops)



    return total_images, total_crops





# =============================================================================

# data.yaml 생성 함수

# =============================================================================

def create_data_yaml(output_root: str, iou_overlap_threshold: float):

    """

    YOLO pose 형식의 data.yaml 생성

    """

    data_yaml = {

        'path': output_root,

        'train': 'images/train',

        'val': 'images/val',

        'kpt_shape': [17, 3],

        'names': {0: 'person'},

        'nc': 1,

        'description': 'Detection-based crop dataset for RTMPose fine-tuning',

        'overlap_iou_threshold': iou_overlap_threshold

    }



    yaml_path = os.path.join(output_root, 'data.yaml')

    with open(yaml_path, 'w') as f:

        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)



    print(f"Created: {yaml_path}")





# =============================================================================

# 메인 함수

# =============================================================================

def main():

    parser = argparse.ArgumentParser(

        description='Create detection-based crop dataset for RTMPose fine-tuning',

        formatter_class=argparse.RawDescriptionHelpFormatter,

        epilog="""

Examples:

  # 기본 실행

  python crop_detection_pose_dataset.py



  # 커스텀 threshold

  python crop_detection_pose_dataset.py --iou-overlap 0.15



  # padding 추가

  python crop_detection_pose_dataset.py --padding 0.1



  # 출력 경로 지정

  python crop_detection_pose_dataset.py --output /path/to/output

        """

    )



    parser.add_argument('--detection-root', default=DETECTION_ROOT,

                        help=f'Detection dataset root (default: {DETECTION_ROOT})')

    parser.add_argument('--pose-root', default=POSE_ROOT,

                        help=f'Pose dataset root (default: {POSE_ROOT})')

    parser.add_argument('--output', default=OUTPUT_ROOT,

                        help=f'Output directory (default: {OUTPUT_ROOT})')

    parser.add_argument('--iou-overlap', type=float, default=IOU_OVERLAP_THRESHOLD,

                        help=f'IoU threshold for overlap detection (default: {IOU_OVERLAP_THRESHOLD})')

    parser.add_argument('--padding', type=float, default=PADDING_RATIO,

                        help=f'Bbox padding ratio (default: {PADDING_RATIO})')

    parser.add_argument('--bbox-npy-root', default=None,

                        help='Bbox NPY file root directory for aria-key based matching (optional)')



    args = parser.parse_args()



    print("=" * 70)

    print("Detection-based Crop Dataset Generator")

    print("=" * 70)

    print(f"Detection root: {args.detection_root}")

    print(f"Pose root:      {args.pose_root}")

    print(f"Output:         {args.output}")

    print(f"IoU overlap:    {args.iou_overlap}")

    print(f"Padding:        {args.padding}")

    if args.bbox_npy_root:

        print(f"Bbox NPY root:  {args.bbox_npy_root}")

    print("=" * 70)



    # 출력 디렉토리 생성

    os.makedirs(args.output, exist_ok=True)



    # 각 split 처리

    total_stats = {'images': 0, 'crops': 0}



    for split in SPLITS:

        print(f"\n--- Processing {split} split ---")

        images, crops = process_split(

            split,

            args.detection_root,

            args.pose_root,

            args.output,

            args.iou_overlap,

            args.padding,

            args.bbox_npy_root

        )



        print(f"  {split}: {images} images -> {crops} crops")

        total_stats['images'] += images

        total_stats['crops'] += crops



    # data.yaml 생성

    create_data_yaml(args.output, args.iou_overlap)



    # 최종 통계 출력

    print("\n" + "=" * 70)

    print("COMPLETED")

    print("=" * 70)

    print(f"Total images with overlaps: {total_stats['images']}")

    print(f"Total crops generated:      {total_stats['crops']}")

    print(f"Output directory:           {args.output}")

    print("=" * 70)





if __name__ == '__main__':

    main()

