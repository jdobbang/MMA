import cv2
import os
import numpy as np
from tqdm import tqdm  # 진행상황 표시용

# 1. 경로 설정
BASE_PATH = "/workspace/MMA/dataset"
POSE_ROOT = os.path.join(BASE_PATH, "yolo_pose_dataset")
BBOX_ROOT = os.path.join(BASE_PATH, "yolodataset")
OUTPUT_ROOT = os.path.join(BASE_PATH, "cropped_dataset_overlap")

splits = ['train', 'val']

def get_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def check_overlap(box1, box2):
    """두 bbox(center_x, center_y, w, h) 간에 겹침이 있는지 확인"""
    b1_x1, b1_y1 = box1[1] - box1[3]/2, box1[2] - box1[4]/2
    b1_x2, b1_y2 = box1[1] + box1[3]/2, box1[2] + box1[4]/2
    
    b2_x1, b2_y1 = box2[1] - box2[3]/2, box2[2] - box2[4]/2
    b2_x2, b2_y2 = box2[1] + box2[3]/2, box2[2] + box2[4]/2
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    return inter_x2 > inter_x1 and inter_y2 > inter_y1

print("🚀 Overlap 상황 데이터 추출 및 크롭을 시작합니다.")

for split in splits:
    img_dir = os.path.join(POSE_ROOT, "images", split)
    pose_label_dir = os.path.join(POSE_ROOT, "labels", split)
    bbox_label_dir = os.path.join(BBOX_ROOT, "labels", split)

    out_img_dir = os.path.join(OUTPUT_ROOT, "images", split)
    out_label_dir = os.path.join(OUTPUT_ROOT, "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    if not os.path.exists(img_dir):
        print(f"⚠️ {split} 폴더를 찾을 수 없어 스킵합니다.")
        continue

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # tqdm 적용: desc로 현재 어떤 split을 처리 중인지 표시
    pbar = tqdm(image_files, desc=f"Processing {split}", unit="img")
    
    success_count = 0
    for img_name in pbar:
        basename = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        pose_path = os.path.join(pose_label_dir, f"{basename}.txt")
        bbox_path = os.path.join(bbox_label_dir, f"{basename}.txt")

        if not (os.path.exists(pose_path) and os.path.exists(bbox_path)):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape

        # Class 0(Person) 필터링
        with open(bbox_path, 'r') as f:
            bboxes = [list(map(float, line.split())) for line in f.readlines() if line.strip().startswith('0')]
        with open(pose_path, 'r') as f:
            poses = [list(map(float, line.split())) for line in f.readlines() if line.strip().startswith('0')]

        if len(bboxes) < 2: continue

        # 2. Overlap 여부 검사
        overlap_indices = set()
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if check_overlap(bboxes[i], bboxes[j]):
                    overlap_indices.add(i)
                    overlap_indices.add(j)

        if not overlap_indices: continue

        # 3. 크롭 및 변환 작업
        for idx in overlap_indices:
            box = bboxes[idx]
            cls, bx, by, bw, bh = box
            
            best_pose = None
            min_dist = float('inf')
            for p in poses:
                dist = get_distance((bx, by), (p[1], p[2]))
                if dist < min_dist:
                    min_dist = dist
                    best_pose = p
            
            if best_pose is None or min_dist > 0.1: continue

            x1, y1 = int((bx - bw/2) * w), int((by - bh/2) * h)
            x2, y2 = int((bx + bw/2) * w), int((by + bh/2) * h)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop_w, crop_h = x2 - x1, y2 - y1

            if crop_w <= 10 or crop_h <= 10: continue

            crop_img = img[y1:y2, x1:x2]
            new_name = f"{basename}_p{idx}"
            cv2.imwrite(os.path.join(out_img_dir, f"{new_name}.jpg"), crop_img)

            kpts = best_pose[5:]
            new_kpts = []
            for i in range(0, len(kpts), 3):
                kx_norm, ky_norm, kv = kpts[i], kpts[i+1], kpts[i+2]
                if kv > 0:
                    nkx = (kx_norm * w - x1) / crop_w
                    nky = (ky_norm * h - y1) / crop_h
                    if 0 <= nkx <= 1 and 0 <= nky <= 1:
                        new_kpts.extend([nkx, nky, kv])
                    else:
                        new_kpts.extend([0.0, 0.0, 0])
                else:
                    new_kpts.extend([0.0, 0.0, 0])

            label_data = [int(cls), 0.5, 0.5, 1.0, 1.0] + new_kpts
            with open(os.path.join(out_label_dir, f"{new_name}.txt"), 'w') as f:
                f.write(" ".join(map(lambda x: f"{x:.6f}", label_data)) + "\n")
            
            success_count += 1
        
        # tqdm 상태바 옆에 현재까지 추출된 크롭 이미지 수 표시
        pbar.set_postfix(crops=success_count)

print(f"\n✅ 작업 완료! 결과 폴더: {OUTPUT_ROOT}")