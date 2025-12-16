import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def yolo_to_xyxy(x_center, y_center, w, h, img_width=1920, img_height=1080):
    """Convert YOLO format to [x1, y1, x2, y2]"""
    x1 = (x_center - w/2) * img_width
    y1 = (y_center - h/2) * img_height
    x2 = (x_center + w/2) * img_width
    y2 = (y_center + h/2) * img_height
    return [x1, y1, x2, y2]


def analyze_player_iou_distribution(source_dir='/workspace/dataset/yolodataset/labels'):
    """
    모든 시퀀스의 0번 클래스(player) 간 IoU 분포 분석
    """
    
    iou_data = {
        'low': [],      # IoU < 0.2
        'medium': [],   # 0.2 <= IoU < 0.5
        'high': [],     # IoU >= 0.5
        'all': []
    }
    
    file_info = {
        'low': [],
        'medium': [],
        'high': []
    }
    
    print("Analyzing player IoU distribution...\n")
    
    # Process train and val splits
    for split in ['train', 'val']:
        split_dir = os.path.join(source_dir, split)
        
        if not os.path.exists(split_dir):
            continue
        
        print(f"Processing {split}...")
        
        txt_files = sorted(glob.glob(f"{split_dir}/*.txt"))
        
        for txt_file in tqdm(txt_files, desc=f"  {split}"):
            # Skip if file doesn't exist (broken symlink)
            if not os.path.exists(txt_file):
                continue
                
            # Read player bboxes (class 0)
            player_boxes = []
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        if class_id == 0:  # player only
                            x_c, y_c, w, h = map(float, parts[1:5])
                            box = yolo_to_xyxy(x_c, y_c, w, h)
                            player_boxes.append(box)
            
            # Calculate IoU if exactly 2 players
            if len(player_boxes) == 2:
                iou = calculate_iou(player_boxes[0], player_boxes[1])
                iou_data['all'].append(iou)
                
                file_path = f"{split}/{os.path.basename(txt_file)}"
                
                if iou < 0.2:
                    iou_data['low'].append(iou)
                    file_info['low'].append((file_path, iou))
                elif iou < 0.5:
                    iou_data['medium'].append(iou)
                    file_info['medium'].append((file_path, iou))
                else:
                    iou_data['high'].append(iou)
                    file_info['high'].append((file_path, iou))    # Print statistics
    print(f"\n{'='*80}")
    print("IoU Distribution Statistics")
    print(f"{'='*80}")
    print(f"Total samples: {len(iou_data['all'])}")
    print(f"\nLow overlap (IoU < 0.2):    {len(iou_data['low']):6d} ({len(iou_data['low'])/len(iou_data['all'])*100:.1f}%)")
    print(f"Medium overlap (0.2-0.5):   {len(iou_data['medium']):6d} ({len(iou_data['medium'])/len(iou_data['all'])*100:.1f}%)")
    print(f"High overlap (IoU >= 0.5):  {len(iou_data['high']):6d} ({len(iou_data['high'])/len(iou_data['all'])*100:.1f}%)")
    
    print(f"\nIoU Statistics:")
    print(f"  Mean: {np.mean(iou_data['all']):.4f}")
    print(f"  Median: {np.median(iou_data['all']):.4f}")
    print(f"  Std: {np.std(iou_data['all']):.4f}")
    print(f"  Min: {np.min(iou_data['all']):.4f}")
    print(f"  Max: {np.max(iou_data['all']):.4f}")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(iou_data['all'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0.2, color='red', linestyle='--', label='Low/Medium threshold (0.2)')
    plt.axvline(0.5, color='orange', linestyle='--', label='Medium/High threshold (0.5)')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.title('Player IoU Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/workspace/dataset/iou_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\nHistogram saved to: /workspace/dataset/iou_distribution.png")
    
    print(f"{'='*80}\n")
    
    return iou_data, file_info


if __name__ == "__main__":
    iou_data, file_info = analyze_player_iou_distribution()