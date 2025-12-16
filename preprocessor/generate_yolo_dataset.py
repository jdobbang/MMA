import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse
from ultralytics import YOLO
import glob


def calculate_dice(box1, box2):
    """
    Calculate Dice coefficient between two boxes
    box format: [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    dice = 2 * intersection / (area1 + area2) if (area1 + area2) > 0 else 0.0
    return dice


def get_max_dice_with_players(yolo_bbox, player_bboxes):
    """Calculate maximum Dice Score between YOLO detection and all player bboxes"""
    max_dice = 0.0
    for player_bbox in player_bboxes:
        dice = calculate_dice(yolo_bbox, player_bbox)
        max_dice = max(max_dice, dice)
    return max_dice


def bbox_to_yolo_format(bbox, img_width, img_height, class_id):
    """
    Convert bbox [x1, y1, x2, y2] to YOLO format [class, x_center, y_center, width, height] (normalized)
    Returns None if bbox is invalid
    """
    x1, y1, x2, y2 = bbox
    
    # Clip coordinates to image boundaries
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    # Check if bbox is valid
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Calculate normalized values
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Validate normalized values
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def process_sequence(sequence_dir, model, yolo_conf=0.1, dice_threshold=0.7):
    """Process a single sequence and generate YOLO annotations"""
    
    print(f"\n{'='*80}")
    print(f"Processing sequence: {os.path.basename(sequence_dir)}")
    print(f"{'='*80}")
    
    exo_dir = os.path.join(sequence_dir, 'exo')
    bbox_dir = os.path.join(sequence_dir, 'processed_data', 'bbox')
    output_dir = os.path.join(sequence_dir, 'processed_data', 'bbox_player_crowd')
    
    # Check if directories exist
    if not os.path.exists(exo_dir) or not os.path.exists(bbox_dir):
        print(f"Skipping {sequence_dir}: missing exo or bbox directory")
        return
    
    # Only process per cam
    cameras = ['cam01']
    print(f"Processing cameras: {cameras}")
    
    # Process each camera
    for camera in cameras:
        print(f"\nProcessing {camera}...")
        
        img_dir = os.path.join(exo_dir, camera, 'images')
        bbox_cam_dir = os.path.join(bbox_dir, camera)
        output_cam_dir = os.path.join(output_dir, camera)
        
        if not os.path.exists(img_dir) or not os.path.exists(bbox_cam_dir):
            print(f"  Skipping {camera}: missing images or bbox directory")
            continue
        
        os.makedirs(output_cam_dir, exist_ok=True)
        
        # Get all frame files
        bbox_files = sorted(glob.glob(os.path.join(bbox_cam_dir, '*.npy')))
        
        # Process each frame
        for bbox_file in tqdm(bbox_files, desc=f"  {camera}"):
            frame_name = os.path.basename(bbox_file).replace('.npy', '')
            img_path = os.path.join(img_dir, f'{frame_name}.jpg')
            output_txt_path = os.path.join(output_cam_dir, f'{frame_name}.txt')
            
            # Skip if output file already exists
            if os.path.exists(output_txt_path):
                continue
            
            if not os.path.exists(img_path):
                continue
            
            # Load image to get dimensions
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            img_height, img_width = image.shape[:2]
            
            # Load player bbox annotations
            player_bboxes_dict = np.load(bbox_file, allow_pickle=True).item()
            player_bboxes = list(player_bboxes_dict.values())
            
            # Prepare YOLO annotations
            yolo_annotations = []
            
            # Add player annotations (class 0)
            for player_bbox in player_bboxes:
                yolo_line = bbox_to_yolo_format(player_bbox, img_width, img_height, class_id=0)
                if yolo_line is not None:
                    yolo_annotations.append(yolo_line)
            
            # Run YOLO detection for crowd
            results = model(image, conf=yolo_conf, classes=[0], verbose=False)
            
            # Process YOLO detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    yolo_bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    # Calculate max Dice Score with player bboxes
                    max_dice = get_max_dice_with_players(yolo_bbox, player_bboxes)
                    
                    # Skip if Dice Score >= threshold (too similar to player)
                    if max_dice >= dice_threshold:
                        continue
                    
                    # Add as crowd (class 1)
                    yolo_line = bbox_to_yolo_format(yolo_bbox, img_width, img_height, class_id=1)
                    if yolo_line is not None:
                        yolo_annotations.append(yolo_line)
            
            # Save YOLO annotation file
            output_txt_path = os.path.join(output_cam_dir, f'{frame_name}.txt')
            with open(output_txt_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))


def main():
    parser = argparse.ArgumentParser(description='Generate YOLO dataset with player and crowd annotations')
    parser.add_argument('--root_dir', required=True, help='Root directory containing all sequences (e.g., 12_mma/12_mma)')
    parser.add_argument('--yolo_model', default='yolo11x.pt', help='YOLO model to use')
    parser.add_argument('--yolo_conf', type=float, default=0.1, help='YOLO confidence threshold')
    parser.add_argument('--dice_threshold', type=float, default=0.7, help='Dice threshold to exclude player duplicates')
    parser.add_argument('--sequences', nargs='+', default=None, help='Specific sequences to process (e.g., 001_mma 002_mma)')
    
    args = parser.parse_args()
    
    # Load YOLO model
    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)
    
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
    
    # Process each sequence sequentially
    for sequence_dir in sequence_dirs:
        process_sequence(sequence_dir, model, args.yolo_conf, args.dice_threshold)
    
    print(f"\n{'='*80}")
    print("All sequences processed!")
    print(f"Annotations saved to: processed_data/bbox_player_crowd/")
    print(f"Format: YOLO txt files (class 0=player, class 1=crowd)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()