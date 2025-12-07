#!/usr/bin/env python3
"""
YOLO Inference and Visualization Script
Run inference on images using trained best.pt model and save visualized results
"""

import argparse
import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import glob


def inference_and_visualize(model_path, source, output_dir, conf_threshold=0.25, save_txt=False):
    """
    Run YOLO inference and save visualized results
    
    Args:
        model_path: Path to best.pt model
        source: Path to image file or directory
        output_dir: Output directory for results
        conf_threshold: Confidence threshold for detections
        save_txt: Save label txt files
    """
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    if save_txt:
        label_dir = output_dir / "labels"
        label_dir.mkdir(exist_ok=True)
    
    # Get image files
    source_path = Path(source)
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(str(source_path / ext)))
            image_files.extend(glob.glob(str(source_path / ext.upper())))
        image_files = sorted([Path(f) for f in image_files])
    else:
        raise ValueError(f"Invalid source: {source}")
    
    if len(image_files) == 0:
        print(f"No images found in {source}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {conf_threshold}")
    print()
    
    # Class names
    class_names = {0: 'player', 1: 'crowd'}
    
    # Confidence-based colors for class 0 (player) only
    # Low: 0.25-0.50 (Yellow), Medium: 0.50-0.75 (Orange), High: 0.75-1.00 (Green)
    def get_color_by_confidence(conf):
        if conf < 0.50:
            return (0, 255, 255)  # Yellow (BGR)
        elif conf < 0.75:
            return (0, 165, 255)  # Orange (BGR)
        else:
            return (0, 255, 0)    # Green (BGR)
    
    # Process each image
    stats = {'player_low': 0, 'player_mid': 0, 'player_high': 0, 'total_images': 0}
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Run inference
        results = model(image, conf=conf_threshold, verbose=False)
        
        # Draw results on image
        vis_image = image.copy()
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates and info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Only visualize class 0 (player)
                if cls != 0:
                    continue
                
                # Update stats based on confidence
                if conf < 0.50:
                    stats['player_low'] += 1
                    conf_level = 'Low'
                elif conf < 0.75:
                    stats['player_mid'] += 1
                    conf_level = 'Mid'
                else:
                    stats['player_high'] += 1
                    conf_level = 'High'
                
                # Get color based on confidence
                color = get_color_by_confidence(conf)
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"player {conf:.2f} ({conf_level})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Save detection for txt file
                if save_txt:
                    img_h, img_w = image.shape[:2]
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    detections.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}")
        
        # Save visualized image
        output_path = vis_dir / img_path.name
        cv2.imwrite(str(output_path), vis_image)
        
        # Save label txt file
        if save_txt and detections:
            label_path = label_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(detections))
        
        stats['total_images'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Inference Summary")
    print("="*60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Total players detected: {stats['player_low'] + stats['player_mid'] + stats['player_high']}")
    print(f"  - Low confidence (-0.50): {stats['player_low']} [Yellow]")
    print(f"  - Mid confidence (0.50-0.75): {stats['player_mid']} [Orange]")
    print(f"  - High confidence (0.75-1.00): {stats['player_high']} [Green]")
    print(f"Average detections per image: {(stats['player_low'] + stats['player_mid'] + stats['player_high']) / max(stats['total_images'], 1):.2f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Visualizations: {vis_dir}")
    if save_txt:
        print(f"  - Labels: {label_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='YOLO Inference and Visualization')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to best.pt model')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--output', type=str, default='/workspace/inference_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='Confidence threshold (default: 0.1)')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save detection results as txt files')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    # Run inference
    inference_and_visualize(
        model_path=args.model,
        source=args.source,
        output_dir=args.output,
        conf_threshold=args.conf,
        save_txt=args.save_txt
    )


if __name__ == "__main__":
    main()

"""
python script/inference_base.py --model log/mma_training_v1_202512052/weights/best.pt --source dataset/yolodataset/images/val --output log/mma_training_v1_202512052/inference_base --conf 0.1
"""