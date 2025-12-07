#!/usr/bin/env python3
"""
Concatenate inference_base and inference_base_bytetrack results side by side
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import glob


def concat_images(base_dir, output_dir):
    """
    Concatenate detection and tracking visualizations horizontally
    
    Args:
        base_dir: Base directory containing inference_base and inference_base_bytetrack folders
        output_dir: Output directory for concatenated results
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths to visualization folders
    detection_dir = base_dir / "inference_base" / "visualizations"
    tracking_dir = base_dir / "inference_base_bytetrack" / "visualizations"
    
    if not detection_dir.exists():
        print(f"Error: Detection directory not found: {detection_dir}")
        return
    
    if not tracking_dir.exists():
        print(f"Error: Tracking directory not found: {tracking_dir}")
        return
    
    # Get all images from detection directory
    detection_images = {}
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for img_path in detection_dir.glob(ext):
            detection_images[img_path.name] = img_path
    
    # Get all images from tracking directory
    tracking_images = {}
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for img_path in tracking_dir.glob(ext):
            tracking_images[img_path.name] = img_path
    
    # Find common images
    common_names = set(detection_images.keys()) & set(tracking_images.keys())
    
    if len(common_names) == 0:
        print("Error: No common images found between detection and tracking folders")
        return
    
    print(f"Found {len(common_names)} common images")
    print(f"Detection dir: {detection_dir}")
    print(f"Tracking dir: {tracking_dir}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Process each common image
    success_count = 0
    
    for img_name in tqdm(sorted(common_names), desc="Concatenating images"):
        detection_path = detection_images[img_name]
        tracking_path = tracking_images[img_name]
        
        # Read both images
        det_img = cv2.imread(str(detection_path))
        track_img = cv2.imread(str(tracking_path))
        
        if det_img is None:
            print(f"Warning: Could not read {detection_path}")
            continue
        
        if track_img is None:
            print(f"Warning: Could not read {tracking_path}")
            continue
        
        # Check if images have the same height
        if det_img.shape[0] != track_img.shape[0]:
            # Resize to match height
            target_height = min(det_img.shape[0], track_img.shape[0])
            det_aspect = det_img.shape[1] / det_img.shape[0]
            track_aspect = track_img.shape[1] / track_img.shape[0]
            
            det_img = cv2.resize(det_img, (int(target_height * det_aspect), target_height))
            track_img = cv2.resize(track_img, (int(target_height * track_aspect), target_height))
        
        # Add text labels at the top
        label_height = 40
        det_with_label = np.zeros((det_img.shape[0] + label_height, det_img.shape[1], 3), dtype=np.uint8)
        det_with_label[:label_height, :] = (50, 50, 50)  # Dark gray background
        det_with_label[label_height:, :] = det_img
        cv2.putText(det_with_label, "Detection (Confidence-based)", (10, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        track_with_label = np.zeros((track_img.shape[0] + label_height, track_img.shape[1], 3), dtype=np.uint8)
        track_with_label[:label_height, :] = (50, 50, 50)
        track_with_label[label_height:, :] = track_img
        cv2.putText(track_with_label, "Tracking (ByteTrack)", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Concatenate horizontally
        concat_img = np.hstack([det_with_label, track_with_label])
        
        # Save concatenated image
        output_path = output_dir / img_name
        cv2.imwrite(str(output_path), concat_img)
        success_count += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Concatenation Summary")
    print("="*60)
    print(f"Total images processed: {success_count}")
    print(f"Output directory: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Concatenate Detection and Tracking Results')
    parser.add_argument('--base-dir', type=str, 
                       default='/workspace/log/mma_training_v1_202512052',
                       help='Base directory containing inference folders')
    parser.add_argument('--output', type=str, 
                       default=None,
                       help='Output directory (default: base_dir/detection_tracking)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output is None:
        args.output = os.path.join(args.base_dir, 'detection_tracking')
    
    # Run concatenation
    concat_images(args.base_dir, args.output)


if __name__ == "__main__":
    main()
