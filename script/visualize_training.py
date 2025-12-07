#!/usr/bin/env python3
"""
Real-time training visualization from results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
from pathlib import Path

def visualize_training(csv_path, refresh_interval=5):
    """
    Visualize training progress from results.csv
    
    Args:
        csv_path: Path to results.csv
        refresh_interval: Update interval in seconds
    """
    # Create output directory
    csv_dir = Path(csv_path).parent
    output_dir = csv_dir / "training_graph"
    output_dir.mkdir(exist_ok=True)
    
    plt.ioff()  # Non-interactive mode
    
    last_mtime = 0
    
    try:
        while True:
            # Check if file was modified
            if not os.path.exists(csv_path):
                time.sleep(refresh_interval)
                continue
            
            current_mtime = os.path.getmtime(csv_path)
            
            if current_mtime > last_mtime:
                last_mtime = current_mtime
                
                # Read CSV
                try:
                    df = pd.read_csv(csv_path)
                    
                    if len(df) == 0:
                        time.sleep(refresh_interval)
                        continue
                    
                    # Create new figure
                    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                    fig.suptitle('YOLO Training Progress', fontsize=16, fontweight='bold')
                    
                    # Clear all axes
                    for ax in axes.flat:
                        ax.clear()
                    
                    # Plot 1: Losses (train)
                    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o')
                    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss', marker='s')
                    axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='^')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].set_title('Training Losses')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # Plot 2: Losses (validation)
                    axes[0, 1].plot(df['epoch'], df['val/box_loss'], label='Box Loss', marker='o')
                    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Class Loss', marker='s')
                    axes[0, 1].plot(df['epoch'], df['val/dfl_loss'], label='DFL Loss', marker='^')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Loss')
                    axes[0, 1].set_title('Validation Losses')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Plot 3: mAP
                    axes[0, 2].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', marker='o', linewidth=2)
                    axes[0, 2].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', marker='s', linewidth=2)
                    axes[0, 2].set_xlabel('Epoch')
                    axes[0, 2].set_ylabel('mAP')
                    axes[0, 2].set_title('Mean Average Precision')
                    axes[0, 2].set_ylim([0, 1])
                    axes[0, 2].legend()
                    axes[0, 2].grid(True, alpha=0.3)
                    
                    # Plot 4: Precision & Recall
                    axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o', linewidth=2)
                    axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='s', linewidth=2)
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Score')
                    axes[1, 0].set_title('Precision & Recall')
                    axes[1, 0].set_ylim([0, 1])
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Plot 5: Learning Rate
                    axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR (pg0)', marker='o')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Learning Rate')
                    axes[1, 1].set_title('Learning Rate Schedule')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Plot 6: Summary Stats
                    axes[1, 2].axis('off')
                    latest = df.iloc[-1]
                    summary_text = f"""
Latest Epoch: {int(latest['epoch'])}
Time: {latest['time']:.1f}s

Training Losses:
  Box: {latest['train/box_loss']:.4f}
  Cls: {latest['train/cls_loss']:.4f}
  DFL: {latest['train/dfl_loss']:.4f}

Validation Losses:
  Box: {latest['val/box_loss']:.4f}
  Cls: {latest['val/cls_loss']:.4f}
  DFL: {latest['val/dfl_loss']:.4f}

Metrics:
  Precision: {latest['metrics/precision(B)']:.4f}
  Recall: {latest['metrics/recall(B)']:.4f}
  mAP@0.5: {latest['metrics/mAP50(B)']:.4f}
  mAP@0.5:0.95: {latest['metrics/mAP50-95(B)']:.4f}

Learning Rate: {latest['lr/pg0']:.6f}
                    """
                    axes[1, 2].text(0.1, 0.5, summary_text, 
                                   fontsize=11, verticalalignment='center',
                                   fontfamily='monospace',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                    
                    plt.tight_layout()
                    
                    # Save only the latest full graph
                    output_path = output_dir / "training_progress.png"
                    fig.savefig(output_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    pass
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path
        csv_path = "/workspace/log/mma_training_v1_202512052/results.csv"
    
    if len(sys.argv) > 2:
        refresh_interval = int(sys.argv[2])
    else:
        refresh_interval = 5
    
    visualize_training(csv_path, refresh_interval)
