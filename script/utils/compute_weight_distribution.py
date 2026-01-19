#!/usr/bin/env python3
"""
Script to compute weight distribution from pose estimation CSV

Usage:
    python compute_weight_distribution.py <input_csv> <output_csv>

Example:
    python compute_weight_distribution.py \
        results/tracking_results/03_grappling2_001_grappling2_cam01/pose_estimation_vitpose.csv \
        results/tracking_results/03_grappling2_001_grappling2_cam01/weight_distribution.csv
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mma.pose.weight_distribution import calculate_weight_from_keypoints_dict


def main(input_csv: str, output_csv: str = None):
    """
    Read pose estimation CSV and compute weight distribution

    Args:
        input_csv: Path to pose estimation CSV
        output_csv: Path to save weight distribution CSV (default: input_csv.weight_dist.csv)
    """
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '.weight_dist.csv')

    print(f"Reading pose data from: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"Processing {len(df)} frames...")

    # Compute weight distribution for each frame
    weight_results = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing frame {idx}/{len(df)}...")

        # Prepare keypoints dict from row
        keypoints = {
            'left_ankle_x': row['left_ankle_x'],
            'left_ankle_y': row['left_ankle_y'],
            'left_ankle_conf': row['left_ankle_conf'],
            'right_ankle_x': row['right_ankle_x'],
            'right_ankle_y': row['right_ankle_y'],
            'right_ankle_conf': row['right_ankle_conf'],
            'left_hip_x': row['left_hip_x'],
            'left_hip_y': row['left_hip_y'],
            'left_hip_conf': row['left_hip_conf'],
            'right_hip_x': row['right_hip_x'],
            'right_hip_y': row['right_hip_y'],
            'right_hip_conf': row['right_hip_conf'],
            'left_knee_x': row['left_knee_x'],
            'left_knee_y': row['left_knee_y'],
            'left_knee_conf': row['left_knee_conf'],
            'right_knee_x': row['right_knee_x'],
            'right_knee_y': row['right_knee_y'],
            'right_knee_conf': row['right_knee_conf'],
        }

        result = calculate_weight_from_keypoints_dict(keypoints)
        weight_results.append(result)

    # Create output dataframe
    weight_df = pd.DataFrame(weight_results)

    # Add frame info from original
    weight_df.insert(0, 'frame', df['frame'])
    weight_df.insert(1, 'track_id', df['track_id'])

    # Save to CSV
    print(f"Saving results to: {output_csv}")
    weight_df.to_csv(output_csv, index=False)

    # Print statistics
    print("\n=== Weight Distribution Statistics ===")
    print(f"\nFront/Rear Weight:")
    print(f"  Front weight: {weight_df['front_weight'].mean():.3f} ± {weight_df['front_weight'].std():.3f}")
    print(f"  Rear weight:  {weight_df['rear_weight'].mean():.3f} ± {weight_df['rear_weight'].std():.3f}")

    print(f"\nLeft/Right Weight:")
    print(f"  Left weight:  {weight_df['left_weight'].mean():.3f} ± {weight_df['left_weight'].std():.3f}")
    print(f"  Right weight: {weight_df['right_weight'].mean():.3f} ± {weight_df['right_weight'].std():.3f}")

    print(f"\nForward Lean:")
    print(f"  Mean: {weight_df['forward_lean'].mean():.3f} ± {weight_df['forward_lean'].std():.3f}")

    print(f"\nStability Score:")
    print(f"  Mean: {weight_df['stability_score'].mean():.3f} ± {weight_df['stability_score'].std():.3f}")

    print(f"\nValid frames: {weight_df['is_valid'].sum()}/{len(weight_df)}")

    print("\n✓ Done!")
    return output_csv


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    main(input_csv, output_csv)
