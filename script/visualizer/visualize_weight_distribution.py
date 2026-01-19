#!/usr/bin/env python3
"""
Script to visualize weight distribution analysis

Usage:
    python visualize_weight_distribution.py <weight_dist_csv> [--output-dir OUTPUT_DIR]

Example:
    python visualize_weight_distribution.py \
        results/tracking_results/03_grappling2_001_grappling2_cam01/pose_estimation_vitpose.weight_dist.csv \
        --output-dir results/analysis/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
from pathlib import Path
import argparse


def plot_weight_time_series(weight_df: pd.DataFrame, output_dir: Path):
    """Plot weight distribution over time"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for track_id in weight_df['track_id'].unique():
        track_data = weight_df[weight_df['track_id'] == track_id]
        label = f"Player {track_id}"
        linestyle = '-' if track_id == 1 else '--'

        # Front/Rear weight
        ax = axes[0]
        ax.plot(track_data['frame'], track_data['front_weight'],
                label=f"{label} (front)", linestyle=linestyle, alpha=0.7)
        ax.fill_between(track_data['frame'], 0, track_data['front_weight'],
                        alpha=0.1, label=f"{label} (front area)")

        # Left/Right weight
        ax = axes[1]
        ax.axhline(y=0.5, color='k', linestyle=':', alpha=0.3)
        ax.plot(track_data['frame'], track_data['left_weight'],
                label=f"{label} (left)", linestyle=linestyle, alpha=0.7)

        # Stability score
        ax = axes[2]
        ax.plot(track_data['frame'], track_data['stability_score'],
                label=label, linestyle=linestyle, alpha=0.7)

    # Formatting
    axes[0].set_ylabel('Front Weight', fontsize=11)
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Weight Distribution Over Time', fontsize=12, fontweight='bold')

    axes[1].set_ylabel('Left Weight', fontsize=11)
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel('Stability Score', fontsize=11)
    axes[2].set_xlabel('Frame', fontsize=11)
    axes[2].set_ylim([0, 1])
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'weight_distribution_timeseries.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_weight_scatter(weight_df: pd.DataFrame, output_dir: Path):
    """Plot front/rear vs left/right weight distribution"""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = {1: 'red', 2: 'blue'}
    for track_id in weight_df['track_id'].unique():
        track_data = weight_df[weight_df['track_id'] == track_id]
        color = colors.get(track_id, 'gray')
        ax.scatter(track_data['left_weight'], track_data['front_weight'],
                   label=f"Player {track_id}", color=color, alpha=0.6, s=30)

    # Add quadrant lines
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0.5, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Add quadrant labels
    ax.text(0.25, 0.75, 'Front-Left\n(aggressive left)', ha='center', va='center',
            fontsize=10, alpha=0.5, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.text(0.75, 0.75, 'Front-Right\n(aggressive right)', ha='center', va='center',
            fontsize=10, alpha=0.5, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(0.25, 0.25, 'Rear-Left\n(defensive left)', ha='center', va='center',
            fontsize=10, alpha=0.5, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.75, 0.25, 'Rear-Right\n(defensive right)', ha='center', va='center',
            fontsize=10, alpha=0.5, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    ax.set_xlabel('Left ←→ Right Weight', fontsize=12)
    ax.set_ylabel('Front ↑ ↓ Rear Weight', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('Weight Distribution 2D Map', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'weight_distribution_2d_map.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_stability_comparison(weight_df: pd.DataFrame, output_dir: Path):
    """Plot stability score comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Stability over time (histogram)
    for track_id in weight_df['track_id'].unique():
        track_data = weight_df[weight_df['track_id'] == track_id]
        ax = axes[0]
        ax.hist(track_data['stability_score'], bins=20, alpha=0.6,
                label=f"Player {track_id}")

    axes[0].set_xlabel('Stability Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_title('Stability Score Distribution', fontsize=12, fontweight='bold')

    # Stability statistics
    ax = axes[1]
    track_ids = sorted(weight_df['track_id'].unique())
    stability_means = [weight_df[weight_df['track_id'] == tid]['stability_score'].mean()
                       for tid in track_ids]
    stability_stds = [weight_df[weight_df['track_id'] == tid]['stability_score'].std()
                      for tid in track_ids]

    x_pos = np.arange(len(track_ids))
    ax.bar(x_pos, stability_means, yerr=stability_stds, capsize=5,
           color=['red', 'blue'][:len(track_ids)], alpha=0.7)
    ax.set_ylabel('Mean Stability Score', fontsize=11)
    ax.set_xlabel('Player', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Player {tid}" for tid in track_ids])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Mean Stability Score by Player', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'stability_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_statistics(weight_df: pd.DataFrame):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("WEIGHT DISTRIBUTION STATISTICS")
    print("="*60)

    for track_id in sorted(weight_df['track_id'].unique()):
        track_data = weight_df[weight_df['track_id'] == track_id]
        print(f"\n--- Player {track_id} ---")

        print(f"Front/Rear Weight:")
        print(f"  Front: {track_data['front_weight'].mean():.3f} ± {track_data['front_weight'].std():.3f}")
        print(f"  Rear:  {track_data['rear_weight'].mean():.3f} ± {track_data['rear_weight'].std():.3f}")

        print(f"Left/Right Weight:")
        print(f"  Left:  {track_data['left_weight'].mean():.3f} ± {track_data['left_weight'].std():.3f}")
        print(f"  Right: {track_data['right_weight'].mean():.3f} ± {track_data['right_weight'].std():.3f}")

        print(f"Forward Lean:")
        print(f"  Mean: {track_data['forward_lean'].mean():.3f} ± {track_data['forward_lean'].std():.3f}")

        print(f"Stability Score:")
        print(f"  Mean: {track_data['stability_score'].mean():.3f} ± {track_data['stability_score'].std():.3f}")
        print(f"  Min:  {track_data['stability_score'].min():.3f}")
        print(f"  Max:  {track_data['stability_score'].max():.3f}")

        print(f"Valid frames: {track_data['is_valid'].sum()}/{len(track_data)}")


def main(weight_dist_csv: str, output_dir: str = None):
    """Main function"""
    weight_dist_csv = Path(weight_dist_csv)

    if output_dir is None:
        output_dir = weight_dist_csv.parent / 'analysis'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading weight distribution from: {weight_dist_csv}")
    weight_df = pd.read_csv(weight_dist_csv)

    print(f"Processing {len(weight_df)} samples from {weight_df['track_id'].nunique()} tracks...")

    # Print statistics
    print_statistics(weight_df)

    # Generate visualizations
    print(f"\nGenerating visualizations in: {output_dir}")
    plot_weight_time_series(weight_df, output_dir)
    plot_weight_scatter(weight_df, output_dir)
    plot_stability_comparison(weight_df, output_dir)

    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize weight distribution analysis')
    parser.add_argument('weight_dist_csv', help='Path to weight distribution CSV')
    parser.add_argument('--output-dir', help='Output directory for visualizations')

    args = parser.parse_args()
    main(args.weight_dist_csv, args.output_dir)
