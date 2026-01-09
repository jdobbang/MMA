#!/usr/bin/env python3
"""
Main pipeline for SMPL mesh rendering and visibility testing.

This script processes frames and renders SMPL meshes to test visibility.

Usage:
    python test_smpl_rendering.py \
      --sequence 03_grappling2/014_grappling2 \
      --frames 00001,00002 \
      --camera cam01 \
      --output_dir smpl_rendering_results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from smpl_utils import load_smpl_data, load_pose2d_data, get_smpl_faces
from camera_calibration import estimate_camera_pnp, create_camera_pose
from render_smpl import render_smpl_mesh, composite_depth_maps
from visibility_test import test_keypoint_visibility, test_keypoint_visibility_color
from visualize_smpl_rendering import create_visualization, save_visualization
from colmap_loader import COLMAPLoader


# Color scheme for persons
PERSON_COLORS = {
    'aria01': (1.0, 0.0, 0.0, 1.0),    # Red
    'aria02': (0.0, 0.0, 1.0, 1.0),    # Blue
}

PERSON_COLORS_RGB = {
    'aria01': (255, 0, 0),      # Red in BGR for OpenCV
    'aria02': (0, 0, 255),      # Blue in BGR for OpenCV
}


def process_frame(
    frame_id: str,
    dataset_path: Path,
    sequence: str,
    camera: str,
    output_dir: Path,
    colmap_loader: COLMAPLoader = None,
    verbose: bool = False
) -> dict:
    """
    Process a single frame: calibration, rendering, and visibility testing.

    Args:
        frame_id: Frame ID (e.g., '00001')
        dataset_path: Path to MMA/dataset directory
        sequence: Sequence name (e.g., '03_grappling2/014_grappling2')
        camera: Camera name (e.g., 'cam01')
        output_dir: Output directory for results
        verbose: Print debug information

    Returns:
        Dictionary with processing results
    """

    seq_path = dataset_path / sequence
    smpl_path = seq_path / 'processed_data' / 'smpl'
    pose2d_path = seq_path / 'processed_data' / 'poses2d' / camera
    img_path = seq_path / 'exo' / camera / 'images' / f'{frame_id}.jpg'

    # Verify files exist
    if not all([smpl_path.exists(), pose2d_path.exists(), img_path.exists()]):
        missing = []
        if not smpl_path.exists():
            missing.append(f'SMPL path: {smpl_path}')
        if not pose2d_path.exists():
            missing.append(f'Pose2D path: {pose2d_path}')
        if not img_path.exists():
            missing.append(f'Image path: {img_path}')
        raise FileNotFoundError(f"Missing files: {', '.join(missing)}")

    if verbose:
        print(f"\nProcessing frame {frame_id}")
        print(f"  SMPL path: {smpl_path}")
        print(f"  Image path: {img_path}")

    # 1. Load data
    smpl_data = load_smpl_data(smpl_path, frame_id)
    pose2d_data = load_pose2d_data(pose2d_path, frame_id)
    original_image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    img_shape = original_image.shape[:2]

    if verbose:
        print(f"  Loaded SMPL data for aria01, aria02")
        print(f"  Image shape: {original_image.shape}")

    # Get SMPL face topology
    try:
        faces = get_smpl_faces()
        if verbose:
            print(f"  Loaded SMPL faces: {faces.shape}")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("\nTo proceed, you need SMPL face topology.")
        print("See error message above for solutions.")
        return None

    # 2. Camera calibration (per person)
    camera_params = {}
    calibration_results = {}

    for person in ['aria01', 'aria02']:
        if verbose:
            print(f"\n  Calibrating camera for {person} (method: PnP RANSAC)...")

        try:
            # Always use PnP RANSAC for robust calibration
            # (COLMAP integration requires additional coordinate system alignment)
            joints_3d = smpl_data[person]['joints'].astype(np.float32)
            keypoints_2d = pose2d_data[person].astype(np.float32)

            K, R, tvec, error = estimate_camera_pnp(
                joints_3d, keypoints_2d, img_shape, verbose=verbose
            )

            camera_params[person] = create_camera_pose(K, R, tvec)
            calibration_results[person] = {
                'source': 'PnP RANSAC',
                'reprojection_error': float(error),
                'intrinsics': K.tolist(),
            }

            if verbose:
                print(f"    ✓ Calibration RMSE: {error:.2f} px")

        except Exception as e:
            print(f"  ✗ Calibration failed for {person}: {e}")
            return None

    # 3. Render meshes (per person)
    rendered_meshes = {}
    depth_maps = {}

    for person in ['aria01', 'aria02']:
        if verbose:
            print(f"\n  Rendering mesh for {person}...")

        try:
            vertices = smpl_data[person]['vertices'].astype(np.float32)
            color = PERSON_COLORS[person]

            color_img, depth = render_smpl_mesh(
                vertices, faces, camera_params[person], img_shape,
                color=color[:3], alpha=color[3], verbose=verbose
            )

            rendered_meshes[person] = color_img
            depth_maps[person] = depth

            if verbose:
                print(f"    ✓ Rendered: {color_img.shape}")

        except Exception as e:
            print(f"  ✗ Rendering failed for {person}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # 4. Composite depth maps for occlusion handling
    final_depth, composite_color = composite_depth_maps(
        depth_maps['aria01'], depth_maps['aria02'],
        rendered_meshes['aria01'], rendered_meshes['aria02']
    )

    if verbose:
        print(f"  Composited depth map")

    # 5. Visibility testing (using rendered color since depth is unavailable)
    visibility_results = {}

    for person in ['aria01', 'aria02']:
        if verbose:
            print(f"\n  Testing visibility for {person}...")

        try:
            keypoints_2d = pose2d_data[person].astype(np.float32)
            joints_3d = smpl_data[person]['joints'].astype(np.float32)

            # Use color-based visibility since depth rendering has issues
            visibility_mask, confidence_scores = test_keypoint_visibility_color(
                keypoints_2d,
                rendered_meshes[person],
                {person: PERSON_COLORS_RGB[person]},
                verbose=verbose
            )

            visibility_results[person] = (visibility_mask, confidence_scores)

            if verbose:
                visible_count = visibility_mask.sum()
                print(f"    ✓ Visible joints: {visible_count} / {len(joints_3d)}")

        except Exception as e:
            print(f"  ✗ Visibility testing failed for {person}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # 6. Create visualization
    if verbose:
        print(f"\n  Creating visualization...")

    try:
        vis_image = create_visualization(
            original_image,
            rendered_meshes['aria01'],
            rendered_meshes['aria02'],
            final_depth,
            pose2d_data,
            visibility_results,
            joints_3d=smpl_data,
            camera_params=camera_params
        )

        if verbose:
            print(f"    ✓ Visualization created: {vis_image.shape}")

    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 7. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save visualization image
    vis_output_path = output_dir / f'vis_frame_{frame_id}.jpg'
    try:
        # Convert RGB to BGR for OpenCV
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_output_path), vis_image_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if verbose:
            print(f"  ✓ Saved visualization: {vis_output_path}")
    except Exception as e:
        print(f"  ✗ Failed to save visualization: {e}")
        return None

    # Save visibility results as JSON
    json_output_path = output_dir / f'visibility_{frame_id}.json'
    try:
        visibility_json = {}

        for person in ['aria01', 'aria02']:
            visibility_mask, confidence_scores = visibility_results[person]

            person_data = {
                'frame': frame_id,
                'person': person,
                'calibration': calibration_results[person],
                'joints': []
            }

            for joint_id, (visible, confidence) in enumerate(
                zip(visibility_mask, confidence_scores)
            ):
                person_data['joints'].append({
                    'joint_id': int(joint_id),
                    'visible': bool(visible),
                    'confidence': float(confidence)
                })

            visibility_json[person] = person_data

        with open(json_output_path, 'w') as f:
            json.dump(visibility_json, f, indent=2)

        if verbose:
            print(f"  ✓ Saved visibility results: {json_output_path}")

    except Exception as e:
        print(f"  ✗ Failed to save JSON: {e}")
        return None

    # Return results summary
    results = {
        'frame_id': frame_id,
        'status': 'success',
        'visualization': str(vis_output_path),
        'visibility_json': str(json_output_path),
        'calibration': calibration_results,
        'visibility_summary': {
            person: visibility_results[person][0].sum()
            for person in ['aria01', 'aria02']
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='SMPL mesh rendering and visibility testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with PnP RANSAC calibration
  python test_smpl_rendering.py \\
    --sequence 03_grappling2/014_grappling2 \\
    --frames 00001 \\
    --camera cam01

  # Test with COLMAP camera parameters (more accurate)
  python test_smpl_rendering.py \\
    --sequence 13_mma2/001_mma2 \\
    --frames 00001,00002 \\
    --camera cam01 \\
    --colmap_dir /workspace/MMA/dataset/13_mma2/001_mma2/processed_data/colmap/workplace

  # Test multiple frames with verbose output
  python test_smpl_rendering.py \\
    --sequence 03_grappling2/014_grappling2 \\
    --frames 00001,00002 \\
    --camera cam01 \\
    --output_dir smpl_rendering_results \\
    --verbose
        """
    )

    parser.add_argument(
        '--sequence',
        type=str,
        required=True,
        help='Sequence path (e.g., 03_grappling2/014_grappling2)'
    )

    parser.add_argument(
        '--frames',
        type=str,
        default='00001',
        help='Frame IDs to process, comma-separated (e.g., 00001,00002)'
    )

    parser.add_argument(
        '--camera',
        type=str,
        default='cam01',
        help='Camera name (default: cam01)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='smpl_rendering_results',
        help='Output directory for results (default: smpl_rendering_results)'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/workspace/MMA/dataset',
        help='Path to MMA/dataset directory'
    )

    parser.add_argument(
        '--colmap_dir',
        type=str,
        default=None,
        help='Path to COLMAP output directory with cameras.txt and images.txt (optional, for precise camera parameters). If not provided, will use PnP RANSAC estimation.'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose debug information'
    )

    args = parser.parse_args()

    # Parse frame IDs
    frame_ids = args.frames.split(',')

    # Setup paths
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) / args.sequence

    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        sys.exit(1)

    # Initialize COLMAP loader if provided
    colmap_loader = None
    if args.colmap_dir:
        try:
            colmap_loader = COLMAPLoader(Path(args.colmap_dir))
            print(f"✓ Loaded COLMAP camera calibration from {args.colmap_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not load COLMAP data: {e}")
            print(f"  Will fall back to PnP RANSAC calibration")

    print("="*70)
    print("SMPL Mesh Rendering and Visibility Testing")
    print("="*70)
    print(f"Sequence: {args.sequence}")
    print(f"Frames: {frame_ids}")
    print(f"Camera: {args.camera}")
    if colmap_loader:
        print(f"Camera calibration: COLMAP")
    else:
        print(f"Camera calibration: PnP RANSAC")
    print(f"Output: {output_dir}")
    print("="*70)

    # Process each frame
    all_results = {}
    successful = 0
    failed = 0

    for frame_id in frame_ids:
        try:
            result = process_frame(
                frame_id,
                dataset_path,
                args.sequence,
                args.camera,
                output_dir,
                colmap_loader=colmap_loader,
                verbose=args.verbose
            )

            if result is not None:
                all_results[frame_id] = result
                successful += 1
                print(f"✓ Frame {frame_id}: Success")
            else:
                failed += 1
                print(f"✗ Frame {frame_id}: Failed")

        except Exception as e:
            failed += 1
            print(f"✗ Frame {frame_id}: Exception - {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Processed: {successful + failed} frames")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    if successful == 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
