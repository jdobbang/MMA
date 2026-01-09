#!/usr/bin/env python3
"""
SMPL utilities for loading and handling SMPL data.

Provides functions for:
- Loading SMPL mesh data from .npy files
- Loading 2D pose data
- Getting SMPL face topology
"""

import numpy as np
from pathlib import Path


def get_smpl_faces():
    """
    Get SMPL face indices (13776, 3).

    Returns:
        np.ndarray: Face indices array of shape (13776, 3) with dtype uint32
    """
    # SMPL standard face topology from official SMPL model
    # These are the pre-computed face indices for the standard SMPL mesh
    # with 6890 vertices. This is a fixed topology used across all SMPL instances.

    # Load from file if available, otherwise use built-in
    smpl_faces_path = Path(__file__).parent / 'smpl_faces.npy'

    if smpl_faces_path.exists():
        return np.load(smpl_faces_path).astype(np.uint32)

    # Fall back to creating/downloading if we need to
    # For now, we'll create a placeholder and note that SMPL faces are standard
    try:
        # Try to get from downloaded SMPL model
        import pickle
        # This would require having the SMPL model file
        # For now, generate using standard SMPL topology
        faces = _get_smpl_faces_standard()
        return faces.astype(np.uint32)
    except Exception as e:
        print(f"Warning: Could not load SMPL faces: {e}")
        return _get_smpl_faces_standard()


def _get_smpl_faces_standard():
    """
    Returns standard SMPL face topology.

    These are the standard 13776 triangular faces for the SMPL body model
    with 6890 vertices. Extracted from the official SMPL model.

    Returns:
        np.ndarray: Face indices array of shape (13776, 3)
    """
    # Standard SMPL faces - these are the pre-computed face indices
    # This is publicly available from the SMPL model definition
    import urllib.request
    import json
    from pathlib import Path

    # Try to download standard SMPL faces
    try:
        # Create a temporary file to store faces
        temp_path = Path('/tmp/smpl_faces_temp.npy')

        # The standard SMPL faces can be obtained from various sources
        # For simplicity, we'll use a known conversion from SMPL topology
        # Ref: https://github.com/vchoutas/smplify-x/blob/master/smplx/body_models.py

        # Instead of downloading, we use the standard topology
        # This is based on the official SMPL body model structure
        faces = _create_standard_faces()

        return faces
    except Exception as e:
        print(f"Error in _get_smpl_faces_standard: {e}")
        return _create_standard_faces()


def _create_standard_faces():
    """
    Create standard SMPL face topology programmatically.

    Returns:
        np.ndarray: Standard SMPL faces (13776, 3)
    """
    # Try multiple sources to get SMPL faces

    # 1. Try loading from cached files
    try:
        import os
        cache_paths = [
            '/workspace/MMA/smpl_model_faces.npy',
            '/workspace/MMA/script/smpl_faces.npy',
            'smpl_model_faces.npy',
            'smpl_faces.npy',
            os.path.expanduser('~/.cache/smpl_faces.npy')
        ]

        for cache_path in cache_paths:
            if os.path.exists(cache_path):
                print(f"Loading SMPL faces from cache: {cache_path}")
                return np.load(cache_path).astype(np.uint32)
    except Exception as e:
        print(f"Cache loading failed: {e}")

    # 2. Try smplx package (recommended)
    try:
        print("Attempting to load SMPL faces using smplx library...")
        import smplx
        # Try to access SMPL model data
        # smplx has SMPL faces built-in that we can use
        smpl_layer = smplx.SMPL(model_path=None)  # Try without path
        faces = smpl_layer.faces
        print(f"✓ Successfully loaded SMPL faces from smplx: shape {faces.shape}")
        return faces.astype(np.uint32)
    except Exception as e:
        print(f"smplx method failed: {e}")

    # 3. Try loading SMPL faces from official model file if available
    try:
        print("Checking for official SMPL model...")
        import pickle
        import os

        # Look for SMPL model in common locations
        possible_paths = [
            'models/smpl/SMPL_NEUTRAL.pkl',
            os.path.expanduser('~/Documents/SMPL/SMPL_NEUTRAL.pkl'),
            '/data/SMPL/SMPL_NEUTRAL.pkl',
        ]

        for model_path in possible_paths:
            if os.path.exists(model_path):
                print(f"Found SMPL model at: {model_path}")
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f, encoding='latin1')
                faces = model_data['f']
                print(f"✓ Loaded SMPL faces from model: shape {faces.shape}")
                return faces.astype(np.uint32)
    except Exception as e:
        print(f"Official model loading failed: {e}")

    # 4. Try using trimesh's built-in body model (fallback)
    try:
        print("Attempting to create faces using trimesh...")
        import trimesh
        # Create a simple body mesh using trimesh primitives
        # This won't be perfect SMPL topology but can work for testing
        # For proper SMPL, we need the actual faces

        # As a fallback for testing, we could create a simplified mesh
        # But this is not recommended for production

        print("Warning: Trimesh fallback would create incorrect topology")
        raise RuntimeError("Need actual SMPL faces for proper rendering")

    except Exception as e:
        print(f"Trimesh fallback failed: {e}")

    # 5. As a last resort, raise informative error
    error_msg = (
        "\n"
        "="*70 + "\n"
        "ERROR: Could not load SMPL face topology\n"
        "="*70 + "\n"
        "\nSMPL face topology is required for 3D rendering.\n"
        "The standard SMPL model has 13776 triangular faces for 6890 vertices.\n"
        "\nTo fix this, do ONE of the following:\n"
        "\n1. [RECOMMENDED] Use smplx package:\n"
        "   pip install smplx\n"
        "\n2. Download SMPL model and extract faces:\n"
        "   - Go to https://smpl.is.tue.mpg.de/\n"
        "   - Download SMPL_NEUTRAL.pkl (requires free registration)\n"
        "   - Save the file to: /workspace/MMA/models/smpl/SMPL_NEUTRAL.pkl\n"
        "   - Run: python -c \"import pickle; "
        "f=pickle.load(open('models/smpl/SMPL_NEUTRAL.pkl','rb'),encoding='latin1'); "
        "import numpy as np; np.save('script/smpl_faces.npy', f['f'])\"\n"
        "\n3. Download pre-extracted faces from GitHub:\n"
        "   - https://github.com/vchoutas/smplify-x\n"
        "   - Download faces and save as: /workspace/MMA/script/smpl_faces.npy\n"
        "\n4. Use alternative body model (SMPL-X, which has built-in faces)\n"
        "\n" + "="*70
    )

    raise RuntimeError(error_msg)


def load_smpl_data(smpl_path, frame_id):
    """
    Load SMPL data for a specific frame.

    Args:
        smpl_path (str or Path): Path to processed_data/smpl directory
        frame_id (str): Frame ID (e.g., '00001')

    Returns:
        dict: Dictionary with keys ['aria01', 'aria02'], each containing:
            - vertices: (6890, 3) array
            - joints: (45, 3) array
            - global_orient: (3,) array
            - transl: (3,) array
            - body_pose: (69,) array
            - betas: (10,) array
            - epoch_loss: float
    """
    smpl_path = Path(smpl_path)
    frame_file = smpl_path / f"{frame_id}.npy"

    if not frame_file.exists():
        raise FileNotFoundError(f"SMPL data not found: {frame_file}")

    # Load with allow_pickle=True to handle object arrays
    data = np.load(frame_file, allow_pickle=True).item()

    return data


def load_pose2d_data(pose2d_path, frame_id):
    """
    Load 2D pose data for a specific frame.

    Args:
        pose2d_path (str or Path): Path to processed_data/poses2d/cam01 directory
        frame_id (str): Frame ID (e.g., '00001')

    Returns:
        dict: Dictionary with keys ['aria01', 'aria02'], each containing:
            - np.ndarray of shape (45, 2) with 2D keypoint coordinates (x, y)
    """
    pose2d_path = Path(pose2d_path)
    frame_file = pose2d_path / f"{frame_id}.npy"

    if not frame_file.exists():
        raise FileNotFoundError(f"Pose2D data not found: {frame_file}")

    data = np.load(frame_file, allow_pickle=True).item()

    return data


def create_smpl_faces_cache():
    """
    Create a cache file for SMPL faces if not present.

    This should be called during environment setup to ensure faces are available.
    """
    import os

    cache_path = '/workspace/MMA/script/smpl_model_faces.npy'

    if os.path.exists(cache_path):
        print(f"SMPL faces cache already exists at {cache_path}")
        return True

    print("Attempting to create SMPL faces cache...")

    try:
        # Try using smplx
        import smplx
        print("Loading SMPL model using smplx...")
        smpl_model = smplx.SMPL(model_path='models/smpl/SMPL_NEUTRAL.pkl',
                                batch_size=1)
        faces = smpl_model.faces
        np.save(cache_path, faces)
        print(f"SMPL faces cached to {cache_path}")
        return True
    except Exception as e:
        print(f"Failed with smplx: {e}")

    print("\nTo use SMPL rendering, you need SMPL face topology.")
    print("Please do one of the following:")
    print("1. Download SMPL model: https://smpl.is.tue.mpg.de/")
    print("   Place SMPL_NEUTRAL.pkl in the 'models/smpl/' directory")
    print("2. Install smplx: pip install smplx")
    print("3. Download pre-extracted faces from: https://github.com/vchoutas/smplify-x")

    return False


if __name__ == '__main__':
    # Test loading utilities
    import sys

    print("Testing SMPL utilities...")

    # Check if we can access test data
    test_smpl_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/smpl')
    test_pose2d_path = Path('/workspace/MMA/dataset/03_grappling2/014_grappling2/processed_data/poses2d/cam01')

    if test_smpl_path.exists():
        print(f"✓ Found SMPL data at {test_smpl_path}")
        try:
            smpl_data = load_smpl_data(test_smpl_path, '00001')
            print(f"✓ Loaded SMPL data for frame 00001")
            print(f"  - aria01 vertices shape: {smpl_data['aria01']['vertices'].shape}")
            print(f"  - aria01 joints shape: {smpl_data['aria01']['joints'].shape}")
        except Exception as e:
            print(f"✗ Error loading SMPL data: {e}")
            sys.exit(1)

    if test_pose2d_path.exists():
        print(f"✓ Found Pose2D data at {test_pose2d_path}")
        try:
            pose2d_data = load_pose2d_data(test_pose2d_path, '00001')
            print(f"✓ Loaded Pose2D data for frame 00001")
            print(f"  - aria01 shape: {pose2d_data['aria01'].shape}")
        except Exception as e:
            print(f"✗ Error loading Pose2D data: {e}")
            sys.exit(1)

    # Try to get SMPL faces
    print("\nAttempting to load SMPL faces...")
    try:
        faces = get_smpl_faces()
        print(f"✓ Successfully loaded SMPL faces with shape {faces.shape}")
    except RuntimeError as e:
        print(f"✗ Could not load SMPL faces")
        print(f"  {e}")
        print("\n  Next steps:")
        print("  1. Download SMPL model from https://smpl.is.tue.mpg.de/")
        print("  2. Extract faces and save to /workspace/MMA/smpl_model_faces.npy")
        sys.exit(1)

    print("\n✓ All SMPL utilities working correctly!")
