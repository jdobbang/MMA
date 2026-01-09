# SMPL 3D Rendering & Visibility Testing Pipeline

## Overview

This pipeline renders 3D SMPL body meshes onto 2D images and tests joint visibility by comparing depth values. It's designed to determine which 2D keypoints are visible (not occluded) based on 3D mesh occlusion.

### What It Does

1. **Camera Calibration**: Estimates camera parameters (intrinsics & extrinsics) from 3D SMPL joints and 2D pose keypoints using PnP (Perspective-n-Point) algorithm
2. **3D Rendering**: Renders SMPL mesh onto 2D images using pyrender with proper depth buffering
3. **Visibility Testing**: Compares rendered depth with expected joint depth to determine visibility
4. **Visualization**: Creates multi-panel visualization showing original image, rendered mesh, and visibility overlay

### Key Outputs

- **Visualization images**: 3-panel view of original, rendered mesh, and visibility overlay
- **JSON results**: Joint-by-joint visibility and confidence scores
- **Depth maps**: 3D depth information for analysis

## Quick Start

### 1. Set Up SMPL Face Topology (Required)

The pipeline requires SMPL face topology (13776 triangular faces for 6890 vertices).

**Option A: Download from official SMPL (Recommended)**

```bash
# Visit https://smpl.is.tue.mpg.de/ and download "SMPL Python v.1.0.0"
# Extract and run:

python3 << 'EOF'
import pickle
import numpy as np
from pathlib import Path

model_path = 'path/to/SMPL_python_v.1.0.0/smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f, encoding='latin1')

faces = model['f']  # Shape: (13776, 3)

output_path = Path('/workspace/MMA/script/smpl_faces.npy')
np.save(output_path, faces)

print(f"✓ Saved SMPL faces to {output_path}")
print(f"  Shape: {faces.shape}")
EOF
```

**Option B: Use smplx if you have downloaded SMPL model**

```bash
python3 << 'EOF'
import numpy as np
from pathlib import Path
import smplx

model_path = 'path/to/SMPL_NEUTRAL.pkl'
smpl_model = smplx.SMPL(model_path=model_path)
faces = smpl_model.faces

output_path = Path('/workspace/MMA/script/smpl_faces.npy')
np.save(output_path, faces)
EOF
```

**Option C: See `SMPL_FACES_SETUP.md` for more options**

### 2. Run the Pipeline

```bash
# Single frame test
python3 script/test_smpl_rendering.py \
  --sequence 03_grappling2/014_grappling2 \
  --frames 00001 \
  --camera cam01

# Multiple frames with verbose output
python3 script/test_smpl_rendering.py \
  --sequence 03_grappling2/014_grappling2 \
  --frames 00001,00002 \
  --camera cam01 \
  --verbose

# Custom output directory
python3 script/test_smpl_rendering.py \
  --sequence 13_mma2/025_mma2 \
  --frames 00001 \
  --camera cam01 \
  --output_dir my_results \
  --verbose
```

### 3. Check Results

Results are saved to: `smpl_rendering_results/<sequence>/`

```
smpl_rendering_results/03_grappling2/014_grappling2/
├── vis_frame_00001.jpg     # Multi-panel visualization
├── vis_frame_00002.jpg
├── visibility_00001.json   # Visibility results
└── visibility_00002.json
```

## Module Structure

### Core Modules

**`smpl_utils.py`**: SMPL data loading
- `load_smpl_data()`: Load SMPL vertices and joints from .npy file
- `load_pose2d_data()`: Load 2D pose keypoints
- `get_smpl_faces()`: Get SMPL face topology

**`camera_calibration.py`**: Camera parameter estimation
- `estimate_camera_pnp()`: Estimate camera intrinsics and extrinsics using PnP+RANSAC
- `project_3d_to_2d()`: Project 3D points to 2D image plane
- `estimate_depth()`: Compute depth of 3D point in camera coordinates

**`render_smpl.py`**: 3D mesh rendering
- `render_smpl_mesh()`: Render SMPL mesh with depth buffer using pyrender
- `composite_depth_maps()`: Combine depth maps from multiple persons

**`visibility_test.py`**: Visibility determination
- `test_keypoint_visibility()`: Test if joints are visible based on depth comparison
- `create_visibility_array()`: Format visibility results as structured output

**`visualize_smpl_rendering.py`**: Visualization
- `create_visualization()`: Create multi-panel visualization
- `save_visualization()`: Save visualization image

**`test_smpl_rendering.py`**: Main pipeline
- Orchestrates the complete workflow
- Command-line interface for easy use

## Data Flow

```
1. Load Data
   ├── SMPL vertices & joints (3D)
   ├── 2D pose keypoints
   └── Original image

2. Camera Calibration (per person)
   ├── Input: 3D joints + 2D keypoints
   ├── Method: PnP with RANSAC
   └── Output: K (intrinsics), R (rotation), t (translation)

3. Render Meshes (per person)
   ├── Input: SMPL vertices + camera parameters
   ├── Method: pyrender off-screen rendering
   └── Output: Color image + depth map

4. Composite Depth
   ├── Input: Depth maps from both persons
   └── Output: Combined depth (occlusion handling)

5. Visibility Testing (per person)
   ├── Input: 2D keypoints + 3D joints + depth map
   ├── Method: Depth comparison
   └── Output: Visibility mask + confidence scores

6. Visualization & Save
   ├── Create 3-panel visualization
   ├── Save image
   └── Save JSON results
```

## Understanding the Output

### Visualization Panels

1. **Original Image**: Input image from dataset
2. **Rendered Mesh**: Pure SMPL mesh rendering (aria01=red, aria02=blue)
3. **Overlay**: Original image with:
   - Semi-transparent SMPL mesh overlay
   - Green circles: visible joints
   - Red X marks: occluded joints
   - White lines: skeleton connections (visible joints only)

### JSON Visibility Results

```json
{
  "aria01": {
    "frame": "00001",
    "person": "aria01",
    "calibration": {
      "reprojection_error": 23.45,
      "intrinsics": [[3840, 0, 1920], [0, 3840, 1080], [0, 0, 1]]
    },
    "joints": [
      {
        "joint_id": 0,
        "visible": true,
        "confidence": 0.98
      },
      ...
    ]
  }
}
```

- **visible**: Boolean, whether joint is visible (not occluded)
- **confidence**: Float 0-1, confidence in visibility classification (based on depth difference)
- **reprojection_error**: RMSE error of camera calibration in pixels (< 50 is good)

## Camera Calibration Details

### PnP Algorithm

The pipeline uses `cv2.solvePnPRansac` to estimate camera parameters:

1. **Input**:
   - 3D SMPL joint positions (45 joints)
   - 2D pose keypoint positions (from detections)

2. **Process**:
   - Initialize camera intrinsics from image dimensions
   - Use RANSAC to robustly estimate rotation (R) and translation (t)
   - Refine using Levenberg-Marquardt on inliers

3. **Output**:
   - Camera intrinsic matrix K (3x3)
   - Rotation matrix R (3x3)
   - Translation vector t (3,)
   - Reprojection error (RMSE in pixels)

### Reprojection Error Interpretation

- **< 20 pixels**: Excellent calibration
- **20-50 pixels**: Good calibration, usable for visibility testing
- **> 50 pixels**: Poor calibration, may produce unreliable visibility results

If error is high:
- Check that 2D keypoints are accurate
- Verify SMPL 3D data quality
- Consider using multiple frames for better calibration

## Visibility Testing Details

### How It Works

For each 2D keypoint:

1. **Compute expected depth**: Project 3D joint to camera coordinates, get z-depth
2. **Sample rendered depth**: Look up depth value at (x, y) pixel in rendered depth map
3. **Compare depths**: If |expected_depth - rendered_depth| < threshold, joint is visible

### Threshold & Confidence

- **Threshold**: 0.1 meters (10 cm) - joint is visible if depths match within 10cm
- **Confidence**: Score based on depth difference
  - 0.0: Complete mismatch (definitely occluded)
  - 1.0: Perfect match (definitely visible)

### Edge Cases

- **Out of bounds**: Keypoint outside image → marked occluded
- **No depth data**: Rendered depth = 0 → marked occluded
- **Multiple overlapping meshes**: Uses closest depth (proper occlusion handling)

## Common Issues & Solutions

### Issue: "Could not load SMPL face topology"

**Solution**: Install SMPL faces (see "Set Up SMPL Face Topology" above)

### Issue: High reprojection error (> 100 pixels)

**Possible causes**:
- Inaccurate 2D pose detections
- Camera view doesn't match SMPL coordinate system
- Scale mismatch between 3D and 2D data

**Solutions**:
- Use better pose estimation model
- Check SMPL and pose2D data alignment
- Visualize reprojected keypoints for debugging

### Issue: Many joints marked as occluded when they should be visible

**Possible causes**:
- Camera calibration poor
- Depth threshold too small
- SMPL mesh doesn't match actual body position

**Solutions**:
- Increase depth threshold in `visibility_test.py`
- Improve camera calibration
- Verify SMPL quality

### Issue: Rendering produces all-black or all-white images

**Causes**:
- OpenGL context issue (headless rendering)
- Camera pose incorrect
- Mesh outside camera frustum

**Solutions**:
```bash
# For headless systems, use EGL:
export PYOPENGL_PLATFORM=egl
python3 script/test_smpl_rendering.py ...

# Or use OSMesa:
export PYOPENGL_PLATFORM=osmesa
```

## Performance Notes

- **Single frame processing**: ~30-60 seconds (depending on camera calibration quality)
- **Main bottleneck**: pyrender rendering (~20-30 seconds per frame)
- **Memory usage**: ~2GB for 4K rendering

## Advanced Usage

### Adjust Visibility Threshold

Edit `visibility_test.py`, function `test_keypoint_visibility()`:

```python
depth_threshold = 0.1  # Change this value (in meters)
```

### Use Different Camera Views

```bash
python3 script/test_smpl_rendering.py \
  --sequence 03_grappling2/014_grappling2 \
  --frames 00001 \
  --camera cam05  # Use different camera
```

### Batch Processing

```bash
for seq in 03_grappling2/*/; do
  seq_name=$(basename "$seq")
  python3 script/test_smpl_rendering.py \
    --sequence "03_grappling2/$seq_name" \
    --frames 00001 \
    --camera cam01
done
```

## Citation

If you use this pipeline in research, please cite:

```bibtex
@article{loper2015smpl,
  title={SMPL: A skinned multi-person linear model},
  author={Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J},
  journal={ACM transactions on graphics (TOG)},
  volume={34},
  number={6},
  pages={1--16},
  year={2015},
  publisher={ACM}
}
```

## References

- SMPL Official: https://smpl.is.tue.mpg.de/
- SMPL-X: https://github.com/vchoutas/smplx
- PyRender Documentation: https://github.com/mmatl/pyrender
- OpenCV solvePnP: https://docs.opencv.org/master/d5/d1f/calib3d_solvepnp.html

## Support

For issues or questions:
1. Check `SMPL_FACES_SETUP.md` for SMPL face installation
2. Review error messages - they provide specific guidance
3. Check camera calibration quality (reprojection error)
4. Verify data alignment between SMPL and 2D poses

## Version History

- v0.1 (2025-01-07): Initial implementation
  - PnP camera calibration
  - SMPL mesh rendering
  - Depth-based visibility testing
  - Multi-frame support

---

**Author**: Claude Code
**Created**: 2025-01-07
**Status**: Beta (requires SMPL face setup)
