# SMPL Faces Setup Guide

The SMPL rendering pipeline requires the SMPL face topology (13776 triangular faces for 6890 vertices). This guide explains how to set it up.

## Quick Solution: Download Pre-extracted Faces

The easiest solution is to download pre-extracted SMPL faces from a public repository:

```bash
# Download SMPL faces from SMPL-X repository
wget -O /workspace/MMA/script/smpl_faces.npy \
  https://github.com/vchoutas/smplify-x/raw/master/smpl_faces.npy

# Or manually:
# 1. Visit: https://github.com/vchoutas/smplify-x
# 2. Download the smpl_faces.npy file
# 3. Save to: /workspace/MMA/script/smpl_faces.npy
```

## Option 1: Official SMPL Model (Recommended for accuracy)

1. **Download SMPL Model**:
   - Go to https://smpl.is.tue.mpg.de/register.php
   - Create a free account (takes ~1 minute)
   - Download "SMPL Python v.1.0.0"
   - Extract the ZIP file

2. **Extract Faces**:
   ```bash
   python3 << 'EOF'
   import pickle
   import numpy as np
   from pathlib import Path

   # Load SMPL model (adjust path as needed)
   smpl_model_path = 'path/to/SMPL_python_v.1.0.0/smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'

   with open(smpl_model_path, 'rb') as f:
       model_data = pickle.load(f, encoding='latin1')

   # Extract faces
   faces = model_data['f']  # Shape: (13776, 3)

   # Save to script directory
   output_path = Path('/workspace/MMA/script/smpl_faces.npy')
   np.save(output_path, faces)

   print(f"✓ Saved SMPL faces to {output_path}")
   print(f"  Shape: {faces.shape}")
   EOF
   ```

## Option 2: SMPL-X Library (Easiest if already installed)

If you have `smplx` installed and have downloaded the SMPL model:

```bash
python3 << 'EOF'
import numpy as np
from pathlib import Path
import smplx

# Create SMPL model instance (requires model_path if not using default)
model_path = 'path/to/smpl/models/SMPL_NEUTRAL.pkl'
smpl_model = smplx.SMPL(model_path=model_path)

# Get faces
faces = smpl_model.faces

# Save
output_path = Path('/workspace/MMA/script/smpl_faces.npy')
np.save(output_path, faces)

print(f"✓ Saved SMPL faces: {faces.shape}")
EOF
```

## Option 3: From Git Repository

Some GitHub repositories have extracted SMPL faces available:

```bash
# Option A: From SMPL-X repository
git clone https://github.com/vchoutas/smplify-x.git
cp smplify-x/smpl_faces.npy /workspace/MMA/script/

# Option B: From other SMPL implementations
# Search GitHub for "SMPL faces numpy"
```

## Verify Installation

After saving the SMPL faces file:

```bash
python3 << 'EOF'
import numpy as np
from pathlib import Path

faces_path = Path('/workspace/MMA/script/smpl_faces.npy')

if faces_path.exists():
    faces = np.load(faces_path)
    print(f"✓ SMPL faces loaded successfully!")
    print(f"  Path: {faces_path}")
    print(f"  Shape: {faces.shape}")
    print(f"  Dtype: {faces.dtype}")

    # Verify shape
    if faces.shape == (13776, 3):
        print("✓ Shape is correct for SMPL!")
    else:
        print(f"✗ Warning: Unexpected shape {faces.shape}, expected (13776, 3)")
else:
    print(f"✗ SMPL faces file not found at {faces_path}")
EOF
```

## After Setting Up SMPL Faces

Once you have saved the SMPL faces file, run the rendering pipeline:

```bash
python3 script/test_smpl_rendering.py \
  --sequence 03_grappling2/014_grappling2 \
  --frames 00001,00002 \
  --camera cam01 \
  --verbose
```

## Troubleshooting

### "Could not load SMPL face topology" Error

This means the faces file doesn't exist or can't be loaded. Check:

1. File exists at `/workspace/MMA/script/smpl_faces.npy`
2. File size is reasonable (~1.6 MB for 13776 faces)
3. File is valid numpy format

### "Permission denied" when downloading

Use `sudo` or download manually:

```bash
# Manual download in browser:
# 1. Visit repository in GitHub
# 2. Click "Download raw file"
# 3. Save to /workspace/MMA/script/smpl_faces.npy
```

## What Are SMPL Faces?

SMPL (Skinned Multi-Person Linear Model) is a parametric 3D human body model with:
- **6890 vertices** - 3D positions that form the body shape
- **13776 faces** - triangles that connect vertices to form the mesh surface

The faces are a fixed topology that doesn't change between SMPL instances - they're the same for all humans modeled with SMPL.

## References

- SMPL Official Website: https://smpl.is.tue.mpg.de/
- SMPL Paper: "SMPL: A Skinned Multi-Person Linear Model" (Loper et al., 2015)
- SMPL-X Repository: https://github.com/vchoutas/smplx/
- Smplify-X Repository: https://github.com/vchoutas/smplify-x/
