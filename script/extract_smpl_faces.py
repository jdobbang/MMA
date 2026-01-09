import sys
import numpy as np

# --- 추가된 부분: NumPy 1.24+ 호환성 패치 ---
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'unicode'):
    np.unicode = str
# ---------------------------------------

import pickle
from pathlib import Path

# 다운로드한 파일 경로
model_path = '/workspace/MMA/model/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'

print(f"Loading SMPL model from: {model_path}")

with open(model_path, 'rb') as f:
    # chumpy 객체를 로드할 때 위에서 선언한 np.bool 등을 참조하게 됩니다.
    model = pickle.load(f, encoding='latin1')

# faces 추출 (chumpy 객체인 경우 .asarray() 또는 np.array()로 변환이 필요할 수 있음)
faces = np.array(model['f'], dtype=np.int32)

print(f"✓ Extracted SMPL faces: {faces.shape}")
print(f"  Data type: {faces.dtype}")

# 저장 경로 설정
output_path = Path('/workspace/MMA/script/smpl_faces.npy')
output_path.parent.mkdir(parents=True, exist_ok=True)
np.save(output_path, faces)

print(f"✓ Saved to: {output_path}")

# 확인
loaded = np.load(output_path)
print(f"✓ Verified: {loaded.shape}")