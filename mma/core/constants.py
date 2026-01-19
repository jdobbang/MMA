"""
Global constants for MMA pipeline

Includes:
- COCO keypoint definitions
- SMPL model information
- Keypoint mappings
- Color palettes
"""

# ===== COCO Keypoints (17 points) =====
COCO_KEYPOINT_NAMES = [
    'nose',             # 0
    'left_eye',         # 1
    'right_eye',        # 2
    'left_ear',         # 3
    'right_ear',        # 4
    'left_shoulder',    # 5
    'right_shoulder',   # 6
    'left_elbow',       # 7
    'right_elbow',      # 8
    'left_wrist',       # 9
    'right_wrist',      # 10
    'left_hip',         # 11
    'right_hip',        # 12
    'left_knee',        # 13
    'right_knee',       # 14
    'left_ankle',       # 15
    'right_ankle',      # 16
]

# COCO Skeleton - connections between keypoints for visualization
COCO_SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Lower body
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# COCO Keypoint groups for visualization
COCO_KEYPOINT_GROUPS = {
    'face': [0, 1, 2, 3, 4],
    'upper_body': [5, 6, 7, 8, 9, 10],
    'torso': [5, 6, 11, 12],
    'lower_body': [11, 12, 13, 14, 15, 16],
}

# ===== SMPL Model Information =====
SMPL_NUM_VERTICES = 6890
SMPL_NUM_FACES = 13776
SMPL_NUM_JOINTS = 45
SMPL_NUM_SHAPE_PARAMS = 10
SMPL_NUM_POSE_PARAMS = 72  # 24 joints * 3 (axis-angle)

# ===== Keypoint Mappings =====
# Mapping from Poses2D (45 keypoints) to COCO (17 keypoints)
# These indices map from SMPL-X joint indices to COCO keypoint indices
POSES2D_TO_COCO_MAPPING = {
    0: 15,   # SMPL nose -> COCO nose
    1: 26,   # SMPL left_eye -> COCO left_eye
    2: 25,   # SMPL right_eye -> COCO right_eye
    3: 27,   # SMPL left_ear -> COCO left_ear
    4: 28,   # SMPL right_ear -> COCO right_ear
    5: 16,   # SMPL left_shoulder -> COCO left_shoulder
    6: 17,   # SMPL right_shoulder -> COCO right_shoulder
    7: 18,   # SMPL left_elbow -> COCO left_elbow
    8: 19,   # SMPL right_elbow -> COCO right_elbow
    9: 20,   # SMPL left_wrist -> COCO left_wrist
    10: 21,  # SMPL right_wrist -> COCO right_wrist
    11: 1,   # SMPL left_hip -> COCO left_hip
    12: 2,   # SMPL right_hip -> COCO right_hip
    13: 4,   # SMPL left_knee -> COCO left_knee
    14: 5,   # SMPL right_knee -> COCO right_knee
    15: 7,   # SMPL left_ankle -> COCO left_ankle
    16: 8,   # SMPL right_ankle -> COCO right_ankle
}

# Reverse mapping from COCO to SMPL-X
COCO_TO_POSES2D_MAPPING = {v: k for k, v in POSES2D_TO_COCO_MAPPING.items()}

# ===== Color Palettes =====
# BGR format for OpenCV

# Default player colors for MMA (2 players)
DEFAULT_PLAYER_COLORS = {
    1: (0, 0, 255),      # Player 1: Red (BGR)
    2: (255, 0, 0),      # Player 2: Blue (BGR)
}

# Extended color palette for up to 20 tracks
TRACK_COLORS = [
    (0, 0, 255),         # Red
    (255, 0, 0),         # Blue
    (0, 255, 0),         # Green
    (255, 255, 0),       # Cyan
    (255, 0, 255),       # Magenta
    (0, 255, 255),       # Yellow
    (128, 0, 0),         # Dark Red
    (0, 128, 0),         # Dark Green
    (0, 0, 128),         # Dark Blue
    (128, 128, 0),       # Dark Cyan
    (128, 0, 128),       # Dark Magenta
    (0, 128, 128),       # Dark Yellow
    (192, 192, 192),     # Light Gray
    (128, 128, 128),     # Gray
    (255, 128, 0),       # Orange
    (0, 255, 128),       # Spring Green
    (128, 255, 0),       # Chartreuse
    (128, 0, 255),       # Violet
    (255, 0, 128),       # Rose
    (0, 128, 255),       # Sky Blue
]

# Keypoint colors for skeleton visualization
KEYPOINT_COLORS = {
    'face': (0, 255, 255),           # Yellow
    'upper_body': (0, 165, 255),     # Orange
    'torso': (255, 0, 0),            # Blue
    'lower_body': (0, 255, 0),       # Green
}

# ===== Arena Specifications =====
# Standard dimensions for grappling and MMA

ARENA_SPECS = {
    'grappling': {
        'name': 'Grappling Mat',
        'width': 3.5,      # meters
        'height': 3.0,     # meters
        'area': 10.5,      # square meters
    },
    'mma': {
        'name': 'MMA Octagon',
        'width': 3.0,      # meters (approximate, reduced for inside)
        'height': 3.0,     # meters
        'area': 9.0,       # square meters
    },
}

# ===== Calibration Defaults =====
# Default camera parameters for initial setup

DEFAULT_CAMERA_INTRINSICS = {
    'fx': 1000.0,
    'fy': 1000.0,
    'cx': 320.0,
    'cy': 240.0,
    'width': 640,
    'height': 480,
}

# ===== Model Paths (relative to asset directory) =====
MODEL_PATHS = {
    'smpl': 'model/SMPL_NEUTRAL.pkl',
    'smpl_x': 'model/SMPLX_NEUTRAL.npz',
    'mano_left': 'model/mano_left.pkl',
    'mano_right': 'model/mano_right.pkl',
}

# ===== Validation Thresholds =====
VALID_CONF_THRESHOLD_RANGE = (0.0, 1.0)
VALID_IOU_THRESHOLD_RANGE = (0.0, 1.0)
MIN_BBOX_SIZE = 10  # pixels
MIN_KEYPOINT_CONFIDENCE = 0.1

# ===== String Constants =====
STAGE_NAMES = ['03_grappling2', '13_mma2']
CAMERA_NAMES = ['aria01', 'aria02']

# CSV column names
CSV_DETECTION_COLUMNS = [
    'image_name', 'x1', 'y1', 'x2', 'y2', 'confidence'
]

CSV_TRACKING_COLUMNS = [
    'image_name', 'frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'
]

CSV_POSE_COLUMNS = [
    'image_name', 'frame', 'track_id',
    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox_conf'
] + [f'{kpt}_{coord}' for kpt in COCO_KEYPOINT_NAMES for coord in ['x', 'y', 'conf']]

# ===== Numeric Constants =====
PI = 3.14159265359
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# ===== Dataset Split Ratios =====
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
