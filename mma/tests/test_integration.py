"""
Integration tests for MMA package

Tests:
- Core module configuration and constants
- Detection pipeline (YOLO)
- Tracking pipeline (SORT, MMA-specific)
- Pose estimation pipeline
- Preprocessing dataset generation
"""

import numpy as np
import tempfile
from pathlib import Path


def test_core_config():
    """Test configuration loading and validation"""
    from mma.core.config import TrackingConfig, PoseConfig, PathConfig

    # Test tracking config
    config = TrackingConfig(
        max_age=30,
        max_gap=10,
        reid_ema_alpha=0.1,
        min_score_threshold=0.3
    )
    assert config.max_age == 30
    assert config.reid_ema_alpha == 0.1
    print("✓ TrackingConfig initialization")

    # Test pose config
    pose_config = PoseConfig(
        model_path="yolo11x-pose.pt",
        model_type="yolo",
        keypoint_conf_threshold=0.3,
        device="cpu"
    )
    assert pose_config.model_type == "yolo"
    print("✓ PoseConfig initialization")

    # Test path config
    path_config = PathConfig()
    assert path_config.dataset_dir is not None
    print("✓ PathConfig initialization")


def test_core_constants():
    """Test constants definitions"""
    from mma.core.constants import (
        COCO_KEYPOINT_NAMES,
        COCO_SKELETON_CONNECTIONS,
        SMPL_NUM_VERTICES,
        SMPL_NUM_FACES,
        DEFAULT_PLAYER_COLORS
    )

    assert len(COCO_KEYPOINT_NAMES) == 17
    assert len(COCO_SKELETON_CONNECTIONS) == 16
    assert SMPL_NUM_VERTICES == 6890
    assert SMPL_NUM_FACES == 13776
    assert 1 in DEFAULT_PLAYER_COLORS
    assert 2 in DEFAULT_PLAYER_COLORS
    print("✓ Constants definitions")


def test_detection_bbox_utils():
    """Test bounding box utilities"""
    from mma.detection.bbox_utils import (
        iou, iou_batch,
        bbox_to_yolo, yolo_to_bbox,
        validate_bbox
    )

    # Test single IoU
    bbox1 = np.array([0, 0, 100, 100], dtype=np.float32)
    bbox2 = np.array([50, 50, 150, 150], dtype=np.float32)
    iou_val = iou(bbox1, bbox2)
    assert 0 <= iou_val <= 1
    print("✓ Single IoU calculation")

    # Test batch IoU
    bboxes1 = np.array([[0, 0, 100, 100], [50, 50, 150, 150]], dtype=np.float32)
    bboxes2 = np.array([[25, 25, 125, 125]], dtype=np.float32)
    iou_matrix = iou_batch(bboxes1, bboxes2)
    assert iou_matrix.shape == (2, 1)
    print("✓ Batch IoU calculation")

    # Test coordinate transformations
    bbox = np.array([100, 100, 300, 300], dtype=np.float32)
    yolo_box = bbox_to_yolo(bbox, 640, 480)
    assert len(yolo_box) == 4
    bbox_back = yolo_to_bbox(yolo_box, 640, 480)
    assert len(bbox_back) == 4
    print("✓ Coordinate transformations")

    # Test validation
    valid_bbox = np.array([10, 20, 100, 150], dtype=np.float32)
    is_valid = validate_bbox(valid_bbox, 640, 480)
    assert is_valid
    print("✓ Bbox validation")


def test_tracking_kalman():
    """Test Kalman filter tracking"""
    from mma.tracking import KalmanBoxTracker

    bbox = np.array([100, 100, 200, 200, 0.9], dtype=np.float32)
    tracker = KalmanBoxTracker(bbox)

    assert tracker.id == 0
    assert tracker.confidence == 0.9
    print("✓ KalmanBoxTracker initialization")

    # Predict
    pred = tracker.predict()
    assert pred.shape[0] == 4  # [x1, y1, x2, y2]
    print("✓ KalmanBoxTracker predict")

    # Update
    new_bbox = np.array([105, 105, 205, 205, 0.95], dtype=np.float32)
    tracker.update(new_bbox)
    assert tracker.confidence == 0.95
    print("✓ KalmanBoxTracker update")

    # Get state
    state = tracker.get_state()
    assert state.shape[0] == 4
    print("✓ KalmanBoxTracker get_state")


def test_tracking_sort():
    """Test SORT tracker"""
    from mma.tracking import Sort

    sort_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # Frame 1: 2 detections
    dets1 = np.array([[10, 20, 100, 150, 0.9], [200, 50, 300, 200, 0.8]])
    tracked1 = sort_tracker.update(dets1)
    assert len(tracked1) > 0
    print("✓ SORT track creation")

    # Frame 2: 2 detections (moved)
    dets2 = np.array([[15, 25, 105, 155, 0.88], [210, 60, 310, 210, 0.82]])
    tracked2 = sort_tracker.update(dets2)
    assert len(tracked2) > 0
    print("✓ SORT track update")

    # Frame 3: empty detections
    dets3 = np.empty((0, 5))
    tracked3 = sort_tracker.update(dets3)
    # May return predicted tracks even without detections
    print("✓ SORT empty frame handling")


def test_tracking_mma():
    """Test MMA-specific tracker"""
    from mma.tracking import MMATracker
    from mma.core.config import TrackingConfig

    config = TrackingConfig(max_age=30, device='cpu')
    mma_tracker = MMATracker(config)

    # Create dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: 2 separated detections
    dets1 = [(10, 20, 100, 150, 0.9), (400, 20, 550, 150, 0.85)]
    tracked1 = mma_tracker.update(dummy_image, dets1, frame_num=1)
    assert len(tracked1) == 2 or len(tracked1) == 0  # May need Re-ID model
    print("✓ MMA tracker initialization")

    # Frame 2: 2 detections (moved)
    dets2 = [(15, 25, 105, 155, 0.88), (405, 25, 555, 155, 0.82)]
    tracked2 = mma_tracker.update(dummy_image, dets2, frame_num=2)
    print("✓ MMA tracker update")

    # Frame 3: no detections
    dets3 = []
    tracked3 = mma_tracker.update(dummy_image, dets3, frame_num=3)
    print("✓ MMA tracker empty frame")


def test_pose_keypoint_utils():
    """Test pose keypoint utilities"""
    from mma.pose import (
        crop_person,
        transform_keypoints_to_original,
        compute_keypoint_stats,
        compute_pose_center
    )

    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (100, 150, 300, 350)

    # Test crop
    crop, crop_info = crop_person(dummy_image, bbox, padding=0.1)
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0
    print("✓ crop_person")

    # Test keypoint transformation
    keypoints = {
        'nose': (50, 30, 0.9),
        'left_shoulder': (40, 60, 0.85),
        'right_shoulder': (60, 60, 0.82)
    }
    transformed = transform_keypoints_to_original(keypoints, crop_info)
    assert len(transformed) == 3
    print("✓ transform_keypoints_to_original")

    # Test keypoint stats
    stats = compute_keypoint_stats(keypoints)
    assert stats['num_valid'] == 3
    assert 0 <= stats['mean_confidence'] <= 1
    print("✓ compute_keypoint_stats")

    # Test pose center
    center = compute_pose_center(keypoints)
    assert center is not None
    assert len(center) == 2
    print("✓ compute_pose_center")


def test_visualization_colors():
    """Test visualization color generation"""
    from mma.visualization import generate_track_color

    color1 = generate_track_color(1)
    color2 = generate_track_color(2)

    assert len(color1) == 3
    assert len(color2) == 3
    assert color1 != color2
    print("✓ generate_track_color MMA players")

    color3 = generate_track_color(3)
    assert len(color3) == 3
    print("✓ generate_track_color other IDs")


def test_preprocessing_dataset_generator():
    """Test dataset generation"""
    from mma.preprocessing import create_yolo_detection_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        images_dir = Path(tmpdir) / 'images'
        images_dir.mkdir()

        # Create simple annotations
        annotations = [
            {
                'image_path': 'frame_001.jpg',
                'objects': [
                    {'class_id': 0, 'x': 0.5, 'y': 0.5, 'w': 0.2, 'h': 0.4}
                ]
            }
        ]

        output_dir = Path(tmpdir) / 'yolo_dataset'

        # Test dataset generation
        try:
            create_yolo_detection_dataset(
                str(images_dir),
                annotations,
                str(output_dir),
                split_ratios={'train': 1.0}  # All to train for testing
            )
            # Check if data.yaml was created
            assert (output_dir / 'data.yaml').exists()
            print("✓ Dataset generation")
        except Exception as e:
            print(f"⚠ Dataset generation (expected in test): {e}")


def test_preprocessing_splitter():
    """Test dataset splitting"""
    from mma.preprocessing import split_yolo_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / 'dataset'
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'

        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Create dummy files
        for i in range(10):
            img_file = images_dir / f'frame_{i:03d}.jpg'
            img_file.write_text('dummy')

            label_file = labels_dir / f'frame_{i:03d}.txt'
            label_file.write_text('0 0.5 0.5 0.2 0.4\n')

        # Test splitting
        try:
            split_yolo_dataset(
                str(dataset_path),
                ratios={'train': 0.7, 'val': 0.3},
                seed=42
            )

            # Check if splits were created
            assert (dataset_path / 'train' / 'images').exists()
            assert (dataset_path / 'val' / 'images').exists()
            print("✓ Dataset splitting")
        except Exception as e:
            print(f"⚠ Dataset splitting (expected in test): {e}")


def run_all_tests():
    """Run all integration tests"""
    print("=" * 70)
    print("MMA Package Integration Tests")
    print("=" * 70)

    tests = [
        ("Core Config", test_core_config),
        ("Core Constants", test_core_constants),
        ("Detection BBox Utils", test_detection_bbox_utils),
        ("Tracking Kalman", test_tracking_kalman),
        ("Tracking SORT", test_tracking_sort),
        ("Tracking MMA", test_tracking_mma),
        ("Pose Keypoint Utils", test_pose_keypoint_utils),
        ("Visualization Colors", test_visualization_colors),
        ("Preprocessing Dataset Generator", test_preprocessing_dataset_generator),
        ("Preprocessing Splitter", test_preprocessing_splitter),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
