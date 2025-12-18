#!/usr/bin/env python3
"""
경기장 좌표 변환 (Homography Transform)
======================================

추적 결과의 bbox 하단 중점을 실제 경기장 좌표(미터)로 변환

Usage:
    python transform_coordinates.py \
        --tracking-csv tracking_results/step2_interpolated.csv \
        --output transformed_coords.csv \
        --preset grappling

    # 배치 모드
    python transform_coordinates.py \
        --batch \
        --tracking-dir mma_tracking_results \
        --preset grappling
"""

import cv2
import csv
import os
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


# ============================================================================
# 경기장 설정 (Presets)
# ============================================================================

STAGE_PRESETS = {
    "grappling": {
        "description": "Grappling 경기장 (4개 시퀀스 공통)",
        "real_size": (3.5, 3.0),  # 가로 3.5m, 세로 3.0m
        "pixel_corners": {
            "top_left": (1600, 900),
            "top_right": (3120, 1000),
            "bottom_left": (510, 3120),
            "bottom_right": (3270, 2150),
        }
    },
    "mma": {
        "description": "MMA 경기장 (4개 시퀀스 공통)",
        "real_size": (3.5, 3.0),  # 가로 3.5m, 세로 3.0m
        "pixel_corners": {
            "top_left": (1210, 1100),
            "top_right": (2710, 1120),
            "bottom_left": (30, 1510),
            "bottom_right": (3580, 2050),
        }
    },
}


def get_homography_matrix(preset_name: str) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    프리셋에서 호모그래피 행렬 계산

    Args:
        preset_name: 프리셋 이름 (예: "grappling", "mma")

    Returns:
        (homography_matrix, real_size)
    """
    if preset_name not in STAGE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(STAGE_PRESETS.keys())}")

    preset = STAGE_PRESETS[preset_name]
    corners = preset["pixel_corners"]
    real_w, real_h = preset["real_size"]

    # 픽셀 좌표 (source points)
    src_points = np.array([
        corners["top_left"],
        corners["top_right"],
        corners["bottom_left"],
        corners["bottom_right"],
    ], dtype=np.float32)

    # 실제 좌표 (destination points)
    # 좌상 = (0, 0), 우상 = (W, 0), 좌하 = (0, H), 우하 = (W, H)
    dst_points = np.array([
        [0, 0],
        [real_w, 0],
        [0, real_h],
        [real_w, real_h],
    ], dtype=np.float32)

    # 호모그래피 행렬 계산
    H, _ = cv2.findHomography(src_points, dst_points)

    return H, (real_w, real_h)


def transform_point(pixel_point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """
    픽셀 좌표를 실제 좌표로 변환

    Args:
        pixel_point: (x, y) 픽셀 좌표
        H: 호모그래피 행렬

    Returns:
        (real_x, real_y) 실제 좌표 (미터)
    """
    pt = np.array([[pixel_point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])


def bbox_bottom_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """
    bbox의 하단 중점 계산 (발 위치 추정)

    Args:
        x1, y1, x2, y2: bbox 좌표

    Returns:
        (center_x, bottom_y)
    """
    center_x = (x1 + x2) / 2
    bottom_y = y2  # 하단
    return center_x, bottom_y


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """두 점 사이의 유클리드 거리 (미터)"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def transform_tracking_results(
    tracking_csv: str,
    output_csv: str,
    preset_name: str,
    fps: float = 30.0
):
    """
    추적 결과를 실제 좌표로 변환

    Args:
        tracking_csv: 입력 추적 CSV
        output_csv: 출력 CSV (실제 좌표 포함)
        preset_name: 경기장 프리셋
        fps: 프레임 레이트 (속도 계산용)
    """
    print(f"Loading tracking results from {tracking_csv}...")

    # 호모그래피 행렬 계산
    H, (real_w, real_h) = get_homography_matrix(preset_name)
    print(f"Preset: {preset_name}")
    print(f"Real stage size: {real_w}m x {real_h}m")

    # 추적 결과 로드
    tracks_by_frame = defaultdict(dict)  # {frame: {player_id: data}}

    with open(tracking_csv, 'r') as f:
        reader = csv.DictReader(f)
        has_image_name = 'image_name' in reader.fieldnames

        for row in reader:
            frame_num = int(row['frame'])
            track_id = int(row['track_id'])
            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])
            confidence = float(row['confidence'])

            # bbox 하단 중점
            pixel_x, pixel_y = bbox_bottom_center(x1, y1, x2, y2)

            # 실제 좌표로 변환
            real_x, real_y = transform_point((pixel_x, pixel_y), H)

            tracks_by_frame[frame_num][track_id] = {
                'image_name': row.get('image_name', ''),
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'real_x': real_x,
                'real_y': real_y,
                'confidence': confidence,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            }

    # 프레임 정렬
    frame_numbers = sorted(tracks_by_frame.keys())
    print(f"Loaded {len(frame_numbers)} frames")

    # 결과 저장
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # 헤더
        writer.writerow([
            'frame', 'time_sec', 'player_id',
            'pixel_x', 'pixel_y',
            'real_x', 'real_y',
            'distance_between_players',
            'velocity_mps',  # meters per second
            'confidence'
        ])

        prev_positions = {}  # {player_id: (real_x, real_y)}

        for frame_num in frame_numbers:
            frame_data = tracks_by_frame[frame_num]
            time_sec = frame_num / fps

            # 두 선수 간 거리 계산
            distance_between = None
            if 1 in frame_data and 2 in frame_data:
                p1 = (frame_data[1]['real_x'], frame_data[1]['real_y'])
                p2 = (frame_data[2]['real_x'], frame_data[2]['real_y'])
                distance_between = calculate_distance(p1, p2)

            # 각 선수별 데이터 작성
            for player_id in sorted(frame_data.keys()):
                data = frame_data[player_id]

                # 속도 계산 (이전 프레임 대비)
                velocity = None
                curr_pos = (data['real_x'], data['real_y'])

                if player_id in prev_positions:
                    prev_pos = prev_positions[player_id]
                    dist_moved = calculate_distance(curr_pos, prev_pos)
                    velocity = dist_moved * fps  # m/s

                prev_positions[player_id] = curr_pos

                writer.writerow([
                    frame_num,
                    f"{time_sec:.3f}",
                    player_id,
                    f"{data['pixel_x']:.1f}",
                    f"{data['pixel_y']:.1f}",
                    f"{data['real_x']:.4f}",
                    f"{data['real_y']:.4f}",
                    f"{distance_between:.4f}" if distance_between else "",
                    f"{velocity:.4f}" if velocity else "",
                    f"{data['confidence']:.4f}"
                ])

    print(f"Saved transformed coordinates to {output_csv}")

    # 통계 출력
    print_statistics(output_csv)


def print_statistics(csv_path: str):
    """변환 결과 통계 출력"""
    print("\n=== Statistics ===")

    distances = []
    velocities = {1: [], 2: []}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['distance_between_players']:
                distances.append(float(row['distance_between_players']))
            if row['velocity_mps']:
                player_id = int(row['player_id'])
                velocities[player_id].append(float(row['velocity_mps']))

    if distances:
        print(f"Distance between players:")
        print(f"  Min: {min(distances):.2f}m")
        print(f"  Max: {max(distances):.2f}m")
        print(f"  Mean: {np.mean(distances):.2f}m")

    for player_id, vels in velocities.items():
        if vels:
            print(f"Player {player_id} velocity:")
            print(f"  Max: {max(vels):.2f} m/s")
            print(f"  Mean: {np.mean(vels):.2f} m/s")


def transform_all_sequences(
    tracking_dir: str,
    preset_name: str,
    csv_filename: str = "step2_interpolated.csv",
    fps: float = 30.0
):
    """
    모든 시퀀스에 대해 좌표 변환 수행

    Args:
        tracking_dir: 추적 결과 디렉토리
        preset_name: 경기장 프리셋
        csv_filename: 입력 CSV 파일명
        fps: 프레임 레이트
    """
    print("="*70)
    print("Batch Coordinate Transform")
    print("="*70)
    print(f"Tracking dir: {tracking_dir}")
    print(f"Preset: {preset_name}")
    print(f"CSV file: {csv_filename}")
    print("="*70)

    # 시퀀스 폴더 찾기
    sequences = []
    for item in os.listdir(tracking_dir):
        seq_path = os.path.join(tracking_dir, item)
        csv_path = os.path.join(seq_path, csv_filename)
        if os.path.isdir(seq_path) and os.path.exists(csv_path):
            sequences.append(item)

    if not sequences:
        print(f"No sequences found with {csv_filename}")
        return

    print(f"Found {len(sequences)} sequences: {sequences}")

    # 각 시퀀스 처리

    for seq_name in sorted(sequences):
        print(f"\n--- Processing: {seq_name} ---")

        # 시퀀스명에 따라 프리셋 자동 선택
        seq_lower = seq_name.lower()
        if "grappling" in seq_lower:
            seq_preset = "grappling"
        elif "mma" in seq_lower:
            seq_preset = "mma"
        else:
            seq_preset = preset_name  # fallback
            print(f"  [경고] 시퀀스명에서 프리셋을 추론할 수 없어 기본값({preset_name}) 사용")

        print(f"  [Info] Preset for this sequence: {seq_preset}")

        input_csv = os.path.join(tracking_dir, seq_name, csv_filename)
        output_csv = os.path.join(tracking_dir, seq_name, "real_coordinates.csv")

        transform_tracking_results(input_csv, output_csv, seq_preset, fps)

    print(f"\n{'='*70}")
    print("All sequences transformed!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="경기장 좌표 변환 (픽셀 → 실제 미터)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 시퀀스
  python transform_coordinates.py \\
      --tracking-csv results/step2_interpolated.csv \\
      --output results/real_coordinates.csv \\
      --preset grappling

  # 배치 모드 (grappling)
  python transform_coordinates.py \\
      --batch \\
      --tracking-dir mma_tracking_results \\
      --preset grappling

Available presets:
  - grappling: Grappling 경기장 (3.5m x 3.0m)
  - mma: MMA 경기장 (3.5m x 3.0m)
        """
    )

    # 모드 선택
    parser.add_argument("--batch", action="store_true",
                        help="배치 모드: 모든 시퀀스 변환")

    # 단일 모드
    parser.add_argument("--tracking-csv", type=str,
                        help="입력 추적 CSV (단일 모드)")
    parser.add_argument("--output", type=str,
                        help="출력 CSV (단일 모드)")

    # 배치 모드
    parser.add_argument("--tracking-dir", type=str, default="mma_tracking_results",
                        help="추적 결과 디렉토리 (배치 모드)")
    parser.add_argument("--csv-file", type=str, default="step2_interpolated.csv",
                        help="입력 CSV 파일명 (배치 모드)")

    # 공통 (단일 모드에서만 필수, 배치 모드에서는 선택적)
    parser.add_argument("--preset", type=str,
                        choices=list(STAGE_PRESETS.keys()),
                        help="경기장 프리셋 (단일 모드 필수, 배치 모드에서는 폴더명 자동 추론, 미지정시 fallback)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="프레임 레이트 (default: 30)")

    args = parser.parse_args()

    if args.batch:
        # 배치 모드: preset은 fallback 용으로만 사용
        transform_all_sequences(
            args.tracking_dir,
            args.preset if args.preset else "mma",
            args.csv_file,
            args.fps
        )
    else:
        if not args.tracking_csv or not args.output or not args.preset:
            parser.print_help()
            print("\nError: --tracking-csv, --output, --preset 모두 단일 모드에서 필요합니다.")
            return

        transform_tracking_results(
            args.tracking_csv,
            args.output,
            args.preset,
            args.fps
        )


if __name__ == "__main__":
    main()
