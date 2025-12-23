#!/usr/bin/env python3
"""
선수별 히트맵 시각화
====================

real_coordinates.csv를 읽어서 각 선수의 위치 히트맵을 생성

Usage:
    # 단일 시퀀스
    python visualize_heatmap.py \
        --input tracking_results/real_coordinates.csv \
        --output heatmap.png \
        --preset grappling

    # 배치 모드
    python visualize_heatmap.py \
        --batch \
        --tracking-dir mma_tracking_results \
        --preset grappling
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 렌더링
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple
from tqdm import tqdm
import cv2


# 경기장 프리셋 (transform_coordinates.py와 동일)
STAGE_PRESETS = {
    "grappling": {
        "description": "Grappling 경기장 (4개 시퀀스 공통)",
        "real_size": (3.5, 3.0),  # 가로 3.5m, 세로 3.0m
    },
    "mma": {
        "description": "MMA 경기장 (4개 시퀀스 공통)",
        "real_size": (3.5, 3.0),  # 가로 3.5m, 세로 3.0m
    },
}


def load_coordinates(csv_path: str) -> Dict[int, List[Tuple[float, float]]]:
    """
    real_coordinates.csv 로드

    Args:
        csv_path: CSV 파일 경로

    Returns:
        {player_id: [(real_x, real_y), ...]}
    """
    coords_by_player = {1: [], 2: []}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_id = int(row['player_id'])
            real_x = float(row['real_x'])
            real_y = float(row['real_y'])

            if player_id in coords_by_player:
                coords_by_player[player_id].append((real_x, real_y))

    return coords_by_player


def load_coordinates_by_frame(csv_path: str) -> Tuple[Dict[int, Dict[int, Tuple[float, float]]], Dict[int, str]]:
    """
    real_coordinates.csv를 프레임별로 로드

    Args:
        csv_path: CSV 파일 경로

    Returns:
        (coords_by_frame, image_name_map)
        coords_by_frame: {frame_num: {player_id: (real_x, real_y)}}
        image_name_map: {frame_num: image_name}
    """
    coords_by_frame = {}
    image_name_map = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            frame_num = int(row['frame'])
            player_id = int(row['player_id'])
            real_x = float(row['real_x'])
            real_y = float(row['real_y'])

            if frame_num not in coords_by_frame:
                coords_by_frame[frame_num] = {}

            coords_by_frame[frame_num][player_id] = (real_x, real_y)

    return coords_by_frame, image_name_map


def create_heatmap(
    coords: List[Tuple[float, float]],
    stage_size: Tuple[float, float],
    bins: int = 50
) -> np.ndarray:
    """
    좌표 리스트로부터 2D 히스토그램(히트맵) 생성

    Args:
        coords: [(x, y), ...] 좌표 리스트
        stage_size: (width, height) 경기장 크기 (미터)
        bins: 히스토그램 bin 수

    Returns:
        2D numpy array (히트맵)
    """
    if not coords:
        return np.zeros((bins, bins))

    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    # 2D 히스토그램
    heatmap, xedges, yedges = np.histogram2d(
        x_coords, y_coords,
        bins=bins,
        range=[[0, stage_size[0]], [0, stage_size[1]]]
    )

    # 가우시안 스무딩
    from scipy.ndimage import gaussian_filter
    heatmap = gaussian_filter(heatmap, sigma=2)

    return heatmap.T  # transpose for correct orientation


def visualize_player_heatmap(
    coords_by_player: Dict[int, List[Tuple[float, float]]],
    stage_size: Tuple[float, float],
    output_path: str,
    title: str = "Player Heatmap",
    bins: int = 50
):
    """
    선수별 히트맵 시각화 및 저장

    Args:
        coords_by_player: {player_id: [(x, y), ...]}
        stage_size: (width, height) 경기장 크기
        output_path: 출력 이미지 경로
        title: 그래프 제목
        bins: 히스토그램 bin 수
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 커스텀 컬러맵
    colors_red = ['white', 'lightyellow', 'yellow', 'orange', 'red', 'darkred']
    colors_blue = ['white', 'lightcyan', 'cyan', 'deepskyblue', 'blue', 'darkblue']
    cmap_red = LinearSegmentedColormap.from_list('red_heat', colors_red)
    cmap_blue = LinearSegmentedColormap.from_list('blue_heat', colors_blue)

    # Player 1 히트맵
    ax1 = axes[0]
    heatmap1 = create_heatmap(coords_by_player.get(1, []), stage_size, bins)
    im1 = ax1.imshow(
        heatmap1,
        extent=[0, stage_size[0], stage_size[1], 0],
        cmap=cmap_red,
        aspect='equal',
        interpolation='bilinear'
    )
    ax1.set_title(f'Player 1 (Red)\n{len(coords_by_player.get(1, []))} frames', fontsize=12)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    draw_stage_boundary(ax1, stage_size)
    plt.colorbar(im1, ax=ax1, label='Density')

    # Player 2 히트맵
    ax2 = axes[1]
    heatmap2 = create_heatmap(coords_by_player.get(2, []), stage_size, bins)
    im2 = ax2.imshow(
        heatmap2,
        extent=[0, stage_size[0], stage_size[1], 0],
        cmap=cmap_blue,
        aspect='equal',
        interpolation='bilinear'
    )
    ax2.set_title(f'Player 2 (Blue)\n{len(coords_by_player.get(2, []))} frames', fontsize=12)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    draw_stage_boundary(ax2, stage_size)
    plt.colorbar(im2, ax=ax2, label='Density')

    # Combined 히트맵 (둘 다 표시)
    ax3 = axes[2]
    # 두 히트맵을 다른 채널로 결합
    combined = np.zeros((heatmap1.shape[0], heatmap1.shape[1], 3))
    # 정규화
    if heatmap1.max() > 0:
        combined[:, :, 0] = heatmap1 / heatmap1.max()  # Red channel
    if heatmap2.max() > 0:
        combined[:, :, 2] = heatmap2 / heatmap2.max()  # Blue channel

    ax3.imshow(
        combined,
        extent=[0, stage_size[0], stage_size[1], 0],
        aspect='equal',
        interpolation='bilinear'
    )
    ax3.set_title('Combined\n(Red: Player 1, Blue: Player 2)', fontsize=12)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    draw_stage_boundary(ax3, stage_size)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 저장
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved heatmap: {output_path}")


def draw_stage_boundary(ax, stage_size: Tuple[float, float]):
    """경기장 경계선 그리기"""
    ax.plot([0, stage_size[0]], [0, 0], 'k-', linewidth=2)
    ax.plot([0, stage_size[0]], [stage_size[1], stage_size[1]], 'k-', linewidth=2)
    ax.plot([0, 0], [0, stage_size[1]], 'k-', linewidth=2)
    ax.plot([stage_size[0], stage_size[0]], [0, stage_size[1]], 'k-', linewidth=2)

    # 그리드
    ax.set_xlim(-0.1, stage_size[0] + 0.1)
    ax.set_ylim(stage_size[1] + 0.1, -0.1)  # Y축 뒤집기 (상단이 0)
    ax.grid(True, alpha=0.3, linestyle='--')


def visualize_trajectory(
    coords_by_player: Dict[int, List[Tuple[float, float]]],
    stage_size: Tuple[float, float],
    output_path: str,
    title: str = "Player Trajectory"
):
    """
    선수 이동 궤적 시각화

    Args:
        coords_by_player: {player_id: [(x, y), ...]}
        stage_size: (width, height) 경기장 크기
        output_path: 출력 이미지 경로
        title: 그래프 제목
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {1: 'red', 2: 'blue'}
    labels = {1: 'Player 1', 2: 'Player 2'}

    for player_id, coords in coords_by_player.items():
        if not coords:
            continue

        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]

        # 궤적 (연한 선)
        ax.plot(x_coords, y_coords, color=colors[player_id], alpha=0.3, linewidth=1)

        # 시작점 (큰 마커)
        ax.scatter(x_coords[0], y_coords[0], color=colors[player_id],
                   s=100, marker='o', edgecolors='black', linewidths=2,
                   label=f'{labels[player_id]} Start', zorder=5)

        # 끝점 (다이아몬드)
        ax.scatter(x_coords[-1], y_coords[-1], color=colors[player_id],
                   s=100, marker='D', edgecolors='black', linewidths=2,
                   label=f'{labels[player_id]} End', zorder=5)

    draw_stage_boundary(ax, stage_size)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved trajectory: {output_path}")


def process_single_sequence(
    input_csv: str,
    output_dir: str,
    preset_name: str,
    sequence_name: str = ""
):
    """
    단일 시퀀스 처리

    Args:
        input_csv: real_coordinates.csv 경로
        output_dir: 출력 디렉토리
        preset_name: 프리셋 이름
        sequence_name: 시퀀스 이름 (제목용)
    """
    if preset_name not in STAGE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    stage_size = STAGE_PRESETS[preset_name]["real_size"]

    # 좌표 로드
    coords_by_player = load_coordinates(input_csv)

    print(f"Loaded coordinates: Player 1 = {len(coords_by_player[1])} frames, "
          f"Player 2 = {len(coords_by_player[2])} frames")


    # visualization 폴더 경로
    vis_dir = os.path.join(output_dir, "visualization")

    # Trajectory 전체를 시간 순서대로 concat하여 하나의 이미지로 저장
    os.makedirs(vis_dir, exist_ok=True)
    title = f"Trajectory - {sequence_name}" if sequence_name else "Player Trajectory"
    trajectory_path = os.path.join(vis_dir, "trajectory_concat.png")
    save_concat_trajectory(coords_by_player, stage_size, trajectory_path, title)

    # 프레임별 concat 이미지 생성 (원본 프레임 + trajectory 맵)
    concat_output_dir = os.path.join(vis_dir, "concat")
    generate_concat_frames(
        coords_csv=input_csv,
        vis_frames_dir=vis_dir,
        output_dir=concat_output_dir,
        stage_size=stage_size,
        sequence_name=sequence_name
    )


def save_concat_trajectory(
    coords_by_player: Dict[int, List[Tuple[float, float]]],
    stage_size: Tuple[float, float],
    output_path: str,
    title: str = "Player Trajectory"
):
    """
    모든 프레임의 궤적을 시간 순서대로 concat하여 하나의 이미지로 저장
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {1: 'red', 2: 'blue'}
    labels = {1: 'Player 1', 2: 'Player 2'}


    # 각 선수의 전체 궤적을 시간 순서대로 한 번에 그림
    for player_id, coords in coords_by_player.items():
        if not coords:
            continue
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        ax.plot(x_coords, y_coords, color=colors[player_id], alpha=0.7, linewidth=2, label=labels[player_id])
        # 시작점
        ax.scatter(x_coords[0], y_coords[0], color=colors[player_id], s=80, marker='o', edgecolors='black', linewidths=2, label=f'{labels[player_id]} Start', zorder=5)
        # 끝점
        ax.scatter(x_coords[-1], y_coords[-1], color=colors[player_id], s=80, marker='D', edgecolors='black', linewidths=2, label=f'{labels[player_id]} End', zorder=5)

    draw_stage_boundary(ax, stage_size)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved concat trajectory: {output_path}")


def create_trajectory_frame(
    trajectory_history: Dict[int, List[Tuple[float, float]]],
    current_positions: Dict[int, Tuple[float, float]],
    stage_size: Tuple[float, float],
    frame_num: int,
    target_height: int = 720
) -> np.ndarray:
    """
    현재 프레임까지의 trajectory + 현재 위치 마커가 있는 이미지 생성

    Args:
        trajectory_history: {player_id: [(x, y), ...]} 지금까지의 궤적
        current_positions: {player_id: (x, y)} 현재 프레임 위치
        stage_size: (width, height) 경기장 크기 (미터)
        frame_num: 현재 프레임 번호
        target_height: 출력 이미지 높이

    Returns:
        numpy array (BGR 이미지)
    """
    # Figure 크기 계산 (경기장 비율 유지)
    aspect_ratio = stage_size[0] / stage_size[1]
    fig_height = 6
    fig_width = fig_height * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    colors = {1: 'red', 2: 'blue'}
    labels = {1: 'Player 1', 2: 'Player 2'}

    # 지나온 궤적 그리기 (연한 선)
    for player_id, history in trajectory_history.items():
        if not history:
            continue
        x_coords = [c[0] for c in history]
        y_coords = [c[1] for c in history]
        ax.plot(x_coords, y_coords, color=colors[player_id], alpha=0.4, linewidth=2)

        # 시작점 (작은 마커)
        ax.scatter(x_coords[0], y_coords[0], color=colors[player_id],
                   s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.5)

    # 현재 위치 (큰 마커)
    for player_id, pos in current_positions.items():
        ax.scatter(pos[0], pos[1], color=colors[player_id],
                   s=200, marker='o', edgecolors='black', linewidths=3,
                   label=labels[player_id], zorder=10)

    # 경기장 경계
    draw_stage_boundary(ax, stage_size)

    # 프레임 번호 표시
    ax.set_title(f'Frame: {frame_num}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # Figure를 numpy array로 변환
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # RGB 3채널만 사용
    img = img[..., :3]
    # BGR로 변환 (OpenCV 호환)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 타겟 높이로 리사이즈
    h, w = img_bgr.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    img_resized = cv2.resize(img_bgr, (new_w, target_height))

    return img_resized


def generate_concat_frames(
    coords_csv: str,
    vis_frames_dir: str,
    output_dir: str,
    stage_size: Tuple[float, float],
    sequence_name: str = ""
):
    """
    각 프레임에 대해 원본 프레임과 trajectory 맵을 concat하여 저장

    Args:
        coords_csv: real_coordinates.csv 경로
        vis_frames_dir: visualization 폴더 경로 (bbox 프레임 이미지)
        output_dir: concat 이미지 출력 디렉토리
        stage_size: (width, height) 경기장 크기
        sequence_name: 시퀀스 이름
    """
    print(f"Generating concat frames for {sequence_name}...")

    # 좌표 로드 (프레임별)
    coords_by_frame, _ = load_coordinates_by_frame(coords_csv)

    if not coords_by_frame:
        print("No coordinate data found!")
        return

    frame_numbers = sorted(coords_by_frame.keys())
    print(f"Found {len(frame_numbers)} frames with coordinates")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 프레임별 파일 목록 확인
    vis_files = {}
    if os.path.exists(vis_frames_dir):
        for f in os.listdir(vis_frames_dir):
            if f.endswith(('.jpg', '.png')):
                vis_files[f] = os.path.join(vis_frames_dir, f)

    if not vis_files:
        print(f"No visualization frames found in {vis_frames_dir}")
        return

    print(f"Found {len(vis_files)} visualization frames")

    # 궤적 히스토리 초기화
    trajectory_history = {1: [], 2: []}

    for frame_num in tqdm(frame_numbers, desc="Generating concat frames"):
        # 현재 프레임 위치
        current_positions = coords_by_frame.get(frame_num, {})

        # 궤적 히스토리 업데이트
        for player_id, pos in current_positions.items():
            trajectory_history[player_id].append(pos)

        # visualization 프레임 찾기
        # 파일명 패턴: frame_000001.jpg 또는 seq_name_00001.jpg 등
        vis_frame_path = None
        for filename, filepath in vis_files.items():
            # 프레임 번호 추출 시도
            base = os.path.splitext(filename)[0]
            parts = base.split('_')
            try:
                # 마지막 부분이 숫자인 경우
                file_frame_num = int(parts[-1])
                if file_frame_num == frame_num:
                    vis_frame_path = filepath
                    break
            except ValueError:
                continue

        if vis_frame_path is None:
            continue

        # 원본 프레임 로드
        orig_frame = cv2.imread(vis_frame_path)
        if orig_frame is None:
            continue

        # 타겟 높이 (원본 프레임 높이 사용)
        target_height = orig_frame.shape[0]

        # Trajectory 맵 생성
        traj_frame = create_trajectory_frame(
            trajectory_history,
            current_positions,
            stage_size,
            frame_num,
            target_height
        )

        # 원본 프레임 높이 맞추기
        if orig_frame.shape[0] != target_height:
            scale = target_height / orig_frame.shape[0]
            new_w = int(orig_frame.shape[1] * scale)
            orig_frame = cv2.resize(orig_frame, (new_w, target_height))

        # 좌우 concat
        concat_frame = np.hstack([orig_frame, traj_frame])

        # 저장
        output_filename = os.path.basename(vis_frame_path)
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, concat_frame)

    print(f"Saved {len(frame_numbers)} concat frames to {output_dir}")


def process_all_sequences(
    tracking_dir: str,
    preset_name: str,
    csv_filename: str = "real_coordinates.csv"
):
    """
    모든 시퀀스에 대해 히트맵 생성

    Args:
        tracking_dir: 추적 결과 디렉토리
        preset_name: 프리셋 이름
        csv_filename: 입력 CSV 파일명
    """
    print("=" * 70)
    print("Batch Heatmap Visualization")
    print("=" * 70)
    print(f"Tracking dir: {tracking_dir}")
    print(f"Preset: {preset_name}")
    print(f"CSV file: {csv_filename}")
    print("=" * 70)

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
        output_dir = os.path.join(tracking_dir, seq_name)

        process_single_sequence(input_csv, output_dir, seq_preset, seq_name)

    print(f"\n{'=' * 70}")
    print("All sequences visualized!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="선수별 히트맵 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 시퀀스
  python visualize_data.py \\
      --input results/real_coordinates.csv \\
      --output-dir results \\
      --preset grappling

  # 배치 모드
  python visualize_data.py \\
      --batch \\
      --tracking-dir mma_tracking_results \\
      --preset mma

Available presets:
  - grappling: Grappling 경기장 (3.5m x 3.0m)
  - mma: MMA 경기장 (3.5m x 3.0m)
        """
    )

    # 모드 선택
    parser.add_argument("--batch", action="store_true",
                        help="배치 모드: 모든 시퀀스 시각화")

    # 단일 모드
    parser.add_argument("--input", type=str,
                        help="입력 CSV (real_coordinates.csv)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="출력 디렉토리")

    # 배치 모드
    parser.add_argument("--tracking-dir", type=str, default="tracking_results",
                        help="추적 결과 디렉토리 (배치 모드)")
    parser.add_argument("--csv-file", type=str, default="real_coordinates.csv",
                        help="입력 CSV 파일명 (배치 모드)")


    # 공통 (단일 모드에서만 필수, 배치 모드에서는 선택적)
    parser.add_argument("--preset", type=str,
                        choices=list(STAGE_PRESETS.keys()),
                        help="경기장 프리셋 (단일 모드 필수, 배치 모드에서는 폴더명 자동 추론, 미지정시 fallback)")

    args = parser.parse_args()

    if args.batch:
        # 배치 모드: preset은 fallback 용으로만 사용
        process_all_sequences(args.tracking_dir, args.preset if args.preset else "mma", args.csv_file)
    else:
        if not args.input or not args.preset:
            parser.print_help()
            print("\nError: --input 및 --preset 모두 단일 모드에서 필요합니다.")
            return

        process_single_sequence(args.input, args.output_dir, args.preset)


if __name__ == "__main__":
    main()
