#!/usr/bin/env python3
"""
MMA 2인 추적 파이프라인
=======================

사용법:
    python run_mma_pipeline.py \
        --detections detection_results.csv \
        --images-dir /path/to/images \
        --output-dir tracking_results

입력 CSV 형식:
    image_name, x1, y1, x2, y2, confidence
"""

import os
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import cv2

from mma_tracker import MMATemplateTracker, interpolate_track


def parse_image_name(image_name: str) -> Tuple[str, int]:
    """
    이미지 이름에서 시퀀스명과 프레임 번호 추출

    Args:
        image_name: '03_grappling2_001_grappling2_cam01_00001.jpg'

    Returns:
        ('03_grappling2_001_grappling2_cam01', 1)
    """
    base_name = os.path.splitext(image_name)[0]
    parts = base_name.rsplit('_', 1)

    if len(parts) == 2:
        try:
            frame_number = int(parts[1])
            return parts[0], frame_number
        except ValueError:
            pass

    return base_name, 0


def load_detections(
    csv_path: str,
    conf_threshold: float = 0.1
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, str]]]:
    """
    검출 CSV 로드 (시퀀스별 그룹화)

    Args:
        csv_path: 검출 CSV 경로
        conf_threshold: 최소 신뢰도 임계값

    Returns:
        (detections_by_sequence, image_name_map)
    """
    print(f"Loading detections from {csv_path}...")

    detections_by_sequence = defaultdict(lambda: defaultdict(list))
    image_name_map = defaultdict(dict)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row['image_name']
            sequence_name, frame_num = parse_image_name(image_name)

            confidence = float(row['confidence'])
            if confidence < conf_threshold:
                continue

            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2'])
            y2 = float(row['y2'])

            detection = [x1, y1, x2, y2, confidence]
            detections_by_sequence[sequence_name][frame_num].append(detection)
            image_name_map[sequence_name][frame_num] = image_name

    # numpy 배열로 변환
    for seq_name in detections_by_sequence:
        for frame_num in detections_by_sequence[seq_name]:
            detections_by_sequence[seq_name][frame_num] = np.array(
                detections_by_sequence[seq_name][frame_num]
            )

    # 통계 출력
    total_sequences = len(detections_by_sequence)
    print(f"Loaded {total_sequences} sequences:")
    for seq_name in sorted(detections_by_sequence.keys()):
        seq_frames = len(detections_by_sequence[seq_name])
        seq_dets = sum(len(dets) for dets in detections_by_sequence[seq_name].values())
        print(f"  - {seq_name}: {seq_frames} frames, {seq_dets} detections")

    return dict(detections_by_sequence), dict(image_name_map)


def save_tracking_results(
    tracks_by_frame: Dict[int, np.ndarray],
    output_csv: str,
    image_name_map: Dict[int, str] = None
):
    """
    추적 결과를 CSV로 저장

    Args:
        tracks_by_frame: {frame_num: [[x1,y1,x2,y2,track_id,conf], ...]}
        output_csv: 출력 CSV 경로
        image_name_map: {frame_num: image_name}
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        if image_name_map:
            writer.writerow(['image_name', 'frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
        else:
            writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])

        for frame_num in sorted(tracks_by_frame.keys()):
            tracks = tracks_by_frame[frame_num]
            for track in tracks:
                x1, y1, x2, y2, track_id, conf = track

                if image_name_map and frame_num in image_name_map:
                    writer.writerow([
                        image_name_map[frame_num],
                        int(frame_num),
                        int(track_id),
                        float(x1), float(y1), float(x2), float(y2),
                        float(conf)
                    ])
                else:
                    writer.writerow([
                        int(frame_num),
                        int(track_id),
                        float(x1), float(y1), float(x2), float(y2),
                        float(conf)
                    ])

    print(f"Saved: {output_csv}")


def run_mma_tracking(
    detections_by_frame: Dict[int, np.ndarray],
    images_dir: str,
    image_name_map: Dict[int, str],
    device: str = 'cuda',
    reid_ema_alpha: float = 0.1
) -> Dict[int, np.ndarray]:
    """
    MMA 템플릿 추적 실행

    Args:
        detections_by_frame: {frame_num: detections_array}
        images_dir: 이미지 디렉토리
        image_name_map: {frame_num: image_name}
        device: 'cuda' or 'cpu'
        reid_ema_alpha: Re-ID EMA 업데이트 비율

    Returns:
        {frame_num: [[x1,y1,x2,y2,track_id,conf], ...]}
    """
    print("\n=== MMA Template Tracking ===")

    tracker = MMATemplateTracker(device=device, reid_ema_alpha=reid_ema_alpha)
    tracks_by_frame = {}

    frame_numbers = sorted(detections_by_frame.keys())

    if not frame_numbers:
        print("No frames to process!")
        return tracks_by_frame

    for frame_num in tqdm(frame_numbers, desc="Tracking"):
        # 이미지 로드
        image_name = image_name_map.get(frame_num)
        if image_name is None:
            continue

        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        frame_img = cv2.imread(image_path)
        if frame_img is None:
            print(f"Warning: Could not read image: {image_path}")
            continue

        # 검출 가져오기
        detections = detections_by_frame.get(frame_num, np.empty((0, 5)))

        # 추적 업데이트
        tracks = tracker.update(frame_img, detections, frame_num)
        tracks_by_frame[frame_num] = tracks

    # 통계
    total_tracks = sum(len(t) for t in tracks_by_frame.values())
    print(f"Tracking complete: {total_tracks} track entries across {len(tracks_by_frame)} frames")

    return tracks_by_frame


def run_interpolation(
    tracks_by_frame: Dict[int, np.ndarray],
    max_gap: int = 30
) -> Dict[int, np.ndarray]:
    """
    추적 결과 보간

    Args:
        tracks_by_frame: 추적 결과
        max_gap: 최대 보간 간격

    Returns:
        보간된 추적 결과
    """
    print(f"\n=== Post-Processing: Interpolation (max_gap={max_gap}) ===")

    # 트랙 ID별로 그룹화
    tracks_by_id = defaultdict(list)
    for frame_num, tracks in tracks_by_frame.items():
        for track in tracks:
            x1, y1, x2, y2, track_id, conf = track
            tracks_by_id[int(track_id)].append((frame_num, [x1, y1, x2, y2], conf))

    # 각 트랙 보간
    interpolated_by_id = {}
    total_added = 0

    for track_id, track_data in tracks_by_id.items():
        original_len = len(track_data)
        interpolated = interpolate_track(track_data, max_gap)
        interpolated_by_id[track_id] = interpolated
        total_added += len(interpolated) - original_len

    print(f"Added {total_added} interpolated entries")

    # 프레임별로 재구성
    result = defaultdict(list)
    for track_id, track_data in interpolated_by_id.items():
        for frame_num, bbox, conf in track_data:
            result[frame_num].append([bbox[0], bbox[1], bbox[2], bbox[3], track_id, conf])

    # numpy 배열로 변환
    for frame_num in result:
        result[frame_num] = np.array(result[frame_num])

    return dict(result)


def run_pipeline_for_sequence(
    sequence_name: str,
    detections_by_frame: Dict[int, np.ndarray],
    images_dir: str,
    output_dir: str,
    image_name_map: Dict[int, str],
    device: str = 'cuda',
    reid_ema_alpha: float = 0.1,
    max_gap: int = 30
) -> Dict[int, np.ndarray]:
    """
    단일 시퀀스에 대한 파이프라인 실행

    Returns:
        최종 추적 결과
    """
    print(f"\n{'='*70}")
    print(f"Processing sequence: {sequence_name}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)

    # 출력 파일 경로
    step1_csv = os.path.join(output_dir, "step1_tracking.csv")
    step2_csv = os.path.join(output_dir, "step2_interpolated.csv")

    # Stage 1: MMA Template Tracking
    tracks_by_frame = run_mma_tracking(
        detections_by_frame,
        images_dir,
        image_name_map,
        device=device,
        reid_ema_alpha=reid_ema_alpha
    )

    save_tracking_results(tracks_by_frame, step1_csv, image_name_map)

    # Stage 2: Interpolation
    final_tracks = run_interpolation(tracks_by_frame, max_gap=max_gap)
    save_tracking_results(final_tracks, step2_csv, image_name_map)

    # 요약
    total_entries = sum(len(t) for t in final_tracks.values())
    unique_ids = set()
    for tracks in final_tracks.values():
        for t in tracks:
            unique_ids.add(int(t[4]))

    print(f"\n--- Sequence {sequence_name} Summary ---")
    print(f"Total: {total_entries} track entries, {len(unique_ids)} unique IDs")

    return final_tracks


def main():
    parser = argparse.ArgumentParser(
        description="MMA 2인 추적 파이프라인 (Template Tracking + Re-ID)"
    )

    # 입출력
    parser.add_argument("--detections", type=str, required=True,
                        help="검출 CSV 파일 경로")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="이미지 디렉토리")
    parser.add_argument("--output-dir", type=str, default="tracking_results",
                        help="출력 디렉토리")

    # 추적 파라미터
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                        help="검출 신뢰도 임계값 (default: 0.1)")
    parser.add_argument("--reid-ema-alpha", type=float, default=0.1,
                        help="Re-ID EMA 업데이트 비율 (default: 0.1)")
    parser.add_argument("--max-gap", type=int, default=30,
                        help="보간 최대 간격 (default: 30)")
    parser.add_argument("--device", type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help="Re-ID 모델 디바이스 (default: cuda)")

    args = parser.parse_args()

    print("="*70)
    print("MMA 2-PLAYER TRACKING PIPELINE")
    print("="*70)
    print(f"Input detections: {args.detections}")
    print(f"Images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*70)

    # 검출 로드
    detections_by_sequence, image_name_map = load_detections(
        args.detections,
        conf_threshold=args.conf_threshold
    )

    # 각 시퀀스 처리
    all_results = {}
    for sequence_name in sorted(detections_by_sequence.keys()):
        detections_by_frame = detections_by_sequence[sequence_name]
        seq_image_map = image_name_map[sequence_name]
        output_subdir = os.path.join(args.output_dir, sequence_name)

        final_tracks = run_pipeline_for_sequence(
            sequence_name=sequence_name,
            detections_by_frame=detections_by_frame,
            images_dir=args.images_dir,
            output_dir=output_subdir,
            image_name_map=seq_image_map,
            device=args.device,
            reid_ema_alpha=args.reid_ema_alpha,
            max_gap=args.max_gap
        )
        all_results[sequence_name] = final_tracks

    # 최종 요약
    print("\n" + "="*70)
    print("ALL SEQUENCES COMPLETE!")
    print("="*70)

    total_entries = 0
    for seq_name, tracks in all_results.items():
        seq_entries = sum(len(t) for t in tracks.values())
        total_entries += seq_entries
        unique_ids = set()
        for t in tracks.values():
            for track in t:
                unique_ids.add(int(track[4]))
        print(f"  {seq_name}: {seq_entries} entries, {len(unique_ids)} IDs")

    print(f"\nTotal: {total_entries} entries across {len(all_results)} sequences")
    print("="*70)


if __name__ == "__main__":
    main()
