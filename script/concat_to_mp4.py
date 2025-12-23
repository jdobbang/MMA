#!/usr/bin/env python3
"""
Image sequence to MP4 video converter using FFmpeg.

Converts concat visualization frames to MP4 video.
Target: tracking_results/*/visualization/concat/

Usage:
    # 전체 시퀀스 배치 처리
    python concat_to_mp4.py --tracking-dir mma_tracking_results

    # 단일 폴더
    python concat_to_mp4.py --input-dir path/to/concat --output video.mp4

    # 프레임 레이트 지정
    python concat_to_mp4.py --tracking-dir mma_tracking_results --fps 30
"""

import os
import argparse
import subprocess
import glob
from pathlib import Path


def get_image_files(input_dir: str) -> list:
    """이미지 파일 목록을 정렬하여 반환"""
    patterns = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pattern)))

    # 파일명 기준 정렬
    files.sort(key=lambda x: os.path.basename(x))
    return files


def images_to_video_ffmpeg(input_dir: str, output_path: str, fps: float = 30.0) -> bool:
    """
    FFmpeg를 사용하여 이미지 시퀀스를 MP4로 변환

    Args:
        input_dir: 이미지가 있는 폴더
        output_path: 출력 MP4 파일 경로
        fps: 프레임 레이트

    Returns:
        성공 여부
    """
    # 이미지 파일 확인
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"  [SKIP] No images found in {input_dir}")
        return False

    print(f"  Found {len(image_files)} images")

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # concat demuxer 사용 (모든 파일명 패턴 지원)
    list_file = os.path.join(input_dir, '_ffmpeg_list.txt')
    with open(list_file, 'w') as f:
        for img in image_files:
            # 각 프레임의 duration 지정
            f.write(f"file '{os.path.abspath(img)}'\n")
            f.write(f"duration {1/fps}\n")
        # 마지막 프레임 한번 더 (FFmpeg concat 버그 방지)
        if image_files:
            f.write(f"file '{os.path.abspath(image_files[-1])}'\n")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-vsync', 'vfr',
        output_path
    ]

    print(f"  Running FFmpeg...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # 임시 파일 삭제
        if os.path.exists(list_file):
            os.remove(list_file)

        if result.returncode == 0:
            # 파일 크기 확인
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  [OK] Created: {output_path} ({size_mb:.1f} MB)")
                return True
            else:
                print(f"  [ERROR] Output file not created")
                return False
        else:
            print(f"  [ERROR] FFmpeg failed:")
            print(f"  {result.stderr[-500:] if result.stderr else 'No error message'}")
            return False

    except FileNotFoundError:
        print("  [ERROR] FFmpeg not found. Please install FFmpeg:")
        print("    Ubuntu/Debian: sudo apt install ffmpeg")
        print("    macOS: brew install ffmpeg")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def process_all_sequences(tracking_dir: str, fps: float = 30.0):
    """
    모든 시퀀스의 concat 폴더를 처리

    Args:
        tracking_dir: 추적 결과 디렉토리 (예: mma_tracking_results)
        fps: 프레임 레이트
    """
    print("=" * 70)
    print("Batch Image-to-Video Conversion")
    print("=" * 70)
    print(f"Tracking dir: {tracking_dir}")
    print(f"FPS: {fps}")
    print("=" * 70)

    # 시퀀스 폴더 찾기
    sequences = []
    for item in os.listdir(tracking_dir):
        concat_dir = os.path.join(tracking_dir, item, "visualization", "concat")
        if os.path.isdir(concat_dir):
            sequences.append((item, concat_dir))

    if not sequences:
        print("No sequences found with visualization/concat folder")
        return

    print(f"Found {len(sequences)} sequences")

    success_count = 0
    fail_count = 0

    for seq_name, concat_dir in sorted(sequences):
        print(f"\n--- {seq_name} ---")

        output_path = os.path.join(tracking_dir, seq_name, f"{seq_name}_concat.mp4")

        if images_to_video_ffmpeg(concat_dir, output_path, fps):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 70}")
    print(f"Completed: {success_count} success, {fail_count} failed")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert image sequences to MP4 video using FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 시퀀스 배치 처리
  python script/concat_to_mp4.py --tracking-dir tracking_results

  # 단일 폴더 처리
  python script/concat_to_mp4.py --input-dir results/visualization/concat --output output.mp4

  # 프레임 레이트 지정
  python script/concat_to_mp4.py --tracking-dir tracking_results --fps 25
        """
    )

    # 배치 모드
    parser.add_argument("--tracking-dir", type=str,
                        help="추적 결과 디렉토리 (배치 모드)")

    # 단일 모드
    parser.add_argument("--input-dir", type=str,
                        help="입력 이미지 폴더 (단일 모드)")
    parser.add_argument("--output", type=str,
                        help="출력 MP4 파일 경로 (단일 모드)")

    # 공통
    parser.add_argument("--fps", type=float, default=30.0,
                        help="프레임 레이트 (default: 30)")

    args = parser.parse_args()

    if args.tracking_dir:
        # 배치 모드
        if not os.path.exists(args.tracking_dir):
            print(f"Error: Directory not found: {args.tracking_dir}")
            return
        process_all_sequences(args.tracking_dir, args.fps)

    elif args.input_dir:
        # 단일 모드
        if not os.path.exists(args.input_dir):
            print(f"Error: Directory not found: {args.input_dir}")
            return

        output_path = args.output or "output.mp4"
        images_to_video_ffmpeg(args.input_dir, output_path, args.fps)

    else:
        parser.print_help()
        print("\nError: --tracking-dir 또는 --input-dir 필요")


if __name__ == "__main__":
    main()
