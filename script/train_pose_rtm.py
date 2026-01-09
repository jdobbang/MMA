#!/usr/bin/env python3
"""
RTMPose Fine-tuning for MMA Dataset
====================================

COCO pretrained RTMPose-m 모델을 MMA 데이터셋으로 fine-tuning

Usage:
  # 기본 설정 (GPUs 자동 감지)
  python train_pose_rtm.py

  # 특정 GPU 사용
  python train_pose_rtm.py --gpu 0,1,2,3

  # 체크포인트에서 재개
  python train_pose_rtm.py --resume-from pose_log/mma_rtm_v1/epoch_10.pth

  # 평가만 수행
  python train_pose_rtm.py --eval-only --checkpoint pose_log/mma_rtm_v1/best_coco.pth
"""

import argparse
import os
import sys

# MMPose 임포트
try:
    from mmengine.config import Config, DictAction
    from mmengine.runner import Runner
    from mmengine.utils import digit_version
    import mmengine
except ImportError:
    print("Error: MMPose dependencies not installed!")
    print("Please run: pip install mmpose mmcv mmengine mmdet")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train RTMPose on MMA dataset')

    # 설정 파일
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rtmpose_configs/rtmpose_x_384x288.py',
        help='train config file path')

    # 체크포인트
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth',
        help='Pretrained checkpoint to load')

    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume training from checkpoint')

    # 출력 디렉토리
    parser.add_argument(
        '--work-dir',
        type=str,
        default='pose_log/mma_rtm_v2',
        help='the dir to save logs and models')

    # GPU 설정
    parser.add_argument(
        '--gpu',
        type=str,
        default=None,
        help='GPU ids to use. e.g., 0,1,2,3 (default: use all)')

    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of gpus')

    # 평가 옵션
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate model without training')

    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10,
        help='Evaluation interval (default: 10 epochs)')

    # 설정 오버라이드
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, e.g., '
        'model.backbone.depth=50 model.backbone.num_stages=4')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 설정 로드
    cfg = Config.fromfile(args.config)

    # 설정 오버라이드
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # 출력 디렉토리 설정
    cfg.work_dir = args.work_dir

    # GPU 설정
    if args.gpu is not None:
        cfg.launcher = 'pytorch'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 평가 인터벌 설정
    if 'val_interval' not in cfg.train_cfg:
        cfg.train_cfg.val_interval = args.eval_interval

    # Runner 생성
    runner = Runner.from_cfg(cfg)

    # 체크포인트 로드
    if args.checkpoint is not None:
        print(f"Loading pretrained checkpoint: {args.checkpoint}")
        runner.load_checkpoint(args.checkpoint)

    # 재개 (평가 후 이전 체크포인트에서)
    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
        runner.load_checkpoint(args.resume_from)

    # 평가만 수행
    if args.eval_only:
        print("Running evaluation only...")
        runner.validate()
    else:
        # 학습 실행
        print("Starting training...")
        runner.train()


if __name__ == '__main__':
    main()