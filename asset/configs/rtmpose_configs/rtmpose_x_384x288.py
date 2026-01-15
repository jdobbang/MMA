# configs/rtmpose_configs/rtmpose_x_384x288.py

# mmpose 기본 설정 로드
_base_ = ['mmpose::_base_/default_runtime.py']

# [필수] mmdet 모델 레지스트리를 강제로 로드하여 KeyError 방지
custom_imports = dict(
    imports=['mmdet.models'],
    allow_failed_imports=False
)

# 기본 설정값
num_keypoints = 17
input_size = (288, 384) # (width, height)

# 런타임 설정 (훈련 시 필요한 파라미터)
max_epochs = 50
stage2_num_epochs = 20
base_lr = 2e-3
train_batch_size = 8
val_batch_size = 4

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# 최적화 및 스케줄러 (3090 Ti는 연산이 빨라 AdamW가 효율적임)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# 코덱 설정 (SimCC 좌표 분류 방식)
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# ------------------------------------------------------------
# 모델 정의 (핵심 부분)
# ------------------------------------------------------------
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',       # mmdet 레지스트리에서 찾도록 명시
        type='CSPNeXt',        # 최신 RTMPose의 표준 백본
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.33,    # RTMPose-x 급의 깊이
        widen_factor=1.25,     # RTMPose-x 급의 너비
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    head=dict(
        type='RTMCCHead',
        in_channels=1280,      # CSPNeXt-x 출력 채널 수
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# 데이터 로더 및 파이프라인
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'dataset/yolo_pose_topdown_dataset/' # 경로: images/train, images/val 포함

# ============================================================
# 학습용 파이프라인 (Train Pipeline)
# ============================================================
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# ============================================================
# 검증용 파이프라인 (Val Pipeline)
# ============================================================
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# ============================================================
# 학습 데이터로더
# ============================================================
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='labels/annotations_train.json',
        pipeline=train_pipeline,
    ))

# ============================================================
# 검증 데이터로더
# ============================================================
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='labels/annotations_val.json',
        test_mode=True,
        pipeline=val_pipeline,
    ))

# 테스트 데이터로더 (검증용과 동일)
test_dataloader = val_dataloader

# ============================================================
# 평가 지표
# ============================================================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'labels/annotations_val.json')
test_evaluator = val_evaluator

# ============================================================
# 체크포인트 저장 및 모니터링 설정
# ============================================================
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=3,
        interval=10)
)