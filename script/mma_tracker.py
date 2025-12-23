#!/usr/bin/env python3
"""
MMA Template Tracker
====================

MMA 2인 추적을 위한 하이브리드 트래커
- SORT의 Kalman Filter (위치 예측)
- Re-ID (외관 특징으로 ID 확정)
- 동적 가중치 전략 (IoU에 따라 가중치 조절)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
import cv2

# KalmanBoxTracker 재사용
from sort_tracker import KalmanBoxTracker, iou_batch, convert_x_to_bbox


@dataclass
class PlayerTemplate:
    """선수 템플릿 (ID 1 or 2)"""
    id: int  # 1 or 2

    # 초기 템플릿 (고정, ID 복구용)
    # - 분리된 프레임에서만 수집하여 평균
    # - 스왑 후에도 원래 ID 복구 가능
    initial_reid_feature: np.ndarray  # 512-dim, 분리 시 평균

    # 적응형 템플릿 (EMA 업데이트, 연속 추적용)
    # - 조명/자세 변화에 적응
    adaptive_reid_feature: np.ndarray  # 512-dim, EMA 업데이트

    # 초기 템플릿 수집용
    initial_feature_sum: np.ndarray = None  # 누적 합
    initial_feature_count: int = 0  # 수집된 프레임 수
    initial_collection_done: bool = False  # 수집 완료 여부 (첫 IoU 발생 시 True)

    # Kalman 상태 (매 프레임 업데이트)
    kalman_tracker: KalmanBoxTracker = None

    # 상태 추적
    last_bbox: List[float] = field(default_factory=list)  # 마지막 검출 bbox
    last_seen_frame: int = 0  # 마지막 검출 프레임
    hit_streak: int = 0  # 연속 검출 횟수
    time_since_update: int = 0  # 마지막 업데이트 이후 프레임


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """두 벡터 간 코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """두 bbox 간 IoU 계산"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


class MMATemplateTracker:
    """
    MMA 2인 추적을 위한 하이브리드 트래커

    핵심 전략:
    - 선수 2명 고정 (새 ID 생성 없음)
    - Kalman Filter로 위치 예측
    - Re-ID로 외관 기반 ID 확정
    - IoU에 따라 가중치 동적 조절
    """

    def __init__(self, device='cuda', reid_ema_alpha=0.1):
        """
        Args:
            device: Re-ID 모델 디바이스 ('cuda' or 'cpu')
            reid_ema_alpha: Re-ID 특징 EMA 업데이트 비율
        """
        self.device = device
        self.reid_ema_alpha = reid_ema_alpha

        # 선수 템플릿 (1, 2)
        self.templates: List[Optional[PlayerTemplate]] = [None, None]
        self.initialized = False
        self.frame_count = 0

        # Re-ID 모델 (lazy loading)
        self.reid_model = None
        self.reid_transform = None

        # 추적 파라미터
        self.max_age = 30  # 검출 없이 유지할 최대 프레임
        self.min_score_threshold = 0.3  # 최소 매칭 스코어

    def _load_reid_model(self):
        """OSNet Re-ID 모델 로드 (lazy loading)"""
        if self.reid_model is not None:
            return

        try:
            import torch
            import torchvision.transforms as T
            import torchreid

            print(f"Loading OSNet Re-ID model (device={self.device})...")

            self.reid_model = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=1000,
                pretrained=True,
                use_gpu=(self.device == 'cuda')
            )
            self.reid_model.eval()

            if self.device == 'cuda':
                self.reid_model = self.reid_model.cuda()

            self.reid_transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            print("OSNet model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "torchreid not installed. Please run: "
                "pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            ) from e

    def extract_reid_features(self, frame_img: np.ndarray, detections: np.ndarray) -> List[np.ndarray]:
        """
        검출된 bbox들에서 Re-ID 특징 추출

        Args:
            frame_img: BGR 이미지 (H, W, 3)
            detections: [[x1, y1, x2, y2, conf], ...]

        Returns:
            List of 512-dim feature vectors
        """
        import torch

        self._load_reid_model()

        if len(detections) == 0:
            return []

        features = []
        batch_tensors = []

        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_img.shape[1], x2), min(frame_img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                # 유효하지 않은 bbox - 0 벡터
                features.append(np.zeros(512))
                continue

            crop = frame_img[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.reid_transform(crop_rgb)
            batch_tensors.append(tensor)

        if len(batch_tensors) == 0:
            return features

        # 배치 처리
        with torch.no_grad():
            batch = torch.stack(batch_tensors)
            if self.device == 'cuda':
                batch = batch.cuda()
            batch_features = self.reid_model(batch)

        # 결과 정리
        feat_idx = 0
        result = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_img.shape[1], x2), min(frame_img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                result.append(np.zeros(512))
            else:
                result.append(batch_features[feat_idx].cpu().numpy())
                feat_idx += 1

        return result

    def initialize(self, frame_img: np.ndarray, detections: np.ndarray, frame_num: int = 0) -> bool:
        """
        첫 프레임에서 2명 검출로 템플릿 초기화

        Args:
            frame_img: BGR 이미지
            detections: [[x1, y1, x2, y2, conf], ...]
            frame_num: 현재 프레임 번호

        Returns:
            초기화 성공 여부
        """
        if len(detections) < 2:
            print(f"Frame {frame_num}: Need 2 detections for initialization, got {len(detections)}")
            return False

        # 상위 신뢰도 2개 선택
        if len(detections) > 2:
            indices = np.argsort(detections[:, 4])[::-1][:2]
            detections = detections[indices]

        # 두 검출 간 IoU 체크 (분리 상태 확인)
        det1_bbox = detections[0, :4]
        det2_bbox = detections[1, :4]
        det_iou = compute_iou(det1_bbox, det2_bbox)

        if det_iou > 0:
            print(f"Frame {frame_num}: Detections overlapping (IoU={det_iou:.2f}), waiting for separation")
            return False

        # Re-ID 특징 추출
        features = self.extract_reid_features(frame_img, detections)

        # 왼쪽/오른쪽으로 Player 1/2 구분
        centers = [(det[0] + det[2]) / 2 for det in detections]
        if centers[0] < centers[1]:
            order = [0, 1]  # 첫 번째가 왼쪽
        else:
            order = [1, 0]  # 두 번째가 왼쪽

        # 템플릿 생성
        for i, idx in enumerate(order):
            det = detections[idx]
            feat = features[idx]

            # Kalman tracker 생성
            kalman = KalmanBoxTracker(det)

            self.templates[i] = PlayerTemplate(
                id=i + 1,  # Player 1, 2
                initial_reid_feature=feat.copy(),  # 초기 템플릿 (고정)
                adaptive_reid_feature=feat.copy(),  # 적응형 템플릿 (EMA)
                initial_feature_sum=feat.copy(),  # 누적 합 초기화
                initial_feature_count=1,  # 수집 카운트
                kalman_tracker=kalman,
                last_bbox=det[:4].tolist(),
                last_seen_frame=frame_num,
                hit_streak=1,
                time_since_update=0
            )

        self.initialized = True
        self.frame_count = frame_num
        print(f"Frame {frame_num}: Initialized 2 player templates (separated, IoU=0)")
        return True

    def compute_dynamic_weights(self, iou_value: float) -> Dict[str, float]:
        """
        IoU 값에 따른 동적 가중치 계산

        Args:
            iou_value: 두 bbox 간 IoU

        Returns:
            {'iou': weight, 'reid_initial': weight, 'reid_adaptive': weight}
        """
        if iou_value == 0:
            # 떨어져 있음: 위치만으로 판단
            return {'iou': 0.0, 'reid_initial': 1.0, 'reid_adaptive': 0.0}
        elif iou_value < 0.5:
            # 약간 겹침: 균형
            return {'iou': 0.0, 'reid_initial': 1.0, 'reid_adaptive': 0.0}
        else:
            # 많이 겹침: Re-ID 중심 (클린치/그라운드)
            return {'iou': 0.5, 'reid_initial': 0.5, 'reid_adaptive': 0.0}

    def compute_cost_matrix(
        self,
        detections: np.ndarray,
        det_features: List[np.ndarray],
        predicted_bboxes: List[np.ndarray]
    ) -> np.ndarray:
        """
        2명 템플릿 vs N개 검출의 비용 행렬 계산

        Args:
            detections: [[x1, y1, x2, y2, conf], ...]
            det_features: 검출별 Re-ID 특징
            predicted_bboxes: Kalman 예측 bbox (2개)

        Returns:
            (2, N) cost matrix (음수 스코어, Hungarian 최소화용)
        """
        n_dets = len(detections)
        cost = np.zeros((2, n_dets))

        for i, template in enumerate(self.templates):
            if template is None:
                cost[i, :] = 1e6  # 초기화 안됨
                continue

            pred_bbox = predicted_bboxes[i].flatten()[:4]

            for j in range(n_dets):
                det_bbox = detections[j, :4]

                # IoU 계산 (예측 위치 vs 검출)
                iou = compute_iou(pred_bbox, det_bbox)

                
                # 동적 가중치 적용                
                weights = self.compute_dynamic_weights(iou)
                
                # - initial: ID 복구용 (스왑 후에도 원래 ID 찾기)
                # - adaptive: 연속 추적용 (조명/자세 변화 대응)
                initial_sim = cosine_similarity(template.initial_reid_feature, det_features[j])
                adaptive_sim = cosine_similarity(template.adaptive_reid_feature, det_features[j])

                # 하이브리드 스코어
                score =iou * weights['iou'] + initial_sim * weights['reid_initial'] + adaptive_sim * weights['reid_adaptive']  

                cost[i, j] = -score  # Hungarian은 최소화

        return cost

    def update(self, frame_img: np.ndarray, detections: np.ndarray, frame_num: int) -> np.ndarray:
        """
        매 프레임 업데이트

        Args:
            frame_img: BGR 이미지
            detections: [[x1, y1, x2, y2, conf], ...]
            frame_num: 현재 프레임 번호

        Returns:
            [[x1, y1, x2, y2, track_id, conf], ...] (최대 2개)
        """
        self.frame_count = frame_num

        # 초기화 안됨
        if not self.initialized:
            if self.initialize(frame_img, detections, frame_num):
                # 초기화 성공 - 현재 결과 반환
                return self._get_output()
            else:
                return np.empty((0, 6))

        # Step 1: Kalman predict
        predicted_bboxes = []
        for template in self.templates:
            if template is not None:
                pred = template.kalman_tracker.predict()
                predicted_bboxes.append(pred)
            else:
                predicted_bboxes.append(np.zeros((1, 4)))

        # 검출 없음
        if len(detections) == 0:
            for template in self.templates:
                if template is not None:
                    template.time_since_update += 1
                    template.hit_streak = 0
            return self._get_output()

        # Step 2: Re-ID 특징 추출
        det_features = self.extract_reid_features(frame_img, detections)

        # Step 3: 비용 행렬 계산
        cost_matrix = self.compute_cost_matrix(detections, det_features, predicted_bboxes)

        # Step 4: Hungarian assignment
        assignments = self._assign(cost_matrix, detections)

        # Step 5: 두 선수 간 IoU 계산 (분리 상태 확인)
        players_separated = False
        players_overlapping = False
        if assignments[0] is not None and assignments[1] is not None:
            det0_bbox = detections[assignments[0], :4]
            det1_bbox = detections[assignments[1], :4]
            players_iou = compute_iou(det0_bbox, det1_bbox)
            players_separated = (players_iou == 0)
            players_overlapping = (players_iou > 0)

        # Step 6: 첫 IoU 발생 시 초기 템플릿 수집 종료
        if players_overlapping:
            for template in self.templates:
                if template is not None and not template.initial_collection_done:
                    template.initial_collection_done = True
                    print(f"Frame {frame_num}: Initial template collection done for Player {template.id} "
                          f"({template.initial_feature_count} frames collected)")

        # Step 7: 템플릿 업데이트
        for player_idx, det_idx in assignments.items():
            template = self.templates[player_idx]
            if template is None:
                continue

            if det_idx is not None:
                det = detections[det_idx]
                feat = det_features[det_idx]

                # Kalman update
                template.kalman_tracker.update(det)

                # 적응형 템플릿 EMA 업데이트 (항상)
                template.adaptive_reid_feature = (
                    (1 - self.reid_ema_alpha) * template.adaptive_reid_feature +
                    self.reid_ema_alpha * feat
                )

                # 초기 템플릿 수집 (분리 상태 + 수집 완료 전에만)
                if players_separated and not template.initial_collection_done:
                    template.initial_feature_sum += feat
                    template.initial_feature_count += 1
                    # 초기 템플릿 = 누적 평균
                    template.initial_reid_feature = (
                        template.initial_feature_sum / template.initial_feature_count
                    )

                # 상태 업데이트
                template.last_bbox = det[:4].tolist()
                template.last_seen_frame = frame_num
                template.hit_streak += 1
                template.time_since_update = 0
            else:
                # 미검출
                template.time_since_update += 1
                template.hit_streak = 0

        return self._get_output()

    def _assign(self, cost_matrix: np.ndarray, detections: np.ndarray) -> Dict[int, Optional[int]]:
        """
        Hungarian algorithm으로 최적 할당

        Args:
            cost_matrix: (2, N) 비용 행렬
            detections: 검출 배열

        Returns:
            {player_idx: detection_idx or None}
        """
        n_dets = len(detections)
        assignments = {0: None, 1: None}

        if n_dets == 0:
            return assignments

        # 검출이 1개인 경우
        if n_dets == 1:
            # 더 낮은 비용(높은 스코어) 템플릿에 할당
            if cost_matrix[0, 0] < cost_matrix[1, 0]:
                # 임계값 체크
                if -cost_matrix[0, 0] > self.min_score_threshold:
                    assignments[0] = 0
            else:
                if -cost_matrix[1, 0] > self.min_score_threshold:
                    assignments[1] = 0
            return assignments

        # 검출이 2개 이상인 경우 - Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            # 임계값 체크
            if -cost_matrix[r, c] > self.min_score_threshold:
                assignments[r] = c

        return assignments

    def _get_output(self) -> np.ndarray:
        """
        현재 추적 결과 반환

        Returns:
            [[x1, y1, x2, y2, track_id, conf], ...]
        """
        results = []

        for template in self.templates:
            if template is None:
                continue

            # max_age 체크
            if template.time_since_update > self.max_age:
                continue

            # 현재 상태 가져오기
            bbox = template.kalman_tracker.get_state()[0].flatten()[:4]
            conf = template.kalman_tracker.confidence

            results.append([
                bbox[0], bbox[1], bbox[2], bbox[3],
                template.id,  # 1 or 2
                conf
            ])

        if len(results) == 0:
            return np.empty((0, 6))

        return np.array(results)

    def reset(self):
        """트래커 초기화"""
        self.templates = [None, None]
        self.initialized = False
        self.frame_count = 0
        # KalmanBoxTracker ID 카운터 리셋
        KalmanBoxTracker.count = 0


# ============================================================================
# Utility Functions
# ============================================================================

def interpolate_track(track_data: List[Tuple[int, List[float], float]], max_gap: int = 30) -> List[Tuple[int, List[float], float]]:
    """
    트랙 내 빈 프레임 보간

    Args:
        track_data: [(frame, [x1,y1,x2,y2], conf), ...]
        max_gap: 보간할 최대 간격

    Returns:
        보간된 트랙 데이터
    """
    if len(track_data) <= 1:
        return track_data

    # 프레임 번호로 정렬
    sorted_data = sorted(track_data, key=lambda x: x[0])
    result = []

    for i in range(len(sorted_data)):
        result.append(sorted_data[i])

        if i < len(sorted_data) - 1:
            curr_frame = sorted_data[i][0]
            next_frame = sorted_data[i + 1][0]
            gap = next_frame - curr_frame - 1

            if 0 < gap <= max_gap:
                curr_bbox = np.array(sorted_data[i][1])
                next_bbox = np.array(sorted_data[i + 1][1])
                curr_conf = sorted_data[i][2]
                next_conf = sorted_data[i + 1][2]

                # 선형 보간
                for t in range(1, gap + 1):
                    alpha = t / (gap + 1)
                    interp_bbox = (1 - alpha) * curr_bbox + alpha * next_bbox
                    interp_conf = (1 - alpha) * curr_conf + alpha * next_conf
                    result.append((curr_frame + t, interp_bbox.tolist(), interp_conf))

    return result
