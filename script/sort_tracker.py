#!/usr/bin/env python3
"""
SORT: Simple Online and Realtime Tracking
A simple and effective tracking algorithm using Kalman Filter and Hungarian Algorithm
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou_batch(bboxes1, bboxes2):
    """
    Compute IoU between two sets of bboxes in the format [x1, y1, x2, y2]

    Args:
        bboxes1: (N, 4) array
        bboxes2: (M, 4) array

    Returns:
        iou: (N, M) array of IoU scores
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    iou = wh / (area1 + area2 - wh)

    return iou


def convert_bbox_to_z(bbox):
    """
    Convert [x1, y1, x2, y2] to [cx, cy, s, r]
    where cx, cy is the center, s is the scale/area, and r is the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert [cx, cy, s, r] back to [x1, y1, x2, y2, score]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects using a Kalman filter.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        bbox: [x1, y1, x2, y2, confidence]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # 더 보수적인(작은) 잡음 및 공분산 설정
        self.kf.R[2:, 2:] *= 5.   # 관측 잡음(스케일, 종횡비) 기존보다 더 작게
        self.kf.R[:2, :2] *= 0.5  # 위치 관측 잡음도 더 작게
        self.kf.P[4:, 4:] *= 500. # 속도 관련 초기 공분산 감소
        self.kf.P *= 5.           # 전체적으로 더 작은 초기 공분산
        self.kf.Q[-1, -1] *= 0.005 # 종횡비 프로세스 잡음 더 감소
        self.kf.Q[4:, 4:] *= 0.005 # 속도 관련 프로세스 잡음 더 감소

        self.kf.x[:4] = convert_bbox_to_z(bbox[:4])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.confidence = bbox[4] if len(bbox) > 4 else 1.0

    def update(self, bbox):
        """
        Update the state with observed bbox.
        bbox: [x1, y1, x2, y2, confidence]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox[:4]))
        self.confidence = bbox[4] if len(bbox) > 4 else 1.0

    def predict(self):
        """
        Advance the state and return the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Return the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class Sort:
    """
    SORT: Simple Online and Realtime Tracking
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Args:
            max_age: Maximum number of frames to keep alive a track without associated detections
            min_hits: Minimum number of associated detections before track is confirmed
            iou_threshold: Minimum IoU for match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        Args:
            detections: numpy array of detections in format [[x1,y1,x2,y2,score], ...]

        Returns:
            numpy array of tracked objects in format [[x1,y1,x2,y2,track_id,score], ...]
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)

        # Return tracked objects
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.confidence])).reshape(1, -1))
            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assigns detections to tracked objects using the Hungarian algorithm

        Returns:
            matched_indices: array of matched [detection_idx, tracker_idx]
            unmatched_detections: array of unmatched detection indices
            unmatched_trackers: array of unmatched tracker indices
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Use Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack([row_ind, col_ind], axis=1)
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out matched with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
