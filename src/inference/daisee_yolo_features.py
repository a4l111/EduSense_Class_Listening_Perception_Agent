"""Feature builders shared by DAiSEE YOLO extraction and realtime inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.inference.daisee_lstm import FEATURE_DIM


def box_iou(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-6)


def _box_from_detection(det: dict[str, Any] | list[float] | np.ndarray) -> np.ndarray:
    if isinstance(det, dict):
        return np.asarray([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32)
    return np.asarray(det, dtype=np.float32)


def _confidence(det: dict[str, Any] | list[float] | np.ndarray) -> float:
    return float(det.get("confidence", 0.0)) if isinstance(det, dict) else 1.0


def keypoint_features(
    keypoints: list[list[float]] | np.ndarray,
    box: np.ndarray,
) -> tuple[list[float], list[float]]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.size == 0 or box[2] <= box[0] or box[3] <= box[1]:
        return [0.0] * 34, [0.0, 0.0, 0.0, 0.0]

    width = max(float(box[2] - box[0]), 1.0)
    height = max(float(box[3] - box[1]), 1.0)
    normalized = []
    for idx in range(17):
        if idx < len(keypoints):
            x, y = float(keypoints[idx][0]), float(keypoints[idx][1])
        else:
            x, y = 0.0, 0.0
        normalized.extend([
            float((x - box[0]) / width),
            float((y - box[1]) / height),
        ])

    nose_y = normalized[1] if len(normalized) > 1 else 0.0
    left_shoulder_y = normalized[11] if len(normalized) > 11 else 0.0
    right_shoulder_y = normalized[13] if len(normalized) > 13 else 0.0
    left_wrist_y = normalized[19] if len(normalized) > 19 else 0.0
    right_wrist_y = normalized[21] if len(normalized) > 21 else 0.0
    shoulder_y = float(np.mean([left_shoulder_y, right_shoulder_y]))
    raised_hand = float(left_wrist_y < shoulder_y or right_wrist_y < shoulder_y)
    head_down = float(nose_y > shoulder_y - 0.1)
    wrist_gap = abs(left_wrist_y - right_wrist_y)
    shoulder_tilt = abs(left_shoulder_y - right_shoulder_y)
    return normalized, [raised_hand, head_down, float(wrist_gap), float(shoulder_tilt)]


def build_feature_vector(
    frame_shape: tuple[int, int, int] | tuple[int, int],
    person_detection: dict[str, Any] | list[float] | np.ndarray,
    object_detections: list[dict[str, Any]] | None = None,
    pose_keypoints: list[list[float]] | np.ndarray | None = None,
) -> np.ndarray:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    person_box = _box_from_detection(person_detection)
    person_conf = _confidence(person_detection)
    object_detections = object_detections or []
    if pose_keypoints is None:
        pose_keypoints = []

    phone_box = None
    phone_conf = 0.0
    for det in object_detections:
        if int(det.get("class_id", -1)) == 67 and float(det.get("confidence", 0.0)) > phone_conf:
            phone_box = _box_from_detection(det)
            phone_conf = float(det.get("confidence", 0.0))

    has_person = person_conf > 0.0 and person_box[2] > person_box[0] and person_box[3] > person_box[1]
    cx = ((person_box[0] + person_box[2]) / 2 / max(width, 1)) if has_person else 0.0
    cy = ((person_box[1] + person_box[3]) / 2 / max(height, 1)) if has_person else 0.0
    bw = ((person_box[2] - person_box[0]) / max(width, 1)) if has_person else 0.0
    bh = ((person_box[3] - person_box[1]) / max(height, 1)) if has_person else 0.0
    phone_overlap = box_iou(person_box, phone_box) if phone_box is not None and has_person else 0.0
    kp, geom = keypoint_features(pose_keypoints, person_box)

    vector = [
        float(has_person),
        float(person_conf),
        float(cx),
        float(cy),
        float(bw),
        float(bh),
        float(phone_conf),
        float(phone_overlap),
        *geom,
        *kp,
    ]
    if len(vector) < FEATURE_DIM:
        vector.extend([0.0] * (FEATURE_DIM - len(vector)))
    return np.asarray(vector[:FEATURE_DIM], dtype=np.float32)
