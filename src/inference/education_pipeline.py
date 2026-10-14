"""Education-scene vision pipeline.

This module keeps the education-specific perception logic separate from the
existing offloading benchmark pipeline.  It supports optional ONNX models for:

- face detection
- expression classification
- head-pose estimation
- gaze estimation

When a model path is not supplied, the stage falls back to lightweight
geometry-based estimates so the gRPC server can still run during integration.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
from loguru import logger

EXPRESSION_LABELS = ["focused", "confused", "tired", "distracted", "neutral"]
BEHAVIOR_LABELS = ["writing", "reading", "using_phone", "raising_hand", "left_seat", "unknown"]
STATE_LABELS = ["focused", "confused", "tired", "distracted", "absent"]


@dataclass
class FaceSignal:
    present: bool = False
    bbox: list[float] = field(default_factory=list)
    confidence: float = 0.0
    expression: str = "neutral"
    expression_confidence: float = 0.0
    eyes_closed: bool = False
    eye_closed_confidence: float = 0.0


@dataclass
class HeadGazeSignal:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_region: str = "unknown"
    confidence: float = 0.0


@dataclass
class BehaviorSignal:
    label: str = "unknown"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class LearningState:
    label: str = "absent"
    confidence: float = 0.0
    intervention: str = "none"
    reasons: list[str] = field(default_factory=list)


@dataclass
class EducationAnalysis:
    face: FaceSignal = field(default_factory=FaceSignal)
    head_gaze: HeadGazeSignal = field(default_factory=HeadGazeSignal)
    behavior: BehaviorSignal = field(default_factory=BehaviorSignal)
    learning_state: LearningState = field(default_factory=LearningState)
    timings_ms: dict[str, float] = field(default_factory=dict)
    num_people_detected: int = 0
    analysis_mode: str = "single"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StudentAnalysis:
    track_id: int
    person_box: list[float] = field(default_factory=list)
    face: FaceSignal = field(default_factory=FaceSignal)
    head_gaze: HeadGazeSignal = field(default_factory=HeadGazeSignal)
    behavior: BehaviorSignal = field(default_factory=BehaviorSignal)
    learning_state: LearningState = field(default_factory=LearningState)
    priority_score: float = 0.0
    needs_attention: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "person_box": self.person_box,
            "face": asdict(self.face),
            "head_gaze": asdict(self.head_gaze),
            "behavior": asdict(self.behavior),
            "learning_state": asdict(self.learning_state),
            "priority_score": self.priority_score,
            "needs_attention": self.needs_attention,
        }


@dataclass
class MultiStudentAnalysis:
    students: list[StudentAnalysis] = field(default_factory=list)
    total_count: int = 0
    attention_needed_count: int = 0
    focused_count: int = 0
    summary: str = ""
    priority_feedback: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "students": [s.to_dict() for s in self.students],
            "total_count": self.total_count,
            "attention_needed_count": self.attention_needed_count,
            "focused_count": self.focused_count,
            "summary": self.summary,
            "priority_feedback": self.priority_feedback,
            "timings_ms": self.timings_ms,
        }


class EducationVisionPipeline:
    """Optional ONNX-backed education perception pipeline."""

    def __init__(
        self,
        ort_module: Any,
        providers: list[str],
        use_gpu: bool = False,
        face_model: str | None = None,
        expression_model: str | None = None,
        head_pose_model: str | None = None,
        gaze_model: str | None = None,
        learning_state_model: str | None = None,
        face_landmarker_model: str | None = None,
    ) -> None:
        self._ort = ort_module
        self._providers = providers
        self._use_gpu = use_gpu
        self._ov_core = None
        self._face = self._load_model(face_model, "face detector")
        self._expression = self._load_model(expression_model, "expression classifier")
        self._head_pose = self._load_model(head_pose_model, "head-pose estimator")
        self._gaze = self._load_model(gaze_model, "gaze estimator")
        self._learning_state = None
        self._face_landmarker = self._init_face_landmarker(face_landmarker_model)
        self._student_temporal: dict[int, dict[str, int]] = {}
        self._closed_eye_threshold = int(os.getenv("TIRED_CLOSED_EYE_FRAMES", "8"))
        self._phone_threshold = int(os.getenv("DISTRACTED_PHONE_FRAMES", "2"))
        self._not_frontal_threshold = int(os.getenv("DISTRACTED_NOT_FRONTAL_FRAMES", "3"))

    def _init_face_landmarker(self, model_path: str | None) -> Any:
        """Initialize MediaPipe FaceLandmarker for eye-closure (EAR) detection."""
        if not model_path:
            logger.info("Face landmarker (MediaPipe EAR): disabled")
            return None
        abs_path = os.path.abspath(model_path)
        if not os.path.exists(abs_path):
            logger.warning(f"Face landmarker: model not found at {abs_path}")
            return None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(model_asset_path=abs_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
            )
            landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            logger.info(f"Face landmarker (MediaPipe EAR): loaded from {abs_path}")
            return landmarker
        except Exception as e:
            logger.warning(f"Face landmarker init failed: {e}; using texture fallback")
            return None

    def _load_learning_state_model(self, model_path: str | None):
        logger.info("Learning-state classifier: rule fusion mode")
        return None

    def _load_model(self, model_path: str | None, label: str, default_path: str | None = None):
        selected_path = model_path or default_path
        if not selected_path:
            logger.info(f"Education {label}: disabled, using fallback logic")
            return None
        abs_path = os.path.abspath(selected_path)
        if not os.path.exists(abs_path):
            logger.warning(f"Education {label}: model not found at {abs_path}; using fallback")
            return None
        if abs_path.lower().endswith(".xml"):
            try:
                try:
                    from openvino import Core
                except ImportError:
                    from openvino.runtime import Core
            except ImportError:
                logger.warning(f"Education {label}: OpenVINO is not installed; using fallback")
                return None
            if self._ov_core is None:
                self._ov_core = Core()
            ov_device = "CPU"
            if self._use_gpu:
                try:
                    available_devices = self._ov_core.available_devices
                    if "GPU" in available_devices:
                        ov_device = "GPU"
                except Exception:
                    pass
            logger.info(f"Loading education {label} (OpenVINO, device={ov_device}): {abs_path}")
            return {
                "kind": "openvino",
                "model": self._ov_core.compile_model(abs_path, ov_device),
            }
        logger.info(f"Loading education {label} (ONNX Runtime): {abs_path}")
        return {
            "kind": "onnx",
            "model": self._ort.InferenceSession(abs_path, providers=self._providers),
        }

    def analyze(
        self,
        frame: np.ndarray,
        person_detections: list[dict],
        object_detections: list[dict] | None = None,
        pose_keypoints: list[list[list[float]]] | None = None,
    ) -> EducationAnalysis:
        result = EducationAnalysis()
        result.num_people_detected = len(person_detections)
        object_detections = object_detections or []
        pose_keypoints = pose_keypoints or []

        t0 = time.perf_counter()
        face = self._analyze_face(frame, person_detections)
        primary_pose = pose_keypoints[0] if pose_keypoints else []
        if face.present:
            face.eyes_closed, face.eye_closed_confidence = self._estimate_eyes_closed(
                frame, face.bbox, primary_pose
            )
        result.timings_ms["face_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        head_gaze = self._analyze_head_gaze(frame, face)
        result.timings_ms["head_gaze_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        behavior = self._analyze_behavior(
            frame,
            person_detections,
            object_detections,
            head_gaze,
            pose_keypoints,
        )
        result.timings_ms["behavior_ms"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        state = self._infer_learning_state_for_person(
            face,
            head_gaze,
            behavior,
            has_person=bool(person_detections),
            track_id=0,
        )
        result.timings_ms["state_ms"] = (time.perf_counter() - t3) * 1000

        result.face = face
        result.head_gaze = head_gaze
        result.behavior = behavior
        result.learning_state = state
        result.analysis_mode = "single" if len(person_detections) <= 1 else "multi"
        result.timings_ms["total_ms"] = sum(result.timings_ms.values())
        return result

    def analyze_smart(
        self,
        frame: np.ndarray,
        person_detections: list[dict],
        object_detections: list[dict] | None = None,
        pose_keypoints: list[list[list[float]]] | None = None,
    ) -> MultiStudentAnalysis:
        """智能多学生分析，自动检测人数并优先反馈问题学生。

        Args:
            frame: 输入帧
            person_detections: 检测到的所有人
            object_detections: 检测到的物体
            pose_keypoints: 姿态关键点

        Returns:
            MultiStudentAnalysis: 包含所有学生的分析结果，按优先级排序
        """
        result = MultiStudentAnalysis()
        object_detections = object_detections or []
        pose_keypoints = pose_keypoints or []

        total_start = time.perf_counter()
        num_people = len(person_detections)
        result.total_count = num_people
        t0 = time.perf_counter()
        all_faces = self._detect_all_faces(frame, person_detections)
        result.timings_ms["face_detection_ms"] = (time.perf_counter() - t0) * 1000

        for idx, person_det in enumerate(person_detections):
            student = StudentAnalysis(track_id=idx)
            student.person_box = [
                person_det["x1"], person_det["y1"],
                person_det["x2"], person_det["y2"]
            ]

            face_signal = all_faces[idx] if idx < len(all_faces) else FaceSignal()
            pose = pose_keypoints[idx] if idx < len(pose_keypoints) else []
            if face_signal.present:
                face_signal.expression, face_signal.expression_confidence = \
                    self._classify_expression(frame, face_signal.bbox)
                face_signal.eyes_closed, face_signal.eye_closed_confidence = \
                    self._estimate_eyes_closed(frame, face_signal.bbox, pose)
            student.face = face_signal

            t1 = time.perf_counter()
            student.head_gaze = self._analyze_head_gaze_for_person(
                frame, face_signal, student.person_box, pose
            )
            result.timings_ms[f"head_gaze_{idx}_ms"] = (time.perf_counter() - t1) * 1000

            student.behavior = self._analyze_behavior_for_person(
                frame, person_det, object_detections, student.head_gaze, pose
            )

            student.learning_state = self._infer_learning_state_for_person(
                student.face,
                student.head_gaze,
                student.behavior,
                has_person=True,
                track_id=idx,
            )

            student.priority_score = self._calculate_priority_score(student)
            student.needs_attention = student.learning_state.label in [
                "distracted", "confused", "tired", "absent"
            ]

            result.students.append(student)

        result.timings_ms["total_analysis_ms"] = (time.perf_counter() - total_start) * 1000

        result.students = self._sort_students_by_priority(result.students)
        result.priority_feedback = self._generate_priority_feedback(result)
        result.summary = self._generate_summary(result)

        result.focused_count = sum(
            1 for s in result.students if s.learning_state.label == "focused"
        )
        result.attention_needed_count = sum(
            1 for s in result.students if s.needs_attention
        )

        return result

    def _detect_all_faces(
        self, frame: np.ndarray, person_detections: list[dict]
    ) -> list[FaceSignal]:
        """检测所有人的脸。"""
        all_faces = []
        detected_faces = self._detect_faces_openvino(frame) if self._face else []

        for idx, person_det in enumerate(person_detections):
            person_box = [person_det["x1"], person_det["y1"],
                         person_det["x2"], person_det["y2"]]

            matched_face = None
            for face_det in detected_faces:
                if self._is_face_in_person_box(face_det, person_box):
                    matched_face = FaceSignal(
                        present=True,
                        bbox=face_det["bbox"],
                        confidence=face_det["confidence"]
                    )
                    break

            if matched_face is None:
                matched_face = self._estimate_face_from_single_box(person_box, person_det["confidence"])

            all_faces.append(matched_face)

        return all_faces

    def _detect_faces_openvino(self, frame: np.ndarray) -> list[dict]:
        """使用OpenVINO检测所有人脸。"""
        if self._face is None or self._face["kind"] != "openvino":
            return []

        input_layer = self._face["model"].input(0)
        _, _, in_h, in_w = input_layer.shape
        h_orig, w_orig = frame.shape[:2]
        inp = cv2.resize(frame, (in_w, in_h)).astype(np.float32)
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]
        output = next(iter(self._face["model"]({input_layer.any_name: inp}).values()))

        detections = []
        for det in np.asarray(output).reshape(-1, 7):
            conf = float(det[2])
            if conf < 0.20:
                continue
            detections.append({
                "bbox": [
                    float(det[3] * w_orig), float(det[4] * h_orig),
                    float(det[5] * w_orig), float(det[6] * h_orig)
                ],
                "confidence": conf,
            })
        return detections

    @staticmethod
    def _is_face_in_person_box(face_det: dict, person_box: list[float],
                               iou_threshold: float = 0.3) -> bool:
        """检查人脸是否在人体框内。"""
        fx1, fy1, fx2, fy2 = face_det["bbox"]
        px1, py1, px2, py2 = person_box

        inter_x1 = max(fx1, px1)
        inter_y1 = max(fy1, py1)
        inter_x2 = min(fx2, px2)
        inter_y2 = min(fy2, py2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        face_area = (fx2 - fx1) * (fy2 - fy1)
        iou = inter_area / (face_area + 1e-6)

        return iou > iou_threshold

    @staticmethod
    def _estimate_face_from_single_box(person_box: list[float],
                                      confidence: float) -> FaceSignal:
        """从单个人体框估计人脸位置。"""
        x1, y1, x2, y2 = person_box
        width = x2 - x1
        height = y2 - y1
        face_box = [x1 + width * 0.28, y1 + height * 0.04,
                   x1 + width * 0.72, y1 + height * 0.34]
        return FaceSignal(present=True, bbox=face_box, confidence=confidence * 0.55)

    def _analyze_head_gaze_for_person(self, frame: np.ndarray,
                                     face: FaceSignal,
                                     person_box: list[float],
                                     pose_keypoints: list[list[float]] | None = None) -> HeadGazeSignal:
        """为单个人分析头部姿态和视线。"""
        # 优先用关键点估算 pitch（低头检测最可靠）
        kp_pitch = self._estimate_head_pitch_from_keypoints(pose_keypoints)

        if not face.present:
            yaw, pitch, roll = self._estimate_head_pose_from_face_box(frame, person_box)
        else:
            crop = self._crop(frame, face.bbox)
            model_angles = self._predict_head_pose(crop)
            if model_angles is not None:
                yaw, pitch, roll = model_angles
            else:
                yaw, pitch, roll = self._estimate_head_pose_from_face_box(frame, face.bbox)

        # 关键点估算的 pitch 更准确（几何估算容易低估低头角度）
        if kp_pitch is not None:
            pitch = kp_pitch

        gaze_region, gaze_conf = self._predict_gaze_region(
            frame, face.bbox if face.present else person_box,
            self._crop(frame, face.bbox if face.present else person_box),
            yaw, pitch, roll
        )

        return HeadGazeSignal(
            yaw=float(yaw), pitch=float(pitch), roll=float(roll),
            gaze_region=gaze_region, confidence=gaze_conf
        )

    def _analyze_behavior_for_person(self, frame: np.ndarray,
                                    person_det: dict,
                                    object_detections: list[dict],
                                    head_gaze: HeadGazeSignal,
                                    pose_keypoints: list[list[float]]) -> BehaviorSignal:
        """为单个人分析行为。"""
        behavior_label = "unknown"
        confidence = 0.35
        evidence = []

        if self._has_phone_near_person(person_det, object_detections, pose_keypoints):
            return BehaviorSignal("using_phone", 0.82, ["cell_phone_near_person"])
        if self._has_coco_object(object_detections, class_ids={67}, min_confidence=0.08):
            return BehaviorSignal("using_phone", 0.68, ["cell_phone_visible"])

        if self._has_raised_hand([pose_keypoints]) if pose_keypoints else False:
            return BehaviorSignal("raising_hand", 0.70, ["movenet_wrist_above_shoulder"])

        if self._has_coco_object(object_detections, class_ids={73}):
            behavior_label = "reading"
            confidence = 0.65
            evidence.append("book_detected")

        if head_gaze.gaze_region == "book":
            behavior_label = "writing" if behavior_label == "unknown" else behavior_label
            confidence = max(confidence, 0.55)
            evidence.append("downward_gaze")

        if behavior_label == "unknown" and head_gaze.gaze_region == "screen":
            behavior_label = "reading"
            confidence = 0.45
            evidence.append("screen_gaze")

        return BehaviorSignal(label=behavior_label, confidence=confidence, evidence=evidence)

    @staticmethod
    def _calculate_iou(box1: dict, boxes: list[dict]) -> dict:
        """计算一个框与多个框的IoU。"""
        ious = {}
        x1_1, y1_1, x2_1, y2_1 = box1["x1"], box1["y1"], box1["x2"], box1["y2"]

        for idx, box2 in enumerate(boxes):
            x1_2, y1_2, x2_2, y2_2 = box2["x1"], box2["y1"], box2["x2"], box2["y2"]

            inter_x1 = max(x1_1, x1_2)
            inter_y1 = max(y1_1, y1_2)
            inter_x2 = min(x2_1, x2_2)
            inter_y2 = min(y2_1, y2_2)

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                ious[idx] = 0.0
                continue

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = box1_area + box2_area - inter_area

            ious[idx] = inter_area / (union_area + 1e-6)

        return ious

    @staticmethod
    def _has_phone_near_person(
        person_det: dict,
        object_detections: list[dict],
        pose_keypoints: list[list[float]] | None = None,
    ) -> bool:
        phones = [
            det for det in object_detections
            if int(det.get("class_id", -1)) == 67 and float(det.get("confidence", 0.0)) >= 0.08
        ]
        if not phones:
            return False

        px1, py1, px2, py2 = (
            float(person_det["x1"]), float(person_det["y1"]),
            float(person_det["x2"]), float(person_det["y2"]),
        )
        pw = max(px2 - px1, 1.0)
        ph = max(py2 - py1, 1.0)
        expanded = [
            px1 - 0.10 * pw, py1 - 0.05 * ph,
            px2 + 0.10 * pw, py2 + 0.08 * ph,
        ]

        wrists = []
        if pose_keypoints and len(pose_keypoints) > 10:
            for idx in (9, 10):
                kp = pose_keypoints[idx]
                if len(kp) >= 2 and (len(kp) < 3 or float(kp[2]) >= 0.15):
                    wrists.append((float(kp[0]), float(kp[1])))

        for phone in phones:
            fx1, fy1, fx2, fy2 = (
                float(phone["x1"]), float(phone["y1"]),
                float(phone["x2"]), float(phone["y2"]),
            )
            fcx = (fx1 + fx2) / 2.0
            fcy = (fy1 + fy2) / 2.0
            if expanded[0] <= fcx <= expanded[2] and expanded[1] <= fcy <= expanded[3]:
                return True

            phone_iou = EducationVisionPipeline._box_iou(
                [px1, py1, px2, py2], [fx1, fy1, fx2, fy2]
            )
            if phone_iou >= 0.015:
                return True

            if wrists:
                phone_diag = max(((fx2 - fx1) ** 2 + (fy2 - fy1) ** 2) ** 0.5, 1.0)
                threshold = max(45.0, phone_diag * 2.2, min(pw, ph) * 0.18)
                for wx, wy in wrists:
                    if ((wx - fcx) ** 2 + (wy - fcy) ** 2) ** 0.5 <= threshold:
                        return True
        return False

    @staticmethod
    def _box_iou(a: list[float], b: list[float]) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        return inter / max(area_a + area_b - inter, 1e-6)

    def _infer_learning_state_for_person(
        self,
        face: FaceSignal,
        head_gaze: HeadGazeSignal,
        behavior: BehaviorSignal,
        has_person: bool = True,
        track_id: int = 0,
    ) -> LearningState:
        """为单个人推断学习状态。"""
        temporal = self._update_temporal_state(track_id, face, head_gaze, behavior, has_person)
        if not has_person or behavior.label == "left_seat":
            return LearningState(
                "absent",
                0.95,
                "notify_absence",
                ["no_person_detected" if not has_person else "left_seat"],
            )

        if temporal["phone"] >= self._phone_threshold:
            return LearningState(
                "distracted",
                0.88,
                "gentle_reminder",
                [f"phone_visible_{temporal['phone']}_frames"],
            )

        if temporal["closed_eye"] >= self._closed_eye_threshold:
            return LearningState(
                "tired",
                0.88,
                "suggest_break",
                [f"closed_eyes_{temporal['closed_eye']}_frames"],
            )

        if temporal["not_frontal"] >= self._not_frontal_threshold:
            return LearningState(
                "distracted",
                0.78,
                "redirect_attention",
                [f"not_frontal_{temporal['not_frontal']}_frames"],
            )

        if behavior.label == "using_phone":
            return LearningState("distracted", 0.82, "gentle_reminder", ["using_phone"])

        if behavior.label == "raising_hand":
            return LearningState("confused", 0.68, "offer_help", ["raising_hand"])

        if head_gaze.gaze_region == "phone":
            return LearningState("distracted", 0.78, "redirect_attention", ["gaze_phone"])

        if face.expression == "tired":
            return LearningState("tired", 0.66, "suggest_break", ["tired_expression"])

        if face.expression == "confused":
            return LearningState("confused", 0.64, "offer_hint", ["confused_expression"])

        if head_gaze.gaze_region == "away":
            return LearningState("distracted", 0.70, "redirect_attention", ["gaze_away"])

        if abs(head_gaze.yaw) > 20:
            return LearningState("distracted", 0.66, "redirect_attention", ["head_turned"])

        if head_gaze.pitch > 38:
            return LearningState("tired", 0.74, "suggest_break", ["head_drooping_severe"])
        if head_gaze.pitch > 25 and behavior.label not in {"writing", "reading"}:
            return LearningState("tired", 0.68, "suggest_break", ["head_drooping"])

        reasons = []
        if face.present:
            reasons.append("face_present")
        if head_gaze.gaze_region in {"screen", "book"}:
            reasons.append(f"gaze_{head_gaze.gaze_region}")
        if behavior.label in {"reading", "writing"}:
            reasons.append(behavior.label)
        if face.expression == "neutral":
            reasons.append("neutral_expression")
        if face.eyes_closed:
            reasons.append(f"eyes_closed_pending_{temporal['closed_eye']}/{self._closed_eye_threshold}")

        confidence = 0.72 if reasons else 0.50
        return LearningState("focused", confidence, "none", reasons)

    @staticmethod
    def _calculate_priority_score(student: StudentAnalysis) -> float:
        """计算学生需要关注程度的优先级分数。

        分数越高表示越需要老师关注。
        """
        state = student.learning_state.label
        confidence = student.learning_state.confidence

        priority_weights = {
            "absent": 100.0,
            "distracted": 80.0,
            "confused": 60.0,
            "tired": 40.0,
            "focused": 0.0,
        }

        base_priority = priority_weights.get(state, 50.0)
        confidence_bonus = confidence * 10.0

        if student.behavior.label == "using_phone":
            base_priority += 30.0
        if student.head_gaze.gaze_region == "away":
            base_priority += 20.0

        return base_priority + confidence_bonus

    @staticmethod
    def _sort_students_by_priority(students: list[StudentAnalysis]) -> list[StudentAnalysis]:
        """按优先级排序，问题学生排在前面。"""
        return sorted(students, key=lambda s: s.priority_score, reverse=True)

    def _generate_priority_feedback(self, analysis: MultiStudentAnalysis) -> list[str]:
        """生成需要优先反馈的学生列表。"""
        feedback = []

        attention_students = [s for s in analysis.students if s.needs_attention]

        for student in attention_students:
            state = student.learning_state
            intervention = state.intervention

            if intervention == "gentle_reminder":
                feedback.append(
                    f"Student #{student.track_id}: {state.label.upper()} - "
                    f"Consider sending a gentle reminder"
                )
            elif intervention == "redirect_attention":
                feedback.append(
                    f"Student #{student.track_id}: Looking AWAY - "
                    f"Try to redirect attention to the lesson"
                )
            elif intervention == "offer_help":
                feedback.append(
                    f"Student #{student.track_id}: RAISING HAND - "
                    f"Needs teacher's help"
                )
            elif intervention == "offer_hint":
                feedback.append(
                    f"Student #{student.track_id}: CONFUSED expression - "
                    f"May need a hint"
                )
            elif intervention == "suggest_break":
                feedback.append(
                    f"Student #{student.track_id}: TIRED - "
                    f"Consider suggesting a short break"
                )
            elif intervention == "notify_absence":
                feedback.append(
                    f"Student #{student.track_id}: ABSENT - "
                    f"Student left the seat"
                )

        return feedback

    def _generate_summary(self, analysis: MultiStudentAnalysis) -> str:
        """生成整体分析摘要。"""
        total = analysis.total_count
        focused = analysis.focused_count
        attention = analysis.attention_needed_count

        if total == 0:
            return "No students detected in the frame."

        if total == 1:
            student = analysis.students[0]
            state = student.learning_state.label.capitalize()
            return f"Single student detected: {state} (confidence: {student.learning_state.confidence:.2f})"

        focus_rate = (focused / total) * 100
        summary = f"Total: {total} students | Focused: {focused} ({focus_rate:.0f}%)"
        if attention > 0:
            summary += f" | Needs Attention: {attention}"
        return summary

    def _analyze_face(self, frame: np.ndarray, person_detections: list[dict]) -> FaceSignal:
        face = self._detect_face_with_model(frame)
        if face.present:
            face.expression, face.expression_confidence = self._classify_expression(frame, face.bbox)
            return face

        face = self._estimate_face_from_person_box(person_detections)
        if face.present:
            face.expression, face.expression_confidence = self._classify_expression(frame, face.bbox)
        return face

    def _detect_face_with_model(self, frame: np.ndarray) -> FaceSignal:
        if self._face is None:
            return FaceSignal()
        if self._face["kind"] == "openvino":
            boxes = self._detect_face_openvino(frame, self._face["model"])
        else:
            session = self._face["model"]
            inp = self._preprocess_image(frame, (640, 640))
            outputs = session.run(None, {session.get_inputs()[0].name: inp})
            boxes = self._postprocess_yolo_like(outputs[0], frame.shape[:2], conf_threshold=0.35)
        if not boxes:
            return FaceSignal()
        best = max(boxes, key=lambda item: item["confidence"])
        return FaceSignal(
            present=True,
            bbox=[best["x1"], best["y1"], best["x2"], best["y2"]],
            confidence=best["confidence"],
        )

    @staticmethod
    def _detect_face_openvino(frame: np.ndarray, compiled_model: Any) -> list[dict]:
        input_layer = compiled_model.input(0)
        _, _, in_h, in_w = input_layer.shape
        h_orig, w_orig = frame.shape[:2]
        inp = cv2.resize(frame, (in_w, in_h)).astype(np.float32)
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]
        output = next(iter(compiled_model({input_layer.any_name: inp}).values()))
        detections = []
        for det in np.asarray(output).reshape(-1, 7):
            conf = float(det[2])
            if conf < 0.35:
                continue
            detections.append({
                "x1": float(det[3] * w_orig),
                "y1": float(det[4] * h_orig),
                "x2": float(det[5] * w_orig),
                "y2": float(det[6] * h_orig),
                "confidence": conf,
                "class_id": 0,
            })
        return detections

    @staticmethod
    def _estimate_face_from_person_box(person_detections: list[dict]) -> FaceSignal:
        if not person_detections:
            return FaceSignal()
        best = max(person_detections, key=lambda item: item["confidence"])
        x1, y1, x2, y2 = best["x1"], best["y1"], best["x2"], best["y2"]
        width = x2 - x1
        height = y2 - y1
        face_box = [x1 + width * 0.28, y1 + height * 0.04, x1 + width * 0.72, y1 + height * 0.34]
        return FaceSignal(present=True, bbox=face_box, confidence=best["confidence"] * 0.55)

    def _classify_expression(self, frame: np.ndarray, bbox: list[float]) -> tuple[str, float]:
        crop = self._crop(frame, bbox)
        if crop.size == 0:
            return "focused", 0.0
        if self._expression is None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            contrast = float(gray.std())
            if brightness < 35:
                return "tired", 0.30
            if contrast < 22:
                return "focused", 0.35
            return "focused", 0.40

        if self._expression["kind"] == "openvino":
            label, conf = self._classify_expression_openvino(crop, self._expression["model"])
            return label, conf

        session = self._expression["model"]
        inp = self._preprocess_image(crop, self._input_hw(session, default=(224, 224)))
        logits = session.run(None, {session.get_inputs()[0].name: inp})[0]
        probs = self._softmax(np.asarray(logits).reshape(-1))
        idx = int(np.argmax(probs[: len(EXPRESSION_LABELS)]))
        return EXPRESSION_LABELS[idx], float(probs[idx])

    def _estimate_eyes_closed_mediapipe(
        self,
        frame: np.ndarray,
        face_bbox: list[float],
    ) -> tuple[bool, float] | None:
        """Detect eye closure via MediaPipe FaceLandmarker + EAR.

        Returns (eyes_closed, confidence) or None when detection fails.
        """
        if self._face_landmarker is None:
            return None
        crop = self._crop(frame, face_bbox)
        if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
            return None
        try:
            import mediapipe as mp

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
            result = self._face_landmarker.detect(mp_image)
            if not result.face_landmarks:
                return None
            landmarks = result.face_landmarks[0]
            # MediaPipe 468-point landmark indices for EAR
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            left_ear = self._calculate_ear(landmarks, left_eye_idx)
            right_ear = self._calculate_ear(landmarks, right_eye_idx)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_threshold = float(os.getenv("EAR_CLOSED_THRESHOLD", "0.20"))
            is_closed = avg_ear < ear_threshold
            confidence = max(0.0, min(1.0, (0.30 - avg_ear) / 0.15))
            return is_closed, confidence
        except Exception:
            return None

    @staticmethod
    def _calculate_ear(landmarks: list, indices: list[int]) -> float:
        """Calculate Eye Aspect Ratio from 6 normalized landmarks.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        where p1..p6 are the 6 eye landmarks clockwise from outer corner.
        """
        pts = [(landmarks[i].x, landmarks[i].y) for i in indices]
        p1, p2, p3, p4, p5, p6 = pts
        d26 = np.sqrt((p2[0] - p6[0]) ** 2 + (p2[1] - p6[1]) ** 2)
        d35 = np.sqrt((p3[0] - p5[0]) ** 2 + (p3[1] - p5[1]) ** 2)
        d14 = np.sqrt((p1[0] - p4[0]) ** 2 + (p1[1] - p4[1]) ** 2)
        if d14 < 1e-6:
            return 0.30
        return float((d26 + d35) / (2.0 * d14))

    def _estimate_eyes_closed(
        self,
        frame: np.ndarray,
        face_bbox: list[float],
        pose_keypoints: list[list[float]] | None = None,
    ) -> tuple[bool, float]:
        """Eye-closure detection: MediaPipe EAR (preferred) or texture heuristic."""
        # Priority 1: MediaPipe FaceLandmarker + EAR
        mp_result = self._estimate_eyes_closed_mediapipe(frame, face_bbox)
        if mp_result is not None:
            return mp_result

        # Fallback: texture heuristic for webcam classroom videos.
        crop = self._crop(frame, face_bbox)
        if crop.size == 0:
            return False, 0.0

        h, w = crop.shape[:2]
        if h < 24 or w < 24:
            return False, 0.0

        eye_band = crop[int(h * 0.26): int(h * 0.56), int(w * 0.12): int(w * 0.88)]
        if eye_band.size == 0:
            return False, 0.0
        gray = cv2.cvtColor(eye_band, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        edges = cv2.Canny(gray, 40, 100)
        edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)

        closed_score = 0.0
        closed_score = 0.0
        if brightness < 50:
            closed_score += 0.15
        if contrast < 25:
            closed_score += 0.28
        if edge_density < 0.075:
            closed_score += 0.38

        if pose_keypoints and len(pose_keypoints) > 2:
            eye_scores = []
            for idx in (1, 2):
                kp = pose_keypoints[idx]
                if len(kp) >= 3:
                    eye_scores.append(float(kp[2]))
            if eye_scores:
                avg_eye_score = sum(eye_scores) / len(eye_scores)
                if avg_eye_score < 0.20:
                    closed_score += 0.38
                elif avg_eye_score > 0.45:
                    closed_score -= 0.22

        confidence = max(0.0, min(1.0, closed_score))
        return confidence >= 0.48, confidence
    def _update_temporal_state(
        self,
        track_id: int,
        face: FaceSignal,
        head_gaze: HeadGazeSignal,
        behavior: BehaviorSignal,
        has_person: bool,
    ) -> dict[str, int]:
        counters = self._student_temporal.setdefault(
            int(track_id),
            {"closed_eye": 0, "phone": 0, "not_frontal": 0, "absent": 0},
        )
        if not has_person:
            counters["absent"] += 1
            counters["closed_eye"] = 0
            counters["phone"] = 0
            counters["not_frontal"] = 0
            return counters

        counters["absent"] = 0
        using_phone = behavior.label == "using_phone"
        counters["closed_eye"] = 0 if using_phone else (counters["closed_eye"] + 1 if face.present and face.eyes_closed else 0)
        counters["phone"] = counters["phone"] + 1 if using_phone else 0

        not_frontal = (
            head_gaze.gaze_region == "away"
            or abs(head_gaze.yaw) > float(os.getenv("DISTRACTED_YAW_THRESHOLD", "20"))
            or abs(head_gaze.roll) > float(os.getenv("DISTRACTED_ROLL_THRESHOLD", "20"))
        )
        counters["not_frontal"] = counters["not_frontal"] + 1 if not_frontal else 0
        return counters

    @staticmethod
    def _classify_expression_openvino(crop: np.ndarray, compiled_model: Any) -> tuple[str, float]:
        # OpenVINO emotion labels are mapped into the five project states.
        label_map = ["focused", "focused", "tired", "confused", "distracted"]
        input_layer = compiled_model.input(0)
        _, _, in_h, in_w = input_layer.shape
        inp = cv2.resize(crop, (in_w, in_h)).astype(np.float32)
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]
        output = next(iter(compiled_model({input_layer.any_name: inp}).values()))
        probs = np.asarray(output).reshape(-1)
        idx = int(np.argmax(probs[: len(label_map)]))
        return label_map[idx], float(probs[idx])

    def _analyze_head_gaze(self, frame: np.ndarray, face: FaceSignal) -> HeadGazeSignal:
        if not face.present:
            return HeadGazeSignal()

        crop = self._crop(frame, face.bbox)
        model_angles = self._predict_head_pose(crop)
        if model_angles is not None:
            yaw, pitch, roll = model_angles
        else:
            yaw, pitch, roll = self._estimate_head_pose_from_face_box(frame, face.bbox)

        gaze_region, gaze_conf = self._predict_gaze_region(frame, face.bbox, crop, yaw, pitch, roll)
        return HeadGazeSignal(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            gaze_region=gaze_region,
            confidence=gaze_conf,
        )

    def _predict_head_pose(self, crop: np.ndarray) -> tuple[float, float, float] | None:
        if self._head_pose is None or crop.size == 0:
            return None
        if self._head_pose["kind"] == "openvino":
            return self._predict_head_pose_openvino(crop, self._head_pose["model"])

        session = self._head_pose["model"]
        inp = self._preprocess_image(crop, self._input_hw(session, default=(60, 60)))
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        flat = np.concatenate([np.asarray(out).reshape(-1) for out in outputs])
        if flat.size < 3:
            return None
        return float(flat[0]), float(flat[1]), float(flat[2])

    @staticmethod
    def _predict_head_pose_openvino(crop: np.ndarray, compiled_model: Any) -> tuple[float, float, float] | None:
        input_layer = compiled_model.input(0)
        _, _, in_h, in_w = input_layer.shape
        inp = cv2.resize(crop, (in_w, in_h)).astype(np.float32)
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]
        result = compiled_model({input_layer.any_name: inp})
        values = {}
        for output in compiled_model.outputs:
            values[output.any_name] = float(np.asarray(result[output]).reshape(-1)[0])
        roll = values.get("angle_r_fc", 0.0)
        pitch = values.get("angle_p_fc", 0.0)
        yaw = values.get("angle_y_fc", 0.0)
        return yaw, pitch, roll

    @staticmethod
    def _estimate_head_pose_from_face_box(
        frame: np.ndarray,
        bbox: list[float],
    ) -> tuple[float, float, float]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        yaw = ((cx / max(w, 1)) - 0.5) * 60.0
        pitch = ((cy / max(h, 1)) - 0.25) * 45.0
        return yaw, pitch, 0.0

    @staticmethod
    def _estimate_head_pitch_from_keypoints(
        pose_keypoints: list[list[float]] | None,
    ) -> float | None:
        """用 YOLO-pose 关键点估算头部低垂角度(pitch)。

        通过鼻子(0)到肩膀(5,6)的垂直距离判断低头程度。
        返回 pitch 估算值(度)，None 表示关键点不足。
        """
        if not pose_keypoints or len(pose_keypoints) < 7:
            return None
        nose = pose_keypoints[0]
        l_shoulder = pose_keypoints[5]
        r_shoulder = pose_keypoints[6]
        if len(nose) < 2 or len(l_shoulder) < 2 or len(r_shoulder) < 2:
            return None
        nose_conf = float(nose[2]) if len(nose) >= 3 else 1.0
        ls_conf = float(l_shoulder[2]) if len(l_shoulder) >= 3 else 1.0
        rs_conf = float(r_shoulder[2]) if len(r_shoulder) >= 3 else 1.0
        if nose_conf < 0.2 or ls_conf < 0.2 or rs_conf < 0.2:
            return None
        nose_y = float(nose[1])
        shoulder_y = (float(l_shoulder[1]) + float(r_shoulder[1])) / 2.0
        shoulder_w = abs(float(r_shoulder[0]) - float(l_shoulder[0]))
        if shoulder_w < 5:
            return None
        ratio = (shoulder_y - nose_y) / shoulder_w
        if ratio > 0.7:
            return 0.0
        # ratio 越小 pitch 越大: ratio=0.3→30°, ratio=0.1→45°, ratio<=0→56°+
        pitch = (0.7 - ratio) * 75.0
        return max(0.0, pitch)

    def _predict_gaze_region(
        self,
        frame: np.ndarray,
        face_bbox: list[float],
        crop: np.ndarray,
        yaw: float,
        pitch: float,
        roll: float,
    ) -> tuple[str, float]:
        if self._gaze is not None and crop.size > 0:
            if self._gaze["kind"] == "openvino":
                return self._predict_gaze_openvino(frame, face_bbox, yaw, pitch, roll, self._gaze["model"])

            session = self._gaze["model"]
            inp = self._preprocess_image(crop, self._input_hw(session, default=(224, 224)))
            outputs = session.run(None, {session.get_inputs()[0].name: inp})
            flat = np.asarray(outputs[0]).reshape(-1)
            if flat.size >= 4:
                idx = int(np.argmax(flat[:4]))
                return ["screen", "book", "phone", "away"][idx], float(self._softmax(flat[:4])[idx])
            if flat.size >= 2:
                yaw, pitch = float(flat[0]), float(flat[1])

        if abs(yaw) > 22:
            return "away", 0.55
        if pitch > 14:
            return "book", 0.50
        if pitch < -12:
            return "screen", 0.45
        return "screen", 0.60

    def _predict_gaze_openvino(
        self,
        frame: np.ndarray,
        face_bbox: list[float],
        yaw: float,
        pitch: float,
        roll: float,
        compiled_model: Any,
    ) -> tuple[str, float]:
        left_eye, right_eye = self._estimate_eye_crops(frame, face_bbox)
        if left_eye.size == 0 or right_eye.size == 0:
            return self._gaze_region_from_angles(yaw, pitch)

        inputs = {}
        for input_layer in compiled_model.inputs:
            name = input_layer.any_name
            shape = list(input_layer.shape)
            if name == "head_pose_angles" or len(shape) == 2:
                inputs[name] = np.array([[yaw, pitch, roll]], dtype=np.float32)
                continue
            if len(shape) != 4:
                continue
            _, _, in_h, in_w = shape
            eye = left_eye if name == "left_eye_image" else right_eye
            inp = cv2.resize(eye, (in_w, in_h)).astype(np.float32)
            inputs[name] = np.transpose(inp, (2, 0, 1))[np.newaxis]
        result = compiled_model(inputs)
        gaze = np.asarray(next(iter(result.values()))).reshape(-1)
        if gaze.size >= 2:
            # OpenVINO ADAS gaze vector: x horizontal, y vertical, z forward.
            gx, gy = float(gaze[0]), float(gaze[1])
            if abs(gx) > 0.35:
                return "away", min(0.85, abs(gx))
            if gy < -0.25:
                return "book", min(0.80, abs(gy))
            if gy > 0.30:
                return "away", min(0.75, abs(gy))
            return "screen", 0.70
        return self._gaze_region_from_angles(yaw, pitch)

    @staticmethod
    def _estimate_eye_crops(frame: np.ndarray, face_bbox: list[float]) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = face_bbox
        fw = x2 - x1
        fh = y2 - y1
        left = [x1 + 0.12 * fw, y1 + 0.28 * fh, x1 + 0.45 * fw, y1 + 0.55 * fh]
        right = [x1 + 0.55 * fw, y1 + 0.28 * fh, x1 + 0.88 * fw, y1 + 0.55 * fh]
        return EducationVisionPipeline._crop(frame, left), EducationVisionPipeline._crop(frame, right)

    @staticmethod
    def _gaze_region_from_angles(yaw: float, pitch: float) -> tuple[str, float]:
        if abs(yaw) > 22:
            return "away", 0.55
        if pitch > 18:
            return "book", 0.50
        if pitch < -12:
            return "screen", 0.45
        return "screen", 0.60

    def _analyze_behavior(
        self,
        frame: np.ndarray,
        person_detections: list[dict],
        object_detections: list[dict],
        head_gaze: HeadGazeSignal,
        pose_keypoints: list[list[list[float]]],
    ) -> BehaviorSignal:
        if not person_detections:
            return BehaviorSignal(label="left_seat", confidence=0.90, evidence=["no_person"])

        evidence = []
        label = "unknown"
        confidence = 0.35

        primary_person = person_detections[0] if person_detections else None
        primary_pose = pose_keypoints[0] if pose_keypoints else []
        if primary_person and self._has_phone_near_person(primary_person, object_detections, primary_pose):
            return BehaviorSignal("using_phone", 0.82, ["cell_phone_near_person"])
        if self._has_coco_object(object_detections, class_ids={67}, min_confidence=0.12):
            return BehaviorSignal("using_phone", 0.68, ["cell_phone_visible"])

        if self._has_raised_hand(pose_keypoints):
            return BehaviorSignal("raising_hand", 0.70, ["movenet_wrist_above_shoulder"])

        if self._has_coco_object(object_detections, class_ids={73}):
            label = "reading"
            confidence = 0.65
            evidence.append("book_detected")

        if head_gaze.gaze_region == "book":
            label = "writing" if label == "unknown" else label
            confidence = max(confidence, 0.55)
            evidence.append("downward_gaze")

        if self._has_low_wrists(pose_keypoints) and head_gaze.gaze_region in {"book", "screen"}:
            label = "writing"
            confidence = max(confidence, 0.62)
            evidence.append("movenet_low_wrists")

        if label == "unknown" and head_gaze.gaze_region == "screen":
            label = "reading"
            confidence = 0.45
            evidence.append("screen_gaze")

        return BehaviorSignal(label=label, confidence=confidence, evidence=evidence)

    @staticmethod
    def _has_raised_hand(pose_keypoints: list[list[list[float]]]) -> bool:
        for keypoints in pose_keypoints:
            if len(keypoints) < 11:
                continue
            left_shoulder, right_shoulder = keypoints[5], keypoints[6]
            left_wrist, right_wrist = keypoints[9], keypoints[10]
            shoulder_y = min(left_shoulder[1], right_shoulder[1])
            for wrist in (left_wrist, right_wrist):
                if wrist[2] > 0.25 and wrist[1] < shoulder_y - 20:
                    return True
        return False

    @staticmethod
    def _has_low_wrists(pose_keypoints: list[list[list[float]]]) -> bool:
        for keypoints in pose_keypoints:
            if len(keypoints) < 13:
                continue
            left_wrist, right_wrist = keypoints[9], keypoints[10]
            left_hip, right_hip = keypoints[11], keypoints[12]
            hip_y = (left_hip[1] + right_hip[1]) / 2.0
            confident_wrists = [w for w in (left_wrist, right_wrist) if w[2] > 0.20]
            if confident_wrists and all(w[1] > hip_y - 60 for w in confident_wrists):
                return True
        return False

    @staticmethod
    def _infer_learning_state(
        face: FaceSignal,
        head_gaze: HeadGazeSignal,
        behavior: BehaviorSignal,
        person_detections: list[dict],
    ) -> LearningState:
        if not person_detections:
            return LearningState("absent", 0.95, "notify_absence", ["no_person_detected"])

        reasons = []
        if behavior.label == "using_phone":
            return LearningState("distracted", 0.82, "gentle_reminder", ["using_phone"])

        if behavior.label == "raising_hand":
            return LearningState("confused", 0.68, "offer_help", ["raising_hand"])

        if head_gaze.gaze_region == "away":
            return LearningState("distracted", 0.70, "redirect_attention", ["gaze_away"])

        if face.expression == "tired":
            return LearningState("tired", 0.66, "suggest_break", ["tired_expression"])

        if face.expression == "confused":
            return LearningState("confused", 0.64, "offer_hint", ["confused_expression"])

        if face.present:
            reasons.append("face_present")
        if head_gaze.gaze_region in {"screen", "book"}:
            reasons.append(f"gaze_{head_gaze.gaze_region}")
        if behavior.label in {"reading", "writing"}:
            reasons.append(behavior.label)

        confidence = 0.72 if reasons else 0.50
        return LearningState("focused", confidence, "none", reasons)

    @staticmethod
    def _crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, min(w, int(bbox[0])))
        y1 = max(0, min(h, int(bbox[1])))
        x2 = max(0, min(w, int(bbox[2])))
        y2 = max(0, min(h, int(bbox[3])))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _preprocess_image(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        img = cv2.resize(frame, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img[np.newaxis]

    @staticmethod
    def _input_hw(session: Any, default: tuple[int, int]) -> tuple[int, int]:
        shape = session.get_inputs()[0].shape
        if len(shape) == 4:
            h, w = shape[-2], shape[-1]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return w, h
        return default

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        values = values - np.max(values)
        exp = np.exp(values)
        return exp / (np.sum(exp) + 1e-9)

    @staticmethod
    def _has_coco_object(
        detections: list[dict],
        class_ids: set[int],
        min_confidence: float = 0.35,
    ) -> bool:
        return any(
            det.get("class_id") in class_ids
            and det.get("confidence", 0.0) >= min_confidence
            for det in detections
        )

    @staticmethod
    def _postprocess_yolo_like(
        output: np.ndarray,
        frame_hw: tuple[int, int],
        conf_threshold: float,
    ) -> list[dict]:
        arr = np.asarray(output)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape[0] < arr.shape[-1]:
            arr = arr.T
        if arr.shape[1] < 5:
            return []

        h_orig, w_orig = frame_hw
        class_scores = arr[:, 4:]
        scores = class_scores.max(axis=1) if class_scores.size else arr[:, 4]
        mask = scores > conf_threshold
        if not np.any(mask):
            return []

        boxes = arr[mask, :4]
        scores = scores[mask]
        results = []
        for box, score in zip(boxes, scores, strict=False):
            cx, cy, bw, bh = box
            results.append({
                "x1": float((cx - bw / 2) * w_orig / 640),
                "y1": float((cy - bh / 2) * h_orig / 640),
                "x2": float((cx + bw / 2) * w_orig / 640),
                "y2": float((cy + bh / 2) * h_orig / 640),
                "confidence": float(score),
                "class_id": 0,
            })
        return results
