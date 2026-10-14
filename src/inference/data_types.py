"""Pipeline A/B/C/D stage 间数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionResult:
    """DETECT stage 输出：目标检测结果。"""

    boxes: np.ndarray  # (N, 4) xyxy 格式
    confidences: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,)

    @property
    def count(self) -> int:
        return len(self.boxes)

    def filter_class(self, class_id: int) -> DetectionResult:
        """过滤指定类别。"""
        mask = self.class_ids == class_id
        return DetectionResult(
            boxes=self.boxes[mask],
            confidences=self.confidences[mask],
            class_ids=self.class_ids[mask],
        )

    @staticmethod
    def empty() -> DetectionResult:
        return DetectionResult(
            boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
            class_ids=np.empty(0, dtype=np.int32),
        )


@dataclass
class TrackResult:
    """TRACK stage 输出：跟踪结果。"""

    boxes: np.ndarray  # (N, 4) xyxy
    track_ids: np.ndarray  # (N,)
    confidences: np.ndarray  # (N,)

    @property
    def count(self) -> int:
        return len(self.boxes)

    @staticmethod
    def empty() -> TrackResult:
        return TrackResult(
            boxes=np.empty((0, 4), dtype=np.float32),
            track_ids=np.empty(0, dtype=np.int64),
            confidences=np.empty(0, dtype=np.float32),
        )


@dataclass
class ReIDResult:
    """REID stage 输出：行人重识别特征。"""

    track_ids: np.ndarray  # (N,)
    embeddings: np.ndarray  # (N, 512)
    boxes: np.ndarray  # (N, 4) xyxy

    @property
    def count(self) -> int:
        return len(self.track_ids)

    @staticmethod
    def empty() -> ReIDResult:
        return ReIDResult(
            track_ids=np.empty(0, dtype=np.int64),
            embeddings=np.empty((0, 512), dtype=np.float32),
            boxes=np.empty((0, 4), dtype=np.float32),
        )


@dataclass
class PipelineResult:
    """Pipeline A 单帧完整执行结果。"""

    dag_id: str
    frame_idx: int
    detection: DetectionResult | None = None
    tracking: TrackResult | None = None
    reid: ReIDResult | None = None
    annotated_frame: np.ndarray | None = None  # 标注后的可视化帧
    total_latency_ms: float = 0.0
    stage_latencies: dict[str, float] = field(default_factory=dict)

    @property
    def num_detections(self) -> int:
        return self.detection.count if self.detection else 0

    @property
    def num_tracks(self) -> int:
        return self.tracking.count if self.tracking else 0

    @property
    def num_reid(self) -> int:
        return self.reid.count if self.reid else 0


# ---- Pipeline B: 姿态分析 ----


@dataclass
class PoseResult:
    """POSE_ESTIMATE stage 输出：姿态估计结果。"""

    keypoints: np.ndarray  # (N, 17, 3) — 17 关键点 x,y,conf
    boxes: np.ndarray  # (N, 4) xyxy
    confidences: np.ndarray  # (N,)

    @property
    def count(self) -> int:
        return len(self.boxes)

    @staticmethod
    def empty() -> PoseResult:
        return PoseResult(
            keypoints=np.empty((0, 17, 3), dtype=np.float32),
            boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
        )


@dataclass
class BehaviorResult:
    """BEHAVIOR_CLASSIFY stage 输出：行为分类结果。"""

    labels: list[str]  # (N,) 行为标签
    confidences: np.ndarray  # (N,)
    alerts: list[dict]  # 告警列表

    @property
    def count(self) -> int:
        return len(self.labels)

    @staticmethod
    def empty() -> BehaviorResult:
        return BehaviorResult(
            labels=[],
            confidences=np.empty(0, dtype=np.float32),
            alerts=[],
        )


@dataclass
class PipelineBResult:
    """Pipeline B 单帧完整执行结果。"""

    dag_id: str
    frame_idx: int
    detection: DetectionResult | None = None
    pose: PoseResult | None = None
    behavior: BehaviorResult | None = None
    annotated_frame: np.ndarray | None = None
    total_latency_ms: float = 0.0
    stage_latencies: dict[str, float] = field(default_factory=dict)

    @property
    def num_detections(self) -> int:
        return self.detection.count if self.detection else 0

    @property
    def num_poses(self) -> int:
        return self.pose.count if self.pose else 0

    @property
    def num_alerts(self) -> int:
        return len(self.behavior.alerts) if self.behavior else 0


# ---- Pipeline C: 车辆监控 ----


@dataclass
class CropResult:
    """CROP stage 输出：裁剪的车牌/ROI 图像。"""

    crops: list[np.ndarray]  # 裁剪的车牌区域
    boxes: np.ndarray  # (N, 4) 车牌区域框
    parent_boxes: np.ndarray  # (N, 4) 车辆检测框

    @property
    def count(self) -> int:
        return len(self.crops)

    @staticmethod
    def empty() -> CropResult:
        return CropResult(
            crops=[],
            boxes=np.empty((0, 4), dtype=np.float32),
            parent_boxes=np.empty((0, 4), dtype=np.float32),
        )


@dataclass
class LPRResult:
    """LPR stage 输出：车牌识别结果。"""

    plates: list[str]  # (N,) 车牌文本
    confidences: np.ndarray  # (N,)
    boxes: np.ndarray  # (N, 4) 车牌位置

    @property
    def count(self) -> int:
        return len(self.plates)

    @staticmethod
    def empty() -> LPRResult:
        return LPRResult(
            plates=[],
            confidences=np.empty(0, dtype=np.float32),
            boxes=np.empty((0, 4), dtype=np.float32),
        )


@dataclass
class PipelineCResult:
    """Pipeline C 单帧完整执行结果。"""

    dag_id: str
    frame_idx: int
    detection: DetectionResult | None = None
    crop: CropResult | None = None
    lpr: LPRResult | None = None
    annotated_frame: np.ndarray | None = None
    total_latency_ms: float = 0.0
    stage_latencies: dict[str, float] = field(default_factory=dict)

    @property
    def num_detections(self) -> int:
        return self.detection.count if self.detection else 0

    @property
    def num_plates(self) -> int:
        return self.lpr.count if self.lpr else 0


# ---- Pipeline D: 场景理解 ----


@dataclass
class SegmentationResult:
    """SEGMENT stage 输出：语义分割结果。"""

    mask: np.ndarray  # (H, W) int 类别 mask
    class_names: list[str]  # 类别名列表
    person_mask: np.ndarray  # (H, W) bool 行人区域

    @staticmethod
    def empty(h: int = 480, w: int = 640) -> SegmentationResult:
        return SegmentationResult(
            mask=np.zeros((h, w), dtype=np.int32),
            class_names=[],
            person_mask=np.zeros((h, w), dtype=bool),
        )


@dataclass
class CrowdCountResult:
    """CROWD_COUNT stage 输出：人群计数结果。"""

    total_count: int
    density_map: np.ndarray  # (H, W) float 密度图
    region_counts: dict[str, int]  # 分区计数

    @staticmethod
    def empty(h: int = 480, w: int = 640) -> CrowdCountResult:
        return CrowdCountResult(
            total_count=0,
            density_map=np.zeros((h, w), dtype=np.float32),
            region_counts={},
        )


@dataclass
class PipelineDResult:
    """Pipeline D 单帧完整执行结果。"""

    dag_id: str
    frame_idx: int
    segmentation: SegmentationResult | None = None
    crowd_count: CrowdCountResult | None = None
    heatmap: np.ndarray | None = None  # (H, W, 3) 热力图
    annotated_frame: np.ndarray | None = None
    total_latency_ms: float = 0.0
    stage_latencies: dict[str, float] = field(default_factory=dict)

    @property
    def person_count(self) -> int:
        return self.crowd_count.total_count if self.crowd_count else 0
