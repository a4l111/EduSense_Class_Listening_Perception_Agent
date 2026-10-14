"""DAG 核心数据结构：Stage、DAGTask 等。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProcessorType(Enum):
    """处理器类型。"""
    CPU = "cpu"
    GPU = "gpu"


class StageType(Enum):
    """视频分析流水线阶段类型。"""
    # 通用
    DECODE = "decode"
    POSTPROCESS = "postprocess"
    # Pipeline A: 行人监控
    DETECT = "detect"
    TRACK = "track"
    REID = "reid"
    # Pipeline B: 姿态分析
    POSE_ESTIMATE = "pose_estimate"
    BEHAVIOR_CLASSIFY = "behavior_classify"
    # Pipeline C: 车辆监控
    CROP = "crop"
    LPR = "lpr"
    # Pipeline D: 场景理解
    SEGMENT = "segment"
    CROWD_COUNT = "crowd_count"
    HEATMAP = "heatmap"
    # Education pipeline: classroom/home learning-state analysis
    FACE_EMOTION = "face_emotion"
    HEAD_GAZE = "head_gaze"
    EDU_BEHAVIOR = "edu_behavior"
    STATE_FUSION = "state_fusion"


@dataclass
class Stage:
    """DAG 中的一个处理阶段。"""
    id: str
    stage_type: StageType
    supported_processors: list[ProcessorType]
    base_latency: dict[ProcessorType, float]  # 每种处理器上的基准延迟 (ms)
    input_size_bytes: int = 0   # 输入数据大小，用于网络传输计算
    output_size_bytes: int = 0  # 输出数据大小
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)


@dataclass
class DAGTask:
    """一个完整的 DAG 任务（一帧视频的处理流水线）。"""
    dag_id: str
    stages: dict[str, Stage]    # stage_id -> Stage
    entry_stages: list[str]     # 入口节点（无前驱）
    exit_stages: list[str]      # 出口节点（无后继）
    deadline_ms: float          # SLO 截止时间
    arrival_time: float = 0.0   # 到达时间戳 (ms)
    density_level: str = "medium"  # 当前帧密度档 ("low"/"medium"/"high")

    def topological_order(self) -> list[str]:
        """返回 DAG 的拓扑排序。"""
        in_degree: dict[str, int] = {sid: len(s.predecessors) for sid, s in self.stages.items()}
        queue = [sid for sid, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            # 按 id 排序保证确定性
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for succ in self.stages[node].successors:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        if len(order) != len(self.stages):
            raise ValueError("DAG contains a cycle")
        return order
