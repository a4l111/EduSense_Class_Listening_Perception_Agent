"""构建视频分析流水线 DAG 的工厂函数。"""

from __future__ import annotations

from .graph import DAGTask, ProcessorType, Stage, StageType

# 默认 5 阶段流水线的基准延迟 (ms) 和数据大小 (bytes)
# 数据来源: ONNX Runtime profiling (CoreML EP / CPU EP, medium 密度 ~10 tracks)
# ByteTrack 为纯 CPU 算法，不支持 GPU
# REID 延迟与目标数量线性相关: GPU ~0.77ms/track, CPU ~2.0ms/track
_DEFAULT_STAGE_SPECS: list[dict] = [
    {
        "stage_type": StageType.DECODE,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 0.5},
        "input_size_bytes": 1_000_000,   # ~1 MB 压缩帧
        "output_size_bytes": 6_220_800,  # 1920x1080x3 原始帧
    },
    {
        "stage_type": StageType.DETECT,
        "supported_processors": [ProcessorType.CPU, ProcessorType.GPU],
        "base_latency": {ProcessorType.CPU: 31.7, ProcessorType.GPU: 9.0},
        "input_size_bytes": 6_220_800,
        "output_size_bytes": 50_000,     # bbox 列表
    },
    {
        "stage_type": StageType.TRACK,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 1.8},
        "input_size_bytes": 50_000,
        "output_size_bytes": 50_000,
    },
    {
        "stage_type": StageType.REID,
        "supported_processors": [ProcessorType.CPU, ProcessorType.GPU],
        "base_latency": {ProcessorType.CPU: 18.7, ProcessorType.GPU: 7.7},
        "input_size_bytes": 50_000,
        "output_size_bytes": 10_000,     # embedding 向量
    },
    {
        "stage_type": StageType.POSTPROCESS,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 0.5},
        "input_size_bytes": 10_000,
        "output_size_bytes": 5_000,
    },
]


def build_video_pipeline(
    dag_id: str = "video_dag_0",
    deadline_ms: float = 200.0,
    arrival_time: float = 0.0,
    latency_scale: float = 1.0,
    stage_specs: list[dict] | None = None,
) -> DAGTask:
    """构建一条 5 阶段线性视频分析流水线 DAG。

    Args:
        dag_id: DAG 标识符
        deadline_ms: SLO 截止时间 (ms)
        arrival_time: DAG 到达时间 (ms)
        latency_scale: 延迟缩放因子（模拟不同模型复杂度）
        stage_specs: 自定义阶段规格，默认使用 5 阶段流水线
    """
    specs = stage_specs or _DEFAULT_STAGE_SPECS
    stages: dict[str, Stage] = {}
    stage_ids: list[str] = []

    for i, spec in enumerate(specs):
        sid = f"{dag_id}_s{i}_{spec['stage_type'].value}"
        stage_ids.append(sid)

        scaled_latency = {
            proc: lat * latency_scale for proc, lat in spec["base_latency"].items()
        }
        predecessors = [stage_ids[i - 1]] if i > 0 else []

        stages[sid] = Stage(
            id=sid,
            stage_type=spec["stage_type"],
            supported_processors=list(spec["supported_processors"]),
            base_latency=scaled_latency,
            input_size_bytes=spec["input_size_bytes"],
            output_size_bytes=spec["output_size_bytes"],
            predecessors=predecessors,
            successors=[],
        )

    # 填充 successors
    for i, sid in enumerate(stage_ids[:-1]):
        stages[sid].successors = [stage_ids[i + 1]]

    return DAGTask(
        dag_id=dag_id,
        stages=stages,
        entry_stages=[stage_ids[0]],
        exit_stages=[stage_ids[-1]],
        deadline_ms=deadline_ms,
        arrival_time=arrival_time,
    )


# ────────────────────────────────────────────────────────────────
# Android Pipeline A (2-fork): DECODE → DETECT → TRACK → (REID ‖ POSE) → POST
# 延迟数据来自 Samsung S10e (Exynos 9820) TFLite profiling
# ────────────────────────────────────────────────────────────────

def build_android_pipeline_a(
    dag_id: str = "android_a_0",
    deadline_ms: float = 500.0,
    arrival_time: float = 0.0,
    density_level: str = "medium",
    node_latencies: dict[str, dict[ProcessorType, float]] | None = None,
) -> DAGTask:
    """构建 Android Pipeline A: DECODE → DETECT → TRACK → (REID ‖ POSE) → POST

    Args:
        node_latencies: 覆盖默认延迟，格式 {"detect": {CPU: x, GPU: y}, ...}
                        用于不同节点（S10e/一加9/Mac）使用不同延迟
    """
    # 默认延迟: S10e TFLite profiling (medium density ~10 tracks)
    lat = {
        "decode":  {ProcessorType.CPU: 0.1},
        "detect":  {ProcessorType.CPU: 288.0, ProcessorType.GPU: 91.0},
        "track":   {ProcessorType.CPU: 0.1},
        "reid":    {ProcessorType.CPU: 20.0, ProcessorType.GPU: 7.7},   # 10 tracks total
        "pose":    {ProcessorType.CPU: 180.0, ProcessorType.GPU: 94.0}, # 10 tracks total
        "post":    {ProcessorType.CPU: 0.1},
    }
    if node_latencies:
        for k, v in node_latencies.items():
            if k in lat:
                lat[k] = v

    # per-track 阶段根据密度缩放
    density_scale = {"low": 0.5, "medium": 1.0, "high": 2.0}
    scale = density_scale.get(density_level, 1.0)

    stages: dict[str, Stage] = {}
    prefix = dag_id

    # DECODE
    stages[f"{prefix}_decode"] = Stage(
        id=f"{prefix}_decode", stage_type=StageType.DECODE,
        supported_processors=[ProcessorType.CPU],
        base_latency=lat["decode"],
        input_size_bytes=100_000,  # JPEG 压缩帧
        output_size_bytes=1_228_800,  # 640×640×3
    )
    # DETECT
    stages[f"{prefix}_detect"] = Stage(
        id=f"{prefix}_detect", stage_type=StageType.DETECT,
        supported_processors=[ProcessorType.CPU, ProcessorType.GPU],
        base_latency=lat["detect"],
        input_size_bytes=1_228_800,
        output_size_bytes=5_000,  # 检测框列表
        predecessors=[f"{prefix}_decode"],
    )
    # TRACK
    stages[f"{prefix}_track"] = Stage(
        id=f"{prefix}_track", stage_type=StageType.TRACK,
        supported_processors=[ProcessorType.CPU],
        base_latency=lat["track"],
        input_size_bytes=5_000,
        output_size_bytes=5_000,
        predecessors=[f"{prefix}_detect"],
    )
    # REID (fork 分支 1) — per-track，延迟随密度缩放
    reid_lat = {p: v * scale for p, v in lat["reid"].items()}
    stages[f"{prefix}_reid"] = Stage(
        id=f"{prefix}_reid", stage_type=StageType.REID,
        supported_processors=[ProcessorType.CPU, ProcessorType.GPU],
        base_latency=reid_lat,
        input_size_bytes=int(5_000 * scale),  # track crops
        output_size_bytes=int(5_000 * scale),
        predecessors=[f"{prefix}_track"],
    )
    # POSE (fork 分支 2) — per-track，延迟随密度缩放
    pose_lat = {p: v * scale for p, v in lat["pose"].items()}
    stages[f"{prefix}_pose"] = Stage(
        id=f"{prefix}_pose", stage_type=StageType.POSE_ESTIMATE,
        supported_processors=[ProcessorType.CPU, ProcessorType.GPU],
        base_latency=pose_lat,
        input_size_bytes=int(5_000 * scale),
        output_size_bytes=int(5_000 * scale),
        predecessors=[f"{prefix}_track"],
    )
    # POSTPROCESS (join)
    stages[f"{prefix}_post"] = Stage(
        id=f"{prefix}_post", stage_type=StageType.POSTPROCESS,
        supported_processors=[ProcessorType.CPU],
        base_latency=lat["post"],
        input_size_bytes=10_000,
        output_size_bytes=5_000,
        predecessors=[f"{prefix}_reid", f"{prefix}_pose"],
    )

    # 填充 successors
    stages[f"{prefix}_decode"].successors = [f"{prefix}_detect"]
    stages[f"{prefix}_detect"].successors = [f"{prefix}_track"]
    stages[f"{prefix}_track"].successors = [f"{prefix}_reid", f"{prefix}_pose"]
    stages[f"{prefix}_reid"].successors = [f"{prefix}_post"]
    stages[f"{prefix}_pose"].successors = [f"{prefix}_post"]

    dag = DAGTask(
        dag_id=dag_id,
        stages=stages,
        entry_stages=[f"{prefix}_decode"],
        exit_stages=[f"{prefix}_post"],
        deadline_ms=deadline_ms,
        arrival_time=arrival_time,
        density_level=density_level,
    )
    return dag


# ────────────────────────────────────────────────────────────────
# Android Pipeline C (3-fork): DECODE → DETECT → TRACK → (REID ‖ POSE ‖ ATTR) → POST
# ────────────────────────────────────────────────────────────────

def build_android_pipeline_c(
    dag_id: str = "android_c_0",
    deadline_ms: float = 500.0,
    arrival_time: float = 0.0,
    density_level: str = "medium",
    node_latencies: dict[str, dict[ProcessorType, float]] | None = None,
) -> DAGTask:
    """构建 Android Pipeline C: DECODE → DETECT → TRACK → (REID ‖ POSE ‖ ATTR) → POST"""
    lat = {
        "decode":  {ProcessorType.CPU: 0.1},
        "detect":  {ProcessorType.CPU: 288.0, ProcessorType.GPU: 91.0},
        "track":   {ProcessorType.CPU: 0.1},
        "reid":    {ProcessorType.CPU: 20.0, ProcessorType.GPU: 7.7},
        "pose":    {ProcessorType.CPU: 180.0, ProcessorType.GPU: 94.0},
        "attr":    {ProcessorType.CPU: 275.0, ProcessorType.GPU: 125.0},  # EfficientNet-Lite0
        "post":    {ProcessorType.CPU: 0.1},
    }
    if node_latencies:
        for k, v in node_latencies.items():
            if k in lat:
                lat[k] = v

    density_scale = {"low": 0.5, "medium": 1.0, "high": 2.0}
    scale = density_scale.get(density_level, 1.0)

    stages: dict[str, Stage] = {}
    prefix = dag_id

    stages[f"{prefix}_decode"] = Stage(
        id=f"{prefix}_decode", stage_type=StageType.DECODE,
        supported_processors=[ProcessorType.CPU],
        base_latency=lat["decode"],
        input_size_bytes=100_000, output_size_bytes=1_228_800,
    )
    stages[f"{prefix}_detect"] = Stage(
        id=f"{prefix}_detect", stage_type=StageType.DETECT,
        supported_processors=[ProcessorType.CPU, ProcessorType.GPU],
        base_latency=lat["detect"],
        input_size_bytes=1_228_800, output_size_bytes=5_000,
        predecessors=[f"{prefix}_decode"],
    )
    stages[f"{prefix}_track"] = Stage(
        id=f"{prefix}_track", stage_type=StageType.TRACK,
        supported_processors=[ProcessorType.CPU],
        base_latency=lat["track"],
        input_size_bytes=5_000, output_size_bytes=5_000,
        predecessors=[f"{prefix}_detect"],
    )

    reid_lat = {p: v * scale for p, v in lat["reid"].items()}
    stages[f"{prefix}_reid"] = Stage(
        id=f"{prefix}_reid", stage_type=StageType.REID,
        supported_processors=[ProcessorType.CPU, ProcessorType.GPU],
        base_latency=reid_lat,
        input_size_bytes=int(5_000 * scale), output_size_bytes=int(5_000 * scale),
        predecessors=[f"{prefix}_track"],
    )

    pose_lat = {p: v * scale for p, v in lat["pose"].items()}
    stages[f"{prefix}_pose"] = Stage(
        id=f"{prefix}_pose", stage_type=StageType.POSE_ESTIMATE,
        supported_processors=[ProcessorType.CPU, ProcessorType.GPU],
        base_latency=pose_lat,
        input_size_bytes=int(5_000 * scale), output_size_bytes=int(5_000 * scale),
        predecessors=[f"{prefix}_track"],
    )

    attr_lat = {p: v * scale for p, v in lat["attr"].items()}
    stages[f"{prefix}_attr"] = Stage(
        id=f"{prefix}_attr", stage_type=StageType.BEHAVIOR_CLASSIFY,
        supported_processors=[ProcessorType.CPU, ProcessorType.GPU],
        base_latency=attr_lat,
        input_size_bytes=int(5_000 * scale), output_size_bytes=int(2_000 * scale),
        predecessors=[f"{prefix}_track"],
    )

    stages[f"{prefix}_post"] = Stage(
        id=f"{prefix}_post", stage_type=StageType.POSTPROCESS,
        supported_processors=[ProcessorType.CPU],
        base_latency=lat["post"],
        input_size_bytes=10_000, output_size_bytes=5_000,
        predecessors=[f"{prefix}_reid", f"{prefix}_pose", f"{prefix}_attr"],
    )

    # successors
    stages[f"{prefix}_decode"].successors = [f"{prefix}_detect"]
    stages[f"{prefix}_detect"].successors = [f"{prefix}_track"]
    stages[f"{prefix}_track"].successors = [
        f"{prefix}_reid", f"{prefix}_pose", f"{prefix}_attr",
    ]
    stages[f"{prefix}_reid"].successors = [f"{prefix}_post"]
    stages[f"{prefix}_pose"].successors = [f"{prefix}_post"]
    stages[f"{prefix}_attr"].successors = [f"{prefix}_post"]

    return DAGTask(
        dag_id=dag_id,
        stages=stages,
        entry_stages=[f"{prefix}_decode"],
        exit_stages=[f"{prefix}_post"],
        deadline_ms=deadline_ms,
        arrival_time=arrival_time,
        density_level=density_level,
    )


# ---- Pipeline B: 行人姿态分析 ----
# DECODE → DETECT → POSE_ESTIMATE → BEHAVIOR_CLASSIFY → POSTPROCESS

def build_education_pipeline(
    dag_id: str = "edu_0",
    deadline_ms: float = 500.0,
    arrival_time: float = 0.0,
    density_level: str = "medium",
    node_latencies: dict[str, dict[ProcessorType, float]] | None = None,
) -> DAGTask:
    """Build education DAG: decode -> face/emotion -> (head/gaze || behavior) -> fusion."""
    lat = {
        "decode": {ProcessorType.CPU: 0.2},
        "face_emotion": {ProcessorType.CPU: 95.0, ProcessorType.GPU: 38.0},
        "head_gaze": {ProcessorType.CPU: 45.0},
        "edu_behavior": {ProcessorType.CPU: 240.0, ProcessorType.GPU: 105.0},
        "state_fusion": {ProcessorType.CPU: 2.0},
    }
    if node_latencies:
        for k, v in node_latencies.items():
            if k in lat:
                lat[k] = v

    density_scale = {"low": 0.6, "medium": 1.0, "high": 1.8}
    scale = density_scale.get(density_level, 1.0)

    stages: dict[str, Stage] = {}
    prefix = dag_id

    stages[f"{prefix}_decode"] = Stage(
        id=f"{prefix}_decode",
        stage_type=StageType.DECODE,
        supported_processors=[ProcessorType.CPU],
        base_latency=lat["decode"],
        input_size_bytes=250_000,
        output_size_bytes=921_600,
    )

    face_lat = {p: v * scale for p, v in lat["face_emotion"].items()}
    stages[f"{prefix}_face_emotion"] = Stage(
        id=f"{prefix}_face_emotion",
        stage_type=StageType.FACE_EMOTION,
        supported_processors=[ProcessorType.CPU, ProcessorType.GPU],
        base_latency=face_lat,
        input_size_bytes=921_600,
        output_size_bytes=int(12_000 * scale),
        predecessors=[f"{prefix}_decode"],
    )

    gaze_lat = {p: v * scale for p, v in lat["head_gaze"].items()}
    stages[f"{prefix}_head_gaze"] = Stage(
        id=f"{prefix}_head_gaze",
        stage_type=StageType.HEAD_GAZE,
        supported_processors=[ProcessorType.CPU],
        base_latency=gaze_lat,
        input_size_bytes=int(12_000 * scale),
        output_size_bytes=int(4_000 * scale),
        predecessors=[f"{prefix}_face_emotion"],
    )

    behavior_lat = {p: v * scale for p, v in lat["edu_behavior"].items()}
    stages[f"{prefix}_behavior"] = Stage(
        id=f"{prefix}_behavior",
        stage_type=StageType.EDU_BEHAVIOR,
        supported_processors=[ProcessorType.CPU, ProcessorType.GPU],
        base_latency=behavior_lat,
        input_size_bytes=921_600,
        output_size_bytes=int(6_000 * scale),
        predecessors=[f"{prefix}_face_emotion"],
    )

    stages[f"{prefix}_fusion"] = Stage(
        id=f"{prefix}_fusion",
        stage_type=StageType.STATE_FUSION,
        supported_processors=[ProcessorType.CPU],
        base_latency=lat["state_fusion"],
        input_size_bytes=int(10_000 * scale),
        output_size_bytes=1_000,
        predecessors=[f"{prefix}_head_gaze", f"{prefix}_behavior"],
    )

    stages[f"{prefix}_decode"].successors = [f"{prefix}_face_emotion"]
    stages[f"{prefix}_face_emotion"].successors = [
        f"{prefix}_head_gaze",
        f"{prefix}_behavior",
    ]
    stages[f"{prefix}_head_gaze"].successors = [f"{prefix}_fusion"]
    stages[f"{prefix}_behavior"].successors = [f"{prefix}_fusion"]

    return DAGTask(
        dag_id=dag_id,
        stages=stages,
        entry_stages=[f"{prefix}_decode"],
        exit_stages=[f"{prefix}_fusion"],
        deadline_ms=deadline_ms,
        arrival_time=arrival_time,
        density_level=density_level,
    )


_POSE_PIPELINE_SPECS: list[dict] = [
    {
        "stage_type": StageType.DECODE,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 5.0},
        "input_size_bytes": 1_000_000,
        "output_size_bytes": 6_220_800,
    },
    {
        "stage_type": StageType.DETECT,
        "supported_processors": [ProcessorType.CPU, ProcessorType.GPU],
        "base_latency": {ProcessorType.CPU: 80.0, ProcessorType.GPU: 15.0},
        "input_size_bytes": 6_220_800,
        "output_size_bytes": 50_000,
    },
    {
        "stage_type": StageType.POSE_ESTIMATE,
        "supported_processors": [ProcessorType.CPU, ProcessorType.GPU],
        "base_latency": {ProcessorType.CPU: 120.0, ProcessorType.GPU: 25.0},
        "input_size_bytes": 6_220_800,  # 需要原始帧 + 检测框
        "output_size_bytes": 200_000,   # 关键点数据
    },
    {
        "stage_type": StageType.BEHAVIOR_CLASSIFY,
        "supported_processors": [ProcessorType.CPU, ProcessorType.GPU],
        "base_latency": {ProcessorType.CPU: 10.0, ProcessorType.GPU: 3.0},
        "input_size_bytes": 200_000,
        "output_size_bytes": 10_000,
    },
    {
        "stage_type": StageType.POSTPROCESS,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 5.0},
        "input_size_bytes": 10_000,
        "output_size_bytes": 5_000,
    },
]


def build_pose_pipeline(
    dag_id: str = "pose_dag_0",
    deadline_ms: float = 300.0,
    arrival_time: float = 0.0,
    latency_scale: float = 1.0,
) -> DAGTask:
    """构建 Pipeline B: 行人姿态分析流水线。

    DECODE → DETECT → POSE_ESTIMATE → BEHAVIOR_CLASSIFY → POSTPROCESS
    """
    return build_video_pipeline(
        dag_id=dag_id,
        deadline_ms=deadline_ms,
        arrival_time=arrival_time,
        latency_scale=latency_scale,
        stage_specs=_POSE_PIPELINE_SPECS,
    )


# ---- Pipeline C: 车辆监控 + 车牌识别 ----
# DECODE → DETECT → CROP → LPR → POSTPROCESS

_VEHICLE_PIPELINE_SPECS: list[dict] = [
    {
        "stage_type": StageType.DECODE,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 5.0},
        "input_size_bytes": 1_000_000,
        "output_size_bytes": 6_220_800,
    },
    {
        "stage_type": StageType.DETECT,
        "supported_processors": [ProcessorType.CPU, ProcessorType.GPU],
        "base_latency": {ProcessorType.CPU: 80.0, ProcessorType.GPU: 15.0},
        "input_size_bytes": 6_220_800,
        "output_size_bytes": 50_000,
    },
    {
        "stage_type": StageType.CROP,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 2.0},
        "input_size_bytes": 6_220_800,
        "output_size_bytes": 500_000,  # 多个裁剪区域
    },
    {
        "stage_type": StageType.LPR,
        "supported_processors": [ProcessorType.CPU, ProcessorType.GPU],
        "base_latency": {ProcessorType.CPU: 50.0, ProcessorType.GPU: 10.0},
        "input_size_bytes": 500_000,
        "output_size_bytes": 5_000,
    },
    {
        "stage_type": StageType.POSTPROCESS,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 3.0},
        "input_size_bytes": 5_000,
        "output_size_bytes": 5_000,
    },
]


def build_vehicle_pipeline(
    dag_id: str = "vehicle_dag_0",
    deadline_ms: float = 200.0,
    arrival_time: float = 0.0,
    latency_scale: float = 1.0,
) -> DAGTask:
    """构建 Pipeline C: 车辆监控 + 车牌识别流水线。

    DECODE → DETECT → CROP → LPR → POSTPROCESS
    """
    return build_video_pipeline(
        dag_id=dag_id,
        deadline_ms=deadline_ms,
        arrival_time=arrival_time,
        latency_scale=latency_scale,
        stage_specs=_VEHICLE_PIPELINE_SPECS,
    )


# ---- Pipeline D: 场景理解 + 人群计数 ----
# DECODE → SEGMENT → CROWD_COUNT → HEATMAP → POSTPROCESS

_SCENE_PIPELINE_SPECS: list[dict] = [
    {
        "stage_type": StageType.DECODE,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 5.0},
        "input_size_bytes": 1_000_000,
        "output_size_bytes": 6_220_800,
    },
    {
        "stage_type": StageType.SEGMENT,
        "supported_processors": [ProcessorType.CPU, ProcessorType.GPU],
        "base_latency": {ProcessorType.CPU: 200.0, ProcessorType.GPU: 40.0},
        "input_size_bytes": 6_220_800,
        "output_size_bytes": 2_073_600,  # 1920x1080 mask
    },
    {
        "stage_type": StageType.CROWD_COUNT,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 8.0},
        "input_size_bytes": 2_073_600,
        "output_size_bytes": 2_073_600,
    },
    {
        "stage_type": StageType.HEATMAP,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 10.0},
        "input_size_bytes": 2_073_600,
        "output_size_bytes": 6_220_800,
    },
    {
        "stage_type": StageType.POSTPROCESS,
        "supported_processors": [ProcessorType.CPU],
        "base_latency": {ProcessorType.CPU: 5.0},
        "input_size_bytes": 6_220_800,
        "output_size_bytes": 5_000,
    },
]


def build_scene_pipeline(
    dag_id: str = "scene_dag_0",
    deadline_ms: float = 350.0,
    arrival_time: float = 0.0,
    latency_scale: float = 1.0,
) -> DAGTask:
    """构建 Pipeline D: 场景理解 + 人群计数流水线。

    DECODE → SEGMENT → CROWD_COUNT → HEATMAP → POSTPROCESS
    """
    return build_video_pipeline(
        dag_id=dag_id,
        deadline_ms=deadline_ms,
        arrival_time=arrival_time,
        latency_scale=latency_scale,
        stage_specs=_SCENE_PIPELINE_SPECS,
    )
