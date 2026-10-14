"""推理后端抽象及实现。"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any

from src.dag.graph import ProcessorType, StageType


class InferenceBackend(ABC):
    """推理后端抽象基类。"""

    @abstractmethod
    def run(
        self, stage_type: StageType, processor: ProcessorType, input_data: Any
    ) -> tuple[Any, float]:
        """执行一次推理。

        Returns:
            (output_data, actual_latency_ms)
        """
        ...


class SimulatedBackend(InferenceBackend):
    """模拟推理后端：用 base_latency + 随机噪声模拟推理延迟。"""

    def __init__(
        self,
        base_latencies: dict[tuple[StageType, ProcessorType], float] | None = None,
        noise_ratio: float = 0.1,
        seed: int | None = None,
        density_scaling: dict[str, dict[str, float]] | None = None,
    ):
        """
        Args:
            base_latencies: (stage_type, proc) -> 基准延迟 (ms)，None 则用默认值
            noise_ratio: 噪声比例 (e.g. 0.1 = ±10%)
            seed: 随机种子
            density_scaling: {stage_type_str: {density_tier: scale_factor}}
                             从 profiling 数据生成，用于密度感知仿真
        """
        self._rng = random.Random(seed)
        self.noise_ratio = noise_ratio
        self.base_latencies = base_latencies or self._default_latencies()
        self.density_scaling = density_scaling or {}

    @classmethod
    def from_profiling(
        cls,
        json_path: str,
        density_tier: str = "medium",
        noise_ratio: float = 0.05,
        seed: int | None = None,
    ) -> SimulatedBackend:
        """从 profiling JSON 构建校准后的 SimulatedBackend。

        未被 profiling 覆盖的阶段（如 heatmap, postprocess）自动回退到默认值。
        同时加载密度缩放表，使 simulate() 能根据帧密度调整延迟。
        """
        from pathlib import Path
        from src.inference.calibrator import (
            load_profiling_results, build_base_latencies,
            generate_density_scaling_table,
        )
        profiling = load_profiling_results(Path(json_path))
        calibrated = build_base_latencies(profiling, density_tier)
        # 以默认延迟为底，用校准值覆盖
        latencies = cls._default_latencies()
        latencies.update(calibrated)
        # 加载密度缩放表
        density_scaling = generate_density_scaling_table(profiling, reference_tier=density_tier)
        return cls(
            base_latencies=latencies, noise_ratio=noise_ratio,
            seed=seed, density_scaling=density_scaling,
        )

    @staticmethod
    def _default_latencies() -> dict[tuple[StageType, ProcessorType], float]:
        return {
            # 通用
            (StageType.DECODE, ProcessorType.CPU): 5.0,
            (StageType.POSTPROCESS, ProcessorType.CPU): 3.0,
            # Pipeline A: 行人监控
            (StageType.DETECT, ProcessorType.CPU): 80.0,
            (StageType.DETECT, ProcessorType.GPU): 15.0,
            (StageType.TRACK, ProcessorType.CPU): 20.0,
            (StageType.TRACK, ProcessorType.GPU): 5.0,
            (StageType.REID, ProcessorType.CPU): 60.0,
            (StageType.REID, ProcessorType.GPU): 12.0,
            # Pipeline B: 姿态分析
            (StageType.POSE_ESTIMATE, ProcessorType.CPU): 120.0,
            (StageType.POSE_ESTIMATE, ProcessorType.GPU): 25.0,
            (StageType.BEHAVIOR_CLASSIFY, ProcessorType.CPU): 10.0,
            (StageType.BEHAVIOR_CLASSIFY, ProcessorType.GPU): 3.0,
            # Pipeline C: 车辆监控
            (StageType.CROP, ProcessorType.CPU): 2.0,
            (StageType.LPR, ProcessorType.CPU): 50.0,
            (StageType.LPR, ProcessorType.GPU): 10.0,
            # Pipeline D: 场景理解
            (StageType.SEGMENT, ProcessorType.CPU): 200.0,
            (StageType.SEGMENT, ProcessorType.GPU): 40.0,
            (StageType.CROWD_COUNT, ProcessorType.CPU): 8.0,
            (StageType.HEATMAP, ProcessorType.CPU): 10.0,
            # Education pipeline
            (StageType.FACE_EMOTION, ProcessorType.CPU): 95.0,
            (StageType.FACE_EMOTION, ProcessorType.GPU): 38.0,
            (StageType.HEAD_GAZE, ProcessorType.CPU): 45.0,
            (StageType.EDU_BEHAVIOR, ProcessorType.CPU): 240.0,
            (StageType.EDU_BEHAVIOR, ProcessorType.GPU): 105.0,
            (StageType.STATE_FUSION, ProcessorType.CPU): 2.0,
        }

    # GPU 并发争用退化因子表：
    # 边缘 GPU（Jetson Nano/Xavier NX）显存和计算单元有限，
    # 多个推理任务并发时需要上下文切换、显存争用，延迟非线性增长。
    # per-object 推理（reid, pose_estimate）受影响最大，因为每个目标
    # 单独推理，高并发时 batch 在 GPU 上排队。
    # whole-image 推理（detect, segment）影响较小。
    #
    # 格式: {stage_type_str: [1并发, 2并发, 3并发, 4+并发]}
    # 值为延迟乘数，1.0 = 无退化
    _GPU_CONTENTION_TABLE: dict[str, list[float]] = {
        # per-object 推理：并发退化严重
        "reid":            [1.0, 1.8, 3.2, 5.0],
        "pose_estimate":   [1.0, 1.8, 3.2, 5.0],
        # tracking 状态争用
        "track":           [1.0, 1.3, 1.8, 2.5],
        # whole-image 推理：退化较小
        "detect":          [1.0, 1.2, 1.5, 1.8],
        "segment":         [1.0, 1.2, 1.5, 1.8],
        "face_emotion":    [1.0, 1.2, 1.5, 1.8],
        "edu_behavior":    [1.0, 1.5, 2.4, 3.8],
        # 轻量任务
        "behavior_classify": [1.0, 1.1, 1.2, 1.3],
        "lpr":             [1.0, 1.2, 1.5, 1.8],
    }

    def _gpu_contention_factor(self, stage_type: StageType, gpu_concurrent: int) -> float:
        """查表得到 GPU 并发争用下的延迟乘数。"""
        table = self._GPU_CONTENTION_TABLE.get(stage_type.value)
        if table is None or gpu_concurrent <= 1:
            return 1.0
        idx = min(gpu_concurrent - 1, len(table) - 1)
        return table[idx]

    def simulate(
        self, stage_type: StageType, processor: ProcessorType,
        density_level: str = "medium",
        call_count: int = 0,
        gpu_concurrent: int = 0,
    ) -> float:
        """返回模拟延迟值（ms），不做 sleep。供离散事件仿真使用。

        模拟三个影响延迟的因素：
        1. 密度缩放：不同人群密度下推理延迟的变化（从 profiling 数据）
        2. GPU 运行时漂移：GPU 持续运行后因热降频等因素延迟逐渐增长
        3. GPU 并发争用：多个推理任务共享 GPU 时的上下文切换和显存争用
           per-object 推理（reid/pose）受影响最大，whole-image 较小

        Args:
            stage_type: 阶段类型
            processor: 处理器类型
            density_level: 当前帧密度档 ("low"/"medium"/"high")
            call_count: 已执行的 stage 总数，驱动 GPU 运行时漂移
            gpu_concurrent: 当前 GPU 上并发执行的 stage 数量
        """
        key = (stage_type, processor)
        base = self.base_latencies.get(key)
        if base is None:
            raise ValueError(f"No latency configured for {stage_type.value} on {processor.value}")
        noise = self._rng.uniform(-self.noise_ratio, self.noise_ratio)

        # 密度缩放
        density_scale = 1.0
        stage_scaling = self.density_scaling.get(stage_type.value)
        if stage_scaling:
            proc_key = "gpu" if processor == ProcessorType.GPU else "cpu"
            proc_tiers = stage_scaling.get(proc_key, stage_scaling.get("cpu", {}))
            density_scale = proc_tiers.get(density_level, 1.0)

        # GPU 运行时漂移：模拟边缘 GPU 持续负载下的热降频
        gpu_drift = 0.0
        if processor == ProcessorType.GPU and call_count > 0:
            base_drift = min(1.20, call_count * 0.0015)
            density_boost = {"low": 0.0, "medium": 0.0, "high": 0.3}
            gpu_drift = base_drift + density_boost.get(density_level, 0.0)

        # GPU 并发争用退化
        contention = 1.0
        if processor == ProcessorType.GPU and gpu_concurrent > 1:
            contention = self._gpu_contention_factor(stage_type, gpu_concurrent)

        return base * density_scale * contention * (1.0 + noise + gpu_drift)

    def run(
        self, stage_type: StageType, processor: ProcessorType, input_data: Any
    ) -> tuple[Any, float]:
        key = (stage_type, processor)
        base = self.base_latencies.get(key)
        if base is None:
            raise ValueError(f"No latency configured for {stage_type.value} on {processor.value}")

        noise = self._rng.uniform(-self.noise_ratio, self.noise_ratio)
        latency = base * (1.0 + noise)

        # 模拟延迟（用 sleep 模拟真实耗时，但可以通过 time_scale 加速）
        time.sleep(latency / 1000.0)

        return {"stage": stage_type.value, "result": "simulated"}, latency


class PyTorchBackend(InferenceBackend):
    """PyTorch MPS 后端（需要 torch + torchvision）。"""

    def __init__(self):
        try:
            import torch
            self._torch = torch
            self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        except ImportError as e:
            raise ImportError("PyTorch is required for PyTorchBackend") from e
        self._models: dict[StageType, Any] = {}

    def _get_model(self, stage_type: StageType) -> Any:
        if stage_type not in self._models:
            if stage_type == StageType.DETECT:
                from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
                model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
            elif stage_type == StageType.REID:
                from torchvision.models import mobilenet_v3_small
                model = mobilenet_v3_small(pretrained=False)
            else:
                return None
            model = model.to(self._device).eval()
            self._models[stage_type] = model
        return self._models[stage_type]

    def run(
        self, stage_type: StageType, processor: ProcessorType, input_data: Any
    ) -> tuple[Any, float]:
        model = self._get_model(stage_type)
        if model is None:
            return {"stage": stage_type.value, "result": "passthrough"}, 0.1

        torch = self._torch
        dummy = torch.randn(1, 3, 224, 224, device=self._device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model(dummy)
        if self._device.type == "mps":
            torch.mps.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

        return {"stage": stage_type.value, "result": "pytorch"}, latency_ms


class ONNXBackend(InferenceBackend):
    """ONNX Runtime 后端。"""

    def __init__(self, model_dir: str = "data/models"):
        try:
            import onnxruntime as ort
            self._ort = ort
        except ImportError as e:
            raise ImportError("onnxruntime is required for ONNXBackend") from e
        self._model_dir = model_dir
        self._sessions: dict[StageType, Any] = {}

    def run(
        self, stage_type: StageType, processor: ProcessorType, input_data: Any
    ) -> tuple[Any, float]:
        import os
        import numpy as np

        model_path = os.path.join(self._model_dir, f"{stage_type.value}.onnx")
        if not os.path.exists(model_path):
            # 没有模型文件时回退到模拟
            return {"stage": stage_type.value, "result": "onnx_fallback"}, 1.0

        if stage_type not in self._sessions:
            providers = ["CPUExecutionProvider"]
            self._sessions[stage_type] = self._ort.InferenceSession(
                model_path, providers=providers
            )

        session = self._sessions[stage_type]
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        latency_ms = (time.perf_counter() - start) * 1000

        return {"stage": stage_type.value, "result": "onnx"}, latency_ms
