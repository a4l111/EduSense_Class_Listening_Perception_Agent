"""DAG-Aware Offloading 策略（创新点二）。

核心理念：选择性卸载 + 拥塞控制。

Greedy 的问题：只要 scheduler 说 needs_offload、邻居有空，就无条件卸载整个 DAG。
在高负载时所有 DAG 都触发卸载 → 网络链路拥塞 → 传输延迟飙升 → 适得其反。

DAO 的改进：
1. DAG 优先级排序：根据 GPU 加速收益对 DAG 进行优先级评分。
   高收益 DAG（如 Pipeline D，SEGMENT GPU加速5.9x）优先卸载。
2. 拥塞感知的准入控制：跟踪最近的卸载频率，当卸载率过高时
   只允许高优先级 DAG 通过，避免低收益 DAG 占用网络带宽。
3. 处理器感知的邻居选择：GPU 密集型 DAG 优先选 GPU 空闲的邻居。

效果：低负载时 greedy ≈ DAO（网络不拥塞，都能卸载成功）；
高负载时 DAO > greedy（DAO 通过拥塞控制避免网络过载，
确保高收益 DAG 优先获得网络带宽）。
"""

from __future__ import annotations

from collections import deque
from typing import Any

from src.dag.graph import DAGTask, ProcessorType
from src.registry import OFFLOAD_REGISTRY

from .base import NodeLoad, OffloadDecision, OffloadPolicy


@OFFLOAD_REGISTRY.register("dao")
class DAGAwareOffloadPolicy(OffloadPolicy):
    """DAG 感知卸载：优先级排序 + 拥塞感知准入控制。"""

    def __init__(
        self,
        bottleneck_ratio: float = 0.3,
        utilization_threshold: float = 0.85,
        network_bandwidth_mbps: float = 100.0,
        congestion_window_ms: float = 500.0,
        max_offloads_per_window: int = 3,
    ):
        """
        Args:
            bottleneck_ratio: 关键路径上存在占比 > 此值的瓶颈 stage 才给予高优先级
            utilization_threshold: 邻居加权利用率低于此值才可作为卸载目标
            network_bandwidth_mbps: 网络带宽（用于估算传输代价）
            congestion_window_ms: 拥塞控制的滑动窗口大小 (ms)
            max_offloads_per_window: 窗口内最大卸载数量
        """
        self.bottleneck_ratio = bottleneck_ratio
        self.utilization_threshold = utilization_threshold
        self.network_bandwidth_mbps = network_bandwidth_mbps
        self.congestion_window_ms = congestion_window_ms
        self.max_offloads_per_window = max_offloads_per_window

        # 卸载历史：记录最近卸载的时间戳
        self._offload_history: deque[float] = deque()
        self._last_trace: dict[str, Any] = {}

    def decide(
        self,
        dag: DAGTask,
        local_node_id: str,
        neighbor_loads: list[NodeLoad],
    ) -> OffloadDecision:
        decision, _ = self.decide_with_trace(dag, local_node_id, neighbor_loads)
        return decision

    def decide_with_trace(
        self,
        dag: DAGTask,
        local_node_id: str,
        neighbor_loads: list[NodeLoad],
    ) -> tuple[OffloadDecision, dict[str, Any]]:
        """返回 DAO 决策以及可展示/可记录的准入控制过程。

        `decide()` 仍保持原接口，仿真实验和旧代码不受影响；实时演示界面
        使用 trace 展示 DAO 的 DAG 优先级、拥塞窗口、候选边缘节点等依据。
        """
        current_time = dag.arrival_time
        priority = self._compute_priority(dag)
        estimated_transfer_ms = self._estimate_transfer_ms(dag)
        self._prune_history(current_time)
        recent_offloads = len(self._offload_history)
        is_gpu_heavy = self._is_gpu_heavy(dag)

        trace: dict[str, Any] = {
            "policy": "DAGAwareOffloadPolicy",
            "dag_id": dag.dag_id,
            "local_node_id": local_node_id,
            "priority": round(float(priority), 4),
            "stage_count": len(dag.stages),
            "deadline_ms": float(dag.deadline_ms),
            "density_level": dag.density_level,
            "is_gpu_heavy": is_gpu_heavy,
            "estimated_transfer_ms": round(float(estimated_transfer_ms), 2),
            "recent_offloads": recent_offloads,
            "congestion_window_ms": float(self.congestion_window_ms),
            "max_offloads_per_window": int(self.max_offloads_per_window),
            "utilization_threshold": float(self.utilization_threshold),
            "admission": "checking",
            "neighbors": [],
        }

        if not neighbor_loads:
            decision = OffloadDecision(should_offload=False)
            trace.update({
                "admission": "reject_no_neighbor",
                "reason": "DAO拒绝：没有可用边缘邻居节点。",
            })
            self._last_trace = trace
            return decision, trace

        # Step 1 + 2: DAG 优先级排序 + 拥塞感知准入控制
        if recent_offloads >= self.max_offloads_per_window:
            if priority < 0.5:
                decision = OffloadDecision(should_offload=False)
                trace.update({
                    "admission": "reject_congested_low_priority",
                    "reason": (
                        "DAO拒绝：滑动窗口内卸载请求已满，"
                        "且当前教育DAG优先级低于高优先级阈值。"
                    ),
                })
                self._last_trace = trace
                return decision, trace
            trace["admission"] = "congested_high_priority_allowed"
        else:
            trace["admission"] = "window_available"

        # Step 3: 处理器感知的邻居选择
        ranked_neighbors = self._rank_neighbors(
            neighbor_loads,
            is_gpu_heavy,
            estimated_transfer_ms,
        )
        trace["neighbors"] = [
            {k: v for k, v in item.items() if k != "node"}
            for item in ranked_neighbors
        ]

        candidates = [item for item in ranked_neighbors if item["accepted"]]
        best = candidates[0]["node"] if candidates else None
        if best is None:
            decision = OffloadDecision(should_offload=False)
            trace.update({
                "admission": "reject_no_qualified_neighbor",
                "reason": "DAO拒绝：边缘节点加权 CPU/GPU 利用率或链路代价过高。",
            })
            self._last_trace = trace
            return decision, trace

        # 记录卸载
        self._offload_history.append(current_time)

        decision = OffloadDecision(
            should_offload=True,
            target_node_id=best.node_id,
            stages_to_offload=list(dag.stages.keys()),
        )
        trace.update({
            "admission": "accept",
            "target_node_id": best.node_id,
            "stages_to_offload": decision.stages_to_offload,
            "reason": "DAO通过：教育DAG具备加速收益，且目标边缘节点负载与链路条件可接受。",
        })
        self._last_trace = trace
        return decision, trace

    def _prune_history(self, current_time: float) -> None:
        while (
            self._offload_history
            and self._offload_history[0] < current_time - self.congestion_window_ms
        ):
            self._offload_history.popleft()

    def _estimate_transfer_ms(self, dag: DAGTask) -> float:
        """估算整条 DAG 卸载时的网络传输代价。"""
        if self.network_bandwidth_mbps <= 0:
            return float("inf")

        entry_bytes = sum(dag.stages[sid].input_size_bytes for sid in dag.entry_stages)
        exit_bytes = sum(dag.stages[sid].output_size_bytes for sid in dag.exit_stages)
        total_bytes = entry_bytes + exit_bytes
        return total_bytes * 8 / (self.network_bandwidth_mbps * 1_000_000) * 1000

    def _compute_priority(self, dag: DAGTask) -> float:
        """计算 DAG 的卸载优先级 (0~1)。

        基于 GPU 加速带来的关键路径缩短比例。
        比例越大 → 优先级越高 → 越应该卸载。
        """
        topo_order = dag.topological_order()

        # CPU-only 关键路径
        cpu_dist: dict[str, float] = {}
        for sid in topo_order:
            stage = dag.stages[sid]
            cpu_lat = stage.base_latency.get(ProcessorType.CPU, float("inf"))
            pred_max = max((cpu_dist[p] for p in stage.predecessors), default=0.0)
            cpu_dist[sid] = pred_max + cpu_lat

        # 最优关键路径
        opt_dist: dict[str, float] = {}
        for sid in topo_order:
            stage = dag.stages[sid]
            min_lat = min(stage.base_latency.values())
            pred_max = max((opt_dist[p] for p in stage.predecessors), default=0.0)
            opt_dist[sid] = pred_max + min_lat

        cpu_critical = max(cpu_dist.values()) if cpu_dist else 0.0
        opt_critical = max(opt_dist.values()) if opt_dist else 0.0

        if cpu_critical == 0:
            return 0.0

        # 加速比例 = (CPU-only - 最优) / CPU-only
        speedup_ratio = (cpu_critical - opt_critical) / cpu_critical
        return min(speedup_ratio, 1.0)

    def _is_gpu_heavy(self, dag: DAGTask) -> bool:
        """判断 DAG 是否 GPU 密集型。"""
        gpu_stages = sum(
            1 for s in dag.stages.values()
            if ProcessorType.GPU in s.supported_processors
        )
        return gpu_stages > len(dag.stages) / 2

    def _select_neighbor(
        self,
        neighbor_loads: list[NodeLoad],
        is_gpu_heavy: bool,
    ) -> NodeLoad | None:
        """处理器感知的邻居选择。"""
        candidates = [
            item for item in self._rank_neighbors(neighbor_loads, is_gpu_heavy, 0.0)
            if item["accepted"]
        ]
        if not candidates:
            return None

        return candidates[0]["node"]

    def _rank_neighbors(
        self,
        neighbor_loads: list[NodeLoad],
        is_gpu_heavy: bool,
        estimated_transfer_ms: float,
    ) -> list[dict[str, Any]]:
        """按 DAO 的处理器感知规则给候选边缘节点排序。"""
        gpu_weight = 0.7 if is_gpu_heavy else 0.3

        ranked: list[dict[str, Any]] = []
        for n in neighbor_loads:
            weighted_util = (
                n.gpu_utilization * gpu_weight
                + n.cpu_utilization * (1 - gpu_weight)
            )
            link_cost_ms = n.network_latency_ms + estimated_transfer_ms
            queue_penalty = min(max(n.queue_length, 0) / 10.0, 1.0)
            link_penalty = min(link_cost_ms / 100.0, 1.0)
            selection_score = (
                weighted_util * 0.60
                + link_penalty * 0.25
                + queue_penalty * 0.15
            )
            ranked.append({
                "node": n,
                "node_id": n.node_id,
                "cpu_utilization": round(float(n.cpu_utilization), 4),
                "gpu_utilization": round(float(n.gpu_utilization), 4),
                "queue_length": int(n.queue_length),
                "network_latency_ms": round(float(n.network_latency_ms), 2),
                "weighted_utilization": round(float(weighted_util), 4),
                "link_cost_ms": round(float(link_cost_ms), 2),
                "selection_score": round(float(selection_score), 4),
                "accepted": weighted_util < self.utilization_threshold,
            })

        return sorted(ranked, key=lambda item: item["selection_score"])
