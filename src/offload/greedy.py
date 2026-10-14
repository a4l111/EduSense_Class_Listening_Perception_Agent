"""Greedy-Nearest 卸载策略 (baseline)。

本地预估延迟超过 deadline 时，选最闲邻居卸载整个 DAG。
不考虑网络拥塞、不做选择性卸载。
"""

from __future__ import annotations

from src.dag.graph import DAGTask
from src.registry import OFFLOAD_REGISTRY

from .base import NodeLoad, OffloadDecision, OffloadPolicy


@OFFLOAD_REGISTRY.register("greedy")
class GreedyOffloadPolicy(OffloadPolicy):
    """贪心卸载：本地过载时选最闲邻居卸载整个 DAG。"""

    def __init__(self, utilization_threshold: float = 0.8):
        self.utilization_threshold = utilization_threshold

    def decide(
        self,
        dag: DAGTask,
        local_node_id: str,
        neighbor_loads: list[NodeLoad],
    ) -> OffloadDecision:
        if not neighbor_loads:
            return OffloadDecision(should_offload=False)

        # 检查本地是否过载（调度器预估延迟 > deadline）
        local_est = getattr(dag, "_local_estimated_ms", None)
        if local_est is not None and local_est <= dag.deadline_ms:
            # 本地能搞定，不卸载
            return OffloadDecision(should_offload=False)

        # 本地过载 → 找最闲的邻居
        best_neighbor = min(
            neighbor_loads,
            key=lambda n: (n.cpu_utilization + n.gpu_utilization) / 2 + n.queue_length * 0.01,
        )

        # 邻居也满载时不卸载
        neighbor_load = (best_neighbor.cpu_utilization + best_neighbor.gpu_utilization) / 2
        if neighbor_load >= self.utilization_threshold:
            return OffloadDecision(should_offload=False)

        return OffloadDecision(
            should_offload=True,
            target_node_id=best_neighbor.node_id,
            stages_to_offload=list(dag.stages.keys()),
        )
