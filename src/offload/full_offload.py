"""Full Offloading 策略 (baseline)。

总是卸载整个 DAG 到负载最低的邻居。
"""

from __future__ import annotations

from src.dag.graph import DAGTask
from src.registry import OFFLOAD_REGISTRY

from .base import NodeLoad, OffloadDecision, OffloadPolicy


@OFFLOAD_REGISTRY.register("full")
class FullOffloadPolicy(OffloadPolicy):
    """全量卸载：总是卸载到最闲的邻居。"""

    def decide(
        self,
        dag: DAGTask,
        local_node_id: str,
        neighbor_loads: list[NodeLoad],
    ) -> OffloadDecision:
        if not neighbor_loads:
            return OffloadDecision(should_offload=False)

        best = min(
            neighbor_loads,
            key=lambda n: (n.cpu_utilization + n.gpu_utilization) / 2,
        )

        return OffloadDecision(
            should_offload=True,
            target_node_id=best.node_id,
            stages_to_offload=list(dag.stages.keys()),
        )
