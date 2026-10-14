"""Random 卸载策略 (baseline)。

以概率 p 随机决定是否卸载整个 DAG，随机选一个邻居。
"""

from __future__ import annotations

import random

from src.dag.graph import DAGTask
from src.registry import OFFLOAD_REGISTRY

from .base import NodeLoad, OffloadDecision, OffloadPolicy


@OFFLOAD_REGISTRY.register("random")
class RandomOffloadPolicy(OffloadPolicy):
    """随机卸载：以固定概率卸载到随机邻居。"""

    def __init__(self, offload_probability: float = 0.5, seed: int = 42):
        self.offload_probability = offload_probability
        self._rng = random.Random(seed)

    def decide(
        self,
        dag: DAGTask,
        local_node_id: str,
        neighbor_loads: list[NodeLoad],
    ) -> OffloadDecision:
        if not neighbor_loads:
            return OffloadDecision(should_offload=False)

        if self._rng.random() >= self.offload_probability:
            return OffloadDecision(should_offload=False)

        target = self._rng.choice(neighbor_loads)
        return OffloadDecision(
            should_offload=True,
            target_node_id=target.node_id,
            stages_to_offload=list(dag.stages.keys()),
        )
