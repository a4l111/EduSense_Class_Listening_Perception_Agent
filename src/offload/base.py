"""卸载决策抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dag.graph import DAGTask


@dataclass
class NodeLoad:
    """节点负载信息。"""
    node_id: str
    cpu_utilization: float       # 0.0 ~ 1.0
    gpu_utilization: float       # 0.0 ~ 1.0
    queue_length: int            # 等待队列长度
    network_latency_ms: float    # 到本节点的网络延迟


@dataclass
class OffloadDecision:
    """卸载决策结果。"""
    should_offload: bool
    target_node_id: str | None = None
    stages_to_offload: list[str] = field(default_factory=list)  # 要卸载的 stage_id 列表


class OffloadPolicy(ABC):
    """卸载策略抽象基类。"""

    @abstractmethod
    def decide(
        self,
        dag: DAGTask,
        local_node_id: str,
        neighbor_loads: list[NodeLoad],
    ) -> OffloadDecision:
        """决定是否卸载以及卸载到哪个节点。

        Args:
            dag: 需要调度的 DAG 任务
            local_node_id: 当前节点 ID
            neighbor_loads: 邻居节点的负载信息
        """
        ...
