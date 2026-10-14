"""Niagara+-style 两步解耦卸载策略 (baseline)。

核心思想 (来自 Niagara+ TSC 2025):
  离线阶段: 对每种 (负载等级, 网络状态) 预计算最优调度模板
  在线阶段: 匹配当前状态到最近模板，直接使用其卸载决策

简化实现:
  - 用 profiling 数据计算各节点真实延迟（不用固定 speedup）
  - 在线时比较本地 EFT vs 卸载 EFT（含传输 + 排队 + 远端计算）
  - 整 DAG 卸载，无拥塞控制
"""

from __future__ import annotations

from src.dag.graph import DAGTask, ProcessorType, StageType
from src.registry import OFFLOAD_REGISTRY

from .base import NodeLoad, OffloadDecision, OffloadPolicy


# 各节点的 DETECT GPU 延迟（用于计算真实加速比）
# 由 profiling 数据驱动，而非固定 speedup
_NODE_DETECT_GPU_MS: dict[str, float] = {
    "node_a": 91.0,   # S10e
    "node_b": 60.0,   # 一加9
    "node_c": 4.6,    # Mac M3 Pro
}

# 各节点的关键路径延迟估算（基于 profiling，Pipeline A high density）
_NODE_CRITICAL_PATH_MS: dict[str, float] = {
    "node_a": 279.0,  # S10e: 0.1 + 91 + 0.1 + max(15.4, 188) + 0.1
    "node_b": 366.0,  # 一加9 真机: 0.1 + 99 + 0.2 + max(149, 267) + 0.1
    "node_c": 25.0,   # Mac: 0.1 + 4.6 + 0.03 + max(6.2, 20) + 0.1
}


@OFFLOAD_REGISTRY.register("niagara_plus")
class NiagaraPlusOffloadPolicy(OffloadPolicy):
    """Niagara+-style: 两步解耦——基于 profiling 的本地 vs 卸载代价比较。"""

    def __init__(
        self,
        network_bandwidth_mbps: float = 100.0,
        node_critical_paths: dict[str, float] | None = None,
    ):
        self.network_bandwidth_mbps = network_bandwidth_mbps
        self._node_cp = node_critical_paths or _NODE_CRITICAL_PATH_MS

    def decide(
        self,
        dag: DAGTask,
        local_node_id: str,
        neighbor_loads: list[NodeLoad],
    ) -> OffloadDecision:
        if not neighbor_loads:
            return OffloadDecision(should_offload=False)

        # Step 1: 本地代价 = 调度器预估（含队列排队）
        local_cost = getattr(dag, "_local_estimated_ms", None)
        if local_cost is None:
            local_cost = self._node_cp.get(local_node_id, 279.0)

        # 本地不过载时不卸载（保守策略，避免不必要的网络开销）
        if local_cost <= dag.deadline_ms * 0.8:
            return OffloadDecision(should_offload=False)

        # Step 2: 对每个邻居计算卸载 EFT
        best_neighbor = None
        best_cost = local_cost  # 只有比本地好才卸载

        for neighbor in neighbor_loads:
            cost = self._estimate_offload_cost(dag, neighbor)
            if cost < best_cost:
                best_cost = cost
                best_neighbor = neighbor

        if best_neighbor is not None:
            return OffloadDecision(
                should_offload=True,
                target_node_id=best_neighbor.node_id,
                stages_to_offload=list(dag.stages.keys()),
            )

        return OffloadDecision(should_offload=False)

    def _estimate_offload_cost(
        self, dag: DAGTask, neighbor: NodeLoad,
    ) -> float:
        """估算卸载到邻居的总延迟 = 传输 + 排队 + 远端计算 + 回传。"""
        # 传输数据量
        upload_bytes = sum(
            dag.stages[sid].input_size_bytes for sid in dag.entry_stages
        )
        download_bytes = sum(
            dag.stages[sid].output_size_bytes for sid in dag.exit_stages
        )

        # 传输延迟
        bw_bps = self.network_bandwidth_mbps * 1_000_000
        upload_ms = (upload_bytes * 8 / bw_bps * 1000) if bw_bps > 0 else float("inf")
        download_ms = (download_bytes * 8 / bw_bps * 1000) if bw_bps > 0 else float("inf")
        rtt_ms = neighbor.network_latency_ms * 2

        # 远端计算: 用 profiling 数据中该节点的关键路径
        remote_compute = self._node_cp.get(neighbor.node_id, 100.0)

        # 远端排队: 基于利用率估算等待时间
        # GPU 利用率高 → 排队等待长
        queue_wait = 0.0
        if neighbor.gpu_utilization > 0.3:
            queue_wait = remote_compute * neighbor.gpu_utilization
        if neighbor.queue_length > 0:
            queue_wait += neighbor.queue_length * remote_compute * 0.5

        return upload_ms + rtt_ms + queue_wait + remote_compute + download_ms
