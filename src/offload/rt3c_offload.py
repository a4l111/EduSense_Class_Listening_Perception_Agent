"""RT3C-style RL 卸载策略 (baseline)。

核心思想 (来自 RT3C TSC 2024):
  基于 Actor-Critic 的强化学习卸载决策。

简化实现:
  - 单智能体 Q-learning（原版是多智能体 MARL + 多头注意力）
  - 6 维离散状态 + ε-greedy 探索
  - 在线学习（仿真引擎每帧回调 report_result）
  - 整 DAG 卸载
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from src.dag.graph import DAGTask, ProcessorType
from src.registry import OFFLOAD_REGISTRY

from .base import NodeLoad, OffloadDecision, OffloadPolicy


def _discretize(value: float, bins: list[float]) -> int:
    """将连续值离散化到 bin index。"""
    for i, threshold in enumerate(bins):
        if value <= threshold:
            return i
    return len(bins)


class QLearningAgent:
    """表格式 Q-learning agent。"""

    def __init__(
        self,
        n_actions: int,
        lr: float = 0.15,
        gamma: float = 0.9,
        epsilon_start: float = 0.5,
        epsilon_decay: float = 0.998,
        epsilon_min: float = 0.05,
        seed: int = 42,
    ):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions)
        )
        self._prev_state: tuple | None = None
        self._prev_action: int | None = None

    def select_action(self, state: tuple) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.n_actions - 1)
        q_values = self.q_table[state]
        max_q = q_values.max()
        best_actions = np.where(q_values == max_q)[0]
        return int(self._np_rng.choice(best_actions))

    def update(self, state: tuple, action: int, reward: float, next_state: tuple):
        best_next_q = self.q_table[next_state].max()
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def step(self, state: tuple, reward: float | None = None) -> int:
        if reward is not None and self._prev_state is not None:
            self.update(self._prev_state, self._prev_action, reward, state)

        action = self.select_action(state)
        self._prev_state = state
        self._prev_action = action
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action


@OFFLOAD_REGISTRY.register("rt3c")
class RT3COffloadPolicy(OffloadPolicy):
    """RT3C-style: Q-learning 驱动的卸载决策。

    状态空间 (6 维，离散化):
      s0: 本地过载程度 = local_est / deadline (4 bins)
      s1: 最优邻居 GPU 利用率 (3 bins)
      s2: 最优邻居排队深度 (3 bins)
      s3: 网络延迟等级 (3 bins)
      s4: DAG 密度等级 (3 bins: low/medium/high)
      s5: 上次卸载是否成功 (2 bins: 0=失败或首次, 1=成功)

    动作空间:
      0: 不卸载
      1+: 卸载到邻居 i（按 GPU 利用率排序）
    """

    def __init__(
        self,
        max_neighbors: int = 3,
        deadline_ms: float = 500.0,
        lr: float = 0.15,
        gamma: float = 0.9,
        seed: int = 42,
    ):
        self.max_neighbors = max_neighbors
        self.deadline_ms = deadline_ms
        n_actions = 1 + max_neighbors

        self._agent = QLearningAgent(
            n_actions=n_actions, lr=lr, gamma=gamma, seed=seed,
        )

        self._last_dag_latency: float | None = None
        self._last_offloaded: bool = False

        # 离散化 bins
        self._overload_bins = [0.5, 0.8, 1.0, 1.5]  # local_est/deadline 比值
        self._util_bins = [0.3, 0.7]
        self._queue_bins = [0, 2, 5]
        self._latency_bins = [10.0, 30.0]
        self._density_map = {"low": 0, "medium": 1, "high": 2}

    def _extract_state(
        self,
        dag: DAGTask,
        neighbor_loads: list[NodeLoad],
    ) -> tuple:
        # s0: 本地过载程度
        local_est = getattr(dag, "_local_estimated_ms", self.deadline_ms)
        overload_ratio = local_est / self.deadline_ms if self.deadline_ms > 0 else 1.0
        s0 = _discretize(overload_ratio, self._overload_bins)

        # s1, s2, s3: 最优邻居信息
        if neighbor_loads:
            best = min(neighbor_loads, key=lambda n: n.gpu_utilization)
            s1 = _discretize(best.gpu_utilization, self._util_bins)
            s2 = _discretize(best.queue_length, self._queue_bins)
            s3 = _discretize(best.network_latency_ms, self._latency_bins)
        else:
            s1, s2, s3 = 2, 3, 2  # 最差状态

        # s4: 密度
        s4 = self._density_map.get(dag.density_level, 1)

        # s5: 上次卸载结果
        s5 = 0
        if self._last_dag_latency is not None:
            s5 = 1 if self._last_dag_latency <= self.deadline_ms else 0

        return (s0, s1, s2, s3, s4, s5)

    def _compute_reward(self, dag_latency: float) -> float:
        """SLO 内越快越好，违约重罚。"""
        if dag_latency <= self.deadline_ms:
            # 归一化到 [-1, 0]：越快越接近 0
            return -dag_latency / self.deadline_ms
        # 违约：根据超时严重程度罚分
        overshoot = dag_latency / self.deadline_ms
        return -1.0 - overshoot  # 超 1× → -2, 超 2× → -3

    def report_result(self, dag_latency: float) -> None:
        """仿真引擎回调：报告 DAG 延迟用于 Q-table 更新。"""
        self._last_dag_latency = dag_latency

    def decide(
        self,
        dag: DAGTask,
        local_node_id: str,
        neighbor_loads: list[NodeLoad],
    ) -> OffloadDecision:
        if not neighbor_loads:
            return OffloadDecision(should_offload=False)

        state = self._extract_state(dag, neighbor_loads)

        # 用上次延迟计算奖励
        reward = None
        if self._last_dag_latency is not None:
            reward = self._compute_reward(self._last_dag_latency)
            self._last_dag_latency = None

        action = self._agent.step(state, reward)

        # action=0: 不卸载
        if action == 0:
            self._last_offloaded = False
            return OffloadDecision(should_offload=False)

        # action=i: 卸载到第 i 个邻居（按 GPU 利用率排序）
        sorted_neighbors = sorted(neighbor_loads, key=lambda n: n.gpu_utilization)
        idx = min(action - 1, len(sorted_neighbors) - 1)
        target = sorted_neighbors[idx]

        self._last_offloaded = True
        return OffloadDecision(
            should_offload=True,
            target_node_id=target.node_id,
            stages_to_offload=list(dag.stages.keys()),
        )
