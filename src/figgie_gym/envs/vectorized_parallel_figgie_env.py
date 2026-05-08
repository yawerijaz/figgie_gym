"""VectorizedParallelFiggieEnv: A simple vectorized wrapper for ParallelFiggieEnv."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from figgie_gym.envs.common import (
    ActionType,
    AgentID,
    InfoType,
    ObsType,
    RewardType,
)
from figgie_gym.envs.parallel_figgie_env import ParallelFiggieEnv


class VectorizedParallelFiggieEnv:
    def __init__(self, envs: Iterable[ParallelFiggieEnv]) -> None:
        self.envs: list[ParallelFiggieEnv] = list(envs)
        self.num_envs: int = len(self.envs)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[dict[AgentID, ObsType]], list[dict[AgentID, InfoType]]]:
        """Reset all environments, injecting a child RNG into each via options['rng']."""
        obs: list[dict[AgentID, ObsType]] = []
        infos: list[dict[AgentID, InfoType]] = []

        parent_rng = np.random.default_rng(seed)
        child_rngs = parent_rng.spawn(self.num_envs)

        for env, child_rng in zip(self.envs, child_rngs, strict=True):
            opts = options or {}

            ob, info = env.reset(options=opts | {"rng": child_rng})
            obs.append(ob)
            infos.append(info)

        return obs, infos

    def step(
        self,
        actions: list[dict[AgentID, ActionType]],
    ) -> tuple[
        list[dict[AgentID, ObsType]],
        list[dict[AgentID, RewardType]],
        list[dict[AgentID, bool]],
        list[dict[AgentID, bool]],
        list[dict[AgentID, InfoType]],
    ]:
        """Step all environments with batched actions."""
        obs: list[dict[AgentID, ObsType]] = []
        rewards: list[dict[AgentID, RewardType]] = []
        terminations: list[dict[AgentID, bool]] = []
        truncations: list[dict[AgentID, bool]] = []
        infos: list[dict[AgentID, InfoType]] = []
        for env, action in zip(self.envs, actions, strict=True):
            o, r, terminated, truncated, info = env.step(action)
            obs.append(o)
            rewards.append(r)
            terminations.append(terminated)
            truncations.append(truncated)
            infos.append(info)
        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        for env in self.envs:
            env.close()
