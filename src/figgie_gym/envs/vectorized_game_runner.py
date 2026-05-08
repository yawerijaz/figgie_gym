from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

type AgentPoolID = int
if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    import numpy as np

    from figgie_gym.envs.common import (
        ActionType,
        Agent,
        AgentID,
        BeliefType,
        ExtraType,
        ObsType,
        RewardType,
    )
    from figgie_gym.envs.vectorized_parallel_figgie_env import (
        VectorizedParallelFiggieEnv,
    )

    # AgentID is the ID in the env - the seat number.
    # AgentPoolID is the ID in the agent pool defining the behavior of the agent.
    type EnvID = int


@dataclass
class EnvAgentStepSnapshot:
    current_observation: ObsType
    belief: BeliefType
    action: ActionType
    reward: RewardType
    next_observation: ObsType
    terminated: bool
    truncated: bool
    extra: ExtraType


class VectorizedGameRunner:
    """Run multiple vectorized environments with a shared agent pool.

    Each environment uses the same set of agent objects, but their seating
    (which pool index maps to which seat/AgentID in the env) is randomized per
    environment to avoid learning seat-order shortcuts. This class handles
    routing observations/actions/rewards between env seat IDs and agent
    identities.
    """

    def __init__(
        self,
        vec_env: VectorizedParallelFiggieEnv,
        agent_pool: Iterable[Agent],
        rng: np.random.Generator,
        num_steps: int,
    ) -> None:
        self.vec_env = vec_env
        self.agent_pool = list(agent_pool)
        self.rng = rng
        self.num_steps = num_steps

        if len(self.agent_pool) == 0:
            msg = "agent_pool must contain at least one agent"
            raise ValueError(msg)

        if any(
            len(env.possible_agents) != len(self.agent_pool)
            for env in self.vec_env.envs
        ):
            msg = "All environments must have the same number of agent seats as the agent_pool length"
            raise ValueError(msg)

        self.num_agents = len(self.agent_pool)

        # Seating map: for each env, map AgentPoolID -> AgentID (seat)
        self.seating: list[dict[AgentPoolID, AgentID]] = [
            dict(
                enumerate(
                    self.rng.permutation(self.num_agents),
                ),
            )
            for _ in range(self.vec_env.num_envs)
        ]
        # Build inverse mapping: for each env map seat (AgentID) -> agent_pool_index
        self.seating_inverse: list[dict[AgentID, AgentPoolID]] = [
            {seat: pool for pool, seat in seating.items()}
            for seating in self.seating
        ]

    def act_batch(
        self,
        env_obs_by_pool: list[dict[AgentPoolID, ObsType]],
    ) -> tuple[
        list[dict[AgentPoolID, BeliefType]],
        list[dict[AgentPoolID, ActionType]],
        list[dict[AgentPoolID, ExtraType]],
    ]:
        """Produce actions in batch.

        Batch observations across envs by agent object, call each agent's act_batch.
        """
        beliefs: list[dict[AgentPoolID, BeliefType]] = [
            {} for _ in range(self.vec_env.num_envs)
        ]
        actions: list[dict[AgentPoolID, ActionType]] = [
            {} for _ in range(self.vec_env.num_envs)
        ]
        extras: list[dict[AgentPoolID, ExtraType]] = [
            {} for _ in range(self.vec_env.num_envs)
        ]
        for agent_pool_id, agent_obj in enumerate(self.agent_pool):
            observations = [x[agent_pool_id] for x in env_obs_by_pool]
            beliefs_actions = agent_obj.act_batch(self.rng, observations)
            for env_idx, (belief, action, extra) in enumerate(beliefs_actions):
                beliefs[env_idx][agent_pool_id] = belief
                actions[env_idx][agent_pool_id] = action
                extras[env_idx][agent_pool_id] = extra
        return beliefs, actions, extras

    def run(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Generator[list[dict[AgentPoolID, EnvAgentStepSnapshot]]]:
        """Run episodes across the vectorized environments and yield per-step snapshots.

        Yields a list of EnvStepSnapshots for each environment where keys inside each snapshot
        are `agent_pool_index`, i.e. which agent from the shared pool.
        """
        # reset environments (Vectorized env will inject its own child RNGs when given seed)
        env_obs_list, _infos = self.vec_env.reset(seed=seed, options=options)

        # Map observations to agent_pool indices according to ordering
        env_obs_by_pool = self.reindex_by_agent_pool_id(env_obs_list)

        for _step in range(self.num_steps + 1):
            beliefs_list_by_pool, actions_list_by_pool, extras_list_by_pool = (
                self.act_batch(
                    env_obs_by_pool,
                )
            )

            (
                next_obs_list,
                rewards_list,
                terminations_list,
                truncations_list,
                _infos,
            ) = self.vec_env.step(
                self.reindex_by_agent_id(actions_list_by_pool),
            )

            env_next_obs_by_pool = self.reindex_by_agent_pool_id(next_obs_list)
            env_rewards_by_pool = self.reindex_by_agent_pool_id(rewards_list)
            env_terminations_by_pool = self.reindex_by_agent_pool_id(
                terminations_list,
            )
            env_truncations_by_pool = self.reindex_by_agent_pool_id(
                truncations_list,
            )

            # Build and yield snapshots
            env_snapshots = [
                {
                    agent_pool_id: EnvAgentStepSnapshot(
                        current_observation=env_obs_by_pool[env_idx][
                            agent_pool_id
                        ],
                        belief=beliefs_list_by_pool[env_idx][agent_pool_id],
                        action=actions_list_by_pool[env_idx][agent_pool_id],
                        reward=env_rewards_by_pool[env_idx][agent_pool_id],
                        next_observation=env_next_obs_by_pool[env_idx][
                            agent_pool_id
                        ],
                        terminated=env_terminations_by_pool[env_idx][
                            agent_pool_id
                        ],
                        truncated=env_truncations_by_pool[env_idx][
                            agent_pool_id
                        ],
                        extra=extras_list_by_pool[env_idx][agent_pool_id],
                    )
                    for agent_pool_id in range(self.num_agents)
                }
                for env_idx in range(self.vec_env.num_envs)
            ]

            yield env_snapshots

            env_obs_by_pool = env_next_obs_by_pool

            are_all_terminated = all(
                all(v.terminated for v in snap.values())
                for snap in env_snapshots
            )
            are_all_truncated = all(
                all(v.truncated for v in snap.values())
                for snap in env_snapshots
            )
            if are_all_terminated or are_all_truncated:
                break

    def reindex_by_agent_id[T](
        self,
        data_by_agent_pool_id: list[dict[AgentPoolID, T]],
    ) -> list[dict[AgentID, T]]:
        return [
            {
                self.seating[env_id][agent_pool_id]: data
                for agent_pool_id, data in d_by_agent_pool_id.items()
            }
            for env_id, d_by_agent_pool_id in enumerate(data_by_agent_pool_id)
        ]

    def reindex_by_agent_pool_id[T](
        self,
        data_by_agent_id: list[dict[AgentID, T]],
    ) -> list[dict[AgentPoolID, T]]:
        return [
            {
                self.seating_inverse[env_id][agent_id]: data
                for agent_id, data in d_by_agent_id.items()
            }
            for env_id, d_by_agent_id in enumerate(data_by_agent_id)
        ]
