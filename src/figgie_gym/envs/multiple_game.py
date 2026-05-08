from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Generator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Self

import numpy as np

from figgie_gym.envs.common import (
    SUIT_TO_CODE,
    ActionType,
    Agent,
    AgentID,
    BeliefType,
    ExtraType,
    ObsType,
    RewardType,
)
from figgie_gym.envs.parallel_figgie_env import ParallelFiggieEnv
from figgie_gym.utilities import flatten

type AgentPoolID = int
type AgentPool = Sequence[tuple[Agent, float]]

type AgentPoolMaker = Callable[
    [np.random.Generator],
    AgentPool,
]


@dataclass
class EnvStepSnapshot:
    current_observations: dict[AgentID, ObsType]
    beliefs: dict[AgentID, BeliefType]
    actions: dict[AgentID, ActionType]
    rewards: dict[AgentID, RewardType]
    next_observations: dict[AgentID, ObsType]
    terminations: dict[AgentID, bool]
    truncations: dict[AgentID, bool]
    extras: dict[AgentID, ExtraType]

    def unroll(self) -> dict[AgentID, EnvAgentStepSnapshot]:
        return {
            agent_id: EnvAgentStepSnapshot(
                self.current_observations[agent_id],
                self.beliefs[agent_id],
                self.actions[agent_id],
                self.rewards[agent_id],
                self.next_observations[agent_id],
                self.terminations[agent_id],
                self.truncations[agent_id],
                extras=self.extras[agent_id],
            )
            for agent_id in self.current_observations
        }


@dataclass
class EnvAgentStepSnapshot:
    current_observations: ObsType
    belief: BeliefType
    actions: ActionType
    rewards: RewardType
    next_observations: ObsType
    terminations: bool
    truncations: bool
    extras: ExtraType


class MultipleGameRunner:
    """Run multiple games with action computed in a vectorized manner."""

    envs_agents: dict[ParallelFiggieEnv, list[tuple[AgentPoolID, Agent]]]
    rng: np.random.Generator
    agent_pool: AgentPool

    def __init__(
        self,
        num_experiments: int,
        num_steps: int,
        num_agents: int,
        game_runner_id: int,
    ) -> None:
        self.num_experiments = num_experiments
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.game_runner_id = game_runner_id

    def construct_env_and_assign_agents(
        self,
        rng: np.random.Generator,
        agent_pool_maker: AgentPoolMaker,
    ) -> Self:
        self.rng = rng
        self.agent_pool = agent_pool_maker(rng)
        agent_objs = [agent for agent, _ in self.agent_pool]
        agent_probs = [prob for _, prob in self.agent_pool]
        agent_choice = self.rng.choice(
            len(agent_objs),
            size=(self.num_experiments, self.num_agents),
            p=agent_probs,
        )
        all_agent_objs: list[list[tuple[AgentPoolID, Agent]]] = [
            [(i, agent_objs[i]) for i in agent_choice_for_game]
            for agent_choice_for_game in agent_choice
        ]
        env_id_start = self.game_runner_id * self.num_experiments
        env_id_end = (self.game_runner_id + 1) * self.num_experiments
        envs = [
            ParallelFiggieEnv(
                self.num_agents,
                self.num_steps,
                render_mode=None,
                env_id=env_id,
            )
            for env_id in range(env_id_start, env_id_end)
        ]
        self.envs_agents = dict(zip(envs, all_agent_objs, strict=True))
        return self

    def run(self) -> Generator[dict[ParallelFiggieEnv, EnvStepSnapshot]]:
        rngs = self.rng.spawn(self.num_experiments)
        env_obs_current: dict[ParallelFiggieEnv, dict[AgentID, ObsType]] = {
            env: env.reset(-1, {"rng": child_rng})[0]
            for env, child_rng in zip(self.envs_agents, rngs, strict=True)
        }
        for _ in range(self.num_steps + 1):
            env_actions = self.act_batch(
                env_obs_current,
            )
            if len(env_actions) != len(env_obs_current):
                msg = f"{len(env_actions)=} != {len(env_obs_current)=}"
                raise ValueError(msg)

            env_snapshots = dict[ParallelFiggieEnv, EnvStepSnapshot]()
            for env, beliefs_actions in env_actions.items():
                beliefs = {
                    agent_id: ba[0] for agent_id, ba in beliefs_actions.items()
                }
                actions = {
                    agent_id: ba[1] for agent_id, ba in beliefs_actions.items()
                }
                extras = {
                    agent_id: ba[2] for agent_id, ba in beliefs_actions.items()
                }
                observations_curr = env_obs_current[env]
                observations, rewards, terminations, truncations, _ = env.step(
                    actions,
                )
                env_snapshots[env] = EnvStepSnapshot(
                    observations_curr,
                    beliefs,
                    actions,
                    rewards,
                    observations,
                    terminations,
                    truncations,
                    extras,
                )

            yield env_snapshots
            env_obs_current = {
                env: snap.next_observations
                for env, snap in env_snapshots.items()
            }
            are_all_terminated = all(
                all(v.terminations.values()) for v in env_snapshots.values()
            )
            are_all_truncated = all(
                all(v.truncations.values()) for v in env_snapshots.values()
            )
            if are_all_terminated or are_all_truncated:
                break

    def act_batch(
        self,
        env_obs_current: dict[ParallelFiggieEnv, dict[AgentID, ObsType]],
    ) -> dict[
        ParallelFiggieEnv,
        dict[AgentID, tuple[BeliefType, ActionType, ExtraType]],
    ]:
        batched_obs_by_agent: dict[
            Agent,
            tuple[list[tuple[ParallelFiggieEnv, AgentID]], list[ObsType]],
        ] = {ag: ([], []) for ag, _ in self.agent_pool}

        for env, agent_objs in self.envs_agents.items():
            for agent_id in env.agents:
                _, agent_obj = agent_objs[agent_id]
                ids_list, obs_list = batched_obs_by_agent[agent_obj]
                ids_list.append((env, agent_id))
                obs_list.append(env_obs_current[env][agent_id])

        env_actions: dict[
            ParallelFiggieEnv,
            dict[AgentID, tuple[BeliefType, ActionType, ExtraType]],
        ] = {env: {} for env in self.envs_agents}

        for agent_obj, (env_ids, observations) in batched_obs_by_agent.items():
            if not observations:
                continue

            # vectorized determination of actions
            actions = agent_obj.act_batch(self.rng, observations)
            for (env, agent_id), action in zip(env_ids, actions, strict=True):
                env_actions[env][agent_id] = action

        return env_actions

    def flattened_env_agent_snapshot_iter(self) -> Generator[dict[str, Any]]:
        for step, env_snapshots in enumerate(self.run()):
            for env, snapshots in env_snapshots.items():
                agent_type_count = Counter[str](
                    [
                        ag.params()["agent_type"]
                        for _, ag in self.envs_agents[env]
                    ],
                )
                for agent_id, snap in snapshots.unroll().items():
                    data = {
                        "game_runner_id": self.game_runner_id,
                        "env_id": env.env_id,
                        "game_runner_step": step,
                        "agent_info": {
                            "agent_id": agent_id,
                            "agent_pool_id": self.envs_agents[env][agent_id][0],
                            "agent_params": self.envs_agents[env][agent_id][
                                1
                            ].params(),
                        },
                        "game_info": {
                            "goal_suit": str(env.goal_suit),
                            "goal_suit_code": SUIT_TO_CODE.get(env.goal_suit),
                            "agent_type_count": agent_type_count,
                        },
                        **asdict(snap),
                    }
                    yield flatten(data)

    def agent_pool_summary(self) -> list[dict[str, int | float]]:
        return [
            {
                "game_runner_id": self.game_runner_id,
                "agent_pool_index": agent_pool_index,
                "sampling_freq": p,
            }
            | ag.params()
            for agent_pool_index, (ag, p) in enumerate(self.agent_pool)
        ]
