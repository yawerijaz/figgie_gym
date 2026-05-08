from typing import Any

import numpy as np

from figgie_gym.envs.common import (
    SUITS,
    ActionOnSuit,
    ActionType,
    Agent,
    AgentID,
    BeliefType,
    ExtraType,
    InfoType,
    ObsOnSuit,
    ObsType,
    RewardType,
    Suit,
)
from figgie_gym.envs.parallel_figgie_env import ParallelFiggieEnv
from figgie_gym.envs.vectorized_game_runner import VectorizedGameRunner
from figgie_gym.envs.vectorized_parallel_figgie_env import (
    VectorizedParallelFiggieEnv,
)


# In this test, cash = snipe_ask = seat = agent id,
# while snipe_bid = quote_bid = quote_ask = agent pool id.
class FakeEnv(ParallelFiggieEnv):
    def __init__(self, num_agents: int, env_id: int) -> None:
        self.possible_agents = list(range(num_agents))
        self.env_id = env_id
        self.last_actions = None
        self.last_reset_options = None

    def reset(
        self,
        seed: int | None = None,  # noqa: ARG002
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, InfoType]]:
        self.last_reset_options = options or {}
        obs = {
            seat: ObsType(
                time=0,
                remaining_time=0,
                remaining_time_fraction=0.0,
                cash=seat,
                per_suit=dict[Suit, ObsOnSuit](),
            )
            for seat in self.possible_agents
        }
        info = {seat: dict[str, str]() for seat in self.possible_agents}
        return obs, info

    def step(
        self,
        actions: dict[AgentID, ActionType],
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, RewardType],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, InfoType],
    ]:
        self.last_actions = dict(actions)
        next_obs = {
            seat: ObsType(
                time=0,
                remaining_time=0,
                remaining_time_fraction=0.0,
                cash=seat,
                per_suit=dict[Suit, ObsOnSuit](),
            )
            for seat in self.possible_agents
        }
        rewards = {seat: float(seat) for seat in self.possible_agents}
        terminations = dict.fromkeys(self.possible_agents, False)
        truncations = dict.fromkeys(self.possible_agents, False)
        infos = {seat: dict[str, str]() for seat in self.possible_agents}
        return next_obs, rewards, terminations, truncations, infos


class MockAgent(Agent):
    def __init__(self, pool_idx: int) -> None:
        self.pool_idx = pool_idx

    def act_batch(
        self,
        random_number_generator: np.random.Generator,  # noqa: ARG002
        observations: list[ObsType],
    ) -> list[tuple[BeliefType, ActionType, ExtraType]]:
        # Return a consistent action that identifies this agent
        return [
            (
                {s: float(self.pool_idx) for s in SUITS},
                {
                    s: ActionOnSuit(
                        self.pool_idx,
                        self.pool_idx,
                        self.pool_idx,
                        obs.cash,
                    )
                    for s in SUITS
                },
                None,
            )
            for obs in observations
        ]

    def act(
        self,
        random_number_generator: np.random.Generator,  # noqa: ARG002
        observation: ObsType,  # noqa: ARG002
    ) -> tuple[BeliefType, ActionType, None]:
        return (
            {s: float(self.pool_idx) for s in SUITS},
            {
                s: ActionOnSuit(
                    self.pool_idx,
                    self.pool_idx,
                    self.pool_idx,
                    self.pool_idx,
                )
                for s in SUITS
            },
            None,
        )


def test_routing_and_seating_randomization() -> None:
    num_envs = 2
    num_agents = 3
    envs = [FakeEnv(num_agents, i) for i in range(num_envs)]
    vec_env = VectorizedParallelFiggieEnv(envs)
    agents = [MockAgent(i) for i in range(num_agents)]
    rng = np.random.default_rng(42)

    runner = VectorizedGameRunner(vec_env, agents, rng, num_steps=1)

    snapshots = list(runner.run(seed=123))
    assert len(snapshots) >= 1

    first_step = snapshots[0]
    for env_idx, agent_snaps in enumerate(first_step):
        # actions keyed by agent_pool_index should match the agent's action

        # each agent produced an action depending on its pool index
        for agent_pool_id, agent_snap in agent_snaps.items():
            expected_action = {
                s: ActionOnSuit(
                    agent_pool_id,
                    agent_pool_id,
                    agent_pool_id,
                    runner.seating[env_idx][agent_pool_id],
                )
                for s in SUITS
            }
            assert agent_snap.action == expected_action

        # verify env received a permutation of those actions at seats
        last_actions = envs[env_idx].last_actions or {}
        seen = set[int]()

        for seat_action in last_actions.values():
            matches = [
                i
                for i, agent_snap in agent_snaps.items()
                if agent_snap.action == seat_action
            ]
            assert len(matches) == 1
            seen.add(matches[0])
        assert seen == set(range(num_agents))


def test_reset_injects_rng_when_seed_provided() -> None:
    num_envs = 3
    num_agents = 2
    envs = [FakeEnv(num_agents, i) for i in range(num_envs)]
    vec_env = VectorizedParallelFiggieEnv(envs)
    agents = [MockAgent(i) for i in range(num_agents)]
    rng = np.random.default_rng(0)

    runner = VectorizedGameRunner(vec_env, agents, rng, num_steps=1)
    # trigger reset with seed; underlying FakeEnv should receive options with 'rng'
    _ = list(runner.run(seed=999))

    for env in envs:
        assert env.last_reset_options is not None
        assert "rng" in env.last_reset_options
