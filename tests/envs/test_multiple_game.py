import numpy as np
import pytest

from figgie_gym.envs.common import (
    SUITS,
    ActionOnSuit,
    ActionType,
    Agent,
    AgentID,
    BeliefType,
    ExtraType,
    ObsType,
)
from figgie_gym.envs.multiple_game import AgentPoolMaker, MultipleGameRunner
from figgie_gym.envs.parallel_figgie_env import ParallelFiggieEnv


class MockAgent(Agent):
    def __init__(self, name: str = "MockAgent") -> None:
        self.name = name
        self.act_batch_calls = 0
        self.last_observations_count = 0

    def act(
        self,
        random_number_generator: np.random.Generator,  # noqa: ARG002
        observation: ObsType,  # noqa: ARG002
    ) -> tuple[BeliefType, ActionType, ExtraType]:
        return (
            dict.fromkeys(SUITS, 1 / len(SUITS)),
            {s: ActionOnSuit(10.0, 11.0, 12.0, 13.0) for s in SUITS},
            None,
        )

    def act_batch(
        self,
        random_number_generator: np.random.Generator,
        observations: list[ObsType],
    ) -> list[tuple[BeliefType, ActionType, ExtraType]]:
        self.act_batch_calls += 1
        self.last_observations_count = len(observations)
        return [self.act(random_number_generator, obs) for obs in observations]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def agent_pool_maker() -> AgentPoolMaker:
    def wrapped(_: np.random.Generator) -> list[tuple[MockAgent, float]]:
        return [(MockAgent("Agent1"), 0.5), (MockAgent("Agent2"), 0.5)]

    return wrapped


def test_runner_initialization(
    rng: np.random.Generator,
    agent_pool_maker: AgentPoolMaker,
) -> None:
    runner = MultipleGameRunner(
        num_experiments=2,
        num_steps=5,
        num_agents=2,
        game_runner_id=1,
    ).construct_env_and_assign_agents(rng, agent_pool_maker)
    assert runner.num_experiments == 2
    assert runner.num_steps == 5
    assert len(runner.agent_pool) == 2


def test_runner_construct_env(
    rng: np.random.Generator,
    agent_pool_maker: AgentPoolMaker,
) -> None:
    runner = MultipleGameRunner(
        num_experiments=3,
        num_steps=5,
        num_agents=4,
        game_runner_id=1,
    ).construct_env_and_assign_agents(rng, agent_pool_maker)
    assert len(runner.envs_agents) == 3
    for env, agents in runner.envs_agents.items():
        assert isinstance(env, ParallelFiggieEnv)
        assert len(agents) == 4
        for agent_pool_id, agent in agents:
            assert agent_pool_id in [0, 1]
            assert isinstance(agent, MockAgent)


def test_act_batch(
    rng: np.random.Generator,
    agent_pool_maker: AgentPoolMaker,
) -> None:
    runner = MultipleGameRunner(
        num_experiments=2,
        num_steps=5,
        num_agents=3,
        game_runner_id=1,
    ).construct_env_and_assign_agents(rng, agent_pool_maker)

    # Reset envs to populate env.agents
    mock_env_obs = dict[ParallelFiggieEnv, dict[AgentID, ObsType]]()
    for env in runner.envs_agents:
        obs, _ = env.reset(seed=42)
        mock_env_obs[env] = obs

    env_actions = runner.act_batch(mock_env_obs)

    assert len(env_actions) == 2
    for actions in env_actions.values():
        assert len(actions) == 3


def test_run_loop_completes(
    rng: np.random.Generator,
    agent_pool_maker: AgentPoolMaker,
) -> None:
    runner = MultipleGameRunner(
        num_experiments=2,
        num_steps=2,
        num_agents=2,
        game_runner_id=1,
    ).construct_env_and_assign_agents(rng, agent_pool_maker)

    steps = list(runner.run())
    assert len(steps) > 0
