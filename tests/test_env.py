from figgie_gym.envs.common import ActionOnSuit
from figgie_gym.envs.parallel_figgie_env import ParallelFiggieEnv
from figgie_gym.market.common import Symbol


def test_env_instantiates_and_steps() -> None:
    env = ParallelFiggieEnv(num_agents=2)
    obs = env.reset()
    assert isinstance(obs, tuple)
    assert len(obs) == 2

    # send noop actions for both agents
    actions = {
        agent: {
            Symbol("Spade"): ActionOnSuit(
                10,
                11,
                12,
                13,
            ),
        }
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)
    assert set(observations.keys()) == set(env.agents)
    assert set(rewards.keys()) == set(env.agents)
    assert set(terminations.keys()) == set(env.agents)
    assert set(truncations.keys()) == set(env.agents)
    assert set(infos.keys()) == set(env.agents)
    assert all(r == 0.0 for r in rewards.values())
