"""Tests for noise agent module."""

from unittest.mock import Mock

import numpy as np

from figgie_gym.agent.common import AGENT_NAME_CODES
from figgie_gym.agent.noise import NoiseAgent
from figgie_gym.envs.common import SUITS, ObsType


def test_agent_instantiation() -> None:
    """Test that NoiseAgent can be instantiated."""
    agent = NoiseAgent()
    assert agent is not None


def test_agent_type_code() -> None:
    """Test that agent has correct type code."""
    agent = NoiseAgent()
    assert agent.agent_type_code == AGENT_NAME_CODES["NoiseAgent"]
    assert agent.agent_type_code == 0


def test_act_returns_tuple() -> None:
    """Test that act method returns a tuple of belief and action."""
    agent = NoiseAgent()
    rng = np.random.default_rng(42)

    # Create a mock observation
    observation = Mock(spec=ObsType)
    observation.per_suit = {s: Mock() for s in SUITS}

    belief, action, _ = agent.act(rng, observation)

    assert isinstance(belief, dict)
    assert isinstance(action, dict)


def test_act_belief_distribution() -> None:
    """Test that belief is uniform distribution over suits."""
    agent = NoiseAgent()
    rng = np.random.default_rng(42)

    observation = Mock(spec=ObsType)
    observation.per_suit = {s: Mock() for s in SUITS}

    belief, _, _ = agent.act(rng, observation)

    # Check that all suits have equal probability
    expected_prob = 1 / len(SUITS)
    for suit in SUITS:
        assert suit in belief
        assert belief[suit] == expected_prob


def test_act_batch_returns_list() -> None:
    """Test that act_batch returns a list."""
    agent = NoiseAgent()
    rng = np.random.default_rng(42)

    observations: list[ObsType] = []
    for _ in range(3):
        obs = Mock(spec=ObsType)
        obs.per_suit = {s: Mock() for s in SUITS}
        observations.append(obs)

    results = agent.act_batch(rng, observations)

    assert isinstance(results, list)
    assert len(results) == 3


def test_act_batch_each_element_is_tuple() -> None:
    """Test that each element in act_batch result is a tuple."""
    agent = NoiseAgent()
    rng = np.random.default_rng(42)

    observations: list[ObsType] = []
    for _ in range(2):
        obs = Mock(spec=ObsType)
        obs.per_suit = {s: Mock() for s in SUITS}
        observations.append(obs)

    results = agent.act_batch(rng, observations)

    for result in results:
        assert isinstance(result, tuple)
        assert len(result) == 3
        belief, action, _ = result
        assert isinstance(belief, dict)
        assert isinstance(action, dict)


def test_act_batch_multiple_observations() -> None:
    """Test act_batch with multiple observations."""
    agent = NoiseAgent()
    rng = np.random.default_rng(42)

    num_obs = 10
    observations: list[ObsType] = []
    for _ in range(num_obs):
        obs = Mock(spec=ObsType)
        obs.per_suit = {s: Mock() for s in SUITS}
        observations.append(obs)

    results = agent.act_batch(rng, observations)

    assert len(results) == num_obs


def test_act_randomness_with_seed() -> None:
    """Test that same seed produces same random actions."""
    agent = NoiseAgent()

    observation = Mock(spec=ObsType)
    observation.per_suit = {s: Mock() for s in SUITS}

    rng1 = np.random.default_rng(123)
    _, action1, _ = agent.act(rng1, observation)

    rng2 = np.random.default_rng(123)
    _, action2, _ = agent.act(rng2, observation)

    # Same seed should produce same results
    for suit in SUITS:
        assert action1[suit].quote_bid == action2[suit].quote_bid
        assert action1[suit].quote_ask == action2[suit].quote_ask
