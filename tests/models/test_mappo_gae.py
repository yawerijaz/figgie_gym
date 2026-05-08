import torch

from figgie_gym.models.ppo_gae import (
    estimate_advantages_and_returns,
    estimate_advantages_and_returns_grouped,
)


def test_grouped_gae_matches_independent_trajectories() -> None:
    """Interleaved rows from two trajectories match per-trajectory GAE."""
    # Trajectory A: env 0, agent 0, times 0..2
    r_a = torch.tensor([1.0, 0.0, 1.0])
    v_a = torch.tensor([0.5, 0.5, 0.5])
    term_a = torch.tensor([False, False, False])
    trunc_a = torch.tensor([False, False, False])
    adv_a, ret_a = estimate_advantages_and_returns(
        r_a.unsqueeze(-1),
        v_a.unsqueeze(-1),
        term_a.unsqueeze(-1),
        trunc_a.unsqueeze(-1),
    )

    # Trajectory B: env 1, agent 0, times 0..2 (different rewards)
    r_b = torch.tensor([0.0, 2.0, 0.0])
    v_b = torch.tensor([0.2, 0.2, 0.2])
    term_b = torch.tensor([False, False, True])
    trunc_b = torch.tensor([False, False, False])
    adv_b, ret_b = estimate_advantages_and_returns(
        r_b.unsqueeze(-1),
        v_b.unsqueeze(-1),
        term_b.unsqueeze(-1),
        trunc_b.unsqueeze(-1),
    )

    # Interleave: A0, B0, A1, B1, A2, B2
    rewards = torch.stack([r_a[0], r_b[0], r_a[1], r_b[1], r_a[2], r_b[2]])
    values = torch.stack([v_a[0], v_b[0], v_a[1], v_b[1], v_a[2], v_b[2]])
    termination = torch.stack(
        [term_a[0], term_b[0], term_a[1], term_b[1], term_a[2], term_b[2]],
    )
    truncation = torch.stack(
        [
            trunc_a[0],
            trunc_b[0],
            trunc_a[1],
            trunc_b[1],
            trunc_a[2],
            trunc_b[2],
        ],
    )
    env_idx = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    agent_slot = torch.zeros(6, dtype=torch.long)
    time = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    adv_g, ret_g = estimate_advantages_and_returns_grouped(
        rewards,
        values,
        termination,
        truncation,
        env_idx,
        agent_slot,
        time,
    )

    order_a = torch.tensor([0, 2, 4])
    order_b = torch.tensor([1, 3, 5])
    assert torch.allclose(adv_g[order_a], adv_a.squeeze(-1))
    assert torch.allclose(ret_g[order_a], ret_a.squeeze(-1))
    assert torch.allclose(adv_g[order_b], adv_b.squeeze(-1))
    assert torch.allclose(ret_g[order_b], ret_b.squeeze(-1))


def test_grouped_gae_two_agent_slots_same_env() -> None:
    """Different agent_slot indices partition the same env_idx."""
    r_0 = torch.ones(2)
    v_0 = torch.zeros(2)
    term = torch.zeros(2, dtype=torch.bool)
    trunc = torch.zeros(2, dtype=torch.bool)
    adv_0, _ret_0 = estimate_advantages_and_returns(
        r_0.unsqueeze(-1),
        v_0.unsqueeze(-1),
        term.unsqueeze(-1),
        trunc.unsqueeze(-1),
    )

    rewards = torch.ones(4)
    values = torch.zeros(4)
    termination = torch.zeros(4, dtype=torch.bool)
    truncation = torch.zeros(4, dtype=torch.bool)
    env_idx = torch.zeros(4, dtype=torch.long)
    agent_slot = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    time = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    adv_g, _ret_g = estimate_advantages_and_returns_grouped(
        rewards,
        values,
        termination,
        truncation,
        env_idx,
        agent_slot,
        time,
    )

    idx_slot0 = torch.tensor([0, 1])
    idx_slot1 = torch.tensor([2, 3])
    assert torch.allclose(adv_g[idx_slot0], adv_0.squeeze(-1))
    assert torch.allclose(adv_g[idx_slot1], adv_0.squeeze(-1))
