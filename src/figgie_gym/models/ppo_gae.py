import torch
from attr import dataclass
from torch import Tensor


@dataclass
class GAESmoothingParams:
    gamma: float = 0.99
    lambda_: float = 0.99


default_gae_params = GAESmoothingParams()


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def estimate_advantages_and_returns(  # noqa: PLR0913
    rewards: Tensor,
    values: Tensor,
    terminations: Tensor,
    truncations: Tensor,
    gae_params: GAESmoothingParams = default_gae_params,
    terminal_bootstrap_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation (GAE)."""
    continued = ~(terminations | truncations)

    device = values.device
    dtype = values.dtype
    bootstrap = torch.full(
        (1, *values.shape[1:]),
        float(terminal_bootstrap_value),
        device=device,
        dtype=dtype,
    )
    next_value = torch.cat((values[1:], bootstrap), dim=0)
    value_estimate = rewards + gae_params.gamma * next_value * continued
    delta = value_estimate - values

    last_advantage = 0
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        advantages[t] = last_advantage = (
            delta[t]
            + gae_params.gamma
            * gae_params.lambda_
            * continued[t]
            * last_advantage
        )
    returns = values + advantages
    return advantages, returns


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def estimate_advantages_and_returns_grouped(  # noqa: PLR0913
    rewards: Tensor,
    values: Tensor,
    terminations: Tensor,
    truncations: Tensor,
    env_idx: Tensor,
    agent_slot: Tensor,
    time: Tensor,
    gae_params: GAESmoothingParams = default_gae_params,
    terminal_bootstrap_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """GAE on a flat rollout by grouping `(env_idx, agent_slot)` trajectories.

    Within each group, transitions are ordered by ascending `time`. Groups are
    independent (used for multi-agent rollouts with a centralized critic).
    """
    if rewards.shape != values.shape:
        msg = "rewards and values must match shape"
        raise ValueError(msg)
    n = rewards.shape[0]
    device = rewards.device
    dtype = rewards.dtype
    advantages = torch.zeros(n, device=device, dtype=dtype)
    returns = torch.zeros(n, device=device, dtype=dtype)

    pairs = torch.stack([env_idx.long(), agent_slot.long()], dim=1)
    unique_pairs = pairs[time == 0]

    for row in unique_pairs:
        e_id, slot = row[0].item(), row[1].item()
        mask = (env_idx.long() == e_id) & (agent_slot.long() == slot)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        order = time[idx].long().argsort()
        sorted_idx = idx[order]

        seg_r = rewards[sorted_idx].unsqueeze(-1)
        seg_v = values[sorted_idx].unsqueeze(-1)
        seg_term = terminations[sorted_idx].unsqueeze(-1)
        seg_trunc = truncations[sorted_idx].unsqueeze(-1)

        seg_adv, seg_ret = estimate_advantages_and_returns(
            seg_r,
            seg_v,
            seg_term,
            seg_trunc,
            gae_params=gae_params,
            terminal_bootstrap_value=terminal_bootstrap_value,
        )
        advantages[sorted_idx] = seg_adv.squeeze(-1)
        returns[sorted_idx] = seg_ret.squeeze(-1)

    return advantages, returns
