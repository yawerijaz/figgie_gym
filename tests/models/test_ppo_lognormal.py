"""Tests for lognormal sampling implementation in PPO."""

import pytest
import torch
from torch import Tensor
from torch.distributions import LogNormal

from figgie_gym.models.ppo import (
    ActorHead,
    PPONetworks,
)


class TestCoordinateInversions:
    """Test coordinate transformation: x ↔ action."""

    def test_x_to_action_reconstruction(self) -> None:
        """Test that x -> action -> x round-trip has minimal error."""
        fair = torch.tensor([[[5.0]]])
        x_sample = torch.tensor([[[[1.0, 2.0, 0.5, 1.5]]]])  # batch=1, suit=1

        action = ActorHead.x_to_action(x_sample, fair)
        x_reconstructed = ActorHead.action_to_x(action, fair)

        error = torch.abs(x_sample - x_reconstructed).max().item()
        assert error < 1e-6, f"Reconstruction error too large: {error}"

    def test_action_round_trip(self) -> None:
        """Test that action -> x -> action round-trip is consistent."""
        fair = torch.tensor([[[5.0]]])
        action = torch.tensor([[[[4.0, 7.0, 4.5, 6.5]]]])

        x = ActorHead.action_to_x(action, fair)
        action_recovered = ActorHead.x_to_action(x, fair)

        error = torch.abs(action - action_recovered).max().item()
        assert error < 1e-6, f"Action error too large: {error}"

    def test_anti_self_match_preservation(self) -> None:
        """Verify that x_to_action maintains anti-self-match constraints."""
        fair = torch.tensor([[[5.0]]])
        x_samples = torch.tensor(
            [
                [[[1.0, 2.0, 0.5, 1.5]]],
                [[[0.5, 1.5, 0.3, 2.0]]],
                [[[2.0, 3.0, 1.0, 2.5]]],
            ],
        )

        action = ActorHead.x_to_action(x_samples, fair)

        y0 = action[..., 0]  # quote_bid
        y1 = action[..., 1]  # quote_ask
        y2 = action[..., 2]  # snipe_bid
        y3 = action[..., 3]  # snipe_ask

        # Since x > 0, bid <= fair, ask >= fair
        assert torch.all(y0 <= fair), "quote_bid should be <= fair"
        assert torch.all(y1 >= fair), "quote_ask should be >= fair"
        assert torch.all(y1 >= y0), "quote_ask should be >= quote_bid"
        assert torch.all(y1 >= y2), "quote_ask should be >= snipe_bid"
        assert torch.all(y3 >= y0), "snipe_ask should be >= quote_bid"


class TestLognormalLogprob:
    """Test log-probability computation without scaling Jacobians."""

    @pytest.fixture(scope="class")
    def lognormal_dist(self) -> LogNormal:
        x_mean = torch.randn(2, 4, 4) * 2 + 5  # batch=2, suit=4
        x_mean = torch.clamp(x_mean, min=0.1)
        action_logstd = torch.full_like(x_mean, -1.0)
        return LogNormal(
            torch.log(torch.clamp(x_mean, min=1e-6)),
            torch.exp(action_logstd),
        )

    @pytest.fixture(scope="class")
    def fair(self) -> torch.Tensor:
        return torch.full((2, 4), 5.0)

    def test_logprob_shape_and_finiteness(
        self,
        lognormal_dist: LogNormal,
        fair: torch.Tensor,
    ) -> None:
        """Test that log-prob has correct shape and is finite."""
        action = torch.randn(2, 4, 4) * 2 + 5

        logprob = PPONetworks.compute_lognormal_logprob_with_jacobian(
            lognormal_dist,
            action,
            fair,
        )

        assert logprob.shape == torch.Size([2]), "log-prob shape incorrect"
        assert torch.all(torch.isfinite(logprob)), "log-prob contains NaN/Inf"

    def test_logprob_is_negative(
        self,
        lognormal_dist: LogNormal,
        fair: torch.Tensor,
    ) -> None:
        """Test that log-prob values are negative (probability < 1)."""
        action = torch.randn(2, 4, 4) * 2 + 5

        logprob = PPONetworks.compute_lognormal_logprob_with_jacobian(
            lognormal_dist,
            action,
            fair,
        )

        assert torch.all(logprob < 0), "log-prob should be negative"


class TestLognormalSampling:
    """Test that lognormal sampling preserves constraints natively."""

    def test_sampled_actions_respect_fair_bounds(self) -> None:
        """Test that sampled actions remain constrained around fair proper."""
        fair = torch.tensor([[[5.0]]])
        x_mean = torch.tensor([[[[1.0, 2.0, 0.5, 1.5]]]])

        action_logstd = torch.full_like(x_mean, -1.0)
        action_std = torch.exp(action_logstd)

        # Sample multiple times
        num_samples = 50
        samples: list[Tensor] = []
        for _ in range(num_samples):
            lognormal_dist = LogNormal(torch.log(x_mean), action_std)
            x_sample = lognormal_dist.rsample()
            action_sample = ActorHead.x_to_action(x_sample, fair)
            samples.append(action_sample)

        samples_tensor = torch.stack(samples)  # (num_samples, batch, suit, 4)

        y0 = samples_tensor[..., 0]
        y1 = samples_tensor[..., 1]

        # Verify bids are <= fair, asks >= fair
        assert torch.all(y0 <= fair.expand_as(y0)), "Bids should be <= fair"
        assert torch.all(y1 >= fair.expand_as(y1)), "Asks should be >= fair"
        assert torch.all(torch.isfinite(samples_tensor)), (
            "Prices should be finite"
        )
