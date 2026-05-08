from typing import cast

import lightning as pl
import torch
from jaxtyping import Float
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.distributions import LogNormal

from figgie_gym.models.building_blocks import SequentialLinear
from figgie_gym.models.equivariant import (
    EquivariantBody,
    EquivariantObservationArchitectureArgs,
)
from figgie_gym.models.ppo_helpers import FINAL_CARD_VALUE
from figgie_gym.utilities import get_device


class ActorHead(nn.Module):
    """Produces the quote and snipe prices for each suit.

    We prohibit self-match by requiring that
    - quote bid < quote ask
    - quote bid < snipe ask
    - snipe bid < quote ask

    Let x = [x0, x1, x2, x3] be the output per suit from the previous layers,
    where -inf < xi < inf.

    Apply x <- Softplus(x), x >= 0.

    We produce positive spreads x which are anchored to a fair expected value:
    - quote bid = center - x0
    - quote ask = center + x1
    - snipe bid = center - x2
    - snipe ask = center + x3

    Because xi >= 0, we trivially satisfy the anti-self-match constraints:
    - bid <= fair <= ask
    - All bids are inherently lower or equal to all asks natively.
    """

    def __init__(
        self,
        suit_embed_dim: int,
        actor_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.model = SequentialLinear(
            suit_embed_dim,
            actor_hidden_dims,
            5,
        )  # 4 prices + 1 bias
        last_layer = cast("nn.Linear", self.model.model[-1])
        # First 4 (margins) initialize to 1.0, 5th (bias) initializes to 0.0
        nn.init.constant_(last_layer.bias[:4], 1.0)
        nn.init.constant_(last_layer.bias[4], 0.0)

    def forward(
        self,
        suit_full_context: Float[Tensor, "batch suit suit_embed_dim"],  # noqa: F722
    ) -> tuple[Float[Tensor, "batch suit 4"], Float[Tensor, "batch suit 1"]]:  # noqa: F722
        # maps to target 4 margins (positive via softplus) + 1 real bias
        out = self.model(suit_full_context)
        spreads = nn.functional.softplus(out[..., :4])
        bias = out[..., 4:]
        return spreads, bias

    @staticmethod
    def action_to_x(
        action: Float[Tensor, "batch suit 4"],  # noqa: F722
        center: Float[Tensor, "..."],
    ) -> Float[Tensor, "batch suit 4"]:  # noqa: F722
        x0 = center - action[..., 0]
        x1 = action[..., 1] - center
        x2 = center - action[..., 2]
        x3 = action[..., 3] - center

        return torch.stack([x0, x1, x2, x3], dim=-1)

    @staticmethod
    def x_to_action(
        x: Float[Tensor, "batch suit 4"],  # noqa: F722
        center: Float[Tensor, "..."],
    ) -> Float[Tensor, "batch suit 4"]:  # noqa: F722
        a0 = center - x[..., 0]
        a1 = center + x[..., 1]
        a2 = center - x[..., 2]
        a3 = center + x[..., 3]

        return torch.stack([a0, a1, a2, a3], dim=-1)


class ValueHead(nn.Module):
    """Estimates the value at a given an observation."""

    def __init__(
        self,
        suit_embed_dim: int,
        value_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.model = SequentialLinear(
            suit_embed_dim,
            value_hidden_dims,
            1,
        )  # 1 value for each suit

    def forward(
        self,
        suit_full_context: Float[Tensor, "batch suit suit_embed_dim"],  # noqa: F722
    ) -> Float[Tensor, "batch"]:  # noqa: F821
        x: Tensor = self.model(suit_full_context)
        return x.sum((-2, -1))


class PricerHead(nn.Module):
    """Returns probability of each suit being the goal suit."""

    def __init__(
        self,
        suit_embed_dim: int,
        pricer_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.model = SequentialLinear(
            suit_embed_dim,
            pricer_hidden_dims,
            1,
        )  # 4 suits

    def forward(
        self,
        suit_full_context: Float[Tensor, "batch suit suit_embed_dim"],  # noqa: F722
    ) -> Float[Tensor, "batch suit"]:  # noqa: F722
        return self.model(suit_full_context).squeeze(-1)


class EquivariantActorNet(pl.LightningModule):
    """An actor with shared blocks."""

    def __init__(
        self,
        equivariant_args: EquivariantObservationArchitectureArgs,
        actor_hidden_dims: tuple[int, ...],
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.body = EquivariantBody(equivariant_args)

        self.actor_head = ActorHead(
            equivariant_args.suit_embed_dim,
            actor_hidden_dims,
        )
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        x: TensorDict,
    ) -> tuple[Float[Tensor, "batch suit 4"], Float[Tensor, "batch suit 1"]]:  # noqa: F722
        y = self.body(x)
        return self.actor_head(y)


class EquivariantValueNet(pl.LightningModule):
    """A value function with shared blocks."""

    def __init__(
        self,
        equivariant_args: EquivariantObservationArchitectureArgs,
        value_hidden_dims: tuple[int, ...],
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.body = EquivariantBody(equivariant_args)

        self.value_head = ValueHead(
            equivariant_args.suit_embed_dim,
            value_hidden_dims,
        )
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: TensorDict) -> Float[Tensor, "batch"]:  # noqa: F821
        y = self.body(x)
        return self.value_head(y)


class EquivariantPricerNet(pl.LightningModule):
    """A pricer network with shared blocks."""

    def __init__(
        self,
        equivariant_args: EquivariantObservationArchitectureArgs,
        pricer_hidden_dims: tuple[int, ...],
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.body = EquivariantBody(equivariant_args)

        self.pricer_head = PricerHead(
            equivariant_args.suit_embed_dim,
            pricer_hidden_dims,
        )
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: TensorDict) -> Float[Tensor, "batch suit"]:  # noqa: F722
        y = self.body(x)
        return self.pricer_head(y)


class PPONetworks(nn.Module):
    def __init__(
        self,
        critic: EquivariantValueNet,
        actor: EquivariantActorNet,
        belief: EquivariantPricerNet,
        belief_embedder: nn.Module,
        bias_bound: float = 3.0,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.actor = actor
        self.critic = critic
        self.belief = belief
        # belief_embedder maps (batch, suit, 1) -> (batch, suit, suit_embed_dim)
        self.belief_embedding = belief_embedder
        # Initialize logstd at 0.0 for LogNormal(mean, exp(0.0)=1) exploration in x-space.
        self.actor_logstd = nn.Parameter(
            torch.full((1, 1, 4), 0.0, device=get_device()),
        )  # log(scale) for lognormal distribution in x-space
        self.bias_bound = (
            bias_bound  # How much we allow the quote mid to deviate from fair
        )

    @staticmethod
    def compute_lognormal_logprob_with_jacobian(
        lognormal_dist: LogNormal,
        action: Float[Tensor, "batch suit 4"],  # noqa: F722
        fair: Float[Tensor, "..."],
    ) -> Float[Tensor, "batch"]:  # noqa: F821
        """Compute log-probability of actions sampled from lognormal distribution.

        The sampling process is:
        1. Action margins x are sampled from LogNormal
        2. Prices are mapped as offsets from fair value

        The log-prob must account for:
        - LogNormal distribution
        - Jacobian of transforming between action space and x space.
          The determinant of this transformation is 1, so log|d/dx| = 0.

        Args:
            lognormal_dist: Distribution from which x was sampled
            action: Sampled actions
            fair: Base fair values per suit

        Returns:
            log_prob: shape (batch,), sum of log-probs across suits and price dimensions

        """
        x = ActorHead.action_to_x(action, fair)
        x_clamped = torch.clamp(x, min=1e-6)
        logprob_lognormal = lognormal_dist.log_prob(x_clamped)

        # Jacobian of transforming between action space and x space.
        # The determinant of this transformation is 1, so log|d/dx| = 0.

        # Sum across suits and price dimensions: (batch, suit, 4) → (batch,)
        return logprob_lognormal.sum((-2, -1))

    def get_value(self, x: TensorDict) -> Tensor:
        return self.critic(x)

    def get_action_and_value(
        self,
        x: TensorDict,
        action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Compute action, log probability, entropy, value estimate and belief logits.

        Uses lognormal sampling in x-space to ensure all samples
        remain positive and preserve the anti-self-match constraints from ActorHead.

        Belief is predicted by a separate equivariant network; its probabilities
        are projected into the actor's suit embedding space and combined with
        the actor body output before producing action means.
        """
        # compute actor body first (expected shape: batch, suit, suit_embed_dim)
        actor_body = self.actor.body(x)

        # belief logits from separate belief network (batch, suit).
        # Returned to caller for gradient descent on cross entropy.
        # stay attached
        belief_logits = self.belief(x)

        # convert to probabilities and project into suit embedding.
        # detach to prevent Actor fuddling with the pricer
        belief_probs = nn.functional.softmax(belief_logits.detach(), dim=-1)

        # apply provided embedder; expect it to produce (batch, suit, suit_embed_dim)
        belief_embed = self.belief_embedding(
            belief_probs.unsqueeze(
                -1,
            ).float(),
        )

        y = actor_body + belief_embed
        x_mean, bias_logit = self.actor.actor_head(y)
        action_logstd = self.actor_logstd.expand_as(x_mean)
        action_std = torch.exp(action_logstd)

        # Convert action_mean to x-space for lognormal sampling
        x_mean_clamped = torch.clamp(x_mean, min=1e-6)
        lognormal_dist = LogNormal(torch.log(x_mean_clamped), action_std)

        fair = belief_probs * FINAL_CARD_VALUE
        bias = self.bias_bound * torch.sigmoid(
            bias_logit,
        )  # allow agent to buy aggressively, was (tanh + 1) / 2
        center = fair + bias.squeeze()

        # Sample action: either from stored action or sample from lognormal
        if action is None:
            x_sample = lognormal_dist.rsample()  # uses reparameterization trick
            action = ActorHead.x_to_action(x_sample, center)
            logprob_curr = PPONetworks.compute_lognormal_logprob_with_jacobian(
                lognormal_dist,
                action,
                center,
            )
        else:
            # We are evaluating given actions
            logprob_curr = PPONetworks.compute_lognormal_logprob_with_jacobian(
                lognormal_dist,
                action,
                center,
            )

        value = self.critic(x)
        entropy = lognormal_dist.entropy().sum(dim=(-1, -2))

        extras = {
            "fair": fair.squeeze().cpu().float(),
            "center": center.squeeze().cpu().float(),
            "bias": bias.squeeze().cpu().float(),
        }

        return action, logprob_curr, entropy, value, belief_logits, extras

    def forward(
        self,
        x: TensorDict,
        action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        return self.get_action_and_value(x, action)
