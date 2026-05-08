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
from figgie_gym.models.ppo import (
    ActorHead,
    EquivariantActorNet,
    EquivariantPricerNet,
)
from figgie_gym.models.ppo_helpers import FINAL_CARD_VALUE
from figgie_gym.utilities import get_device


class CentralizedCriticNet(nn.Module):
    """Value function over joint observations for MAPPO (CTDE).

    Encodes each agent's observation with a dedicated :class:`EquivariantBody`,
    pools suit axes, concatenates agent embeddings, and predicts one baseline per
    learned agent slot.
    """

    def __init__(
        self,
        equivariant_args: EquivariantObservationArchitectureArgs,
        num_learned_agents: int,
        fusion_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.num_learned_agents = num_learned_agents
        self.body = EquivariantBody(equivariant_args)
        fused_in = num_learned_agents * equivariant_args.suit_embed_dim
        self.head = SequentialLinear(
            fused_in,
            fusion_hidden_dims,
            num_learned_agents,
        )

    def forward(self, joint_x: TensorDict) -> Float[Tensor, "batch agents"]:  # noqa: F722
        """Predict critic values from stacked local observations.

        Args:
            joint_x: Preprocessed observation tensordict with ``batch_size ==
                (batch, num_learned_agents)``, same nested layout as single-agent
                ``x``.

        Returns:
            Tensor of shape ``(batch, num_learned_agents)``.

        """
        batch_dims = joint_x.batch_size
        if len(batch_dims) != 2:  # noqa: PLR2004
            msg = f"joint_x must have batch rank 2, got {tuple(batch_dims)}"
            raise ValueError(msg)
        batch_n, k = int(batch_dims[0]), int(batch_dims[1])
        if k != self.num_learned_agents:
            msg = f"joint_x agent dimension {k} != {self.num_learned_agents}"
            raise ValueError(msg)

        flat_x = joint_x.reshape(batch_n * k)  # pyright: ignore[reportUnknownMemberType]
        body_out = self.body(flat_x)
        pooled = body_out.sum(dim=-2)
        fused = pooled.reshape(batch_n, k * pooled.shape[-1])
        return self.head(fused)


class MAPPONetworks(nn.Module):
    """Decentralized actors + belief + centralized critic for MAPPO."""

    def __init__(
        self,
        centralized_critic: CentralizedCriticNet,
        actor: EquivariantActorNet,
        belief: EquivariantPricerNet,
        belief_embedder: nn.Module,
        bias_bound: float = 3.0,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.centralized_critic = centralized_critic
        self.actor = actor
        self.belief = belief
        self.belief_embedding = belief_embedder
        self.actor_logstd = nn.Parameter(
            torch.full((1, 1, 4), 0.0, device=get_device()),
        )
        self.bias_bound = bias_bound

    def get_joint_values(self, joint_x: TensorDict) -> Tensor:
        return self.centralized_critic(joint_x)

    def get_action_and_value(
        self,
        x: TensorDict,
        action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Local actor forward; value slot is zeros (use critic at train time)."""
        actor_body = self.actor.body(x)
        belief_logits = self.belief(x)
        belief_probs = nn.functional.softmax(belief_logits.detach(), dim=-1)
        belief_embed = self.belief_embedding(
            belief_probs.unsqueeze(-1).float(),
        )
        y = actor_body + belief_embed
        x_mean, bias_logit = self.actor.actor_head(y)
        action_logstd = self.actor_logstd.expand_as(x_mean)
        action_std = torch.exp(action_logstd)

        x_mean_clamped = torch.clamp(x_mean, min=1e-6)
        lognormal_dist = LogNormal(torch.log(x_mean_clamped), action_std)

        fair = belief_probs * FINAL_CARD_VALUE
        bias = self.bias_bound * torch.sigmoid(bias_logit)
        center = fair + bias.squeeze()

        if action is None:
            x_sample = lognormal_dist.rsample()
            action = ActorHead.x_to_action(x_sample, center)
            logprob_curr = self.compute_lognormal_logprob_with_jacobian(
                lognormal_dist,
                action,
                center,
            )
        else:
            logprob_curr = self.compute_lognormal_logprob_with_jacobian(
                lognormal_dist,
                action,
                center,
            )

        value = torch.zeros(
            x.batch_size[0],
            device=x.device,
            dtype=action.dtype,
        )
        entropy = lognormal_dist.entropy().sum(dim=(-1, -2))

        extras = {
            "fair": fair.squeeze().cpu().float(),
            "center": center.squeeze().cpu().float(),
            "bias": bias.squeeze().cpu().float(),
        }

        return action, logprob_curr, entropy, value, belief_logits, extras

    @staticmethod
    def compute_lognormal_logprob_with_jacobian(
        lognormal_dist: LogNormal,
        action: Float[Tensor, "batch suit 4"],  # noqa: F722
        fair: Float[Tensor, "..."],
    ) -> Float[Tensor, "batch"]:  # noqa: F821
        x = ActorHead.action_to_x(action, fair)
        x_clamped = torch.clamp(x, min=1e-6)
        logprob_lognormal = lognormal_dist.log_prob(x_clamped)
        return logprob_lognormal.sum((-2, -1))

    def forward(
        self,
        x: TensorDict,
        action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        return self.get_action_and_value(x, action)
