from typing import cast

import lightning as pl
import torch
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.optim import Adam

from figgie_gym.models.ppo import (
    PPONetworks,
)
from figgie_gym.models.ppo_gae import estimate_advantages_and_returns


class PPOLightningModule(pl.LightningModule):
    def __init__(  # noqa: PLR0913
        self,
        ppo_networks: PPONetworks,
        ppo_epsilon: float,
        critic_loss_coef: float,
        actor_entropy_loss_coef: float,
        belief_loss_coef: float = 0.1,
        learning_rate: float = 0.002,
    ) -> None:
        super().__init__()
        super().save_hyperparameters()
        self.ppo_epsilon = ppo_epsilon
        self.ppo_networks = ppo_networks
        self.critic_loss_coef = critic_loss_coef
        self.actor_entropy_loss_coef = actor_entropy_loss_coef
        self.belief_loss_coef = belief_loss_coef
        self.learning_rate = learning_rate

    def training_step(
        self,
        batch: TensorDict,  # keys: same as rollout buffer
        batch_idx: int,  # noqa: ARG002
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss and return both loss and breakdown dict for logging.

        Returns:
            (loss, metrics_dict) where metrics_dict contains policy_loss, value_loss, entropy_loss, loss

        """
        obs = cast("TensorDict", batch["observation"])
        actions = cast("Tensor", batch["action"])
        rewards = cast("Tensor", batch["reward"])
        logprob_buffer = cast("Tensor", batch["logprob"])
        termination = cast("Tensor", batch["termination"])
        truncation = cast("Tensor", batch["truncation"])

        _, logprob_curr, entropy, value, belief_logits, extras = (
            self.ppo_networks.get_action_and_value(obs, actions)
        )

        logratio = logprob_curr - logprob_buffer
        ratio = logratio.exp()

        advantage, return_estimates = estimate_advantages_and_returns(
            rewards,
            value,
            termination,
            truncation,
        )
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        clamped_ratio_adv = (
            torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
            * advantage
        )
        ratio_adv = ratio * advantage
        policy_loss = -torch.minimum(ratio_adv, clamped_ratio_adv).mean()

        # clamping value seems to help with stablilizing value loss too
        value_pred_clipped = torch.clamp(
            value,
            min=return_estimates.detach() - self.ppo_epsilon,
            max=return_estimates.detach() + self.ppo_epsilon,
        )
        value_loss = torch.max(
            (return_estimates.detach() - value) ** 2,
            (return_estimates.detach() - value_pred_clipped) ** 2,
        ).mean()
        entropy_loss = -entropy.mean()

        goal_labels = (
            cast("Tensor", batch["goal_suit_code"])
            .to(belief_logits.device)
            .long()
        )
        belief_loss = nn.CrossEntropyLoss()(belief_logits, goal_labels)

        loss = (
            policy_loss
            + self.critic_loss_coef * value_loss
            + self.actor_entropy_loss_coef * entropy_loss
            + self.belief_loss_coef * belief_loss
        )

        metrics = {
            "train/policy_loss": float(policy_loss.detach()),
            "train/value_loss": float(value_loss.detach()),
            "train/entropy_loss": float(entropy_loss.detach()),
            "train/belief_loss": float(belief_loss.detach()),
            "train/mean_fair": float(extras["fair"].mean().detach()),
            "train/mean_center": float(extras["center"].mean().detach()),
            "train/mean_bias": float(extras["bias"].mean().detach()),
            "train/loss": float(loss.detach()),
        }

        return loss, metrics

    def configure_optimizers(self) -> Adam:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
