from typing import cast

import lightning as pl
import torch
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.optim import Adam

from figgie_gym.models.mappo import MAPPONetworks
from figgie_gym.models.ppo_gae import (
    estimate_advantages_and_returns_grouped,
)


class MAPPOLightningModule(pl.LightningModule):
    """PPO training with centralized critic and grouped GAE."""

    def __init__(  # noqa: PLR0913
        self,
        mappo_networks: MAPPONetworks,
        num_learned_agents: int,
        ppo_epsilon: float,
        critic_loss_coef: float,
        actor_entropy_loss_coef: float,
        belief_loss_coef: float = 0.1,
        learning_rate: float = 0.002,
    ) -> None:
        super().__init__()
        super().save_hyperparameters(ignore=["mappo_networks"])
        self.mappo_networks = mappo_networks
        self.num_learned_agents = num_learned_agents
        self.ppo_epsilon = ppo_epsilon
        self.critic_loss_coef = critic_loss_coef
        self.actor_entropy_loss_coef = actor_entropy_loss_coef
        self.belief_loss_coef = belief_loss_coef
        self.learning_rate = learning_rate

    def training_step(
        self,
        batch: TensorDict,
        batch_idx: int,  # noqa: ARG002
    ) -> tuple[torch.Tensor, dict[str, float]]:
        obs = cast("TensorDict", batch["observation"])
        joint_obs = cast("TensorDict", batch["joint_observation"])
        actions = cast("Tensor", batch["action"])
        rewards = cast("Tensor", batch["reward"])
        logprob_buffer = cast("Tensor", batch["logprob"])
        termination = cast("Tensor", batch["termination"])
        truncation = cast("Tensor", batch["truncation"])
        agent_slot = cast("Tensor", batch["agent_slot"])
        env_idx = cast("Tensor", batch["env_idx"])
        time = cast("Tensor", batch["time"])

        _, logprob_curr, entropy, _, belief_logits, extras = (
            self.mappo_networks.get_action_and_value(obs, actions)
        )

        joint_values = self.mappo_networks.get_joint_values(joint_obs)
        slot_idx = agent_slot.long().clamp(
            0,
            self.num_learned_agents - 1,
        )
        value = joint_values[torch.arange(joint_values.shape[0]), slot_idx]

        logratio = logprob_curr - logprob_buffer
        ratio = logratio.exp()

        advantage, return_estimates = estimate_advantages_and_returns_grouped(
            rewards,
            value.detach(),
            termination,
            truncation,
            env_idx,
            agent_slot,
            time,
        )
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        clamped_ratio_adv = (
            torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
            * advantage
        )
        ratio_adv = ratio * advantage
        policy_loss = -torch.minimum(ratio_adv, clamped_ratio_adv).mean()

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
