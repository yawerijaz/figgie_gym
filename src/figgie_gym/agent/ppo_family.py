from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import nn

from figgie_gym.envs.common import (
    SUITS,
    ActionOnSuit,
    ActionType,
    Agent,
    BeliefType,
    ExtraType,
    ObsType,
)
from figgie_gym.models.mappo import MAPPONetworks
from figgie_gym.models.mappo_blocks import MAPPOLightningModule
from figgie_gym.models.ppo import (
    PPONetworks,
)
from figgie_gym.models.ppo_blocks import PPOLightningModule
from figgie_gym.pipelines.observations_to_tensordict import (
    observations_to_tensordict_direct,
)
from figgie_gym.pipelines.tendordict_preprocess import (
    preprocessors,
)
from figgie_gym.utilities import get_device

if TYPE_CHECKING:
    from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
        TensorDict,  # pyright: ignore[reportUnknownVariableType]
    )


class PPOAgent(Agent):
    """PPO policy agent."""

    def __init__(
        self,
        lightning_module: PPOLightningModule | MAPPOLightningModule,
        networks: PPONetworks | MAPPONetworks,
    ) -> None:
        self.lightning_module = lightning_module
        self.networks = networks

    def act(
        self,
        random_number_generator: np.random.Generator,
        observation: ObsType,
    ) -> tuple[BeliefType, ActionType, float]:
        return self.act_batch(random_number_generator, [observation])[0]

    def act_batch(
        self,
        random_number_generator: np.random.Generator,  # noqa: ARG002
        observations: list[ObsType],
    ) -> list[tuple[BeliefType, ActionType, ExtraType]]:
        self.networks.eval()
        with torch.no_grad():
            td = observations_to_tensordict_direct(observations)
            processed_td = preprocessors["nested"].preprocess(
                td.unflatten_keys(),
                apply_soft_clip_on_prices=True,
                stage="inference",
            )
            x = cast(
                "TensorDict",
                processed_td["x"].to(device=get_device()),  # pyright: ignore[reportUnknownMemberType]
            )
            (
                action_tensor,
                action_log_prob_tensor,
                _entropy,
                _value,
                belief_logits,
                extras,
            ) = self.networks(x)

            action_tensor = action_tensor.cpu()
            probs = nn.functional.softmax(belief_logits.cpu(), dim=-1)
            batch_size, _, _ = action_tensor.shape
            actions: list[tuple[BeliefType, ActionType, ExtraType]] = []
            for i in range(batch_size):
                belief: BeliefType = {
                    s: float(probs[i, s_idx]) for s_idx, s in enumerate(SUITS)
                }
                action: ActionType = {
                    s: ActionOnSuit(*action_tensor[i, s_idx].tolist())
                    for s_idx, s in enumerate(SUITS)
                }
                log_prob = float(action_log_prob_tensor[i])
                extra = {
                    "pricing": {
                        s: {
                            "fair": extras["fair"][i, s_idx].item(),
                            "bias": extras["bias"][i, s_idx].item(),
                            "center": extras["center"][i, s_idx].item(),
                        }
                        for s_idx, s in enumerate(SUITS)
                    },
                    "log_prob": log_prob,
                }
                actions.append((belief, action, extra))
            return actions


class MAPPOAgent(PPOAgent): ...
