import torch
from jaxtyping import Float
from tensordict import TensorDict  # pyright: ignore[reportMissingTypeStubs]
from torch import Tensor

from figgie_gym.envs.common import SUITS
from figgie_gym.models.building_blocks import CommonTensorDictLightningModule


class NaiveClassifier(CommonTensorDictLightningModule):
    def forward(self, x: TensorDict) -> Float[Tensor, "batch suit"]:  # noqa: F722
        return torch.full((len(x), len(SUITS)), 0.25)
