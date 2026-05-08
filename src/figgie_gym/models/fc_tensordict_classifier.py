from typing import cast

import structlog
import torch
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,
    stack,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn

from figgie_gym.models.building_blocks import (
    CommonTensorDictLightningModule,
    SequentialLinear,
)

logger = structlog.get_logger(__name__)


class FCTensorDictClassifier(CommonTensorDictLightningModule):
    """A simple fully-connected classifier accepting TensorDicts.

    Args:
        main_body_dims: input, intermediate and output dimensions
        lr: learning rate
        dropout: dropout probability between layers

    """

    def __init__(
        self,
        main_body_dims: tuple[int, tuple[int, ...], int],
        lr: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        main_body_in_dim, main_body_hidden_dims, num_classes = main_body_dims

        self.main_body = SequentialLinear(
            main_body_in_dim,
            main_body_hidden_dims,
            num_classes,
            dropout,
        )

        self.lr = lr

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: TensorDict) -> torch.Tensor:
        stacked = cast("Tensor", stack(list(x.values()), dim=-1))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        return self.main_body(stacked)
