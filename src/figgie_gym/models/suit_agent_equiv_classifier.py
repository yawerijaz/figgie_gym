# pyright: reportPrivateImportUsage=false
"""Equivarant classifier on suits."""

from typing import cast

import structlog
import torch
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,
    stack,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, concat, nn

from figgie_gym.models.building_blocks import (
    CommonTensorDictLightningModule,
    SequentialLinear,
)

logger = structlog.get_logger(__name__)


class SuitAgentEquivClassifier(CommonTensorDictLightningModule):
    """A classifier with shared blocks."""

    def __init__(
        self,
        main_body_dims: tuple[int, tuple[int, ...], int],
        per_suit_per_agent_dims: tuple[int, tuple[int, ...], int],
        lr: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        (
            per_suit_per_agent_in_dim,
            per_suit_per_agent_hidden_dims,
            per_suit_per_agent_out_dim,
        ) = per_suit_per_agent_dims

        main_body_in_dim, main_body_hidden_dims, num_classes = main_body_dims

        self.suit_agent_shared = SequentialLinear(
            per_suit_per_agent_in_dim,
            per_suit_per_agent_hidden_dims,
            per_suit_per_agent_out_dim,
            dropout,
        )
        self.main_body = SequentialLinear(
            main_body_in_dim,
            main_body_hidden_dims,
            num_classes,
            dropout,
        )

        self.lr = lr

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: TensorDict) -> torch.Tensor:
        agent_trade_summaries_dict = cast(
            "TensorDict",
            x["per_suit", "trade_summaries"],
        )
        agent_trade_summaries_tensors = [
            cast("Tensor", agent_trade_summaries_dict[k])
            for k in [
                "buy_quantity",
                "buy_consideration",
                "sell_quantity",
                "sell_consideration",
                "min_net_quantity_change",
            ]
        ]
        agent_trade_summaries = stack(agent_trade_summaries_tensors, dim=-1)
        agent_trade_summaries_out = self.suit_agent_shared(
            agent_trade_summaries,
        )

        suit_dict = cast(
            "TensorDict",
            x["per_suit"],
        )
        suit_tensors = [
            cast("Tensor", suit_dict[k])
            for k in [
                "known_count",
                "market_bid_price",
                "market_bid_quantity",
                "market_ask_price",
                "market_ask_quantity",
                "last_price",
                "volume",
                "self_position",
            ]
        ]

        suit = concat(
            [
                stack(suit_tensors, dim=-1),
                cast("Tensor", agent_trade_summaries_out.flatten(-2)),
            ],
            dim=-1,
        )

        top_level_tensors = [
            cast("Tensor", x[k])
            for k in [
                "time",
                "remaining_time",
                "remaining_time_fraction",
                "cash",
            ]
        ]

        top = concat([stack(top_level_tensors, -1), suit.flatten(-2)], dim=-1)

        return self.main_body(top)
