# ruff: noqa: F722
from typing import NamedTuple, cast

import torch
from jaxtyping import Float
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,
    stack,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn

from figgie_gym.models.building_blocks import (
    CommonTensorDictLightningModule,
    SequentialLinear,
)


class TradeSummaryEmbedding(nn.Module):
    """Embeds trade summaries using linear layers."""

    def __init__(
        self,
        trade_summary_input_dim: int,
        trade_summary_hidden_dims: tuple[int, ...],
        trade_summary_embed_dim: int,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.model = SequentialLinear(
            trade_summary_input_dim,
            trade_summary_hidden_dims,
            trade_summary_embed_dim,
        )

    def forward(
        self,
        x: Float[Tensor, "batch suit player ts_input"],
    ) -> Float[Tensor, "batch suit player ts_embed"]:
        return self.model(x)


class SameSuitContext(nn.Module):
    """Contextualizes each player's knowledge about a suit with other players' actions in the same suit.

    Think of it like adding all users' publicly known trades in a same suit together.
    Since the game is symmetric on players' seat, this layer is permutation invariant.
    """

    def __init__(
        self,
        trade_summary_embed_dim: int,
        known_trade_aggregate_hidden_dims: tuple[int, ...],
        known_trade_aggregate_embed_dim: int,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.prepool_model = SequentialLinear(
            trade_summary_embed_dim,
            known_trade_aggregate_hidden_dims,
            known_trade_aggregate_embed_dim,
        )

    def forward(
        self,
        trade_summary_embedding: Float[Tensor, "batch suit player ts_embed"],
    ) -> Float[Tensor, "batch suit known_trade_agg_embed"]:
        prepool_transformed = self.prepool_model(trade_summary_embedding)
        return prepool_transformed.sum(dim=2)


class AllSingleSuitContext(nn.Module):
    """Combines player's proprietary holding information with pooled knowledge learned from other player's action."""

    def __init__(
        self,
        private_holding_in_dim: int,
        known_trade_aggregate_embed_dim: int,
        suit_embed_hidden_dims: tuple[int, ...],
        suit_embed_dim: int,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.model = SequentialLinear(
            private_holding_in_dim + known_trade_aggregate_embed_dim,
            suit_embed_hidden_dims,
            suit_embed_dim,
        )

    def forward(
        self,
        private_suit_info: Float[Tensor, "batch suit private"],
        known_trade_embed: Float[Tensor, "batch suit known_trade_agg_embed"],
    ) -> Float[Tensor, "batch suit suit_embed_dim"]:
        all_suit_info = torch.concat(  # pyright: ignore[reportPrivateImportUsage]
            [private_suit_info, known_trade_embed],
            dim=-1,
        )
        return self.model(all_suit_info)


class AllOtherSuitsContext(nn.Module):
    """Contextualize information regarding the other suit of the same colored.

    Equivariant layer within color - swapping Club and Spade is the same game with everything swapped at each decision point.

    Equivariant layer within color - swapping Club and Spade is the same game with everything swapped at each decision point.
    Invariant to opposite colored suits.

    Equivariant with respect to colors. Swapping red and black changes the game symmetrically.

    E.g. When contextualizing Clubs, swapping Diamonds and Hearts doesn't change the game.
    But Swapping Clubs and Spades would change the game symmetrically.

    Swapping Heart and Spade will be a totally different, unrelated game.

    A simple flat model for now.

    A simple way to do this is to group by color, and for each suit, concatenate the other suit of the same color, and the sum of all suits of the other color as context.

    """

    def __init__(
        self,
        suit_embed_dim: int,
        other_suit_context_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.model = SequentialLinear(
            3 * suit_embed_dim,
            other_suit_context_hidden_dims,
            suit_embed_dim,
        )

    def forward(
        self,
        suit_info: Float[Tensor, "batch suit suit_embed_dim"],
    ) -> Float[Tensor, "batch suit suit_embed_dim"]:
        batch_dim, suit_dim, suit_embed_dim = suit_info.shape
        color_dim = 2  # red / black
        grouped_by_color = suit_info.reshape(
            batch_dim,
            color_dim,
            suit_dim // color_dim,
            suit_embed_dim,
        )  # batch color suit suit_embed_dim
        grouped_by_color_concat_with_other_suits = torch.concat(  # pyright: ignore[reportPrivateImportUsage]
            [
                grouped_by_color,
                grouped_by_color.flip(2),
                grouped_by_color.flip(1)
                .mean(dim=2, keepdim=True)
                .repeat(1, 1, 2, 1),
            ],
            dim=-1,
        )
        grouped_by_color_with_all_suits_context = self.model(
            grouped_by_color_concat_with_other_suits,
        )

        return grouped_by_color_with_all_suits_context.reshape(
            batch_dim,
            suit_dim,
            suit_embed_dim,
        )


class BeliefHead(nn.Module):
    def __init__(
        self,
        suit_embed_dim: int,
        belief_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.model = SequentialLinear(suit_embed_dim, belief_hidden_dims, 1)

    def forward(
        self,
        suit_full_context: Float[Tensor, "batch suit suit_embed_dim"],
    ) -> Float[Tensor, "batch suit"]:
        return self.model(suit_full_context).squeeze(-1)


class EquivariantClassifier(CommonTensorDictLightningModule):
    """A classifier with shared blocks."""

    def __init__(  # noqa: PLR0913
        self,
        trade_summary_input_dim: int,
        trade_summary_hidden_dims: tuple[int, ...],
        trade_summary_embed_dim: int,
        known_trade_aggregate_hidden_dims: tuple[int, ...],
        known_trade_aggregate_embed_dim: int,
        private_holding_in_dim: int,
        suit_embed_hidden_dims: tuple[int, ...],
        suit_embed_dim: int,
        other_suit_context_hidden_dims: tuple[int, ...],
        belief_hidden_dims: tuple[int, ...],
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.trade_summary_embedding = TradeSummaryEmbedding(
            trade_summary_input_dim,
            trade_summary_hidden_dims,
            trade_summary_embed_dim,
        )
        self.same_suit_context = SameSuitContext(
            trade_summary_embed_dim,
            known_trade_aggregate_hidden_dims,
            known_trade_aggregate_embed_dim,
        )
        self.all_single_suit_context = AllSingleSuitContext(
            private_holding_in_dim,
            known_trade_aggregate_embed_dim,
            suit_embed_hidden_dims,
            suit_embed_dim,
        )
        self.all_other_suits_context = AllOtherSuitsContext(
            suit_embed_dim,
            other_suit_context_hidden_dims,
        )
        self.belief_head = BeliefHead(suit_embed_dim, belief_hidden_dims)

        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: TensorDict) -> Float[Tensor, "batch suit"]:
        # Extract trade summaries: (batch, suit, 5 features)
        trade_summaries_dict = cast(
            "TensorDict",
            x["per_suit", "trade_summaries"],
        )
        trade_summaries_tensors = [
            cast("Tensor", trade_summaries_dict[k])
            for k in [
                "buy_quantity",
                "buy_consideration",
                "sell_quantity",
                "sell_consideration",
                "min_net_quantity_change",
            ]
        ]
        trade_summaries = stack(trade_summaries_tensors, dim=-1).float()

        # Embed trade summaries
        ts_embed = self.trade_summary_embedding(trade_summaries)

        # Get same suit context (pooled across players)
        # FLAW: We don't distingush self vs other's info in the pooling step. Handle it later.
        same_suit_context = self.same_suit_context(ts_embed)

        # Extract private suit info: (batch, suit, features)
        suit_dict = cast("TensorDict", x["per_suit"])
        private_suit_tensors = [
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
        # Stack into (batch, suit, features) instead of concatenating
        private_suit_info = torch.stack(private_suit_tensors, dim=-1).float()  # pyright: ignore[reportPrivateImportUsage]

        # Combine private info with same suit context
        all_single_suit_context = self.all_single_suit_context(
            private_suit_info,
            same_suit_context,
        )

        # Contextualize with other suits (equivariant across colors)
        all_other_suits_context = self.all_other_suits_context(
            all_single_suit_context,
        )

        # Generate beliefs for each suit
        logits = self.belief_head(all_other_suits_context)

        # Pool across suits to get final prediction
        return logits  # noqa: RET504


class EquivariantObservationArchitectureArgs(NamedTuple):
    trade_summary_input_dim: int
    trade_summary_hidden_dims: tuple[int, ...]
    trade_summary_embed_dim: int
    known_trade_aggregate_hidden_dims: tuple[int, ...]
    known_trade_aggregate_embed_dim: int
    private_holding_in_dim: int
    suit_embed_hidden_dims: tuple[int, ...]
    suit_embed_dim: int
    other_suit_context_hidden_dims: tuple[int, ...]


class EquivariantBody(nn.Module):
    """An actor with shared blocks."""

    def __init__(
        self,
        equivariant_args: EquivariantObservationArchitectureArgs,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        self.trade_summary_embedding = TradeSummaryEmbedding(
            equivariant_args.trade_summary_input_dim,
            equivariant_args.trade_summary_hidden_dims,
            equivariant_args.trade_summary_embed_dim,
        )
        self.same_suit_context = SameSuitContext(
            equivariant_args.trade_summary_embed_dim,
            equivariant_args.known_trade_aggregate_hidden_dims,
            equivariant_args.known_trade_aggregate_embed_dim,
        )
        self.all_single_suit_context = AllSingleSuitContext(
            equivariant_args.private_holding_in_dim,
            equivariant_args.known_trade_aggregate_embed_dim,
            equivariant_args.suit_embed_hidden_dims,
            equivariant_args.suit_embed_dim,
        )
        self.all_other_suits_context = AllOtherSuitsContext(
            equivariant_args.suit_embed_dim,
            equivariant_args.other_suit_context_hidden_dims,
        )

    def forward(
        self,
        x: TensorDict,
    ) -> Float[Tensor, "batch suit suit_embed_dim"]:
        # Extract trade summaries: (batch, suit, 5 features)
        trade_summaries_dict = cast(
            "TensorDict",
            x["per_suit", "trade_summaries"],
        )
        trade_summaries_tensors = [
            cast("Tensor", trade_summaries_dict[k])
            for k in [
                "buy_quantity",
                "buy_consideration",
                "sell_quantity",
                "sell_consideration",
                "min_net_quantity_change",
            ]
        ]
        trade_summaries = stack(trade_summaries_tensors, dim=-1).float()

        # Embed trade summaries
        ts_embed = self.trade_summary_embedding(trade_summaries)

        # Get same suit context (pooled across players)
        # FLAW: We don't distingush self vs other's info in the pooling step. Handle it later.
        same_suit_context = self.same_suit_context(ts_embed)

        # Extract private suit info: (batch, suit, features)
        suit_dict = cast("TensorDict", x["per_suit"])
        private_suit_tensors = [
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
        # Stack into (batch, suit, features) instead of concatenating
        private_suit_info = torch.stack(private_suit_tensors, dim=-1).float()  # pyright: ignore[reportPrivateImportUsage]

        # Combine private info with same suit context
        all_single_suit_context = self.all_single_suit_context(
            private_suit_info,
            same_suit_context,
        )

        # Contextualize with other suits (equivariant across colors)
        return self.all_other_suits_context(
            all_single_suit_context,
        )
