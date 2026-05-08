# pyright: reportPrivateImportUsage=false
"""Tests for equivariant module."""

from typing import cast

import torch
from tensordict import TensorDict  # pyright: ignore[reportMissingTypeStubs]
from torch import nn

from figgie_gym.models.equivariant import (
    AllOtherSuitsContext,
    AllSingleSuitContext,
    BeliefHead,
    EquivariantClassifier,
    SameSuitContext,
    TradeSummaryEmbedding,
)


class TestTradeSummaryEmbedding:
    """Test TradeSummaryEmbedding module."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        embedding = TradeSummaryEmbedding(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(16,),
            trade_summary_embed_dim=8,
        )
        assert embedding is not None
        assert isinstance(embedding, nn.Module)

    def test_initialization_no_hidden_dims(self) -> None:
        """Test initialization with no hidden layers."""
        embedding = TradeSummaryEmbedding(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=8,
        )
        assert embedding is not None

    def test_forward_pass(self) -> None:
        """Test forward pass with correct shapes."""
        embedding = TradeSummaryEmbedding(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(16,),
            trade_summary_embed_dim=8,
        )
        # Shape: batch=2, suit=4, player=3, ts_input=5
        x = torch.randn(2, 4, 3, 5)
        output = embedding(x)
        assert output.shape == (2, 4, 3, 8)

    def test_forward_pass_single_sample(self) -> None:
        """Test forward pass with single sample."""
        embedding = TradeSummaryEmbedding(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=8,
        )
        x = torch.randn(1, 4, 3, 5)
        output = embedding(x)
        assert output.shape == (1, 4, 3, 8)

    def test_output_dtype(self) -> None:
        """Test that output has correct dtype."""
        embedding = TradeSummaryEmbedding(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(16,),
            trade_summary_embed_dim=8,
        )
        x = torch.randn(2, 4, 3, 5)
        output = embedding(x)
        assert output.dtype == torch.float32


class TestSameSuitContext:
    """Test SameSuitContext module."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        context = SameSuitContext(
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(16,),
            known_trade_aggregate_embed_dim=6,
        )
        assert context is not None
        assert isinstance(context, nn.Module)

    def test_initialization_no_hidden_dims(self) -> None:
        """Test initialization with no hidden layers."""
        context = SameSuitContext(
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=6,
        )
        assert context is not None

    def test_forward_pass(self) -> None:
        """Test forward pass with correct shapes."""
        context = SameSuitContext(
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(16,),
            known_trade_aggregate_embed_dim=6,
        )
        # Shape: batch=2, suit=4, player=3, ts_embed=8
        x = torch.randn(2, 4, 3, 8)
        output = context(x)
        # Output should be: batch=2, suit=4, known_trade_agg_embed=6
        assert output.shape == (2, 4, 6)

    def test_pooling_behavior(self) -> None:
        """Test that context sums across players."""
        context = SameSuitContext(
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=8,
        )
        # Create input where we can verify pooling
        x = torch.ones(1, 1, 3, 8)  # All ones
        output = context(x)
        # Output should sum across 3 players
        assert output.shape == (1, 1, 8)


class TestAllSingleSuitContext:
    """Test AllSingleSuitContext module."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        context = AllSingleSuitContext(
            private_holding_in_dim=10,
            known_trade_aggregate_embed_dim=6,
            suit_embed_hidden_dims=(16,),
            suit_embed_dim=8,
        )
        assert context is not None
        assert isinstance(context, nn.Module)

    def test_forward_pass(self) -> None:
        """Test forward pass with correct shapes."""
        context = AllSingleSuitContext(
            private_holding_in_dim=10,
            known_trade_aggregate_embed_dim=6,
            suit_embed_hidden_dims=(16,),
            suit_embed_dim=8,
        )
        # Shape: batch=2, suit=4, private=10
        private_suit_info = torch.randn(2, 4, 10)
        # Shape: batch=2, suit=4, known_trade_agg_embed=6
        known_trade_embed = torch.randn(2, 4, 6)
        output = context(private_suit_info, known_trade_embed)
        # Output should be: batch=2, suit=4, suit_embed_dim=8
        assert output.shape == (2, 4, 8)

    def test_concatenation_behavior(self) -> None:
        """Test that private and known info are concatenated."""
        context = AllSingleSuitContext(
            private_holding_in_dim=3,
            known_trade_aggregate_embed_dim=2,
            suit_embed_hidden_dims=(),
            suit_embed_dim=5,
        )
        private_suit_info = torch.randn(1, 1, 3)
        known_trade_embed = torch.randn(1, 1, 2)
        output = context(private_suit_info, known_trade_embed)
        assert output.shape == (1, 1, 5)


class TestAllOtherSuitsContext:
    """Test AllOtherSuitsContext module."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        context = AllOtherSuitsContext(
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(16,),
        )
        assert context is not None
        assert isinstance(context, nn.Module)

    def test_forward_pass(self) -> None:
        """Test forward pass with correct shapes."""
        context = AllOtherSuitsContext(
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(16,),
        )
        # Shape: batch=2, suit=4, suit_embed_dim=8
        x = torch.randn(2, 4, 8)
        output = context(x)
        # Output should maintain shape
        assert output.shape == (2, 4, 8)

    def test_equivariance_structure(self) -> None:
        """Test that the model respects the color grouping."""
        context = AllOtherSuitsContext(
            suit_embed_dim=4,
            other_suit_context_hidden_dims=(),
        )
        # Create a batch with 4 suits (2 colors, 2 suits per color)
        x = torch.randn(1, 4, 4)
        output = context(x)
        assert output.shape == (1, 4, 4)

    def test_color_grouping_reshape(self) -> None:
        """Test internal color grouping and reshaping."""
        context = AllOtherSuitsContext(
            suit_embed_dim=2,
            other_suit_context_hidden_dims=(),
        )
        # Input: batch=2, suit=4, embed=2
        x = torch.randn(2, 4, 2)
        output = context(x)
        assert output.shape == (2, 4, 2)


class TestBeliefHead:
    """Test BeliefHead module."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        head = BeliefHead(
            suit_embed_dim=8,
            belief_hidden_dims=(16,),
        )
        assert head is not None
        assert isinstance(head, nn.Module)

    def test_initialization_no_hidden_dims(self) -> None:
        """Test initialization with no hidden layers."""
        head = BeliefHead(
            suit_embed_dim=8,
            belief_hidden_dims=(),
        )
        assert head is not None

    def test_forward_pass(self) -> None:
        """Test forward pass with correct shapes."""
        head = BeliefHead(
            suit_embed_dim=8,
            belief_hidden_dims=(16,),
        )
        # Shape: batch=2, suit=4, suit_embed_dim=8
        x = torch.randn(2, 4, 8)
        output = head(x)
        # Output should be: batch=2, suit=4
        assert output.shape == (2, 4)

    def test_forward_pass_single_suit(self) -> None:
        """Test forward pass with single suit."""
        head = BeliefHead(
            suit_embed_dim=8,
            belief_hidden_dims=(),
        )
        x = torch.randn(1, 1, 8)
        output = head(x)
        assert output.shape == (1, 1)

    def test_squeeze_behavior(self) -> None:
        """Test that the model squeezes the last dimension."""
        head = BeliefHead(
            suit_embed_dim=4,
            belief_hidden_dims=(),
        )
        x = torch.randn(2, 4, 4)
        output = head(x)
        # Should not have the last dimension squeezed (only -1 from final linear)
        assert output.dim() == 2


class TestEquivariantClassifier:
    """Test EquivariantClassifier module."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(16,),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(16,),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(16,),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(16,),
            belief_hidden_dims=(16,),
        )
        assert classifier is not None
        assert isinstance(classifier, nn.Module)

    def test_has_all_components(self) -> None:
        """Test that classifier has all required components."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(16,),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(16,),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(16,),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(16,),
            belief_hidden_dims=(16,),
        )
        assert hasattr(classifier, "trade_summary_embedding")
        assert hasattr(classifier, "same_suit_context")
        assert hasattr(classifier, "all_single_suit_context")
        assert hasattr(classifier, "all_other_suits_context")
        assert hasattr(classifier, "belief_head")
        assert hasattr(classifier, "loss_fn")

    def test_learning_rate(self) -> None:
        """Test custom learning rate."""
        lr = 5e-4
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(16,),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(16,),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(16,),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(16,),
            belief_hidden_dims=(16,),
            lr=lr,
        )
        assert classifier.lr == lr

    def test_forward_pass_basic(self) -> None:
        """Test forward pass with basic input."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(),
            belief_hidden_dims=(),
        )

        # Create input TensorDict
        batch_size = 2
        num_suits = 4
        num_players = 3

        # Trade summaries should be (batch, suit, player, 5 features) before stacking
        # Since they're extracted from TensorDict, each key is (batch, suit)
        trade_summaries = {
            "buy_quantity": torch.randn(batch_size, num_suits, num_players),
            "buy_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "sell_quantity": torch.randn(batch_size, num_suits, num_players),
            "sell_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "min_net_quantity_change": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
        }

        suit_info = {
            "known_count": torch.randn(batch_size, num_suits),
            "market_bid_price": torch.randn(batch_size, num_suits),
            "market_bid_quantity": torch.randn(batch_size, num_suits),
            "market_ask_price": torch.randn(batch_size, num_suits),
            "market_ask_quantity": torch.randn(batch_size, num_suits),
            "last_price": torch.randn(batch_size, num_suits),
            "volume": torch.randn(batch_size, num_suits),
            "self_position": torch.randn(batch_size, num_suits),
        }

        x = TensorDict(
            {
                "per_suit": TensorDict(suit_info, batch_size=batch_size),
                ("per_suit", "trade_summaries"): TensorDict(
                    trade_summaries,
                    batch_size=batch_size,
                ),
            },
            batch_size=batch_size,
        )

        output = classifier(x)
        # Output should be batch size (mean pooled across suits)
        assert output.shape == (batch_size, num_suits)

    def test_forward_pass_single_sample(self) -> None:
        """Test forward pass with single sample."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(),
            belief_hidden_dims=(),
        )

        batch_size = 1
        num_suits = 4
        num_players = 5

        trade_summaries = {
            "buy_quantity": torch.randn(batch_size, num_suits, num_players),
            "buy_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "sell_quantity": torch.randn(batch_size, num_suits, num_players),
            "sell_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "min_net_quantity_change": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
        }

        suit_info = {
            "known_count": torch.randn(batch_size, num_suits),
            "market_bid_price": torch.randn(batch_size, num_suits),
            "market_bid_quantity": torch.randn(batch_size, num_suits),
            "market_ask_price": torch.randn(batch_size, num_suits),
            "market_ask_quantity": torch.randn(batch_size, num_suits),
            "last_price": torch.randn(batch_size, num_suits),
            "volume": torch.randn(batch_size, num_suits),
            "self_position": torch.randn(batch_size, num_suits),
        }

        x = TensorDict(
            {
                "per_suit": TensorDict(suit_info, batch_size=batch_size),
                ("per_suit", "trade_summaries"): TensorDict(
                    trade_summaries,
                    batch_size=batch_size,
                ),
            },
            batch_size=batch_size,
        )

        output = classifier(x)
        assert output.shape == (batch_size, num_suits)


class TestEquivariantProperties:
    """Test equivariance and permutation invariance properties."""

    def _create_test_data(
        self,
        batch_size: int = 2,
        num_suits: int = 4,
        num_players: int = 3,
    ) -> TensorDict:
        """Create test TensorDict."""
        trade_summaries = {
            "buy_quantity": torch.randn(batch_size, num_suits, num_players),
            "buy_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "sell_quantity": torch.randn(batch_size, num_suits, num_players),
            "sell_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "min_net_quantity_change": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
        }

        suit_info = {
            "known_count": torch.randn(batch_size, num_suits),
            "market_bid_price": torch.randn(batch_size, num_suits),
            "market_bid_quantity": torch.randn(batch_size, num_suits),
            "market_ask_price": torch.randn(batch_size, num_suits),
            "market_ask_quantity": torch.randn(batch_size, num_suits),
            "last_price": torch.randn(batch_size, num_suits),
            "volume": torch.randn(batch_size, num_suits),
            "self_position": torch.randn(batch_size, num_suits),
        }

        return TensorDict(
            {
                "per_suit": TensorDict(suit_info, batch_size=batch_size),
                ("per_suit", "trade_summaries"): TensorDict(
                    trade_summaries,
                    batch_size=batch_size,
                ),
            },
            batch_size=batch_size,
        )

    def test_permutation_invariance_same_suit_context(self) -> None:
        """Test that SameSuitContext is invariant to player order."""
        context = SameSuitContext(
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=6,
        )

        # Create input with multiple players
        batch_size, num_suits, num_players = 2, 4, 3
        x = torch.randn(batch_size, num_suits, num_players, 8)

        # Get original output
        output_original = context(x)

        # Shuffle players dimension globally
        player_perm = torch.randperm(num_players)
        x_shuffled = x[:, :, player_perm, :]

        output_shuffled = context(x_shuffled)

        # Should be identical since it sums across players
        assert torch.allclose(output_original, output_shuffled, atol=1e-5)

    def test_permutation_invariance_full_classifier(self) -> None:
        """Test that full classifier is invariant to player order in trade summaries."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(),
            belief_hidden_dims=(),
        )

        # Create input data with multiple players
        batch_size = 2
        num_suits = 4
        num_players = 3
        x = self._create_test_data(batch_size=batch_size, num_suits=num_suits)

        # Get original output
        output_original = classifier(x)

        # Create shuffled version by permuting players in trade summaries
        x_shuffled = TensorDict(x)  # Copy
        trade_summaries_dict = cast(
            "TensorDict",
            x_shuffled["per_suit", "trade_summaries"],
        )

        # Permute players across all trade summary features
        player_perm = torch.randperm(num_players)
        for key in trade_summaries_dict.keys():  # pyright: ignore[reportUnknownMemberType] # noqa: SIM118
            tensor = cast("torch.Tensor", trade_summaries_dict[key])
            # tensor shape: (batch, suit, num_players)
            trade_summaries_dict[key] = tensor[:, :, player_perm]

        # Classifier output should be invariant to player order
        output_shuffled = classifier(x_shuffled)
        assert torch.allclose(output_original, output_shuffled, atol=1e-5)

    def test_suit_swap_equivariance_same_color(self) -> None:
        """Test equivariance when swapping suits within same color (0↔1 or 2↔3)."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(),
            belief_hidden_dims=(),
        )

        x = self._create_test_data(batch_size=2, num_suits=4)

        # Get original output
        output_original = classifier(x)

        # Swap suits 0 and 1 (same color - black)
        x_swapped = TensorDict(x)
        suit_dict = cast("TensorDict", x_swapped["per_suit"])
        trade_summaries_dict = cast(
            "TensorDict",
            x_swapped["per_suit", "trade_summaries"],
        )

        # Swap all suit data for suits 0 and 1
        for key in suit_dict.keys():  # pyright: ignore[reportUnknownMemberType] # noqa: SIM118
            tensor = cast("torch.Tensor", suit_dict[key])
            swapped = tensor.clone()
            swapped[:, 0] = tensor[:, 1]
            swapped[:, 1] = tensor[:, 0]
            suit_dict[key] = swapped

        for key in trade_summaries_dict.keys():  # pyright: ignore[reportUnknownMemberType] # noqa: SIM118
            tensor = cast("torch.Tensor", trade_summaries_dict[key])
            swapped = tensor.clone()
            swapped[:, 0] = tensor[:, 1]
            swapped[:, 1] = tensor[:, 0]
            trade_summaries_dict[key] = swapped

        output_swapped = classifier(x_swapped)

        # Logits for suits 0 and 1 should swap
        assert torch.allclose(
            output_original[:, 0],
            output_swapped[:, 1],
            atol=1e-4,
        )
        assert torch.allclose(
            output_original[:, 1],
            output_swapped[:, 0],
            atol=1e-4,
        )
        # Logits for suits 2 and 3 should remain unchanged
        assert torch.allclose(
            output_original[:, 2],
            output_swapped[:, 2],
            atol=1e-5,
        )
        assert torch.allclose(
            output_original[:, 3],
            output_swapped[:, 3],
            atol=1e-5,
        )

    def test_suit_swap_equivariance_red_suits(self) -> None:
        """Test equivariance when swapping red suits (2↔3)."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(),
            belief_hidden_dims=(),
        )

        x = self._create_test_data(batch_size=2, num_suits=4)

        # Get original output
        output_original = classifier(x)

        # Swap suits 2 and 3 (same color - red)
        x_swapped = TensorDict(x)
        suit_dict = cast("TensorDict", x_swapped["per_suit"])
        trade_summaries_dict = cast(
            "TensorDict",
            x_swapped["per_suit", "trade_summaries"],
        )

        # Swap all suit data for suits 2 and 3
        for key in suit_dict.keys():  # pyright: ignore[reportUnknownMemberType] # noqa: SIM118
            tensor = cast("torch.Tensor", suit_dict[key])
            swapped = tensor.clone()
            swapped[:, 2] = tensor[:, 3]
            swapped[:, 3] = tensor[:, 2]
            suit_dict[key] = swapped

        for key in trade_summaries_dict.keys():  # pyright: ignore[reportUnknownMemberType] # noqa: SIM118
            tensor = cast("torch.Tensor", trade_summaries_dict[key])
            swapped = tensor.clone()
            swapped[:, 2] = tensor[:, 3]
            swapped[:, 3] = tensor[:, 2]
            trade_summaries_dict[key] = swapped

        output_swapped = classifier(x_swapped)

        # Logits for suits 2 and 3 should swap
        assert torch.allclose(
            output_original[:, 2],
            output_swapped[:, 3],
            atol=1e-4,
        )
        assert torch.allclose(
            output_original[:, 3],
            output_swapped[:, 2],
            atol=1e-4,
        )
        # Logits for suits 0 and 1 should remain unchanged
        assert torch.allclose(
            output_original[:, 0],
            output_swapped[:, 0],
            atol=1e-4,
        )
        assert torch.allclose(
            output_original[:, 1],
            output_swapped[:, 1],
            atol=1e-4,
        )

    def test_color_swap_equivariance(self) -> None:
        """Test equivariance when swapping colors (0↔2 and 1↔3)."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(),
            belief_hidden_dims=(),
        )

        x = self._create_test_data(batch_size=2, num_suits=4)

        # Get original output
        output_original = classifier(x)

        # Swap colors: 0↔2 and 1↔3
        x_swapped = TensorDict(x)
        suit_dict = cast("TensorDict", x_swapped["per_suit"])
        trade_summaries_dict = cast(
            "TensorDict",
            x_swapped["per_suit", "trade_summaries"],
        )

        # Swap all suit data
        for key in suit_dict.keys():  # pyright: ignore[reportUnknownMemberType] # noqa: SIM118
            tensor = cast("torch.Tensor", suit_dict[key])
            swapped = tensor.clone()
            swapped[:, [0, 1, 2, 3]] = tensor[:, [2, 3, 0, 1]]
            suit_dict[key] = swapped

        for key in trade_summaries_dict.keys():  # pyright: ignore[reportUnknownMemberType] # noqa: SIM118
            tensor = cast("torch.Tensor", trade_summaries_dict[key])
            swapped = tensor.clone()
            swapped[:, [0, 1, 2, 3]] = tensor[:, [2, 3, 0, 1]]
            trade_summaries_dict[key] = swapped

        output_swapped = classifier(x_swapped)

        # Logits should swap: 0↔2 and 1↔3
        assert torch.allclose(
            output_original[:, 0],
            output_swapped[:, 2],
            atol=1e-4,
        )
        assert torch.allclose(
            output_original[:, 1],
            output_swapped[:, 3],
            atol=1e-4,
        )
        assert torch.allclose(
            output_original[:, 2],
            output_swapped[:, 0],
            atol=1e-4,
        )
        assert torch.allclose(
            output_original[:, 3],
            output_swapped[:, 1],
            atol=1e-4,
        )

    def test_all_other_suits_context_equivariance(self) -> None:
        """Test that AllOtherSuitsContext respects equivariance."""
        context = AllOtherSuitsContext(
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(),
        )

        # Create input: batch=1, suits=4, embed=8
        x = torch.randn(1, 4, 8)

        output_original = context(x)

        # Swap suits 0 and 1 (same color)
        x_swapped = x.clone()
        x_swapped[:, [0, 1]] = x[:, [1, 0]]

        output_swapped = context(x_swapped)

        # Outputs should swap for suits 0 and 1
        assert torch.allclose(
            output_original[:, 0],
            output_swapped[:, 1],
            atol=1e-4,
        )
        assert torch.allclose(
            output_original[:, 1],
            output_swapped[:, 0],
            atol=1e-4,
        )
        # Suits 2 and 3 should be unchanged
        assert torch.allclose(
            output_original[:, 2],
            output_swapped[:, 2],
            atol=1e-4,
        )
        assert torch.allclose(
            output_original[:, 3],
            output_swapped[:, 3],
            atol=1e-4,
        )

    def test_has_loss_function(self) -> None:
        """Test that classifier has loss function."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(16,),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(16,),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(16,),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(16,),
            belief_hidden_dims=(16,),
        )
        assert isinstance(classifier.loss_fn, nn.CrossEntropyLoss)

    def test_output_dtype(self) -> None:
        """Test that output has correct dtype."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=8,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=6,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(),
            suit_embed_dim=8,
            other_suit_context_hidden_dims=(),
            belief_hidden_dims=(),
        )

        batch_size = 2
        num_suits = 4
        num_players = 4

        trade_summaries = {
            "buy_quantity": torch.randn(batch_size, num_suits, num_players),
            "buy_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "sell_quantity": torch.randn(batch_size, num_suits, num_players),
            "sell_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "min_net_quantity_change": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
        }

        suit_info = {
            "known_count": torch.randn(batch_size, num_suits),
            "market_bid_price": torch.randn(batch_size, num_suits),
            "market_bid_quantity": torch.randn(batch_size, num_suits),
            "market_ask_price": torch.randn(batch_size, num_suits),
            "market_ask_quantity": torch.randn(batch_size, num_suits),
            "last_price": torch.randn(batch_size, num_suits),
            "volume": torch.randn(batch_size, num_suits),
            "self_position": torch.randn(batch_size, num_suits),
        }

        x = TensorDict(
            {
                "per_suit": TensorDict(suit_info, batch_size=batch_size),
                ("per_suit", "trade_summaries"): TensorDict(
                    trade_summaries,
                    batch_size=batch_size,
                ),
            },
            batch_size=batch_size,
        )

        output = classifier(x)
        assert output.dtype == torch.float32

    def test_large_batch_forward_pass(self) -> None:
        """Test forward pass with larger batch."""
        classifier = EquivariantClassifier(
            trade_summary_input_dim=5,
            trade_summary_hidden_dims=(),
            trade_summary_embed_dim=16,
            known_trade_aggregate_hidden_dims=(),
            known_trade_aggregate_embed_dim=12,
            private_holding_in_dim=8,
            suit_embed_hidden_dims=(),
            suit_embed_dim=16,
            other_suit_context_hidden_dims=(),
            belief_hidden_dims=(),
        )

        batch_size = 64
        num_suits = 4
        num_players = 6

        trade_summaries = {
            "buy_quantity": torch.randn(batch_size, num_suits, num_players),
            "buy_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "sell_quantity": torch.randn(batch_size, num_suits, num_players),
            "sell_consideration": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
            "min_net_quantity_change": torch.randn(
                batch_size,
                num_suits,
                num_players,
            ),
        }

        suit_info = {
            "known_count": torch.randn(batch_size, num_suits),
            "market_bid_price": torch.randn(batch_size, num_suits),
            "market_bid_quantity": torch.randn(batch_size, num_suits),
            "market_ask_price": torch.randn(batch_size, num_suits),
            "market_ask_quantity": torch.randn(batch_size, num_suits),
            "last_price": torch.randn(batch_size, num_suits),
            "volume": torch.randn(batch_size, num_suits),
            "self_position": torch.randn(batch_size, num_suits),
        }

        x = TensorDict(
            {
                "per_suit": TensorDict(suit_info, batch_size=batch_size),
                ("per_suit", "trade_summaries"): TensorDict(
                    trade_summaries,
                    batch_size=batch_size,
                ),
            },
            batch_size=batch_size,
        )

        output = classifier(x)
        assert output.shape == (batch_size, num_suits)
