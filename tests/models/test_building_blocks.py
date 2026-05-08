"""Tests for building_blocks module."""

import torch
from tensordict import TensorDict
from torch import nn

from figgie_gym.models.building_blocks import (
    SequentialLinear,
    unpack,
)


class TestSequentialLinear:
    """Test SequentialLinear module."""

    def test_initialization_simple(self) -> None:
        """Test simple initialization with default parameters."""
        model = SequentialLinear(
            input_dim=10,
            output_dim=4,
        )
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_initialization_with_hidden_layers(self) -> None:
        """Test initialization with hidden layers."""
        model = SequentialLinear(
            input_dim=20,
            hidden_dims=(64, 32),
            output_dim=4,
        )
        assert model is not None
        assert isinstance(model.model, nn.Sequential)

    def test_initialization_with_dropout(self) -> None:
        """Test initialization with custom dropout."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
            dropout=0.5,
        )
        assert model is not None

    def test_initialization_no_dropout(self) -> None:
        """Test initialization with no dropout."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
            dropout=0.0,
        )
        assert model is not None

    def test_forward_pass_simple(self) -> None:
        """Test forward pass with simple model."""
        model = SequentialLinear(
            input_dim=10,
            output_dim=4,
        )
        x = torch.randn(2, 10)
        output = model.forward(x)
        assert output.shape == (2, 4)

    def test_forward_pass_with_hidden_layers(self) -> None:
        """Test forward pass with hidden layers."""
        model = SequentialLinear(
            input_dim=20,
            hidden_dims=(64, 32),
            output_dim=10,
        )
        x = torch.randn(4, 20)
        output = model.forward(x)
        assert output.shape == (4, 10)

    def test_forward_pass_batch_size_one(self) -> None:
        """Test forward pass with batch size 1."""
        model = SequentialLinear(
            input_dim=8,
            hidden_dims=(16,),
            output_dim=2,
        )
        x = torch.randn(1, 8)
        output = model.forward(x)
        assert output.shape == (1, 2)

    def test_forward_pass_large_batch(self) -> None:
        """Test forward pass with large batch."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
        )
        x = torch.randn(1000, 10)
        output = model.forward(x)
        assert output.shape == (1000, 4)

    def test_model_structure_contains_layernorm(self) -> None:
        """Test that model structure contains LayerNorm layers."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
        )
        # Check that model contains LayerNorm
        has_layernorm = any(isinstance(m, nn.LayerNorm) for m in model.model)
        assert has_layernorm

    def test_model_structure_contains_relu(self) -> None:
        """Test that model structure contains ReLU activation."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
        )
        # Check that model contains ReLU
        has_relu = any(isinstance(m, nn.ReLU) for m in model.model)
        assert has_relu

    def test_model_structure_no_relu_at_output(self) -> None:
        """Test that output layer does not have ReLU."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
        )
        # Get last layers
        layers_list = list(model.model)
        # Last layer should be Linear, not ReLU
        assert isinstance(layers_list[-1], nn.Linear)

    def test_dropout_layer_presence(self) -> None:
        """Test that dropout layers are added when dropout > 0."""
        model_with_dropout = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
            dropout=0.3,
        )
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in model_with_dropout.model
        )
        assert has_dropout

        model_no_dropout = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
            dropout=0.0,
        )
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in model_no_dropout.model
        )
        assert not has_dropout

    def test_model_parameter_count(self) -> None:
        """Test that model has trainable parameters."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
        )
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_model_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
        )
        x = torch.randn(2, 10, requires_grad=True)
        output = model.forward(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None

        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None

    def test_output_dtype(self) -> None:
        """Test that output has correct dtype."""
        model = SequentialLinear(
            input_dim=10,
            output_dim=4,
        )
        x = torch.randn(2, 10, dtype=torch.float32)
        output = model.forward(x)
        assert output.dtype == torch.float32

    def test_single_hidden_layer(self) -> None:
        """Test model with single hidden layer."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(32,),
            output_dim=4,
        )
        x = torch.randn(5, 10)
        output = model.forward(x)
        assert output.shape == (5, 4)

    def test_multiple_hidden_layers(self) -> None:
        """Test model with multiple hidden layers."""
        model = SequentialLinear(
            input_dim=10,
            hidden_dims=(64, 32, 16),
            output_dim=4,
        )
        x = torch.randn(5, 10)
        output = model.forward(x)
        assert output.shape == (5, 4)


class TestUnpack:
    """Test the unpack utility function."""

    def test_unpack_basic(self) -> None:
        """Test basic unpacking of TensorDict."""
        x_dict = TensorDict(
            {"a": torch.randn(4, 3), "b": torch.randn(4, 2)}, batch_size=4
        )
        y = torch.tensor([0, 1, 2, 3])
        y_card_counter = torch.randn(4, 4)

        batch = TensorDict(
            {
                "x": x_dict,
                "y": y,
                "y_card_counter": y_card_counter,
            },
            batch_size=4,
        )

        x_out, y_out, y_cc_out = unpack(batch)

        assert isinstance(x_out, TensorDict)
        assert torch.equal(y_out, y)
        assert torch.equal(y_cc_out, y_card_counter)

    def test_unpack_preserves_data(self) -> None:
        """Test that unpack preserves data correctly."""
        x_dict = TensorDict(
            {"a": torch.tensor([[1.0, 2.0]]), "b": torch.tensor([[3.0, 4.0]])},
            batch_size=1,
        )
        y = torch.tensor([1])
        y_card_counter = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        batch = TensorDict(
            {
                "x": x_dict,
                "y": y,
                "y_card_counter": y_card_counter,
            },
            batch_size=1,
        )

        _, y_out, y_cc_out = unpack(batch)

        assert y_out.item() == 1
        assert torch.allclose(y_cc_out[0], torch.tensor([0.1, 0.2, 0.3, 0.4]))
