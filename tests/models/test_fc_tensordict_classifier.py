"""Tests for fc_tensordict_classifier module."""

import torch
from torch import nn

from figgie_gym.models.building_blocks import SequentialLinear
from figgie_gym.models.fc_tensordict_classifier import FCTensorDictClassifier


def test_initialization_basic() -> None:
    """Test basic initialization with default parameters."""
    main_body_dims = (10, (32, 16), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)
    assert model is not None


def test_initialization_with_lr() -> None:
    """Test initialization with custom learning rate."""
    main_body_dims = (10, (32,), 4)
    lr = 1e-4
    model = FCTensorDictClassifier(
        main_body_dims=main_body_dims,
        lr=lr,
    )
    assert model.lr == lr


def test_initialization_with_dropout() -> None:
    """Test initialization with custom dropout."""
    main_body_dims = (10, (32,), 4)
    dropout = 0.5
    model = FCTensorDictClassifier(
        main_body_dims=main_body_dims,
        dropout=dropout,
    )
    assert model is not None


def test_has_main_body() -> None:
    """Test that model has main_body attribute."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)
    assert hasattr(model, "main_body")


def test_has_loss_fn() -> None:
    """Test that model has loss function."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)
    assert hasattr(model, "loss_fn")
    assert isinstance(model.loss_fn, nn.CrossEntropyLoss)


def test_main_body_forward_pass() -> None:
    """Test forward pass with main_body directly."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    # Test main body directly with raw tensor
    x = torch.randn(2, 10)
    output = model.main_body(x)
    assert output.shape == (2, 4)


def test_main_body_single_sample() -> None:
    """Test main body forward pass with single sample."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    x = torch.randn(1, 10)
    output = model.main_body(x)
    assert output.shape == (1, 4)


def test_main_body_large_batch() -> None:
    """Test main body forward pass with large batch."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    x = torch.randn(1000, 10)
    output = model.main_body(x)
    assert output.shape == (1000, 4)


def test_output_shape_matches_num_classes() -> None:
    """Test that output shape matches number of classes."""
    num_classes = 3
    main_body_dims = (10, (32,), num_classes)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    x = torch.randn(5, 10)
    output = model.main_body(x)
    assert output.shape[1] == num_classes


def test_output_dtype() -> None:
    """Test that output has correct dtype."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    x = torch.randn(2, 10, dtype=torch.float32)
    output = model.main_body(x)
    assert output.dtype == torch.float32


def test_hyperparameters_saved() -> None:
    """Test that hyperparameters are saved."""
    main_body_dims = (10, (32,), 4)
    lr = 1e-4
    dropout = 0.2
    model = FCTensorDictClassifier(
        main_body_dims=main_body_dims,
        lr=lr,
        dropout=dropout,
    )
    assert hasattr(model, "hparams")


def test_batch_processing_various_sizes() -> None:
    """Test processing of different batch sizes."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    batch_sizes = [1, 2, 8, 16, 32]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 10)
        output = model.main_body(x)
        assert output.shape[0] == batch_size
        assert output.shape[1] == 4


def test_main_body_is_sequential_linear() -> None:
    """Test that main_body is a SequentialLinear instance."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    assert isinstance(model.main_body, SequentialLinear)


def test_loss_function_is_crossentropy() -> None:
    """Test that loss function is CrossEntropyLoss."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)
    assert isinstance(model.loss_fn, nn.CrossEntropyLoss)


def test_different_hidden_dims() -> None:
    """Test model with various hidden dimension configurations."""
    test_configs = [
        (10, (), 4),  # No hidden layers
        (10, (32,), 4),  # Single hidden layer
        (10, (64, 32), 4),  # Two hidden layers
        (10, (128, 64, 32), 4),  # Three hidden layers
    ]

    for input_dim, hidden_dims, output_dim in test_configs:
        main_body_dims = (input_dim, hidden_dims, output_dim)
        model = FCTensorDictClassifier(main_body_dims=main_body_dims)

        x = torch.randn(2, input_dim)
        output = model.main_body(x)
        assert output.shape == (2, output_dim)


def test_different_learning_rates() -> None:
    """Test model with different learning rates."""
    main_body_dims = (10, (32,), 4)
    lrs = [1e-5, 1e-4, 1e-3, 1e-2]

    for lr in lrs:
        model = FCTensorDictClassifier(
            main_body_dims=main_body_dims,
            lr=lr,
        )
        assert model.lr == lr


def test_model_parameters_trainable() -> None:
    """Test that model has trainable parameters."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0

    # Check that all parameters require grad
    for param in model.parameters():
        assert param.requires_grad


def test_model_gradient_flow() -> None:
    """Test that gradients flow through the model."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    x = torch.randn(2, 10, requires_grad=True)
    output = model.main_body(x)
    loss = output.sum()
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None

    # Check that model parameters have gradients
    for param in model.parameters():
        assert param.grad is not None


def test_different_input_dimensions() -> None:
    """Test model with different input dimensions."""
    input_dims = [5, 10, 20, 50, 100]
    for input_dim in input_dims:
        main_body_dims = (input_dim, (32,), 4)
        model = FCTensorDictClassifier(main_body_dims=main_body_dims)

        x = torch.randn(2, input_dim)
        output = model.main_body(x)
        assert output.shape == (2, 4)


def test_different_output_dimensions() -> None:
    """Test model with different output dimensions."""
    output_dims = [2, 3, 4, 5, 10]
    for output_dim in output_dims:
        main_body_dims = (10, (32,), output_dim)
        model = FCTensorDictClassifier(main_body_dims=main_body_dims)

        x = torch.randn(2, 10)
        output = model.main_body(x)
        assert output.shape[1] == output_dim


def test_model_eval_mode() -> None:
    """Test that model can be set to eval mode."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    model.eval()

    # In eval mode, dropout should be disabled
    x = torch.randn(2, 10)
    output = model.main_body(x)
    assert output.shape == (2, 4)


def test_model_train_mode() -> None:
    """Test that model can be set to train mode."""
    main_body_dims = (10, (32,), 4)
    model = FCTensorDictClassifier(main_body_dims=main_body_dims)

    model.train()

    x = torch.randn(2, 10)
    output = model.main_body(x)
    assert output.shape == (2, 4)
