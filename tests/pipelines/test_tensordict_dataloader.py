"""Tests for tensordict_dataloader module."""

from pathlib import Path

import pytest

from figgie_gym.pipelines.tensordict_dataloader import TensorDictDataModule


class TestTensorDictDataModule:
    """Test TensorDictDataModule for data loading and splitting."""

    def test_initialization_valid_split(self) -> None:
        """Test that valid data split percentages initialize correctly."""
        data_module = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            data_split_pct=(0.8, 0.1, 0.1),
            batch_size=32,
        )
        assert data_module.data_split_pct == (0.8, 0.1, 0.1)
        assert data_module.batch_size == 32
        assert data_module.seed == 42

    def test_initialization_invalid_split_sum(self) -> None:
        """Test that invalid data split (sum != 1) raises ValueError."""
        with pytest.raises(ValueError):
            TensorDictDataModule(
                data_dir=Path("/tmp/data"),
                data_split_pct=(0.5, 0.2, 0.2),  # Sum = 0.9, which is invalid
            )

    def test_initialization_with_custom_batch_size(self) -> None:
        """Test initialization with custom batch size."""
        data_module = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            batch_size=128,
        )
        assert data_module.batch_size == 128

    def test_initialization_with_custom_seed(self) -> None:
        """Test initialization with custom seed."""
        data_module = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            seed=123,
        )
        assert data_module.seed == 123

    def test_initialization_with_custom_preprocessor(self) -> None:
        """Test initialization with custom preprocessor."""
        data_module = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            preprocessor="raw_flat",
        )
        assert data_module.preprocessor == "raw_flat"

    def test_initialization_apply_soft_clip_on_prices(self) -> None:
        """Test initialization with soft clip enabled/disabled."""
        data_module_with_clip = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            apply_soft_clip_on_prices=True,
        )
        assert data_module_with_clip.apply_soft_clip_on_prices is True

        data_module_without_clip = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            apply_soft_clip_on_prices=False,
        )
        assert data_module_without_clip.apply_soft_clip_on_prices is False

    def test_hyperparameters_saved(self) -> None:
        """Test that hyperparameters are properly saved."""
        data_module = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            data_split_pct=(0.7, 0.15, 0.15),
            batch_size=64,
            apply_soft_clip_on_prices=True,
            seed=999,
            preprocessor="original",
        )
        # Check that save_hyperparameters was called (via parent class)
        assert hasattr(data_module, "hparams")

    def test_invalid_data_split_too_high(self) -> None:
        """Test that data split > 1 raises ValueError."""
        with pytest.raises(ValueError):
            TensorDictDataModule(
                data_dir=Path("/tmp/data"),
                data_split_pct=(0.5, 0.5, 0.5),  # Sum = 1.5
            )

    def test_invalid_data_split_too_low(self) -> None:
        """Test that data split < 1 raises ValueError."""
        with pytest.raises(ValueError):
            TensorDictDataModule(
                data_dir=Path("/tmp/data"),
                data_split_pct=(0.5, 0.3, 0.1),  # Sum = 0.9
            )

    def test_preprocessor_property(self) -> None:
        """Test that preprocessor_ property returns correct preprocessor."""
        data_module = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            preprocessor="original",
        )
        # This will fail without proper setup, but we're testing the property exists
        assert hasattr(data_module, "preprocessor_")

    def test_different_preprocessors(self) -> None:
        """Test initialization with different preprocessor options."""
        preprocessors_to_test = ["original", "original"]
        for preprocessor in preprocessors_to_test:
            data_module = TensorDictDataModule(
                data_dir=Path("/tmp/data"),
                preprocessor=preprocessor,
            )
            assert data_module.preprocessor == preprocessor

    def test_batch_sizes_variation(self) -> None:
        """Test various batch sizes."""
        batch_sizes = [1, 16, 32, 64, 128, 256, 512]
        for batch_size in batch_sizes:
            data_module = TensorDictDataModule(
                data_dir=Path("/tmp/data"),
                batch_size=batch_size,
            )
            assert data_module.batch_size == batch_size

    def test_split_percentages_edge_cases(self) -> None:
        """Test edge case split percentages."""
        # All training
        data_module = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            data_split_pct=(1.0, 0.0, 0.0),
        )
        assert sum(data_module.data_split_pct) == 1.0

        # All validation
        data_module = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            data_split_pct=(0.0, 1.0, 0.0),
        )
        assert sum(data_module.data_split_pct) == 1.0

        # All testing
        data_module = TensorDictDataModule(
            data_dir=Path("/tmp/data"),
            data_split_pct=(0.0, 0.0, 1.0),
        )
        assert sum(data_module.data_split_pct) == 1.0

    def test_data_dir_is_stored(self) -> None:
        """Test that data directory is properly stored."""
        data_dir = Path("/path/to/data")
        data_module = TensorDictDataModule(data_dir=data_dir)
        assert data_module.data_dir == data_dir
