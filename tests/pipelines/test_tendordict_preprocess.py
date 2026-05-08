"""Tests for tendordict_preprocess module."""

import torch

from figgie_gym.pipelines.tendordict_preprocess import (
    DEFAULT_PRICE,
    DEFAULT_QUANTITY,
    OriginalTensorDictPreprocessor,
    impute_nan,
    imputed_value_given_key,
)


class TestImputedValueGivenKey:
    """Test the imputed_value_given_key function."""

    def test_price_keys(self) -> None:
        """Test that price keys return DEFAULT_PRICE."""
        price_keys = [
            "last_price",
            "ask.price",
            "bid.price",
            "per_suit.Spade.last_price",
            "per_suit.Club.ask.price",
            "per_suit.Heart.bid.price",
        ]
        for key in price_keys:
            assert imputed_value_given_key(key) == DEFAULT_PRICE

    def test_quantity_keys(self) -> None:
        """Test that quantity keys return DEFAULT_QUANTITY."""
        quantity_keys = [
            "ask.quantity",
            "bid.quantity",
            "per_suit.Spade.ask.quantity",
            "per_suit.Club.bid.quantity",
        ]
        for key in quantity_keys:
            assert imputed_value_given_key(key) == DEFAULT_QUANTITY

    def test_unknown_keys(self) -> None:
        """Test that unknown keys return None."""
        unknown_keys = [
            "cash",
            "time",
            "unknown_field",
            "per_suit.Spade.known_count",
        ]
        for key in unknown_keys:
            assert imputed_value_given_key(key) is None

    def test_default_constants(self) -> None:
        """Test that default constants have expected values."""
        assert DEFAULT_PRICE == 5
        assert DEFAULT_QUANTITY == 0


class TestImputeNan:
    """Test the impute_nan function."""

    def test_impute_nan_with_price_key(self) -> None:
        """Test NaN imputation with price key."""
        tensor = torch.tensor([1.0, float("nan"), 3.0, float("nan")])
        result = impute_nan(
            imputed_value_given_key,
            "last_price",
            tensor,
        )
        assert torch.isnan(result).sum() == 0
        assert result[1].item() == DEFAULT_PRICE
        assert result[3].item() == DEFAULT_PRICE

    def test_impute_nan_with_quantity_key(self) -> None:
        """Test NaN imputation with quantity key."""
        tensor = torch.tensor([1.0, float("nan"), 3.0, float("nan")])
        result = impute_nan(
            imputed_value_given_key,
            "ask.quantity",
            tensor,
        )
        assert torch.isnan(result).sum() == 0
        assert result[1].item() == DEFAULT_QUANTITY
        assert result[3].item() == DEFAULT_QUANTITY

    def test_impute_nan_with_unknown_key(self) -> None:
        """Test that unknown keys return tensor unchanged."""
        tensor = torch.tensor([1.0, float("nan"), 3.0])
        result = impute_nan(
            imputed_value_given_key,
            "unknown_key",
            tensor,
        )
        # Should return as-is
        assert torch.isnan(result[1]).item() is True

    def test_impute_nan_no_nans(self) -> None:
        """Test imputation when tensor has no NaNs."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = impute_nan(
            imputed_value_given_key,
            "last_price",
            tensor,
        )
        # Should be unchanged
        assert torch.equal(result, tensor)

    def test_impute_nan_all_nans(self) -> None:
        """Test imputation when entire tensor is NaN."""
        tensor = torch.full((4,), float("nan"))
        result = impute_nan(
            imputed_value_given_key,
            "bid.price",
            tensor,
        )
        assert torch.isnan(result).sum() == 0
        assert torch.all(result == DEFAULT_PRICE)

    def test_impute_nan_custom_function(self) -> None:
        """Test imputation with custom imputation function."""

        def custom_impute(key: str) -> float | None:
            if key == "custom_key":
                return 999.0
            return None

        tensor = torch.tensor([1.0, float("nan"), 3.0])
        result = impute_nan(custom_impute, "custom_key", tensor)
        assert result[1].item() == 999.0


class TestOriginalTensorDictPreprocessor:
    """Test OriginalTensorDictPreprocessor."""

    def test_preprocessor_target_field(self) -> None:
        """Test that target field is correctly set."""
        assert (
            OriginalTensorDictPreprocessor.target_field
            == "hidden.goal_suit_code"
        )

    def test_preprocessor_feature_fields(self) -> None:
        """Test that feature fields are correctly defined."""
        expected_fields = [
            "step",
            "steps_remaining",
            "remaining_game_portion",
            "cash",
            "per_suit",
        ]
        assert OriginalTensorDictPreprocessor.feature_fields == expected_fields

    def test_preprocessor_has_preprocess_method(self) -> None:
        """Test that preprocessor has preprocess method."""
        preprocessor = OriginalTensorDictPreprocessor()
        assert hasattr(preprocessor, "preprocess")
        assert callable(preprocessor.preprocess)

    def test_preprocessor_has_split_dataset_method(self) -> None:
        """Test that preprocessor has split_dataset method."""
        preprocessor = OriginalTensorDictPreprocessor()
        assert hasattr(preprocessor, "split_dataset")
        assert callable(preprocessor.split_dataset)

    def test_preprocessor_has_calculate_common_fields_method(self) -> None:
        """Test that preprocessor has calculate_common_fields method."""
        preprocessor = OriginalTensorDictPreprocessor()
        assert hasattr(preprocessor, "calculate_common_fields")
        assert callable(preprocessor.calculate_common_fields)


class TestTensorDictPreprocessorIntegration:
    """Integration tests for preprocessor functionality."""

    def test_preprocessor_instantiation(self) -> None:
        """Test that preprocessor can be instantiated."""
        preprocessor = OriginalTensorDictPreprocessor()
        assert preprocessor is not None

    def test_feature_fields_not_empty(self) -> None:
        """Test that feature fields are not empty."""
        assert len(OriginalTensorDictPreprocessor.feature_fields) > 0

    def test_imputation_function_consistency(self) -> None:
        """Test consistency of imputation across multiple calls."""
        tensor1 = torch.tensor([float("nan"), 1.0, float("nan")])
        tensor2 = torch.tensor([float("nan"), 1.0, float("nan")])

        result1 = impute_nan(imputed_value_given_key, "last_price", tensor1)
        result2 = impute_nan(imputed_value_given_key, "last_price", tensor2)

        assert torch.equal(result1, result2)
