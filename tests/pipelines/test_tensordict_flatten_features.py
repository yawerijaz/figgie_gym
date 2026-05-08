"""Tests for tensordict_flatten_features module."""

import pytest

from figgie_gym.pipelines.tensordict_flatten_features import (
    get_feature_list,
    known_count,
    last_price,
    raw_all,
    self_count,
)


class TestGetFeatureList:
    """Test feature list generation for different feature spaces."""

    def test_known_count_flat(self) -> None:
        """Test that known_count_flat returns correct feature list."""
        features = get_feature_list("known_count_flat")
        assert features == known_count
        assert len(features) == 4
        assert all("known_count" in f for f in features)

    def test_raw_flat(self) -> None:
        """Test that raw_flat returns all features."""
        features = get_feature_list("raw_flat")
        assert features == raw_all
        assert len(features) > 50  # raw_all has many features
        assert "time" in features
        assert "cash" in features

    def test_known_and_price_flat(self) -> None:
        """Test that known_and_price_flat combines known count, self position, and last price."""
        features = get_feature_list("known_and_price_flat")
        expected = known_count + self_count + last_price
        assert features == expected
        assert len(features) == 12
        assert all(
            "known_count" in f or "self_position" in f or "last_price" in f
            for f in features
        )

    def test_invalid_feature_space(self) -> None:
        """Test that invalid feature space raises error."""
        with pytest.raises(Exception):
            # This should raise an assert_never exception
            get_feature_list("invalid_feature_space")  # type: ignore


class TestFeatureListConstants:
    """Test feature list constant definitions."""

    def test_known_count_features(self) -> None:
        """Test known_count features structure."""
        assert len(known_count) == 4
        suits = ["Spade", "Club", "Heart", "Diamond"]
        for i, suit in enumerate(suits):
            assert suit in known_count[i]
            assert "known_count" in known_count[i]

    def test_self_count_features(self) -> None:
        """Test self_count features structure."""
        assert len(self_count) == 4
        suits = ["Spade", "Club", "Heart", "Diamond"]
        for i, suit in enumerate(suits):
            assert suit in self_count[i]
            assert "self_position" in self_count[i]

    def test_last_price_features(self) -> None:
        """Test last_price features structure."""
        assert len(last_price) == 4
        suits = ["Spade", "Club", "Heart", "Diamond"]
        for i, suit in enumerate(suits):
            assert suit in last_price[i]
            assert "last_price" in last_price[i]

    def test_raw_all_features(self) -> None:
        """Test raw_all features structure."""
        assert len(raw_all) > 50
        # Check for key features
        assert "time" in raw_all
        assert "remaining_time" in raw_all
        assert "cash" in raw_all
        # Check for suit-specific features
        suits = ["Spade", "Club", "Heart", "Diamond"]
        for suit in suits:
            suit_features = [f for f in raw_all if suit in f]
            assert len(suit_features) > 0
            assert any("last_price" in f for f in suit_features)
            assert any("known_count" in f for f in suit_features)

    def test_feature_consistency(self) -> None:
        """Test that feature paths follow consistent naming convention."""
        all_features = known_count + self_count + last_price + raw_all
        # Remove duplicates for checking
        unique_features = set(all_features)

        # Check that features with "per_suit" contain one of the suits
        suits = ["Spade", "Club", "Heart", "Diamond"]
        for feature in unique_features:
            if "per_suit" in feature:
                assert any(suit in feature for suit in suits), (
                    f"Feature {feature} has per_suit but no suit"
                )
