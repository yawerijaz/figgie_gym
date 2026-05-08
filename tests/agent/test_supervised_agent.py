"""Tests for supervised agent module."""

import numpy as np

from figgie_gym.agent.supervised import (
    SimplifiedExpectedValueGeometricAggressive,
)


def test_initialization_basic() -> None:
    """Test basic initialization."""
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    assert calc is not None


def test_full_pot_stored() -> None:
    """Test that full_pot is stored correctly."""
    full_pot = 100
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=full_pot,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    assert calc.full_pot == full_pot


def test_goal_symbol_value_stored() -> None:
    """Test that goal_symbol_value is stored correctly."""
    goal_symbol_value = 10
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=goal_symbol_value,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    assert calc.goal_symbol_value == goal_symbol_value


def test_num_symbols_stored() -> None:
    """Test that num_symbols is stored correctly."""
    num_symbols = 4
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=num_symbols,
        late_aggressiveness_factor=1.5,
    )
    assert calc.num_symbols == num_symbols


def test_late_aggressiveness_factor_stored() -> None:
    """Test that late_aggressiveness_factor is stored and minimum enforced."""
    # Test with value > 1
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    assert calc.late_aggressiveness_factor == 1.5


def test_late_aggressiveness_minimum_enforced() -> None:
    """Test that late_aggressiveness_factor has minimum value."""
    # Test with value < 1 (should be adjusted to minimum)
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=0.5,
    )
    # Should be at least 1 + epsilon
    assert calc.late_aggressiveness_factor > 1.0


def test_symbol_value_computed() -> None:
    """Test that symbol values are computed."""
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    assert hasattr(calc, "symbol_value")
    assert "intrinsic" in calc.symbol_value
    assert "pot" in calc.symbol_value
    assert "all" in calc.symbol_value


def test_symbol_value_intrinsic_shape() -> None:
    """Test that intrinsic symbol value has correct shape."""
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    intrinsic = calc.symbol_value["intrinsic"]
    assert len(intrinsic) == calc.MAX_HOLDING + 1


def test_symbol_value_pot_shape() -> None:
    """Test that pot symbol value has correct shape."""
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    pot = calc.symbol_value["pot"]
    assert len(pot) == calc.MAX_HOLDING + 1


def test_symbol_value_all_shape() -> None:
    """Test that all symbol value has correct shape."""
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    all_val = calc.symbol_value["all"]
    assert len(all_val) == calc.MAX_HOLDING + 1


def test_max_holding_constant() -> None:
    """Test that MAX_HOLDING constant is set."""
    assert SimplifiedExpectedValueGeometricAggressive.MAX_HOLDING == 12


def test_compute_symbol_value_goal_only_intrinsic() -> None:
    """Test compute_symbol_value_goal_only with intrinsic include."""
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    value = calc.compute_symbol_value_goal_only("intrinsic")
    # All values should equal goal_symbol_value for intrinsic
    assert len(value) == 13  # MAX_HOLDING + 1
    assert np.all(value == 10)


def test_compute_symbol_value_goal_only_pot() -> None:
    """Test compute_symbol_value_goal_only with pot include."""
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    value = calc.compute_symbol_value_goal_only("pot")
    assert len(value) == 13
    # Pot values should be non-negative
    assert np.all(value >= 0)


def test_compute_symbol_value_goal_only_all() -> None:
    """Test compute_symbol_value_goal_only with all include."""
    calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=100,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.5,
    )
    value = calc.compute_symbol_value_goal_only("all")
    assert len(value) == 13
    # All should be >= intrinsic
    assert np.all(value >= 10)


def test_different_pots() -> None:
    """Test with different full pot values."""
    pots = [50, 100, 200, 500]
    for pot in pots:
        calc = SimplifiedExpectedValueGeometricAggressive(
            full_pot=pot,
            goal_symbol_value=10,
            num_symbols=4,
            late_aggressiveness_factor=1.5,
        )
        assert calc.full_pot == pot


def test_different_goal_values() -> None:
    """Test with different goal symbol values."""
    goal_values = [5, 10, 15, 20]
    for goal_val in goal_values:
        calc = SimplifiedExpectedValueGeometricAggressive(
            full_pot=100,
            goal_symbol_value=goal_val,
            num_symbols=4,
            late_aggressiveness_factor=1.5,
        )
        assert calc.goal_symbol_value == goal_val


def test_different_aggressiveness_factors() -> None:
    """Test with different aggressiveness factors."""
    factors = [1.1, 1.5, 2.0, 3.0]
    for factor in factors:
        calc = SimplifiedExpectedValueGeometricAggressive(
            full_pot=100,
            goal_symbol_value=10,
            num_symbols=4,
            late_aggressiveness_factor=factor,
        )
        assert calc.late_aggressiveness_factor == factor


def test_different_num_symbols() -> None:
    """Test with different number of symbols."""
    num_syms = [2, 3, 4, 5, 6]
    for num_sym in num_syms:
        calc = SimplifiedExpectedValueGeometricAggressive(
            full_pot=100,
            goal_symbol_value=10,
            num_symbols=num_sym,
            late_aggressiveness_factor=1.5,
        )
        assert calc.num_symbols == num_sym
