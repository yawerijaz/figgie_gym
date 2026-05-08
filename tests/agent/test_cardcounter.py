import numpy as np

from figgie_gym.agent.cardcounter import (
    ExpectedValueGeometricAggressive,
    PosteriorProbabilies,
    get_symbol_quantity_permutations,
    posterior_probabilities_on_symbol_quantity_permutation,
)


def test_get_symbol_quantity_permutations() -> None:
    assert len(get_symbol_quantity_permutations()) == 12


def test_posterior_probabilities_on_symbol_quantity_permutation() -> None:
    probs = posterior_probabilities_on_symbol_quantity_permutation(
        [
            [12, 10, 8, 10],
            [8, 10, 12, 10],
        ],
        [3, 2, 1, 2],
    )
    assert probs[1] < probs[0]

    probs = posterior_probabilities_on_symbol_quantity_permutation(
        [
            [12, 10, 10, 8],
            [8, 10, 12, 10],
            [8, 10, 12, 12],
        ],
        [3, 2, 1, 2],
    )
    assert probs[1] < probs[0]

    probs = posterior_probabilities_on_symbol_quantity_permutation(
        [
            [12, 10, 10, 8],
            [10, 8, 10, 12],
            [8, 10, 10, 12],
        ],
        [8, 8, 8, 9],
    )
    assert probs[0] == 0
    assert probs[1] == probs[2]

    probs = posterior_probabilities_on_symbol_quantity_permutation(
        [
            [12, 10, 10, 8],
            [10, 8, 10, 12],
            [8, 10, 10, 12],
        ],
        [4, 4, 4, 4],
    )
    assert probs[0] == probs[1] == probs[2]

    probs1 = posterior_probabilities_on_symbol_quantity_permutation(
        [
            [12, 10, 10, 8],
            [8, 10, 12, 10],
            [8, 10, 12, 12],
        ],
        [3, 2, 1, 2],
    )
    probs2 = posterior_probabilities_on_symbol_quantity_permutation(
        [
            [12, 10, 10, 8],
            [8, 10, 12, 10],
            [8, 10, 12, 12],
        ],
        [5, 4, 3, 4],
    )
    assert probs1[0] < probs2[0]


def test_expected_value() -> None:
    perm = get_symbol_quantity_permutations()
    prob = PosteriorProbabilies(perm)
    calc = ExpectedValueGeometricAggressive(
        full_pot=200,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=8.0,
        posterior_calculator=prob,
    )

    known_holdings = [2, 2, 2, 2]
    current_holdings = [0, 0, 0, 0]
    evs = calc.compute_expected_value(known_holdings, current_holdings)
    assert (evs == evs[0]).all(), evs

    known_holdings = [2, 2, 2, 5]
    current_holdings = [2, 2, 2, 2]
    evs = calc.compute_expected_value(known_holdings, current_holdings)
    assert (evs.max() == evs[2]).all(), evs

    known_holdings = [4, 2, 6, 6]
    current_holdings = [4, 2, 6, 5]
    evs1 = calc.compute_expected_value(known_holdings, current_holdings)
    current_holdings = [4, 2, 6, 6]
    evs2 = calc.compute_expected_value(known_holdings, current_holdings)
    assert evs1[3] - evs2[3] > 5

    known_holdings = [10, 12, 10, 8]
    current_holdings = [10, 12, 10, 8]
    evs1 = calc.compute_expected_value(known_holdings, current_holdings)
    assert (evs1 == np.array([0.0, 0.0, 0.0, 0.0])).all(), evs1
    current_holdings = [9, 12, 10, 8]
    evs2 = calc.compute_expected_value(known_holdings, current_holdings)
    assert (evs2 == np.array([10.0, 0.0, 0.0, 0.0])).all(), evs2
