import numpy as np
from numpy.typing import NDArray

from figgie_gym.agent.cardcounter import (
    PosteriorProbabilies,
    get_symbol_quantity_permutations,
)


def card_counter_prediction(
    suit_known_counts: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compute probablity of each suit being the target.

    Order assumed: Spade, Club, Heart, Diamond
    """
    permutations = get_symbol_quantity_permutations()
    prob = PosteriorProbabilies(permutations)
    long_prob = np.apply_along_axis(
        prob.compute_long_suit_probability_given_known_holdings,
        1,
        suit_known_counts,
    )
    return long_prob[:, [1, 0, 3, 2]]  # S, C, H, D assumed
