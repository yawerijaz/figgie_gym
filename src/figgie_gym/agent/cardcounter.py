from __future__ import annotations

from functools import cache
from itertools import permutations
from typing import TYPE_CHECKING, Any, Literal, cast, override

import numpy as np
from numpy import float64, int32
from scipy.special import factorial

from figgie_gym.envs.common import (
    SUITS,
    ActionOnSuit,
    ActionType,
    Agent,
    BeliefType,
    ExtraType,
    ObsType,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CardCounterAgent(Agent):
    def __init__(
        self,
        ev_calculator: ExpectedValueGeometricAggressive,
        quote_spread: int,
    ) -> None:
        self.quote_spread = quote_spread
        self.ev_calculator = ev_calculator
        self.late_aggressiveness_factor = (
            ev_calculator.late_aggressiveness_factor
        )

    @override
    def params(self) -> dict[str, Any]:
        return super().params() | {
            "quote_spread": self.quote_spread,
            "late_aggressiveness_factor": self.late_aggressiveness_factor,
        }

    def act(
        self,
        random_number_generator: np.random.Generator,  # noqa: ARG002
        observation: ObsType,
    ) -> tuple[BeliefType, ActionType, ExtraType]:
        suit_obs = observation.per_suit
        my_positions = [int(suit_obs[sym].self_position) for sym in SUITS]
        known_positions = [
            sum(
                s.net_quantity_change() - s.min_net_quantity_change
                for s in suit_obs[sym].other_trade_summaries
            )
            + suit_obs[sym].self_position
            for sym in SUITS
        ]
        probs = self.ev_calculator.posterior_calculator.compute_long_suit_probability_given_known_holdings(
            known_positions,
        )[[1, 0, 3, 2]]
        fair_buy = self.ev_calculator.breakeven_buy_price(
            known_positions,
            my_positions,
        )
        fair_sell = self.ev_calculator.breakeven_sell_price(
            known_positions,
            my_positions,
        )
        fair_mid = (fair_buy + fair_sell) / 2
        quote_bid = (
            np.minimum(fair_mid, fair_buy) - self.quote_spread / 2
        ).clip(min=0)
        quote_ask = np.maximum(fair_mid, fair_sell) + self.quote_spread / 2

        return (
            dict(zip(SUITS, probs, strict=True)),
            {
                sym: ActionOnSuit(
                    quote_bid=quote_bid[i],
                    quote_ask=quote_ask[i],
                    snipe_bid=fair_buy[i],
                    snipe_ask=fair_sell[i],
                )
                for i, sym in enumerate(SUITS)
            },
            None,
        )

    def act_batch(
        self,
        random_number_generator: np.random.Generator,
        observations: list[ObsType],
    ) -> list[tuple[BeliefType, ActionType, None]]:
        return super().act_batch(random_number_generator, observations)


class PosteriorProbabilies:
    def __init__(
        self,
        symbol_quantity_permutations: list[list[int]] | NDArray[int32],
    ) -> None:
        self.symbol_quantity_permutations = np.array(
            symbol_quantity_permutations,
        )

        self._compute_given_known_holdings_cached = cache(
            self._compute_given_known_holdings,
        )

    def compute_given_known_holdings(
        self,
        known_holdings: list[int] | NDArray[int32],
    ) -> NDArray[float64]:
        if isinstance(known_holdings, np.ndarray):
            known_holdings_list = cast("list[int]", known_holdings.tolist())
        else:
            known_holdings_list = known_holdings
        known_holdings_tuple = cast("tuple[int]", tuple(known_holdings_list))
        return self._compute_given_known_holdings_cached(known_holdings_tuple)

    def _compute_given_known_holdings(
        self,
        known_holdings: tuple[int],
    ) -> NDArray[float64]:
        permut = self.symbol_quantity_permutations
        known_holdings_array = np.array(known_holdings)

        unknown_quantities = permut - known_holdings_array
        partial_factorial = np.array(
            [
                np.prod(
                    np.array(
                        [factorial(y) if y >= 0 else float("inf") for y in x],
                    ),
                )
                for x in unknown_quantities
            ],
        )
        weight = 1 / partial_factorial
        if weight.sum() == 0:
            msg = f"{weight.sum()=}, {known_holdings_array=}"
            raise ValueError(msg)
        return weight / weight.sum()

    def compute_long_suit_probability_given_known_holdings(
        self,
        known_holdings: list[int] | NDArray[int32],
    ) -> NDArray[float64]:
        long_card_suit = self.symbol_quantity_permutations.argmax(axis=1)
        long_prob = np.bincount(
            long_card_suit,
            self.compute_given_known_holdings(known_holdings),
        )
        return np.array(long_prob, dtype=float64)


def posterior_probabilities_on_symbol_quantity_permutation(
    symbol_quantity_permutations: list[list[int]] | NDArray[np.int32],
    known_holdings: list[int] | NDArray[np.int32],
) -> NDArray[float64]:
    return PosteriorProbabilies(
        symbol_quantity_permutations,
    ).compute_given_known_holdings(known_holdings)


def get_symbol_quantity_permutations(
    symbol_quantities: list[int] | None = None,
) -> NDArray[np.int32]:
    symbol_quantities = symbol_quantities or [10, 10, 12, 8]
    return np.array(
        [(list(permut)) for permut in set(permutations(symbol_quantities))],
    )


class ExpectedValueGeometricAggressive:
    MAX_HOLDING = 12

    def __init__(
        self,
        full_pot: int,
        goal_symbol_value: int,
        num_symbols: int,
        late_aggressiveness_factor: float,
        posterior_calculator: PosteriorProbabilies,
    ) -> None:
        self.posterior_calculator = posterior_calculator
        self.permutations = posterior_calculator.symbol_quantity_permutations
        self.full_pot = full_pot
        self.goal_symbol_value = goal_symbol_value
        self.num_symbols = num_symbols
        self.late_aggressiveness_factor = max(
            late_aggressiveness_factor,
            1 + 1e-6,
        )

        self.symbol_value = {
            "intrinsic": self.compute_symbol_value_all_permuation_symbol_holding(
                "intrinsic",
            ),
            "pot": self.compute_symbol_value_all_permuation_symbol_holding(
                "pot",
            ),
            "all": self.compute_symbol_value_all_permuation_symbol_holding(
                "all",
            ),
        }

    def compute_symbol_value_all_permuation_symbol_holding(
        self,
        include: Literal["intrinsic", "pot", "all"] = "all",
    ) -> NDArray[float64]:
        permus = self.permutations
        r = self.late_aggressiveness_factor

        goal = np.choose(permus.argmax(axis=1), [1, 0, 3, 2])  # (permus)
        goal_quantity = permus[
            np.arange(len(permus), dtype=int),
            goal,
        ]  # (permus)
        holding_guarantee_pot = goal_quantity // 2 + 1  # (permus)
        pot = self.full_pot - goal_quantity * self.goal_symbol_value  # (permus)
        value_of_first = (
            pot * (1 - r) / (1 - r**holding_guarantee_pot)
        )  # (permus)

        value_of_each = value_of_first[:, np.newaxis] * r ** np.arange(
            self.MAX_HOLDING + 1,
        )  # (permus x holding)
        mask = (
            np.arange(self.MAX_HOLDING + 1)
            < holding_guarantee_pot[:, np.newaxis]
        )  # (permus x holding)
        value_of_each = np.where(
            mask,
            value_of_each,
            0,
        ).round()  # (permus x holding)

        match include:
            case "all":
                all_value_by_permu_symbol_holding = (
                    value_of_each + self.goal_symbol_value
                )
            case "intrinsic":
                all_value_by_permu_symbol_holding = (
                    np.zeros_like(value_of_each) + self.goal_symbol_value
                )
            case "pot":
                all_value_by_permu_symbol_holding = value_of_each

        value_from_pot_by_permu_symbol_holding = np.zeros(
            (len(permus), self.num_symbols, self.MAX_HOLDING + 1),
        )  # (permus x symbol x holding)
        mask = (
            np.arange(self.MAX_HOLDING + 1) < goal_quantity[:, np.newaxis]
        )  # (permus x holding)
        value_from_pot_by_permu_symbol_holding[np.arange(len(permus)), goal] = (
            np.where(mask, all_value_by_permu_symbol_holding, 0)
        )
        return value_from_pot_by_permu_symbol_holding

    def compute_expected_value(
        self,
        known_holdings: list[int] | NDArray[np.int32],
        current_holdings: list[int] | NDArray[np.int32],
        include: Literal["intrinsic", "pot", "all"] = "all",
    ) -> NDArray[float64]:
        current_holdings = np.array(current_holdings)
        probs = self.posterior_calculator.compute_given_known_holdings(
            known_holdings,
        )
        return (
            probs[:, np.newaxis]
            * self.symbol_value[include][
                :,
                np.arange(self.num_symbols),
                np.maximum(0, np.minimum(current_holdings, self.MAX_HOLDING)),
            ]
        ).sum(axis=0)

    def breakeven_buy_price(
        self,
        known_holdings: list[int] | NDArray[np.int32],
        current_holdings: list[int] | NDArray[np.int32],
    ) -> NDArray[np.float64]:
        return self.compute_expected_value(known_holdings, current_holdings)

    def breakeven_sell_price(
        self,
        known_holdings: list[int] | NDArray[np.int32],
        current_holdings: list[int] | NDArray[np.int32],
    ) -> NDArray[np.float64]:
        return self.compute_expected_value(
            known_holdings,
            (np.array(current_holdings) - 1),
        )
